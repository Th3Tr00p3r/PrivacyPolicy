import gzip
import json
import logging
import logging.config
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List
from winsound import Beep

import aiofiles  # type: ignore
import gensim
import httpx
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.doc2vec import TaggedDocument

from ppa.utils import timer

# TODO: I should separate between two cases for URL-like strings - companyname.com should be <COMPANYNAME>, www.companyname.com should be <URL>.
# Also, can I use the special token method of the Dictionary for them?
URL_PATTERN = r"(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?\/[a-zA-Z0-9]{2,}|((https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?)|(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}(\.[a-zA-Z0-9]{2,})?"
EMAIL_PATTERN = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

RAW_DATA_FPATH = Path("tosdr_raw.json")
DATABASE_FPATH = Path("tosdr_db.json")


@dataclass
class ToSDRDataLoader:
    """Doc."""

    api_url: str = "https://api.tosdr.org/search/v4/?query="
    raw_data_fpath: Path = RAW_DATA_FPATH
    database_fpath: Path = DATABASE_FPATH

    async def load_data(
        self,
        queries: List[str],
        force_extract=False,
        force_transform=False,
        beep=True,
        **kwargs,
    ) -> pd.DataFrame:
        """Doc."""

        # Extract fresh raw data
        if not self.raw_data_fpath.exists() or force_extract:
            await self._extract_data(queries, **kwargs)
            # Optionally beep when done (Assuming Windows OS)
            if beep:
                Beep(1000, 500)

        # Transform and load afresh
        if not self.database_fpath.exists() or force_transform:
            # Transform the raw data into a dictionary object
            df = self._transform_data()
            # Save to a JSON file
            df.to_json(self.database_fpath)

        # Read the JSON file into a Pandas DataFrame
        return pd.read_json(self.database_fpath)

    def _transform_data(self) -> pd.DataFrame:
        """Doc."""

        # Read the JSON file into a Pandas DataFrame
        df = pd.read_json(self.raw_data_fpath, lines=True)

        # prepare the "parameters" columns for JSON normalization
        df["parameters"] = df["parameters"].apply(lambda x: x["services"][0])

        # normalize the JSON inside the "parameters" column
        parameters_df = pd.json_normalize(df["parameters"], max_level=1)

        # build a new DataFrame from the relevant data, and rename
        df = pd.concat([df["message"], parameters_df[["id", "rating.letter"]]], axis=1)
        df.rename(columns={"message": "tag", "rating.letter": "rating"}, inplace=True)

        # replace "N/A" strings with NaNs
        df["rating"].replace("N/A", np.nan, inplace=True)

        # ignore nulls
        df = df[df["rating"].notnull()]

        return df

    async def _extract_data(self, queries: List[str], timeout_s=10, **kwargs):
        """Doc."""

        async def write_response(response):
            """
            Write a response to the data file.

            Args:
                response: The response to write to the file.

            Returns:
                None
            """
            async with aiofiles.open(self.raw_data_fpath, "a") as json_file:
                #                await json_file.write(json.dumps(data, indent=4)[1:-2] + ",\n")
                await json_file.write(f"{json.dumps(response.json())}\n")

        logging.info("Started loading data.")

        # Ensure file exists
        with open(self.raw_data_fpath, "w"):
            pass

        # Make requests asynchronously
        async with httpx.AsyncClient() as client:
            for query in queries:
                try:
                    # call API (sibgle query)
                    response = await client.get(f"{self.api_url}{query}", timeout=timeout_s)

                    # If succesful call
                    if response.status_code == 200:
                        # Check that data exists (not empty)
                        if not self._is_data_empty(response):
                            await write_response(response)
                            logging.info(f"{query}: Success!")
                        else:
                            logging.info(f"{query}: Data is empty!")
                    else:
                        logging.info(
                            f"{query}: Failed to fetch data. Status code: {response.status_code}"
                        )

                except httpx.TimeoutException as exc:
                    logging.info(f"{query}: Timed-out while fetching data. Error: {exc}")
                    continue

        logging.info("Finished loading data.")

    def _is_data_empty(self, response) -> bool:
        """Doc."""

        return not bool(response.json()["parameters"]["services"])


@dataclass
class SampleGenerator:
    """Doc."""

    fpath: Path
    n_samples: int
    offset: int = 0
    id_corpus: bool = False
    dct: Dictionary = None

    def __repr__(self):
        return f"SampleGenerator({self.n_samples - self.offset} samples. fpath={self.fpath}, offset={self.offset}, id_corpus={self.id_corpus})"

    def __iter__(self):
        """Doc."""

        #         print(f"Iterating over {self.n_samples - self.offset} samples.") # TESTESTEST

        # check that the generator is not meant to be empty, to avoid loading the entire file for nothing
        if self.offset == self.n_samples:
            return

        offset = self.offset
        with gzip.open(self.fpath, "rb") as input_file:
            try:
                for _ in range(self.n_samples):
                    # Deserialize and yield one document at a time
                    tagged_doc = pickle.load(input_file)
                    # skip 'offset' first samples
                    if offset:
                        #                         print(f"Skipping sample #{self.offset - offset + 1}") # TESTESTEST
                        offset -= 1
                        continue  # skip this sample
                    if self.id_corpus:
                        #                         if self.offset: # TESTESTEST
                        #                             print("Yielding sample") # TESTESTEST
                        yield TaggedDocument(
                            [self.dct[id] for id in tagged_doc.words], tagged_doc.tags
                        )
                    else:
                        yield tagged_doc
            except EOFError:
                pass  # End of file


@dataclass
class CorpusProcessor:
    """Doc."""

    fpaths: List[Path]
    save_dir_path: Path
    should_filter_stopwords: bool = False
    url_pattern: str = URL_PATTERN
    email_pattern: str = EMAIL_PATTERN
    max_tokens: int = 100_000
    min_tokens: int = 0
    seed: int = None
    dct: Dictionary = None

    def __post_init__(self) -> None:
        self.total_samples: int = len(self.fpaths)
        # shuffle the paths (reproducible with seed)
        if self.seed is not None:
            random.seed(self.seed)
            random.shuffle(self.fpaths)

    def __repr__(self):
        return f"CorpusProcessor({len(self.fpaths)} docs, should_filter_stopwords={self.should_filter_stopwords}, seed={self.seed}, min_tokens={self.min_tokens}, max_tokens={self.max_tokens})"

    @timer()
    def process(self, force=False, id_corpus=True, **kwargs):
        """Doc."""

        if not force:
            # try loading existing dict
            self.dct = Dictionary.load(str(self.save_dir_path / "dictionary.pkl"))
            print("Loaded existing dictionary.")
            print("Using existing training data.")

        else:
            print(
                f"Processing and saving {self.total_samples:,} TaggedDocument objects to disk: ",
                end="",
            )
            # initialize a Dictionary object
            self.dct = Dictionary()
            with gzip.open(self.save_dir_path / "train_data.pkl.gz", "wb") as output_file:
                # re-iterate, this time converting the tokens to integers according to dict ID, then saving
                for fidx, fpath in enumerate(self.fpaths):
                    # track visually
                    if not (fidx + 1) % (self.total_samples // 100):
                        print("o", end="")
                    # open and process each file
                    tokenized_doc = self._preprocess_document(fpath)
                    # ignore very short/long documents
                    if self.min_tokens <= len(tokenized_doc) <= self.max_tokens:
                        # add to dictionary
                        self.dct.add_documents([tokenized_doc])
                        # create a TaggedDocument instance
                        tagged_doc = TaggedDocument(words=tokenized_doc, tags=[fpath.stem])
                        # Serialize the document tokens using pickle and write to the compressed file
                        pickle.dump(tagged_doc, output_file, protocol=pickle.HIGHEST_PROTOCOL)
                print(" - Done.")

            print("Saving Dictionary... ", end="")
            self.dct.save(str(self.save_dir_path / "dictionary.pkl"))
            print("Done.")

        # ID corpus
        if id_corpus and force:
            print(
                f"Re-Processing and saving {self.total_samples:,} ndarrays to disk: ",
                end="",
            )
            with gzip.open(self.save_dir_path / "train_data_id.pkl.gz", "wb") as output_file:
                # re-iterate, this time converting the tokens to integers according to dict ID, then saving
                for fidx, tagged_doc in enumerate(self.generate_samples(id_corpus=False)):
                    # track visually
                    if not (fidx + 1) % (self.total_samples // 100):
                        print("o", end="")
                    # open and process each file
                    tagged_id_doc = TaggedDocument(
                        words=self.dct.doc2idx(tagged_doc.words), tags=tagged_doc.tags
                    )
                    # Serialize the document tokens using pickle and write to the compressed file
                    pickle.dump(tagged_id_doc, output_file, protocol=pickle.HIGHEST_PROTOCOL)
                print(" - Done.")
            # delete the tokenized corpus
            (self.save_dir_path / "train_data.pkl.gz").unlink()

    def generate_train_test_sets(
        self, n_samples: int = None, test_frac: float = 0.2, id_corpus=True
    ):
        """Doc."""

        n_samples = n_samples or self.total_samples
        fpath = self.save_dir_path / f"train_data{('_id' if id_corpus else '')}.pkl.gz"

        # initialize re-generators
        n_train_samples = round(n_samples * (1 - test_frac))
        train_sample_loader = SampleGenerator(
            fpath, n_train_samples, id_corpus=id_corpus, dct=self.dct
        )
        test_sample_loader = SampleGenerator(
            fpath, n_samples, offset=n_train_samples, id_corpus=id_corpus, dct=self.dct
        )
        return train_sample_loader, test_sample_loader

    def generate_samples(self, n_samples: int = None, id_corpus=True):
        """Doc."""
        # TODO: change name to "sample_generator"

        n_samples = n_samples or self.total_samples
        fpath = self.save_dir_path / f"train_data{('_id' if id_corpus else '')}.pkl.gz"

        # initialize re-generator
        return SampleGenerator(fpath, n_samples, id_corpus=id_corpus, dct=self.dct)

    #     def _reservoir_sampling(self, iterable: Iterable, n_samples: int):
    #         """Doc."""

    #         sample = []
    #         for idx, item in enumerate(iterable):
    #             if idx < n_samples:
    #                 sample.append(item)
    #             else:
    #                 switch_idx = random.randint(0, idx)
    #                 if switch_idx < n_samples:
    #                     sample[switch_idx] = item
    #         return sample

    def _preprocess_document(self, fpath: Path):
        """Doc."""

        # read all but the header (probably not the most efficient method)
        with open(fpath, "r", encoding="utf-8") as f:
            _, *doc_lines = f.read().split("\n")
            doc = "\n".join(doc_lines)

        # Replace URLs with "<URL>", email addresses with "<EMAILADD>"
        if self.url_pattern:
            doc = re.sub(self.url_pattern, "<URL>", doc)
        if self.email_pattern:
            doc = re.sub(self.email_pattern, "<EMAILADD>", doc)

        # tokenize the text
        simp_proc_doc = gensim.utils.simple_preprocess(doc)

        # remove stopwords and return
        if self.should_filter_stopwords:
            return gensim.parsing.preprocessing.remove_stopword_tokens(simp_proc_doc)
        else:
            return simp_proc_doc
