import gzip
import json
import logging
import logging.config
import pickle
import random
import re
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from winsound import Beep

import aiofiles  # type: ignore
import gensim
import httpx
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.doc2vec import TaggedDocument

from ppa.utils import timer

# TODO: Separate between two cases for URL-like strings - companyname.com should be <COMPANYNAME>, www.companyname.com should be <URL>.
# Also, can I use the special token method of the Dictionary for them?
URL_PATTERN = r"(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?\/[a-zA-Z0-9]{2,}|((https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?)|(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}(\.[a-zA-Z0-9]{2,})?"
EMAIL_PATTERN = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

RAW_DATA_FPATH = Path("tosdr_raw.json")
DATABASE_FPATH = Path("tosdr_db.json")


@dataclass
class ToSDRDataLoader:
    """
    Load data from the ToSDR API and process it into a DataFrame.

    Args:
        api_url (str): The URL of the ToSDR API.
        raw_data_fpath (Path): The path to the raw data file.
        database_fpath (Path): The path to the database file.

    Methods:
        load_data(queries: List[str], force_extract=False, force_transform=False, beep=True, **kwargs) -> pd.DataFrame:
            Load data from the ToSDR API, transform it, and return it as a DataFrame.
    """

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
        """
        Load data from the ToSDR API, transform it, and return it as a DataFrame.

        Args:
            queries (List[str]): A list of queries to search in the ToSDR API.
            force_extract (bool): Force data extraction from the API (default is False).
            force_transform (bool): Force data transformation (default is False).
            beep (bool): Enable a beep notification (default is True).

        Returns:
            pd.DataFrame: A DataFrame containing the processed data.
        """
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
        """
        Transform raw data into a structured DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the transformed data.
        """
        # Read the JSON file into a Pandas DataFrame
        df = pd.read_json(self.raw_data_fpath, lines=True)

        # Prepare the "parameters" columns for JSON normalization
        df["parameters"] = df["parameters"].apply(lambda x: x["services"][0])

        # Normalize the JSON inside the "parameters" column
        parameters_df = pd.json_normalize(df["parameters"], max_level=1)

        # Build a new DataFrame from the relevant data, and rename
        df = pd.concat([df["message"], parameters_df[["id", "rating.letter"]]], axis=1)
        df.rename(columns={"message": "tag", "rating.letter": "rating"}, inplace=True)

        # Replace "N/A" strings with NaNs
        df["rating"].replace("N/A", np.nan, inplace=True)

        # Ignore nulls
        df = df[df["rating"].notnull()]

        return df

    async def _extract_data(self, queries: List[str], timeout_s=10, **kwargs):
        """
        Extract data from the ToSDR API asynchronously and write it to a file.

        Args:
            queries (List[str]): A list of queries to search in the ToSDR API.
            timeout_s (int): Timeout duration in seconds (default is 10).

        Returns:
            None
        """

        async def write_response(response):
            """
            Write a response to the data file.

            Args:
                response: The response to write to the file.

            Returns:
                None
            """
            async with aiofiles.open(self.raw_data_fpath, "a") as json_file:
                await json_file.write(f"{json.dumps(response.json())}\n")

        logging.info("Started loading data.")

        # Ensure file exists
        with open(self.raw_data_fpath, "w"):
            pass

        # Make requests asynchronously
        async with httpx.AsyncClient() as client:
            for query in queries:
                try:
                    # Call API (single query)
                    response = await client.get(f"{self.api_url}{query}", timeout=timeout_s)

                    # If successful call
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
        """
        Check if the response data from the API is empty.

        Args:
            response: The response from the API.

        Returns:
            bool: True if the response data is empty, False otherwise.
        """
        return not bool(response.json()["parameters"]["services"])


@dataclass
class SampleGenerator:
    """
    Generate samples from a data file.

    Args:
        fpath (Path): The path to the data file.
        n_samples (int): The number of samples to generate.
        offset (int): The offset for skipping samples (default is 0).
        id_corpus (bool): Whether to generate samples with integer IDs (default is False).
        dct (Dictionary): The Gensim Dictionary for ID conversion (default is None).
    """

    fpath: Path
    n_samples: int
    offset: int = 0
    id_corpus: bool = False
    dct: Dictionary = None

    def __repr__(self):
        return f"SampleGenerator(({(self.n_samples - self.offset):,} samples) fpath={self.fpath}, offset={self.offset}, id_corpus={self.id_corpus})"

    def __iter__(self):
        """
        Iterate over samples in the data file.
        """
        # Check that the generator is not meant to be empty to avoid loading the entire file for nothing
        if self.offset == self.n_samples:
            return

        offset = self.offset
        with gzip.open(self.fpath, "rb") as input_file:
            try:
                for _ in range(self.n_samples):
                    # Deserialize and yield one document at a time
                    tagged_doc = pickle.load(input_file)
                    # Skip 'offset' first samples
                    if offset:
                        offset -= 1
                        continue  # Skip this sample
                    if self.id_corpus:
                        yield TaggedDocument(
                            [self.dct[id] for id in tagged_doc.words], tagged_doc.tags
                        )
                    else:
                        yield tagged_doc

            except EOFError:
                pass  # End of file


@dataclass
class CorpusProcessor:
    """
    Process a corpus of documents into a format suitable for training a model.

    Args:
        fpaths (List[Path]): A list of file paths to the documents.
        save_dir_path (Path): The path to the directory for saving processed data.
        should_filter_stopwords (bool): Whether to filter out stopwords (default is False).
        url_pattern (str): A regular expression pattern for matching URLs (default is URL_PATTERN).
        email_pattern (str): A regular expression pattern for matching email addresses (default is EMAIL_PATTERN).
        max_tokens (int): The maximum number of tokens in a document (default is 100,000).
        min_tokens (int): The minimum number of tokens in a document (default is 0).
        seed (int): Seed for reproducible shuffling (default is None).
        dct (Dictionary): The Gensim Dictionary for document ID conversion (default is None).

    Methods:
        process(force=False, id_corpus=True, **kwargs):
            Process the corpus and save it to the specified directory.
        generate_train_test_sets(n_samples=None, test_frac=0.2, id_corpus=True):
            Generate training and testing sets from the processed data.
        generate_samples(n_samples=None, id_corpus=True):
            Generate document samples from the processed data.
    """

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
        # Shuffle the paths (reproducible with seed)
        if self.seed is not None:
            random.seed(self.seed)
            random.shuffle(self.fpaths)

    def __repr__(self):
        return f"CorpusProcessor({len(self.fpaths)} docs, should_filter_stopwords={self.should_filter_stopwords}, seed={self.seed}, min_tokens={self.min_tokens}, max_tokens={self.max_tokens})"

    @timer()
    def process(self, force=False, id_corpus=True, **kwargs):
        # TODO: make id_corpus the norm
        """
        Process the corpus and save it to the specified directory.

        Args:
            force (bool): Force reprocessing (default is False).
            id_corpus (bool): Whether to generate samples with integer IDs (default is True).

        Returns:
            None
        """
        if not force:
            # Try loading an existing dictionary
            self.dct = Dictionary.load(str(self.save_dir_path / "dictionary.pkl"))
            print("Loaded existing dictionary.")
            print("Using existing training data.")
        else:
            print(
                f"Processing and saving {self.total_samples:,} TaggedDocument objects to disk: ",
                end="",
            )
            # Initialize a Dictionary object
            self.dct = Dictionary()
            with gzip.open(self.save_dir_path / "train_data.pkl.gz", "wb") as output_file:
                # Re-iterate, this time converting the tokens to integers according to dict ID, then saving
                for fidx, fpath in enumerate(self.fpaths):
                    # Track progress visually
                    if not (fidx + 1) % (self.total_samples // 100):
                        print("o", end="")
                    # Open and process each file
                    tokenized_doc = self._preprocess_document(fpath)
                    # Ignore very short/long documents
                    if self.min_tokens <= len(tokenized_doc) <= self.max_tokens:
                        # Add to the dictionary
                        self.dct.add_documents([tokenized_doc])
                        # Create a TaggedDocument instance
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
                f"Re-Processing and saving {self.total_samples:,} TaggedDocument objects to disk: ",
                end="",
            )
            with gzip.open(self.save_dir_path / "train_data_id.pkl.gz", "wb") as output_file:
                # Re-iterate, this time converting the tokens to integers according to dict ID, then saving
                for fidx, tagged_doc in enumerate(self.generate_samples(id_corpus=False)):
                    # Track progress visually
                    if not (fidx + 1) % (self.total_samples // 100):
                        print("o", end="")
                    # Open and process each file
                    tagged_id_doc = TaggedDocument(
                        words=self.dct.doc2idx(tagged_doc.words), tags=tagged_doc.tags
                    )
                    # Serialize the document tokens using pickle and write to the compressed file
                    pickle.dump(tagged_id_doc, output_file, protocol=pickle.HIGHEST_PROTOCOL)
                print(" - Done.")
            # Delete the tokenized corpus
            (self.save_dir_path / "train_data.pkl.gz").unlink()

    @timer()
    def add_label_tags(self, tag_label_dict: Dict[str, str], force=False):
        """Doc."""

        if not force:
            # TODO: check also if file exists! if it doesn't, do the labeling even if force=False!
            print("Using existing labeled data.")

        else:
            print(
                f"Adding labels and saving {self.total_samples:,} TaggedDocument objects to disk: ",
                end="",
            )
            with gzip.open(self.save_dir_path / "labeled_corpus.pkl.gz", "wb") as output_file:
                # Re-iterate over all saved samples, adding labels as a second tag where available, then saving
                for fidx, tagged_doc in enumerate(self.generate_samples()):
                    # Track progress visually
                    if not (fidx + 1) % (self.total_samples // 100):
                        print("o", end="")
                    # add label if available in `tag_label_dict`
                    with suppress(KeyError):
                        tagged_doc.tags.insert(
                            1, tag_label_dict[tagged_doc.tags[0]]
                        )  # always insert after first tag (URL)
                        # remove all but the first 2 tags, second being the label
                        while len(tagged_doc.tags) > 2:
                            tagged_doc.tags.pop()
                    # Serialize the document tokens using pickle and write to the compressed file
                    pickle.dump(tagged_doc, output_file, protocol=pickle.HIGHEST_PROTOCOL)
                print(" - Done.")

    def generate_train_test_sets(
        self, n_samples: int = None, test_frac: float = 0.2, labeled=False, id_corpus=True
    ):
        """
        Generate training and testing sets from the processed data.

        Args:
            n_samples (int): The number of samples to generate (default is None).
            test_frac (float): The fraction of samples to use for testing (default is 0.2).
            id_corpus (bool): Whether to generate samples with integer IDs (default is True).

        Returns:
            Tuple[SampleGenerator, SampleGenerator]: A tuple of training and testing sample generators.
        """
        n_samples = n_samples or self.total_samples
        if labeled:
            fpath = self.save_dir_path / "labeled_corpus.pkl.gz"
        else:
            fpath = self.save_dir_path / f"train_data{('_id' if id_corpus else '')}.pkl.gz"

        # Initialize re-generators
        n_train_samples = round(n_samples * (1 - test_frac))
        train_sample_loader = SampleGenerator(
            fpath, n_train_samples, id_corpus=id_corpus, dct=self.dct
        )
        test_sample_loader = SampleGenerator(
            fpath, n_samples, offset=n_train_samples, id_corpus=id_corpus, dct=self.dct
        )
        return train_sample_loader, test_sample_loader

    def generate_samples(self, *args, **kwargs):
        """
        Generate document samples from the processed data.
        This is essentially a partial function for `generate_train_test_sets`,
        with test_frac=0.0, which returns a single iterator.

        Returns:
            SampleGenerator: A sample generator for the processed data.
        """

        # Initialize and return re-generator
        sample_gen, _ = self.generate_train_test_sets(*args, test_frac=0.0, **kwargs)
        return sample_gen

    def _preprocess_document(self, fpath: Path):
        """
        Preprocess a document from the corpus.

        Args:
            fpath (Path): The path to the document file.

        Returns:
            List[str]: A list of preprocessed tokens from the document.
        """
        # Read all but the header (probably not the most efficient method)
        with open(fpath, "r", encoding="utf-8") as f:
            _, *doc_lines = f.read().split("\n")
            doc = "\n".join(doc_lines)

        # Replace URLs with "<URL>" and email addresses with "<EMAILADD>"
        if self.url_pattern:
            doc = re.sub(self.url_pattern, "<URL>", doc)
        if self.email_pattern:
            doc = re.sub(self.email_pattern, "<EMAILADD>", doc)

        # Tokenize the text
        simp_proc_doc = gensim.utils.simple_preprocess(doc)

        # Remove stopwords and return
        if self.should_filter_stopwords:
            return gensim.parsing.preprocessing.remove_stopword_tokens(simp_proc_doc)
        else:
            return simp_proc_doc
