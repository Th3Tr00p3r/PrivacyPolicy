import json
import logging
import logging.config
import re
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, List
from winsound import Beep

import aiofiles  # type: ignore
import gensim
import httpx
import numpy as np
import pandas as pd
import wordninja
from gensim.corpora import Dictionary
from gensim.models.doc2vec import TaggedDocument
from gensim.models.phrases import Phrases
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from ppa.utils import (
    IndexedFile,
    combine_with_separators,
    concatenate_nearest_neighbors,
    get_file_index_path,
    replace_most_common_phrase,
    timer,
)

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
    """Doc."""

    fpath: Path
    label_index_path: Path
    start_pos_list: List[int]
    shuffled: bool = False
    index_suffix: InitVar[str] = "_idx"
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    text_only: bool = False

    @property
    def labels(self):
        """Doc."""

        return self._labels or self._get_labels()

    def __post_init__(self, index_suffix: str):
        """Doc."""

        self.indexed_file = partial(
            IndexedFile,
            self.fpath,
            "read",
            self.rng,
            self.start_pos_list,
            index_suffix,
        )
        self.in_memory: bool = False
        self._labels: List[str] = []

    def __repr__(self):
        return f"SampleGenerator({len(self):,} `TaggedDocument` objects, fpath={self.fpath})"

    def __iter__(self) -> TaggedDocument:
        """
        Iterate over samples in the data file.
        """

        # samples on disk
        if not self.in_memory:
            for deserialized_obj in self.indexed_file(shuffled=self.shuffled).read_all():
                if not self.text_only:
                    yield TaggedDocument(*deserialized_obj)
                else:
                    yield deserialized_obj[0]

        # samples in RAM
        else:
            if self.shuffled:
                self.rng.shuffle(self._sample_list)
            yield from self._sample_list

    def __getitem__(self, pos_idx: int | slice) -> TaggedDocument:
        """Doc."""

        # samples on disk
        if not self.in_memory:
            # handle slice objects
            if isinstance(pos_idx, slice):
                start, stop, step = pos_idx.indices(len(self))
                pos_idxs: List[int] | range = range(start, stop, step)
            else:
                pos_idxs = [pos_idx]

            # get samples by index/indices
            samples = []
            for pos_idx in pos_idxs:
                with self.indexed_file(shuffled=False) as idx_input_file:
                    samples.append(TaggedDocument(*idx_input_file.read_idx(pos_idx)))

            # if more than one sample, return as list
            if len(samples) > 1:
                return samples
            else:
                return samples[0]

        # samples in RAM (regular list __getitem__)
        else:
            return self._sample_list[pos_idx]

    def __len__(self):
        """Doc."""

        try:
            return len(self.start_pos_list)
        except TypeError:
            len_ = 0
            for _ in self.indexed_file().read_all():
                len_ += 1
            return len_

    def _get_labels(self) -> List[str]:
        """Get labels corresponding to the labels index"""

        urls = [td.tags[0] for td in self]
        url_label_dict = {}
        with open(self.label_index_path, "r") as label_idx_file:
            for line in label_idx_file:
                _, url, label = json.loads(line.strip())
                url_label_dict[url] = label

        self._labels = [url_label_dict[url] for url in urls]
        return self._labels

    def load_to_memory(self):
        """Doc."""

        self._get_labels()
        self._sample_list = list(self)
        self.in_memory = True


@dataclass
class CorpusProcessor:
    """
    Process a corpus of documents into a format suitable for training a model.

    Args:
        fpaths (List[Path]): A list of file paths to the documents.
        save_dir_path (Path): The path to the directory for saving processed data.
        url_pattern (str): A regular expression pattern for matching URLs (default is URL_PATTERN).
        email_pattern (str): A regular expression pattern for matching email addresses (default is EMAIL_PATTERN).
        max_tokens (int): The maximum number of tokens in a document (default is 100,000).
        min_tokens (int): The minimum number of tokens in a document (default is 0).
        seed (int): Seed for reproducible shuffling (default is None).
        dct (Dictionary): The Gensim Dictionary for document ID conversion (default is None).

    Methods:
        process(force=False, **kwargs):
            Process the corpus and save it to the specified directory.
        generate_train_test_sets(n_samples=None, test_frac=0.2):
            Generate training and testing sets from the processed data.
        generate_samples(n_samples=None):
            Generate document samples from the processed data.
    """

    fpaths: List[Path]
    save_dir_path: Path
    seed: int = None
    dct: Dictionary = None

    def __post_init__(self) -> None:
        self.total_samples: int = len(self.fpaths)

        # file paths:
        self.dict_path = self.save_dir_path / "dictionary.pkl"
        self.corpus_path = self.save_dir_path / "corpus.json"
        self.labeled_corpus_path = self.save_dir_path / "labeled_corpus.json"

        # Instantiate a reproducible (if used with integer seed) random number generator for shuffling
        self.rng = np.random.default_rng(self.seed)

    def __repr__(self):
        return f"CorpusProcessor({self.total_samples:,} docs, seed={self.seed})"

    @timer(1000)
    def process(
        self, force: bool = False, filter_tokens: bool = True, bigrams: bool = True, **kwargs
    ):
        """
        Process the corpus and save it to the specified directory.

        Args:
            force (bool): Force reprocessing (default is False).

        Returns:
            None
        """
        if not force:
            # Try loading an existing dictionary
            self.dct = Dictionary.load(str(self.dict_path))
            logging.info("[CorpusProcessor.process] Loaded existing dictionary.")
            logging.info("[CorpusProcessor.process] Using existing training data.")
        else:
            logging.info(
                f"[CorpusProcessor.process] Processing {self.total_samples:,} documents and saving TaggedDocument objects to disk..."
            )
            # Initialize a Dictionary object
            self.dct = Dictionary()
            with IndexedFile(self.corpus_path, "write") as idx_output_file:
                # Re-iterate, this time converting the tokens to integers according to dict ID, then saving
                for fidx, fpath in enumerate(self.fpaths):
                    # Track progress visually
                    if not (fidx + 1) % (self.total_samples / 10):
                        logging.info(f"{(fidx+1)/(self.total_samples):.1%}... ")
                    # Open and process each file
                    tokenized_doc = self._preprocess_document(fpath, **kwargs)
                    # Add to the dictionary
                    self.dct.add_documents([tokenized_doc])
                    # Create a TaggedDocument instance
                    td = TaggedDocument(words=tokenized_doc, tags=[fpath.stem])
                    # Serialize the document tokens using pickle and write to the compressed file
                    idx_output_file.write(td, notes=td.tags)

            # process n-grams
            if bigrams:
                self._unite_bigrams(**kwargs)

            # filter and remove the filtered documents from the total count
            if filter_tokens:
                self.total_samples -= self._filter_tokens(**kwargs)

            logging.info(f"[CorpusProcessor.process] Saving Dictionary to '{self.dict_path}'. ")
            self.dct.save(str(self.dict_path))

    def _unite_bigrams(
        self,
        threshold: float = 0.1,
        min_count: int = None,
        **kwargs,
    ) -> None:
        """Doc."""

        if not min_count:
            n_total_tokens = self.dct.num_pos
            n_unique_tokens = len(self.dct)
            min_count = round(n_total_tokens / n_unique_tokens)

        logging.info(f"[CorpusProcessor._unite_bigrams] Getting bigrams (min_count={min_count})...")
        bigram = Phrases(
            self.generate_samples(text_only=True),
            min_count=min_count,
            threshold=threshold,
            scoring="npmi",
            delimiter="",
        )

        #         self.sorted_ngrams = dict(sorted(bigram.export_phrases().items(), key=lambda item: item[1])) # TESTESTEST
        #         print("ngrams:\n-------", "\n".join(f"{k}: {v}" for k, v in self.sorted_ngrams.items())) # TESTESTEST

        n_unique_tokens_before = len(self.dct)

        # re-initialize the Dictionary
        self.dct = Dictionary()
        logging.info("[CorpusProcessor._unite_bigrams] Re-processing...")
        with IndexedFile(self.corpus_path, "write") as idx_output_file:
            # Re-iterate over all saved samples, adding labels as a second tag where available, then saving
            for fidx, td in enumerate(self.generate_samples()):
                # Track progress visually
                if not (fidx + 1) % (self.total_samples / 10):
                    logging.info(f"{(fidx+1)/(self.total_samples):.1%}... ")

                # Convert the document using the `Phrases` object
                words_with_bigrams = bigram[td.words]

                # Add to the new dictionary
                self.dct.add_documents([words_with_bigrams])
                td = TaggedDocument(words=words_with_bigrams, tags=td.tags)
                # Serialize the document tokens using JSON and write to file
                idx_output_file.write(td, notes=td.tags)
        logging.info(
            f"[CorpusProcessor._unite_bigrams] Unique token delta: {len(self.dct) - n_unique_tokens_before}"
        )

    def _filter_tokens(
        self,
        n_below: int = 5,
        no_above: float = 0.99,
        min_percentile: int = 5,
        max_percentile: int = 95,
        **kwargs,
    ) -> int:
        """Doc."""

        # get document lengths by sampling 1000 random document, and get the min/max tokens utilizing percentiles
        doc_lengths = [
            len(tagged_doc.words)
            for tagged_doc in self.generate_samples(
                n_samples=1000, shuffled_idx=True, shuffled_gen=True
            )
        ]
        min_tokens, max_tokens = np.percentile(
            doc_lengths, [min_percentile, max_percentile]
        ).astype(int)
        logging.info(
            f"[CorpusProcessor._filter_tokens] Document length range: {min_tokens}-{max_tokens} tokens."
        )
        logging.info(
            f"[CorpusProcessor._filter_tokens] Required token appearance range: {n_below:,}-{round(no_above * self.dct.num_docs):,} documents."
        )

        n_unique_tokens_before = len(self.dct)

        # filter words appearing in less then n_below documents, or more then above 'no_above' fraction of documents
        fltrd_dct = deepcopy(self.dct)
        fltrd_dct.filter_extremes(
            no_below=n_below,
            no_above=no_above,
        )

        n_docs_filtered = 0
        # re-initialize the Dictionary
        self.dct = Dictionary()
        logging.info("[CorpusProcessor._filter_tokens] Filtering tokens and document lengths...")
        with IndexedFile(self.corpus_path, "write") as idx_output_file:
            # Re-iterate over all saved samples, adding labels as a second tag where available, then saving
            for fidx, td in enumerate(self.generate_samples()):
                # Track progress visually
                if not (fidx + 1) % (self.total_samples / 10):
                    logging.info(f"{(fidx+1)/(self.total_samples):.1%}... ")

                # remove tokens not in filtered Dictionary
                filtered_tokens = [
                    token_ for token_ in td.words if fltrd_dct.token2id.get(token_) is not None
                ]
                filtered_tokens = self._remove_consecutive_duplicates(filtered_tokens)

                # Ignore very short/long documents
                if min_tokens <= len(filtered_tokens) <= max_tokens:
                    # Add to the new dictionary
                    self.dct.add_documents([filtered_tokens])
                    td = TaggedDocument(words=filtered_tokens, tags=td.tags)
                    # Serialize the document tokens using JSON and write to file
                    idx_output_file.write(td, notes=td.tags)

                else:
                    n_docs_filtered += 1

        # Filter once more, to get rid of leftover '< n_below' tokens after the doc length filtering
        # filter words appearing in less then n_below documents, or more then above 'no_above' fraction of documents
        fltrd_dct = deepcopy(self.dct)
        fltrd_dct.filter_extremes(
            no_below=n_below,
            no_above=no_above,
        )
        self.dct = Dictionary()
        logging.info("[CorpusProcessor._filter_tokens] Re-filtering tokens...")
        with IndexedFile(self.corpus_path, "write") as idx_output_file:
            # Re-iterate over all saved samples, adding labels as a second tag where available, then saving
            for fidx, td in enumerate(self.generate_samples()):
                # Track progress visually
                if not (fidx + 1) % (self.total_samples / 10):
                    logging.info(f"{(fidx+1)/(self.total_samples):.1%}... ")

                # remove tokens not in filtered Dictionary
                filtered_tokens = [
                    token_ for token_ in td.words if fltrd_dct.token2id.get(token_) is not None
                ]
                filtered_tokens = self._remove_consecutive_duplicates(filtered_tokens)
                # Add to the new dictionary
                self.dct.add_documents([filtered_tokens])
                td = TaggedDocument(words=filtered_tokens, tags=td.tags)
                # Serialize the document tokens using JSON and write to file
                idx_output_file.write(td, notes=td.tags)

        logging.info(
            f"[CorpusProcessor._filter_tokens] Filtered {n_docs_filtered:,} documents and {n_unique_tokens_before - len(self.dct):,}/{n_unique_tokens_before:,} ({(n_unique_tokens_before - len(self.dct))/n_unique_tokens_before:.1%}) unique tokens."
        )

        return n_docs_filtered

    def dict_info(self):
        """Doc."""

        print(
            f"Dictionary was created by processing {self.dct.num_pos:,} tokens from a corpus of {self.dct.num_docs:,} documents."
        )
        print(f"It contains {len(self.dct):,} unique tokens.")
        print(
            f"Each document, on average, contains {self.dct.num_nnz // self.dct.num_docs:,} unique tokens."
        )

    @timer(1000)
    def add_label_tags(self, tag_label_dict: Dict[str, str], force=False):
        """Doc."""

        if not force:
            # TODO: check also if file exists! if it doesn't, do the labeling even if force=False!
            logging.info("[CorpusProcessor.add_label_tags] Using existing labeled data.")

        else:
            logging.info(
                f"[CorpusProcessor.add_label_tags] Adding labels and saving {self.total_samples:,} TaggedDocument objects to disk...",
            )
            with IndexedFile(self.labeled_corpus_path, "write") as idx_output_file:
                # Re-iterate over all saved samples, adding labels as a second tag where available, then saving
                for fidx, td in enumerate(self.generate_samples()):
                    # Track progress visually
                    if not (fidx + 1) % (self.total_samples / 10):
                        logging.info(f"{(fidx+1)/(self.total_samples):.1%}... ")

                    # add label if available in `tag_label_dict`
                    try:
                        # always insert after first tag (URL)
                        td.tags.insert(1, tag_label_dict[td.tags[0]])
                    except KeyError:
                        td.tags.insert(1, "unlabeled")

                    # Serialize the document tokens using pickle and write to file
                    idx_output_file.write(td, notes=td.tags)

                print("Done.")

    @timer(1000)
    def generate_train_test_sets(  # noqa: C901
        self,
        fpath: Path = None,
        test_frac: float = 0.2,
        labeled=False,
        shuffled_idx=False,
        shuffled_gen=False,
        n_samples: int = None,
        text_only: bool = False,
    ):
        """
        Generate training and testing sets from the processed data.

        Args:
            n_samples (int): The number of samples to generate (default is None).
            test_frac (float): The fraction of samples to use for testing (default is 0.2).

        Returns:
            Tuple[SampleGenerator, SampleGenerator]: A tuple of training and testing sample generators.
        """

        if labeled:
            fpath = fpath or self.labeled_corpus_path
        else:
            fpath = fpath or self.corpus_path

        file_idx_path = get_file_index_path(fpath)

        n_samples = n_samples or self.total_samples

        if labeled:
            # Sort the index into a dictionary
            index_dict: Dict[str, List[int]] = {"good": [], "bad": [], "unlabeled": []}
            with open(file_idx_path, "r") as idx_file:
                while True:
                    try:
                        start_pos, _, label = json.loads(idx_file.readline())
                        index_dict[label].append(start_pos)
                    except json.JSONDecodeError:
                        break

            # Shuffle the index (optional) - this means choosing different train/test sets
            if shuffled_idx:
                for index_list in index_dict.values():
                    self.rng.shuffle(index_list)

            # Count the number of "good," "bad," and "unlabeled" policies in training set (stratified)
            label_counts_dict = {label: len(list_) for label, list_ in index_dict.items()}

            # trim each list according to the requested number of samples and its own length
            if n_samples < self.total_samples:
                sample_frac = n_samples / self.total_samples
                for label in index_dict.keys():
                    n_reduced = int(label_counts_dict[label] * sample_frac)
                    index_dict[label] = index_dict[label][:n_reduced]

                # Recount needed after trimming
                label_counts_dict = {label: len(list_) for label, list_ in index_dict.items()}

            # Collect test set file index (optional)
            test_start_positions = []
            if test_frac:
                n_test_good = int(label_counts_dict["good"] * test_frac)
                if not n_test_good:
                    raise RuntimeError("No 'good' samples in the test set - check your data!")
                n_test_bad = int(label_counts_dict["bad"] * test_frac)
                n_test_unlabeled = int(label_counts_dict["unlabeled"] * test_frac)

                # Collect training set file index
                for _ in range(n_test_good):
                    test_start_positions.append(index_dict["good"].pop())
                for _ in range(n_test_bad):
                    test_start_positions.append(index_dict["bad"].pop())
                for _ in range(n_test_unlabeled):
                    test_start_positions.append(index_dict["unlabeled"].pop())

            # use the rest as training set file index
            train_start_positions = index_dict["good"] + index_dict["bad"] + index_dict["unlabeled"]

            # Shuffle again so that labeled samples are mixed in with the unlabeled samples
            self.rng.shuffle(train_start_positions)
            self.rng.shuffle(test_start_positions)

        # all unlabeled
        else:
            # Get the entire file index as a list
            index_list = []
            with open(file_idx_path, "r") as idx_file:
                while True:
                    try:
                        start_pos, _ = json.loads(idx_file.readline())
                        index_list.append(start_pos)
                    except json.JSONDecodeError:
                        break

            # Shuffle the index (optional) - this means choosing different train/test sets
            if shuffled_idx:
                self.rng.shuffle(index_list)

            # create train/test file indices
            n_train = int(n_samples * (1 - test_frac))
            train_start_positions = index_list[:n_train]
            test_start_positions = index_list[n_train:n_samples]

        # Initialize re-generators
        label_index_path = get_file_index_path(self.labeled_corpus_path)
        if not test_start_positions:
            return SampleGenerator(
                fpath,
                label_index_path,
                train_start_positions,
                shuffled=shuffled_gen,
                text_only=text_only,
            )
        else:
            return SampleGenerator(
                fpath,
                label_index_path,
                train_start_positions,
                shuffled=shuffled_gen,
            ), SampleGenerator(fpath, label_index_path, test_start_positions, shuffled=shuffled_gen)

    def generate_samples(self, *args, **kwargs):
        """
        Generate document samples from the processed data.
        This is essentially a partial function for `generate_train_test_sets`,
        with test_frac=0.0, which returns a single iterator.

        Returns:
            SampleGenerator: A sample generator for the processed data.
        """

        # Initialize and return re-generator
        return self.generate_train_test_sets(*args, test_frac=0.0, **kwargs)

    def _preprocess_document(self, fpath: Path, separators=["", " ", "-"], **kwargs):

        # Read all but the header
        with open(fpath, "r", encoding="utf-8") as f:
            _, *doc_lines = f.readlines()
            doc = "\n".join(doc_lines)[2:]

        # Replace ordinal numbers (1st, 2nd, 3rd, etc.) with their word equivalents
        ordinal_pattern = r"\b([1-9]|10)(st|nd|rd|th)\b"
        doc = re.sub(
            ordinal_pattern, lambda match: self._convert_ordinal_to_words(match.group(1)), doc
        )

        # Initialize special tokens list
        special_tokens = {}

        # Define a regular expression pattern to match common date formats
        date_pattern = r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},?\s\d{2,4}\b"
        doc = re.sub(date_pattern, "_datetoken_", doc)
        special_tokens["datetoken"] = "<DATE>"

        # Find and replace email links with a special token
        email_link_pattern = r"\[([^\]]*?)\]\(mailto:([^\)]*?)\)"
        doc = re.sub(email_link_pattern, "_emladdrstoken_", doc)
        special_tokens["emladdrstoken"] = "<EMAIL>"

        # Find and replace links with the text inside the square brackets, and add a special token afterwards
        md_link_pattern = r"\[([^\]]*?)\]\(.*?\)"
        doc = re.sub(md_link_pattern, r"\1 _urltoken_", doc)
        special_tokens["urltoken"] = "<URL>"

        # Find and replace email addresses
        email_pattern = r"[^@ \t\r\n\v\f]+@[^@ \t\r\n\v\f]+\.[^@ \t\r\n\v\f]+"
        doc = re.sub(email_pattern, "_emladdrstoken_", doc)

        # Replace URLs with "<URL>" and email addresses with "<EMAILADD>"
        url_pattern = (
            r"(https?://)?[a-zA-Z0-9-]+(\.[a-zA-Z]{2,})+(\.[a-zA-Z]{2,})?(/[a-zA-Z0-9-]+)*/?"
        )
        doc = re.sub(url_pattern, "_urltoken_", doc)

        # Look for possible appearances of company name using the fpath/URL
        possible_company_words = concatenate_nearest_neighbors(
            wordninja.split(fpath.name.split(".")[0]), 4
        )
        possbile_company_names = combine_with_separators(possible_company_words, separators)
        doc = replace_most_common_phrase(possbile_company_names, doc, "_companynametoken_")
        special_tokens["companynametoken"] = "<COMPANY>"

        # remove underscores (which are caused by Markdown)
        doc = re.sub("_", " ", doc)

        # Tokenize the text
        tokens = self._tokenize_text(doc, special_tokens, **kwargs)

        return tokens

    def _tokenize_text(
        self,
        text: str,
        specials_dict: Dict[str, str],
        lemmatize: bool = True,
        filter_stopwords: bool = True,
        **kwargs,
    ):
        """Doc."""

        # Tokenize the text using simple_preprocess
        tokens = gensim.utils.simple_preprocess(text, min_len=2, max_len=20)

        # insert special tokens
        text = " ".join(tokens)
        for k, v in specials_dict.items():
            text = re.sub(k, v, text)
        tokens = text.split()

        # Lemmatize (optional)
        if lemmatize:
            # Initialize the WordNet lemmatizer
            lemmatizer = WordNetLemmatizer()
            # Lemmatize the tokens
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # TODO: Privacy terms / highlighting should be performed here!
        # (before stopwords are removed, which might ruint things such as 'opt out/in')

        # Remove stopwords (optional)
        if filter_stopwords:
            keep_list = ["not"]  # , 'but', 'if', 'and', 'or']
            custom_stopwords = ["ii"]
            stop_words = set(stopwords.words("english")) - set(keep_list) | set(custom_stopwords)
            tokens = [token for token in tokens if token not in stop_words]

        # Remove consecutive duplicates
        tokens = self._remove_consecutive_duplicates(tokens)

        return tokens

    def _remove_consecutive_duplicates(self, tokens: List[str]):
        """Doc."""

        no_dups = [tokens[0]]
        for i in range(1, len(tokens)):
            if tokens[i] != tokens[i - 1]:
                no_dups.append(tokens[i])
        return no_dups

    def _convert_ordinal_to_words(self, number):
        ordinals = {
            "1": "first",
            "2": "second",
            "3": "third",
            "4": "fourth",
            "5": "fifth",
            "6": "sixth",
            "7": "seventh",
            "8": "eighth",
            "9": "ninth",
            "10": "tenth",
        }
        return ordinals.get(number, number)
