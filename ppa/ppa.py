import json
import logging
import logging.config
import random
import re
from contextlib import suppress
from copy import deepcopy
from dataclasses import InitVar, dataclass
from functools import partial
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
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from ppa.utils import IndexedFile, get_file_index_path, timer

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
    Generate samples from file.

    Args:
        fpath (Path): The path to the data file.
        n_samples (int): The number of samples to generate.
        offset (int): The offset for skipping samples (default is 0).
        dct (Dictionary): The Gensim Dictionary for ID conversion (default is None).
    """

    fpath: Path
    start_pos_list: List[int] = None
    index_suffix: InitVar[str] = "_idx"

    def __post_init__(self, index_suffix: str):
        """Doc."""

        self.indexed_file = partial(
            IndexedFile,
            self.fpath,
            "read",
            self.start_pos_list,
            index_suffix,
        )

    def __repr__(self):
        return f"SampleGenerator({len(self):,} `TaggedDocument` objects, fpath={self.fpath})"

    def __iter__(self) -> TaggedDocument:
        """
        Iterate over samples in the data file.
        """

        for deserialized_obj in self.indexed_file().read_all():
            yield TaggedDocument(*deserialized_obj)

    def __getitem__(self, pos_idx: int) -> TaggedDocument:
        """Doc."""

        with self.indexed_file(shuffled=False) as idx_input_file:
            return TaggedDocument(*idx_input_file.read_idx(pos_idx))

    def __len__(self):
        """Doc."""

        try:
            return len(self.start_pos_list)
        except TypeError:
            len_ = 0
            for _ in self.indexed_file().read_all():
                len_ += 1
            return len_


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

        # Shuffle the paths (reproducible with seed)
        if self.seed is not None:
            random.seed(self.seed)

    def __repr__(self):
        return f"CorpusProcessor({len(self.fpaths)} docs, seed={self.seed}"

    @timer(1000)
    def process(self, force=False, **kwargs):
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
            print("Loaded existing dictionary.")
            print("Using existing training data.")
        else:
            print(
                f"Processing and saving {self.total_samples:,} TaggedDocument objects to disk: ",
                end="",
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
                    tagged_doc = TaggedDocument(words=tokenized_doc, tags=[fpath.stem])
                    # Serialize the document tokens using pickle and write to the compressed file
                    idx_output_file.write(tagged_doc)
                print(" - Done.")

            # filter tokens
            print(
                "Filtering tokens and re-saving TaggedDocument objects to disk: ",
                end="",
            )
            self._filter_tokens(**kwargs)

            print("Saving Dictionary... ", end="")
            self.dct.save(str(self.dict_path))
            print("Done.")

    def _filter_tokens(
        self,
        n_below: int = 3,
        no_above: float = 1.0,
        min_tokens: int = 10,
        max_tokens: int = 10_000,
        **kwargs,
    ):
        """Doc."""

        # filter words appearing in less then n_below documents, or more then above 'no_above' fraction of documents
        fltrd_dct = deepcopy(self.dct)
        fltrd_dct.filter_extremes(
            no_below=n_below,
            no_above=no_above,
            # keep_n=1_000_000,
            # keep_tokens=None
        )

        # re-initialize the Dictionary
        self.dct = Dictionary()
        with IndexedFile(self.corpus_path, "write") as idx_output_file:
            # Re-iterate over all saved samples, adding labels as a second tag where available, then saving
            for fidx, tagged_doc in enumerate(self.generate_samples()):
                # Track progress visually
                if not (fidx + 1) % (self.total_samples / 10):
                    logging.info(f"{(fidx+1)/(self.total_samples):.1%}... ")

                # remove tokens not in filtered Dictionary
                filtered_tokens = [
                    token_ for token_ in tagged_doc.words if fltrd_dct.token2id.get(token_)
                ]
                # Ignore very short/long documents
                if min_tokens <= len(filtered_tokens) <= max_tokens:
                    # Add to the dictionary
                    self.dct.add_documents([filtered_tokens])
                    tagged_doc = TaggedDocument(words=filtered_tokens, tags=tagged_doc.tags)

                    # Serialize the document tokens using pickle and write to file
                    # TODO: why do I need the 'note=tagged_doc.tags[1]' if it is contained in the tags?
                    try:
                        idx_output_file.write(tagged_doc, note=tagged_doc.tags[1])
                    except IndexError:
                        idx_output_file.write(tagged_doc)

    @timer(1000)
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
            with IndexedFile(self.labeled_corpus_path, "write") as idx_output_file:
                # Re-iterate over all saved samples, adding labels as a second tag where available, then saving
                for fidx, tagged_doc in enumerate(self.generate_samples()):
                    # Track progress visually
                    if not (fidx + 1) % (self.total_samples / 10):
                        logging.info(f"{(fidx+1)/(self.total_samples):.1%}... ")

                    # add label if available in `tag_label_dict`
                    with suppress(KeyError):
                        # always insert after first tag (URL)
                        tagged_doc.tags.insert(1, tag_label_dict[tagged_doc.tags[0]])
                        # remove all but the first 2 tags, second being the label
                        while len(tagged_doc.tags) > 2:
                            tagged_doc.tags.pop()

                    # Serialize the document tokens using pickle and write to file
                    # TODO: why do I need the 'note=tagged_doc.tags[1]' if it is contained in the tags?
                    try:
                        idx_output_file.write(tagged_doc, note=tagged_doc.tags[1])
                    except IndexError:
                        idx_output_file.write(tagged_doc)

                print(" - Done.")

    @timer(1000)
    def generate_train_test_sets(  # noqa: C901
        self,
        fpath: Path = None,
        n_samples: int = None,
        test_frac: float = 0.2,
        labeled=False,
        shuffled=False,
    ):
        """
        Generate training and testing sets from the processed data.

        Args:
            n_samples (int): The number of samples to generate (default is None).
            test_frac (float): The fraction of samples to use for testing (default is 0.2).

        Returns:
            Tuple[SampleGenerator, SampleGenerator]: A tuple of training and testing sample generators.
        """
        n_samples = n_samples or self.total_samples
        if labeled:
            fpath = fpath or self.labeled_corpus_path
        else:
            fpath = fpath or self.corpus_path

        file_idx_path = get_file_index_path(fpath)

        # Calculate the number of training samples
        n_train = int(n_samples * (1 - test_frac))

        if labeled:
            # Sort the index into a dictionary
            index_dict: Dict[str, List[int]] = {"good": [], "bad": [], "unlabeled": []}
            with open(file_idx_path, "r") as idx_file:
                while True:
                    try:
                        start_pos, note = json.loads(idx_file.readline())
                        index_dict[note].append(start_pos)
                    except json.JSONDecodeError:
                        break

            # Shuffle the index (optional) - this means choosing different train/test sets
            if shuffled:
                for start_pos_list in index_dict.values():
                    random.shuffle(start_pos_list)

            # Calculate the number of "good," "bad," and "unlabeled" policies in training set (stratified)
            label_counts_dict = {label: len(list_) for label, list_ in index_dict.items()}
            train_frac = 1 - test_frac
            subset_factor = n_samples / self.total_samples
            train_factor = train_frac * subset_factor
            n_train_good = int(label_counts_dict["good"] * train_factor)
            if not n_train_good:
                n_train_good = int(label_counts_dict["good"] * train_frac)
                n_train_bad = int(label_counts_dict["bad"] * train_frac)
            else:
                n_train_bad = int(label_counts_dict["bad"] * train_factor)
            n_train_unlabeled = int(label_counts_dict["unlabeled"] * train_factor)

            # Collect training set file index
            train_start_positions = []
            # Collect training set file index
            for _ in range(n_train_good):
                train_start_positions.append(index_dict["good"].pop())
            for _ in range(n_train_bad):
                train_start_positions.append(index_dict["bad"].pop())
            for _ in range(n_train_unlabeled):
                train_start_positions.append(index_dict["unlabeled"].pop())

            # use the rest as test set file index
            n_test_good = n_samples - n_train_good
            n_test_bad = n_samples - n_train_bad
            n_test_unlabeled = n_samples - n_train_unlabeled
            try:
                test_start_positions = (
                    index_dict["good"][:n_test_good]
                    + index_dict["bad"][:n_test_bad]
                    + index_dict["unlabeled"][:n_test_unlabeled]
                )
            except KeyError as exc:
                raise RuntimeError(f"Not all types of labels exist! [{exc}]")

        # all unlabeled
        else:
            if shuffled:
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
                random.shuffle(index_list)

                # create train/test file indices
                train_start_positions = index_list[:n_train]
                test_start_positions = index_list[n_train:n_samples]

            # if shuffling isn't needed (such as when processing), no indexing is needed
            else:
                train_start_positions = None
                test_start_positions = []

        # Initialize re-generators
        if not test_start_positions:
            return SampleGenerator(fpath, train_start_positions)
        else:
            return SampleGenerator(fpath, train_start_positions), SampleGenerator(
                fpath, test_start_positions
            )

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

    def _preprocess_document(self, fpath: Path, **kwargs):

        # Read all but the header
        with open(fpath, "r", encoding="utf-8") as f:
            _, *doc_lines = f.readlines()
            doc = "\n".join(doc_lines)[2:]

        # Initialize special tokens list
        special_tokens = {}

        # Define a regular expression pattern to match common date formats
        date_pattern = r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},?\s\d{2,4}\b"
        doc = re.sub(date_pattern, "datetoken", doc)
        special_tokens["datetoken"] = "<DATE>"

        # Find and replace email links with a special token
        email_link_pattern = r"\[([^\]]*?)\]\(mailto:([^\)]*?)\)"
        doc = re.sub(email_link_pattern, "emladdrstoken", doc)
        special_tokens["emladdrstoken"] = "<EMAIL>"

        # Find and replace links with the text inside the square brackets, and add a special token afterwards
        md_link_pattern = r"\[([^\]]*?)\]\(.*?\)"
        doc = re.sub(md_link_pattern, r"\1 urltoken", doc)
        special_tokens["urltoken"] = "<URL>"

        # Find and replace email addresses
        email_pattern = r"[^@ \t\r\n\v\f]+@[^@ \t\r\n\v\f]+\.[^@ \t\r\n\v\f]+"
        doc = re.sub(email_pattern, "emladdrstoken", doc)

        # Replace URLs with "<URL>" and email addresses with "<EMAILADD>"
        url_pattern = (
            r"(https?://)?[a-zA-Z0-9-]+(\.[a-zA-Z]{2,})+(\.[a-zA-Z]{2,})?(/[a-zA-Z0-9-]+)*/?"
        )
        doc = re.sub(url_pattern, "urltoken", doc)

        # Tokenize the text
        tokens = self._tokenize_text(doc, special_tokens, **kwargs)

        return tokens

    def _tokenize_text(
        self, text: str, specials_dict: Dict[str, str], filter_stopwords: bool = True, **kwargs
    ):
        """Doc."""

        hyphenated_tokens = self._get_hyphenated_tokens(text, **kwargs)

        # TODO: Privacy terms / highlighting should be performed here!
        # (before stopwords are removed, which might ruint things such as 'opt out/in')

        # Remove stopwords (optional)
        if filter_stopwords:
            keep_list = ["not"]  # , 'but', 'if', 'and', 'or']
            custom_stopwords = ["ii"]
            stop_words = set(stopwords.words("english")) - set(keep_list) | set(custom_stopwords)
            hyphenated_tokens = [token for token in hyphenated_tokens if token not in stop_words]

        # Remove consecutive duplicates
        tokens = [hyphenated_tokens[0]]
        for i in range(1, len(hyphenated_tokens)):
            if hyphenated_tokens[i] != hyphenated_tokens[i - 1]:
                tokens.append(hyphenated_tokens[i])

        # insert special tokens
        text = " ".join(tokens)
        for k, v in specials_dict.items():
            text = re.sub(k, v, text)
        tokens = text.split()

        return tokens

    def _get_hyphenated_tokens(self, text: str, lemmatize: bool = True, **kwargs):
        """Doc."""

        def unify_tokens(tokens, unifying_token):
            result = []
            i = 0

            while i < len(tokens):
                if tokens[i] == unifying_token:
                    i += 1  # Skip the unifying token
                    continue

                unified_token = tokens[i]

                while i + 2 < len(tokens) and tokens[i + 1] == unifying_token:
                    # Continue building the unified token
                    unified_token += "-" + tokens[i + 2]
                    i += 2  # Skip the next token as it's already unified

                result.append(unified_token)
                i += 1

            return result

        # Use regular expressions to find hyphenated words and replace hyphens with " hyph "
        text_with_hyph = re.sub(r"([A-Za-z]+)-([A-Za-z]+)", r"\1 hyph \2", text)

        # Tokenize the text using simple_preprocess
        tokens = gensim.utils.simple_preprocess(text_with_hyph, min_len=2, max_len=20)

        # Lemmatize (optional)
        if lemmatize:
            # Initialize the WordNet lemmatizer
            lemmatizer = WordNetLemmatizer()
            # Lemmatize the tokens
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Merge tokens with "hyph" between them into hyphenated tokens
        return unify_tokens(tokens, "hyph")
