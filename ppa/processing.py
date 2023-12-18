import json
import logging
import logging.config
import re
from collections import Counter
from copy import copy, deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import aiofiles  # type: ignore
import gensim
import httpx
import nltk
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
    longest_most_common_phrase,
    timer,
)

# Define paths
CURRENT_DPATH = Path(__file__).resolve().parent

RAW_DATA_FPATH = CURRENT_DPATH / "notebooks" / "tosdr_raw.json"
DATABASE_FPATH = CURRENT_DPATH / "notebooks" / "tosdr_db.json"

CORPUS_DPATH = CURRENT_DPATH / "corpus"
CORPUS_FPATH = CORPUS_DPATH / "corpus.json"
DICT_FPATH = CORPUS_DPATH / "dictionary.pkl"
BIGRAMS_FPATH = CORPUS_DPATH / "bigrams.pkl"

# Download NLTK datasets (if necessary)
nltk.data.path.append(str(CORPUS_DPATH))  # set the path to the location in the repo
# Check if wordnet dataset is present, if not, download it
if not (CORPUS_DPATH / "corpora" / "wordnet.zip").exists():
    nltk.download("wordnet", download_dir=CORPUS_DPATH)

# Check if stopwords dataset is present, if not, download it
if not (CORPUS_DPATH / "corpora" / "stopwords.zip").exists():
    nltk.download("stopwords", download_dir=CORPUS_DPATH)


@dataclass
class SampleGenerator:
    """
    Generates samples from a data file based on provided parameters.

    Parameters
    ----------
    corpus_fpath : Path
        Path to the data file.
    start_pos_list : List[int]
        List of starting positions in the file for sampling.
    rng : np.random.Generator, optional
        Random number generator, by default np.random.default_rng().
    tokens_only : bool, optional
        Flag to indicate sampling text only, by default False.
    """

    corpus_fpath: Path
    start_pos_list: List[int] = None
    labeled: bool = False
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    tokens_only: bool = False
    positive_keys: List[str] = None
    negative_keys: List[str] = None

    @property
    def labels(self) -> List[str]:
        """
        Retrieve labels sorted in the sample order.

        Returns
        -------
        List[str]
            List of labels.
        """

        return self._labels or self._get_keys_and_labels()[1]

    @property
    def keys(self) -> List[str]:
        """
        Retrieve keys sorted in the sample order.

        Returns
        -------
        List[str]
            List of labels.
        """

        return self._keys or self._get_keys_and_labels()[0]

    def __post_init__(self) -> None:
        """
        Initialize SampleGenerator.
        """

        if self.positive_keys and self.negative_keys:
            raise ValueError("Only one of `positive_keys` or `positive_keys` can be supplied!")

        # initialize indexed file
        self.indexed_file = IndexedFile(
            self.corpus_fpath,
            "read",
            self.start_pos_list,
        )
        # initialize empties for labels and keys (obtained through property methods)
        self._labels: List[str] = []
        self._keys: List[str] = []

        # initialize using keys instead of file start positions
        if self.positive_keys or self.negative_keys:
            # initialize file start positions
            self.start_pos_list = []
            # use supplied keys
            if self.positive_keys:
                for key in self.positive_keys:
                    try:
                        pos, _ = self.indexed_file.key2poslabel[key]
                    except KeyError:
                        pass
                    else:
                        self.start_pos_list.append(pos)
            # use all but suplied keys
            if self.negative_keys:
                valid_keys = set(self.indexed_file.key2poslabel.keys()) - set(self.negative_keys)
                for key in valid_keys:
                    pos, _ = self.indexed_file.key2poslabel[key]
                    self.start_pos_list.append(pos)

            # re-initialize indexed file with the created 'start_pos_list'
            self.indexed_file = IndexedFile(
                self.corpus_fpath,
                "read",
                self.start_pos_list,
            )

    def __repr__(self) -> str:
        """
        Return a string representation of SampleGenerator.

        Returns
        -------
        str
            String representation.
        """
        counter_str = ", ".join([f"{k}: {v:,}" for k, v in Counter(self.labels).items()])
        return f"SampleGenerator({len(self):,} `TaggedDocument` objects ({counter_str}), corpus_fpath={self.corpus_fpath})"

    def sample(
        self, n: int = None, idxs: List[int] | np.ndarray = None, tokens_only=None
    ) -> "SampleGenerator":
        """
        Generate a new SampleGenerator instance with sampled data.

        Parameters
        ----------
        n : int, optional
            Number of samples to retrieve, by default None.
        idxs : List[int] | np.ndarray, optional
            List of indices to sample, by default None.

        Returns
        -------
        SampleGenerator
            Sampled instance of SampleGenerator.
        """

        start_pos_list = copy(self.start_pos_list)
        if idxs is not None:
            start_pos_list = [pos for idx, pos in enumerate(start_pos_list) if idx in idxs]
        if n:
            self.rng.shuffle(start_pos_list)
            start_pos_list = start_pos_list[:n]

        return SampleGenerator(
            self.corpus_fpath,
            start_pos_list,
            self.labeled,
            self.rng,
            tokens_only or self.tokens_only,
        )

    def get_labeled(self, label=None, tokens_only=None) -> "SampleGenerator":
        """Doc."""

        if not label:
            labeled_idxs = [
                self.index(key)
                for key, label_ in zip(self.keys, self.labels)
                if label_ != "unlabeled"
            ]
        else:
            labeled_idxs = [
                self.index(key) for key, label_ in zip(self.keys, self.labels) if label_ == label
            ]
        return self.sample(idxs=labeled_idxs, tokens_only=tokens_only)

    def __iter__(self) -> TaggedDocument:
        """
        Iterate over samples in the data file.

        Yields
        ------
        TaggedDocument
            Yielded samples.
        """

        for tokens, key, label in self.indexed_file.read_all():
            if not self.tokens_only:
                yield TaggedDocument(
                    tokens, ([key, label] if (label != "unlabeled") and self.labeled else [key])
                )
            else:
                yield tokens

    def __getitem__(self, pos_idx: int | slice | str) -> "TaggedDocument" | List["TaggedDocument"]:
        """
        Retrieve samples by position index or slice.

        Parameters
        ----------
        pos_idx : int | slice | str
            Index or slice to retrieve samples.

        Returns
        -------
        TaggedDocument | List[TaggedDocument]
            Retrieved sample(s).
        """

        # handle slice objects
        if isinstance(pos_idx, slice):
            start, stop, step = pos_idx.indices(len(self))
            pos_idxs: List[int | str] | range = range(start, stop, step)
        else:
            pos_idxs = [pos_idx]

        # get samples by index/indices
        samples = []
        for pos_idx in pos_idxs:
            # convert key to index (optional getting by URL)
            if isinstance(pos_idx, str):
                pos_idx = self.indexed_file.key_to_pos_idx(pos_idx)
            with self.indexed_file as idx_input_file:
                tokens, key, label = idx_input_file.read_idx(pos_idx)
                samples.append(
                    TaggedDocument(
                        tokens, ([key, label] if (label != "unlabeled") and self.labeled else [key])
                    )
                )

        # if more than one sample, return as list
        if len(samples) > 1:
            return samples
        else:
            return samples[0]

    def index(self, key) -> int:
        """Get index of key"""

        return self.indexed_file.key_to_pos_idx(key)

    def __len__(self) -> int:
        """
        Return the number of samples in SampleGenerator.

        Returns
        -------
        int
            Number of samples.
        """

        try:
            return len(self.start_pos_list)
        except TypeError:
            len_ = 0
            for _ in self.indexed_file.read_all():
                len_ += 1
            return len_

    def _get_keys_and_labels(self) -> Tuple[List[str], List[str]]:
        """
        Retrieve labels corresponding to the labels index.

        Returns
        -------
        List[str]
            List of labels.
        """

        pos2keylabel: Dict[int, Tuple[str, str]] = {}
        with open(self.indexed_file.idx_fpath, "r") as label_idx_file:
            for line in label_idx_file:
                pos, key, label = json.loads(line.strip())
                pos2keylabel[pos] = key, label

        self._keys, self._labels = map(  # type: ignore
            list, zip(*[pos2keylabel[pos] for pos in self.start_pos_list])
        )
        return self._keys, self._labels


@dataclass
class CorpusProcessor:
    """
    Process and manipulate a corpus of text documents.
    """

    corpus_fpath: Path = CORPUS_FPATH
    dict_fpath: Path = DICT_FPATH
    bigrams_fpath: Path = BIGRAMS_FPATH

    def __post_init__(self) -> None:
        """
        Initialize the CorpusProcessor after the object creation.
        """

        try:
            self.dct = Dictionary.load(str(self.dict_fpath))
        except FileNotFoundError:
            logging.warning("Dictionary missing!")
        try:
            self.bigram = Dictionary.load(str(self.bigrams_fpath))
        except FileNotFoundError:
            logging.warning("FrozenPhrases (bigrams) missing!")

    def __repr__(self):
        """
        Return a string representation of the CorpusProcessor object.
        """

        return f"CorpusProcessor({self.dct.num_docs:,} docs)"

    @timer(1000)
    def process_corpus(
        self, fpaths: List[Path], filter_tokens: bool = True, bigrams: bool = True, **kwargs
    ):
        """
        Process the corpus for dictionary creation and tokenization.

        Parameters
        ----------
        filter_tokens : bool, optional
            Toggle token filtering, by default True.
        bigrams : bool, optional
            Toggle bigram creation, by default True.
        """

        n_paths = len(fpaths)

        logging.info(
            f"[{self.__class__.__name__}.process_corpus] Processing {n_paths:,} documents and serializing to disk..."
        )
        # Initialize a Dictionary object
        self.dct = Dictionary()
        with IndexedFile(self.corpus_fpath, "write") as corpus_file:
            # Re-iterate, this time converting the tokens to integers according to dict ID, then saving
            for fidx, fpath in enumerate(fpaths):
                # Track progress visually
                if not (fidx + 1) % (n_paths / 10):
                    logging.info(f"{(fidx+1)/n_paths:.1%}... ")
                # Open and process each file
                tokenized_doc, domain_name = self._preprocess_document(fpath, **kwargs)
                # Add to the dictionary
                self.dct.add_documents([tokenized_doc])
                # Serialize the document tokens using pickle and write to file
                corpus_file.write(tokenized_doc, notes=(domain_name, "unlabeled"))

        # process n-grams
        if bigrams:
            self._bigram_corpus(**kwargs)

        # filter and remove the filtered documents from the total count
        if filter_tokens:
            self._filter_corpus(**kwargs)

        # save
        self.dct.save(str(self.dict_fpath))

    def process_document(self, doc: str, url: str) -> TaggedDocument:
        """Process a single document based on Dictionary and FrozenPhrases objects learned form entire corpus."""

        # tokenize
        tokenized_doc, domain_name = self._preprocess_document(doc=doc, url=url)

        # Deal with unhyphenated bigrams
        bigram_tokenized_doc = self._bigram_tokens(tokenized_doc)

        # filter according to master Dictionary
        filtered_tokenized_doc = [
            token_ for token_ in bigram_tokenized_doc if self.dct.token2id.get(token_) is not None
        ]

        # remove consecutive duplicates after filtering
        final_tokenized_doc = self._remove_consecutive_duplicates(filtered_tokenized_doc)

        return TaggedDocument(final_tokenized_doc, [domain_name])

    def membership_test(self, tokenized_doc: List[str], verbose=False) -> float:
        """
        Estimate if a tokenized document belongs to the type/group represented by the corpus.
        The idea is that since the corpus Dictionary's IDs are ordered according to prevalence, outlier documents
        or documents of a type different from the corpus will inevitably have either a different distribution of common tokens.
        """

        # Length ratio (penalizing short final lengths)
        len_ratio = 1 / (1 + len(tokenized_doc) / 4)

        # In-dictionary factor (penalzing documents with token requencies differing from those in the Dictionary)
        token_id_counts = Counter([idx for idx in self.dct.doc2idx(tokenized_doc)])
        median_common_id = np.median(
            sorted(token_id_counts, key=lambda k: token_id_counts[k], reverse=True)[:15]
        )
        median2total_ratio = median_common_id / (len(self.dct) / 3)

        if verbose:
            print(
                f"\nCorpusProcessor.membership_test ratios:\nlen_ratio={len_ratio:.3f}\nmedian2total_ratio={median2total_ratio:.3f}"
            )
            print(
                f"\nCorpusProcessor.membership_test factors:\nlen_ratio={np.exp(-len_ratio):.3f}\nmedian2total_ratio={np.exp(-median2total_ratio):.3f}\n"
            )

        # combine factors in decaying exponent
        return np.exp(-(median2total_ratio + len_ratio))

    def _bigram_corpus(
        self,
        bigram_threshold: float = 0.1,
        min_count: int = None,
        **kwargs,
    ) -> None:
        """
        Generate and unite bigrams in the corpus.

        Parameters
        ----------
        bigram_threshold : float, optional
            Threshold for bigram detection, by default 0.1.
        min_count : int, optional
            Minimum count for bigram creation, by default None.
        """

        if not min_count:
            n_total_tokens = self.dct.num_pos
            n_unique_tokens = len(self.dct)
            min_count = round(n_total_tokens / n_unique_tokens)

        # train Phrases mode on entire corpus
        logging.info(
            f"[{self.__class__.__name__}._bigram_corpus] Getting bigrams (min_count={min_count})..."
        )
        self.bigram = Phrases(
            self.generate_samples(tokens_only=True),
            min_count=min_count,
            threshold=bigram_threshold,
            scoring="npmi",
            delimiter="-",
        ).freeze()

        # save to disk
        self.bigram.save(str(self.bigrams_fpath))

        n_unique_tokens_before = len(self.dct)

        # re-initialize the Dictionary
        self.dct = Dictionary()
        # keep the hyphenated (regular) bigrams and un-hyphenated bigrams in lists to avoid re-computation for each document
        unhyphenated_bigrams = [bigram.replace("-", "") for bigram in self.bigram.phrasegrams]
        hyphenated_bigrams = list(self.bigram.phrasegrams.keys())
        logging.info(f"[{self.__class__.__name__}._bigram_corpus] Re-processing...")
        with IndexedFile(self.corpus_fpath, "write") as corpus_file:
            # Re-iterate over all saved samples, adding labels as a second tag where available, then saving
            for fidx, td in enumerate(all_samples := self.generate_samples()):
                # Track progress visually
                if not (fidx + 1) % (len(all_samples) / 10):
                    logging.info(f"{(fidx+1)/(len(all_samples)):.1%}... ")

                # Unite bigrams in document
                bigram_tokens = self._bigram_tokens(
                    td.words, unhyphenated_bigrams, hyphenated_bigrams
                )

                # Add to the new dictionary
                self.dct.add_documents([bigram_tokens])
                # Serialize the document tokens using JSON and write to file
                corpus_file.write(bigram_tokens, notes=td.tags)
        logging.info(
            f"[{self.__class__.__name__}._bigram_corpus] Unique token delta: {len(self.dct) - n_unique_tokens_before}"
        )

    def _filter_corpus(
        self,
        n_below: int = 10,
        no_above: float = 0.99,
        min_percentile: int = 1,
        max_percentile: int = 99,
        **kwargs,
    ) -> None:
        """
        Filter tokens based on document statistics.

        Parameters
        ----------
        n_below : int, optional
            Tokens appearing below this number are filtered, by default None.
        no_above : float, optional
            Fraction of documents for token filtering, by default 0.99.
        min_percentile : int, optional
            Minimum percentile for token filtering, by default 1.
        max_percentile : int, optional
            Maximum percentile for token filtering, by default 99.

        Returns
        -------
        int
            Number of documents filtered.
        """

        # get document lengths by sampling 1000 random document, and get the min/max tokens utilizing percentiles
        doc_lengths = [
            len(tagged_doc.words)
            for tagged_doc in self.generate_samples(
                n_samples=1000,
            )
        ]
        min_tokens, max_tokens = np.percentile(
            doc_lengths, [min_percentile, max_percentile]
        ).astype(int)
        logging.info(
            f"[{self.__class__.__name__}._filter_corpus] Document length range: {min_tokens}-{max_tokens} tokens."
        )
        n_below = n_below or round(self.dct.num_docs ** (1 / 3))
        logging.info(
            f"[{self.__class__.__name__}._filter_corpus] Required token appearance range: {n_below:,}-{round(no_above * self.dct.num_docs):,} documents."
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
        logging.info(
            f"[{self.__class__.__name__}._filter_corpus] Filtering tokens and document lengths..."
        )
        with IndexedFile(self.corpus_fpath, "write") as corpus_file:
            # Re-iterate over all saved samples, adding labels as a second tag where available, then saving
            for fidx, td in enumerate(all_samples := self.generate_samples()):
                # Track progress visually
                if not (fidx + 1) % (len(all_samples) / 10):
                    logging.info(f"{(fidx+1)/(len(all_samples)):.1%}... ")

                # remove tokens not in filtered Dictionary
                filtered_tokens = [
                    token_ for token_ in td.words if fltrd_dct.token2id.get(token_) is not None
                ]
                filtered_tokens = self._remove_consecutive_duplicates(filtered_tokens)

                # Ignore very short/long documents
                if min_tokens <= len(filtered_tokens) <= max_tokens:
                    # Add to the new dictionary
                    self.dct.add_documents([filtered_tokens])
                    # Serialize the document tokens using JSON and write to file
                    corpus_file.write(filtered_tokens, notes=td.tags)

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
        logging.info(f"[{self.__class__.__name__}._filter_corpus] Re-filtering tokens...")
        with IndexedFile(self.corpus_fpath, "write") as corpus_file:
            # Re-iterate over all saved samples, adding labels as a second tag where available, then saving
            for fidx, td in enumerate(all_samples := self.generate_samples()):
                # Track progress visually
                if not (fidx + 1) % (len(all_samples) / 10):
                    logging.info(f"{(fidx+1)/(len(all_samples)):.1%}... ")

                # remove tokens not in filtered Dictionary
                filtered_tokens = [
                    token_ for token_ in td.words if fltrd_dct.token2id.get(token_) is not None
                ]
                filtered_tokens = self._remove_consecutive_duplicates(filtered_tokens)
                # Add to the new dictionary
                self.dct.add_documents([filtered_tokens])
                # Serialize the document tokens using JSON and write to file
                corpus_file.write(filtered_tokens, notes=td.tags)

        logging.info(
            f"[{self.__class__.__name__}._filter_corpus] Filtered {n_docs_filtered:,} documents and {n_unique_tokens_before - len(self.dct):,}/{n_unique_tokens_before:,} ({(n_unique_tokens_before - len(self.dct))/n_unique_tokens_before:.1%}) unique tokens."
        )

    def _bigram_tokens(
        self,
        tokenized_doc: List[str],
        unhyphenated_bigrams: List[str] = None,
        hyphenated_bigrams: List[str] = None,
    ):
        """Doc."""

        # get unhyphenated bigrams
        unhyphenated_bigrams = unhyphenated_bigrams or [
            bigram.replace("-", "") for bigram in self.bigram.phrasegrams
        ]
        # keep the bigrams as a list
        bigrams = hyphenated_bigrams or list(self.bigram.phrasegrams.keys())

        # find bigrams in the document using ForzenPhrases
        bigram_tokenized_doc = self.bigram[tokenized_doc]

        # Deal with unhyphenated bigrams
        corrected_bigram_tokenized_doc = []
        for token_ in bigram_tokenized_doc:
            try:
                bigram_idx = unhyphenated_bigrams.index(token_)
            except ValueError:
                # token_ is a regular word or bigram
                corrected_bigram_tokenized_doc.append(token_)
            else:
                # token is an unhyphenated bigram!
                corrected_bigram_tokenized_doc.append(bigrams[bigram_idx])

        return corrected_bigram_tokenized_doc

    def dict_info(self) -> None:
        """
        Gather & display information about the dictionary/corpus.
        """

        print(
            f"Dictionary was created by processing {self.dct.num_pos:,} tokens from a corpus of {self.dct.num_docs:,} documents."
        )
        print(f"It contains {len(self.dct):,} unique tokens.")
        print(
            f"Each document, on average, contains {self.dct.num_nnz // self.dct.num_docs:,} unique tokens."
        )

    @timer(1000)
    def add_label_tags(self, key2label: Dict[str, str]):
        """
        Add label tags to the index file according to label dictionary.

        Parameters
        ----------
        key2label : Dict[str, str]
            Dictionary mapping tags to labels.
        """

        logging.info(
            f"[{self.__class__.__name__}.add_label_tags] Adding labels to index file where available...",
        )
        with IndexedFile(self.corpus_fpath, "reindex") as corpus_idx_file:
            # iterate over all existing keys (domain names)
            for key, (pos, _) in corpus_idx_file.key2poslabel.items():
                # "rewrite" the index line with a new label, if exists in key2label dictionary
                corpus_idx_file.write((pos, key, key2label.get(key, "unlabeled")))

    @timer(1000)
    def generate_train_test_sets(  # noqa: C901
        self,
        fpath: Path = None,
        test_frac: float = 0.2,
        seed: int = None,
        n_samples: int = None,
        upsampling: bool = False,
        **kwargs,
    ):
        """
        Generate training and testing sets from the corpus.

        Parameters
        ----------
        fpath : Path, optional
            File path for the corpus, by default None.
        test_frac : float, optional
            Fraction of data for the test set, by default 0.2.
        labeled : bool, optional
            Flag for labeled corpus, by default False.
        seed : int, optional
            Integer seed for deterministic train/test splitting.
        n_samples : int, optional
            Number of samples, by default None.
        tokens_only : bool, optional
            Toggle text-only samples, by default False.

        Returns
        -------
        Tuple[SampleGenerator, SampleGenerator]
            Training and testing sample generators.
        """

        # Instantiate a reproducible (if used with integer seed) random number generator for shuffling
        rng = np.random.default_rng(seed)

        fpath = fpath or self.corpus_fpath

        file_idx_path = get_file_index_path(fpath)

        n_samples = n_samples or self.dct.num_docs

        # Sort the index into a dictionary
        index_dict: Dict[str, List[int]] = {"good": [], "bad": [], "unlabeled": []}
        with open(file_idx_path, "r") as idx_file:
            while True:
                try:
                    start_pos, _, label = json.loads(idx_file.readline())
                    label = label or "unlabeled"  # treat None as unlabeled
                    index_dict[label].append(start_pos)
                except json.JSONDecodeError:
                    break

        # Shuffle the index - this means choosing different train/test sets for individual seeds
        for index_list in index_dict.values():
            rng.shuffle(index_list)

        # Count the number of "good," "bad," and "unlabeled" policies in training set (stratified)
        label_counts_dict = {label: len(list_) for label, list_ in index_dict.items()}

        # trim each list according to the requested number of samples and its own length
        if n_samples < self.dct.num_docs:
            sample_frac = n_samples / self.dct.num_docs
            for label in index_dict.keys():
                if not (upsampling and label == "good"):
                    n_reduced = round(label_counts_dict[label] * sample_frac)
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

            # Collect test set file index
            for _ in range(n_test_good):
                test_start_positions.append(index_dict["good"].pop())
            for _ in range(n_test_bad):
                test_start_positions.append(index_dict["bad"].pop())
            for _ in range(n_test_unlabeled):
                test_start_positions.append(index_dict["unlabeled"].pop())

        # oversample minority class (optional)
        if upsampling:
            upsampling_factor = int(label_counts_dict["bad"] / label_counts_dict["good"])
            index_dict["good"] *= upsampling_factor
            rng.shuffle(index_dict["good"])

            # trim good list AFTER upsampling (if relevant)
            if n_samples < self.dct.num_docs:
                n_reduced = int(len(index_dict["good"]) * sample_frac)
                index_dict["good"] = index_dict["good"][:n_reduced]

        # use the rest as training set file index
        train_start_positions = index_dict["good"] + index_dict["bad"] + index_dict["unlabeled"]

        # Shuffle again so that labeled samples are mixed in with the unlabeled samples
        rng.shuffle(train_start_positions)
        rng.shuffle(test_start_positions)

        # Initialize re-generators
        if not test_start_positions:
            return SampleGenerator(
                fpath,
                train_start_positions,
                **kwargs,
            )
        else:
            return SampleGenerator(
                fpath,
                train_start_positions,
                **kwargs,
            ), SampleGenerator(fpath, test_start_positions, **kwargs)

    def generate_samples(self, *args, **kwargs):
        """
        Generate samples from the corpus.
        """

        # Initialize and return re-generator
        return self.generate_train_test_sets(*args, test_frac=0.0, **kwargs)

    def _preprocess_document(
        self,
        fpath: Path = None,
        doc: str = None,
        url: str = None,
        separators=["", " ", "-"],
        **kwargs,
    ):
        """
        Preprocess a document from the corpus.

        Parameters
        ----------
        fpath : Path
            File path of the document.
        separators : List[str], optional
            List of separators for word combination, by default ["", " ", "-"].

        Returns
        -------
        List[str]
            List of preprocessed tokens.
        """

        if fpath:
            with open(fpath, "r", encoding="utf-8") as f:
                # Read all but the header
                _, *doc_lines = f.readlines()
                doc = "\n".join(doc_lines)[2:]

        elif not doc or not url:
            raise ValueError(
                "Must provide either a file path or a text together with a URL (for guessing the company name)."
            )

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

        # get domain name
        if url:
            domain_pattern = r"^(?:https?:\/\/)?(?:ftp?:\/\/)?(?:[^@\n]+@)?(?:www\.)?([^:\/\n?]+)"
            domain_name = re.match(domain_pattern, url).group(1)
        else:
            domain_name = fpath.stem

        # Look for possible appearances of company names using domain_name
        sliced_domain_name = wordninja.split(domain_name.split(".")[0])
        possible_company_words = concatenate_nearest_neighbors(sliced_domain_name, 6)
        single_word_domain = len(sliced_domain_name) == 1
        possbile_company_names = combine_with_separators(
            possible_company_words, separators, min_words=1 if single_word_domain else 2
        )
        lmc_phrase = longest_most_common_phrase(possbile_company_names, doc)
        if lmc_phrase:
            doc = re.sub(re.escape(lmc_phrase), "_companynametoken_", doc, flags=re.IGNORECASE)
            special_tokens["companynametoken"] = "<COMPANY>"

        # remove underscores (which are caused by Markdown)
        doc = re.sub("_", " ", doc)

        # Tokenize the text
        tokens = self._tokenize_text(doc, special_tokens, **kwargs)

        return tokens, domain_name

    def _tokenize_text(
        self,
        text: str,
        specials_dict: Dict[str, str],
        lemmatize: bool = True,
        filter_stopwords: bool = True,
        keep_list: List[str] = ["not"],  # , 'but', 'if', 'and', 'or']
        custom_stopwords: List[str] = ["ii", "llc", "inc"],
        **kwargs,
    ):
        """
        Tokenize text from the document.

        Parameters
        ----------
        text : str
            Text to be tokenized.
        specials_dict : Dict[str, str]
            Dictionary for special tokens.
        lemmatize : bool, optional
            Toggle lemmatization, by default True.
        filter_stopwords : bool, optional
            Toggle stopwords filtering, by default True.

        Returns
        -------
        List[str]
            List of tokenized words.
        """

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
            tokens = [lemmatizer.lemmatize(token_) for token_ in tokens]

        # TODO: Privacy terms / highlighting should be performed here!
        # (before stopwords are removed, which might ruint things such as 'opt out/in')

        # Remove stopwords (optional)
        if filter_stopwords:
            stop_words = set(stopwords.words("english")) - set(keep_list) | set(custom_stopwords)
            tokens = [token_ for token_ in tokens if token_ not in stop_words]

        # Remove consecutive duplicates
        tokens = self._remove_consecutive_duplicates(tokens)

        return tokens

    def _remove_consecutive_duplicates(self, tokens: List[str]):
        """
        Remove consecutive duplicates from a list of tokens.

        Parameters
        ----------
        tokens : List[str]
            List of tokens.

        Returns
        -------
        List[str]
            List of tokens without consecutive duplicates.
        """

        no_dups = [tokens[0]]
        for i in range(1, len(tokens)):
            if tokens[i] != tokens[i - 1]:
                no_dups.append(tokens[i])
        return no_dups

    def _convert_ordinal_to_words(self, number):
        """
        Convert ordinal numbers to their word equivalents.

        Parameters
        ----------
        number : int
            Ordinal number to convert.

        Returns
        -------
        str
            Ordinal number in word form.
        """

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

    @timer(1000, beep=True)
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
