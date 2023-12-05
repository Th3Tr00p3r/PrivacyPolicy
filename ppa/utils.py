import asyncio
import functools
import json
import logging
import random
import re
import shutil
import time
from collections.abc import Iterable
from dataclasses import InitVar, dataclass, field
from itertools import combinations, permutations
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, List
from winsound import Beep

import yaml  # type: ignore


@dataclass
class IndexedFile:
    """
    Class for handling indexed file operations.
    """

    fpath: Path
    mode: str
    start_pos_list: List[int] = field(default_factory=list)
    index_suffix: InitVar[str] = "_idx"

    def __post_init__(self, index_suffix: str):
        """
        Initialize the IndexedFile after object creation.
        """

        self.idx_fpath = get_file_index_path(self.fpath, index_suffix)

        if self.mode in {"write", "reindex"}:
            # build index file-path
            self.temp_file = NamedTemporaryFile(mode="w", delete=False)

        if self.mode in {"read", "reindex"}:
            # prepare key -> start position dictionary
            self.key2poslabel = {}
            with open(self.idx_fpath, "r") as file:
                lines = file.readlines()
                for line in lines:
                    pos, key, label = json.loads(line)
                    #                    print("pos, key, label: ", pos, key, label) # TESTESTEST
                    self.key2poslabel[key] = (pos, label)
            # keep the reverse dict, too
            self.pos2keylabel = {
                pos: (key, label) for key, (pos, label) in self.key2poslabel.items()
            }

    def __enter__(self):
        """
        Enter the context manager.
        """

        if self.mode == "read":
            self.file = open(self.fpath, "r")
            self.pos_idx = 0  # keep track of the file position index

        elif self.mode in {"write", "reindex"}:
            self.file = self.temp_file
            self.notes = []

        return self

    def __exit__(self, *args):
        """
        Exit the context manager.
        """

        # close the data file
        self.file.close()

        if self.mode == "write":
            # Move the temporary file to the supplied fpath
            shutil.move(self.temp_file.name, self.fpath)
            # create the index file from the collected lists (start positions and notes)
            with open(self.idx_fpath, "w") as idx_file:
                for start_pos, notes in zip(self.start_pos_list, self.notes):
                    json.dump((start_pos, *notes), idx_file)
                    idx_file.write("\n")

        elif self.mode == "reindex":
            # Move the temporary file to the supplied fpath
            shutil.move(self.temp_file.name, self.idx_fpath)

    def write(self, obj, notes: List[str] = None):
        """
        Write an object to the file.

        Parameters
        ----------
        obj : Any
            Object to be written to the file.
        notes : Iterable[str]
            Iterable containing notes for the object.
        """

        if notes and len(notes) < 2:
            notes.append("unlabeled")

        if self.mode == "write":
            start_pos = self.file.tell()
            self.notes.append(notes)
            self.start_pos_list.append(start_pos)
            json.dump(obj, self.file)
            self.file.write("\n")

        elif self.mode == "reindex":
            json.dump(obj, self.file)
            self.file.write("\n")

        else:
            raise TypeError(f"You are attempting to write while mode={self.mode}!")

    def read_idx(self, pos_idx: int) -> Any:
        """
        Read an object by index position.

        Parameters
        ----------
        pos_idx : int
            Index position of the object to be read.

        Returns
        -------
        Any
            Object read from the file.
        """

        if self.mode != "read":
            raise TypeError(f"You are attempting to read while mode={self.mode}!")

        pos = self.start_pos_list[pos_idx]
        self.file.seek(pos)
        tokens = json.loads(self.file.readline())
        key, label = self.pos2keylabel[pos]
        return tokens, key, label

    def read_all(self) -> Iterable[Any]:
        """
        Read all objects from the file.

        Returns
        -------
        Iterable[Any]
            Generator yielding all objects from the file.
        """

        if self.mode != "read":
            raise TypeError(f"You are attempting to read while mode={self.mode}!")

        with open(self.fpath, "r") as file:
            for pos in self.start_pos_list:
                file.seek(pos)
                tokens = json.loads(file.readline())
                key, label = self.pos2keylabel[pos]
                yield tokens, key, label

    def key_to_pos_idx(self, key: str) -> int:
        """
        Get the position index of a key in the file.

        Parameters
        ----------
        key : str
            Key to search for in the file.

        Returns
        -------
        int
            Position index of the key in the file.
        """
        return self.start_pos_list.index(self.key2poslabel[key][0])


def longest_most_common_phrase(phrases, text):
    """
    Return the most common phrase found in the text with a special token.
    If more than one phrase has same number of appearances in text, the longest is returned.

    Parameters
    ----------
    phrases : list[str]
        List of phrases to search for in the text.
    text : str
        The text in which to search for the phrases.
    special_token : str
        The special token to replace the most common phrase found.

    Returns
    -------
    str
        Longest of the most common phrases.
    """

    # Use a list comprehension to count occurrences of each phrase in the text
    counts_dict = {phrase: len(re.findall(phrase, text, flags=re.IGNORECASE)) for phrase in phrases}

    if any(counts_dict.values()):

        # Find the maximum count
        max_count = max(counts_dict.values())

        # Filter phrases that have the maximum count
        phrases_with_max_count = [
            phrase for phrase, count in counts_dict.items() if count == max_count
        ]

        # Return the longest phrase among those with the same maximum count
        return max(phrases_with_max_count, key=len)


def concatenate_nearest_neighbors(strings, n_max: int) -> List[str]:
    """
    Concatenate nearest neighbors from a list of strings.

    Parameters
    ----------
    strings : list[str]
        List of strings to perform concatenation.
    n : int
        Number of strings to retain after concatenation.

    Returns
    -------
    list[str]
        List of strings after concatenation of nearest neighbors.
    """

    while len(strings) > n_max:
        # Calculate distances (you might use a specific method based on your criteria)
        # For this example, let's randomly select neighbors to concatenate
        idx = random.randint(0, len(strings) - 2)
        concatenated = strings[idx] + strings[idx + 1]

        # Replace the nearest neighbors with the concatenated string
        strings = strings[:idx] + [concatenated] + strings[idx + 2 :]

    return strings


def combine_with_separators(words, separators, min_length=4, min_words: int = 1):
    """
    Combine words with separators, allowing for omitted words while ensuring at least one word appears.

    Parameters
    ----------
    words : list[str]
        List of words to combine.
    separators : list[str]
        List of separators for combination.

    Returns
    -------
    list[str]
        List of combined strings with separators and omitted words.
    """

    all_combinations = set()  # Using a set to ensure uniqueness

    # Generate all permutations of word indices to allow for omitted words
    word_idxs = range(
        min_words, len(words) + 1
    )  # Start from 'min_words' to ensure at least 'min_words' words
    word_combinations = [
        comb for word_idx in word_idxs for comb in combinations(range(len(words)), word_idx)
    ]

    for word_comb in word_combinations:
        for perm in permutations(word_comb):
            selected_words = [words[idx] for idx in perm]

            for sep in separators:
                # Combine selected words with the current separator
                combined = sep.join(selected_words)
                # enforce minimum length
                if len(combined) >= min_length:
                    all_combinations.add(combined)

    return list(all_combinations)


def get_file_index_path(fpath: Path, index_suffix: str = "_idx"):
    """
    Get the file index path based on the given file path and suffix.

    Parameters
    ----------
    fpath : Path
        Path to the file.
    index_suffix : str, optional
        Index suffix for the file, by default "_idx".

    Returns
    -------
    Path
        The file index path.
    """

    return Path(fpath.parent) / f"{fpath.stem}{index_suffix}{''.join(fpath.suffixes)}"


def config_logging(log_path: Path = Path.cwd().parent.parent / "logs"):
    """
    Configure the logging package and ensure folder and initial files exist.

    Parameters
    ----------
    log_path : Path, optional
        Path to the log files directory, by default parent directory + "/logs".
    """

    Path.mkdir(log_path, parents=True, exist_ok=True)
    init_log_file_list = ["log.txt"]
    for init_log_file in init_log_file_list:
        log_file_path = log_path / init_log_file
        open(log_file_path, "a").close()

    with open(log_path.parent / "logging_config.yaml", "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)


def timer(threshold_ms: float = 0.0, beep=True) -> Callable:
    """
    Decorator for function timing used for testing.

    Parameters
    ----------
    threshold_ms : float, optional
        Threshold for time in milliseconds, by default 0.0
    beep : bool, optional
        Boolean indicating whether to beep on threshold exceedance, by default True

    Returns
    -------
    Callable
        Wrapper function for timing the decorated function.
    """

    def outer_wrapper(func) -> Callable:
        """
        Outer wrapper function to handle timing for both regular and asynchronous functions.
        """

        if asyncio.iscoroutinefunction(func):
            # timing async funcitons
            @functools.wraps(func)
            async def wrapper(*args, should_time: bool = True, **kwargs):
                """
                Wrapper function for timing asynchronous functions.

                Parameters
                ----------
                args : tuple
                    Positional arguments for the function.
                should_time : bool, optional
                    Boolean indicating whether to calculate and log time, by default True
                kwargs : dict
                    Keyword arguments for the function.

                Returns
                -------
                Any
                    Value returned by the decorated asynchronous function.
                """

                if should_time:
                    tic = time.perf_counter()
                    value = await func(*args, **kwargs)
                    toc = time.perf_counter()
                    elapsed_time_ms = (toc - tic) * 1e3
                    if elapsed_time_ms > threshold_ms:
                        in_s = elapsed_time_ms > 1000
                        logging.info(
                            f"***TIMER*** Function '{func.__name__}()' took {elapsed_time_ms * (1e-3 if in_s else 1):.2f} {'s' if in_s else 'ms'}.\n"
                        )
                        if beep:
                            Beep(1000, 500)  # Beep at 1000 Hz for 500 ms
                    return value
                else:
                    return func(*args, **kwargs)

        else:

            @functools.wraps(func)
            def wrapper(*args, should_time: bool = True, **kwargs):
                """
                Wrapper function for timing regular functions.

                Parameters
                ----------
                args : tuple
                    Positional arguments for the function.
                should_time : bool, optional
                    Boolean indicating whether to calculate and log time, by default True
                kwargs : dict
                    Keyword arguments for the function.

                Returns
                -------
                Any
                    Value returned by the decorated function.
                """

                if should_time:
                    tic = time.perf_counter()
                    value = func(*args, **kwargs)
                    toc = time.perf_counter()
                    elapsed_time_ms = (toc - tic) * 1e3
                    if elapsed_time_ms > threshold_ms:
                        in_s = elapsed_time_ms > 1000
                        logging.info(
                            f"***TIMER*** Function '{func.__name__}()' took {elapsed_time_ms * (1e-3 if in_s else 1):.2f} {'s' if in_s else 'ms'}."
                        )
                        if beep:
                            Beep(1000, 500)  # Beep at 1000 Hz for 500 ms
                    return value
                else:
                    return func(*args, **kwargs)

        return wrapper

    return outer_wrapper
