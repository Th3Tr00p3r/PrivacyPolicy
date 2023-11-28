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
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, List
from winsound import Beep

import numpy as np
import yaml  # type: ignore


@dataclass
class IndexedFile:
    """
    Class for handling indexed file operations.
    """

    fpath: Path
    mode: str
    rng: np.random.Generator = None
    start_pos_list: List[int] = field(default_factory=list)
    index_suffix: InitVar[str] = "_idx"
    shuffled: bool = False

    def __post_init__(self, index_suffix: str):
        """
        Initialize the IndexedFile after object creation.
        """

        self.idx_fpath = get_file_index_path(self.fpath, index_suffix)

        if self.mode == "write":
            # build index file-path
            self.temp_file = NamedTemporaryFile(mode="w", delete=False)

        elif self.mode == "read":
            # prepare key -> start position dictionary
            self.key_pos_dict = {}
            with open(self.idx_fpath, "r") as file:
                lines = file.readlines()
                for line in lines:
                    pos, key, *_ = json.loads(line)
                    self.key_pos_dict[key] = pos

    def __enter__(self):
        """
        Enter the context manager.
        """

        if self.mode == "read":
            self.file = open(self.fpath, "r")
            self.pos_idx = 0  # keep track of the file position index
            if self.shuffled:
                # shuffle each time entered
                self.rng.shuffle(self.start_pos_list)
        elif self.mode == "write":
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
                    json.dump([start_pos] + [note for note in notes], idx_file)
                    idx_file.write("\n")

    def write(self, obj, notes: Iterable[str]):
        """
        Write an object to the file.

        Parameters
        ----------
        obj : Any
            Object to be written to the file.
        notes : Iterable[str]
            Iterable containing notes for the object.
        """

        if self.mode != "write":
            raise TypeError(f"You are attempting to write while mode={self.mode}!")

        start_pos = self.file.tell()
        self.notes.append(notes)
        self.start_pos_list.append(start_pos)
        json.dump(obj, self.file)
        self.file.write("\n")

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

        self.file.seek(self.start_pos_list[pos_idx])
        return json.loads(self.file.readline())

    def read_all(self) -> Iterable[Any]:
        """
        Read all objects from the file.

        Returns
        -------
        Iterable[Any]
            Generator yielding all objects from the file.
        """

        if self.shuffled:
            self.rng.shuffle(self.start_pos_list)
        with open(self.fpath, "r") as file:
            for pos in self.start_pos_list:
                file.seek(pos)
                yield json.loads(file.readline())

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
        return self.start_pos_list.index(self.key_pos_dict[key])


def replace_most_common_phrase(phrases, text, special_token):
    """
    Replace the most common phrase found in the text with a special token.

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
        Text with the most common phrase replaced by the special token.
    """

    # Use a list comprehension to count occurrences of each phrase in the text
    counts_dict = {phrase: len(re.findall(phrase, text, flags=re.IGNORECASE)) for phrase in phrases}
    if any(counts_dict.values()):

        # Find the phrase with the maximum count
        phrase_to_replace, max_count = max(counts_dict.items(), key=lambda item: item[1])

        # Replace and return the most common phrase found in the text with the special token
        return re.sub(phrase_to_replace, special_token, text, flags=re.IGNORECASE)

    else:
        return text


def concatenate_nearest_neighbors(strings, n):
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

    while len(strings) > n:
        # Calculate distances (you might use a specific method based on your criteria)
        # For this example, let's randomly select neighbors to concatenate
        idx = random.randint(0, len(strings) - 2)
        concatenated = strings[idx] + strings[idx + 1]

        # Replace the nearest neighbors with the concatenated string
        strings = strings[:idx] + [concatenated] + strings[idx + 2 :]

    return strings


def combine_with_separators(words, separators):
    """
    Combine words with separators to generate all combinations.

    Parameters
    ----------
    words : list[str]
        List of words to combine.
    separators : list[str]
        List of separators for combination.

    Returns
    -------
    list[str]
        List of combined strings with separators.
    """

    # Generate all combinations of separators for the given number of words
    separator_combinations = product(separators, repeat=len(words) - 1)

    result = []
    for sep_comb in separator_combinations:
        combined = [f"{word}{sep}" for word, sep in zip(words, sep_comb)]
        result.append("".join(combined + [words[-1]]))

    return result


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

        return wrapper

    return outer_wrapper
