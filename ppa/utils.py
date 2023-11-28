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
from typing import Callable, List
from winsound import Beep

import numpy as np
import yaml  # type: ignore


@dataclass
class IndexedFile:

    fpath: Path
    mode: str
    rng: np.random.Generator = None
    start_pos_list: List[int] = field(default_factory=list)
    index_suffix: InitVar[str] = "_idx"
    shuffled: bool = False

    def __post_init__(self, index_suffix: str):
        """Doc."""

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
        """Doc."""

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
        """Doc."""

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
        """Doc."""

        if self.mode != "write":
            raise TypeError(f"You are attempting to write while mode={self.mode}!")

        start_pos = self.file.tell()
        self.notes.append(notes)
        self.start_pos_list.append(start_pos)
        json.dump(obj, self.file)
        self.file.write("\n")

    def read_idx(self, pos_idx: int):
        """Doc."""

        if self.mode != "read":
            raise TypeError(f"You are attempting to read while mode={self.mode}!")

        self.file.seek(self.start_pos_list[pos_idx])
        return json.loads(self.file.readline())

    def read_all(self):
        """Doc."""

        if self.shuffled:
            self.rng.shuffle(self.start_pos_list)
        with open(self.fpath, "r") as file:
            for pos in self.start_pos_list:
                file.seek(pos)
                yield json.loads(file.readline())

    def key_to_pos_idx(self, key: str) -> int:
        """Doc."""

        return self.start_pos_list.index(self.key_pos_dict[key])


def replace_most_common_phrase(phrases, text, special_token):
    # Use a list comprehension to count occurrences of each phrase in the text
    counts_dict = {phrase: len(re.findall(phrase, text, flags=re.IGNORECASE)) for phrase in phrases}
    if any(counts_dict.values()):

        # Find the phrase with the maximum count
        phrase_to_replace, max_count = max(counts_dict.items(), key=lambda item: item[1])

        # Replace and return the most common phrase found in the text with the special token
        return re.sub(phrase_to_replace, special_token, text, flags=re.IGNORECASE)

    else:
        return text


# Function to find nearest neighbors and concatenate randomly
def concatenate_nearest_neighbors(strings, n):
    while len(strings) > n:
        # Calculate distances (you might use a specific method based on your criteria)
        # For this example, let's randomly select neighbors to concatenate
        idx = random.randint(0, len(strings) - 2)
        concatenated = strings[idx] + strings[idx + 1]

        # Replace the nearest neighbors with the concatenated string
        strings = strings[:idx] + [concatenated] + strings[idx + 2 :]

    return strings


def combine_with_separators(words, separators):
    # Generate all combinations of separators for the given number of words
    separator_combinations = product(separators, repeat=len(words) - 1)

    result = []
    for sep_comb in separator_combinations:
        combined = [f"{word}{sep}" for word, sep in zip(words, sep_comb)]
        result.append("".join(combined + [words[-1]]))

    return result


def get_file_index_path(fpath: Path, index_suffix: str = "_idx"):
    """Doc."""

    return Path(fpath.parent) / f"{fpath.stem}{index_suffix}{''.join(fpath.suffixes)}"


def config_logging(log_path: Path = Path.cwd().parent.parent / "logs"):
    """
    Configure the logging package for the whole application and ensure folder and initial files exist.
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
    Meant to be used as a decorator (@helper.timer(threshold))
    for quickly setting up function timing for testing.
    Works for both regular and asynchronous functions.
    NOTE - asynchronous function timing may include stuff that happens
        while function 'awaits' other coroutines.
    """

    def outer_wrapper(func) -> Callable:
        """Doc."""

        if asyncio.iscoroutinefunction(func):
            # timing async funcitons
            @functools.wraps(func)
            async def wrapper(*args, should_time: bool = True, **kwargs):
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
