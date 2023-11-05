import asyncio
import functools
import gzip
import logging
import pickle
import random
import time
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Callable, List
from winsound import Beep

import yaml  # type: ignore


@dataclass
class IndexedFile:

    fpath: Path
    mode: str
    start_pos_list: List[int] = field(default_factory=lambda: [0])
    index_suffix: InitVar[str] = "_idx"
    should_shufle: bool = True

    def __post_init__(self, index_suffix: str):
        """Doc."""

        if self.mode == "read" and not self.start_pos_list:
            raise ValueError("Must supply 'start_pos_list' in 'read' mode!")

        if self.mode == "write":
            # build index file-path
            self.idx_fpath = get_file_index_path(self.fpath, index_suffix)

    def __enter__(self):
        if self.mode == "read":
            self.file = gzip.open(self.fpath, "rb")
            self.pos_idx = 0  # keep track of the file position index
            if self.should_shufle:
                # shuffle each time entered
                random.shuffle(self.start_pos_list)
        elif self.mode == "write":
            self.file = gzip.open(self.fpath, "wb")
            self.notes = []

        return self

    def __exit__(self, *args):
        """Doc."""

        # close the data file
        self.file.close()
        if self.mode == "write":
            # create the index file from the collected lists (start positions and notes)
            with gzip.open(self.idx_fpath, "wb") as idx_file:
                for start_pos, note in zip(self.start_pos_list, self.notes):
                    pickle.dump((start_pos, note), idx_file, protocol=pickle.HIGHEST_PROTOCOL)

    def write(self, obj, note: str = "unlabeled"):
        """Doc."""

        if self.mode != "write":
            raise TypeError(f"You are attempting to write while mode={self.mode}!")

        start_pos = self.file.tell()
        self.notes.append(note)
        self.start_pos_list.append(start_pos)
        pickle.dump(obj, self.file, protocol=pickle.HIGHEST_PROTOCOL)

    def read(self):
        """Doc."""

        if self.mode != "read":
            raise TypeError(f"You are attempting to read while mode={self.mode}!")

        self.file.seek(self.start_pos_list[self.pos_idx])
        self.pos_idx += 1
        return pickle.load(self.file)

    def read_idx(self, pos_idx: int):
        """Doc."""

        if self.mode != "read":
            raise TypeError(f"You are attempting to read while mode={self.mode}!")

        self.file.seek(self.start_pos_list[pos_idx])
        return pickle.load(self.file)

    def read_all(self):
        """Doc."""

        if self.should_shufle:
            random.shuffle(self.start_pos_list)

        with gzip.open(self.fpath, "rb") as file:
            for pos in self.start_pos_list:
                file.seek(pos)
                yield pickle.load(file)


def deep_stem_path(p: Path):
    """Doc."""

    p = Path(p.stem)
    while True:
        stemmed_p = Path(p.stem)
        if stemmed_p != p:
            p = stemmed_p
        else:
            return str(p)


def get_file_index_path(fpath: Path, index_suffix: str = "_idx"):
    """Doc."""

    return Path(fpath.parent) / f"{deep_stem_path(fpath)}{index_suffix}{''.join(fpath.suffixes)}"


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
                        print(
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
                        print(
                            f"***TIMER*** Function '{func.__name__}()' took {elapsed_time_ms * (1e-3 if in_s else 1):.2f} {'s' if in_s else 'ms'}.\n"
                        )
                        if beep:
                            Beep(1000, 500)  # Beep at 1000 Hz for 500 ms
                    return value

        return wrapper

    return outer_wrapper
