import asyncio
import functools
import logging
import time
from pathlib import Path
from typing import Callable
from winsound import Beep

import yaml  # type: ignore


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
