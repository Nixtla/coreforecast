__all__ = ["GroupedArray", "get_start_all_threads", "set_start_all_threads"]

from ._lib.grouped_array import (
    GroupedArray,
    _set_start_all_threads,
    _start_all_threads,
)


def set_start_all_threads(start_all: bool) -> bool:
    """Choose whether grouped calls start every thread they're given.

    `num_threads` is the most threads a grouped call uses. By default the
    calling thread first works alone for a few microseconds and starts more
    threads only for the work it projects is left, so small calls don't pay
    for threads they can't use. Passing True makes every call start
    `min(num_threads, n_groups)` threads at once, which suits calls known to be
    large, such as a few long series. The setting is process-wide, so it is
    meant for applications rather than libraries, and it applies to calls that
    start after it returns.

    Args:
        start_all (bool): Whether every grouped call starts all of its threads.

    Returns:
        bool: The setting it replaces, to restore it with."""
    return _set_start_all_threads(start_all)


def get_start_all_threads() -> bool:
    """Whether grouped calls start every thread they're given.

    Returns:
        bool: The value last passed to `set_start_all_threads`, False by default."""
    return _start_all_threads()
