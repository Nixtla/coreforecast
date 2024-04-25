__all__ = [
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
]

import ctypes
from typing import Callable, Optional

import numpy as np

from ._lib import _LIB
from .utils import (
    _data_as_void_ptr,
    _ensure_float,
    _float_arr_to_prefix,
)


def _rolling_stat(
    x: np.ndarray,
    stat: str,
    window_size: int,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    out = np.empty_like(x)
    if min_samples is None:
        min_samples = window_size
    _LIB[f"{prefix}_Rolling{stat}Transform"](
        _data_as_void_ptr(x),
        ctypes.c_int(x.size),
        ctypes.c_int(window_size),
        ctypes.c_int(min_samples),
        _data_as_void_ptr(out),
    )
    return out


def _rolling_docstring(*args, **kwargs) -> Callable:
    base_docstring = """Compute the {} of the input array.

    Args:
        x (np.ndarray): Input array.
        window_size (int): The size of the rolling window.
        min_samples (int, optional): The minimum number of samples required to compute the statistic.
            If None, it is set to `window_size`.

    Returns:
        np.ndarray: Array with rolling statistic
    """

    def docstring_decorator(function: Callable):
        function.__doc__ = base_docstring.format(function.__name__)
        return function

    return docstring_decorator(*args, **kwargs)


@_rolling_docstring
def rolling_mean(
    x: np.ndarray, window_size: int, min_samples: Optional[int] = None
) -> np.ndarray:
    return _rolling_stat(x, "Mean", window_size, min_samples)


@_rolling_docstring
def rolling_std(
    x: np.ndarray, window_size: int, min_samples: Optional[int] = None
) -> np.ndarray:
    return _rolling_stat(x, "Std", window_size, min_samples)


@_rolling_docstring
def rolling_min(
    x: np.ndarray, window_size: int, min_samples: Optional[int] = None
) -> np.ndarray:
    return _rolling_stat(x, "Min", window_size, min_samples)


@_rolling_docstring
def rolling_max(
    x: np.ndarray, window_size: int, min_samples: Optional[int] = None
) -> np.ndarray:
    return _rolling_stat(x, "Max", window_size, min_samples)
