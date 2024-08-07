__all__ = ["exponentially_weighted_mean"]

import numpy as np

from ._lib import exponentially_weighted as _ew
from .utils import _ensure_float


def exponentially_weighted_mean(x: np.ndarray, alpha: float) -> np.ndarray:
    """Compute the exponentially weighted mean of the input array.

    Args:
        x (np.ndarray): Input array.
        alpha (float): Weight parameter.

    Returns:
        np.ndarray: Array with the exponentially weighted mean.
    """
    x = _ensure_float(x)
    out = np.empty_like(x)
    _ew.exponentially_weighted_mean(x, alpha, out)
    return out
