import numpy as np

from ._lib import diff as _diff
from .differences import num_seas_diffs
from .utils import _ensure_float


def find_season_length(x: np.ndarray, max_season_length: int) -> int:
    """Find the length of the seasonal period of the time series.
    Returns 0 if no seasonality is found.

    Args:
        x (np.ndarray): Array with the time series.

    Returns:
        int: Season period."""
    x = _ensure_float(x)
    period = _diff.period(x, max_season_length)
    return num_seas_diffs(x, period, 1) * period
