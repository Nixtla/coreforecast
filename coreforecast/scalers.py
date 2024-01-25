import ctypes

import numpy as np

from .grouped_array import _LIB, GroupedArray


__all__ = [
    "boxcox_lambda",
    "LocalMinMaxScaler",
    "LocalStandardScaler",
    "LocalRobustScaler",
]


_LIB.BoxCoxLambda_Guerrero.restype = ctypes.c_double


def boxcox_lambda(
    x: np.ndarray,
    season_length: int,
    lower: float = -1.0,
    upper: float = 2.0,
    method: str = "guerrero",
) -> float:
    """Find optimum lambda for the Box-Cox transformation

    Args:
        x (np.ndarray): Array with data to transform.
        season_length (int): Length of the seasonal period.
        lower (float): Lower bound for the lambda.
        upper (float): Upper bound for the lambda.
        method (str): Method to use. Valid options are 'guerrero'.

    Returns:
        float: Optimum lambda."""
    if method != "guerrero":
        raise NotImplementedError(f"Method {method} not implemented")
    if any(x <= 0):
        raise ValueError("All values in x must be positive")
    if lower >= upper:
        raise ValueError("lower must be less than upper")
    return _LIB.BoxCoxLambda_Guerrero(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(x.size),
        ctypes.c_int(season_length),
        ctypes.c_double(lower),
        ctypes.c_double(upper),
    )


class _BaseLocalScaler:
    _scaler_type: str

    def fit(self, ga: GroupedArray) -> "_BaseLocalScaler":
        """Compute the statistics for each group.

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            self: The fitted scaler object."""
        self.stats_ = ga._scaler_fit(self._scaler_type)
        return self

    def transform(self, ga: GroupedArray) -> np.ndarray:
        """Use the computed statistics to apply the transformation.

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the transformed data."""
        return ga._scaler_transform(self.stats_)

    def inverse_transform(self, ga: GroupedArray) -> np.ndarray:
        """Use the computed statistics to invert the transformation.

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the inverted transformation."""
        return ga._scaler_inverse_transform(self.stats_)


class LocalMinMaxScaler(_BaseLocalScaler):
    """Scale each group to the [0, 1] interval"""

    _scaler_type = "MinMax"


class LocalStandardScaler(_BaseLocalScaler):
    """Scale each group to have zero mean and unit variance"""

    _scaler_type = "Standard"


class LocalRobustScaler(_BaseLocalScaler):
    """Scale each group using robust statistics

    Args:
        scale (str): Type of robust scaling to use. Valid options are 'iqr' and 'mad'.
            If 'iqr' will use the inter quartile range as the scale.
            If 'mad' will use median absolute deviation as the scale."""

    def __init__(self, scale: str):
        if scale == "iqr":
            self._scaler_type = "RobustIqr"
        else:
            self._scaler_type = "RobustMad"
