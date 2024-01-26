import ctypes
from typing import Union

import numpy as np

from .grouped_array import _LIB, _data_as_void_ptr, _ensure_float, GroupedArray


__all__ = [
    "boxcox_lambda",
    "LocalBoxCoxScaler",
    "LocalMinMaxScaler",
    "LocalRobustScaler",
    "LocalStandardScaler",
]


_LIB.Float32_BoxCoxLambdaGuerrero.restype = ctypes.c_float
_LIB.Float64_BoxCoxLambdaGuerrero.restype = ctypes.c_double


def _pyfloat_to_np_c(x: float, t: np.dtype) -> Union[ctypes.c_float, ctypes.c_double]:
    if t == np.float32:
        return ctypes.c_float(x)
    return ctypes.c_double(x)


def boxcox_lambda(
    x: np.ndarray,
    season_length: int,
    lower: float = -0.9,
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
    x = _ensure_float(x)
    if x.dtype == np.float32:
        fn = "Float32_BoxCoxLambdaGuerrero"
    else:
        fn = "Float64_BoxCoxLambdaGuerrero"
    # _LIB[fn] doesn't get the restype assignment (keeps c_int)
    return getattr(_LIB, fn)(
        _data_as_void_ptr(x),
        ctypes.c_int(x.size),
        ctypes.c_int(season_length),
        _pyfloat_to_np_c(lower, x.dtype),
        _pyfloat_to_np_c(upper, x.dtype),
    )


def boxcox(x: np.ndarray, lmbda: float) -> np.ndarray:
    x = _ensure_float(x)
    if x.dtype == np.float32:
        fn = "Float32_BoxCoxTransform"
    else:
        fn = "Float64_BoxCoxTransform"
    out = np.empty_like(x)
    getattr(_LIB, fn)(
        _data_as_void_ptr(x),
        ctypes.c_int(x.size),
        _pyfloat_to_np_c(lmbda, x.dtype),
        _data_as_void_ptr(out),
    )
    return out


def inv_boxcox(x: np.ndarray, lmbda: float) -> np.ndarray:
    x = _ensure_float(x)
    if x.dtype == np.float32:
        fn = "Float32_BoxCoxInverseTransform"
    else:
        fn = "Float64_BoxCoxInverseTransform"
    out = np.empty_like(x)
    getattr(_LIB, fn)(
        _data_as_void_ptr(x),
        ctypes.c_int(x.size),
        _pyfloat_to_np_c(lmbda, x.dtype),
        _data_as_void_ptr(out),
    )
    return out


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


class LocalBoxCoxScaler(_BaseLocalScaler):
    def __init__(
        self,
        season_length: int,
        lower: float = -0.9,
        upper: float = 2.0,
        method="guerrero",
    ):
        self.season_length = season_length
        self.lower = lower
        self.upper = upper
        if method != "guerrero":
            raise NotImplementedError(f"Method {method} not implemented")
        self.method = method.capitalize()

    def fit(self, ga: GroupedArray) -> "_BaseLocalScaler":
        self.stats = np.empty_like(ga.data, shape=(len(ga), 1))
        _LIB[f"{ga.prefix}_BoxCoxLambda{self.method}"](
            ga._handle,
            ctypes.c_int(self.season_length),
            _pyfloat_to_np_c(self.lower, ga.data.dtype),
            _pyfloat_to_np_c(self.upper, ga.data.dtype),
            _data_as_void_ptr(self.stats),
        )
        # this is to use ones as scale. I know.
        self.stats = np.hstack([self.stats, np.ones_like(self.stats)])
        return self

    def transform(self, ga: GroupedArray) -> np.ndarray:
        out = np.empty_like(ga.data)
        _LIB[f"{ga.prefix}_BoxCoxTransform"](
            ga._handle,
            _data_as_void_ptr(self.stats),
            _data_as_void_ptr(out),
        )
        return out

    def inverse_transform(self, ga: GroupedArray) -> np.ndarray:
        out = np.empty_like(ga.data)
        _LIB[f"{ga.prefix}_BoxCoxInverseTransform"](
            ga._handle,
            _data_as_void_ptr(self.stats),
            _data_as_void_ptr(out),
        )
        return out
