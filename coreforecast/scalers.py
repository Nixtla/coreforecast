import ctypes
from typing import List, Optional

import numpy as np

from ._lib import _LIB, _indptr_dtype
from .grouped_array import GroupedArray
from .utils import (
    _data_as_void_ptr,
    _diffs_to_indptr,
    _ensure_float,
    _float_arr_to_prefix,
    _pyfloat_to_np_c,
)


__all__ = [
    "AutoDifferences",
    "AutoSeasonalDifferences",
    "AutoSeasonalityAndDifferences",
    "LocalBoxCoxScaler",
    "LocalMinMaxScaler",
    "LocalRobustScaler",
    "LocalStandardScaler",
    "boxcox",
    "boxcox_lambda",
    "inv_boxcox",
]


_LIB.Float32_BoxCoxLambdaGuerrero.restype = ctypes.c_float
_LIB.Float64_BoxCoxLambdaGuerrero.restype = ctypes.c_double


def boxcox_lambda(
    x: np.ndarray,
    season_length: int,
    lower: float = -0.9,
    upper: float = 2.0,
    method: str = "guerrero",
) -> float:
    """Find optimum lambda for the Box-Cox transformation (supports negative numbers)

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
    if lower >= upper:
        raise ValueError("lower must be less than upper")
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    fn = f"{prefix}_BoxCoxLambdaGuerrero"
    # _LIB[fn] doesn't get the restype assignment (keeps c_int)
    return getattr(_LIB, fn)(
        _data_as_void_ptr(x),
        ctypes.c_int(x.size),
        ctypes.c_int(season_length),
        _pyfloat_to_np_c(lower, x.dtype),
        _pyfloat_to_np_c(upper, x.dtype),
    )


def boxcox(x: np.ndarray, lmbda: float) -> np.ndarray:
    """Apply the Box-Cox transformation

    Args:
        x (np.ndarray): Array with data to transform.
        lmbda (float): Lambda value to use.

    Returns:
        np.ndarray: Array with the transformed data."""
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    fn = f"{prefix}_BoxCoxTransform"
    out = np.empty_like(x)
    getattr(_LIB, fn)(
        _data_as_void_ptr(x),
        ctypes.c_int(x.size),
        _pyfloat_to_np_c(lmbda, x.dtype),
        _data_as_void_ptr(out),
    )
    return out


def inv_boxcox(x: np.ndarray, lmbda: float) -> np.ndarray:
    """Invert the Box-Cox transformation

    Args:
        x (np.ndarray): Array with data to transform.
        lmbda (float): Lambda value to use.

    Returns:
        np.ndarray: Array with the inverted transformation."""
    x = _ensure_float(x)
    prefix = _float_arr_to_prefix(x)
    fn = f"{prefix}_BoxCoxInverseTransform"
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

    def fit_transform(self, ga: GroupedArray) -> np.ndarray:
        """ "Compute the statistics for each group and apply the transformation.

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the transformed data."""
        return self.fit(ga).transform(ga)

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
    """Find the optimum lambda for the Box-Cox transformation by group and apply it

    Args:
        season_length (int): Length of the seasonal period.
        lower (float): Lower bound for the lambda.
        upper (float): Upper bound for the lambda.
        method (str): Method to use. Valid options are 'guerrero'."""

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
        """Compute the statistics for each group.

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            self: The fitted scaler object."""
        self.stats = ga._boxcox_fit(
            self.season_length, self.lower, self.upper, self.method
        )
        return self

    def transform(self, ga: GroupedArray) -> np.ndarray:
        """Use the computed lambdas to apply the transformation.

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the transformed data."""
        return ga._boxcox_transform(self.stats)

    def inverse_transform(self, ga: GroupedArray) -> np.ndarray:
        """Use the computed lambdas to invert the transformation.

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the inverted transformation."""
        return ga._boxcox_inverse_transform(self.stats)


class AutoDifferences:
    """Find and apply the optimal number of differences to each group.

    Args:
        max_diffs (int): Maximum number of differences to apply."""

    def __init__(self, max_diffs: int):
        if not isinstance(max_diffs, int) or max_diffs <= 0:
            raise ValueError("max_diff must be a positive integer")
        self.max_diffs = max_diffs

    def _transform(self, ga: GroupedArray, season_length: int) -> np.ndarray:
        max_d = int(self.diffs_.max())
        transformed = ga.data.copy()
        self.tails_ = []
        for i in range(max_d):
            ga = ga._with_data(transformed)
            mask = self.diffs_ > i
            ds = np.where(mask, _indptr_dtype(season_length), _indptr_dtype(0))
            tails_indptr = _diffs_to_indptr(ds)
            self.tails_.append(ga._tails(tails_indptr))
            transformed = ga._diffs(ds)
        return transformed

    def fit_transform(self, ga: GroupedArray) -> np.ndarray:
        """Compute and apply the optimal number of differences for each group

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the transformed data."""
        self.diffs_ = ga._num_diffs(self.max_diffs)
        return self._transform(ga, season_length=1)

    def _inverse_transform(self, ga: GroupedArray, season_length: int) -> np.ndarray:
        transformed = ga.data.copy()
        n_diffs = len(self.tails_)
        for i, tails in enumerate(self.tails_[::-1]):
            ga = ga._with_data(transformed)
            mask = self.diffs_ >= n_diffs - i
            ds = np.where(mask, _indptr_dtype(season_length), _indptr_dtype(0))
            transformed = ga._inv_diffs(ds, tails)
        return transformed

    def inverse_transform(self, ga: GroupedArray) -> np.ndarray:
        """Invert the differences

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the inverted transformation."""
        return self._inverse_transform(ga, 1)

    def _update(self, ga: GroupedArray, season_length: int) -> np.ndarray:
        tail_indptr = np.arange(
            0, season_length * ga.indptr.size, season_length, dtype=_indptr_dtype
        )
        new_tails = []
        dtype = ga.data.dtype.type
        transformed = ga.data.copy()
        for i, tail in enumerate(self.tails_):
            ga = ga._with_data(transformed)
            tail_ga = GroupedArray(tail, tail_indptr, num_threads=ga.num_threads)
            combined = tail_ga._append(ga)
            new_tails.append(combined._tail(season_length))
            ds = np.where(
                self.diffs_ > i, _indptr_dtype(season_length), _indptr_dtype(0)
            )
            combined_transformed = combined._diffs(ds)
            transformed = combined._with_data(combined_transformed)._tails(ga.indptr)
        self.tails_ = new_tails
        return transformed

    def update(self, ga: GroupedArray) -> np.ndarray:
        """Update the last observations from each serie

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the updated data."""
        return self._update(ga, season_length=1)


class AutoSeasonalDifferences(AutoDifferences):
    """Find and apply the optimal number of seasonal differences to each group.

    Args:
        season_length (int): Length of the seasonal period.
        max_diffs (int): Maximum number of differences to apply.
        n_seasons (int | None): Number of seasons to use to determine the number of differences. Defaults to 10.
            If `None` will use all samples, otherwise `season_length` * `n_seasons` samples will be used for the test.
            Smaller values will be faster but could be less accurate.
    """

    def __init__(
        self,
        season_length: int,
        max_diffs: int,
        n_seasons: Optional[int] = 10,
    ):
        self.season_length = season_length
        self.max_diffs = max_diffs
        if isinstance(n_seasons, int) and n_seasons < 2:
            raise ValueError("n_seasons must be at least 2")
        self.n_seasons = n_seasons

    def fit_transform(self, ga: GroupedArray) -> np.ndarray:
        """Compute and apply the optimal number of seasonal differences for each group

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the transformed data."""
        if self.n_seasons is None:
            tails_ga = ga
        else:
            n_samples = self.season_length * self.n_seasons
            tails_ga = GroupedArray(
                ga._tail(n_samples),
                np.arange(0, (len(ga) + 1) * n_samples, n_samples),
                num_threads=ga.num_threads,
            )
        self.diffs_ = tails_ga._num_seas_diffs(self.season_length, self.max_diffs)
        return self._transform(ga, self.season_length)

    def inverse_transform(self, ga: GroupedArray) -> np.ndarray:
        """Invert the seasonal differences

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the inverted transformation."""
        return self._inverse_transform(ga, self.season_length)

    def update(self, ga: GroupedArray) -> np.ndarray:
        """Update the last observations from each serie

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the updated data."""
        return self._update(ga, self.season_length)


class AutoSeasonalityAndDifferences:
    """Find the length of the seasonal period and apply the optimal number of differences to each group.

    Args:
        max_season_length (int): Maximum length of the seasonal period.
        max_diffs (int): Maximum number of differences to apply.
        n_seasons (int | None): Number of seasons to use to determine the number of differences. Defaults to 10.
            If `None` will use all samples, otherwise `max_season_length` * `n_seasons` samples will be used for the test.
            Smaller values will be faster but could be less accurate.
    """

    def __init__(
        self, max_season_length: int, max_diffs: int, n_seasons: Optional[int] = 10
    ):
        self.max_season_length = max_season_length
        self.max_diffs = max_diffs
        if isinstance(n_seasons, int) and n_seasons < 2:
            raise ValueError("n_seasons must be at least 2")
        self.n_seasons = n_seasons

    def fit_transform(self, ga: GroupedArray) -> np.ndarray:
        """Compute the optimal length of the seasonal period and apply the optimal number of differences for each group

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the transformed data."""
        self.diffs_ = []
        self.tails_ = []
        transformed = ga.data.copy()
        for i in range(self.max_diffs):
            ga = ga._with_data(transformed)
            if self.n_seasons is None:
                tails_ga = ga
            else:
                n_samples = self.max_season_length * self.n_seasons
                tails_ga = GroupedArray(
                    ga._tail(n_samples),
                    np.arange(0, (len(ga) + 1) * n_samples, n_samples),
                    num_threads=ga.num_threads,
                )
            periods = tails_ga._periods(self.max_season_length)
            diffs = periods * tails_ga._num_seas_diffs_periods(1, periods)
            diffs = diffs.astype(_indptr_dtype)
            self.diffs_.append(diffs)
            tails_indptr = _diffs_to_indptr(diffs)
            self.tails_.append(ga._tails(tails_indptr))
            transformed = ga._diffs(diffs)
        return transformed

    def inverse_transform(self, ga: GroupedArray) -> np.ndarray:
        """Invert the seasonal differences

        Args:
            ga (GroupedArray): Array with grouped data.

        Returns:
            np.ndarray: Array with the inverted transformation."""
        transformed = ga.data.copy()
        for diffs, tails in zip(self.diffs_[::-1], self.tails_[::-1]):
            ga = ga._with_data(transformed)
            transformed = ga._inv_diffs(diffs, tails)
        return transformed

    def update(self, ga: GroupedArray) -> np.ndarray:
        new_tails = []
        transformed = ga.data.copy()
        for i in range(self.max_diffs):
            ga = ga._with_data(transformed)
            tails_indptr = _diffs_to_indptr(self.diffs_[i])
            tails_ga = GroupedArray(
                self.tails_[i], tails_indptr, num_threads=ga.num_threads
            )
            combined = tails_ga._append(ga)
            new_tails.append(combined._tails(tails_indptr))
            combined_transformed = combined._diffs(self.diffs_[i])
            transformed = combined._with_data(combined_transformed)._tails(ga.indptr)
        self.tails_ = new_tails
        return transformed
