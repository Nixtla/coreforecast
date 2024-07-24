import warnings
from typing import Optional, Union

import numpy as np

from ._lib import ga as _ga_fns
from .utils import _diffs_to_indptr, _ensure_float, _indptr_dtype


class GroupedArray:
    """Array of grouped data

    Args:
        data (np.ndarray): 1d array with the values.
        indptr (np.ndarray): 1d array with the group boundaries.
        num_threads (int): Number of threads to use when computing transformations."""

    def __init__(self, data: np.ndarray, indptr: np.ndarray, num_threads: int = 1):
        if data.ndim != 1:
            raise ValueError("data must be a 1d array")
        if indptr.ndim != 1:
            raise ValueError("indptr must be a 1d array")
        if indptr[-1] != data.size:
            raise ValueError("Last element of indptr must be equal to the size of data")
        if num_threads < 1:
            warnings.warn(
                f"num_threads must be a positive integer, got: {num_threads}. "
                "Setting num_threads=1."
            )
            num_threads = 1
        self.data = np.ascontiguousarray(data, dtype=data.dtype)
        self.data = _ensure_float(self.data)
        if self.data.dtype == np.float32:
            self.prefix = "GroupedArrayFloat32"
        else:
            self.prefix = "GroupedArrayFloat64"
        self.indptr = indptr.astype(_indptr_dtype, copy=False)
        self.num_threads = num_threads

    def __len__(self):
        return self.indptr.size - 1

    def __getitem__(self, i):
        return self.data[self.indptr[i] : self.indptr[i + 1]]

    def _with_data(self, data: np.ndarray) -> "GroupedArray":
        if data.size != self.data.size:
            raise ValueError(
                "New data must have the same size as the original data. "
                f"Original size: {self.data.size:,}. New size: {data.size:,}"
            )
        data = data.astype(self.data.dtype, copy=False)
        data = np.ascontiguousarray(data)
        return GroupedArray(data, self.indptr, self.num_threads)

    def _append(self, other: "GroupedArray") -> "GroupedArray":
        if self.data.dtype != other.data.dtype:
            other = other._with_data(other.data.astype(self.data.dtype))
        if self.indptr.size != other.indptr.size:
            raise ValueError("Can only append arrays with the same number of groups")
        new_indptr = self.indptr + other.indptr
        new_data = np.empty_like(self.data, shape=new_indptr[-1])
        _ga_fns.append(self.data, self.indptr, self.num_threads, other.data, other.indptr, new_data, new_indptr)
        return GroupedArray(new_data, new_indptr, self.num_threads)

    def _scaler_fit(self, scaler_type: str) -> np.ndarray:
        stats = np.empty_like(self.data, shape=(len(self), 2))
        getattr(_ga_fns, f"{scaler_type}_scaler_stats")(self.data, self.indptr, self.num_threads, stats)
        return stats

    def _scaler_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        stats = stats.astype(self.data.dtype, copy=False)
        _ga_fns.scaler_transform(self.data, self.indptr, self.num_threads, stats, out)
        return out

    def _scaler_inverse_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        stats = stats.astype(self.data.dtype, copy=False)
        _ga_fns.scaler_inverse_transform(self.data, self.indptr, self.num_threads, stats, out)
        return out

    def _index_from_end(self, k: int) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _ga_fns.index_from_end(self.data, self.indptr, self.num_threads, k, out)
        return out

    def _head(self, k: int) -> np.ndarray:
        out = np.empty_like(self.data, shape=k * len(self))
        _ga_fns.head(self.data, self.indptr, self.num_threads, k, out)
        return out

    def _tail(self, k: int) -> np.ndarray:
        out = np.empty_like(self.data, shape=k * len(self))
        _ga_fns.tail(self.data, self.indptr, self.num_threads, k, out)
        return out

    def _tails(self, indptr_out: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data, shape=indptr_out[-1])
        _ga_fns.tails(self.data, self.indptr, self.num_threads, out, indptr_out)
        return out

    def _take(self, idxs: np.ndarray) -> np.ndarray:
        return np.hstack([self[i] for i in idxs])

    def _lag_transform(self, lag: int) -> np.ndarray:
        out = np.empty_like(self.data)
        _ga_fns.lag(self.data, self.indptr, self.num_threads, lag, out)
        return out

    def _rolling_transform(
        self, stat_name: str, lag: int, window_size: int, min_samples: int
    ) -> np.ndarray:
        out = np.empty_like(self.data)
        getattr(_ga_fns, f"rolling_{stat_name}_transform")(
            self.data, self.indptr, self.num_threads, lag, window_size, min_samples, out
        )
        return out

    def _rolling_quantile_transform(
        self, lag: int, p: float, window_size: int, min_samples: int
    ) -> np.ndarray:
        out = np.empty_like(self.data)
        _ga_fns.rolling_quantile_transform(
            self.data, self.indptr, self.num_threads, lag, p, window_size, min_samples, out
        )
        return out

    def _rolling_update(
        self, stat_name: str, lag: int, window_size: int, min_samples: int
    ) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        getattr(_ga_fns, f"rolling_{stat_name}_update")(
            self.data, self.indptr, self.num_threads, lag, window_size, min_samples, out
        )
        return out

    def _rolling_quantile_update(
        self, lag: int, p: float, window_size: int, min_samples: int
    ) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _ga_fns.rolling_quantile_update(
            self.data, self.indptr, self.num_threads, lag, p, window_size, min_samples, out
        )
        return out

    def _seasonal_rolling_transform(
        self,
        stat_name: str,
        lag: int,
        season_length: int,
        window_size: int,
        min_samples: int,
    ) -> np.ndarray:
        out = np.empty_like(self.data)
        getattr(_ga_fns, f"seasonal_rolling_{stat_name}_transform")(
            self.data, self.indptr, self.num_threads, lag, season_length, window_size, min_samples, out
        )
        return out

    def _seasonal_rolling_update(
        self,
        stat_name: str,
        lag: int,
        season_length: int,
        window_size: int,
        min_samples: int,
    ) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        getattr(_ga_fns, f"seasonal_rolling_{stat_name}_update")(
            self.data, self.indptr, self.num_threads, lag, season_length, window_size, min_samples, out
        )
        return out

    def _seasonal_rolling_quantile_transform(
        self,
        lag: int,
        p: float,
        season_length: int,
        window_size: int,
        min_samples: int,
    ) -> np.ndarray:
        out = np.empty_like(self.data)
        _ga_fns.seasonal_rolling_quantile_transform(
            self.data, self.indptr, self.num_threads, lag, season_length, p, window_size, min_samples, out
        )
        return out

    def _seasonal_rolling_quantile_update(
        self,
        lag: int,
        p: float,
        season_length: int,
        window_size: int,
        min_samples: int,
    ) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _ga_fns.seasonal_rolling_quantile_update(
            self.data, self.indptr, self.num_threads, lag, season_length, p, window_size, min_samples, out
        )
        return out

    def _expanding_transform_with_aggs(
        self,
        stat_name: str,
        lag: int,
        aggs: np.ndarray,
    ) -> np.ndarray:
        out = np.empty_like(self.data)
        getattr(_ga_fns, f"expanding_{stat_name}_transform")(
            self.data, self.indptr, self.num_threads, lag, out, aggs
        )
        return out

    def _expanding_transform(
        self,
        stat_name: str,
        lag: int,
    ) -> np.ndarray:
        out = np.empty_like(self.data)
        getattr(_ga_fns, f"expanding_{stat_name}_transform")(
            self.data, self.indptr, self.num_threads, lag, out
        )
        return out

    def _expanding_quantile_transform(self, lag: int, p: float) -> np.ndarray:
        out = np.empty_like(self.data)
        _ga_fns.expanding_quantile_transform(
            self.data, self.indptr, self.num_threads, lag, p, out
        )
        return out

    def _expanding_quantile_update(self, lag: int, p: float) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _ga_fns.expanding_quantile_update(
            self.data, self.indptr, self.num_threads, lag, p, out
        )
        return out

    def _exponentially_weighted_transform(
        self,
        stat_name: str,
        lag: int,
        alpha: float,
    ) -> np.ndarray:
        out = np.empty_like(self.data)
        getattr(_ga_fns, f"exponentially_weighted_{stat_name}_transform")(
            self.data, self.indptr, self.num_threads, lag, alpha, out
        )
        return out

    def _boxcox_fit(
        self, method: str, season_length: Optional[int], lower: float, upper: float
    ) -> np.ndarray:
        out = np.empty_like(self.data, shape=(len(self), 2))
        if method == "guerrero":
            assert season_length is not None
            _ga_fns.boxcox_lambda_guerrero(
                self.data, self.indptr, self.num_threads, season_length, lower, upper, out
            )
        else:
            _ga_fns.boxcox_lambda_loglik(
                self.data, self.indptr, self.num_threads, lower, upper, out
            )
        return out

    def _boxcox_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        stats = stats.astype(self.data.dtype, copy=False)
        _ga_fns.boxcox_transform(
            self.data, self.indptr, self.num_threads, stats, out
        )
        return out

    def _boxcox_inverse_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        stats = stats.astype(self.data.dtype, copy=False)
        _ga_fns.boxcox_inverse_transform(
            self.data, self.indptr, self.num_threads, stats, out
        )
        return out

    def _num_diffs(self, max_d: int = 1) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _ga_fns.num_diffs(
            self.data, self.indptr, self.num_threads, max_d, out
        )
        return out

    def _num_seas_diffs(self, season_length: int, max_d: int = 1) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _ga_fns.num_seas_diffs(
            self.data, self.indptr, self.num_threads, season_length, max_d, out
        )
        return out

    def _num_seas_diffs_periods(self, max_d: int, periods: np.ndarray) -> np.ndarray:
        periods_and_out = np.empty_like(self.data, shape=(len(self), 2))
        periods_and_out[:, 0] = periods
        _ga_fns.num_seas_diffs_periods(
            self.data, self.indptr, self.num_threads, max_d, periods_and_out
        )
        return periods_and_out[:, 1]

    def _periods(self, max_period: int) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _ga_fns.period(
            self.data, self.indptr, self.num_threads, max_period, out
        )
        return out

    def _diff(self, d: int) -> np.ndarray:
        out = np.empty_like(self.data)
        _ga_fns.difference(
            self.data, self.indptr, self.num_threads, d, out
        )
        return out

    def _diffs(self, ds: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        _ga_fns.differences(
            self.data, self.indptr, self.num_threads, ds, out
        )
        return out

    def _inv_diff(self, d: int, tails: np.ndarray) -> np.ndarray:
        ds = np.full(len(self), d, dtype=_indptr_dtype)
        return self._inv_diffs(ds, tails)

    def _inv_diffs(self, ds: np.ndarray, tails: np.ndarray) -> np.ndarray:
        tails_indptr = _diffs_to_indptr(ds)
        tails = tails.astype(self.data.dtype, copy=False)
        out = np.empty_like(self.data)
        _ga_fns.invert_differences(
            self.data, self.indptr, self.num_threads, tails, tails_indptr, self.indptr, out
        )
        return out
