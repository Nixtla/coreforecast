import ctypes
import warnings
from typing import Optional, Union

import numpy as np

from ._lib import _LIB, _indptr_dtype, _indptr_t
from .utils import _diffs_to_indptr, _ensure_float, _data_as_void_ptr, _pyfloat_to_np_c


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
        _LIB[f"{self.prefix}_Append"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            _data_as_void_ptr(other.data),
            _data_as_void_ptr(other.indptr),
            _indptr_t(other.indptr.size),
            _data_as_void_ptr(new_indptr),
            _data_as_void_ptr(new_data),
        )
        return GroupedArray(new_data, new_indptr, self.num_threads)

    def _pyfloat_to_c(self, x: float) -> Union[ctypes.c_float, ctypes.c_double]:
        if self.prefix == "GroupedArrayFloat32":
            return ctypes.c_float(x)
        return ctypes.c_double(x)

    def _scaler_fit(self, scaler_type: str) -> np.ndarray:
        stats = np.empty_like(self.data, shape=(len(self), 2))
        _LIB[f"{self.prefix}_{scaler_type}ScalerStats"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            _data_as_void_ptr(stats),
        )
        return stats

    def _scaler_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        stats = stats.astype(self.data.dtype, copy=False)
        _LIB[f"{self.prefix}_ScalerTransform"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            _data_as_void_ptr(stats),
            _data_as_void_ptr(out),
        )
        return out

    def _scaler_inverse_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        stats = stats.astype(self.data.dtype, copy=False)
        _LIB[f"{self.prefix}_ScalerInverseTransform"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            _data_as_void_ptr(stats),
            _data_as_void_ptr(out),
        )
        return out

    def _index_from_end(self, k: int) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _LIB[f"{self.prefix}_IndexFromEnd"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(k),
            _data_as_void_ptr(out),
        )
        return out

    def _head(self, k: int) -> np.ndarray:
        out = np.empty_like(self.data, shape=k * len(self))
        _LIB[f"{self.prefix}_Head"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(k),
            _data_as_void_ptr(out),
        )
        return out

    def _tail(self, k: int) -> np.ndarray:
        out = np.empty_like(self.data, shape=k * len(self))
        _LIB[f"{self.prefix}_Tail"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(k),
            _data_as_void_ptr(out),
        )
        return out

    def _tails(self, indptr_out: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data, shape=indptr_out[-1])
        _LIB[f"{self.prefix}_Tails"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            _data_as_void_ptr(indptr_out),
            _data_as_void_ptr(out),
        )
        return out

    def _take(self, idxs: np.ndarray) -> np.ndarray:
        return np.hstack([self[i] for i in idxs])

    def _lag_transform(self, lag: int) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_LagTransform"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(lag),
            _data_as_void_ptr(out),
        )
        return out

    def _rolling_transform(
        self, stat_name: str, lag: int, window_size: int, min_samples: int
    ) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_Rolling{stat_name}Transform"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(lag),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
        )
        return out

    def _rolling_quantile_transform(
        self, lag: int, p: float, window_size: int, min_samples: int
    ) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_RollingQuantileTransform"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(lag),
            self._pyfloat_to_c(p),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
        )
        return out

    def _rolling_update(
        self, stat_name: str, lag: int, window_size: int, min_samples: int
    ) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _LIB[f"{self.prefix}_Rolling{stat_name}Update"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(lag),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
        )
        return out

    def _rolling_quantile_update(
        self, lag: int, p: float, window_size: int, min_samples: int
    ) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _LIB[f"{self.prefix}_RollingQuantileUpdate"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(lag),
            self._pyfloat_to_c(p),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
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
        _LIB[f"{self.prefix}_SeasonalRolling{stat_name}Transform"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(lag),
            ctypes.c_int(season_length),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
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
        _LIB[f"{self.prefix}_SeasonalRolling{stat_name}Update"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(lag),
            ctypes.c_int(season_length),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
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
        _LIB[f"{self.prefix}_SeasonalRollingQuantileTransform"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(lag),
            ctypes.c_int(season_length),
            self._pyfloat_to_c(p),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
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
        _LIB[f"{self.prefix}_SeasonalRollingQuantileUpdate"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(lag),
            ctypes.c_int(season_length),
            self._pyfloat_to_c(p),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
        )
        return out

    def _expanding_transform_with_aggs(
        self,
        stat_name: str,
        lag: int,
        aggs: np.ndarray,
    ) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_Expanding{stat_name}Transform"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(lag),
            _data_as_void_ptr(out),
            _data_as_void_ptr(aggs),
        )
        return out

    def _expanding_transform(
        self,
        stat_name: str,
        lag: int,
    ) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_Expanding{stat_name}Transform"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(lag),
            _data_as_void_ptr(out),
        )
        return out

    def _expanding_quantile_transform(self, lag: int, p: float) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_ExpandingQuantileTransform"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(lag),
            self._pyfloat_to_c(p),
            _data_as_void_ptr(out),
        )
        return out

    def _expanding_quantile_update(self, lag: int, p: float) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _LIB[f"{self.prefix}_ExpandingQuantileUpdate"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(lag),
            self._pyfloat_to_c(p),
            _data_as_void_ptr(out),
        )
        return out

    def _exponentially_weighted_transform(
        self,
        stat_name: str,
        lag: int,
        alpha: float,
    ) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_ExponentiallyWeighted{stat_name}Transform"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(lag),
            self._pyfloat_to_c(alpha),
            _data_as_void_ptr(out),
        )
        return out

    def _boxcox_fit(
        self, method: str, season_length: Optional[int], lower: float, upper: float
    ) -> np.ndarray:
        out = np.empty_like(self.data, shape=(len(self), 2))
        if method == "guerrero":
            assert season_length is not None
            _LIB[f"{self.prefix}_BoxCoxLambdaGuerrero"](
                _data_as_void_ptr(self.data),
                _data_as_void_ptr(self.indptr),
                _indptr_t(self.indptr.size),
                ctypes.c_int(self.num_threads),
                ctypes.c_int(season_length),
                _pyfloat_to_np_c(lower, self.data.dtype),
                _pyfloat_to_np_c(upper, self.data.dtype),
                _data_as_void_ptr(out),
            )
        else:
            _LIB[f"{self.prefix}_BoxCoxLambdaLogLik"](
                _data_as_void_ptr(self.data),
                _data_as_void_ptr(self.indptr),
                _indptr_t(self.indptr.size),
                ctypes.c_int(self.num_threads),
                _pyfloat_to_np_c(lower, self.data.dtype),
                _pyfloat_to_np_c(upper, self.data.dtype),
                _data_as_void_ptr(out),
            )
        return out

    def _boxcox_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        stats = stats.astype(self.data.dtype, copy=False)
        _LIB[f"{self.prefix}_BoxCoxTransform"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            _data_as_void_ptr(stats),
            _data_as_void_ptr(out),
        )
        return out

    def _boxcox_inverse_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        stats = stats.astype(self.data.dtype, copy=False)
        _LIB[f"{self.prefix}_BoxCoxInverseTransform"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            _data_as_void_ptr(stats),
            _data_as_void_ptr(out),
        )
        return out

    def _num_diffs(self, max_d: int = 1) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _LIB[f"{self.prefix}_NumDiffs"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(max_d),
            _data_as_void_ptr(out),
        )
        return out

    def _num_seas_diffs(self, season_length: int, max_d: int = 1) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _LIB[f"{self.prefix}_NumSeasDiffs"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(season_length),
            ctypes.c_int(max_d),
            _data_as_void_ptr(out),
        )
        return out

    def _num_seas_diffs_periods(self, max_d: int, periods: np.ndarray) -> np.ndarray:
        periods_and_out = np.empty_like(self.data, shape=(len(self), 2))
        periods_and_out[:, 0] = periods
        _LIB[f"{self.prefix}_NumSeasDiffsPeriods"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(max_d),
            _data_as_void_ptr(periods_and_out),
        )
        return periods_and_out[:, 1]

    def _periods(self, max_period: int) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _LIB[f"{self.prefix}_Period"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_size_t(max_period),
            _data_as_void_ptr(out),
        )
        return out

    def _diff(self, d: int) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_Difference"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            ctypes.c_int(d),
            _data_as_void_ptr(out),
        )
        return out

    def _diffs(self, ds: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_Differences"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            _data_as_void_ptr(ds),
            _data_as_void_ptr(out),
        )
        return out

    def _inv_diff(self, d: int, tails: np.ndarray) -> np.ndarray:
        ds = np.full(len(self), d, dtype=_indptr_dtype)
        return self._inv_diffs(ds, tails)

    def _inv_diffs(self, ds: np.ndarray, tails: np.ndarray) -> np.ndarray:
        tails_indptr = _diffs_to_indptr(ds)
        tails = tails.astype(self.data.dtype, copy=False)
        out = np.empty_like(self.data)
        _LIB[f"{self.prefix}_InvertDifferences"](
            _data_as_void_ptr(self.data),
            _data_as_void_ptr(self.indptr),
            _indptr_t(self.indptr.size),
            ctypes.c_int(self.num_threads),
            _data_as_void_ptr(tails),
            _data_as_void_ptr(tails_indptr),
            _indptr_t(tails_indptr.size),
            _data_as_void_ptr(self.indptr),
            _data_as_void_ptr(out),
        )
        return out
