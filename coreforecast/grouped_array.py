import ctypes
import platform
import sys

import numpy as np

if sys.version_info < (3, 10):
    from importlib_resources import files
else:
    from importlib.resources import files


DTYPE_FLOAT32 = ctypes.c_int(0)
DTYPE_FLOAT64 = ctypes.c_int(1)

if platform.system() in ("Windows", "Microsoft"):
    prefix = "Release"
    extension = "dll"
else:
    prefix = ""
    extension = "so"

_LIB = ctypes.CDLL(
    str(files("coreforecast").joinpath("lib", prefix, f"libcoreforecast.{extension}"))
)


def _data_as_void_ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))


class GroupedArray:
    def __init__(self, data: np.ndarray, indptr: np.ndarray, num_threads: int = 1):
        if data.dtype == np.float32:
            self.dtype = DTYPE_FLOAT32
        elif data.dtype == np.float64:
            self.dtype = DTYPE_FLOAT64
        else:
            self.dtype = DTYPE_FLOAT32
            data = data.astype(np.float32)
        self.data = data
        if indptr.dtype != np.int32:
            indptr = indptr.astype(np.int32)
        self.indptr = indptr
        self._handle = ctypes.c_void_p()
        _LIB.GroupedArray_Create(
            _data_as_void_ptr(data),
            ctypes.c_int32(data.size),
            indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int(indptr.size),
            ctypes.c_int(num_threads),
            self.dtype,
            ctypes.byref(self._handle),
        )

    def __del__(self):
        _LIB.GroupedArray_Delete(self._handle, self.dtype)

    def __len__(self):
        return self.indptr.size - 1

    def __getitem__(self, i):
        return self.data[self.indptr[i] : self.indptr[i + 1]]

    def scaler_fit(self, stats_fn_name: str) -> np.ndarray:
        stats = np.full_like(self.data, np.nan, shape=(len(self), 2))
        _LIB[stats_fn_name](
            self._handle,
            self.dtype,
            _data_as_void_ptr(stats),
        )
        return stats

    def scaler_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.full_like(self.data, np.nan)
        _LIB.GroupedArray_ScalerTransform(
            self._handle,
            _data_as_void_ptr(stats),
            self.dtype,
            _data_as_void_ptr(out),
        )
        return out

    def scaler_inverse_transform(self, stats: np.ndarray) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB.GroupedArray_ScalerInverseTransform(
            self._handle,
            _data_as_void_ptr(stats),
            self.dtype,
            _data_as_void_ptr(out),
        )
        return out

    def lag_transform(self, lag: int) -> np.ndarray:
        out = np.empty_like(self.data)
        _LIB.GroupedArray_LagTransform(
            self._handle,
            self.dtype,
            ctypes.c_int(lag),
            _data_as_void_ptr(out),
        )
        return out

    def rolling_transform(
        self, tfm_name: str, lag: int, window_size: int, min_samples: int
    ) -> np.ndarray:
        out = np.full_like(self.data, np.nan)
        _LIB[f"GroupedArray_Rolling{tfm_name}Transform"](
            self._handle,
            self.dtype,
            ctypes.c_int(lag),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
        )
        return out

    def rolling_update(
        self, tfm_name: str, lag: int, window_size: int, min_samples: int
    ) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _LIB[f"GroupedArray_Rolling{tfm_name}Update"](
            self._handle,
            self.dtype,
            ctypes.c_int(lag),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
        )
        return out

    def seasonal_rolling_transform(
        self,
        tfm_name: str,
        lag: int,
        season_length: int,
        window_size: int,
        min_samples: int,
    ) -> np.ndarray:
        out = np.full_like(self.data, np.nan)
        _LIB[f"GroupedArray_SeasonalRolling{tfm_name}Transform"](
            self._handle,
            self.dtype,
            ctypes.c_int(lag),
            ctypes.c_int(season_length),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
        )
        return out

    def seasonal_rolling_update(
        self,
        tfm_name: str,
        lag: int,
        season_length: int,
        window_size: int,
        min_samples: int,
    ) -> np.ndarray:
        out = np.empty_like(self.data, shape=len(self))
        _LIB[f"GroupedArray_SeasonalRolling{tfm_name}Update"](
            self._handle,
            self.dtype,
            ctypes.c_int(lag),
            ctypes.c_int(season_length),
            ctypes.c_int(window_size),
            ctypes.c_int(min_samples),
            _data_as_void_ptr(out),
        )
        return out

    def expanding_transform_with_aggs(
        self,
        tfm_name: str,
        lag: int,
        aggs: np.ndarray,
    ) -> np.ndarray:
        out = np.full_like(self.data, np.nan)
        _LIB[f"GroupedArray_Expanding{tfm_name}Transform"](
            self._handle,
            self.dtype,
            ctypes.c_int(lag),
            _data_as_void_ptr(out),
            _data_as_void_ptr(aggs),
        )
        return out

    def expanding_transform(
        self,
        tfm_name: str,
        lag: int,
    ) -> np.ndarray:
        out = np.full_like(self.data, np.nan)
        _LIB[f"GroupedArray_Expanding{tfm_name}Transform"](
            self._handle,
            self.dtype,
            ctypes.c_int(lag),
            _data_as_void_ptr(out),
        )
        return out
