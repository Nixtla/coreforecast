import numpy as np
import pytest
from coreforecast.grouped_array import GroupedArray
from coreforecast.lag_transforms import RollingMean
from coreforecast.scalers import LocalStandardScaler

from . import dtypes


@pytest.fixture
def indptr(rng):
    lengths = rng.integers(low=1_000, high=2_000, size=5_000)
    return np.append(0, lengths.cumsum()).astype(np.int32)


@pytest.fixture
def data(rng, indptr):
    return rng.normal(size=indptr[-1])


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("num_threads", [1, 2])
def test_scalers(benchmark, data, indptr, dtype, num_threads):
    ga = GroupedArray(data.astype(dtype), indptr, num_threads=num_threads)
    scaler = LocalStandardScaler()
    benchmark(scaler.fit, ga)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("num_threads", [1, 2])
def test_lag_transforms(benchmark, dtype, data, indptr, num_threads):
    ga = GroupedArray(data.astype(dtype), indptr, num_threads=num_threads)
    tfm = RollingMean(lag=1, window_size=10, min_samples=5)
    benchmark(tfm.transform, ga)
