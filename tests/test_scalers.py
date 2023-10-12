import numpy as np
import pytest
from utilsforecast.target_transforms import (
    LocalStandardScaler as UtilsStandardScaler,
    LocalMinMaxScaler as UtilsMinMaxScaler,
    LocalRobustScaler as UtilsRobustScaler,
)

from coreforecast.grouped_array import GroupedArray
from coreforecast.scalers import (
    LocalMinMaxScaler,
    LocalRobustScaler,
    LocalStandardScaler,
)


@pytest.fixture
def indptr():
    lengths = np.random.randint(low=1_000, high=2_000, size=5_000)
    return np.append(0, lengths.cumsum()).astype(np.int32)


@pytest.fixture
def data(indptr):
    return np.random.randn(indptr[-1])


def std_scaler_stats(x):
    return np.nanmean(x), np.nanstd(x)


def minmax_scaler_stats(x):
    min, max = np.nanmin(x), np.nanmax(x)
    return min, max - min


def robust_scaler_iqr_stats(x):
    q25, median, q75 = np.nanquantile(x, [0.25, 0.5, 0.75])
    return median, q75 - q25


def robust_scaler_mad_stats(x):
    median = np.nanmedian(x)
    mad = np.median(np.abs(x - median))
    return median, mad


def scaler_transform(x, stats):
    offset, scale = stats
    return (x - offset) / scale


def scaler_inverse_transform(x, stats):
    offset, scale = stats
    return x * scale + offset


scaler2fns = {
    "standard": std_scaler_stats,
    "minmax": minmax_scaler_stats,
    "robust-iqr": robust_scaler_iqr_stats,
    "robust-mad": robust_scaler_mad_stats,
}
scaler2core = {
    "standard": LocalStandardScaler(),
    "minmax": LocalMinMaxScaler(),
    "robust-iqr": LocalRobustScaler("iqr"),
    "robust-mad": LocalRobustScaler("mad"),
}
scaler2utils = {
    "standard": UtilsStandardScaler(),
    "minmax": UtilsMinMaxScaler(),
    "robust-iqr": UtilsRobustScaler("iqr"),
    "robust-mad": UtilsRobustScaler("mad"),
}
scalers = list(scaler2fns.keys())


@pytest.mark.parametrize("scaler_name", scalers)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_correctness(data, indptr, scaler_name, dtype):
    ga = GroupedArray(data.astype(dtype), indptr)
    scaler = scaler2core[scaler_name]
    stats_fn = scaler2fns[scaler_name]
    scaler.fit(ga)

    # stats
    expected_stats = np.hstack([stats_fn(grp) for grp in ga]).reshape(-1, 2)
    np.testing.assert_allclose(scaler.stats_, expected_stats, atol=1e-6, rtol=1e-6)

    # transform
    transformed = scaler.transform(ga)
    expected_transformed = np.hstack(
        [scaler_transform(grp, scaler.stats_[i]) for i, grp in enumerate(ga)]
    )
    np.testing.assert_allclose(transformed, expected_transformed, atol=1e-6, rtol=1e-6)

    # inverse transform
    transformed_ga = GroupedArray(transformed, ga.indptr)
    restored = scaler.inverse_transform(transformed_ga)
    expected_restored = np.hstack(
        [
            scaler_inverse_transform(grp, scaler.stats_[i])
            for i, grp in enumerate(transformed_ga)
        ]
    )
    np.testing.assert_allclose(restored, expected_restored, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("scaler_name", scalers)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("lib", ["core", "utils"])
def test_performance(benchmark, data, indptr, scaler_name, dtype, lib):
    ga = GroupedArray(data.astype(dtype), indptr)
    if lib == "core":
        scaler = scaler2core[scaler_name]
    else:
        scaler = scaler2utils[scaler_name]
    benchmark(lambda: scaler.fit_transform(ga))
