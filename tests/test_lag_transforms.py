import functools

import coreforecast.lag_transforms as lag_tf
import numpy as np
import pandas as pd
import pytest
from coreforecast.grouped_array import GroupedArray

from . import lag_tfms_map, min_samples, season_length, window_size

lag = 2
lengths = np.random.default_rng(seed=0).integers(low=100, high=200, size=100)
indptr = np.append(0, lengths.cumsum()).astype(np.int32)


def pd_transform(data, indptr, updates_only, lag, op, *args) -> np.ndarray:
    func, agg = op.rsplit("_", maxsplit=1)
    seasonal = func.startswith("seasonal")
    func = func.replace("seasonal_", "")
    sizes = np.diff(indptr)
    n_series = sizes.size
    df = pd.DataFrame(
        {
            "unique_id": np.repeat(np.arange(n_series), sizes),
            "ds": np.hstack([np.arange(size) for size in sizes]),
            "y": data,
        }
    )
    df["lagged"] = df.groupby("unique_id", observed=True)["y"].shift(lag)
    if seasonal:
        season_length, *args = args
        grouped_lagged = df.groupby(
            ["unique_id", np.arange(df.shape[0]) % season_length], observed=True
        )["lagged"]
    else:
        grouped_lagged = df.groupby("unique_id", observed=True)["lagged"]
    if func == "ewm":
        res = grouped_lagged.ewm(alpha=args[0], adjust=False).mean()
    else:
        res = getattr(getattr(grouped_lagged, func)(*args), agg)()
    res = res.sort_index(level=-1)
    if updates_only:
        res = res.groupby("unique_id", observed=True).tail(1)
    return res.to_numpy()


def pd_rolling_quantile(x, lag, p, window_size, min_samples):
    return (
        pd.Series(x)
        .shift(lag)
        .rolling(window=window_size, min_periods=min_samples)
        .quantile(p)
    )


def pd_seasonal_rolling_quantile(x, lag, p, season_length, window_size, min_samples):
    out = np.empty_like(x)
    x = pd.Series(x).shift(lag).to_numpy()
    for season in range(season_length):
        out[season::season_length] = pd_rolling_quantile(
            x[season::season_length], 0, p, window_size, min_samples
        )
    return out


def pd_expanding_quantile(x, lag, p):
    return pd.Series(x).shift(lag).expanding().quantile(p)


@pytest.fixture
def data(rng):
    return 10 * rng.random(indptr[-1])


def test_lag():
    ga = GroupedArray(np.array([1, 2, 3, 10, 11]), np.array([0, 3, 5]))

    lag2 = lag_tf.Lag(2)
    np.testing.assert_allclose(
        lag2.transform(ga),
        np.array([np.nan, np.nan, 1, np.nan, np.nan]),
    )
    np.testing.assert_allclose(
        lag2.update(ga),
        np.array([2, 10]),
    )

    lag3 = lag_tf.Lag(3)
    np.testing.assert_allclose(
        lag3.transform(ga),
        np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
    )
    np.testing.assert_allclose(
        lag3.update(ga),
        np.array([1, np.nan]),
    )


@pytest.mark.parametrize("comb", list(lag_tfms_map.keys()))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_correctness(data, comb, dtype):
    # pandas, computing in float64, stands in for the exact result. The float32
    # sliding std drifts by up to 8.3e-5 over a group here, and that error is
    # absolute: a window of nearly equal values has no relative accuracy left.
    if dtype == np.float32:
        rtol, atol = 1e-5, 1e-3 if "std" in comb else 1e-4
    else:
        rtol, atol = 1e-7, 0.0
    data = data.astype(dtype, copy=True)
    ga = GroupedArray(data, indptr)
    cf, args = lag_tfms_map[comb]
    # transform
    pd_res = pd_transform(data, indptr, False, lag, comb, *args)
    cobj = cf(lag, *args)
    cres = cobj.transform(ga)
    np.testing.assert_allclose(pd_res, cres, atol=atol, rtol=rtol)
    # update
    pd_res = pd_transform(data, indptr, True, lag - 1, comb, *args)
    cres = cobj.update(ga)
    np.testing.assert_allclose(pd_res, cres, atol=atol, rtol=rtol)
    # stack
    combined = cobj.stack([cobj, cobj])
    if hasattr(cobj, "stats_"):
        assert combined.stats_.shape[0] == 2 * cobj.stats_.shape[0]
    else:
        assert combined is cobj


@pytest.mark.parametrize("window_type", ["rolling", "seasonal_rolling", "expanding"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("p", [0.01, 0.1, 0.5, 0.9, 0.99])
def test_correctness_quantiles(data, dtype, p, window_type):
    rtol = 1e-5 if dtype == np.float32 else 1e-7
    data = data.astype(dtype, copy=True)
    ga = GroupedArray(data, indptr)
    if window_type == "rolling":
        core_cls = lag_tf.RollingQuantile(lag, p, window_size, min_samples)
        pd_fun = pd_rolling_quantile
        pd_kwargs = dict(window_size=window_size, min_samples=min_samples)
    elif window_type == "seasonal_rolling":
        core_cls = lag_tf.SeasonalRollingQuantile(
            lag, p, season_length, window_size, min_samples
        )
        pd_fun = pd_seasonal_rolling_quantile
        pd_kwargs = dict(
            season_length=season_length,
            window_size=window_size,
            min_samples=min_samples,
        )
    else:
        core_cls = lag_tf.ExpandingQuantile(lag, p)
        pd_fun = pd_expanding_quantile
        pd_kwargs = {}
    cres = core_cls.transform(ga)
    core_cls.lag = lag + 1
    cres_upd = core_cls.update(ga)
    pres = np.hstack([pd_fun(ga[i], lag, p, **pd_kwargs) for i in range(len(ga))])
    pres_upd = pres[indptr[1:] - 1]
    np.testing.assert_allclose(cres, pres, rtol=rtol)
    np.testing.assert_allclose(cres_upd, pres_upd, rtol=rtol)


# An update must produce what the transform would put at the same position, so
# the C++ transform is the single definition of what skipna means.
update_consistency_tfms = {
    "RollingMean": lambda lag, skipna: lag_tf.RollingMean(
        lag, window_size, min_samples, skipna
    ),
    "RollingStd": lambda lag, skipna: lag_tf.RollingStd(
        lag, window_size, min_samples, skipna
    ),
    "RollingMin": lambda lag, skipna: lag_tf.RollingMin(
        lag, window_size, min_samples, skipna
    ),
    "RollingMax": lambda lag, skipna: lag_tf.RollingMax(
        lag, window_size, min_samples, skipna
    ),
    "RollingQuantile": lambda lag, skipna: lag_tf.RollingQuantile(
        lag, 0.5, window_size, min_samples, skipna
    ),
    "SeasonalRollingMean": lambda lag, skipna: lag_tf.SeasonalRollingMean(
        lag, season_length, window_size, min_samples, skipna
    ),
    "SeasonalRollingStd": lambda lag, skipna: lag_tf.SeasonalRollingStd(
        lag, season_length, window_size, min_samples, skipna
    ),
    "SeasonalRollingMin": lambda lag, skipna: lag_tf.SeasonalRollingMin(
        lag, season_length, window_size, min_samples, skipna
    ),
    "SeasonalRollingMax": lambda lag, skipna: lag_tf.SeasonalRollingMax(
        lag, season_length, window_size, min_samples, skipna
    ),
    "SeasonalRollingQuantile": lambda lag, skipna: lag_tf.SeasonalRollingQuantile(
        lag, 0.5, season_length, window_size, min_samples, skipna
    ),
    "ExpandingMean": lambda lag, skipna: lag_tf.ExpandingMean(lag, skipna),
    "ExpandingStd": lambda lag, skipna: lag_tf.ExpandingStd(lag, skipna),
    "ExpandingMin": lambda lag, skipna: lag_tf.ExpandingMin(lag, skipna),
    "ExpandingMax": lambda lag, skipna: lag_tf.ExpandingMax(lag, skipna),
    "ExpandingQuantile": lambda lag, skipna: lag_tf.ExpandingQuantile(lag, 0.5, skipna),
    "ExponentiallyWeightedMean": lambda lag, skipna: lag_tf.ExponentiallyWeightedMean(
        lag, 0.5, skipna
    ),
}


def test_update_consistency_covers_every_transform():
    # Lag has no statistic to accumulate and no skipna argument
    assert set(update_consistency_tfms) == set(lag_tf.__all__) - {"Lag"}
    windowed = {name for name in update_consistency_tfms if "Rolling" in name}
    assert windowed | set(accumulating_tfms) == set(update_consistency_tfms)
    assert not windowed & set(accumulating_tfms)


n_hist = 40
# enough updates for a NaN that arrives through one to leave the windows again
n_obs = 50


def consistency_series(name, rng):
    x = rng.uniform(1.0, 11.0, size=n_obs)
    if name == "clean":
        return x
    if name == "leading_nans":
        x[:5] = np.nan
    elif name == "all_nan_history":
        # the kernel never runs, so its stats row is NaN and the update seeds
        x[:n_hist] = np.nan
    elif name == "single_valid":
        x[: n_hist - 1] = np.nan
    elif name == "interior_nan":
        # far enough back that every window, the seasonal phases included, has
        # moved past it before the first update
        x[7] = np.nan
    elif name == "leading_and_interior_nans":
        x[:3] = np.nan
        x[9] = np.nan
    elif name == "nan_via_update":
        x[n_hist + 1] = np.nan
    elif name == "long_nan_run":
        x[n_hist : n_hist + 3] = np.nan
    else:
        raise ValueError(f"unknown series: {name}")
    return x


# the only NaNs skipna=False supports
leading_run_series = [
    "clean",
    "leading_nans",
    "all_nan_history",
    "single_valid",
]
# NaNs anywhere else. The transforms report NaN from one of those to the end
# of the group; a windowed update would have to scan the group's values every
# time to know, which is the work skipna=False exists to skip.
interior_nan_series = [
    "interior_nan",
    "leading_and_interior_nans",
    "nan_via_update",
    "long_nan_run",
]
# the transforms whose update accumulates over the whole group: the NaN is
# already in the state they resume from, so keeping it costs nothing
accumulating_tfms = [
    "ExpandingMean",
    "ExpandingStd",
    "ExpandingMin",
    "ExpandingMax",
    "ExpandingQuantile",
    "ExponentiallyWeightedMean",
]


def consistency_ga(arrays, dtype):
    return GroupedArray(
        np.hstack(arrays).astype(dtype, copy=False),
        np.append(0, np.cumsum([len(a) for a in arrays])).astype(np.int32),
    )


def update_and_transform(factory, arrays, dtype):
    tfm = factory()
    tfm.transform(consistency_ga([a[:n_hist] for a in arrays], dtype))
    updates, expected = [], []
    for n in range(n_hist, n_obs + 1):
        step = [a[:n] for a in arrays]
        updates.append(tfm.update(consistency_ga(step, dtype)))
        # the padding takes the position the update computed, and is never read
        padded = [np.append(a, 0.0) for a in step]
        out = factory().transform(consistency_ga(padded, dtype))
        expected.append(out[np.cumsum([len(a) for a in padded]) - 1])
    return np.array(updates), np.array(expected)


def check_update_matches_transform(name, skipna, series, dtype):
    rng = np.random.default_rng(0)
    arrays = [consistency_series("clean", rng), consistency_series(series, rng)]
    rtol, atol = (1e-3, 1e-4) if dtype == np.float32 else (1e-9, 1e-12)
    for lag in (1, 2):
        got, expected = update_and_transform(
            functools.partial(update_consistency_tfms[name], lag, skipna),
            arrays,
            dtype,
        )
        np.testing.assert_allclose(
            got,
            expected,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            err_msg=f"{name} lag={lag} skipna={skipna} series={series}",
        )


@pytest.mark.parametrize("name", list(update_consistency_tfms))
@pytest.mark.parametrize("series", leading_run_series)
@pytest.mark.parametrize("skipna", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_update_matches_transform(name, skipna, series, dtype):
    check_update_matches_transform(name, skipna, series, dtype)


@pytest.mark.parametrize("name", list(update_consistency_tfms))
@pytest.mark.parametrize("series", interior_nan_series)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_update_matches_transform_with_skipna(name, series, dtype):
    check_update_matches_transform(name, True, series, dtype)
