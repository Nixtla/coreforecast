import functools

import coreforecast.lag_transforms as lag_tf
import numpy as np
import pandas as pd
import pytest
from coreforecast.grouped_array import GroupedArray

from . import lag_tfms_map, min_samples, season_length, window_size

lag = 2
lengths = np.random.randint(low=100, high=200, size=100)
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
def data():
    return 10 * np.random.rand(indptr[-1])


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
    atol = 1e-4
    rtol = 1e-5 if dtype == np.float32 else 1e-7
    if dtype == np.float32:
        if "rolling_std" in comb:
            rtol = 1e-2
    if "expanding_std" in comb:
        rtol *= 100
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


# An update must produce what the transform would put at the same position.
# The expanding and EWM updates carry their accumulator in Python while every
# other update delegates to a kernel, so this keeps the C++ transform as the
# single definition of what skipna means for all of them.
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
    # Lag is the only transform with no statistic to accumulate and no skipna
    # argument, so it is the only one the property below doesn't apply to.
    assert set(update_consistency_tfms) == set(lag_tf.__all__) - {"Lag"}


n_hist = 40
n_obs = 45


def consistency_series(name, rng):
    x = rng.uniform(1.0, 11.0, size=n_obs)
    if name == "clean":
        return x
    if name == "leading_nans":
        x[:5] = np.nan
    elif name == "all_nan_history":
        # the kernel never runs for this group, so the transform leaves a
        # NaN-filled stats row behind and the update has to seed from scratch
        x[:n_hist] = np.nan
    elif name == "single_valid":
        x[: n_hist - 1] = np.nan
    elif name == "interior_nan":
        x[17] = np.nan
    elif name == "nan_via_update":
        x[n_hist + 1] = np.nan
    elif name == "long_nan_run":
        x[n_hist : n_hist + 3] = np.nan
    else:
        raise ValueError(f"unknown series: {name}")
    return x


# with skipna=False only a leading run of NaNs is defined behaviour, so the
# series that put one anywhere else are exercised with skipna=True only
update_consistency_cases = [
    (False, "clean"),
    (False, "leading_nans"),
    (False, "all_nan_history"),
    (False, "single_valid"),
    (True, "clean"),
    (True, "leading_nans"),
    (True, "all_nan_history"),
    (True, "single_valid"),
    (True, "interior_nan"),
    (True, "nan_via_update"),
    (True, "long_nan_run"),
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
        # the transform's value one position past the end of step is taken over
        # exactly the observations the update has consumed; the padding sits at
        # that position and is never read
        padded = [np.append(a, 0.0) for a in step]
        out = factory().transform(consistency_ga(padded, dtype))
        expected.append(out[np.cumsum([len(a) for a in padded]) - 1])
    return np.array(updates), np.array(expected)


@pytest.mark.parametrize("name", list(update_consistency_tfms))
@pytest.mark.parametrize("skipna,series", update_consistency_cases)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_update_matches_transform(name, skipna, series, dtype):
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
