import coreforecast.rolling as cf_rolling
import numpy as np
import pandas as pd
import pytest

from .test_lag_transforms import pd_rolling_quantile, pd_seasonal_rolling_quantile

quantile_ops = ["rolling_quantile", "seasonal_rolling_quantile"]
other_ops = [op for op in cf_rolling.__all__ if op not in quantile_ops]


@pytest.mark.parametrize("op", other_ops)
@pytest.mark.parametrize("min_samples", [None, 5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rolling(op, min_samples, dtype):
    window_size = 10
    season_length = 4
    x = np.random.rand(100).astype(dtype)
    serie = pd.Series(x)
    if op.startswith("seasonal"):
        args = (season_length, window_size, min_samples)
        serie = serie.groupby(np.arange(serie.size) % season_length)
    else:
        args = (window_size, min_samples)
    cf_res = getattr(cf_rolling, op)(x, *args)
    pd_res = getattr(
        serie.rolling(window=window_size, min_periods=min_samples),
        op.replace("rolling_", "").replace("seasonal_", ""),
    )()
    if op.startswith("seasonal"):
        pd_res = pd_res.reset_index(level=0, drop=True).sort_index()
    np.testing.assert_allclose(cf_res, pd_res, rtol=1e-5)

    if min_samples is not None:
        if op.startswith("seasonal"):
            n = season_length * (min_samples - 1)
        else:
            n = min_samples - 1
        x2 = np.random.rand(n)
        res2 = getattr(cf_rolling, op)(x2, *args)
        np.testing.assert_array_equal(res2, np.full_like(x2, np.nan))

        x3 = np.random.rand(n + 1)
        res3 = getattr(cf_rolling, op)(x3, *args)
        assert np.sum(~np.isnan(res3)) == 1


@pytest.mark.parametrize("op", quantile_ops)
@pytest.mark.parametrize("min_samples", [None, 5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_quantiles(op, min_samples, dtype, p):
    window_size = 10
    season_length = 4
    x = np.random.rand(100).astype(dtype)
    if op == "seasonal_rolling_quantile":
        pd_res = pd_seasonal_rolling_quantile(
            x, 0, p, season_length, window_size, min_samples
        )
        cf_res = cf_rolling.seasonal_rolling_quantile(
            x, p, season_length, window_size, min_samples
        )
    else:
        pd_res = pd_rolling_quantile(x, 0, p, window_size, min_samples)
        cf_res = cf_rolling.rolling_quantile(x, p, window_size, min_samples)
    np.testing.assert_allclose(pd_res, cf_res, atol=1e-5)
