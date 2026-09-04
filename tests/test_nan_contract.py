"""The two supported NaN paths.

skipna=False assumes NaNs form a leading run only; skipna=True ignores them
wherever they are. Both must behave the same through the free functions and
through GroupedArray.
"""

import numpy as np
import pandas as pd
import pytest

from coreforecast.expanding import (
    expanding_max,
    expanding_mean,
    expanding_min,
    expanding_quantile,
    expanding_std,
)
from coreforecast.exponentially_weighted import exponentially_weighted_mean
from coreforecast.grouped_array import GroupedArray
from coreforecast.rolling import (
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_quantile,
    rolling_std,
    seasonal_rolling_max,
    seasonal_rolling_mean,
    seasonal_rolling_min,
    seasonal_rolling_quantile,
    seasonal_rolling_std,
)

window_size = 3
min_samples = 2
season_length = 2

# (name, free function, equivalent GroupedArray call)
equivalences = [
    ("rolling_mean", lambda x: rolling_mean(x, window_size, min_samples),
     lambda ga: ga._rolling_mean(0, window_size, min_samples)),
    ("rolling_std", lambda x: rolling_std(x, window_size, min_samples),
     lambda ga: ga._rolling_std(0, window_size, min_samples)),
    ("rolling_min", lambda x: rolling_min(x, window_size, min_samples),
     lambda ga: ga._rolling_min(0, window_size, min_samples)),
    ("rolling_max", lambda x: rolling_max(x, window_size, min_samples),
     lambda ga: ga._rolling_max(0, window_size, min_samples)),
    ("rolling_quantile", lambda x: rolling_quantile(x, 0.5, window_size, min_samples),
     lambda ga: ga._rolling_quantile(0, 0.5, window_size, min_samples)),
    ("seasonal_rolling_mean",
     lambda x: seasonal_rolling_mean(x, season_length, window_size, min_samples),
     lambda ga: ga._seasonal_rolling_mean(0, season_length, window_size, min_samples)),
    ("seasonal_rolling_std",
     lambda x: seasonal_rolling_std(x, season_length, window_size, min_samples),
     lambda ga: ga._seasonal_rolling_std(0, season_length, window_size, min_samples)),
    ("seasonal_rolling_min",
     lambda x: seasonal_rolling_min(x, season_length, window_size, min_samples),
     lambda ga: ga._seasonal_rolling_min(0, season_length, window_size, min_samples)),
    ("seasonal_rolling_max",
     lambda x: seasonal_rolling_max(x, season_length, window_size, min_samples),
     lambda ga: ga._seasonal_rolling_max(0, season_length, window_size, min_samples)),
    ("seasonal_rolling_quantile",
     lambda x: seasonal_rolling_quantile(x, 0.5, season_length, window_size, min_samples),
     lambda ga: ga._seasonal_rolling_quantile(0, 0.5, season_length, window_size, min_samples)),
    ("expanding_mean", expanding_mean, lambda ga: ga._expanding_mean(0)[0]),
    ("expanding_std", expanding_std, lambda ga: ga._expanding_std(0)[0]),
    ("expanding_min", expanding_min, lambda ga: ga._expanding_min(0)),
    ("expanding_max", expanding_max, lambda ga: ga._expanding_max(0)),
    ("expanding_quantile", lambda x: expanding_quantile(x, 0.5),
     lambda ga: ga._expanding_quantile(0, 0.5)),
    ("ewm_mean", lambda x: exponentially_weighted_mean(x, 0.5),
     lambda ga: ga._exponentially_weighted_mean(0, 0.5)),
]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_leading", [0, 1, 3])
@pytest.mark.parametrize("name,free_fn,ga_fn", equivalences, ids=[e[0] for e in equivalences])
def test_leading_nans_match_grouped_array(name, free_fn, ga_fn, n_leading, dtype):
    x = np.concatenate(
        [np.full(n_leading, np.nan), np.arange(1, 9)]
    ).astype(dtype)
    ga = GroupedArray(x, np.array([0, x.size], dtype=np.int32))
    np.testing.assert_allclose(free_fn(x), ga_fn(ga), equal_nan=True, rtol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_leading_nans_are_preserved_and_the_rest_is_computed(dtype):
    x = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    res = rolling_mean(x, window_size=3, min_samples=1)
    assert np.isnan(res[:2]).all()
    np.testing.assert_allclose(res[2:], [1.0, 1.5, 2.0, 3.0, 4.0], rtol=1e-6)


@pytest.mark.parametrize(
    "fn", [rolling_mean, rolling_min, rolling_max, expanding_mean, expanding_min]
)
def test_degenerate_inputs(fn):
    kwargs = {} if fn.__name__.startswith("expanding") else {"window_size": 3}
    np.testing.assert_array_equal(fn(np.array([]), **kwargs), np.array([]))
    assert np.isnan(fn(np.full(4, np.nan), **kwargs)).all()


def test_skipna_min_max_expire_stale_front_on_a_nan_step():
    # regression: the monotonic deque's front expires once per *step*, so it must
    # be checked on NaN steps too, otherwise an out-of-window entry is returned
    x = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0])
    # index 3's window is [2, 3, nan]; position 0 has left it
    np.testing.assert_allclose(
        rolling_min(x, 3, 1, skipna=True), [1.0, 1.0, 1.0, 2.0, 3.0, 5.0, 5.0]
    )
    np.testing.assert_allclose(
        rolling_max(x, 3, 1, skipna=True), [1.0, 2.0, 3.0, 3.0, 5.0, 6.0, 7.0]
    )


@pytest.mark.parametrize("stat", ["min", "max"])
def test_skipna_min_max_match_pandas_with_scattered_nans(stat, rng):
    # min_samples=1 is where coreforecast's positional min_samples and pandas'
    # observation-based min_periods agree, so pandas is a valid oracle there
    fn = {"min": rolling_min, "max": rolling_max}[stat]
    for _ in range(200):
        n = int(rng.integers(2, 30))
        w = int(rng.integers(1, 10))
        x = rng.normal(size=n)
        x[rng.random(n) < 0.4] = np.nan
        expected = getattr(pd.Series(x).rolling(w, min_periods=1), stat)().to_numpy()
        np.testing.assert_allclose(
            fn(x, w, 1, skipna=True), expected, equal_nan=True
        )
