import numpy as np
import pytest
from window_ops.expanding import *
from window_ops.ewm import ewm_mean
from window_ops.rolling import *
from window_ops.shift import shift_array

from coreforecast.grouped_array import GroupedArray
from coreforecast.lag_transforms import *

lag = 2
season_length = 7
window_size = 4
min_samples = 2
lengths = np.random.randint(low=100, high=200, size=100)
indptr = np.append(0, lengths.cumsum()).astype(np.int32)
data = 10 * np.random.rand(indptr[-1])
ga = GroupedArray(data, indptr)


def transform(data, indptr, updates_only, lag, func, *args) -> np.ndarray:
    n_series = len(indptr) - 1
    if updates_only:
        out = np.empty_like(data[:n_series])
        for i in range(n_series):
            lagged = shift_array(data[indptr[i] : indptr[i + 1]], lag)
            out[i] = func(lagged, *args)[-1]
    else:
        out = np.empty_like(data)
        for i in range(n_series):
            lagged = shift_array(data[indptr[i] : indptr[i + 1]], lag)
            out[indptr[i] : indptr[i + 1]] = func(lagged, *args)
    return out


@pytest.mark.parametrize(
    "comb",
    [
        (rolling_mean, RollingMean, [window_size, min_samples]),
        (rolling_std, RollingStd, [window_size, min_samples]),
        (rolling_min, RollingMin, [window_size, min_samples]),
        (rolling_max, RollingMax, [window_size, min_samples]),
        (
            seasonal_rolling_mean,
            SeasonalRollingMean,
            [season_length, window_size, min_samples],
        ),
        (
            seasonal_rolling_std,
            SeasonalRollingStd,
            [season_length, window_size, min_samples],
        ),
        (
            seasonal_rolling_min,
            SeasonalRollingMin,
            [season_length, window_size, min_samples],
        ),
        (
            seasonal_rolling_max,
            SeasonalRollingMax,
            [season_length, window_size, min_samples],
        ),
        (expanding_mean, ExpandingMean, []),
        (expanding_std, ExpandingStd, []),
        (expanding_min, ExpandingMin, []),
        (expanding_max, ExpandingMax, []),
        (ewm_mean, ExponentiallyWeightedMean, [0.8]),
    ],
)
def test_correctness(comb):
    wf, cf, args = comb
    # transform
    wres = transform(data, indptr, False, lag, wf, *args)
    cobj = cf(lag, *args)
    cres = cobj.transform(ga)
    np.testing.assert_allclose(wres, cres)
    # update
    wres = transform(data, indptr, True, lag - 1, wf, *args)
    cres = cobj.update(ga)
    np.testing.assert_allclose(wres, cres)
