import numpy as np
import pytest
import window_ops.rolling as wops_rolling

import coreforecast.rolling as cf_rolling


@pytest.mark.parametrize("op", cf_rolling.__all__)
@pytest.mark.parametrize("min_samples", [None, 5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rolling(op, min_samples, dtype):
    window_size = 10
    x = np.random.rand(100).astype(dtype)
    cf_res = getattr(cf_rolling, op)(x, window_size, min_samples)
    wo_res = getattr(wops_rolling, op)(x, window_size, min_samples)
    np.testing.assert_allclose(cf_res, wo_res, rtol=1e-5)
