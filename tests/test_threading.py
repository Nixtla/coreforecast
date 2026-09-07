"""The multi-threaded GroupedArray path.

Groups are split across threads, so results must not depend on how many are
used, and a failure inside a worker must surface as a Python exception rather
than taking the interpreter down with it.
"""

import numpy as np
import pytest

from coreforecast._lib.grouped_array import (
    _GroupedArrayFloat32,
    _GroupedArrayFloat64,
)
from coreforecast.grouped_array import GroupedArray

thread_counts = [2, 3, 8]

operations = [
    ("rolling_mean", lambda ga: ga._rolling_mean(1, 5, 2)),
    ("rolling_std", lambda ga: ga._rolling_std(1, 5, 2)),
    ("rolling_min", lambda ga: ga._rolling_min(1, 5, 2)),
    ("rolling_max", lambda ga: ga._rolling_max(1, 5, 2)),
    ("rolling_quantile", lambda ga: ga._rolling_quantile(1, 0.5, 5, 2)),
    ("seasonal_rolling_mean", lambda ga: ga._seasonal_rolling_mean(1, 7, 3, 2)),
    ("expanding_mean", lambda ga: ga._expanding_mean(1)[0]),
    ("expanding_std", lambda ga: ga._expanding_std(1)[0]),
    ("ewm_mean", lambda ga: ga._exponentially_weighted_mean(1, 0.8)),
    ("minmax_stats", lambda ga: ga._minmax_stats()),
    ("standard_stats", lambda ga: ga._standard_stats()),
    ("robust_iqr_stats", lambda ga: ga._robust_iqr_stats()),
    ("robust_mad_stats", lambda ga: ga._robust_mad_stats()),
    ("num_diffs", lambda ga: ga._num_diffs(2)),
    ("diff", lambda ga: ga._diff(1)),
    ("tail", lambda ga: ga._tail(3)),
]


@pytest.fixture
def grouped(rng):
    lengths = rng.integers(low=40, high=90, size=50)
    indptr = np.append(0, lengths.cumsum()).astype(np.int32)
    return rng.normal(size=indptr[-1]), indptr


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("num_threads", thread_counts)
@pytest.mark.parametrize("name,op", operations, ids=[o[0] for o in operations])
def test_results_do_not_depend_on_thread_count(name, op, num_threads, grouped, dtype):
    data, indptr = grouped
    data = data.astype(dtype)
    serial = op(GroupedArray(data, indptr, num_threads=1))
    parallel = op(GroupedArray(data, indptr, num_threads=num_threads))
    np.testing.assert_array_equal(serial, parallel)


@pytest.mark.parametrize("num_threads", [-5, 0, 1, 64])
def test_degenerate_thread_counts(num_threads):
    # more threads than groups used to spawn workers with empty ranges
    data = np.arange(6, dtype=np.float64)
    indptr = np.array([0, 3, 6], dtype=np.int32)
    ga = GroupedArray(data, indptr, num_threads=num_threads)
    # min_samples=1, so each group's first element is its own mean
    np.testing.assert_allclose(ga._rolling_mean(0, 2, 1), [0.0, 0.5, 1.5, 3.0, 3.5, 4.5])


@pytest.mark.parametrize("num_threads", [1, 4])
def test_worker_exception_surfaces_as_python_exception(grouped, num_threads):
    # a negative window is rejected by the kernel, so the throw happens inside
    # the worker and used to reach std::terminate
    data, indptr = grouped
    ga = GroupedArray(data, indptr, num_threads=num_threads)
    with pytest.raises(Exception):
        ga._rolling_mean_update(0, -1, -1)
    # the object is still usable afterwards
    assert np.isfinite(ga._rolling_mean(0, 3, 1)).any()


def test_num_threads_is_mutable_at_runtime(grouped):
    data, indptr = grouped
    ga = GroupedArray(data, indptr, num_threads=1)
    expected = ga._rolling_mean(1, 5, 2)
    ga.num_threads = 4
    np.testing.assert_array_equal(ga._rolling_mean(1, 5, 2), expected)


class TestIndptrValidation:
    def test_rejects_values_beyond_the_32_bit_range(self):
        with pytest.raises(ValueError, match="32-bit"):
            GroupedArray(np.zeros(3), np.array([0, 1, 2**31 + 5], dtype=np.int64))

    def test_rejects_a_value_that_would_wrap_onto_the_data_size(self):
        # 2**32 + 3 truncates to exactly 3, so the size check alone would pass it
        with pytest.raises(ValueError, match="32-bit"):
            GroupedArray(np.zeros(3), np.array([0, 1, 2**32 + 3], dtype=np.int64))

    def test_rejects_a_mismatched_last_element(self):
        with pytest.raises(ValueError, match="Last element"):
            GroupedArray(np.zeros(3), np.array([0, 1, 5], dtype=np.int64))

    @pytest.mark.parametrize(
        "indptr",
        [
            [0, 2**31 + 100, 3],  # wraps to a large negative offset
            [0, 2**32, 3],  # wraps to zero
            [0, -1, 3],  # negative outright
        ],
    )
    def test_rejects_out_of_range_intermediate_entries(self, indptr):
        # every entry is an offset into data, so checking only the last one
        # leaves the rest free to wrap during the int32 cast
        with pytest.raises(ValueError, match="non-negative"):
            GroupedArray(np.zeros(3), np.array(indptr, dtype=np.int64))

    @pytest.mark.parametrize(
        "data_size,indptr",
        [
            (1, [0, 3, 1]),  # decreasing, yet last still equals len(data)
            (6, [0, 5, 2, 6]),
        ],
    )
    def test_rejects_non_monotonic_indptr(self, data_size, indptr):
        # a decreasing entry makes a group span past the end of its own data
        with pytest.raises(ValueError, match="non-decreasing"):
            GroupedArray(np.zeros(data_size), np.array(indptr, dtype=np.int64))

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_accepts_valid_integer_indptr(self, dtype):
        ga = GroupedArray(np.zeros(3), np.array([0, 1, 3], dtype=dtype))
        assert len(ga) == 2

    def test_accepts_empty_groups(self):
        # equal consecutive entries are a legitimate empty group
        ga = GroupedArray(np.zeros(6), np.array([0, 3, 3, 6], dtype=np.int64))
        assert len(ga) == 3

    @pytest.mark.parametrize("indptr", [[0, 3, 6], (0, 3, 6)])
    def test_accepts_non_ndarray_sequences(self, indptr):
        # validating in 64 bits must not cost the flexibility of forcecast
        ga = GroupedArray(np.arange(6.0), indptr, 1)
        assert len(ga) == 2
        np.testing.assert_array_equal(np.asarray(ga.indptr), [0, 3, 6])

    @pytest.mark.parametrize(
        "cls", [_GroupedArrayFloat32, _GroupedArrayFloat64], ids=["f32", "f64"]
    )
    def test_typed_constructors_validate_too(self, cls):
        # these are module attributes, so they must not bypass the checks the
        # GroupedArray factory applies
        with pytest.raises(ValueError, match="non-negative"):
            cls(np.zeros(3), np.array([0, 1, 2**31 + 5], dtype=np.int64), 1)
        with pytest.raises(ValueError, match="non-decreasing"):
            cls(np.zeros(6), np.array([0, 5, 2, 6], dtype=np.int64), 1)
        assert len(cls(np.zeros(6), [0, 3, 6], 1)) == 2
