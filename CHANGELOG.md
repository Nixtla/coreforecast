# Changelog

## Unreleased

### Breaking changes

- The free functions in `coreforecast.rolling`, `coreforecast.expanding` and
  `coreforecast.exponentially_weighted` now preserve a leading run of NaNs and
  compute the statistic over what follows it. They used to return all NaN as
  soon as the series started with one. With `skipna=True`, `min_samples` counts
  positions after that run, which is what pandas does. The seasonal variants
  split what follows the run into seasons, so a leading run whose length isn't a
  multiple of `season_length` now groups different positions together than it
  used to.
- Input without a float dtype (integer arrays, lists, Series) is now widened to
  float64 instead of float32. Values from `2**24` on were silently rounded
  before; the result now takes twice the memory. Pass a float32 array to get
  float32 back, which was already the only way to be sure of it.
- Arguments that used to read or write out of bounds are rejected with a
  `ValueError` or an `IndexError` instead: non-positive `window_size`,
  `min_samples` and `season_length`, negative `lag`, `d` and `k`, quantile
  levels outside `[0, 1]`, `indptr` entries that are negative or decreasing,
  out-of-range group indices, and `ds`, `stats`,
  `periods` and `tails` arrays whose size doesn't match the group count.
  Calling `update` on a lag transform built with `lag=0` also raises now: an
  update reads the value that isn't in the array yet, so there is nothing for
  it to consume. A negative `max_season_length` is a `ValueError` like the
  other counts rather than a `TypeError` from the binding.
- `GroupedArray.indptr` is stored and returned as `int64` instead of `int32`,
  so arrays are no longer limited to `2**31` elements. Any integer dtype is
  still accepted as input; the values that used to be rejected as "not
  representable with 32-bit integers" are now checked like every other offset.
  It is signed rather than unsigned so that arithmetic mixing it with numpy's
  default integer stays integral instead of promoting to float64.

### Bug fixes

- `LocalBoxCoxScaler.stats_` had an uninitialised second column: the lambda
  kernels write one value per group and the array holding them was never
  cleared. Transforms were unaffected since that column is not read, but the
  attribute differed between identical fits. It is now deterministic padding:
  zero for a group that produced a lambda, NaN for an empty or all-NaN group,
  whose whole row the driver fills.

### Performance

- Threads take groups off a shared counter in chunks as they finish instead of
  being handed an equal share of the group count up front, and the chunks go
  out heaviest first so the longest groups are started while there is still
  other work to run alongside them. Uneven group lengths used to leave threads
  idle: on four threads here, 1000 series ordered longest first went from 3.0x
  to 3.5x, and a panel holding one series with half the elements from 1.5x to
  1.9x wherever in the panel that series sits. Evenly sized groups are
  unchanged, and so are the results: every group writes its own slice of the
  output whichever thread runs it. A single group is never split across
  threads, so one holding half the elements caps the speedup at 2x however
  many threads are used.

### Build

- The Eigen submodule is gone; the handful of reductions it backed are plain
  loops now. Source builds no longer need to fetch it. The box-cox lambda
  estimators run about twice as fast because their transform is evaluated once
  instead of once per statistic; results may differ from previous versions in
  the last bits because the summation order changed.
- The kernels have their own tests under `tests/cpp`, built with
  `-DCOREFORECAST_BUILD_TESTS=ON` against the `external_libs/doctest`
  submodule; existing checkouts need `git submodule update --init` for it.
  They are not part of the sdist or the wheels.
- The private `_lib.rolling.rolling_quantile` and `seasonal_rolling_quantile`
  take `p` right after `data`, the order the public functions already used.

### Documentation

- The local scalers' `skipna` documentation said an interior NaN "may result in
  NaN statistics". It doesn't for min-max, where the result is whichever value
  wins an unordered comparison. With `skipna=False` only a leading run of NaNs
  is supported; pass `skipna=True` for anything else.
