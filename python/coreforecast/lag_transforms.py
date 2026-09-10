__all__ = [
    "Lag",
    "RollingMean",
    "RollingStd",
    "RollingMin",
    "RollingMax",
    "RollingQuantile",
    "SeasonalRollingMean",
    "SeasonalRollingStd",
    "SeasonalRollingMin",
    "SeasonalRollingMax",
    "SeasonalRollingQuantile",
    "ExpandingMean",
    "ExpandingStd",
    "ExpandingMin",
    "ExpandingMax",
    "ExpandingQuantile",
    "ExponentiallyWeightedMean",
]

import abc
import copy
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from ._lib.grouped_array import _GroupedArrayFloat32, _GroupedArrayFloat64

    GroupedArray = _GroupedArrayFloat32 | _GroupedArrayFloat64


class _BaseLagTransform(abc.ABC):
    lag: int
    stats_: np.ndarray

    @property
    def _update_lag(self) -> int:
        # an update consumes the value that isn't in ga yet, so it reads one
        # position further back than the transform does
        if self.lag < 1:
            raise ValueError(f"lag must be greater than 0 to update, got {self.lag}")
        return self.lag - 1

    @abc.abstractmethod
    def transform(self, ga: "GroupedArray") -> np.ndarray:
        """Apply the transformation by group.

        Args:
            ga (GroupedArray): Array with the grouped data.

        Returns:
            np.ndarray: Array with the transformed data."""
        ...

    @abc.abstractmethod
    def update(self, ga: "GroupedArray") -> np.ndarray:
        """Compute the most recent value of the transformation for each group.

        Args:
            ga (GroupedArray): Array with the grouped data.

        Returns:
            np.ndarray: Array with the updates for each group."""
        ...

    def take(self, _idxs: np.ndarray) -> "_BaseLagTransform":
        return self

    @staticmethod
    def stack(transforms: Sequence["_BaseLagTransform"]) -> "_BaseLagTransform":
        first_tfm = transforms[0]
        if not hasattr(first_tfm, "stats_"):
            # transform doesn't save state, we can return any of them
            return first_tfm
        out = copy.deepcopy(first_tfm)
        if first_tfm.stats_.ndim == 1:
            concat_fn = np.hstack
        else:
            concat_fn = np.vstack
        out.stats_ = concat_fn([tfm.stats_ for tfm in transforms])
        return out


class Lag(_BaseLagTransform):
    """Simple lag operator

    Args:
        lag (int): Number of periods to offset"""

    def __init__(self, lag: int):
        self.lag = lag

    def transform(self, ga: "GroupedArray") -> np.ndarray:
        return ga._lag(self.lag)

    def update(self, ga: "GroupedArray") -> np.ndarray:
        return ga._index_from_end(self._update_lag)


class _RollingBase(_BaseLagTransform):
    stat_name: str
    window_size: int
    min_samples: int
    skipna: bool

    def __init__(
        self,
        lag: int,
        window_size: int,
        min_samples: Optional[int] = None,
        skipna: bool = False,
    ):
        self.lag = lag
        if min_samples is None:
            min_samples = window_size
        if min_samples > window_size:
            min_samples = window_size
        self.window_size = window_size
        self.min_samples = min_samples
        self.skipna = skipna

    def transform(self, ga: "GroupedArray") -> np.ndarray:
        return getattr(ga, f"_rolling_{self.stat_name}")(
            self.lag, self.window_size, self.min_samples, self.skipna
        )

    def update(self, ga: "GroupedArray") -> np.ndarray:
        return getattr(ga, f"_rolling_{self.stat_name}_update")(
            self._update_lag, self.window_size, self.min_samples, self.skipna
        )


class RollingMean(_RollingBase):
    """Rolling Mean

    Args:
        lag (int): Number of periods to offset by before applying the transformation.
        window_size (int): Length of the rolling window.
        min_samples (int, optional): Minimum number of samples required to compute the statistic.
            If None, defaults to window_size.
        skipna (bool): If True, exclude NaN values from calculations.
            When False (default), only a leading run of NaNs is supported; an
            interior one leaves the result unspecified, so pass True for that."""

    stat_name = "mean"


class RollingStd(_RollingBase):
    """Rolling Standard Deviation

    Args:
        lag (int): Number of periods to offset by before applying the transformation.
        window_size (int): Length of the rolling window.
        min_samples (int, optional): Minimum number of samples required to compute the statistic.
            If None, defaults to window_size.
        skipna (bool): If True, exclude NaN values from calculations.
            When False (default), only a leading run of NaNs is supported; an
            interior one leaves the result unspecified, so pass True for that."""

    stat_name = "std"


class RollingMin(_RollingBase):
    """Rolling Minimum

    Args:
        lag (int): Number of periods to offset by before applying the transformation.
        window_size (int): Length of the rolling window.
        min_samples (int, optional): Minimum number of samples required to compute the statistic.
            If None, defaults to window_size.
        skipna (bool): If True, exclude NaN values from calculations.
            When False (default), only a leading run of NaNs is supported; an
            interior one leaves the result unspecified, so pass True for that."""

    stat_name = "min"


class RollingMax(_RollingBase):
    """Rolling Maximum

    Args:
        lag (int): Number of periods to offset by before applying the transformation.
        window_size (int): Length of the rolling window.
        min_samples (int, optional): Minimum number of samples required to compute the statistic.
            If None, defaults to window_size.
        skipna (bool): If True, exclude NaN values from calculations.
            When False (default), only a leading run of NaNs is supported; an
            interior one leaves the result unspecified, so pass True for that."""

    stat_name = "max"


class RollingQuantile(_RollingBase):
    """Rolling quantile

    Args:
        lag (int): Number of periods to offset by before applying the transformation
        p (float): Quantile to compute
        window_size (int): Length of the rolling window
        min_samples (int, optional): Minimum number of samples required to compute the statistic.
            If None, defaults to window_size.
        skipna (bool): If True, exclude NaN values from calculations.
            When False (default), only a leading run of NaNs is supported; an
            interior one leaves the result unspecified, so pass True for that."""

    def __init__(
        self,
        lag: int,
        p: float,
        window_size: int,
        min_samples: Optional[int] = None,
        skipna: bool = False,
    ):
        super().__init__(
            lag=lag, window_size=window_size, min_samples=min_samples, skipna=skipna
        )
        self.p = p

    def transform(self, ga: "GroupedArray") -> np.ndarray:
        return ga._rolling_quantile(
            self.lag, self.p, self.window_size, self.min_samples, self.skipna
        )

    def update(self, ga: "GroupedArray") -> np.ndarray:
        return ga._rolling_quantile_update(
            self._update_lag, self.p, self.window_size, self.min_samples, self.skipna
        )


class _SeasonalRollingBase(_RollingBase):
    season_length: int

    def __init__(
        self,
        lag: int,
        season_length: int,
        window_size: int,
        min_samples: Optional[int] = None,
        skipna: bool = False,
    ):
        super().__init__(lag, window_size, min_samples, skipna)
        self.season_length = season_length

    def transform(self, ga: "GroupedArray") -> np.ndarray:
        return getattr(ga, f"_seasonal_rolling_{self.stat_name}")(
            self.lag,
            self.season_length,
            self.window_size,
            self.min_samples,
            self.skipna,
        )

    def update(self, ga: "GroupedArray") -> np.ndarray:
        return getattr(ga, f"_seasonal_rolling_{self.stat_name}_update")(
            self._update_lag,
            self.season_length,
            self.window_size,
            self.min_samples,
            self.skipna,
        )


class SeasonalRollingMean(_SeasonalRollingBase):
    """Seasonal rolling Mean

    Args:
        lag (int): Number of periods to offset by before applying the transformation
        season_length (int): Length of the seasonal period, e.g. 7 for weekly data
        window_size (int): Length of the rolling window
        min_samples (int, optional): Minimum number of samples required to compute the statistic.
            If None, defaults to window_size.
        skipna (bool): If True, exclude NaN values from calculations.
            When False (default), only a leading run of NaNs is supported; an
            interior one leaves the result unspecified, so pass True for that."""

    stat_name = "mean"


class SeasonalRollingStd(_SeasonalRollingBase):
    """Seasonal rolling Standard Deviation

    Args:
        lag (int): Number of periods to offset by before applying the transformation
        season_length (int): Length of the seasonal period, e.g. 7 for weekly data
        window_size (int): Length of the rolling window
        min_samples (int, optional): Minimum number of samples required to compute the statistic.
            If None, defaults to window_size.
        skipna (bool): If True, exclude NaN values from calculations.
            When False (default), only a leading run of NaNs is supported; an
            interior one leaves the result unspecified, so pass True for that."""

    stat_name = "std"


class SeasonalRollingMin(_SeasonalRollingBase):
    """Seasonal rolling Minimum

    Args:
        lag (int): Number of periods to offset by before applying the transformation
        season_length (int): Length of the seasonal period, e.g. 7 for weekly data
        window_size (int): Length of the rolling window
        min_samples (int, optional): Minimum number of samples required to compute the statistic.
            If None, defaults to window_size.
        skipna (bool): If True, exclude NaN values from calculations.
            When False (default), only a leading run of NaNs is supported; an
            interior one leaves the result unspecified, so pass True for that."""

    stat_name = "min"


class SeasonalRollingMax(_SeasonalRollingBase):
    """Seasonal rolling Maximum

    Args:
        lag (int): Number of periods to offset by before applying the transformation
        season_length (int): Length of the seasonal period, e.g. 7 for weekly data
        window_size (int): Length of the rolling window
        min_samples (int, optional): Minimum number of samples required to compute the statistic.
            If None, defaults to window_size.
        skipna (bool): If True, exclude NaN values from calculations.
            When False (default), only a leading run of NaNs is supported; an
            interior one leaves the result unspecified, so pass True for that."""

    stat_name = "max"


class SeasonalRollingQuantile(_SeasonalRollingBase):
    """Seasonal rolling statistic

    Args:
        lag (int): Number of periods to offset by before applying the transformation
        p (float): Quantile to compute
        season_length (int): Length of the seasonal period, e.g. 7 for weekly data
        window_size (int): Length of the rolling window
        min_samples (int, optional): Minimum number of samples required to compute the statistic.
            If None, defaults to window_size.
        skipna (bool): If True, exclude NaN values from calculations.
            When False (default), only a leading run of NaNs is supported; an
            interior one leaves the result unspecified, so pass True for that."""

    def __init__(
        self,
        lag: int,
        p: float,
        season_length: int,
        window_size: int,
        min_samples: Optional[int] = None,
        skipna: bool = False,
    ):
        super().__init__(
            lag=lag,
            season_length=season_length,
            window_size=window_size,
            min_samples=min_samples,
            skipna=skipna,
        )
        self.p = p

    def transform(self, ga: "GroupedArray") -> np.ndarray:
        return ga._seasonal_rolling_quantile(
            self.lag,
            self.p,
            self.season_length,
            self.window_size,
            self.min_samples,
            self.skipna,
        )

    def update(self, ga: "GroupedArray") -> np.ndarray:
        return ga._seasonal_rolling_quantile_update(
            self._update_lag,
            self.p,
            self.season_length,
            self.window_size,
            self.min_samples,
            self.skipna,
        )


class _ExpandingBase(_BaseLagTransform):
    stats_: np.ndarray
    skipna: bool

    def __init__(self, lag: int, skipna: bool = False):
        self.lag = lag
        self.skipna = skipna

    def _skip(self, x: np.ndarray) -> np.ndarray:
        """Which incoming values must not enter the accumulator.

        For the accumulators that keep their count of values in the first
        column of stats_; the running comparisons and the EWM read emptiness
        off their NaN state instead."""
        skip = np.isnan(x)
        if not self.skipna:
            # the accumulator still has to walk through the leading run of
            # NaNs that the transform strips, so a NaN only propagates once a
            # real value has been seen
            skip &= self.stats_[:, 0] == 0.0
        return skip

    def take(self, idxs: np.ndarray) -> "_ExpandingBase":
        out = self.__class__(self.lag, self.skipna)
        out.stats_ = self.stats_[idxs].copy()
        return out


class ExpandingMean(_ExpandingBase):
    """Expanding Mean

    Args:
        lag (int): Number of periods to offset by before applying the transformation
        skipna (bool): If True, exclude NaN values from calculations.
            When False (default), only a leading run of NaNs is supported; an
            interior one leaves the result unspecified, so pass True for that."""

    def transform(self, ga: "GroupedArray") -> np.ndarray:
        out, n = ga._expanding_mean(self.lag, self.skipna)
        cumsum = n * out[ga.indptr[1:] - 1]
        # the driver fills the whole row of a group it skipped, which means it
        # saw no value rather than a poisoned accumulator, so an update has to
        # be able to seed from it. A NaN count is the only way to tell the two
        # apart: an interior NaN without skipna leaves the count finite.
        empty = np.isnan(n)
        n = np.where(empty, 0.0, n)
        cumsum = np.where(empty, 0.0, cumsum)
        self.stats_ = np.hstack([n[:, None], cumsum[:, None]])
        return out

    def update(self, ga: "GroupedArray") -> np.ndarray:
        x = ga._index_from_end(self._update_lag)
        skip = self._skip(x)
        self.stats_[:, 0] += ~skip
        self.stats_[:, 1] += np.where(skip, 0.0, x)
        n = self.stats_[:, 0]
        return np.divide(
            self.stats_[:, 1], n, out=np.full_like(n, np.nan), where=n > 0.0
        )


class ExpandingStd(_ExpandingBase):
    """Expanding Standard Deviation

    Args:
        lag (int): Number of periods to offset by before applying the transformation
        skipna (bool): If True, exclude NaN values from calculations.
            When False (default), only a leading run of NaNs is supported; an
            interior one leaves the result unspecified, so pass True for that."""

    def transform(self, ga: "GroupedArray") -> np.ndarray:
        out, self.stats_ = ga._expanding_std(self.lag, self.skipna)
        # see ExpandingMean.transform: a skipped group's row is "nothing seen
        # yet", which is a zeroed Welford state. Only a whole NaN row is one;
        # an interior NaN without skipna leaves the count finite and has to
        # keep propagating.
        self.stats_[np.isnan(self.stats_).all(axis=1)] = 0.0
        return out

    def update(self, ga: "GroupedArray") -> np.ndarray:
        x = ga._index_from_end(self._update_lag)
        prev_n = self.stats_[:, 0].copy()
        prev_avg = self.stats_[:, 1].copy()
        prev_m2 = self.stats_[:, 2].copy()
        skip = self._skip(x)
        n = prev_n + 1.0
        curr_avg = prev_avg + (x - prev_avg) / n
        # avoid possible loss of precision
        m2 = np.maximum(prev_m2 + (x - prev_avg) * (x - curr_avg), 0.0)
        # a skipped value leaves the count, the mean and M2 untouched
        self.stats_[:, 0] = np.where(skip, prev_n, n)
        self.stats_[:, 1] = np.where(skip, prev_avg, curr_avg)
        self.stats_[:, 2] = np.where(skip, prev_m2, m2)
        n = self.stats_[:, 0]
        # the kernel needs two values before it reports a deviation
        var = np.divide(
            self.stats_[:, 2], n - 1.0, out=np.full_like(n, np.nan), where=n > 1.0
        )
        return np.sqrt(var)


class _ExpandingComp(_ExpandingBase):
    stat: str
    _comp_fn: Callable
    _skipna_comp_fn: Callable

    def transform(self, ga: "GroupedArray") -> np.ndarray:
        out = getattr(ga, f"_expanding_{self.stat}")(self.lag, self.skipna)
        self.stats_ = out[ga.indptr[1:] - 1]
        return out

    def update(self, ga: "GroupedArray") -> np.ndarray:
        x = ga._index_from_end(self._update_lag)
        comp_fn = self._skipna_comp_fn if self.skipna else self._comp_fn
        # a NaN state is a group the driver skipped, so the first value it gets
        # seeds the statistic instead of being compared against a NaN that
        # np.minimum/np.maximum would keep forever. Without skipna a group with
        # nothing after its leading run of NaNs is the only defined way to get
        # here, and that is exactly the case that has to seed.
        self.stats_ = np.where(np.isnan(self.stats_), x, comp_fn(self.stats_, x))
        return self.stats_


class ExpandingMin(_ExpandingComp):
    """Expanding Minimum

    Args:
        lag (int): Number of periods to offset by before applying the transformation
        skipna (bool): If True, exclude NaN values from calculations.
            When False (default), only a leading run of NaNs is supported; an
            interior one leaves the result unspecified, so pass True for that."""

    stat = "min"
    _comp_fn = np.minimum
    _skipna_comp_fn = np.fmin


class ExpandingMax(_ExpandingComp):
    """Expanding Maximum

    Args:
        lag (int): Number of periods to offset by before applying the transformation
        skipna (bool): If True, exclude NaN values from calculations.
            When False (default), only a leading run of NaNs is supported; an
            interior one leaves the result unspecified, so pass True for that."""

    stat = "max"
    _comp_fn = np.maximum
    _skipna_comp_fn = np.fmax


class ExpandingQuantile(_BaseLagTransform):
    """Expanding quantile

    Args:
        lag (int):
            Number of periods to offset by before applying the transformation
        p (float):
            Quantile to compute
        skipna (bool): If True, exclude NaN values from calculations.
            When False (default), only a leading run of NaNs is supported; an
            interior one leaves the result unspecified, so pass True for that."""

    def __init__(self, lag: int, p: float, skipna: bool = False):
        self.lag = lag
        self.p = p
        self.skipna = skipna

    def transform(self, ga: "GroupedArray") -> np.ndarray:
        return ga._expanding_quantile(self.lag, self.p, self.skipna)

    def update(self, ga: "GroupedArray") -> np.ndarray:
        return ga._expanding_quantile_update(self._update_lag, self.p, self.skipna)


class ExponentiallyWeightedMean(_BaseLagTransform):
    """Exponentially weighted mean

    Args:
        lag (int): Number of periods to offset by before applying the transformation
        alpha (float): Smoothing factor
        skipna (bool): If True, exclude NaN values from calculations using forward-fill behavior.
            When False (default), only a leading run of NaNs is supported; an
            interior one leaves the result unspecified, so pass True for that."""

    def __init__(self, lag: int, alpha: float, skipna: bool = False):
        self.lag = lag
        self.alpha = alpha
        self.skipna = skipna

    def transform(self, ga: "GroupedArray") -> np.ndarray:
        out = ga._exponentially_weighted_mean(self.lag, self.alpha, self.skipna)
        self.stats_ = out[ga.indptr[1:] - 1]
        return out

    def update(self, ga: "GroupedArray") -> np.ndarray:
        x = ga._index_from_end(self._update_lag)
        ewm = self.alpha * x + (1 - self.alpha) * self.stats_
        # a NaN state is a group the driver skipped, so the first value it gets
        # seeds the mean, which is what the kernel does with the first value
        # after a leading run of NaNs
        ewm = np.where(np.isnan(self.stats_), x, ewm)
        if self.skipna:
            # a NaN forward-fills the previous mean
            ewm = np.where(np.isnan(x), self.stats_, ewm)
        self.stats_ = ewm
        return self.stats_

    def take(self, idxs: np.ndarray) -> "ExponentiallyWeightedMean":
        out = copy.deepcopy(self)
        out.stats_ = out.stats_[idxs].copy()
        return out
