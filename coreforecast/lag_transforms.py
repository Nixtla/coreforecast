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
from typing import Callable, Optional

import numpy as np

from .grouped_array import GroupedArray


class _BaseLagTransform(abc.ABC):
    @abc.abstractmethod
    def transform(self, ga: GroupedArray) -> np.ndarray:
        """Apply the transformation by group.

        Args:
            ga (GroupedArray): Array with the grouped data.

        Returns:
            np.ndarray: Array with the transformed data."""
        ...

    @abc.abstractmethod
    def update(self, ga: GroupedArray) -> np.ndarray:
        """Compute the most recent value of the transformation for each group.

        Args:
            ga (GroupedArray): Array with the grouped data.

        Returns:
            np.ndarray: Array with the updates for each group."""
        ...


class Lag(_BaseLagTransform):
    """Simple lag operator

    Args:
        lag (int): Number of periods to offset"""

    def __init__(self, lag: int):
        self.lag = lag

    def transform(self, ga: GroupedArray) -> np.ndarray:
        return ga._lag_transform(self.lag)

    def update(self, ga: GroupedArray) -> np.ndarray:
        return ga._index_from_end(self.lag - 1)


_rolling_base_docstring = """Rolling {stat_name}

    Args:
        lag (int): Number of periods to offset by before applying the transformation.
        window_size (int): Length of the rolling window.
        min_samples (int, optional): Minimum number of samples required to compute the statistic.
            If None, defaults to window_size."""


class _RollingBase(_BaseLagTransform):
    stat_name: str
    lag: int
    window_size: int
    min_samples: int

    def __init__(self, lag: int, window_size: int, min_samples: Optional[int] = None):
        self.lag = lag
        if min_samples is None:
            min_samples = window_size
        if min_samples > window_size:
            min_samples = window_size
        self.window_size = window_size
        self.min_samples = min_samples

    def transform(self, ga: GroupedArray) -> np.ndarray:
        return ga._rolling_transform(
            self.stat_name, self.lag, self.window_size, self.min_samples
        )

    def update(self, ga: GroupedArray) -> np.ndarray:
        return ga._rolling_update(
            self.stat_name, self.lag - 1, self.window_size, self.min_samples
        )


class RollingMean(_RollingBase):
    stat_name = "Mean"


RollingMean.__doc__ = _rolling_base_docstring.format(stat_name="Mean")


class RollingStd(_RollingBase):
    stat_name = "Std"


RollingStd.__doc__ = _rolling_base_docstring.format(stat_name="Standard Deviation")


class RollingMin(_RollingBase):
    stat_name = "Min"


RollingMin.__doc__ = _rolling_base_docstring.format(stat_name="Minimum")


class RollingMax(_RollingBase):
    stat_name = "Max"


RollingMax.__doc__ = _rolling_base_docstring.format(stat_name="Maximum")


class RollingQuantile(_RollingBase):
    """Rolling quantile

    Args:
        lag (int): Number of periods to offset by before applying the transformation
        p (float): Quantile to compute
        window_size (int): Length of the rolling window
        min_samples (int, optional): Minimum number of samples required to compute the statistic.
            If None, defaults to window_size."""

    def __init__(
        self, lag: int, p: float, window_size: int, min_samples: Optional[int] = None
    ):
        super().__init__(lag=lag, window_size=window_size, min_samples=min_samples)
        self.p = p

    def transform(self, ga: GroupedArray) -> np.ndarray:
        return ga._rolling_quantile_transform(
            self.lag, self.p, self.window_size, self.min_samples
        )

    def update(self, ga: GroupedArray) -> np.ndarray:
        return ga._rolling_quantile_update(
            self.lag - 1, self.p, self.window_size, self.min_samples
        )


_seasonal_rolling_docstring = """Seasonal rolling {stat_name}

    Args:
        lag (int): Number of periods to offset by before applying the transformation
        season_length (int): Length of the seasonal period, e.g. 7 for weekly data
        window_size (int): Length of the rolling window
        min_samples (int, optional): Minimum number of samples required to compute the statistic.
            If None, defaults to window_size."""


class _SeasonalRollingBase(_RollingBase):
    season_length: int

    def __init__(
        self,
        lag: int,
        season_length: int,
        window_size: int,
        min_samples: Optional[int] = None,
    ):
        super().__init__(lag, window_size, min_samples)
        self.season_length = season_length

    def transform(self, ga: GroupedArray) -> np.ndarray:
        return ga._seasonal_rolling_transform(
            self.stat_name,
            self.lag,
            self.season_length,
            self.window_size,
            self.min_samples,
        )

    def update(self, ga: GroupedArray) -> np.ndarray:
        return ga._seasonal_rolling_update(
            self.stat_name,
            self.lag - 1,
            self.season_length,
            self.window_size,
            self.min_samples,
        )


class SeasonalRollingMean(_SeasonalRollingBase):
    stat_name = "Mean"


SeasonalRollingMean.__doc__ = _seasonal_rolling_docstring.format(stat_name="Mean")


class SeasonalRollingStd(_SeasonalRollingBase):
    stat_name = "Std"


SeasonalRollingStd.__doc__ = _seasonal_rolling_docstring.format(
    stat_name="Standard Deviation"
)


class SeasonalRollingMin(_SeasonalRollingBase):
    stat_name = "Min"


SeasonalRollingMin.__doc__ = _seasonal_rolling_docstring.format(stat_name="Minimum")


class SeasonalRollingMax(_SeasonalRollingBase):
    stat_name = "Max"


SeasonalRollingMax.__doc__ = _seasonal_rolling_docstring.format(stat_name="Maximum")


class SeasonalRollingQuantile(_SeasonalRollingBase):
    """Seasonal rolling statistic

    Args:
        lag (int): Number of periods to offset by before applying the transformation
        p (float): Quantile to compute
        season_length (int): Length of the seasonal period, e.g. 7 for weekly data
        window_size (int): Length of the rolling window
        min_samples (int, optional): Minimum number of samples required to compute the statistic.
            If None, defaults to window_size."""

    def __init__(
        self,
        lag: int,
        p: float,
        season_length: int,
        window_size: int,
        min_samples: Optional[int] = None,
    ):
        super().__init__(
            lag=lag,
            season_length=season_length,
            window_size=window_size,
            min_samples=min_samples,
        )
        self.p = p

    def transform(self, ga: GroupedArray) -> np.ndarray:
        return ga._seasonal_rolling_quantile_transform(
            lag=self.lag,
            p=self.p,
            season_length=self.season_length,
            window_size=self.window_size,
            min_samples=self.min_samples,
        )

    def update(self, ga: GroupedArray) -> np.ndarray:
        return ga._seasonal_rolling_quantile_update(
            lag=self.lag - 1,
            p=self.p,
            season_length=self.season_length,
            window_size=self.window_size,
            min_samples=self.min_samples,
        )


_expanding_docstring = """Expanding {stat_name}

    Args:
        lag (int): Number of periods to offset by before applying the transformation"""


class _ExpandingBase(_BaseLagTransform):
    def __init__(self, lag: int):
        self.lag = lag


class ExpandingMean(_ExpandingBase):
    def transform(self, ga: GroupedArray) -> np.ndarray:
        self.n = np.empty_like(ga.data, shape=len(ga))
        out = ga._expanding_transform_with_aggs("Mean", self.lag, self.n)
        self.cumsum = out[ga.indptr[1:] - 1] * self.n
        return out

    def update(self, ga: GroupedArray) -> np.ndarray:
        self.n += 1
        self.cumsum += ga._index_from_end(self.lag - 1)
        return self.cumsum / self.n


ExpandingMean.__doc__ = _expanding_docstring.format(stat_name="Mean")


class ExpandingStd(_ExpandingBase):
    def transform(self, ga: GroupedArray) -> np.ndarray:
        self.stats = np.empty_like(ga.data, shape=(len(ga), 3))
        out = ga._expanding_transform_with_aggs("Std", self.lag, self.stats)
        return out

    def update(self, ga: GroupedArray) -> np.ndarray:
        x = ga._index_from_end(self.lag - 1)
        self.stats[:, 0] += 1.0
        n = self.stats[:, 0]
        prev_avg = self.stats[:, 1].copy()
        self.stats[:, 1] = prev_avg + (x - prev_avg) / n
        self.stats[:, 2] += (x - prev_avg) * (x - self.stats[:, 1])
        self.stats[:, 2] = np.maximum(self.stats[:, 2], 0.0)
        return np.sqrt(self.stats[:, 2] / (n - 1))


ExpandingStd.__doc__ = _expanding_docstring.format(stat_name="Standard Deviation")


class _ExpandingComp(_ExpandingBase):
    stat: str
    _comp_fn: Callable

    def transform(self, ga: GroupedArray) -> np.ndarray:
        out = ga._expanding_transform(self.stat, self.lag)
        self.stats = out[ga.indptr[1:] - 1]
        return out

    def update(self, ga: GroupedArray) -> np.ndarray:
        self.stats = self._comp_fn(self.stats, ga._index_from_end(self.lag - 1))
        return self.stats


class ExpandingMin(_ExpandingComp):
    stat = "Min"
    _comp_fn = np.minimum


ExpandingMin.__doc__ = _expanding_docstring.format(stat_name="Minimum")


class ExpandingMax(_ExpandingComp):
    stat = "Max"
    _comp_fn = np.maximum


ExpandingMax.__doc__ = _expanding_docstring.format(stat_name="Maximum")


class ExpandingQuantile(_BaseLagTransform):
    """Expanding quantile

    Args:
        lag (int):
            Number of periods to offset by before applying the transformation
        p (float):
            Quantile to compute"""

    def __init__(self, lag: int, p: float):
        self.lag = lag
        self.p = p

    def transform(self, ga: GroupedArray) -> np.ndarray:
        return ga._expanding_quantile_transform(self.lag, self.p)

    def update(self, ga: GroupedArray) -> np.ndarray:
        return ga._expanding_quantile_update(self.lag - 1, self.p)


class ExponentiallyWeightedMean(_BaseLagTransform):
    """Exponentially weighted mean

    Args:
        lag (int): Number of periods to offset by before applying the transformation
        alpha (float): Smoothing factor"""

    def __init__(self, lag: int, alpha: float):
        self.lag = lag
        self.alpha = alpha

    def transform(self, ga: GroupedArray) -> np.ndarray:
        out = ga._exponentially_weighted_transform("Mean", self.lag, self.alpha)
        self.ewm = out[ga.indptr[1:] - 1]
        return out

    def update(self, ga: GroupedArray) -> np.ndarray:
        x = ga._index_from_end(self.lag - 1)
        self.ewm = self.alpha * x + (1 - self.alpha) * self.ewm
        return self.ewm
