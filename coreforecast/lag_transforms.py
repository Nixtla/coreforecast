from typing import Optional

import numpy as np

from .grouped_array import GroupedArray


class Lag:
    def __init__(self, lag: int):
        self.lag = lag

    def transform(self, ga: GroupedArray) -> np.ndarray:
        return ga.lag_transform(self.lag)

    def update(self, ga: GroupedArray) -> np.ndarray:
        return ga.data[ga.indptr[1:] - self.lag]


class RollingBase:
    tfm_name: str
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
        return ga.rolling_transform(
            self.tfm_name, self.lag, self.window_size, self.min_samples
        )

    def update(self, ga: GroupedArray) -> np.ndarray:
        return ga.rolling_update(
            self.tfm_name, self.lag - 1, self.window_size, self.min_samples
        )


class RollingMean(RollingBase):
    tfm_name = "Mean"


class RollingStd(RollingBase):
    tfm_name = "Std"


class RollingMin(RollingBase):
    tfm_name = "Min"


class RollingMax(RollingBase):
    tfm_name = "Max"


class SeasonalRollingBase(RollingBase):
    season_length: int

    def __init__(
        self, lag: int, season_length: int, window_size: int, min_samples: int
    ):
        super().__init__(lag, window_size, min_samples)
        self.season_length = season_length

    def transform(self, ga: GroupedArray) -> np.ndarray:
        return ga.seasonal_rolling_transform(
            self.tfm_name,
            self.lag,
            self.season_length,
            self.window_size,
            self.min_samples,
        )

    def update(self, ga: GroupedArray) -> np.ndarray:
        return ga.seasonal_rolling_update(
            self.tfm_name,
            self.lag - 1,
            self.season_length,
            self.window_size,
            self.min_samples,
        )


class SeasonalRollingMean(SeasonalRollingBase):
    tfm_name = "Mean"


class SeasonalRollingStd(SeasonalRollingBase):
    tfm_name = "Std"


class SeasonalRollingMin(SeasonalRollingBase):
    tfm_name = "Min"


class SeasonalRollingMax(SeasonalRollingBase):
    tfm_name = "Max"


class ExpandingBase:
    def __init__(self, lag: int):
        self.lag = lag


class ExpandingMean(ExpandingBase):
    def transform(self, ga: GroupedArray) -> np.ndarray:
        self.n = np.empty_like(ga.data, shape=len(ga))
        out = ga.expanding_transform_with_aggs("Mean", self.lag, self.n)
        self.cumsum = out[ga.indptr[1:] - 1] * self.n
        return out

    def update(self, ga: GroupedArray) -> np.ndarray:
        self.n += 1
        self.cumsum += ga.data[ga.indptr[1:] - self.lag]
        return self.cumsum / self.n


class ExpandingStd(ExpandingBase):
    def transform(self, ga: GroupedArray) -> np.ndarray:
        self.stats = np.empty_like(ga.data, shape=(len(ga), 3))
        out = ga.expanding_transform_with_aggs("Std", self.lag, self.stats)
        return out

    def update(self, ga: GroupedArray) -> np.ndarray:
        x = ga.data[ga.indptr[1:] - self.lag]
        self.stats[:, 0] += 1
        n = self.stats[:, 0]
        prev_avg = self.stats[:, 1]
        self.stats[:, 1] = prev_avg + (x - prev_avg) / n
        self.stats[:, 2] += (x - prev_avg) * (x - self.stats[:, 1])
        return np.sqrt(self.stats[:, 2] / (n - 1))


class ExpandingMin(ExpandingBase):
    def transform(self, ga: GroupedArray) -> np.ndarray:
        out = ga.expanding_transform("Min", self.lag)
        self.mins = out[ga.indptr[1:] - 1]
        return out

    def update(self, ga: GroupedArray) -> np.ndarray:
        self.mins = np.minimum(self.mins, ga.data[ga.indptr[1:] - self.lag])
        return self.mins


class ExpandingMax(ExpandingBase):
    def transform(self, ga: GroupedArray) -> np.ndarray:
        out = ga.expanding_transform("Max", self.lag)
        self.maxs = out[ga.indptr[1:] - 1]
        return out

    def update(self, ga: GroupedArray) -> np.ndarray:
        self.maxs = np.maximum(self.maxs, ga.data[ga.indptr[1:] - self.lag])
        return self.maxs
