from typing import Optional

import numpy as np

from .grouped_array import GroupedArray


class Identity:
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
            self.tfm_name, self.lag, self.window_size, self.min_samples
        )


class RollingMean(RollingBase):
    tfm_name = "Mean"


class SeasonalRollingBase(RollingBase):
    season_length: int

    def __init__(
        self, lag: int, season_length: int, window_size: int, min_samples: int
    ):
        super().__init__(lag, window_size, min_samples)
        self.season_length = season_length

    def transform(self, ga: GroupedArray) -> np.ndarray:
        return ga.rolling_transform(
            self.tfm_name,
            self.lag,
            self.season_length,
            self.window_size,
            self.min_samples,
        )

    def update(self, ga: GroupedArray) -> np.ndarray:
        return ga.seasonal_rolling_update(
            self.tfm_name,
            self.lag,
            self.season_length,
            self.window_size,
            self.min_samples,
        )


class SeasonalRollingMean(SeasonalRollingBase):
    tfm_name = "Mean"
