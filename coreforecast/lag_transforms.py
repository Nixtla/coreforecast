from typing import Optional

import numpy as np

from .grouped_array import GroupedArray


class RollingBase:
    tfm_name: str
    window_size: int
    min_samples: int

    def __init__(self, window_size: int, min_samples: Optional[int] = None):
        if min_samples is None:
            min_samples = window_size
        if min_samples > window_size:
            min_samples = window_size
        self.window_size = window_size
        self.min_samples = min_samples

    def fit_transform(self, ga: GroupedArray, lag: int) -> np.ndarray:
        return ga.rolling_transform(
            self.tfm_name, lag, self.window_size, self.min_samples
        )

    def update(self, ga: GroupedArray, lag: int) -> np.ndarray:
        return ga.rolling_update(self.tfm_name, lag, self.window_size, self.min_samples)


class RollingMean(RollingBase):
    tfm_name = "RollingMean"


class SeasonalRollingMean(RollingBase):
    tfm_name = "SeasonalRollingMean"
