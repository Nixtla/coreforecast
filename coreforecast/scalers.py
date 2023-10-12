import numpy as np

from .grouped_array import GroupedArray


class BaseLocalScaler:
    stats_fn_name: str

    def fit(self, ga: GroupedArray) -> "BaseLocalScaler":
        self.stats_ = ga.scaler_fit(self.stats_fn_name)
        return self

    def transform(self, ga: GroupedArray) -> np.ndarray:
        return ga.scaler_transform(self.stats_)

    def fit_transform(self, ga: GroupedArray) -> np.ndarray:
        return self.fit(ga).transform(ga)

    def inverse_transform(self, ga: GroupedArray) -> np.ndarray:
        return ga.scaler_inverse_transform(self.stats_)


class LocalMinMaxScaler(BaseLocalScaler):
    stats_fn_name = "GroupedArray_MinMaxScalerStats"


class LocalStandardScaler(BaseLocalScaler):
    stats_fn_name = "GroupedArray_StandardScalerStats"


class LocalRobustScaler(BaseLocalScaler):
    def __init__(self, scale: str):
        if scale == "iqr":
            self.stats_fn_name = "GroupedArray_RobustScalerIqrStats"
        else:
            self.stats_fn_name = "GroupedArray_RobustScalerMadStats"
