# cython: language_level=3
# distutils: language=c++
cimport numpy as np


cdef class GroupedArrayFloat32:
    cdef GroupedArray[float] *_cpp_obj

    def __cinit__(
        self,
        float[::1] data,
        indptr_t[::1] indptr,
        int num_threads,
    ):
        self._cpp_obj = new GroupedArray[float](
            &data[0],
            &indptr[0],
            indptr.size,
            num_threads,
        )

    def __dealloc__(self):
        if self._cpp_obj != NULL:
            del self._cpp_obj

    def _append(self, GroupedArrayFloat32& other, indptr_t[::1] out_indptr, float[::1] out_data):
        self._cpp_obj.Append(other._cpp_obj, out_indptr, out_data)

    def _standard_stats(self, float[:, ::1] out):
        self._cpp_obj.StandardScalerStats(&out.data)

    def _minmax_stats(self, float[:, ::1] out):
        self._cpp_obj.MinMaxScalerStats(&out.data)

    def _robustiqr_stats(self, float[:, ::1] out):
        self._cpp_obj.RobustIqrScalerStats(&out[0])

    def _robustmad_stats(self, float[:, ::1] out):
        self._cpp_obj.RobustMadScalerStats(&out[0])

    def _scaler_transform(self, float[:, ::1] stats, float[::1] out):
        self._cpp_obj.ApplyScaler(&stats[0], &out[0])

    def _scaler_inverse_transform(self, float[:, ::1] stats, float[::1] out):
        self._cpp_obj.InvertScaler(&stats[0], &out[0])

    def _index_from_end(self, int k, float[::1] out):
        self._cpp_obj.IndexFromEnd(k, &out[0])

    def _head(self, int k, float[::1] out):
        self._cpp_obj.Head(k, &out[0])

    def _tail(self, int k, float[::1] out):
        self._cpp_obj.Tail(k, &out[0])

    def _tails(self, indptr_t[::1] out_indptr, float[::1] out_data):
        self._cpp_obj.Tails(out_indptr, out_data)

    def _lag_transform(self, int k, float[::1] out):
        self._cpp_obj.LagTransform(k, &out[0])

    def _rolling_mean(self, int lag, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.RollingMean(lag, window_size, min_samples, &out[0])

    def _rolling_std(self, int lag, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.RollingStd(lag, window_size, min_samples, &out[0])

    def _rolling_min(self, int lag, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.RollingMin(lag, window_size, min_samples, &out[0])

    def _rolling_max(self, int lag, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.RollingMax(lag, window_size, min_samples, &out[0])

    def _rolling_quantile(self, int lag, float p, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.RollingQuantile(lag, p, window_size, min_samples, &out[0])

    def _rolling_mean_update(self, int lag, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.RollingMeanUpdate(lag, window_size, min_samples, &out[0])

    def _rolling_std_update(self, int lag, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.RollingStdUpdate(lag, window_size, min_samples, &out[0])

    def _rolling_min_update(self, int lag, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.RollingMinUpdate(lag, window_size, min_samples, &out[0])

    def _rolling_max_update(self, int lag, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.RollingMaxUpdate(lag, window_size, min_samples, &out[0])

    def _rolling_quantile_update(self, int lag, float p, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.RollingQuantileUpdate(lag, p, window_size, min_samples, &out[0])

    def _seasonal_rolling_mean(self, int lag, int season_length, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.SeasonalRollingMeanTransform(lag, season_length, window_size, min_samples, &out[0])

    def _seasonal_rolling_std(self, int lag, int season_length, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.SeasonalRollingStdTransform(lag, season_length, window_size, min_samples, &out[0])

    def _seasonal_rolling_min(self, int lag, int season_length, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.SeasonalRollingMinTransform(lag, season_length, window_size, min_samples, &out[0])

    def _seasonal_rolling_max(self, int lag, int season_length, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.SeasonalRollingMaxTransform(lag, season_length, window_size, min_samples, &out[0])

    def _seasonal_rolling_quantile(self, int lag, int season_length, float p, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.SeasonalRollingQuantileTransform(lag, season_length, p, window_size, min_samples, &out[0])

    def _seasonal_rolling_mean_update(self, int lag, int season_length, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.SeasonalRollingMeanUpdate(lag, season_length, window_size, min_samples, &out[0])

    def _seasonal_rolling_std_update(self, int lag, int season_length, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.SeasonalRollingStdUpdate(lag, season_length, window_size, min_samples, &out[0])

    def _seasonal_rolling_min_update(self, int lag, int season_length, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.SeasonalRollingMinUpdate(lag, season_length, window_size, min_samples, &out[0])

    def _seasonal_rolling_max_update(self, int lag, int season_length, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.SeasonalRollingMaxUpdate(lag, season_length, window_size, min_samples, &out[0])

    def _seasonal_rolling_quantile_update(self, int lag, int season_length, float p, int window_size, int min_samples, float[::1] out):
        self._cpp_obj.SeasonalRollingQuantileUpdate(lag, season_length, p, window_size, min_samples, &out[0])

    def _expanding_mean(self, int lag, float[::1] out, float[:, ::1] agg):
        self._cpp_obj.ExpandingMean(lag, &out[0], &agg[0])

    def _expanding_std(self, int lag, float[::1] out, float[:, ::1] agg):
        self._cpp_obj.ExpandingStd(lag, &out[0], &agg[0])

    def _expanding_mean(self, int lag, float[::1] out):
        self._cpp_obj.ExpandingMean(lag, &out[0])

    def _expanding_min(self, int lag, float[::1] out):
        self._cpp_obj.ExpandingMin(lag, &out[0])

    def _expanding_max(self, int lag, float[::1] out):
        self._cpp_obj.ExpandingMax(lag, &out[0])

    def _expanding_quantile(self, int lag, float p, float[::1] out):
        self._cpp_obj.ExpandingQuantile(lag, p, &out[0])

    def _expanding_quantile_update(self, int lag, float p, float[::1] out):
        self._cpp_obj.ExpandingQuantileUpdate(lag, p, &out[0])

    def _exponentially_weighted_mean(self, int lag, float alpha, float[::1] out):
        self._cpp_obj.ExponentiallyWeightedMean(lag, alpha, &out[0])

    def _boxcox_lambda_guerrero(self, int period, float lower, float upper, float[::1] out):
        self._cpp_obj.BoxCoxLambdaGuerrero(period, lower, upper, &out[0])

    def _boxcox_lambda_loglik(self, float lower, float upper, float[::1] out):
        self._cpp_obj.BoxCoxLambdaLogLik(lower, upper, &out.data[0])

    def _boxcox_transform(self, float[:, ::1] lmbdas, float[::1] out):
        self._cpp_obj.BoxCoxTransform(&lmbdas[0], &out[0])

    def _boxcox_inverse_transform(self, float[:, ::1] lmbdas, float[::1] out):
        self._cpp_obj.BoxCoxInverseTransform(&lmbdas[0], &out[0])

    def _num_diffs(self, int max_d, float[::1] out):
        self._cpp_obj.NumDiffs(max_d, &out[0])

    def _num_seas_diffs(self, int season_length, int max_d, float[::1] out):
        self._cpp_obj.NumSeasDiffs(season_length, max_d, &out[0])

    def _num_seas_diffs_periods(self, int max_d, int periods, float[:, ::1] out):
        self._cpp_obj.NumSeasDiffsPeriods(max_d, periods, &out[0])

    def _periods(self, int max_period, float[::1] out):
        self._cpp_obj.Periods(max_period, &out[0])

    def _diff(self, int d, float[::1] out):
        self._cpp_obj.Diff(d, &out[0])

    def _diffs(self, indptr_t[::1] ds, float[::1] out):
        self._cpp_obj.Diffs(&ds[0], &out[0])

    def _inv_diffs(self, GroupedArrayFloat32 &other, indptr_t[::1] ds, float[::1] out):
        self._cpp_obj.InvertDifferences(other.cpp_obj, &ds.data, &out.data)
