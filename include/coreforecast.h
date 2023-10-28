#pragma once

#include <cstdint>

#ifdef _MSC_VER
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

using GroupedArrayHandle = void *;
using indptr_t = int32_t;

#define DTYPE_FLOAT32 (0)
#define DTYPE_FLOAT64 (1)

extern "C" {
DLL_EXPORT int GroupedArray_Create(const void *data, int32_t n_data,
                                   int32_t *indptr, int32_t n_groups,
                                   int num_threads, int data_type,
                                   GroupedArrayHandle *out);

DLL_EXPORT int GroupedArray_Delete(GroupedArrayHandle handle, int data_type);

DLL_EXPORT int GroupedArray_MinMaxScalerStats(GroupedArrayHandle handle,
                                              int data_type, void *out);

DLL_EXPORT int GroupedArray_StandardScalerStats(GroupedArrayHandle handle,
                                                int data_type, void *out);

DLL_EXPORT int GroupedArray_RobustScalerIqrStats(GroupedArrayHandle handle,
                                                 int data_type, void *out);

DLL_EXPORT int GroupedArray_RobustScalerMadStats(GroupedArrayHandle handle,
                                                 int data_type, void *out);

DLL_EXPORT int GroupedArray_ScalerTransform(GroupedArrayHandle handle,
                                            const void *stats, int data_type,
                                            void *out);

DLL_EXPORT int GroupedArray_ScalerInverseTransform(GroupedArrayHandle handle,
                                                   const void *stats,
                                                   int data_type, void *out);

DLL_EXPORT int GroupedArray_LagTransform(GroupedArrayHandle handle,
                                         int data_type, int lag, void *out);

DLL_EXPORT int GroupedArray_RollingMeanTransform(GroupedArrayHandle handle,
                                                 int data_type, int lag,
                                                 int window_size,
                                                 int min_samples, void *out);

DLL_EXPORT int GroupedArray_RollingStdTransform(GroupedArrayHandle handle,
                                                int data_type, int lag,
                                                int window_size,
                                                int min_samples, void *out);

DLL_EXPORT int GroupedArray_RollingMinTransform(GroupedArrayHandle handle,
                                                int data_type, int lag,
                                                int window_size,
                                                int min_samples, void *out);
DLL_EXPORT int GroupedArray_RollingMaxTransform(GroupedArrayHandle handle,
                                                int data_type, int lag,
                                                int window_size,
                                                int min_samples, void *out);

DLL_EXPORT int GroupedArray_RollingMeanUpdate(GroupedArrayHandle handle,
                                              int data_type, int lag,
                                              int window_size, int min_samples,
                                              void *out);

DLL_EXPORT int GroupedArray_RollingStdUpdate(GroupedArrayHandle handle,
                                             int data_type, int lag,
                                             int window_size, int min_samples,
                                             void *out);

DLL_EXPORT int GroupedArray_RollingMinUpdate(GroupedArrayHandle handle,
                                             int data_type, int lag,
                                             int window_size, int min_samples,
                                             void *out);

DLL_EXPORT int GroupedArray_RollingMaxUpdate(GroupedArrayHandle handle,
                                             int data_type, int lag,
                                             int window_size, int min_samples,
                                             void *out);

DLL_EXPORT int GroupedArray_SeasonalRollingMeanTransform(
    GroupedArrayHandle handle, int data_type, int lag, int season_length,
    int window_size, int min_samples, void *out);

DLL_EXPORT int GroupedArray_SeasonalRollingStdTransform(
    GroupedArrayHandle handle, int data_type, int lag, int season_length,
    int window_size, int min_samples, void *out);

DLL_EXPORT int GroupedArray_SeasonalRollingMinTransform(
    GroupedArrayHandle handle, int data_type, int lag, int season_length,
    int window_size, int min_samples, void *out);

DLL_EXPORT int GroupedArray_SeasonalRollingMaxTransform(
    GroupedArrayHandle handle, int data_type, int lag, int season_length,
    int window_size, int min_samples, void *out);

DLL_EXPORT int GroupedArray_SeasonalRollingMeanUpdate(
    GroupedArrayHandle handle, int data_type, int lag, int season_length,
    int window_size, int min_samples, void *out);

DLL_EXPORT int GroupedArray_SeasonalRollingStdUpdate(
    GroupedArrayHandle handle, int data_type, int lag, int season_length,
    int window_size, int min_samples, void *out);

DLL_EXPORT int GroupedArray_SeasonalRollingMinUpdate(
    GroupedArrayHandle handle, int data_type, int lag, int season_length,
    int window_size, int min_samples, void *out);

DLL_EXPORT int GroupedArray_SeasonalRollingMaxUpdate(
    GroupedArrayHandle handle, int data_type, int lag, int season_length,
    int window_size, int min_samples, void *out);

DLL_EXPORT int GroupedArray_ExpandingMeanTransform(GroupedArrayHandle handle,
                                                   int data_type, int lag,
                                                   void *out, void *agg);

DLL_EXPORT int GroupedArray_ExpandingStdTransform(GroupedArrayHandle handle,
                                                  int data_type, int lag,
                                                  void *out, void *agg);

DLL_EXPORT int GroupedArray_ExpandingMinTransform(GroupedArrayHandle handle,
                                                  int data_type, int lag,
                                                  void *out);

DLL_EXPORT int GroupedArray_ExpandingMaxTransform(GroupedArrayHandle handle,
                                                  int data_type, int lag,
                                                  void *out);
}
