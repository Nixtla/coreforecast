#pragma once

#include <cstdint>

#ifdef _MSC_VER
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

#define C_EXPORT extern "C" DLL_EXPORT

typedef void *GroupedArrayHandle;

C_EXPORT {
  int GroupedArray_CreateFromArrays(float *data, int32_t n_data,
                                    int32_t *indptr, int32_t n_groups,
                                    GroupedArrayHandle *out);

  int GroupedArray_Delete(GroupedArrayHandle handle);

  int GroupedArray_MinMaxScalerStats(GroupedArrayHandle handle, double *out);

  int GroupedArray_StandardScalerStats(GroupedArrayHandle handle, double *out);

  int GroupedArray_RobustScalerIqrStats(GroupedArrayHandle handle, double *out);

  int GroupedArray_RobustScalerMadStats(GroupedArrayHandle handle, double *out);

  int GroupedArray_ScalerTransform(GroupedArrayHandle handle, double *stats,
                                   float *out);

  int GroupedArray_ScalerInverseTransform(GroupedArrayHandle handle,
                                          double *stats, float *out);
}
