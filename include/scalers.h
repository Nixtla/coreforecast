#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>

#include "export.h"
#include "grouped_array.h"
#include "types.h"

extern "C" {
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
}
