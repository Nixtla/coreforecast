#pragma once

#include "export.h"
#include "grouped_array.h"
#include "rolling.h"

extern "C" {
DLL_EXPORT int GroupedArray_SeasonalRollingMeanTransform(
    GroupedArrayHandle handle, int data_type, int lag, int season_length,
    int window_size, int min_samples, void *out);
DLL_EXPORT int GroupedArray_SeasonalRollingMeanUpdate(
    GroupedArrayHandle handle, int data_type, int lag, int season_length,
    int window_size, int min_samples, void *out);
}
