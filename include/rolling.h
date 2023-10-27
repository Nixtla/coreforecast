#pragma once

#include <cmath>

#include "grouped_array.h"

int GroupedArray_RollingMeanTransform(GroupedArrayHandle handle, int data_type,
                                      int lag, int window_size, int min_samples,
                                      void *out);

int GroupedArray_RollingStdTransform(GroupedArrayHandle handle, int data_type,
                                     int lag, int window_size, int min_samples,
                                     void *out);

int GroupedArray_RollingMinTransform(GroupedArrayHandle handle, int data_type,
                                     int lag, int window_size, int min_samples,
                                     void *out);

int GroupedArray_RollingMaxTransform(GroupedArrayHandle handle, int data_type,
                                     int lag, int window_size, int min_samples,
                                     void *out);
