#pragma once

#include "grouped_array.h"
#include <algorithm>

int GroupedArray_LagTransform(GroupedArrayHandle handle, int data_type, int lag,
                              void *out);
