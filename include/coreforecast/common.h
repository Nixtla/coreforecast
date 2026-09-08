#pragma once

// Shared by every kernel. Nothing here may depend on Python or pybind11: the
// kernels are compiled and tested on their own; the binding-side helpers live
// in bindings.h.

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

using indptr_t = int32_t;

// Counts that index into the buffers: zero or less makes the growing loops in
// the kernels start at -1 and read and write one element before them.
inline void RequirePositive(const char *name, int value) {
  if (value > 0) {
    return;
  }
  throw std::invalid_argument(std::string(name) + " must be greater than 0");
}

// Offsets into the buffers: a negative one reads and writes before their start.
inline void RequireNonNegative(const char *name, int value) {
  if (value < 0) {
    throw std::invalid_argument(std::string(name) + " must be non-negative");
  }
}

// Quantile levels index into the sorted window, so a value outside [0, 1]
// indexes past its end. NaN fails both comparisons and is rejected too.
inline void RequireProbability(const char *name, double value) {
  if (value >= 0.0 && value <= 1.0) {
    return;
  }
  throw std::invalid_argument(std::string(name) + " must be between 0 and 1");
}

template <typename T> inline indptr_t FirstNotNaN(const T *data, indptr_t n) {
  indptr_t i = 0;
  while (i < n && std::isnan(data[i])) {
    ++i;
  }
  return i;
}

template <typename T>
inline indptr_t FirstNotNaN(const T *data, indptr_t n, T *out) {
  indptr_t i = 0;
  while (i < n && std::isnan(data[i])) {
    out[i++] = std::numeric_limits<T>::quiet_NaN();
  }
  return i;
}
