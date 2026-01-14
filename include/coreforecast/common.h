#pragma once

#include <cstdint>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

using indptr_t = int32_t;
namespace py = pybind11;

template <typename T> inline indptr_t FirstNotNaN(const T *data, indptr_t n) {
  indptr_t i = 0;
  while (std::isnan(data[i]) && i < n) {
    ++i;
  }
  return i;
}

template <typename T>
inline indptr_t FirstNotNaN(const T *data, indptr_t n, T *out) {
  indptr_t i = 0;
  while (std::isnan(data[i]) && i < n) {
    out[i++] = std::numeric_limits<T>::quiet_NaN();
  }
  return i;
}

template <typename T> inline bool IsNaN(T value) { return std::isnan(value); }

template <typename T> inline int CountValidValues(const T *data, int n) {
  int count = 0;
  for (int i = 0; i < n; ++i) {
    if (!std::isnan(data[i])) {
      ++count;
    }
  }
  return count;
}

template <typename T>
inline void FilterValidValues(const T *data, int n, std::vector<T> &out) {
  out.clear();
  out.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (!std::isnan(data[i])) {
      out.push_back(data[i]);
    }
  }
}
