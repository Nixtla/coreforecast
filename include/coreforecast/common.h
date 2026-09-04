#pragma once

#include <cstdint>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

using indptr_t = int32_t;
namespace py = pybind11;

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

// The kernels assume NaNs only appear as a leading run, so the prefix is copied
// to the output and only the valid tail is handed to `f`, which is what
// GroupedArray::Transform already does for the grouped entry points.
template <typename T, typename Func>
inline py::array_t<T> SkipLeadingNaN(const py::array_t<T> data, Func f) {
  py::array_t<T> out(data.size());
  auto n = static_cast<indptr_t>(data.size());
  indptr_t start = FirstNotNaN(data.data(), n, out.mutable_data());
  if (start < n) {
    f(data.data() + start, n - start, out.mutable_data() + start);
  }
  return out;
}
