#pragma once

// Helpers for the pybind11 layer in src/. Kernels must not include this.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "common.h"

namespace py = pybind11;

// Contiguous input array. The kernels take a raw pointer, so a strided view
// would be read as if it were contiguous and silently give results for the
// wrong elements.
template <typename T>
using CArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

// Ensures contiguity without putting c_style on the parameter itself: doing
// that makes array_t::check_ fail for a strided array, so pybind's
// no-conversion pass matches no overload and the conversion pass picks the
// first one registered, silently converting the input to that overload's dtype
// (float64, which is the one registered first).
template <typename T>
inline CArray<T> AsContiguous(const py::array_t<T> &data) {
  return CArray<T>::ensure(data);
}

// The kernels assume NaNs only appear as a leading run, so the prefix is copied
// to the output and only the valid tail is handed to `f`, which is what
// GroupedArray::Transform already does for the grouped entry points.
template <typename T, typename Func>
inline py::array_t<T> SkipLeadingNaN(const py::array_t<T> &data_arg, Func f) {
  const auto data = AsContiguous(data_arg);
  py::array_t<T> out(data.size());
  auto n = static_cast<indptr_t>(data.size());
  indptr_t start = FirstNotNaN(data.data(), n, out.mutable_data());
  if (start < n) {
    f(data.data() + start, n - start, out.mutable_data() + start);
  }
  return out;
}
