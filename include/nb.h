#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

template <typename T>
using Vector = nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <typename T>
using Matrix = nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
