#include "bindings.h"

#include "exponentially_weighted.h"

template <typename T>
py::array_t<T> ExponentiallyWeightedMean(const py::array_t<T> data, T alpha,
                                         bool skipna = false) {
  return SkipLeadingNaN<T>(data, [&](std::span<const T> in, std::span<T> out) {
    exponentially_weighted::MeanTransform<T>(in, out, alpha, skipna);
  });
}

void init_ew(py::module_ &m) {
  py::module_ ew = m.def_submodule("exponentially_weighted");
  ew.def("exponentially_weighted_mean", &ExponentiallyWeightedMean<double>,
         py::arg("data"), py::arg("alpha"), py::arg("skipna") = false);
  ew.def("exponentially_weighted_mean", &ExponentiallyWeightedMean<float>,
         py::arg("data"), py::arg("alpha"), py::arg("skipna") = false);
}
