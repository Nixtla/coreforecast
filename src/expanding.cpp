#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "expanding.h"
#include "nb.h"

void init_expanding(nb::module_ &m) {
  nb::module_ expanding = m.def_submodule("expanding");
  // Float32 Methods
  expanding.def("expanding_mean", [](const Vector<float> x, Vector<float> out) {
    float tmp;
    ExpandingMeanTransform<float>(x.data(), static_cast<int>(x.size()),
                                  out.data(), &tmp);
  });
  expanding.def("expanding_std", [](const Vector<float> x, Vector<float> out) {
    float tmp;
    int length = static_cast<int>(x.size());
    RollingStdTransformWithStats<float>(x.data(), length, out.data(), &tmp,
                                        false, length, 2);
  });
  expanding.def("expanding_min", [](const Vector<float> x, Vector<float> out) {
    ExpandingMinTransform<float>()(x.data(), static_cast<int>(x.size()),
                                   out.data());
  });
  expanding.def("expanding_max", [](const Vector<float> x, Vector<float> out) {
    ExpandingMaxTransform<float>()(x.data(), static_cast<int>(x.size()),
                                   out.data());
  });
  expanding.def("expanding_quantile",
                [](const Vector<float> x, float p, Vector<float> out) {
                  ExpandingQuantileTransform<float>(
                      x.data(), static_cast<int>(x.size()), out.data(), p);
                });

  // Float64 Methods
  expanding.def("expanding_mean",
                [](const Vector<double> x, Vector<double> out) {
                  double tmp;
                  ExpandingMeanTransform<double>(
                      x.data(), static_cast<int>(x.size()), out.data(), &tmp);
                });
  expanding.def("expanding_std",
                [](const Vector<double> x, Vector<double> out) {
                  double tmp;
                  int length = static_cast<int>(x.size());
                  RollingStdTransformWithStats<double>(
                      x.data(), length, out.data(), &tmp, false, length, 2);
                });
  expanding.def("expanding_min",
                [](const Vector<double> x, Vector<double> out) {
                  ExpandingMinTransform<double>()(
                      x.data(), static_cast<int>(x.size()), out.data());
                });
  expanding.def("expanding_max",
                [](const Vector<double> x, Vector<double> out) {
                  ExpandingMaxTransform<double>()(
                      x.data(), static_cast<int>(x.size()), out.data());
                });
  expanding.def("expanding_quantile",
                [](const Vector<double> x, double p, Vector<double> out) {
                  ExpandingQuantileTransform<double>(
                      x.data(), static_cast<int>(x.size()), out.data(), p);
                });
}
