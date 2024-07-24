#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "diff.h"
#include "nb.h"

void init_diff(nb::module_ &m) {
  nb::module_ diff = m.def_submodule("diff");
  // Float32 Methods
  diff.def("diff", [](const Vector<float> x, int d, Vector<float> out) {
    Difference<float>(x.data(), static_cast<int>(x.size()), out.data(), d);
  });
  diff.def("num_diffs", [](const Vector<float> x, int max_d) {
    float out;
    NumDiffs(x.data(), static_cast<int>(x.size()), &out, max_d);
    return static_cast<int>(out);
  });
  diff.def("num_seas_diffs", [](const Vector<float> x, int period, int max_d) {
    float out;
    NumSeasDiffs(x.data(), static_cast<int>(x.size()), &out, period, max_d);
    return static_cast<int>(out);
  });
  diff.def("period", [](const Vector<float> x, int max_lag) {
    float out;
    GreatestAutocovariance(x.data(), static_cast<int>(x.size()), &out, max_lag);
    return static_cast<int>(out);
  });

  // Float64 Methods
  diff.def("diff", [](const Vector<double> x, int d, Vector<double> out) {
    Difference<double>(x.data(), static_cast<int>(x.size()), out.data(), d);
  });
  diff.def("num_diffs", [](const Vector<double> x, int max_d) {
    double out;
    NumDiffs(x.data(), static_cast<int>(x.size()), &out, max_d);
    return static_cast<int>(out);
  });
  diff.def("num_seas_diffs", [](const Vector<double> x, int period, int max_d) {
    double out;
    NumSeasDiffs(x.data(), static_cast<int>(x.size()), &out, period, max_d);
    return static_cast<int>(out);
  });
  diff.def("period", [](const Vector<double> x, int max_lag) {
    double out;
    GreatestAutocovariance(x.data(), static_cast<int>(x.size()), &out, max_lag);
    return static_cast<int>(out);
  });
}
