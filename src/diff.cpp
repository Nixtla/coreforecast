#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "diff.h"
#include "nb.h"

void init_diff(nb::module_ &m) {
  nb::module_ diff = m.def_submodule("diff");
  // Float32 Methods
  m.def("diff", [](const Vector<float> x, int d, Vector<float> out) {
    Difference<float>(x.data(), static_cast<int>(x.size()), out.data(), d);
  });
  m.def("num_diffs", [](const Vector<float> x, int max_d) {
    float out;
    NumDiffs(x.data(), static_cast<int>(x.size()), &out, max_d);
    return static_cast<int>(out);
  });
  m.def("num_seas_diffs", [](const Vector<float> x, int period, int max_d) {
    float out;
    NumSeasDiffs(x.data(), static_cast<int>(x.size()), &out, period, max_d);
    return static_cast<int>(out);
  });
  m.def("period", [](const Vector<float> x, int max_lag) {
    float out;
    GreatestAutocovariance(x.data(), static_cast<int>(x.size()), &out, max_lag);
    return static_cast<int>(out);
  });
}
