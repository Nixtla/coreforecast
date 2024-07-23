
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "exponentially_weighted.h"
#include "nb.h"

void init_exp_weigh(nb::module_ &m) {
  nb::module_ exp_weigh = m.def_submodule("exponentially_weighted");
  // Float32 Methods
  exp_weigh.def("exponentially_weighted_mean",
                [](const Vector<float> x, float alpha, Vector<float> out) {
                  ExponentiallyWeightedMeanTransform<float>(
                      x.data(), static_cast<int>(x.size()), out.data(), alpha);
                });
}
