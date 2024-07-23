#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "nb.h"
#include "scalers.h"

void init_scalers(nb::module_ &m) {
  nb::module_ scalers = m.def_submodule("scalers");
  // Float32 Methods
  scalers.def("boxcox_lambda_guerrero", [](const Vector<float> x, int period,
                                           float lower, float upper) {
    float out;
    BoxCoxLambda_Guerrero<float>(x.data(), static_cast<int>(x.size()), &out,
                                 period, lower, upper);
    return out;
  });
  scalers.def("boxcox_lambda_loglik",
              [](const Vector<float> x, float lower, float upper) {
                float out;
                BoxCoxLambda_LogLik<float>(x.data(), static_cast<int>(x.size()),
                                           &out, lower, upper);
                return out;
              });
  scalers.def("boxcox",
              [](const Vector<float> x, float lambda, Vector<float> out) {
                std::transform(x.data(), x.data() + x.size(), out.data(),
                               [lambda](float x) {
                                 return BoxCoxTransform<float>(x, lambda, 0.0);
                               });
              });
  scalers.def(
      "inv_boxcox", [](const Vector<float> x, float lambda, Vector<float> out) {
        std::transform(x.data(), x.data() + x.size(), out.data(),
                       [lambda](float x) {
                         return BoxCoxInverseTransform<float>(x, lambda, 0.0);
                       });
      });
}
