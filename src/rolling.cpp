#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "nb.h"
#include "rolling.h"

void init_rolling(nb::module_ &m) {
  nb::module_ rolling = m.def_submodule("rolling");
  // Float32 Methods
  // Rolling
  rolling.def("rolling_mean", [](const Vector<float> x, int window_size,
                                 int min_samples, Vector<float> out) {
    RollingMeanTransform<float>(x.data(), static_cast<int>(x.size()),
                                out.data(), window_size, min_samples);
  });
  rolling.def("rolling_std", [](const Vector<float> x, int window_size,
                                int min_samples, Vector<float> out) {
    RollingStdTransform<float>(x.data(), static_cast<int>(x.size()), out.data(),
                               window_size, min_samples);
  });
  rolling.def("rolling_min", [](const Vector<float> x, int window_size,
                                int min_samples, Vector<float> out) {
    RollingMinTransform<float>(x.data(), static_cast<int>(x.size()), out.data(),
                               window_size, min_samples);
  });
  rolling.def("rolling_max", [](const Vector<float> x, int window_size,
                                int min_samples, Vector<float> out) {
    RollingMaxTransform<float>(x.data(), static_cast<int>(x.size()), out.data(),
                               window_size, min_samples);
  });
  rolling.def("rolling_quantile", [](const Vector<float> x, float p,
                                     int window_size, int min_samples,
                                     Vector<float> out) {
    RollingQuantileTransform<float>(x.data(), static_cast<int>(x.size()),
                                    out.data(), window_size, min_samples, p);
  });

  // Seasonal rolling
  rolling.def("seasonal_rolling_mean", [](const Vector<float> x,
                                          int season_length, int window_size,
                                          int min_samples, Vector<float> out) {
    SeasonalRollingMeanTransform<float>()(x.data(), static_cast<int>(x.size()),
                                          out.data(), season_length,
                                          window_size, min_samples);
  });
  rolling.def("seasonal_rolling_std", [](const Vector<float> x,
                                         int season_length, int window_size,
                                         int min_samples, Vector<float> out) {
    SeasonalRollingStdTransform<float>()(x.data(), static_cast<int>(x.size()),
                                         out.data(), season_length, window_size,
                                         min_samples);
  });
  rolling.def("seasonal_rolling_min", [](const Vector<float> x,
                                         int season_length, int window_size,
                                         int min_samples, Vector<float> out) {
    SeasonalRollingMinTransform<float>()(x.data(), static_cast<int>(x.size()),
                                         out.data(), season_length, window_size,
                                         min_samples);
  });
  rolling.def("seasonal_rolling_max", [](const Vector<float> x,
                                         int season_length, int window_size,
                                         int min_samples, Vector<float> out) {
    SeasonalRollingMaxTransform<float>()(x.data(), static_cast<int>(x.size()),
                                         out.data(), season_length, window_size,
                                         min_samples);
  });
  rolling.def("seasonal_rolling_quantile",
              [](const Vector<float> x, int season_length, float p,
                 int window_size, int min_samples, Vector<float> out) {
                SeasonalRollingQuantileTransform<float>()(
                    x.data(), static_cast<int>(x.size()), out.data(),
                    season_length, window_size, min_samples, p);
              });
}
