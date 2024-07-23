#include <nanobind/ndarray.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

template <typename T>
using Vector = nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <typename T>
using VectorView = nb::ndarray_view<T, nb::ndim<1>, 'C'>;


template <typename T>
inline void RollingMeanTransform(const VectorView<T> data, VectorView<T> out, int window_size,
                                 int min_samples) {
  T accum = static_cast<T>(0.0);
  int n = static_cast<int>(data.shape(0));
  int upper_limit = std::min(window_size, n);
  for (int i = 0; i < upper_limit; ++i) {
    accum += data(i);
    if (i + 1 < min_samples) {
      out(i) = std::numeric_limits<T>::quiet_NaN();
    } else {
      out(i) = accum / (i + 1);
    }
  }

  for (int i = window_size; i < n; ++i) {
    accum += data(i) - data(i - window_size);
    out(i) = accum / window_size;
  }
}


template <typename T>
inline void NB_RollingMeanTransform(const Vector<T> data, Vector<T> out, int window_size,
                                 int min_samples) {
    RollingMeanTransform<T>(data.view(), out.view(), window_size, min_samples);
}

NB_MODULE(_coreforecast, m) {
    m.def("rolling_mean", &NB_RollingMeanTransform<float>);
    m.def("rolling_mean", &NB_RollingMeanTransform<double>);
}
