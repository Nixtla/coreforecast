#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using indptr_t = int32_t;

template <typename T>
void MeanTransform(const T *data, int n, T *out, int window_size,
                   int min_samples) {
  T accum = static_cast<T>(0.0);
  int upper_limit = std::min(window_size, n);
  for (int i = 0; i < upper_limit; ++i) {
    accum += data[i];
    if (i + 1 < min_samples) {
      out[i] = std::numeric_limits<T>::quiet_NaN();
    } else {
      out[i] = accum / (i + 1);
    }
  }

  for (int i = window_size; i < n; ++i) {
    accum += data[i] - data[i - window_size];
    out[i] = accum / window_size;
  }
}

template <class T> class GroupedArray {
public:
  GroupedArray(const py::array_t<T> &data, const py::array_t<indptr_t> &indptr,
               int num_threads)
      : data_(data), indptr_(indptr), num_threads_(num_threads) {}

  py::array_t<T> RollingMeanTransform(int window_size, int min_samples) {
    py::array_t<T> out(data_.size());
    MeanTransform(data_.data(), static_cast<int>(data_.size()),
                  out.mutable_data(), window_size, min_samples);
    return out;
  }

private:
  py::array_t<T> data_;
  py::array_t<indptr_t> indptr_;
  int num_threads_;
};

template <typename T> void bind_ga(py::module &m, const std::string &name) {
  py::class_<GroupedArray<T>>(m, name.c_str())
      .def(py::init<const py::array_t<T> &, const py::array_t<indptr_t> &,
                    int>())
      .def("rolling_mean", &GroupedArray<T>::RollingMeanTransform);
}

PYBIND11_MODULE(ga, m) {
  bind_ga<float>(m, "GroupedArray");
  bind_ga<double>(m, "GroupedArray");
}
