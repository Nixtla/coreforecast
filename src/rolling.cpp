#include "rolling.h"

template <typename T>
inline void RollingMeanTransform(const T *data, int n, T *out, int window_size,
                                 int min_samples) {
  T accum = static_cast<T>(0.0);
  int upper_limit = std::min(window_size, n);
  for (int i = 0; i < upper_limit; ++i) {
    accum += data[i];
    if (i + 1 >= min_samples)
      out[i] = accum / (i + 1);
  }

  for (int i = window_size; i < n; ++i) {
    accum += data[i] - data[i - window_size];
    out[i] = accum / window_size;
  }
}

template <typename T>
inline void RollingStdTransform(const T *data, int n, T *out, int window_size,
                                int min_samples) {
  T prev_avg = static_cast<T>(0.0);
  T curr_avg = data[0];
  T m2 = static_cast<T>(0.0);
  int upper_limit = std::min(window_size, n);
  for (int i = 0; i < upper_limit; ++i) {
    prev_avg = curr_avg;
    curr_avg = prev_avg + (data[i] - prev_avg) / (i + 1);
    m2 += (data[i] - prev_avg) * (data[i] - curr_avg);
    if (i + 1 >= min_samples)
      out[i] = sqrt(m2 / i);
  }
  for (int i = window_size; i < n; ++i) {
    T delta = data[i] - data[i - window_size];
    prev_avg = curr_avg;
    curr_avg = prev_avg + delta / window_size;
    m2 += delta * (data[i] - curr_avg + data[i - window_size] - prev_avg);
    out[i] = sqrt(m2 / (window_size - 1));
  }
}

template <typename Func, typename T>
inline void RollingCompTransform(const T *data, int n, T *out, Func comp,
                                 int window_size, int min_samples) {
  int upper_limit = std::min(window_size, n);
  T pivot = data[0];
  for (int i = 0; i < upper_limit; ++i) {
    if (comp(data[i], pivot)) {
      pivot = data[i];
    }
    if (i + 1 >= min_samples) {
      out[i] = pivot;
    }
  }
  for (int i = window_size; i < n; ++i) {
    pivot = data[i];
    for (int j = 0; j < window_size; ++j) {
      if (comp(data[i - j], pivot)) {
        pivot = data[i - j];
      }
    }
    out[i] = pivot;
  }
}

template <typename T> struct RollingMinTransform {
  void operator()(const T *data, int n, T *out, int window_size,
                  int min_samples) {
    RollingCompTransform(data, n, out, std::less<T>(), window_size,
                         min_samples);
  }
};

template <typename T> struct RollingMaxTransform {
  void operator()(const T *data, int n, T *out, int window_size,
                  int min_samples) const {
    RollingCompTransform(data, n, out, std::greater<T>(), window_size,
                         min_samples);
  }
};

int GroupedArray_RollingMeanTransform(GroupedArrayHandle handle, int data_type,
                                      int lag, int window_size, int min_samples,
                                      void *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->Transform(RollingMeanTransform<float>, lag, static_cast<float *>(out),
                  window_size, min_samples);
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->Transform(RollingMeanTransform<double>, lag, static_cast<double *>(out),
                  window_size, min_samples);
  }
  return 0;
}

int GroupedArray_RollingStdTransform(GroupedArrayHandle handle, int data_type,
                                     int lag, int window_size, int min_samples,
                                     void *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->Transform(RollingStdTransform<float>, lag, static_cast<float *>(out),
                  window_size, min_samples);
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->Transform(RollingStdTransform<double>, lag, static_cast<double *>(out),
                  window_size, min_samples);
  }
  return 0;
}

int GroupedArray_RollingMinTransform(GroupedArrayHandle handle, int data_type,
                                     int lag, int window_size, int min_samples,
                                     void *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->Transform(RollingMinTransform<float>(), lag, static_cast<float *>(out),
                  window_size, min_samples);
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->Transform(RollingMinTransform<double>(), lag,
                  static_cast<double *>(out), window_size, min_samples);
  }
  return 0;
}

int GroupedArray_RollingMaxTransform(GroupedArrayHandle handle, int data_type,
                                     int lag, int window_size, int min_samples,
                                     void *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->Transform(RollingMaxTransform<float>(), lag, static_cast<float *>(out),
                  window_size, min_samples);
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->Transform(RollingMaxTransform<double>(), lag,
                  static_cast<double *>(out), window_size, min_samples);
  }
  return 0;
}

template <typename Func, typename T>
inline void RollingUpdate(Func RollingTfm, const T *data, int n, T *out,
                          int window_size, int min_samples) {
  if (n < min_samples) {
    *out = std::numeric_limits<T>::quiet_NaN();
    return;
  }
  int n_samples = std::min(window_size, n);
  T *buffer = new T[n_samples];
  RollingTfm(data, n_samples, buffer, window_size, min_samples);
  *out = buffer[n_samples - 1];
  delete[] buffer;
}

/* int GroupedArray_RollingMeanUpdate(GroupedArrayHandle handle, int data_type,
                                   int lag, int window_size, int min_samples,
                                   void *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->Reduce(RollingMeanUpdate<float>, 1, static_cast<float *>(out), lag,
               window_size, min_samples);
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->Reduce(RollingMeanUpdate<double>, 1, static_cast<double *>(out), lag,
               window_size, min_samples);
  }
  return 0;
} */
