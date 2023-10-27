#include <algorithm>
#include <cmath>
#include <numeric>

#include "coreforecast.h"

template <typename T>
inline T CommonScalerTransform(T data, T scale, T offset) {
  return (data - offset) / scale;
}

template <typename T>
inline T CommonScalerInverseTransform(T data, T scale, T offset) {
  return data * scale + offset;
}

template <typename T> inline indptr_t FirstNotNaN(const T *data, indptr_t n) {
  indptr_t i = 0;
  while (std::isnan(data[i]) && i < n) {
    ++i;
  }
  return i;
}

template <typename T>
inline void MinMaxScalerStats(const T *data, int n, T *stats) {
  T min = std::numeric_limits<T>::infinity();
  T max = -std::numeric_limits<T>::infinity();
  for (int i = 0; i < n; ++i) {
    if (data[i] < min)
      min = data[i];
    if (data[i] > max)
      max = data[i];
  }
  stats[0] = min;
  stats[1] = max - min;
}

template <typename T>
inline void StandardScalerStats(const T *data, int n, T *stats) {
  double sum = std::accumulate(data, data + n, 0.0);
  double mean = sum / n;
  double sum_sq = 0.0;
  for (int i = 0; i < n; ++i) {
    sum_sq += (data[i] - mean) * (data[i] - mean);
  }
  stats[0] = static_cast<T>(mean);
  stats[1] = static_cast<T>(sqrt(sum_sq / n));
}

template <typename T> inline T Quantile(T *data, T p, int n) {
  T i_plus_g = p * (n - 1);
  int i = static_cast<int>(i_plus_g);
  T g = i_plus_g - i;
  std::nth_element(data, data + i, data + n);
  T out = data[i];
  if (g > 0.0) {
    std::nth_element(data, data + i + 1, data + n);
    out += g * (data[i + 1] - out);
  }
  return out;
}

template <typename T>
inline void RobustScalerIqrStats(const T *data, int n, T *stats) {
  T *buffer = new T[n];
  std::copy(data, data + n, buffer);
  T median = Quantile(buffer, static_cast<T>(0.5), n);
  T q1 = Quantile(buffer, static_cast<T>(0.25), n);
  T q3 = Quantile(buffer, static_cast<T>(0.75), n);
  stats[0] = median;
  stats[1] = q3 - q1;
  delete[] buffer;
}

template <typename T>
inline void RobustScalerMadStats(const T *data, int n, T *stats) {
  T *buffer = new T[n];
  std::copy(data, data + n, buffer);
  const T median = Quantile(buffer, static_cast<T>(0.5), n);
  for (int i = 0; i < n; ++i) {
    buffer[i] = std::abs(buffer[i] - median);
  }
  T mad = Quantile(buffer, static_cast<T>(0.5), n);
  stats[0] = median;
  stats[1] = mad;
  delete[] buffer;
}

template <typename T> inline void LagTransform(const T *data, int n, T *out) {
  std::copy(data, data + n, out);
}

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
inline void RollingStdTransform(const T *data, int n, T *out, T *agg,
                                int window_size, int min_samples) {
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
  agg[0] = curr_avg;
  agg[1] = m2;
}

template <typename T>
inline void RollingStdUpdate(const T *data, int n, T *out, T *agg,
                             int window_size, int min_samples) {
  T prev_avg = agg[0];
  if (n < window_size) {
    *agg[0] = prev_avg + (data[n - 1] - prev_avg) / n;
    *agg[1] = (data[n - 1] - prev_avg) * (data[n - 1] - *agg[0]);
  } else {
    T old = data[n - window_size - 1];
    *agg[0] = prev_avg + (data[n - 1] - old) / window_size;
    *agg[1] = (data[n - 1] - old) * (data[n - 1] - *agg[0] + old - prev_avg);
  }
  // possible loss of precision
  *agg[1] = std::max(*agg[1], static_cast<T>(0.0));
  if (n < min_samples) {
    *out = std::numeric_limits<T>::quiet_NaN();
  } else {
    *out = sqrt(*agg[1] / (window_size - 1));
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
      out[i] = data[i];
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

template <typename T> struct LessThanCompFunctor {
  void operator()(const T *data, int n, T *out, int window_size,
                  int min_samples) {
    RollingCompTransform(data, n, out, std::less<T>(), window_size,
                         min_samples);
  }
};

template <typename T> struct GreaterThanCompFuctor {
  void operator()(const T *data, int n, T *out, int window_size,
                  int min_samples) const {
    RollingCompTransform(data, n, out, std::greater<T>(), window_size,
                         min_samples);
  }
};

template <typename T>
inline void SeasonalRollingMeanTransform(const T *data, int n, T *out,
                                         int season_length, int window_size,
                                         int min_samples) {
  int buff_size = n / season_length + (n % season_length > 0);
  T *season_data = new T[buff_size];
  T *season_out = new T[buff_size];
  std::fill_n(season_out, buff_size, std::numeric_limits<T>::quiet_NaN());
  for (int i = 0; i < season_length; ++i) {
    int season_n = n / season_length + (i < n % season_length);
    for (int j = 0; j < season_n; ++j) {
      season_data[j] = data[i + j * season_length];
    }
    RollingMeanTransform(season_data, season_n, season_out, window_size,
                         min_samples);
    for (int j = 0; j < season_n; ++j) {
      out[i + j * season_length] = season_out[j];
    }
  }
  delete[] season_data;
  delete[] season_out;
}

template <typename T>
inline void RollingMeanUpdate(const T *data, int n, T *out, int window_size,
                              int min_samples) {
  if (n < min_samples) {
    *out = std::numeric_limits<T>::quiet_NaN();
    return;
  }
  int n_samples = std::min(window_size, n);
  T accum = std::accumulate(data + n - n_samples, data + n, 0.0);
  *out = accum / n_samples;
}

template <typename T>
inline void SeasonalRollingMeanUpdate(const T *data, int n, T *out,
                                      int season_length, int window_size,
                                      int min_samples) {
  int season = n % season_length;
  int season_n = n / season_length + (season > 0);
  if (season_n < min_samples) {
    *out = std::numeric_limits<T>::quiet_NaN();
    return;
  }
  int n_copy = std::min(window_size, season_n);
  T *season_data = new T[n_copy];
  for (int i = 0; i < n_copy; ++i) {
    season_data[i] = data[n - (n_copy - i) * season_length];
  }
  RollingMeanUpdate(season_data, n_copy, out, window_size, min_samples);
}

template <typename T>
inline void ExpandingMeanTransform(const T *data, int n, T *out, T *agg) {
  T accum = static_cast<T>(0.0);
  for (int i = 0; i < n; ++i) {
    accum += data[i];
    out[i] = accum / (i + 1);
  }
  *agg = static_cast<T>(n);
}

template <class T> class GroupedArray {
private:
  const T *data_;
  indptr_t n_data_;
  const indptr_t *indptr_;
  int n_groups_;
  int num_threads_;

public:
  GroupedArray(const T *data, indptr_t n_data, const indptr_t *indptr,
               int n_indptr, int num_threads)
      : data_(data), n_data_(n_data), indptr_(indptr), n_groups_(n_indptr - 1),
        num_threads_(num_threads) {}
  ~GroupedArray() {}
  template <typename Func, typename... Args>

  void Reduce(Func f, int n_out, T *out, int lag,
              Args &&...args) const noexcept {
#pragma omp parallel for schedule(static) num_threads(num_threads_)
    for (int i = 0; i < n_groups_; ++i) {
      indptr_t start = indptr_[i];
      indptr_t end = indptr_[i + 1];
      indptr_t n = end - start;
      if (lag >= n)
        continue;
      indptr_t start_idx = FirstNotNaN(data_ + start, n);
      if (start_idx == n)
        continue;
      f(data_ + start + start_idx, n - start_idx - lag, out + n_out * i,
        std::forward<Args>(args)...);
    }
  }

  template <typename Func>
  void ScalerTransform(Func f, const T *stats, T *out) const noexcept {
#pragma omp parallel for schedule(static) num_threads(num_threads_)
    for (int i = 0; i < n_groups_; ++i) {
      indptr_t start = indptr_[i];
      indptr_t end = indptr_[i + 1];
      T offset = stats[2 * i];
      T scale = stats[2 * i + 1];
      if (std::abs(scale) < std::numeric_limits<T>::epsilon()) {
        scale = static_cast<T>(1.0);
      }
      for (indptr_t j = start; j < end; ++j) {
        out[j] = f(data_[j], scale, offset);
      }
    }
  }

  template <typename Func, typename... Args>
  void Transform(Func f, int lag, T *out, Args &&...args) const noexcept {
#pragma omp parallel for schedule(static) num_threads(num_threads_)
    for (int i = 0; i < n_groups_; ++i) {
      indptr_t start = indptr_[i];
      indptr_t end = indptr_[i + 1];
      indptr_t n = end - start;
      indptr_t start_idx = FirstNotNaN(data_ + start, n);
      if (start_idx == n) {
        continue;
      }
      start += start_idx;
      f(data_ + start, n - start_idx - lag, out + start + lag,
        std::forward<Args>(args)...);
    }
  }

  template <typename Func>
  void TransformAndReduce(Func f, int lag, T *out, int n_agg,
                          T *agg) const noexcept {
#pragma omp parallel for schedule(static) num_threads(num_threads_)
    for (int i = 0; i < n_groups_; ++i) {
      indptr_t start = indptr_[i];
      indptr_t end = indptr_[i + 1];
      indptr_t n = end - start;
      indptr_t start_idx = FirstNotNaN(data_ + start, n);
      if (start_idx == n) {
        continue;
      }
      start += start_idx;
      f(data_ + start, n - start_idx - lag, out + start + lag, agg + i * n_agg);
    }
  }
};

int GroupedArray_Create(const void *data, indptr_t n_data, indptr_t *indptr,
                        indptr_t n_groups, int num_threads, int data_type,
                        GroupedArrayHandle *out) {
  if (data_type == DTYPE_FLOAT32) {
    *out = new GroupedArray<float>(static_cast<const float *>(data), n_data,
                                   indptr, n_groups, num_threads);
  } else {
    *out = new GroupedArray<double>(static_cast<const double *>(data), n_data,
                                    indptr, n_groups, num_threads);
  }
  return 0;
}

int GroupedArray_Delete(GroupedArrayHandle handle, int data_type) {
  if (data_type == DTYPE_FLOAT32) {
    delete reinterpret_cast<GroupedArray<float> *>(handle);
  } else {
    delete reinterpret_cast<GroupedArray<double> *>(handle);
  }
  return 0;
}

int GroupedArray_MinMaxScalerStats(GroupedArrayHandle handle, int data_type,
                                   void *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->Reduce(MinMaxScalerStats<float>, 2, static_cast<float *>(out), 0);
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->Reduce(MinMaxScalerStats<double>, 2, static_cast<double *>(out), 0);
  }
  return 0;
}

int GroupedArray_StandardScalerStats(GroupedArrayHandle handle, int data_type,
                                     void *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->Reduce(StandardScalerStats<float>, 2, static_cast<float *>(out), 0);
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->Reduce(StandardScalerStats<double>, 2, static_cast<double *>(out), 0);
  }
  return 0;
}

int GroupedArray_RobustScalerIqrStats(GroupedArrayHandle handle, int data_type,
                                      void *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->Reduce(RobustScalerIqrStats<float>, 2, static_cast<float *>(out), 0);
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->Reduce(RobustScalerIqrStats<double>, 2, static_cast<double *>(out), 0);
  }
  return 0;
}

int GroupedArray_RobustScalerMadStats(GroupedArrayHandle handle, int data_type,
                                      void *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->Reduce(RobustScalerMadStats<float>, 2, static_cast<float *>(out), 0);
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->Reduce(RobustScalerMadStats<double>, 2, static_cast<double *>(out), 0);
  }
  return 0;
}

int GroupedArray_ScalerTransform(GroupedArrayHandle handle, const void *stats,
                                 int data_type, void *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->ScalerTransform(CommonScalerTransform<float>,
                        static_cast<const float *>(stats),
                        static_cast<float *>(out));
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->ScalerTransform(CommonScalerTransform<double>,
                        static_cast<const double *>(stats),
                        static_cast<double *>(out));
  }
  return 0;
}

int GroupedArray_ScalerInverseTransform(GroupedArrayHandle handle,
                                        const void *stats, int data_type,
                                        void *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->ScalerTransform(CommonScalerInverseTransform<float>,
                        static_cast<const float *>(stats),
                        static_cast<float *>(out));
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->ScalerTransform(CommonScalerInverseTransform<double>,
                        static_cast<const double *>(stats),
                        static_cast<double *>(out));
  }
  return 0;
}

int GroupedArray_LagTransform(GroupedArrayHandle handle, int data_type, int lag,
                              void *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->Transform(LagTransform<float>, lag, static_cast<float *>(out));
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->Transform(LagTransform<double>, lag, static_cast<double *>(out));
  }
  return 0;
}

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

int GroupedArray_RollingMeanUpdate(GroupedArrayHandle handle, int data_type,
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
}

int GroupedArray_ExpandingMeanTransform(GroupedArrayHandle handle,
                                        int data_type, int lag, void *out,
                                        void *agg) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->TransformAndReduce(ExpandingMeanTransform<float>, lag,
                           static_cast<float *>(out), 1,
                           static_cast<float *>(agg));
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->TransformAndReduce(ExpandingMeanTransform<double>, lag,
                           static_cast<double *>(out), 1,
                           static_cast<double *>(agg));
  }
  return 0;
}

int GroupedArray_SeasonalRollingMeanTransform(GroupedArrayHandle handle,
                                              int data_type, int lag,
                                              int season_length,
                                              int window_size, int min_samples,
                                              void *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->Transform(SeasonalRollingMeanTransform<float>, lag,
                  static_cast<float *>(out), season_length, window_size,
                  min_samples);
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->Transform(SeasonalRollingMeanTransform<double>, lag,
                  static_cast<double *>(out), season_length, window_size,
                  min_samples);
  }
  return 0;
}

int GroupedArray_SeasonalRollingMeanUpdate(GroupedArrayHandle handle,
                                           int data_type, int lag,
                                           int season_length, int window_size,
                                           int min_samples, void *out) {

  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->Reduce(SeasonalRollingMeanUpdate<float>, 1, static_cast<float *>(out),
               lag, season_length, window_size, min_samples);
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->Reduce(SeasonalRollingMeanUpdate<double>, 1, static_cast<double *>(out),
               lag, season_length, window_size, min_samples);
  }
  return 0;
}

int GroupedArray_RollingMinTransform(GroupedArrayHandle handle, int data_type,
                                     int lag, int window_size, int min_samples,
                                     void *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->Transform(LessThanCompFunctor<float>(), lag, static_cast<float *>(out),
                  window_size, min_samples);
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->Transform(LessThanCompFunctor<double>(), lag,
                  static_cast<double *>(out), window_size, min_samples);
  }
}

int GroupedArray_RollingMaxTransform(GroupedArrayHandle handle, int data_type,
                                     int lag, int window_size, int min_samples,
                                     void *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->Transform(GreaterThanCompFuctor<float>(), lag,
                  static_cast<float *>(out), window_size, min_samples);
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->Transform(GreaterThanCompFuctor<double>(), lag,
                  static_cast<double *>(out), window_size, min_samples);
  }
}
