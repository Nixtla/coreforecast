#include <algorithm>
#include <cmath>
#include <numeric>

#include "coreforecast.h"

template <typename T>
inline float CommonScalerTransform(T data, double scale, double offset) {
  return static_cast<T>((data - offset) / scale);
}

template <typename T>
inline T CommonScalerInverseTransform(T data, double scale, double offset) {
  return static_cast<T>(data * scale + offset);
}

template <typename T> inline int FirstNotNaN(const T *data, int n) {
  int i = 0;
  while (std::isnan(data[i]) && i < n) {
    ++i;
  }
  return i;
}

template <typename T>
inline void MinMaxScalerStats(const T *data, int n, double *stats) {
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
inline void StandardScalerStats(const T *data, int n, double *stats) {
  double sum = std::accumulate(data, data + n, 0.0);
  double mean = sum / n;
  double sum_sq = 0.0;
  for (int i = 0; i < n; ++i) {
    sum_sq += (data[i] - mean) * (data[i] - mean);
  }
  stats[0] = mean;
  stats[1] = sqrt(sum_sq / n);
}

template <typename T> inline double Quantile(T *data, float p, int n) {
  double i_plus_g = p * (n - 1);
  int i = static_cast<int>(i_plus_g);
  double g = i_plus_g - i;
  return data[i] + g * (data[i + 1] - data[i]);
}

template <typename T>
inline void RobustScalerIqrStats(const T *data, int n, double *stats) {
  T *buffer = new T[n];
  std::copy(data, data + n, buffer);
  std::sort(buffer, buffer + n);
  double median = Quantile(buffer, 0.5F, n);
  double q1 = Quantile(buffer, 0.25F, n);
  double q3 = Quantile(buffer, 0.75F, n);
  stats[0] = median;
  stats[1] = q3 - q1;
  delete[] buffer;
}

template <typename T>
inline void RobustScalerMadStats(const T *data, int n, double *stats) {
  T *buffer = new T[n];
  std::copy(data, data + n, buffer);
  std::sort(buffer, buffer + n);
  const T median = static_cast<T>(Quantile(buffer, 0.5F, n));
  for (int i = 0; i < n; ++i) {
    buffer[i] = std::abs(buffer[i] - median);
  }
  std::sort(buffer, buffer + n);
  double mad = Quantile(buffer, 0.5F, n);
  stats[0] = median;
  stats[1] = mad;
  delete[] buffer;
}

template <class T> class GroupedArray {
private:
  const T *data_;
  int32_t n_data_;
  int32_t *indptr_;
  int32_t n_groups_;

public:
  GroupedArray(const T *data, int32_t n_data, int32_t *indptr, int32_t n_indptr)
      : data_(data), n_data_(n_data), indptr_(indptr), n_groups_(n_indptr - 1) {
  }
  ~GroupedArray() {}
  template <typename Func> void ComputeStats(Func f, double *out) const {
    for (int i = 0; i < n_groups_; ++i) {
      int32_t start = indptr_[i];
      int32_t end = indptr_[i + 1];
      int32_t n = end - start;
      int start_idx = FirstNotNaN(data_ + start, n);
      if (start_idx == n)
        continue;
      f(data_ + start + start_idx, n - start_idx, out + 2 * i);
    }
  }

  template <typename Func>
  void ScalerTransform(Func f, const double *stats, T *out) const {
    for (int i = 0; i < n_groups_; ++i) {
      int32_t start = indptr_[i];
      int32_t end = indptr_[i + 1];
      double offset = stats[2 * i];
      double scale = stats[2 * i + 1];
      for (int32_t j = start; j < end; ++j) {
        out[j] = f(data_[j], scale, offset);
      }
    }
  }
};

int GroupedArray_CreateFromArrays(const void *data, int32_t n_data,
                                  int32_t *indptr, int32_t n_groups,
                                  int data_type, GroupedArrayHandle *out) {
  if (data_type == DTYPE_FLOAT32) {
    *out = new GroupedArray<float>(static_cast<const float *>(data), n_data,
                                   indptr, n_groups);
  } else {
    *out = new GroupedArray<double>(static_cast<const double *>(data), n_data,
                                    indptr, n_groups);
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
                                   double *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->ComputeStats(MinMaxScalerStats<float>, out);
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->ComputeStats(MinMaxScalerStats<double>, out);
  }
  return 0;
}

int GroupedArray_StandardScalerStats(GroupedArrayHandle handle, int data_type,
                                     double *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->ComputeStats(StandardScalerStats<float>, out);
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->ComputeStats(StandardScalerStats<double>, out);
  }
  return 0;
}

int GroupedArray_RobustScalerIqrStats(GroupedArrayHandle handle, int data_type,
                                      double *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->ComputeStats(RobustScalerIqrStats<float>, out);
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->ComputeStats(RobustScalerIqrStats<double>, out);
  }
  return 0;
}

int GroupedArray_RobustScalerMadStats(GroupedArrayHandle handle, int data_type,
                                      double *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->ComputeStats(RobustScalerMadStats<float>, out);
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->ComputeStats(RobustScalerMadStats<double>, out);
  }
  return 0;
}

int GroupedArray_ScalerTransform(GroupedArrayHandle handle, const double *stats,
                                 int data_type, void *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->ScalerTransform(CommonScalerTransform<float>, stats,
                        static_cast<float *>(out));
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->ScalerTransform(CommonScalerTransform<double>, stats,
                        static_cast<double *>(out));
  }
  return 0;
}

int GroupedArray_ScalerInverseTransform(GroupedArrayHandle handle,
                                        const double *stats, int data_type,
                                        void *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->ScalerTransform(CommonScalerInverseTransform<float>, stats,
                        static_cast<float *>(out));
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->ScalerTransform(CommonScalerInverseTransform<double>, stats,
                        static_cast<double *>(out));
  }
  return 0;
}
