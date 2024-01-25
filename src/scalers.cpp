#include "scalers.h"
#include "brent.h"
#include "stats.h"

#include <numeric>
#include <vector>

template <typename T>
inline T CommonScalerTransform(T data, T scale, T offset) {
  return (data - offset) / scale;
}

template <typename T>
inline T CommonScalerInverseTransform(T data, T scale, T offset) {
  return data * scale + offset;
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
  double mean = Mean(data, n);
  double std = StandardDeviation(data, n, mean);
  stats[0] = static_cast<T>(mean);
  stats[1] = static_cast<T>(std);
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

double GuerreroCV(double lambda, const std::vector<double> &x_mean,
                  const std::vector<double> &x_std) {
  auto x_rat = std::vector<double>(x_std.size());
  int start_idx = 0;
  for (size_t i = 0; i < x_rat.size(); ++i) {
    if (std::isnan(x_std[i])) {
      start_idx++;
      continue;
    }
    x_rat[i] = x_std[i] / std::pow(x_mean[i], 1.0 - lambda);
  }
  double mean = Mean(x_rat.data() + start_idx, x_rat.size() - start_idx);
  double std = StandardDeviation(x_rat.data() + start_idx,
                                 x_rat.size() - start_idx, mean, 1);
  if (std::isnan(std)) {
    return std::numeric_limits<double>::max();
  }
  return std / mean;
}

double BoxCoxLambda_Guerrero(const double *x, int n, int period, double lower,
                             double upper) {
  int n_seasons = n / period;
  int n_full = n_seasons * period;
  // build matrix with subseries having full periods
  auto x_mat = std::vector<double>(n_seasons * period);
  std::copy(x + n - n_full, x + n, x_mat.begin());
  // means of subseries
  auto x_mean = std::vector<double>(n_seasons, 0.0);
  auto x_n = std::vector<int>(n_seasons, 0);
  for (int i = 0; i < n_seasons; ++i) {
    for (int j = 0; j < period; ++j) {
      if (std::isnan(x_mat[i * period + j])) {
        continue;
      }
      x_mean[i] += x_mat[i * period + j];
      x_n[i]++;
    }
    x_mean[i] /= x_n[i];
  }
  // stds of subseries
  auto x_std = std::vector<double>(x_mean.size(), 0.0);
  for (size_t i = 0; i < x_std.size(); ++i) {
    if (std::isnan(x_mean[i]) || x_n[i] < 2) {
      x_std[i] = std::numeric_limits<double>::quiet_NaN();
      continue;
    }
    for (int j = 0; j < period; ++j) {
      if (std::isnan(x_mat[i * period + j])) {
        continue;
      }
      double tmp = x_mat[i * period + j] - x_mean[i];
      x_std[i] += tmp * tmp / (x_n[i] - 1);
    }
    x_std[i] = std::sqrt(x_std[i]);
  }
  double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.25);
  return Brent(GuerreroCV, lower, upper, tol, x_mean, x_std);
}

int GroupedArrayFloat32_MinMaxScalerStats(GroupedArrayHandle handle,
                                          float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(MinMaxScalerStats<float>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat64_MinMaxScalerStats(GroupedArrayHandle handle,
                                          double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(MinMaxScalerStats<double>, 2, out, 0);
  return 0;
}

int GroupedArrayFloat32_StandardScalerStats(GroupedArrayHandle handle,
                                            float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(StandardScalerStats<float>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat64_StandardScalerStats(GroupedArrayHandle handle,
                                            double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(StandardScalerStats<double>, 2, out, 0);
  return 0;
}

int GroupedArrayFloat32_RobustIqrScalerStats(GroupedArrayHandle handle,
                                             float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RobustScalerIqrStats<float>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat64_RobustIqrScalerStats(GroupedArrayHandle handle,
                                             double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RobustScalerIqrStats<double>, 2, out, 0);
  return 0;
}

int GroupedArrayFloat32_RobustMadScalerStats(GroupedArrayHandle handle,
                                             float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(RobustScalerMadStats<float>, 2, out, 0);
  return 0;
}
int GroupedArrayFloat64_RobustMadScalerStats(GroupedArrayHandle handle,
                                             double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(RobustScalerMadStats<double>, 2, out, 0);
  return 0;
}

int GroupedArrayFloat32_ScalerTransform(GroupedArrayHandle handle,
                                        const float *stats, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->ScalerTransform(CommonScalerTransform<float>, stats, out);
  return 0;
}
int GroupedArrayFloat64_ScalerTransform(GroupedArrayHandle handle,
                                        const double *stats, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->ScalerTransform(CommonScalerTransform<double>, stats, out);
  return 0;
}

int GroupedArrayFloat32_ScalerInverseTransform(GroupedArrayHandle handle,
                                               const float *stats, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->ScalerTransform(CommonScalerInverseTransform<float>, stats, out);
  return 0;
}
int GroupedArrayFloat64_ScalerInverseTransform(GroupedArrayHandle handle,
                                               const double *stats,
                                               double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->ScalerTransform(CommonScalerInverseTransform<double>, stats, out);
  return 0;
}
