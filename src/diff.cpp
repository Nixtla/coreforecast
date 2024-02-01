#include "diff.h"
#include "kpss.h"
#include "seasonal.h"

template <typename T> inline bool IsConstant(const T *data, int n) {
  for (int i = 1; i < n; ++i) {
    if (data[i] != data[0]) {
      return false;
    }
  }
  return true;
}

template <typename T> void Difference(const T *data, int n, T *out, int d) {
  std::fill(out, out + d, std::numeric_limits<T>::quiet_NaN());
  for (int i = d; i < n; ++i) {
    out[i] = data[i] - data[i - d];
  }
}

template <typename T>
void InvertDifference(const T *data, int n, T *out, T *tails, int d) {
  int upper = std::min(d, n);
  for (int i = 0; i < upper; ++i) {
    out[i] = data[i] + tails[i];
  }
  for (int i = upper; i < n; ++i) {
    out[i] = data[i] + out[i - d];
  }
}

template <typename T> void NumDiffs(const T *x, indptr_t n, T *out, int max_d) {
  // assume there are only NaNs at the start
  indptr_t start_idx = FirstNotNaN(x, n);
  x += start_idx;
  n -= start_idx;
  if (n < 3) {
    *out = 0;
    return;
  }
  constexpr T threshold = 0.463; // alpha = 0.05
  int d = 0;
  int n_lags = std::floor(3 * std::sqrt(n) / 13);
  bool do_diff = KPSS(x, n, n_lags) > threshold;
  std::vector<T> x_vec(n);
  std::copy(x, x + n, x_vec.begin());
  std::vector<T> diff_x(n);
  while (do_diff && d < max_d) {
    ++d;
    Difference(x_vec.data(), x_vec.size(), diff_x.data(), 1);
    if (IsConstant(diff_x.data() + d, diff_x.size() - d)) {
      *out = d;
      return;
    }
    std::copy(diff_x.begin(), diff_x.end(), x_vec.begin());
    if (n > d) {
      // we've taken d differences, so we have d NaNs
      do_diff = KPSS(x_vec.data() + d, n - d, n_lags) > threshold;
    } else {
      do_diff = false;
    }
  }
  *out = d;
  return;
}

template <typename T>
void NumSeasDiffs(const T *x, indptr_t n, T *out, int period, int max_d) {
  // assume there are only NaNs at the start
  indptr_t start_idx = FirstNotNaN(x, n);
  x += start_idx;
  n -= start_idx;
  if (n < 2 * period) {
    *out = 0;
    return;
  }
  constexpr T threshold = 0.64;
  int d = 0;
  bool do_diff = SeasHeuristic(x, n, period) > threshold;
  std::vector<T> x_vec(n);
  std::copy(x, x + n, x_vec.begin());
  std::vector<T> diff_x(n);
  while (do_diff && d < max_d) {
    ++d;
    Difference(x_vec.data(), x_vec.size(), diff_x.data(), period);
    if (IsConstant(diff_x.data() + d * period, n - d * period)) {
      *out = d;
      return;
    }
    std::copy(diff_x.begin(), diff_x.end(), x_vec.begin());
    // we'll have d * period NaNs and we need 2 * period samples for the STL
    if (n > (d + 2) * period && d < max_d) {
      do_diff = SeasHeuristic(x_vec.data() + d * period, n - d * period,
                              period) > threshold;
    } else {
      do_diff = false;
    }
  }
  *out = d;
  return;
}

template <typename T>
void ConditionalDifference(const T *data, int n, T *out, T *apply, int period) {
  if (apply[0] == 0) {
    std::copy(data, data + n, out);
    return;
  }
  Difference(data, n, out, period);
}

template <typename T>
void ConditionalInvertDifference(const T *data, int n, T *out,
                                 T *apply_and_tails, int period) {
  if (apply_and_tails[0] == 0) {
    std::copy(data, data + n, out);
    return;
  }
  InvertDifference(data, n, out, apply_and_tails + 1, period);
}

void Float32_Difference(const float *x, indptr_t n, int d, float *out) {
  Difference<float>(x, n, out, d);
}
void Float64_Difference(const double *x, indptr_t n, int d, double *out) {
  Difference<double>(x, n, out, d);
}

int Float32_NumDiffs(const float *x, indptr_t n, int max_d) {
  float out;
  NumDiffs(x, n, &out, max_d);
  return static_cast<int>(out);
}
int Float64_NumDiffs(const double *x, indptr_t n, int max_d) {
  double out;
  NumDiffs(x, n, &out, max_d);
  return static_cast<int>(out);
}

int Float32_NumSeasDiffs(const float *x, indptr_t n, int period, int max_d) {
  float out;
  NumSeasDiffs(x, n, &out, period, max_d);
  return static_cast<int>(out);
}
int Float64_NumSeasDiffs(const double *x, indptr_t n, int period, int max_d) {
  double out;
  NumSeasDiffs(x, n, &out, period, max_d);
  return static_cast<int>(out);
}

void GroupedArrayFloat32_NumDiffs(GroupedArrayHandle handle, int max_d,
                                  float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(NumDiffs<float>, 1, out, 0, max_d);
}
void GroupedArrayFloat64_NumDiffs(GroupedArrayHandle handle, int max_d,
                                  double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(NumDiffs<double>, 1, out, 0, max_d);
}

void GroupedArrayFloat32_NumSeasDiffs(GroupedArrayHandle handle, int period,
                                      int max_d, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Reduce(NumSeasDiffs<float>, 1, out, 0, period, max_d);
}
void GroupedArrayFloat64_NumSeasDiffs(GroupedArrayHandle handle, int period,
                                      int max_d, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Reduce(NumSeasDiffs<double>, 1, out, 0, period, max_d);
}

void GroupedArrayFloat32_Difference(GroupedArrayHandle handle, int d,
                                    float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->Transform(Difference<float>, 0, out, d);
}
void GroupedArrayFloat64_Difference(GroupedArrayHandle handle, int d,
                                    double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->Transform(Difference<double>, 0, out, d);
}

void GroupedArrayFloat32_InvertDifference(GroupedArrayHandle handle, int d,
                                          float *tails, float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->TransformAndReduce(InvertDifference<float>, 0, out, d, tails, d);
}
void GroupedArrayFloat64_InvertDifference(GroupedArrayHandle handle, int d,
                                          double *tails, double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->TransformAndReduce(InvertDifference<double>, 0, out, d, tails, d);
}

void GroupedArrayFloat32_ConditionalDifference(GroupedArrayHandle handle,
                                               int period, float *apply,
                                               float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->TransformAndReduce(ConditionalDifference<float>, 0, out, 1, apply,
                         period);
}
void GroupedArrayFloat64_ConditionalDifference(GroupedArrayHandle handle,
                                               int period, double *apply,
                                               double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->TransformAndReduce(ConditionalDifference<double>, 0, out, 1, apply,
                         period);
}

void GroupedArrayFloat32_ConditionalInvertDifference(GroupedArrayHandle handle,
                                                     int period,
                                                     float *apply_and_tails,
                                                     float *out) {
  auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
  ga->TransformAndReduce(ConditionalInvertDifference<float>, 0, out, period + 1,
                         apply_and_tails, period);
}
void GroupedArrayFloat64_ConditionalInvertDifference(GroupedArrayHandle handle,
                                                     int period,
                                                     double *apply_and_tails,
                                                     double *out) {
  auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
  ga->TransformAndReduce(ConditionalInvertDifference<double>, 0, out,
                         period + 1, apply_and_tails, period);
}