#include "seasonal_rolling.h"

template <typename Func, typename T>
inline void SeasonalRollingTransform(Func rolling_tfm, const T *data, int n,
                                     T *out, int season_length, int window_size,
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
    rolling_tfm(season_data, season_n, season_out, window_size, min_samples);
    for (int j = 0; j < season_n; ++j) {
      out[i + j * season_length] = season_out[j];
    }
  }
  delete[] season_data;
  delete[] season_out;
}

template <typename T> struct SeasonalRollingMeanTransform {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingTransform(RollingMeanTransform<T>, data, n, out,
                             season_length, window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingStdTransform {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingTransform(RollingStdTransform<T>, data, n, out,
                             season_length, window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingMinTransform {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingTransform(RollingMinTransform<T>(), data, n, out,
                             season_length, window_size, min_samples);
  }
};

template <typename T> struct SeasonalRollingMaxTransform {
  void operator()(const T *data, int n, T *out, int season_length,
                  int window_size, int min_samples) {
    SeasonalRollingTransform(RollingMaxTransform<T>(), data, n, out,
                             season_length, window_size, min_samples);
  }
};

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

int GroupedArray_SeasonalRollingMeanTransform(GroupedArrayHandle handle,
                                              int data_type, int lag,
                                              int season_length,
                                              int window_size, int min_samples,
                                              void *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->Transform(SeasonalRollingMeanTransform<float>(), lag,
                  static_cast<float *>(out), season_length, window_size,
                  min_samples);
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->Transform(SeasonalRollingMeanTransform<double>(), lag,
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
