#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

#include "diff.h"
#include "expanding.h"
#include "exponentially_weighted.h"
#include "grouped_array_functions.h"
#include "lag.h"
#include "rolling.h"
#include "scalers.h"
#include "seasonal.h"

using indptr_t = int32_t;

template <typename T> inline indptr_t FirstNotNaN(const T *data, indptr_t n) {
  indptr_t i = 0;
  while (std::isnan(data[i]) && i < n) {
    ++i;
  }
  return i;
}

template <typename T>
inline indptr_t FirstNotNaN(const T *data, indptr_t n, T *out) {
  indptr_t i = 0;
  while (std::isnan(data[i]) && i < n) {
    out[i] = std::numeric_limits<T>::quiet_NaN();
    ++i;
  }
  return i;
}

template <typename T> inline void SkipLags(T *out, int n, int lag) {
  int replacements = std::min(lag, n);
  for (int i = 0; i < replacements; ++i) {
    out[i] = std::numeric_limits<T>::quiet_NaN();
  }
}

template <class T> class GroupedArray {
private:
  const T *data_;
  const indptr_t *indptr_;
  int n_groups_;
  int num_threads_;

public:
  GroupedArray(const T *data, const indptr_t *indptr, int n_indptr,
               int num_threads)
      : data_(data), indptr_(indptr), n_groups_(n_indptr - 1),
        num_threads_(num_threads) {}

  template <typename Func> void Parallelize(Func f) const noexcept {
    std::vector<std::thread> threads;
    int groups_per_thread = n_groups_ / num_threads_;
    int remainder = n_groups_ % num_threads_;
    for (int t = 0; t < num_threads_; ++t) {
      int start_group = t * groups_per_thread + std::min(t, remainder);
      int end_group = (t + 1) * groups_per_thread + std::min(t + 1, remainder);
      threads.emplace_back(f, start_group, end_group);
    }
    for (auto &thread : threads) {
      thread.join();
    }
  }

  template <typename Func, typename... Args>
  void Reduce(Func f, int n_out, T *out, int lag,
              Args &&...args) const noexcept {
    Parallelize([data = data_, indptr = indptr_, &f, n_out, out, lag,
                 &args...](int start_group, int end_group) {
      for (int i = start_group; i < end_group; ++i) {
        indptr_t start = indptr[i];
        indptr_t end = indptr[i + 1];
        indptr_t n = end - start;
        indptr_t start_idx = FirstNotNaN(data + start, n);
        if (start_idx + lag >= n) {
          for (int j = 0; j < n_out; ++j) {
            out[n_out * i + j] = std::numeric_limits<T>::quiet_NaN();
          }
          continue;
        }
        start += start_idx;
        f(data + start, n - start_idx - lag, out + n_out * i,
          std::forward<Args>(args)...);
      }
    });
  }

  template <typename Func, typename... Args>
  void VariableReduce(Func f, const indptr_t *indptr_out, T *out,
                      Args &&...args) const noexcept {
    Parallelize([data = data_, indptr = indptr_, &f, indptr_out, out,
                 &args...](int start_group, int end_group) {
      for (int i = start_group; i < end_group; ++i) {
        indptr_t start = indptr[i];
        indptr_t end = indptr[i + 1];
        indptr_t n = end - start;
        indptr_t out_n = indptr_out[i + 1] - indptr_out[i];
        f(data + start, n, out + indptr_out[i], out_n,
          std::forward<Args>(args)...);
      }
    });
  }

  template <typename Func>
  void ScalerTransform(Func f, const T *stats, T *out) const noexcept {
    Parallelize([data = data_, indptr = indptr_, &f, stats,
                 out](int start_group, int end_group) {
      for (int i = start_group; i < end_group; ++i) {
        indptr_t start = indptr[i];
        indptr_t end = indptr[i + 1];
        T offset = stats[2 * i];
        T scale = stats[2 * i + 1];
        if (std::abs(scale) < std::numeric_limits<T>::epsilon()) {
          scale = static_cast<T>(1.0);
        }
        for (indptr_t j = start; j < end; ++j) {
          out[j] = f(data[j], offset, scale);
        }
      }
    });
  }

  template <typename Func, typename... Args>
  void Transform(Func f, int lag, T *out, Args &&...args) const noexcept {
    Parallelize([data = data_, indptr = indptr_, &f, lag, out,
                 &args...](int start_group, int end_group) {
      for (int i = start_group; i < end_group; ++i) {
        indptr_t start = indptr[i];
        indptr_t end = indptr[i + 1];
        indptr_t n = end - start;
        indptr_t start_idx = FirstNotNaN(data + start, n, out + start);
        SkipLags(out + start + start_idx, n - start_idx, lag);
        if (start_idx + lag >= n) {
          continue;
        }
        start += start_idx;
        f(data + start, n - start_idx - lag, out + start + lag,
          std::forward<Args>(args)...);
      }
    });
  }

  template <typename Func>
  void VariableTransform(Func f, const indptr_t *params,
                         T *out) const noexcept {
    Parallelize([data = data_, indptr = indptr_, &f, params,
                 out](int start_group, int end_group) {
      for (int i = start_group; i < end_group; ++i) {
        indptr_t start = indptr[i];
        indptr_t end = indptr[i + 1];
        indptr_t n = end - start;
        indptr_t start_idx = FirstNotNaN(data + start, n, out + start);
        if (start_idx >= n) {
          continue;
        }
        start += start_idx;
        f(data + start, n - start_idx, params[i], out + start);
      }
    });
  }

  template <typename Func, typename... Args>
  void TransformAndReduce(Func f, int lag, T *out, int n_agg, T *agg,
                          Args &&...args) const noexcept {
    Parallelize([data = data_, indptr = indptr_, &f, lag, out, n_agg, agg,
                 &args...](int start_group, int end_group) {
      for (int i = start_group; i < end_group; ++i) {
        indptr_t start = indptr[i];
        indptr_t end = indptr[i + 1];
        indptr_t n = end - start;
        indptr_t start_idx = FirstNotNaN(data + start, n, out + start);
        SkipLags(out + start + start_idx, n - start_idx, lag);
        if (start_idx + lag >= n)
          continue;
        start += start_idx;
        f(data + start, n - start_idx - lag, out + start + lag, agg + i * n_agg,
          std::forward<Args>(args)...);
      }
    });
  }

  template <typename Func>
  void Zip(Func f, const GroupedArray<T> &other, const indptr_t *out_indptr,
           T *out) const noexcept {
    Parallelize([data = data_, indptr = indptr_, &f, other_data = other.data_,
                 other_indptr = other.indptr_, out_indptr,
                 out](int start_group, int end_group) {
      for (int i = start_group; i < end_group; ++i) {
        indptr_t start = indptr[i];
        indptr_t end = indptr[i + 1];
        indptr_t n = end - start;
        indptr_t other_start = other_indptr[i];
        indptr_t other_end = other_indptr[i + 1];
        indptr_t other_n = other_end - other_start;
        f(data + start, n, other_data + other_start, other_n,
          out + out_indptr[i]);
      }
    });
  }

  // lag
  void LagTransform(int k, T *out) { Transform(lag::LagTransform<T>, k, out); }

  // manipulation
  void IndexFromEnd(int k, T *out) {
    Transform(grouped_array_functions::IndexFromEnd<T>, k, out);
  }
  void Head(int k, T *out) {
    Reduce(grouped_array_functions::Head<T>, k, out, 0, k);
  }
  void Tail(int k, T *out) {
    Reduce(grouped_array_functions::Tail<T>, k, out, 0, k);
  }
  void Tails(const indptr_t *out_indptr, T *out_data) {
    VariableReduce(grouped_array_functions::Tail<T>, out_indptr, out_data);
  }
  void Append(GroupedArray<T> &other, const indptr_t *out_indptr, T *out_data) {
    Zip(grouped_array_functions::Append<T>, other, out_indptr, out_data);
  }

  // rolling
  void RollingMeanTransform(int lag, int window_size, int min_samples, T *out) {
    Transform(rolling::MeanTransform<T>, lag, out, window_size, min_samples);
  }
  void RollingStdTransform(int lag, int window_size, int min_samples, T *out) {
    Transform(rolling::StdTransform<T>, lag, out, window_size, min_samples);
  }
  void RollingMinTransform(int lag, int window_size, int min_samples, T *out) {
    Transform(rolling::MinTransform<T>, lag, out, window_size, min_samples);
  }
  void RollingMaxTransform(int lag, int window_size, int min_samples, T *out) {
    Transform(rolling::MaxTransform<T>, lag, out, window_size, min_samples);
  }
  void RollingQuantileTransform(int lag, T p, int window_size, int min_samples,
                                T *out) {
    Transform(rolling::QuantileTransform<T>, lag, out, window_size, min_samples,
              p);
  }
  void RollingMeanUpdate(int lag, int window_size, int min_samples, T *out) {
    Reduce(rolling::MeanUpdate<T>(), 1, out, lag, window_size, min_samples);
  }
  void RollingStdUpdate(int lag, int window_size, int min_samples, T *out) {
    Reduce(rolling::StdUpdate<T>(), 1, out, lag, window_size, min_samples);
  }
  void RollingMinUpdate(int lag, int window_size, int min_samples, T *out) {
    Reduce(rolling::MinUpdate<T>(), 1, out, lag, window_size, min_samples);
  }
  void RollingMaxUpdate(int lag, int window_size, int min_samples, T *out) {
    Reduce(rolling::MaxUpdate<T>(), 1, out, lag, window_size, min_samples);
  }
  void RollingQuantileUpdate(int lag, T p, int window_size, int min_samples,
                             T *out) {
    Reduce(rolling::QuantileUpdate<T>(), 1, out, lag, window_size, min_samples,
           p);
  }

  // seasonal rolling
  void SeasonalRollingMeanTransform(int lag, int season_length, int window_size,
                                    int min_samples, T *out) {
    Transform(rolling::SeasonalMeanTransform<T>(), lag, out, season_length,
              window_size, min_samples);
  }
  void SeasonalRollingStdTransform(int lag, int season_length, int window_size,
                                   int min_samples, T *out) {
    Transform(rolling::SeasonalStdTransform<T>(), lag, out, season_length,
              window_size, min_samples);
  }
  void SeasonalRollingMinTransform(int lag, int season_length, int window_size,
                                   int min_samples, T *out) {
    Transform(rolling::SeasonalMinTransform<T>(), lag, out, season_length,
              window_size, min_samples);
  }
  void SeasonalRollingMaxTransform(int lag, int season_length, int window_size,
                                   int min_samples, T *out) {
    Transform(rolling::SeasonalMaxTransform<T>(), lag, out, season_length,
              window_size, min_samples);
  }
  void SeasonalRollingQuantileTransform(int lag, int season_length, T p,
                                        int window_size, int min_samples,
                                        T *out) {
    Transform(rolling::SeasonalQuantileTransform<T>(), lag, out, season_length,
              window_size, min_samples, p);
  }
  void SeasonalRollingMeanUpdate(int lag, int season_length, int window_size,
                                 int min_samples, T *out) {
    Reduce(rolling::SeasonalMeanUpdate<T>(), 1, out, lag, season_length,
           window_size, min_samples);
  }
  void SeasonalRollingStdUpdate(int lag, int season_length, int window_size,
                                int min_samples, T *out) {
    Reduce(rolling::SeasonalStdUpdate<T>(), 1, out, lag, season_length,
           window_size, min_samples);
  }
  void SeasonalRollingMinUpdate(int lag, int season_length, int window_size,
                                int min_samples, T *out) {
    Reduce(rolling::SeasonalMinUpdate<T>(), 1, out, lag, season_length,
           window_size, min_samples);
  }
  void SeasonalRollingMaxUpdate(int lag, int season_length, int window_size,
                                int min_samples, T *out) {
    Reduce(rolling::SeasonalMaxUpdate<T>(), 1, out, lag, season_length,
           window_size, min_samples);
  }
  void SeasonalRollingQuantileUpdate(int lag, int season_length, T p,
                                     int window_size, int min_samples, T *out) {
    Reduce(rolling::SeasonalQuantileUpdate<T>(), 1, out, lag, season_length,
           window_size, min_samples, p);
  }

  // expanding
  void ExpandingMeanTransform(int lag, T *out, T *agg) {
    TransformAndReduce(expanding::MeanTransform<T>, lag, out, 1, agg);
  }
  void ExpandingStdTransform(int lag, T *out, T *agg) {
    TransformAndReduce(expanding::StdTransform<T>, lag, out, 3, agg);
  }
  void ExpandingMinTransform(int lag, T *out) {
    Transform(expanding::MinTransform<T>(), lag, out);
  }
  void ExpandingMaxTransform(int lag, T *out) {
    Transform(expanding::MaxTransform<T>(), lag, out);
  }
  void ExpandingQuantileTransform(int lag, T p, T *out) {
    Transform(expanding::MaxTransform<T>(), lag, out, p);
  }
  void ExpandingQuantileUpdate(int lag, T p, T *out) {
    Reduce(expanding::QuantileUpdate<T>, 1, out, lag, p);
  }

  // exponentially weighted
  void ExponentiallyWeightedMeanTransform(int lag, T alpha, T *out) {
    Transform(exponentially_weighted::MeanTransform<T>, lag, out, alpha);
  }

  // scalers
  void MinMaxScalerStats(T *out) {
    Reduce(scalers::MinMaxScalerStats<T>, 2, out, 0);
  }
  void StandardScalerStats(T *out) {
    Reduce(scalers::StandardScalerStats<T>, 2, out, 0);
  }
  void RobustIqrScalerStats(T *out) {
    Reduce(scalers::RobustScalerIqrStats<T>, 2, out, 0);
  }
  void RobustMadScalerStats(T *out) {
    Reduce(scalers::RobustScalerMadStats<T>, 2, out, 0);
  }
  void ApplyScaler(const T *stats, T *out) {
    ScalerTransform(scalers::CommonScalerTransform<T>, stats, out);
  }
  void InvertScaler(const T *stats, T *out) {
    ScalerTransform(scalers::CommonScalerInverseTransform<T>, stats, out);
  }
  void BoxCoxLambdaGuerrero(int period, T lower, T upper, T *out) {
    Reduce(scalers::BoxCoxLambdaGuerrero<T>, 2, out, 0, period, lower, upper);
  }
  void BoxCoxLambdaLogLik(int period, T lower, T upper, T *out) {
    Reduce(scalers::BoxCoxLambdaLogLik<T>, 2, out, 0, period, lower, upper);
  }
  void BoxCoxTransform(const T *lambdas, T *out) {
    ScalerTransform(scalers::BoxCoxTransform<T>, lambdas, out);
  }
  void BoxCoxInverseTransform(const T *lambdas, T *out) {
    ScalerTransform(scalers::BoxCoxInverseTransform<T>, lambdas, out);
  }

  // differences
  void NumDiffs(int max_d, T *out) {
    Reduce(diff::NumDiffs<T>, 1, out, 0, max_d);
  }
  void NumSeasDiffs(int period, int max_d, T *out) {
    Reduce(diff::NumSeasDiffs<T>, 1, out, 0, period, max_d);
  }
  void NumSeasDiffsPeriods(int max_d, T *periods_and_out) {
    Reduce(diff::NumSeasDiffsPeriods<T>, 2, periods_and_out, 0, max_d);
  }
  void Period(size_t max_lag, T *out) {
    Reduce(seasonal::GreatestAutocovariance<T>, 1, out, 0, max_lag);
  }
  void Difference(int d, T *out) {
    Transform(seasonal::Difference<T>, 0, out, d);
  }
  void Differences(const indptr_t *ds, T *out) {
    VariableTransform(diff::Differences<T>, ds, out);
  }
  void InvertDifferences(const GroupedArray &other, const indptr_t out_indptr,
                         T *out_data) {
    Zip(diff::InvertDifference<T>, other, out_indptr, out_data);
  }
};
