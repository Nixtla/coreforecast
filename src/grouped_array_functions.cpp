#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "diff.h"
#include "expanding.h"
#include "exponentially_weighted.h"
#include "grouped_array_functions.h"
#include "lag.h"
#include "nb.h"
#include "rolling.h"
#include "scalers.h"
#include "seasonal.h"

void init_ga(nb::module_ &m) {
  nb::module_ ga = m.def_submodule("ga");
  // Float32 Methods
  // Lag
  ga.def("lag", [](const Vector<float> data, const Vector<indptr_t> indptr,
                   int num_threads, int lag, Vector<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Transform(LagTransform<float>, lag, out.data());
  });

  // Manipulation
  ga.def("index_from_end", [](const Vector<float> data,
                              const Vector<indptr_t> indptr, int num_threads,
                              int k, Vector<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Reduce(IndexFromEnd<float>, 1, out.data(), 0, k);
  });
  ga.def("head", [](const Vector<float> data, const Vector<indptr_t> indptr,
                    int num_threads, int k, Vector<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Reduce(Head<float>, k, out.data(), 0, k);
  });
  ga.def("tail", [](const Vector<float> data, const Vector<indptr_t> indptr,
                    int num_threads, int k, Vector<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Reduce(Tail<float>, k, out.data(), 0, k);
  });
  ga.def("append", [](const Vector<float> data, const Vector<indptr_t> indptr,
                      int num_threads, const Vector<float> other_data,
                      const Vector<indptr_t> other_indptr,
                      Vector<float> out_data, Vector<indptr_t> out_indptr) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    auto other =
        GroupedArray<float>(other_data.data(), other_indptr.data(),
                            static_cast<int>(other_indptr.size()), num_threads);
    ga.Zip(Append<float>, other, out_indptr.data(), out_data.data());
  });
  ga.def("tails", [](const Vector<float> data, const Vector<indptr_t> indptr,
                     int num_threads, Vector<float> out_data,
                     Vector<indptr_t> out_indptr) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.VariableReduce(Tail<float>, out_indptr.data(), out_data.data());
  });

  // Lag
  // Rolling
  ga.def("rolling_mean_transform", [](const Vector<float> data,
                                      const Vector<indptr_t> indptr,
                                      int num_threads, int lag, int window_size,
                                      int min_samples, Vector<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Transform(RollingMeanTransform<float>, lag, out.data(), window_size,
                 min_samples);
  });
  ga.def("rolling_std_transform", [](const Vector<float> data,
                                     const Vector<indptr_t> indptr,
                                     int num_threads, int lag, int window_size,
                                     int min_samples, Vector<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Transform(RollingStdTransform<float>, lag, out.data(), window_size,
                 min_samples);
  });
  ga.def("rolling_min_transform", [](const Vector<float> data,
                                     const Vector<indptr_t> indptr,
                                     int num_threads, int lag, int window_size,
                                     int min_samples, Vector<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Transform(RollingMinTransform<float>, lag, out.data(), window_size,
                 min_samples);
  });
  ga.def("rolling_max_transform", [](const Vector<float> data,
                                     const Vector<indptr_t> indptr,
                                     int num_threads, int lag, int window_size,
                                     int min_samples, Vector<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Transform(RollingMaxTransform<float>, lag, out.data(), window_size,
                 min_samples);
  });
  ga.def("rolling_quantile_transform",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, int lag, float p, int window_size, int min_samples,
            Vector<float> out) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.Transform(RollingQuantileTransform<float>, lag, out.data(),
                        window_size, min_samples, p);
         });
  ga.def("rolling_mean_update", [](const Vector<float> data,
                                   const Vector<indptr_t> indptr,
                                   int num_threads, int lag, int window_size,
                                   int min_samples, Vector<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Reduce(RollingMeanUpdate<float>(), 1, out.data(), lag, window_size,
              min_samples);
  });
  ga.def("rolling_std_update", [](const Vector<float> data,
                                  const Vector<indptr_t> indptr,
                                  int num_threads, int lag, int window_size,
                                  int min_samples, Vector<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Reduce(RollingStdUpdate<float>(), 1, out.data(), lag, window_size,
              min_samples);
  });
  ga.def("rolling_min_update", [](const Vector<float> data,
                                  const Vector<indptr_t> indptr,
                                  int num_threads, int lag, int window_size,
                                  int min_samples, Vector<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Reduce(RollingMinUpdate<float>(), 1, out.data(), lag, window_size,
              min_samples);
  });
  ga.def("rolling_max_update", [](const Vector<float> data,
                                  const Vector<indptr_t> indptr,
                                  int num_threads, int lag, int window_size,
                                  int min_samples, Vector<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Reduce(RollingMaxUpdate<float>(), 1, out.data(), lag, window_size,
              min_samples);
  });
  ga.def("rolling_quantile_update",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, int lag, float p, int window_size, int min_samples,
            Vector<float> out) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.Reduce(RollingQuantileUpdate<float>(), 1, out.data(), lag,
                     window_size, min_samples, p);
         });

  // Seasonal rolling
  ga.def("seasonal_rolling_mean_transform",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, int lag, int season_length, int window_size,
            int min_samples, Vector<float> out) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.Transform(SeasonalRollingMeanTransform<float>(), lag, out.data(),
                        season_length, window_size, min_samples);
         });
  ga.def("seasonal_rolling_std_transform",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, int lag, int season_length, int window_size,
            int min_samples, Vector<float> out) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.Transform(SeasonalRollingStdTransform<float>(), lag, out.data(),
                        season_length, window_size, min_samples);
         });
  ga.def("seasonal_rolling_min_transform",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, int lag, int season_length, int window_size,
            int min_samples, Vector<float> out) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.Transform(SeasonalRollingMinTransform<float>(), lag, out.data(),
                        season_length, window_size, min_samples);
         });
  ga.def("seasonal_rolling_max_transform",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, int lag, int season_length, int window_size,
            int min_samples, Vector<float> out) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.Transform(SeasonalRollingMaxTransform<float>(), lag, out.data(),
                        season_length, window_size, min_samples);
         });
  ga.def("seasonal_rolling_quantile_transform",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, int lag, int season_length, float p,
            int window_size, int min_samples, Vector<float> out) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.Transform(SeasonalRollingQuantileTransform<float>(), lag,
                        out.data(), season_length, window_size, min_samples, p);
         });
  ga.def("seasonal_rolling_mean_update",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, int lag, int season_length, int window_size,
            int min_samples, Vector<float> out) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.Reduce(SeasonalRollingMeanUpdate<float>(), 1, out.data(), lag,
                     season_length, window_size, min_samples);
         });
  ga.def("seasonal_rolling_std_update",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, int lag, int season_length, int window_size,
            int min_samples, Vector<float> out) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.Reduce(SeasonalRollingStdUpdate<float>(), 1, out.data(), lag,
                     season_length, window_size, min_samples);
         });
  ga.def("seasonal_rolling_min_update",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, int lag, int season_length, int window_size,
            int min_samples, Vector<float> out) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.Reduce(SeasonalRollingMinUpdate<float>(), 1, out.data(), lag,
                     season_length, window_size, min_samples);
         });
  ga.def("seasonal_rolling_max_update",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, int lag, int season_length, int window_size,
            int min_samples, Vector<float> out) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.Reduce(SeasonalRollingMaxUpdate<float>(), 1, out.data(), lag,
                     season_length, window_size, min_samples);
         });
  ga.def("seasonal_rolling_quantile_update",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, int lag, int season_length, float p,
            int window_size, int min_samples, Vector<float> out) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.Reduce(SeasonalRollingQuantileUpdate<float>(), 1, out.data(), lag,
                     season_length, window_size, min_samples, p);
         });

  // Expanding
  ga.def("expanding_mean_transform",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, int lag, Vector<float> out, Vector<float> agg) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.TransformAndReduce(ExpandingMeanTransform<float>, lag, out.data(),
                                 1, agg.data());
         });
  ga.def("expanding_std_transform",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, int lag, Vector<float> out, Matrix<float> agg) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.TransformAndReduce(ExpandingMeanTransform<float>, lag, out.data(),
                                 3, agg.data());
         });
  ga.def("expanding_min_transform",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, int lag, Vector<float> out) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.Transform(ExpandingMinTransform<float>(), lag, out.data());
         });
  ga.def("expanding_max_transform",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, int lag, Vector<float> out) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.Transform(ExpandingMaxTransform<float>(), lag, out.data());
         });
  ga.def("expanding_quantile_transform",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, int lag, float p, Vector<float> out) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.Transform(ExpandingQuantileTransform<float>, lag, out.data(), p);
         });
  ga.def("expanding_quantile_update",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, int lag, float p, Vector<float> out) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.Reduce(ExpandingQuantileUpdate<float>, 1, out.data(), lag, p);
         });

  // Exponentially weighted
  ga.def("exponentially_weighted_mean_transform",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, int lag, float alpha, Vector<float> out) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.Transform(ExponentiallyWeightedMeanTransform<float>, lag,
                        out.data(), alpha);
         });

  // Scalers
  ga.def("min_max_scaler_stats", [](const Vector<float> data,
                                    const Vector<indptr_t> indptr,
                                    int num_threads, Matrix<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Reduce(MinMaxScalerStats<float>, 2, out.data(), 0);
  });
  ga.def("standard_scaler_stats", [](const Vector<float> data,
                                     const Vector<indptr_t> indptr,
                                     int num_threads, Matrix<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Reduce(StandardScalerStats<float>, 2, out.data(), 0);
  });
  ga.def("robust_iqr_scaler_stats", [](const Vector<float> data,
                                       const Vector<indptr_t> indptr,
                                       int num_threads, Matrix<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Reduce(RobustScalerIqrStats<float>, 2, out.data(), 0);
  });
  ga.def("robust_mad_scaler_stats", [](const Vector<float> data,
                                       const Vector<indptr_t> indptr,
                                       int num_threads, Matrix<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Reduce(RobustScalerMadStats<float>, 2, out.data(), 0);
  });
  ga.def("scaler_transform", [](const Vector<float> data,
                                const Vector<indptr_t> indptr, int num_threads,
                                const Matrix<float> stats, Vector<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.ScalerTransform(CommonScalerTransform<float>, stats.data(), out.data());
  });
  ga.def("scaler_inverse_transform",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, const Matrix<float> stats, Vector<float> out) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.ScalerTransform(CommonScalerInverseTransform<float>, stats.data(),
                              out.data());
         });
  ga.def("boxcox_lambda_guerrero", [](const Vector<float> data,
                                      const Vector<indptr_t> indptr,
                                      int num_threads, int period, float lower,
                                      float upper, Matrix<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Reduce(BoxCoxLambda_Guerrero<float>, 2, out.data(), 0, period, lower,
              upper);
  });
  ga.def("boxcox_lambda_loglik", [](const Vector<float> data,
                                    const Vector<indptr_t> indptr,
                                    int num_threads, float lower, float upper,
                                    Matrix<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Reduce(BoxCoxLambda_LogLik<float>, 2, out.data(), 0, lower, upper);
  });
  ga.def("boxcox_transform", [](const Vector<float> data,
                                const Vector<indptr_t> indptr, int num_threads,
                                const Matrix<float> lambdas,
                                Vector<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.ScalerTransform(BoxCoxTransform<float>, lambdas.data(), out.data());
  });
  ga.def("boxcox_inverse_transform",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, const Matrix<float> lambdas, Vector<float> out) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           ga.ScalerTransform(BoxCoxInverseTransform<float>, lambdas.data(),
                              out.data());
         });

  // Differences
  ga.def("num_diffs", [](const Vector<float> data,
                         const Vector<indptr_t> indptr, int num_threads,
                         int max_d, Vector<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Reduce(NumDiffs<float>, 1, out.data(), 0, max_d);
  });
  ga.def("num_seas_diffs", [](const Vector<float> data,
                              const Vector<indptr_t> indptr, int num_threads,
                              int period, int max_d, Vector<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Reduce(NumSeasDiffs<float>, 1, out.data(), 0, period, max_d);
  });
  ga.def("num_seas_diffs_periods", [](const Vector<float> data,
                                      const Vector<indptr_t> indptr,
                                      int num_threads, int max_d,
                                      Matrix<float> periods_and_out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Reduce(NumSeasDiffsPeriods<float>, 2, periods_and_out.data(), 0, max_d);
  });
  ga.def("period", [](const Vector<float> data, const Vector<indptr_t> indptr,
                      int num_threads, int max_lag, Vector<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Reduce(GreatestAutocovariance<float>, 1, out.data(), 0, max_lag);
  });
  ga.def("difference", [](const Vector<float> data,
                          const Vector<indptr_t> indptr, int num_threads, int d,
                          Vector<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.Transform(Difference<float>, 0, out.data(), d);
  });
  ga.def("differences", [](const Vector<float> data,
                           const Vector<indptr_t> indptr, int num_threads,
                           const Vector<indptr_t> ds, Vector<float> out) {
    auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                  static_cast<int>(indptr.size()), num_threads);
    ga.VariableTransform(Differences<float>, ds.data(), out.data());
  });
  ga.def("invert_differences",
         [](const Vector<float> data, const Vector<indptr_t> indptr,
            int num_threads, const Vector<float> other_data,
            const Vector<indptr_t> other_indptr,
            const Vector<indptr_t> out_indptr, Vector<float> out_data) {
           auto ga = GroupedArray<float>(data.data(), indptr.data(),
                                         static_cast<int>(indptr.size()),
                                         num_threads);
           auto tails_ga = GroupedArray<float>(
               other_data.data(), other_indptr.data(),
               static_cast<int>(other_indptr.size()), num_threads);
           ga.Zip(InvertDifference<float>, tails_ga, out_indptr.data(),
                  out_data.data());
         });
}
