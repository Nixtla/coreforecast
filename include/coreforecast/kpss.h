#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <numeric>
#include <span>

#include "common.h"

template <typename T> T KPSS(std::span<const T> x, index_t lags) {
  const index_t n = std::ssize(x);
  const Eigen::Map<const Eigen::VectorX<T>> v(x.data(), n);
  T mean = v.mean();
  Eigen::VectorX<T> resids = v.array() - mean;
  Eigen::VectorX<T> cresids(n);
  std::partial_sum(resids.begin(), resids.end(), cresids.begin());
  T eta = cresids.squaredNorm() / (n * n);
  T s = resids.array().square().sum();
  for (index_t i = 1; i < lags + 1; ++i) {
    T tmp = resids.head(n - i).dot(resids.tail(n - i));
    s += 2 * tmp * (1.0 - (i / (lags + 1.0)));
  }
  return n * eta / s;
}
