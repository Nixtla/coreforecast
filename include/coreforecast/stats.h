#pragma once

#include "SkipList.h"

#include <algorithm>
#include <span>

#include "common.h"

namespace stats {
// numpy's default linear interpolation. Partially sorts `data` in place.
template <typename T> T Quantile(std::span<T> data, T p) {
  const index_t n = std::ssize(data);
  const T i_plus_g = p * (n - 1);
  const auto i = static_cast<index_t>(i_plus_g);
  const T g = i_plus_g - i;

  auto nth = data.begin() + i;
  std::nth_element(data.begin(), nth, data.end());
  T out = *nth;

  if (g > T{0}) {
    auto min = std::min_element(nth + 1, data.end());
    out += g * (*min - out);
  }
  return out;
}

template <typename T>
T SortedQuantile(OrderedStructs::SkipList::HeadNode<T> &data, T p, index_t n) {
  const T i_plus_g = p * (n - 1);
  const auto i = static_cast<index_t>(i_plus_g);
  const T g = i_plus_g - i;
  T out = data.at(i);
  if (g > T{0.0}) {
    out += g * (data.at(i + 1) - out);
  }
  return out;
}
} // namespace stats
