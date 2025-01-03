#pragma once

#include "SkipList.h"
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <iterator>
#include <limits>
#include <ranges>

namespace stats {
template <std::contiguous_iterator Iterator>
  requires std::totally_ordered<std::iter_value_t<Iterator>>
auto Quantile(Iterator begin, Iterator end, std::iter_value_t<Iterator> p) {
  using T = std::iter_value_t<Iterator>;
  auto range = std::ranges::subrange(begin, end);

  if (range.empty()) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  T i_plus_g = p * (range.size() - 1);
  auto i = static_cast<size_t>(i_plus_g);
  T g = i_plus_g - i;

  auto nth = begin + i;
  std::ranges::nth_element(range, nth);
  T out = *nth;

  if (g > T{0}) {
    auto min = std::ranges::min_element(std::ranges::subrange(nth + 1, end));
    out += g * (*min - out);
  }
  return out;
}

template <typename T>
auto SortedQuantile(OrderedStructs::SkipList::HeadNode<T> &data, T p,
                    size_t n) {
  T i_plus_g = p * (n - 1);
  int i = static_cast<size_t>(i_plus_g);
  T g = i_plus_g - i;
  T out = data.at(i);
  if (g > T{0.0}) {
    out += g * (data.at(i + 1) - out);
  }
  return out;
}
} // namespace stats
