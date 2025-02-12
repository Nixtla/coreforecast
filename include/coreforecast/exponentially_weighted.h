#pragma once

namespace exponentially_weighted {
template <typename T>
inline void MeanTransform(const T *data, int n, T *out, T alpha, ptrdiff_t stride) {
  out[0] = data[0];
  for (int i = 1; i < n; ++i) {
    out[i] = alpha * data[i * stride / sizeof(T)] + (1 - alpha) * out[i - 1];
  }
}
} // namespace exponentially_weighted
