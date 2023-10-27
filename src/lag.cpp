#include "lag.h"

template <typename T> inline void LagTransform(const T *data, int n, T *out) {
  std::copy(data, data + n, out);
}

int GroupedArray_LagTransform(GroupedArrayHandle handle, int data_type, int lag,
                              void *out) {
  if (data_type == DTYPE_FLOAT32) {
    auto ga = reinterpret_cast<GroupedArray<float> *>(handle);
    ga->Transform(LagTransform<float>, lag, static_cast<float *>(out));
  } else {
    auto ga = reinterpret_cast<GroupedArray<double> *>(handle);
    ga->Transform(LagTransform<double>, lag, static_cast<double *>(out));
  }
  return 0;
}
