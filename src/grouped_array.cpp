#include "grouped_array.h"
#include "types.h"

template <class T>
GroupedArray<T>::GroupedArray(const T *data, indptr_t n_data,
                              const indptr_t *indptr, int n_indptr,
                              int num_threads)
    : data_(data), n_data_(n_data), indptr_(indptr), n_groups_(n_indptr - 1),
      num_threads_(num_threads) {}

template <class T> GroupedArray<T>::~GroupedArray() {}

template <class T>
template <typename Func, typename... Args>
void GroupedArray<T>::Reduce(Func f, int n_out, T *out, int lag,
                             Args &&...args) const noexcept {
#pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int i = 0; i < n_groups_; ++i) {
    indptr_t start = indptr_[i];
    indptr_t end = indptr_[i + 1];
    indptr_t n = end - start;
    if (lag >= n)
      continue;
    indptr_t start_idx = FirstNotNaN(data_ + start, n);
    if (start_idx == n)
      continue;
    f(data_ + start + start_idx, n - start_idx - lag, out + n_out * i,
      std::forward<Args>(args)...);
  }
}

template <class T>
template <typename Func>
void GroupedArray<T>::ScalerTransform(Func f, const T *stats,
                                      T *out) const noexcept {
#pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int i = 0; i < n_groups_; ++i) {
    indptr_t start = indptr_[i];
    indptr_t end = indptr_[i + 1];
    T offset = stats[2 * i];
    T scale = stats[2 * i + 1];
    if (std::abs(scale) < std::numeric_limits<T>::epsilon()) {
      scale = static_cast<T>(1.0);
    }
    for (indptr_t j = start; j < end; ++j) {
      out[j] = f(data_[j], scale, offset);
    }
  }
}

template <class T>
template <typename Func, typename... Args>
void GroupedArray<T>::Transform(Func f, int lag, T *out,
                                Args &&...args) const noexcept {
#pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int i = 0; i < n_groups_; ++i) {
    indptr_t start = indptr_[i];
    indptr_t end = indptr_[i + 1];
    indptr_t n = end - start;
    indptr_t start_idx = FirstNotNaN(data_ + start, n);
    if (start_idx == n) {
      continue;
    }
    start += start_idx;
    f(data_ + start, n - start_idx - lag, out + start + lag,
      std::forward<Args>(args)...);
  }
}

template <class T>
template <typename Func>
void GroupedArray<T>::TransformAndReduce(Func f, int lag, T *out, int n_agg,
                                         T *agg) const noexcept {
#pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int i = 0; i < n_groups_; ++i) {
    indptr_t start = indptr_[i];
    indptr_t end = indptr_[i + 1];
    indptr_t n = end - start;
    indptr_t start_idx = FirstNotNaN(data_ + start, n);
    if (start_idx == n) {
      continue;
    }
    start += start_idx;
    f(data_ + start, n - start_idx - lag, out + start + lag, agg + i * n_agg);
  }
}

int GroupedArray_Create(const void *data, indptr_t n_data, indptr_t *indptr,
                        indptr_t n_groups, int num_threads, int data_type,
                        GroupedArrayHandle *out) {
  if (data_type == DTYPE_FLOAT32) {
    *out = new GroupedArray<float>(static_cast<const float *>(data), n_data,
                                   indptr, n_groups, num_threads);
  } else {
    *out = new GroupedArray<double>(static_cast<const double *>(data), n_data,
                                    indptr, n_groups, num_threads);
  }
  return 0;
}

int GroupedArray_Delete(GroupedArrayHandle handle, int data_type) {
  if (data_type == DTYPE_FLOAT32) {
    delete reinterpret_cast<GroupedArray<float> *>(handle);
  } else {
    delete reinterpret_cast<GroupedArray<double> *>(handle);
  }
  return 0;
}
