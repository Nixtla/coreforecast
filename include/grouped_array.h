#pragma once

#include <algorithm>
#include <utility>

#include "export.h"
#include "types.h"

template <class T> class GroupedArray {
private:
  const T *data_;
  indptr_t n_data_;
  const indptr_t *indptr_;
  int n_groups_;
  int num_threads_;

public:
  GroupedArray(const T *data, indptr_t n_data, const indptr_t *indptr,
               int n_indptr, int num_threads);
  ~GroupedArray();

  template <typename Func, typename... Args>
  void Reduce(Func f, int n_out, T *out, int lag,
              Args &&...args) const noexcept;

  template <typename Func>
  void ScalerTransform(Func f, const T *stats, T *out) const noexcept;

  template <typename Func, typename... Args>
  void Transform(Func f, int lag, T *out, Args &&...args) const noexcept;

  template <typename Func>
  void TransformAndReduce(Func f, int lag, T *out, int n_agg,
                          T *agg) const noexcept;
};

extern "C" {
DLL_EXPORT int GroupedArray_Create(const void *data, indptr_t n_data,
                                   indptr_t *indptr, indptr_t n_groups,
                                   int num_threads, int data_type,
                                   GroupedArrayHandle *out);

DLL_EXPORT int GroupedArray_Delete(GroupedArrayHandle handle, int data_type);
}
