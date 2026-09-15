#pragma once

// The scheduler the grouped operations run through. Nothing here may depend on
// Python or pybind11: the GIL is released on the binding side, so this file is
// compiled into the C++ tests on its own and the sanitizers see the parallel
// path without an interpreter in the way.

#include <algorithm>
#include <exception>
#include <thread>
#include <vector>

#include "common.h"

namespace parallel {

// Splits [0, n_groups) into n_threads contiguous ranges of equal group count
// and calls f(start_group, end_group) on each. Exceptions can't cross a thread
// boundary, so each worker stashes its own and we rethrow once everything has
// been joined.
template <typename Func>
inline void ForEach(index_t n_groups, int n_threads, Func f) {
  if (n_threads < 2) {
    f(0, n_groups);
    return;
  }
  std::vector<std::exception_ptr> errors(n_threads);
  std::vector<std::thread> threads;
  threads.reserve(n_threads);
  const index_t groups_per_thread = n_groups / n_threads;
  const index_t remainder = n_groups % n_threads;
  std::exception_ptr spawn_error;
  try {
    for (int t = 0; t < n_threads; ++t) {
      const index_t start_group =
          t * groups_per_thread + std::min<index_t>(t, remainder);
      const index_t end_group =
          (t + 1) * groups_per_thread + std::min<index_t>(t + 1, remainder);
      threads.emplace_back([&f, &errors, t, start_group, end_group]() {
        try {
          f(start_group, end_group);
        } catch (...) {
          errors[t] = std::current_exception();
        }
      });
    }
  } catch (...) {
    spawn_error = std::current_exception();
  }
  for (auto &thread : threads) {
    thread.join();
  }
  if (spawn_error) {
    std::rethrow_exception(spawn_error);
  }
  for (auto &error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
}

} // namespace parallel
