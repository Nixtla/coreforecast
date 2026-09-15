#pragma once

// The scheduler the grouped operations run through. Nothing here may depend on
// Python or pybind11: the GIL is released on the binding side, so this file is
// compiled into the C++ tests on its own and the sanitizers see the parallel
// path without an interpreter in the way.

#include <algorithm>
#include <atomic>
#include <exception>
#include <thread>
#include <vector>

#include "common.h"

namespace parallel {

// Chunks per thread. Enough of them that a thread which drew short groups can
// take work off one that drew long ones, few enough that the atomic fetch
// costs nothing next to the groups it hands out.
inline constexpr index_t kChunksPerThread = 16;

// Calls f(start_group, end_group) over consecutive chunks of [0, n_groups)
// until they run out. Groups are handed out as workers finish rather than
// split up front, so a thread that drew short groups takes more of them
// instead of idling while another works through the long ones. A group is
// never split, so one holding half the elements still caps the speedup at 2x.
// Every group writes only its own slice of the output, so which thread runs it
// never changes the result. Exceptions can't cross a thread boundary, so each
// worker stashes its own and we rethrow once everything has been joined.
template <typename Func>
inline void ForEach(index_t n_groups, int n_threads, Func f) {
  if (n_threads < 2) {
    f(0, n_groups);
    return;
  }
  const index_t chunk =
      std::max<index_t>(1, n_groups / (kChunksPerThread * n_threads));
  std::atomic<index_t> next_group{0};
  std::vector<std::exception_ptr> errors(n_threads);
  std::vector<std::thread> threads;
  threads.reserve(n_threads);
  std::exception_ptr spawn_error;
  try {
    for (int t = 0; t < n_threads; ++t) {
      threads.emplace_back([&f, &errors, &next_group, t, chunk, n_groups]() {
        try {
          for (index_t start = next_group.fetch_add(chunk); start < n_groups;
               start = next_group.fetch_add(chunk)) {
            f(start, std::min(start + chunk, n_groups));
          }
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
