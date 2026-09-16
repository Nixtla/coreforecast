#pragma once

// The scheduler the grouped operations run through. Nothing here may depend on
// Python or pybind11: the GIL is released on the binding side, so this file is
// compiled into the C++ tests on its own and the sanitizers see the parallel
// path without an interpreter in the way.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <exception>
#include <span>
#include <thread>
#include <vector>

#include "common.h"

namespace parallel {

// Chunks per thread. Enough of them that a thread which drew cheap chunks takes
// more, few enough that the atomic fetch and the ordering below cost nothing
// next to the groups they hand out.
inline constexpr index_t kChunksPerThread = 16;

inline index_t ChunkSize(index_t n_groups, int n_threads) {
  return std::max<index_t>(1, n_groups / (kChunksPerThread * n_threads));
}

// The start offset of every chunk of [0, n_groups), heaviest chunk first,
// where indptr[i + 1] - indptr[i] is how many elements group i holds.
//
// The order matters because a group is never split: a thread that starts the
// longest one last has nothing left to overlap it with, and the call can't
// finish before that group does. Handing chunks out in index order costs about
// a fifth of the four-thread speedup when the longest group sits at the end.
// Chunks of equal weight keep their index order, so evenly sized groups are
// walked exactly as they are laid out.
inline std::vector<index_t> ChunkStarts(std::span<const index_t> indptr,
                                        index_t chunk) {
  assert(!indptr.empty());
  assert(chunk > 0);
  const index_t n_groups = std::ssize(indptr) - 1;
  std::vector<index_t> starts;
  starts.reserve(static_cast<size_t>((n_groups + chunk - 1) / chunk));
  for (index_t start = 0; start < n_groups; start += chunk) {
    starts.push_back(start);
  }
  const auto elements = [indptr, chunk, n_groups](index_t start) {
    return indptr[std::min(start + chunk, n_groups)] - indptr[start];
  };
  std::sort(starts.begin(), starts.end(), [&elements](index_t a, index_t b) {
    const index_t wa = elements(a);
    const index_t wb = elements(b);
    return wa != wb ? wa > wb : a < b;
  });
  return starts;
}

// Calls f(start_group, end_group) over the chunks of [0, n_groups), which is
// indptr.size() - 1 groups, until they run out. Chunks go out off a shared
// counter in the order ChunkStarts puts them in, so a thread that drew short
// groups takes more of them instead of idling while another works through the
// long ones. One group holding half the elements still caps the speedup at 2x.
//
// Every group writes only its own slice of the output, so neither the thread
// count nor the order the chunks go out in changes the result. Exceptions
// can't cross a thread boundary, so each worker stashes its own and we rethrow
// once everything has been joined.
template <typename Func>
inline void ForEach(std::span<const index_t> indptr, int n_threads, Func f) {
  assert(!indptr.empty());
  const index_t n_groups = std::ssize(indptr) - 1;
  if (n_threads < 2) {
    f(0, n_groups);
    return;
  }
  const index_t chunk = ChunkSize(n_groups, n_threads);
  const std::vector<index_t> starts = ChunkStarts(indptr, chunk);
  std::atomic<size_t> next_chunk{0};
  std::vector<std::exception_ptr> errors(n_threads);
  std::vector<std::thread> threads;
  threads.reserve(n_threads);
  std::exception_ptr spawn_error;
  try {
    for (int t = 0; t < n_threads; ++t) {
      threads.emplace_back(
          [&f, &errors, &next_chunk, &starts, t, chunk, n_groups]() {
            try {
              for (size_t i = next_chunk.fetch_add(1); i < starts.size();
                   i = next_chunk.fetch_add(1)) {
                f(starts[i], std::min(starts[i] + chunk, n_groups));
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
