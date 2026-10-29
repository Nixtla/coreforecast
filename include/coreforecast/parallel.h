#pragma once

// The scheduler the grouped operations run through. Nothing here may depend on
// Python or pybind11: the GIL is released on the binding side, so this file is
// compiled into the C++ tests on its own and the sanitizers see the parallel
// path without an interpreter in the way.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <exception>
#include <span>
#include <thread>
#include <vector>

#include "common.h"

namespace parallel {

using std::chrono::nanoseconds;

// Chunks per thread. Enough of them that a thread which drew cheap chunks takes
// more, few enough that the atomic fetch and the ordering below cost nothing
// next to the groups they hand out.
inline constexpr index_t kChunksPerThread = 16;

// How much work each thread has to be left for starting it to pay off.
// Starting and joining a thread costs about 20 us on both a Neoverse N1 and
// GitHub's x86 runners, and a call only broke even on two threads once it held
// about 70 us of work.
inline constexpr nanoseconds kWorkPerThread{50'000};

// How long the calling thread works alone to measure how fast the call goes.
// Long enough for a few chunks next to a ~30 ns clock read, short enough that
// a call which does need threads loses little by starting them this late.
inline constexpr nanoseconds kSoloBudget{10'000};

// The budget ForEach uses when it isn't given one. Process-wide so that setting
// it to zero starts every thread asked for in every array, derived ones
// included.
inline std::atomic<index_t> solo_budget_ns{kSoloBudget.count()};

inline nanoseconds SoloBudget() {
  return nanoseconds{solo_budget_ns.load(std::memory_order_relaxed)};
}

// Returns the budget it replaces.
inline nanoseconds SetSoloBudget(nanoseconds budget) {
  return nanoseconds{solo_budget_ns.exchange(budget.count())};
}

// How long the chunks run so far took, and how much they weighed.
struct Progress {
  nanoseconds elapsed{0};
  index_t done = 0;

  // How long `weight` more takes at the rate so far, or zero before anything
  // has run.
  nanoseconds Projected(index_t weight) const {
    if (done == 0) {
      return nanoseconds{0};
    }
    const double ns = static_cast<double>(elapsed.count()) *
                      static_cast<double>(weight) / static_cast<double>(done);
    // converting a double past the range of index_t is undefined
    return nanoseconds{static_cast<index_t>(std::min(ns, 1e18))};
  }
};

// Whether the calling thread can take a chunk of next_weight alone and still
// finish it within the budget. The first chunk has no rate to go by, so it
// always runs.
inline bool KeepSolo(const Progress &progress, index_t next_weight,
                     nanoseconds budget) {
  return progress.elapsed + progress.Projected(next_weight) <= budget;
}

// Whether batch_weight more can run before the clock is read again: what fits
// in half the budget left, so that a rate gone wrong overshoots by at most
// that half. Before there is a rate, only one chunk runs.
inline bool FitsInBatch(const Progress &progress, index_t batch_weight,
                        nanoseconds budget) {
  return progress.done > 0 &&
         2 * progress.Projected(batch_weight) <= budget - progress.elapsed;
}

// How many threads the work left after the solo phase keeps busy: one per
// per_thread of it, but no more than requested or than there are chunks, and
// at least one.
inline int ThreadsForRemaining(int requested, nanoseconds remaining,
                               index_t remaining_chunks,
                               nanoseconds per_thread) {
  assert(per_thread.count() > 0);
  return static_cast<int>(
      std::max<index_t>(1, std::min<index_t>({requested, remaining / per_thread,
                                              remaining_chunks})));
}

// indptr.size() - 1, except that an empty indptr has no groups rather than -1
// of them.
inline index_t NumGroups(std::span<const index_t> indptr) {
  return indptr.empty() ? 0 : std::ssize(indptr) - 1;
}

inline index_t ChunkSize(index_t n_groups, int n_threads) {
  return std::max<index_t>(1, n_groups / (kChunksPerThread * n_threads));
}

// The start offset of every chunk of [0, n_groups), heaviest chunk first,
// where indptr[i + 1] - indptr[i] is how many elements group i holds.
//
// The order matters because a group is never split: a thread that starts the
// longest one last has nothing left to overlap it with, and the call can't
// finish before that group does. Chunks of equal weight keep their index
// order, so evenly sized groups are walked exactly as they are laid out.
inline std::vector<index_t> ChunkStarts(std::span<const index_t> indptr,
                                        index_t chunk) {
  assert(chunk > 0);
  const index_t n_groups = NumGroups(indptr);
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

// Calls f(start_group, end_group) over the chunks of [0, n_groups) until they
// run out, on at most n_threads threads.
//
// Starting a thread costs more than a small call's work, and what that work
// costs depends on the kernel, its parameters and the CPU, so it is measured
// rather than predicted: the calling thread first runs the lightest chunks
// alone for up to solo_budget, and a call that finishes inside it never starts
// a thread. Otherwise the rest gets one thread per kWorkPerThread it is
// projected to take at the rate measured so far. A budget of zero skips the
// solo phase and starts every thread asked for.
//
// Chunks then go out off a shared counter in the order ChunkStarts puts them
// in, so a thread that drew short groups takes more of them instead of idling
// while another works through the long ones. The calling thread is worker 0
// and at most one thread per chunk runs, so no thread ever starts without a
// chunk to run and a spawn that fails only costs a thread.
//
// Every group writes only its own slice of the output, so neither the thread
// count nor the order the chunks go out in changes the result. Exceptions
// can't cross a thread boundary, so a worker that throws stashes its own and
// stops the counter, the rest hand back after the chunk they are on, and it
// is rethrown once everything has been joined.
template <typename Func>
inline void ForEach(std::span<const index_t> indptr, int n_threads, Func f,
                    nanoseconds solo_budget = SoloBudget()) {
  const index_t n_groups = NumGroups(indptr);
  n_threads = static_cast<int>(std::min<index_t>(n_threads, n_groups));
  if (n_threads < 2) {
    f(0, n_groups);
    return;
  }
  const index_t chunk = ChunkSize(n_groups, n_threads);
  const std::vector<index_t> starts = ChunkStarts(indptr, chunk);
  const auto end_of = [chunk, n_groups](index_t start) {
    return std::min(start + chunk, n_groups);
  };
  // groups count too, so that a chunk of empty ones still weighs something
  const auto weight = [indptr, &end_of](index_t start) {
    const index_t end = end_of(start);
    return indptr[end] - indptr[start] + end - start;
  };
  // starts[0, n_chunks) are the chunks still to run
  size_t n_chunks = starts.size();
  if (solo_budget.count() > 0) {
    const auto solo_start = std::chrono::steady_clock::now();
    Progress progress;
    while (n_chunks > 0 &&
           KeepSolo(progress, weight(starts[n_chunks - 1]), solo_budget)) {
      // reading the clock after every chunk costs more than a tiny call's work
      index_t batch = 0;
      do {
        const index_t start = starts[--n_chunks];
        f(start, end_of(start));
        batch += weight(start);
      } while (n_chunks > 0 &&
               FitsInBatch(progress, batch + weight(starts[n_chunks - 1]),
                           solo_budget));
      progress.elapsed = std::chrono::steady_clock::now() - solo_start;
      progress.done += batch;
    }
    const index_t total = indptr[n_groups] - indptr[0] + n_groups;
    n_threads = ThreadsForRemaining(
        n_threads, progress.Projected(total - progress.done),
        static_cast<index_t>(n_chunks), kWorkPerThread);
    if (n_threads < 2) {
      for (size_t i = 0; i < n_chunks; ++i) {
        f(starts[i], end_of(starts[i]));
      }
      return;
    }
  }
  std::atomic<size_t> next_chunk{0};
  std::vector<std::exception_ptr> errors(n_threads);
  const auto worker = [&f, &errors, &next_chunk, &starts, &end_of,
                       n_chunks](int t) {
    try {
      for (size_t i = next_chunk.fetch_add(1); i < n_chunks;
           i = next_chunk.fetch_add(1)) {
        f(starts[i], end_of(starts[i]));
      }
    } catch (...) {
      errors[t] = std::current_exception();
      next_chunk.store(n_chunks);
    }
  };
  std::vector<std::thread> threads;
  threads.reserve(n_threads - 1);
  for (int t = 1; t < n_threads; ++t) {
    try {
      threads.emplace_back(worker, t);
    } catch (...) {
      break;
    }
  }
  worker(0);
  for (auto &thread : threads) {
    thread.join();
  }
  for (auto &error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
}

} // namespace parallel
