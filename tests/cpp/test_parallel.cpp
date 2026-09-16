#include <algorithm>
#include <atomic>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "parallel.h"
#include "rolling.h"

#include "helpers.h"

using namespace helpers;

namespace {

// n_groups groups of one element each, which makes every chunk weigh the same.
std::vector<index_t> FlatIndptr(index_t n_groups) {
  std::vector<index_t> indptr(n_groups + 1);
  std::iota(indptr.begin(), indptr.end(), index_t{0});
  return indptr;
}

// One counter per group, so a group left out or run twice shows up as a count
// other than one.
std::vector<int> VisitCounts(index_t n_groups, int n_threads) {
  std::vector<std::atomic<int>> visits(n_groups);
  const auto indptr = FlatIndptr(n_groups);
  parallel::ForEach(indptr, n_threads,
                    [&visits](index_t start_group, index_t end_group) {
                      for (index_t i = start_group; i < end_group; ++i) {
                        visits[i].fetch_add(1);
                      }
                    });
  return {visits.begin(), visits.end()};
}

void CheckAllVisitedOnce(index_t n_groups, int n_threads) {
  INFO("n_groups ", n_groups, ", n_threads ", n_threads);
  const auto visits = VisitCounts(n_groups, n_threads);
  REQUIRE(std::ssize(visits) == n_groups);
  CHECK(std::all_of(visits.begin(), visits.end(),
                    [](int count) { return count == 1; }));
}

} // namespace

TEST_CASE("every group is visited exactly once") {
  for (index_t n_groups : {0, 1, 7, 1000}) {
    for (int n_threads : {1, 2, 3, 8}) {
      CheckAllVisitedOnce(n_groups, n_threads);
    }
  }
}

TEST_CASE("equal chunks keep their index order") {
  const auto indptr = FlatIndptr(10);
  CHECK(parallel::ChunkStarts(indptr, 1) ==
        std::vector<index_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  // 4 doesn't divide 10, so the last chunk is short and lighter than the rest;
  // it still has to come last rather than be dropped
  CHECK(parallel::ChunkStarts(indptr, 4) == std::vector<index_t>{0, 4, 8});
  CHECK(parallel::ChunkStarts(indptr, 100) == std::vector<index_t>{0});
  CHECK(parallel::ChunkStarts(FlatIndptr(0), 1).empty());
}

TEST_CASE("the heaviest chunks are handed out first") {
  // group 7 holds almost everything and sits near the end, where going in
  // index order would start it last and leave nothing to overlap it with
  std::vector<index_t> indptr = FlatIndptr(10);
  for (index_t i = 8; i < std::ssize(indptr); ++i) {
    indptr[i] += 1000;
  }
  CHECK(parallel::ChunkStarts(indptr, 1) ==
        std::vector<index_t>{7, 0, 1, 2, 3, 4, 5, 6, 8, 9});
  // chunk 6 covers groups 6 and 7, so it is the heavy one now
  CHECK(parallel::ChunkStarts(indptr, 2) ==
        std::vector<index_t>{6, 0, 2, 4, 8});
}

TEST_CASE("a throwing chunk surfaces once the other workers have joined") {
  constexpr index_t kGroups = 64;
  constexpr int kThreads = 4;
  std::atomic<int> order{0};
  std::atomic<bool> threw{false};
  std::atomic<int> started{0};
  std::atomic<int> finished{0};
  const auto indptr = FlatIndptr(kGroups);
  // Whichever chunk runs first throws, and the rest park until it has, so
  // returning without joining leaves workers reading these locals: a use after
  // return for ASan, a race for TSan, and started != finished here. Keying on
  // the running order rather than on a group index keeps the parked workers
  // from waiting on a chunk that the scheduler hasn't handed out yet.
  auto run_and_throw = [&order, &threw, &started, &finished, &indptr] {
    parallel::ForEach(indptr, kThreads,
                      [&order, &threw, &started, &finished](index_t, index_t) {
                        if (order.fetch_add(1) == 0) {
                          threw.store(true);
                          throw std::runtime_error("boom");
                        }
                        started.fetch_add(1);
                        while (!threw.load()) {
                        }
                        finished.fetch_add(1);
                      });
  };
  CHECK_THROWS_AS(run_and_throw(), std::runtime_error);
  CHECK(started.load() > 0);
  CHECK(finished.load() == started.load());

  CheckAllVisitedOnce(kGroups, kThreads);
}

TEST_CASE("rolling quantile over threads equals the serial result") {
  constexpr index_t kGroups = 200;
  constexpr index_t kSize = 120;
  const auto data = Random<double>(kGroups * kSize);
  const auto w = Window::Checked(10, 1, false);
  std::vector<index_t> indptr(kGroups + 1);
  for (index_t i = 0; i <= kGroups; ++i) {
    indptr[i] = i * kSize;
  }
  auto run = [&data, &w, &indptr](std::vector<double> &out, int n_threads) {
    parallel::ForEach(
        indptr, n_threads,
        [&data, &w, &out](index_t start_group, index_t end_group) {
          for (index_t i = start_group; i < end_group; ++i) {
            rolling::QuantileTransform<double>(
                In(data).subspan(i * kSize, kSize),
                Out(out).subspan(i * kSize, kSize), w, 0.5);
          }
        });
  };
  std::vector<double> serial(data.size());
  std::vector<double> threaded(data.size());
  run(serial, 1);
  run(threaded, 4);
  // Every group owns its own slice of the output, so neither the split across
  // threads nor the order the chunks go out in may change a bit of it
  for (index_t i = 0; i < std::ssize(serial); ++i) {
    INFO("index ", i);
    REQUIRE(threaded[i] == serial[i]);
  }
}
