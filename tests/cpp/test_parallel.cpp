#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <vector>

#include "parallel.h"
#include "rolling.h"

#include "helpers.h"

using namespace helpers;

namespace {

// One counter per group, so a group left out or run twice shows up as a count
// other than one.
std::vector<int> VisitCounts(index_t n_groups, int n_threads) {
  std::vector<std::atomic<int>> visits(n_groups);
  parallel::ForEach(n_groups, n_threads,
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

TEST_CASE("a throwing chunk surfaces once the other workers have joined") {
  constexpr index_t kGroups = 64;
  constexpr int kThreads = 4;
  std::atomic<bool> threw{false};
  std::atomic<int> started{0};
  std::atomic<int> finished{0};
  // Every chunk but the first parks until the throw has happened, so returning
  // without joining leaves workers reading these locals: a use after return for
  // ASan, a race for TSan, and started != finished here.
  auto run_and_throw = [&threw, &started, &finished] {
    parallel::ForEach(kGroups, kThreads,
                      [&threw, &started, &finished](index_t start_group,
                                                    index_t) {
                        if (start_group == 0) {
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
  auto run = [&data, &w](std::vector<double> &out, int n_threads) {
    parallel::ForEach(
        kGroups, n_threads,
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
  // Every group owns its own slice of the output, so the split across threads
  // must not change a bit of it
  for (index_t i = 0; i < std::ssize(serial); ++i) {
    INFO("index ", i);
    REQUIRE(threaded[i] == serial[i]);
  }
}
