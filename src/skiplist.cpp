// The vendored skip list declares these three free functions in its header and
// defines them in a translation unit we don't build, so that we can provide
// them here. Its tossCoin() draws from libc's rand, which glibc serialises on
// a process-wide lock (and the BSD libc doesn't guard at all), so every worker
// running a rolling quantile contended on it.

#include <cstdint>
#include <string>

#include "SkipList.h"

namespace OrderedStructs {
namespace SkipList {
namespace {
// Any non-zero value; xorshift is stuck at zero.
constexpr std::uint64_t kDefaultSeed = 0x9E3779B97F4A7C15ULL;

// Per thread, so node heights don't depend on what the other threads are doing
// and no two threads share the state.
thread_local std::uint64_t state = kDefaultSeed;

// xorshift64*
std::uint64_t Next() {
  state ^= state >> 12;
  state ^= state << 25;
  state ^= state >> 27;
  return state * 0x2545F4914F6CDD1DULL;
}
} // namespace

bool tossCoin() { return (Next() >> 63) != 0; }

void seedRand(unsigned seed) {
  state = seed * kDefaultSeed;
  if (state == 0) {
    state = kDefaultSeed;
  }
}

void _throw_exceeds_size(size_t /* index */) {
  throw IndexError("Index out of range.");
}

} // namespace SkipList
} // namespace OrderedStructs
