"""Times the grouped kernels at 1, 2 and 4 threads and prints the speedup table.

Not part of tests/test_efficiency.py: that suite runs under CodSpeed in
simulation mode, which counts instructions and so can't see what threads do.
With --check the four-thread speedups on the uniform dataset have to clear the
floors below, which is what the CI job runs.
"""

import argparse
import os
import sys
import time

import numpy as np
from coreforecast.grouped_array import GroupedArray

THREAD_COUNTS = [1, 2, 4]

# The floor is the four-thread speedup a --check run has to clear. Each is
# about half of what a quiet four-core box does, so the noise of a shared
# runner doesn't trip it but the regression it guards against does: before the
# skip list got a per-thread coin toss the quantiles ran at 0.47x. rolling_mean
# gets a lower one still: at 2.6 ms serial it is the kernel where spawning the
# threads and the noise of the machine weigh most.
KERNELS = [
    ("rolling_mean", lambda ga: ga._rolling_mean(1, 7, 1), 1.5),
    ("rolling_max", lambda ga: ga._rolling_max(1, 7, 1), 2.5),
    ("robust_iqr_stats", lambda ga: ga._robust_iqr_stats(), 2.5),
    ("boxcox_loglik", lambda ga: ga._boxcox_loglik(-0.9, 2.0), 2.5),
    ("rolling_quantile (w=7)", lambda ga: ga._rolling_quantile(1, 0.5, 7, 1), 2.5),
    ("rolling_quantile (w=100)", lambda ga: ga._rolling_quantile(1, 0.5, 100, 1), 2.5),
]


def build(rng, lengths):
    indptr = np.append(0, np.cumsum(lengths)).astype(np.int64)
    # positive, since box-cox takes the log of it
    return rng.uniform(low=1.0, high=100.0, size=indptr[-1]), indptr


def uniform(rng):
    return build(rng, rng.integers(low=1000, high=2000, size=1000))


def huge_group(rng):
    lengths = rng.integers(low=1000, high=2000, size=1000)
    return build(rng, np.append(lengths, lengths.sum()))


def tiny(rng):
    return build(rng, np.full(200, 20))


# Only the uniform case is held to the floors. The other two are what the rest
# of the threading plan is about: handing work out dynamically so one huge
# group doesn't pin a thread, and not spawning at all for an array this small.
DATASETS = [
    ("uniform", uniform, 5, True),
    ("one huge group", huge_group, 5, False),
    ("tiny", tiny, 200, False),
]


def available_cores():
    # not hardware_concurrency: taskset and cgroup cpusets are what a CI runner
    # hands us, and only the affinity mask reflects them
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def time_it(fn, ga, repeats):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn(ga)
        best = min(best, time.perf_counter() - start)
    return best * 1e3


def run(name, dataset, repeats, floors):
    data, indptr = dataset
    print(f"\n## {name}: {len(indptr) - 1} groups, {data.size} elements")
    header = "| kernel".ljust(28) + "".join(f"| {t:>2}t (ms) " for t in THREAD_COUNTS)
    header += "".join(f"| {t}t speedup " for t in THREAD_COUNTS[1:]) + "|"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    below = []
    for kernel_name, fn, floor in KERNELS:
        times = [
            time_it(fn, GroupedArray(data, indptr, num_threads=t), repeats)
            for t in THREAD_COUNTS
        ]
        speedups = [times[0] / t for t in times[1:]]
        row = f"| {kernel_name}".ljust(28)
        row += "".join(f"| {t:>9.3f} " for t in times)
        row += "".join(f"| {s:>10.2f} " for s in speedups)
        print(row + "|")
        if floors and speedups[-1] < floor:
            below.append(
                f"{kernel_name}: {speedups[-1]:.2f}x on 4 threads, floor {floor:.2f}x"
            )
    return below


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if a four-thread speedup is under its floor",
    )
    args = parser.parse_args()
    cores = available_cores()
    print(f"{cores} cores available")
    if args.check and cores < max(THREAD_COUNTS):
        # passing here would make the gate disappear the day a runner shrinks
        sys.exit(f"--check needs {max(THREAD_COUNTS)} cores to mean anything")
    rng = np.random.default_rng(0)
    below = []
    for name, dataset, repeats, floors in DATASETS:
        below += run(name, dataset(rng), repeats, floors and args.check)
    if below:
        print("\nBelow the floor:")
        for line in below:
            print(f"  {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
