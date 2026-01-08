#!/usr/bin/env python3
"""Parallel mutation testing by module.

Runs mutmut on each module with only the tests that import it,
dramatically reducing test execution time per mutation.

Usage:
    python scripts/mutmut_parallel.py [--dry-run] [--modules fitting,results]

Example (estimate speedup):
    python scripts/mutmut_parallel.py --dry-run

Example (run specific modules in parallel terminals):
    # Terminal 1:
    python scripts/mutmut_parallel.py --modules fitting
    # Terminal 2:
    python scripts/mutmut_parallel.py --modules results
"""

import argparse
import os
import subprocess
import sys

# Module -> Test files mapping (only tests that import this module)
MODULE_TEST_MAP = {
    "fitting": [
        "tests/test_fitting.py",
        "tests/test_heavy_tail.py",
    ],
    "discrete_fitting": [
        "tests/test_discrete_fitting.py",
    ],
    "results": [
        "tests/test_results.py",
        "tests/test_serialization.py",
        "tests/test_bounded_fitting.py",
    ],
    "distributions": [
        "tests/test_distributions.py",
        "tests/test_partition_strategy.py",
    ],
    "config": [
        "tests/test_config.py",
    ],
    "histogram": [
        "tests/test_histogram.py",
    ],
    "copula": [
        "tests/test_copula.py",
    ],
    "serialization": [
        "tests/test_serialization.py",
    ],
    "truncated": [
        "tests/test_fitting.py",  # truncated is tested via fitting
    ],
    "continuous_fitter": [
        "tests/test_numerical_stability.py",
        "tests/test_property_based.py",
    ],
    "discrete_fitter": [
        "tests/test_numerical_stability.py",
    ],
    "core": [
        "tests/test_core.py",
    ],
    "backends.local": [
        "tests/test_backends.py",
        "tests/test_backend_factory.py",
    ],
    "backends.factory": [
        "tests/test_backend_factory.py",
    ],
}


def estimate_speedup():
    """Estimate speedup from per-module test selection."""
    print("=" * 60)
    print("MUTATION TESTING SPEEDUP ESTIMATION")
    print("=" * 60)

    # Count tests per module vs all tests
    all_tests_count = 590  # From earlier measurement
    all_tests_time = 35  # seconds

    print(f"\nCurrent approach: ALL {all_tests_count} tests per mutation (~{all_tests_time}s)")
    print(f"3800 mutations × {all_tests_time}s = ~{3800 * all_tests_time / 3600:.0f} hours\n")

    print("Per-module approach:")
    total_estimated_time = 0

    for module, tests in MODULE_TEST_MAP.items():
        # Rough estimate: each test file has ~50 tests, each test ~0.06s
        est_tests = len(tests) * 50
        est_time = est_tests * 0.06 + 1  # +1s overhead
        # Estimate mutations per module (rough: 200-400 per module)
        est_mutations = 300
        module_time = est_mutations * est_time
        total_estimated_time += module_time

        print(f"  {module:20s}: {len(tests):2d} test files, ~{est_time:.1f}s/mutation")

    print(f"\nEstimated total: ~{total_estimated_time / 3600:.1f} hours")
    print(f"Speedup: ~{(3800 * all_tests_time) / total_estimated_time:.1f}x faster")
    print("\nRun with: python scripts/mutmut_parallel.py --modules <module>")


def run_module(module: str, dry_run: bool = False):
    """Run mutmut on a single module with its specific tests."""
    tests = MODULE_TEST_MAP.get(module)
    if not tests:
        print(f"Unknown module: {module}")
        print(f"Available: {', '.join(MODULE_TEST_MAP.keys())}")
        return 1

    # Build the mutmut command
    src_path = f"src/spark_bestfit/{module.replace('.', '/')}.py"
    cache_dir = f".mutmut-cache-{module.replace('.', '_')}"

    cmd = [
        "mutmut",
        "run",
        f"--paths-to-mutate={src_path}",
        "--runner",
        f"pytest {' '.join(tests)} -x -q --tb=no -p no:spark -m 'not spark'",
    ]

    # Set unique cache directory via environment
    env = os.environ.copy()
    env["MUTMUT_CACHE_FILE"] = cache_dir

    print(f"\n{'=' * 60}")
    print(f"Module: {module}")
    print(f"Source: {src_path}")
    print(f"Tests:  {', '.join(tests)}")
    print(f"Cache:  {cache_dir}")
    print(f"{'=' * 60}")

    if dry_run:
        print(f"Would run: {' '.join(cmd)}")
        return 0

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Parallel mutation testing by module")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--modules", help="Comma-separated modules to test (default: show estimate)")
    parser.add_argument("--all", action="store_true", help="Run all modules sequentially")
    args = parser.parse_args()

    if not args.modules and not args.all:
        estimate_speedup()
        return 0

    if args.all:
        modules = list(MODULE_TEST_MAP.keys())
    else:
        modules = [m.strip() for m in args.modules.split(",")]

    for module in modules:
        ret = run_module(module, args.dry_run)
        if ret != 0 and not args.dry_run:
            print(f"Module {module} failed with code {ret}")

    return 0


def print_parallel_commands():
    """Print commands for running modules in parallel terminals."""
    print("\n" + "=" * 60)
    print("PARALLEL EXECUTION (copy to separate terminals)")
    print("=" * 60)

    # Group modules by estimated time for load balancing
    modules_by_size = sorted(MODULE_TEST_MAP.items(), key=lambda x: len(x[1]), reverse=True)

    # Split into 4 groups for 4 parallel terminals
    groups = [[], [], [], []]
    for i, (module, _) in enumerate(modules_by_size):
        groups[i % 4].append(module)

    for i, group in enumerate(groups):
        modules_str = ",".join(group)
        print(f"\n# Terminal {i + 1}:")
        print(f"python scripts/mutmut_parallel.py --modules {modules_str}")

    print("\n# Or run all sequentially:")
    print("python scripts/mutmut_parallel.py --all")
    print("\n# Estimated parallel time with 4 terminals: ~1.5 hours")


if __name__ == "__main__":
    if "--show-parallel" in sys.argv:
        print_parallel_commands()
        sys.exit(0)
    sys.exit(main())
