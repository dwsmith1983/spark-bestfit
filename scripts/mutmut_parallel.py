#!/usr/bin/env python3
"""Parallel mutation testing by module.

Runs mutmut on each module with only the tests that import it,
dramatically reducing test execution time per mutation.

Usage:
    python scripts/mutmut_parallel.py [--dry-run] [--modules fitting,results]
    python scripts/mutmut_parallel.py --all
    python scripts/mutmut_parallel.py --show-survivors fitting

Example (estimate speedup):
    python scripts/mutmut_parallel.py --dry-run

Example (run specific modules in parallel terminals):
    # Terminal 1:
    python scripts/mutmut_parallel.py --modules fitting
    # Terminal 2:
    python scripts/mutmut_parallel.py --modules results

After running, view survivors:
    python scripts/mutmut_parallel.py --show-survivors fitting
"""

import argparse
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

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

RESULTS_DIR = Path(".mutmut-results")
SUMMARY_FILE = RESULTS_DIR / "summary.txt"


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    RESULTS_DIR.mkdir(exist_ok=True)


def parse_mutmut_output(output: str) -> dict:
    """Parse mutmut output to extract statistics."""
    # Look for pattern like: ⠧ 498/498  🎉 239  ⏰ 0  🤔 0  🙁 259  🔇 0
    pattern = r"(\d+)/(\d+)\s+🎉\s*(\d+)\s+⏰\s*(\d+)\s+🤔\s*(\d+)\s+🙁\s*(\d+)\s+🔇\s*(\d+)"
    match = re.search(pattern, output)
    if match:
        return {
            "completed": int(match.group(1)),
            "total": int(match.group(2)),
            "killed": int(match.group(3)),
            "timeout": int(match.group(4)),
            "suspicious": int(match.group(5)),
            "survived": int(match.group(6)),
            "skipped": int(match.group(7)),
        }
    return {}


def get_stats_from_cache() -> dict:
    """Get mutation stats directly from the mutmut cache sqlite database."""
    cache_file = Path(".mutmut-cache")
    if not cache_file.exists():
        return {}

    try:
        conn = sqlite3.connect(cache_file)
        cursor = conn.cursor()

        # Count mutants by status
        cursor.execute(
            """
            SELECT status, COUNT(*) FROM Mutant GROUP BY status
        """
        )
        rows = cursor.fetchall()
        conn.close()

        # Map status values to our stats dict
        status_map = {
            "ok_killed": "killed",
            "bad_survived": "survived",
            "bad_timeout": "timeout",
            "ok_suspicious": "suspicious",
            "skipped": "skipped",
        }

        stats = {
            "killed": 0,
            "survived": 0,
            "timeout": 0,
            "suspicious": 0,
            "skipped": 0,
        }

        total = 0
        for status, count in rows:
            total += count
            if status in status_map:
                stats[status_map[status]] = count

        stats["total"] = total
        stats["completed"] = total
        return stats
    except Exception:
        return {}


def backup_cache(module: str):
    """Backup the mutmut cache for this module."""
    cache_file = Path(".mutmut-cache")
    if cache_file.exists():
        backup_path = RESULTS_DIR / f"cache-{module.replace('.', '_')}.sqlite"
        shutil.copy(cache_file, backup_path)
        return backup_path
    return None


def save_module_result(module: str, stats: dict, duration: float):
    """Save module result to summary file."""
    ensure_results_dir()

    with open(SUMMARY_FILE, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        kill_rate = (
            stats["killed"] / (stats["killed"] + stats["survived"]) * 100
            if (stats["killed"] + stats["survived"]) > 0
            else 0
        )
        f.write(f"\n[{timestamp}] {module}\n")
        f.write(f"  Total: {stats.get('total', 'N/A')}\n")
        f.write(f"  Killed: {stats.get('killed', 'N/A')} 🎉\n")
        f.write(f"  Survived: {stats.get('survived', 'N/A')} 🙁\n")
        f.write(f"  Suspicious: {stats.get('suspicious', 'N/A')} 🤔\n")
        f.write(f"  Timeout: {stats.get('timeout', 'N/A')} ⏰\n")
        f.write(f"  Kill Rate: {kill_rate:.1f}%\n")
        f.write(f"  Duration: {duration:.1f}s\n")


def get_mutation_diff(cache_file: Path, mutant_id: int) -> Optional[str]:
    """Get the mutation diff using mutmut show."""
    env = os.environ.copy()
    env["MUTMUT_CACHE"] = str(cache_file)
    env["PYTHONPATH"] = "src"

    try:
        result = subprocess.run(
            ["mutmut", "show", str(mutant_id)],
            capture_output=True,
            text=True,
            env=env,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def parse_mutation_diff(diff: str) -> tuple[Optional[str], Optional[str]]:
    """Extract the before/after from a mutation diff."""
    lines = diff.split("\n")
    before = None
    after = None

    for line in lines:
        if line.startswith("-") and not line.startswith("---"):
            before = line[1:].strip()
        elif line.startswith("+") and not line.startswith("+++"):
            after = line[1:].strip()

    return before, after


def show_survivors(module: str, show_diffs: bool = True):
    """Show survived mutants for a module from its cached database."""
    cache_file = RESULTS_DIR / f"cache-{module.replace('.', '_')}.sqlite"

    if not cache_file.exists():
        print(f"No cache found for module: {module}")
        print(f"Expected: {cache_file}")
        print("\nAvailable caches:")
        for f in RESULTS_DIR.glob("cache-*.sqlite"):
            print(f"  {f.stem.replace('cache-', '').replace('_', '.')}")
        return 1

    conn = sqlite3.connect(cache_file)
    cursor = conn.cursor()

    # Get survived mutants with file and line info
    cursor.execute(
        """
        SELECT
            sf.filename,
            l.line_number,
            l.line as original_code,
            m.id as mutant_id
        FROM Mutant m
        JOIN Line l ON m.line = l.id
        JOIN SourceFile sf ON l.sourcefile = sf.id
        WHERE m.status = 'bad_survived'
        ORDER BY l.line_number
    """
    )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print(f"No survived mutants for module: {module}")
        return 0

    print(f"\n{'=' * 70}")
    print(f"SURVIVED MUTANTS: {module} ({len(rows)} total)")
    print(f"{'=' * 70}")

    # Try to get actual mutations via mutmut show
    mutmut_available = False
    if show_diffs:
        # Test if mutmut is available
        diff = get_mutation_diff(cache_file, rows[0][3])
        mutmut_available = diff is not None

    current_file = None
    for filename, line_num, code, mutant_id in rows:
        if filename != current_file:
            current_file = filename
            print(f"\n{filename}:")

        if mutmut_available:
            diff = get_mutation_diff(cache_file, mutant_id)
            if diff:
                before, after = parse_mutation_diff(diff)
                if before and after:
                    # Show compact before → after
                    before_short = before[:40] + "..." if len(before) > 40 else before
                    after_short = after[:40] + "..." if len(after) > 40 else after
                    print(f"  Line {line_num:4d} [#{mutant_id}]:")
                    print(f"    - {before_short}")
                    print(f"    + {after_short}")
                    continue

        # Fallback: just show line info
        code_preview = code[:60] + "..." if len(code) > 60 else code
        print(f"  Line {line_num:4d} [#{mutant_id}]: {code_preview}")

    print(f"\n{'=' * 70}")
    if not mutmut_available:
        print("Note: Install mutmut to see actual mutations")
        print("")
    print("To see a specific mutation:")
    print(f"  MUTMUT_CACHE={cache_file} mutmut show <mutant_id>")
    print("To apply and inspect:")
    print(f"  MUTMUT_CACHE={cache_file} mutmut apply <mutant_id>")
    print("  git diff  # see the mutation")
    print("  git checkout .  # revert")

    return 0


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
    import time

    tests = MODULE_TEST_MAP.get(module)
    if not tests:
        print(f"Unknown module: {module}")
        print(f"Available: {', '.join(MODULE_TEST_MAP.keys())}")
        return 1

    # Build the mutmut command
    src_path = f"src/spark_bestfit/{module.replace('.', '/')}.py"

    cmd = [
        "mutmut",
        "run",
        f"--paths-to-mutate={src_path}",
        "--runner",
        f"pytest {' '.join(tests)} -x -q --tb=no -p no:spark -m 'not spark'",
    ]

    print(f"\n{'=' * 60}")
    print(f"Module: {module}")
    print(f"Source: {src_path}")
    print(f"Tests:  {', '.join(tests)}")
    print(f"{'=' * 60}")

    if dry_run:
        print(f"Would run: {' '.join(cmd)}")
        return 0

    print(f"Running: {' '.join(cmd)}\n")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time

    # Print output
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Parse results - check both stdout and stderr (mutmut uses stderr for progress)
    combined_output = result.stdout + "\n" + result.stderr
    stats = parse_mutmut_output(combined_output)
    if not stats:
        # Regex didn't match, get stats directly from sqlite
        stats = get_stats_from_cache()
        if stats:
            print(f"(Stats from sqlite: {stats['killed']} killed, {stats['survived']} survived)")

    if stats:
        ensure_results_dir()
        save_module_result(module, stats, duration)

        # Backup cache before next module overwrites it
        backup_path = backup_cache(module)
        if backup_path:
            print(f"\nCache saved to: {backup_path}")
            print(f"View survivors: python scripts/mutmut_parallel.py --show-survivors {module}")

    return result.returncode


def get_stats_from_sqlite(cache_file: Path) -> dict:
    """Get mutation stats from a specific sqlite cache file."""
    if not cache_file.exists():
        return {}

    try:
        conn = sqlite3.connect(cache_file)
        cursor = conn.cursor()

        cursor.execute("SELECT status, COUNT(*) FROM Mutant GROUP BY status")
        rows = cursor.fetchall()
        conn.close()

        status_map = {
            "ok_killed": "killed",
            "bad_survived": "survived",
            "bad_timeout": "timeout",
            "ok_suspicious": "suspicious",
            "skipped": "skipped",
        }

        stats = {
            "killed": 0,
            "survived": 0,
            "timeout": 0,
            "suspicious": 0,
            "skipped": 0,
        }

        total = 0
        for status, count in rows:
            total += count
            if status in status_map:
                stats[status_map[status]] = count

        stats["total"] = total
        return stats
    except Exception:
        return {}


def rebuild_summary():
    """Rebuild summary from all cached sqlite files."""
    if not RESULTS_DIR.exists():
        print("No results directory found.")
        return 1

    cache_files = sorted(RESULTS_DIR.glob("cache-*.sqlite"))
    if not cache_files:
        print("No cache files found.")
        return 1

    # Clear and rebuild summary
    if SUMMARY_FILE.exists():
        SUMMARY_FILE.unlink()

    print(f"Rebuilding summary from {len(cache_files)} cache files...")

    for cache_file in cache_files:
        module = cache_file.stem.replace("cache-", "").replace("_", ".")
        stats = get_stats_from_sqlite(cache_file)

        if stats and stats.get("total", 0) > 0:
            save_module_result(module, stats, 0.0)
            kill_rate = (
                stats["killed"] / (stats["killed"] + stats["survived"]) * 100
                if (stats["killed"] + stats["survived"]) > 0
                else 0
            )
            print(f"  {module}: {stats['killed']}/{stats['total']} killed ({kill_rate:.1f}%)")

    print(f"\nSummary rebuilt to {SUMMARY_FILE}")
    return 0


def print_summary():
    """Print summary of all module results."""
    if not SUMMARY_FILE.exists():
        print("No results found. Run mutation testing first.")
        return

    print(f"\n{'=' * 70}")
    print("MUTATION TESTING SUMMARY")
    print(f"{'=' * 70}")
    print(SUMMARY_FILE.read_text())


def main():
    parser = argparse.ArgumentParser(description="Parallel mutation testing by module")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--modules", help="Comma-separated modules to test (default: show estimate)")
    parser.add_argument("--all", action="store_true", help="Run all modules sequentially")
    parser.add_argument("--show-survivors", metavar="MODULE", help="Show survived mutants for a module")
    parser.add_argument("--summary", action="store_true", help="Show summary of all results")
    parser.add_argument("--rebuild-summary", action="store_true", help="Rebuild summary from cached sqlite files")
    parser.add_argument("--resume", action="store_true", help="Skip modules that already have cached results")
    args = parser.parse_args()

    if args.show_survivors:
        return show_survivors(args.show_survivors)

    if args.rebuild_summary:
        return rebuild_summary()

    if args.summary:
        print_summary()
        return 0

    if not args.modules and not args.all:
        estimate_speedup()
        return 0

    # Get list of modules to run
    if args.all:
        ensure_results_dir()
        modules = list(MODULE_TEST_MAP.keys())
    else:
        modules = [m.strip() for m in args.modules.split(",")]

    # Filter out completed modules if resuming
    if args.resume:
        # Get completed module cache names (with dots replaced by underscores)
        completed_caches = set()
        for cache_file in RESULTS_DIR.glob("cache-*.sqlite"):
            cache_name = cache_file.stem.replace("cache-", "")
            completed_caches.add(cache_name)

        # Match modules by converting their names the same way
        def is_completed(module: str) -> bool:
            cache_name = module.replace(".", "_")
            return cache_name in completed_caches

        completed = {m for m in modules if is_completed(m)}

        original_count = len(modules)
        modules = [m for m in modules if m not in completed]
        skipped = original_count - len(modules)

        if skipped > 0:
            print(f"Resuming: skipping {skipped} completed modules")
            print(f"Remaining: {len(modules)} modules\n")

        if not modules:
            print("All modules already completed!")
            return 0
    elif args.all and not args.resume:
        # Clear summary file for fresh run (only if not resuming)
        if SUMMARY_FILE.exists():
            SUMMARY_FILE.unlink()

    for module in modules:
        ret = run_module(module, args.dry_run)
        if ret != 0 and not args.dry_run:
            print(f"Module {module} failed with code {ret}")

    if not args.dry_run:
        print_summary()

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
