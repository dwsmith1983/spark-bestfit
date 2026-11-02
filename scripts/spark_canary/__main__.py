"""Command line entry point: ``python -m spark_canary <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from .bench import Comparison, compare, load_rounds, regressions
from .findings import (
    Finding,
    TargetResult,
    classify_pip_error,
    derive_install_flags,
    findings_for_bench,
    findings_for_target,
    is_installable,
    parse_freeze,
    scan_junit,
    workflow_failure_finding,
)
from .issues import gh, upsert
from .versions import build_plan, canary_targets, fetch_json

PYPI_INDEX_URL = "https://pypi.org/pypi/pyspark/json"
PYPI_RELEASE_URL = "https://pypi.org/pypi/pyspark/{version}/json"
OUTCOMES = ("success", "failure", "skipped")
TESTS_PASSED = {"success": True, "failure": False, "skipped": None}
MAX_READ_BYTES = 2_000_000


def resolve(pyproject: Path) -> dict:
    """Fetch PySpark release data from PyPI and build the canary plan."""
    releases = fetch_json(PYPI_INDEX_URL)["releases"]
    classifiers = {
        str(version): fetch_json(PYPI_RELEASE_URL.format(version=version))["info"]["classifiers"]
        for version in canary_targets(releases)
    }
    return build_plan(releases, classifiers, pyproject)


def _cmd_resolve(args: argparse.Namespace) -> int:
    print(json.dumps(resolve(args.pyproject), indent=2))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    rows = [] if args.skip else compare(load_rounds(args.baseline), load_rounds(args.target))
    result = {
        "baseline": args.baseline_version,
        "target": args.target_version,
        "skipped": args.skip,
        "python": args.python or None,
        "rows": [row.to_json() for row in rows],
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for name in regressions(rows):
        print(name)
    return 0


def _read_optional(path: Path, limit: int = MAX_READ_BYTES) -> str:
    """The last ``limit`` bytes of ``path`` as text, or "" when the step never wrote it."""
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - limit))
        return handle.read(limit).decode("utf-8", errors="replace")


def _cmd_record(args: argparse.Namespace) -> int:
    constrained_ok, unconstrained_ok = derive_install_flags(args.constrained, args.unconstrained)
    failed, failed_total = scan_junit(args.junit) if args.junit.is_file() else ((), 0)
    result = TargetResult(
        spark=args.spark,
        python=args.python or None,
        in_range=args.in_range == "true",
        constrained_ok=constrained_ok,
        unconstrained_ok=unconstrained_ok,
        tests_passed=TESTS_PASSED[args.tests],
        failed_tests=failed,
        failed_total=failed_total,
        packages=parse_freeze(_read_optional(args.freeze)),
        pip_error=_read_optional(args.pip_error),
    )
    would_file_install = not unconstrained_ok or (result.in_range and not constrained_ok)
    if result.python is not None and would_file_install and classify_pip_error(result.pip_error) == "transient":
        print(
            f"pyspark {result.spark}: pip failed without a resolution error (network or index trouble?); "
            "not recording a result so this counts as a canary failure, not a dependency finding.",
            file=sys.stderr,
        )
        return 1
    args.out.write_text(json.dumps(result.to_json(), indent=2) + "\n")
    return 0


def _log_summary(line: str) -> None:
    """Print to the job log and, on Actions, to the run's step summary."""
    print(line)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(line + "\n\n")


def _target_findings(targets: list[dict], results_dir: Path, run_url: str) -> tuple[list[Finding], bool]:
    findings: list[Finding] = []
    failed = False
    for target in targets:
        path = results_dir / f"result-{target['spark']}.json"
        try:
            result = TargetResult.from_json(json.loads(path.read_text()))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"pyspark {target['spark']}: no usable result at {path}: {exc!r}", file=sys.stderr)
            failed = True
            continue
        if not result.in_range and not is_installable(result):
            _log_summary(
                f"pyspark {result.spark} (outside the declared range) could not be installed; "
                "informational only, no issue filed."
            )
        findings.extend(findings_for_target(result, run_url))
    return findings, failed


def _bench_findings(comparison: Path, run_url: str) -> tuple[list[Finding], bool]:
    """Regression findings from ``comparison``; ``freeze-base.txt`` and ``freeze-target.txt`` beside it are optional."""
    try:
        data = json.loads(comparison.read_text())
        if data.get("skipped"):
            _log_summary(f"Benchmark comparison skipped: {data['skipped']}")
            return [], False
        rows = [Comparison.from_json(row) for row in data["rows"]]
        findings = findings_for_bench(
            rows,
            data["baseline"],
            data["target"],
            run_url,
            baseline_packages=parse_freeze(_read_optional(comparison.parent / "freeze-base.txt")),
            target_packages=parse_freeze(_read_optional(comparison.parent / "freeze-target.txt")),
            python=data.get("python"),
        )
        return findings, False
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"benchmark comparison {comparison} unusable: {exc!r}", file=sys.stderr)
        return [], True


def _file_all(findings: list[Finding], dry_run: bool) -> bool:
    """Upsert every finding; returns True when any upsert failed."""
    failed = False
    for finding in findings:
        try:
            print(f"{upsert(finding, gh, dry_run=dry_run)}: {finding.title}")
        except Exception as exc:  # noqa: BLE001 - keep filing the rest, then fail the run
            print(f"failed to file {finding.title!r}: {exc}", file=sys.stderr)
            failed = True
    return failed


def _cmd_report(args: argparse.Namespace) -> int:
    target_findings, targets_failed = _target_findings(args.targets, args.results_dir, args.run_url)
    bench_findings, bench_failed = _bench_findings(args.comparison, args.run_url)
    upsert_failed = _file_all(target_findings + bench_findings, args.dry_run)
    return 1 if targets_failed or bench_failed or upsert_failed else 0


def _cmd_report_failure(args: argparse.Namespace) -> int:
    finding = workflow_failure_finding(args.needs, args.run_url)
    print(f"{upsert(finding, gh, dry_run=args.dry_run)}: {finding.title}")
    return 0


def _json_arg(kind: type) -> Callable[[str], object]:
    """argparse ``type=`` that parses JSON and checks the top-level type."""

    def parse(text: str) -> object:
        value = json.loads(text)
        if not isinstance(value, kind):
            raise argparse.ArgumentTypeError(f"expected a JSON {kind.__name__}, got {type(value).__name__}")
        return value

    parse.__name__ = f"JSON {kind.__name__}"
    return parse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spark_canary", description="Spark release canary helpers")
    commands = parser.add_subparsers(dest="command", required=True)

    resolve_cmd = commands.add_parser("resolve", help="print the canary target plan as JSON")
    resolve_cmd.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    resolve_cmd.set_defaults(func=_cmd_resolve)

    compare_cmd = commands.add_parser(
        "compare", help="compare benchmark medians, write JSON, print regressed node ids one per line"
    )
    compare_cmd.add_argument("--baseline", type=Path, nargs="+", help="baseline --benchmark-json files")
    compare_cmd.add_argument("--target", type=Path, nargs="+", help="target --benchmark-json files")
    compare_cmd.add_argument("--baseline-version")
    compare_cmd.add_argument("--target-version")
    compare_cmd.add_argument("--python", help="Python both sides ran on; empty means unknown")
    compare_cmd.add_argument("--skip", metavar="REASON", help="record that the comparison was skipped and why")
    compare_cmd.add_argument("--out", type=Path, required=True)
    compare_cmd.set_defaults(func=_cmd_compare)

    record_cmd = commands.add_parser("record", help="write one compatibility target's result JSON")
    record_cmd.add_argument("--spark", required=True)
    record_cmd.add_argument("--python", required=True, help='planned Python, or "" when none is supported')
    record_cmd.add_argument("--in-range", required=True, choices=("true", "false"))
    for step in ("constrained", "unconstrained", "tests"):
        record_cmd.add_argument(f"--{step}", required=True, choices=OUTCOMES, help="GitHub steps.<id>.outcome")
    record_cmd.add_argument("--junit", type=Path, required=True)
    record_cmd.add_argument("--freeze", type=Path, required=True)
    record_cmd.add_argument("--pip-error", type=Path, required=True)
    record_cmd.add_argument("--out", type=Path, required=True)
    record_cmd.set_defaults(func=_cmd_record)

    report_cmd = commands.add_parser("report", help="file or update issues for every finding")
    report_cmd.add_argument("--targets", type=_json_arg(list), required=True, help="the resolve plan's targets list")
    report_cmd.add_argument("--results-dir", type=Path, required=True)
    report_cmd.add_argument("--comparison", type=Path, required=True)
    report_cmd.add_argument("--run-url", required=True)
    report_cmd.add_argument("--dry-run", action="store_true")
    report_cmd.set_defaults(func=_cmd_report)

    failure_cmd = commands.add_parser("report-failure", help="file or update the workflow-failing issue")
    failure_cmd.add_argument("--needs", type=_json_arg(dict), required=True, help="toJSON(needs)")
    failure_cmd.add_argument("--run-url", required=True)
    failure_cmd.add_argument("--dry-run", action="store_true")
    failure_cmd.set_defaults(func=_cmd_report_failure)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "compare" and not args.skip and not (args.baseline and args.target):
        parser.error("compare needs --baseline and --target files unless --skip is given")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
