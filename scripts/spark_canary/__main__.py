"""Command line entry point: ``python -m spark_canary <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bench import compare, load_rounds, regressions
from .versions import build_plan, canary_targets, fetch_json

PYPI_INDEX_URL = "https://pypi.org/pypi/pyspark/json"
PYPI_RELEASE_URL = "https://pypi.org/pypi/pyspark/{version}/json"


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
        "rows": [row.to_json() for row in rows],
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for name in regressions(rows):
        print(name)
    return 0


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
    compare_cmd.add_argument("--skip", metavar="REASON", help="record that the comparison was skipped and why")
    compare_cmd.add_argument("--out", type=Path, required=True)
    compare_cmd.set_defaults(func=_cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "compare" and not args.skip and not (args.baseline and args.target):
        parser.error("compare needs --baseline and --target files unless --skip is given")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
