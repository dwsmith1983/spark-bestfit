"""Command line entry point: ``python -m spark_canary <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spark_canary", description="Spark release canary helpers")
    commands = parser.add_subparsers(dest="command", required=True)

    resolve_cmd = commands.add_parser("resolve", help="print the canary target plan as JSON")
    resolve_cmd.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    resolve_cmd.set_defaults(func=_cmd_resolve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
