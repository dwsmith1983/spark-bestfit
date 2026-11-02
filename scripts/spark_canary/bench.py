"""Compare pytest-benchmark results between a baseline and a target PySpark release."""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

REGRESSION_RATIO = 1.10
SPEEDUP_RATIO = 0.90

Status = Literal["regression", "speedup", "ok", "missing"]


def load_rounds(paths: list[Path]) -> dict[str, list[float]]:
    """Merge the per-round timings (``stats.data``) of each benchmark across ``--benchmark-json`` files.

    A missing file raises ``FileNotFoundError``; a file that is not valid JSON, or has a benchmark
    without round data, raises ``ValueError`` naming the file.
    """
    rounds: dict[str, list[float]] = {}
    for path in paths:
        try:
            benchmarks = json.loads(path.read_text())["benchmarks"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}: not a pytest-benchmark JSON file ({exc!r})") from exc
        for bench in benchmarks:
            name = bench.get("fullname", "<unnamed>")
            data = bench.get("stats", {}).get("data")
            if not data:
                raise ValueError(f"{path}: benchmark {name} has no stats.data (run with --benchmark-json)")
            rounds.setdefault(name, []).extend(float(value) for value in data)
    return rounds


@dataclass(frozen=True)
class Comparison:
    """Median timing of one benchmark on each side; ``None`` when that side did not run it."""

    name: str
    baseline_median: float | None
    target_median: float | None

    @property
    def ratio(self) -> float | None:
        if self.baseline_median is None or self.target_median is None:
            return None
        return self.target_median / self.baseline_median

    @property
    def status(self) -> Status:
        ratio = self.ratio
        if ratio is None:
            return "missing"
        if ratio > REGRESSION_RATIO:
            return "regression"
        if ratio < SPEEDUP_RATIO:
            return "speedup"
        return "ok"

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "baseline_median": self.baseline_median, "target_median": self.target_median}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Comparison:
        return cls(name=d["name"], baseline_median=d["baseline_median"], target_median=d["target_median"])


def _median(rounds: list[float] | None) -> float | None:
    return statistics.median(rounds) if rounds else None


def compare(baseline: dict[str, list[float]], target: dict[str, list[float]]) -> list[Comparison]:
    """One row per benchmark that ran on either side, sorted by name."""
    return [
        Comparison(name, _median(baseline.get(name)), _median(target.get(name)))
        for name in sorted(baseline.keys() | target.keys())
    ]


def regressions(comparisons: Iterable[Comparison]) -> list[str]:
    """Full pytest node ids of the benchmarks that regressed."""
    return [row.name for row in comparisons if row.status == "regression"]
