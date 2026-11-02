"""Turn canary results into issue findings: title, labels, Markdown body and a noise-guard fingerprint."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .bench import Comparison

Outcome = Literal["success", "failure", "skipped"]
PipErrorKind = Literal["resolution", "transient"]

CANARY_LABEL = "spark-canary"
MAX_BODY_CHARS = 60_000
MAX_FAILED_TESTS = 50
MAX_JUNIT_FAILURES = 1_000
MAX_PIP_ERROR_LINES = 100
MAX_PIP_ERROR_CHARS = 20_000
TRACKED_PACKAGES = ("pyspark", "numpy", "pandas", "pyarrow")
# Only these mean "the network or the index let us down"; anything else pip prints is a real install failure.
TRANSIENT_PATTERN = re.compile(
    r"ReadTimeoutError|ConnectTimeoutError|ConnectionError|NewConnectionError|Max retries exceeded"
    r"|Temporary failure in name resolution|ProtocolError|HTTP error (?:5\d\d|429)|Too Many Requests"
    r"|\b5\d\d (?:Server Error|Service Unavailable|Bad Gateway|Gateway Time-?out|Internal Server Error)"
)
REASON_MARKERS = (
    "ResolutionImpossible",
    "No matching distribution found",
    "Could not find a version that satisfies",
    "requires a different Python",
)
PASSES_OUTSIDE_SUFFIX = " passes, outside the declared range"
WORKFLOW_FAILING_TITLE = "Spark canary: workflow failing"
TRUNCATED_NOTE = "\n\n_(body truncated to fit GitHub's size limit; see the run for the rest)_"
ZERO_WIDTH_SPACE = "\u200b"

DOCS_CHECKLIST = (
    "### When fixing\n"
    "- [ ] Update the `pyspark` range in `pyproject.toml` if support changes\n"
    "- [ ] Update the Compatibility table in `README.md`\n"
    "- [ ] Update the Compatibility Matrix in `docs/quickstart.rst`"
)


def tests_fail_title(version: str) -> str:
    return f"Spark canary: tests fail on pyspark {version}"


def regression_title(target: str, baseline: str) -> str:
    return f"Spark canary: performance regression on pyspark {target} vs {baseline}"


def cannot_install_title(version: str) -> str:
    return f"Spark canary: spark-bestfit[spark] cannot install with pyspark {version}"


def passes_outside_title(version: str) -> str:
    return f"Spark canary: pyspark {version}{PASSES_OUTSIDE_SUFFIX}"


def is_passes_outside_title(title: str) -> bool:
    """Titles deduped across all issue states, so a closed one is never recreated."""
    return title.startswith("Spark canary: pyspark ") and title.endswith(PASSES_OUTSIDE_SUFFIX)


def derive_install_flags(constrained: Outcome, unconstrained: Outcome) -> tuple[bool, bool]:
    """``(constrained_ok, unconstrained_ok)`` from GitHub step outcomes; a constrained success counts for both."""
    constrained_ok = constrained == "success"
    return constrained_ok, constrained_ok or unconstrained == "success"


def classify_pip_error(text: str) -> PipErrorKind:
    """``transient`` only for known network or index trouble; build and resolve failures are ``resolution``.

    A resolver verdict wins over network noise: pip logs retry warnings and still reaches a real answer.
    """
    if any(marker in text for marker in REASON_MARKERS):
        return "resolution"
    return "transient" if TRANSIENT_PATTERN.search(text) else "resolution"


@dataclass(frozen=True)
class TargetResult:
    """What the compatibility job saw for one PySpark release.

    ``failed_tests`` may be capped while parsing JUnit; ``failed_total`` keeps the real count
    (``None`` means ``len(failed_tests)``).
    """

    spark: str
    python: str | None
    in_range: bool
    constrained_ok: bool
    unconstrained_ok: bool
    tests_passed: bool | None
    failed_tests: tuple[str, ...]
    packages: dict[str, str] = field(hash=False)
    pip_error: str
    failed_total: int | None = None

    @property
    def failed_count(self) -> int:
        return len(self.failed_tests) if self.failed_total is None else self.failed_total

    def to_json(self) -> dict[str, Any]:
        return {
            "spark": self.spark,
            "python": self.python,
            "in_range": self.in_range,
            "constrained_ok": self.constrained_ok,
            "unconstrained_ok": self.unconstrained_ok,
            "tests_passed": self.tests_passed,
            "failed_tests": list(self.failed_tests),
            "failed_total": self.failed_total,
            "packages": dict(self.packages),
            "pip_error": self.pip_error,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> TargetResult:
        return cls(
            spark=d["spark"],
            python=d["python"],
            in_range=d["in_range"],
            constrained_ok=d["constrained_ok"],
            unconstrained_ok=d["unconstrained_ok"],
            tests_passed=d["tests_passed"],
            failed_tests=tuple(d["failed_tests"]),
            failed_total=d["failed_total"],
            packages=dict(d["packages"]),
            pip_error=d["pip_error"],
        )


@dataclass(frozen=True)
class Finding:
    """One issue to create or comment on, keyed by its exact title."""

    title: str
    labels: tuple[str, ...]
    body: str
    fingerprint: str


def scan_junit(junit: Path, limit: int = MAX_JUNIT_FAILURES) -> tuple[tuple[str, ...], int]:
    """``(first `limit` failing ids, total failing count)``; ids are ``classname::name`` of testcases
    with a ``failure`` or ``error``, in report order and deduped."""
    kept: dict[str, None] = {}
    seen: set[str] = set()
    for _, case in ET.iterparse(junit):
        if case.tag != "testcase":
            continue
        if case.find("failure") is not None or case.find("error") is not None:
            test_id = f"{case.get('classname', '')}::{case.get('name', '')}"
            if test_id not in seen:
                seen.add(test_id)
                if len(kept) < limit:
                    kept[test_id] = None
        case.clear()
    return tuple(kept), len(seen)


def failed_tests(junit: Path) -> tuple[str, ...]:
    """``classname::name`` of testcases with ``failure`` or ``error``, capped at ``MAX_JUNIT_FAILURES``."""
    return scan_junit(junit)[0]


def parse_freeze(text: str) -> dict[str, str]:
    """Versions of the packages the report shows, from ``pip freeze`` output."""
    versions = {}
    for line in text.splitlines():
        name, sep, version = line.strip().partition("==")
        canonical = name.strip().lower().replace("_", "-")
        if sep and canonical in TRACKED_PACKAGES:
            versions[canonical] = version.strip()
    return versions


def fingerprint_marker(fingerprint: str) -> str:
    return f"<!-- canary-fingerprint: {fingerprint} -->"


def _fingerprint(details: str) -> str:
    return hashlib.sha256(details.encode("utf-8")).hexdigest()


# --- Markdown safety for text that came from pip, pytest or PyPI ---


def _defuse(text: str) -> str:
    """Break ``@mentions`` so issue text never pings anyone."""
    return text.replace("@", "@" + ZERO_WIDTH_SPACE)


def _break_fences(text: str) -> str:
    """Split runs of three or more backticks so untrusted text can never close our code fence."""
    return re.sub(r"`{3,}", lambda m: ZERO_WIDTH_SPACE.join(m.group(0)), text)


def _cell(text: str) -> str:
    """Plain table cell: no mentions, no column breaks, one line."""
    return _defuse(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _code(text: str, in_table: bool = False) -> str:
    """Inline code span that survives backticks (and ``|`` inside tables) in ``text``."""
    text = _defuse(text).replace("\r", " ").replace("\n", " ")
    if in_table:
        text = text.replace("|", "\\|")
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    delimiter = "`" * (longest + 1)
    pad = " " if longest else ""
    return f"{delimiter}{pad}{text}{pad}{delimiter}"


def _open_fence(main: str) -> bool:
    return sum(1 for line in main.splitlines() if line.startswith("```")) % 2 == 1


def _assemble(details: str, sections: list[str]) -> tuple[str, str]:
    """``(body, fingerprint)``: marker first, docs checklist last, total under ``MAX_BODY_CHARS``."""
    fingerprint = _fingerprint(details)
    head = fingerprint_marker(fingerprint) + "\n"
    tail = "\n\n" + DOCS_CHECKLIST
    main = "\n\n".join(sections)
    budget = MAX_BODY_CHARS - 1 - len(head) - len(tail) - len(TRUNCATED_NOTE) - len("\n```")
    if len(head) + len(main) + len(tail) >= MAX_BODY_CHARS:
        main = main[:budget]
        if _open_fence(main):
            main += "\n```"
        main += TRUNCATED_NOTE
    return head + main + tail, fingerprint


def _environment(r: TargetResult, run_url: str) -> str:
    def package(name: str) -> str:
        return r.packages.get(name, "not installed")

    where = "inside" if r.in_range else "outside"
    rows = [
        ("pyspark", r.packages.get("pyspark", r.spark)),
        ("Python", r.python or "none (no supported Python matches requires-python)"),
        ("numpy", package("numpy")),
        ("pandas", package("pandas")),
        ("pyarrow", package("pyarrow")),
    ]
    table = "\n".join(["| package | version |", "|---|---|"] + [f"| {name} | {_cell(value)} |" for name, value in rows])
    return f"pyspark {r.spark} is {where} the declared range in `pyproject.toml`.\n\n{table}\n\nRun: {run_url}"


def _tests_result(tests_passed: bool | None) -> str:
    if tests_passed is None:
        return "tests did not run"
    return "tests passed" if tests_passed else "tests failed"


def _pip_error_tail(text: str) -> str:
    lines = text.rstrip().splitlines()
    tail = "\n".join(lines[-MAX_PIP_ERROR_LINES:])[-MAX_PIP_ERROR_CHARS:]
    note = f"Last {MAX_PIP_ERROR_LINES} of {len(lines)} lines:\n" if len(lines) > MAX_PIP_ERROR_LINES else ""
    return f"### pip error\n{note}```text\n{_break_fences(_defuse(tail))}\n```"


def _pip_error_details(text: str) -> str:
    """The lines that say why pip gave up, so download noise and paths don't change the fingerprint."""
    lines = [line.strip() for line in text.splitlines()]
    reasons = [line for line in lines if line.lower().startswith("error:") or any(m in line for m in REASON_MARKERS)]
    return "\n".join([classify_pip_error(text), *dict.fromkeys(reasons)])


def _failed_tests_section(r: TargetResult) -> tuple[str, list[str]]:
    """``(Markdown section, sorted ids)``; lists at most ``MAX_FAILED_TESTS`` then "... and N more"."""
    names = sorted(r.failed_tests)
    listed = [f"- {_code(name)}" for name in names[:MAX_FAILED_TESTS]]
    if r.failed_count > len(listed):
        listed.append(f"- ... and {r.failed_count - len(listed)} more")
    if not listed:
        listed = ["No failing test ids were found in the JUnit report; see the run log."]
    return f"### Failed tests ({r.failed_count})\n" + "\n".join(listed), names


def _cannot_install(r: TargetResult, run_url: str) -> Finding:
    sections: list[str] = []
    if r.python is None:
        reason = "no supported Python: no version in the pyspark classifiers is allowed by `requires-python`"
        details = "no python"
    elif not r.unconstrained_ok:
        reason = "pip could not install pyspark, even without the project's caps on numpy, pandas and pyarrow."
        details = "unconstrained\n" + _pip_error_details(r.pip_error)
    else:
        result = _tests_result(r.tests_passed)
        reason = (
            "`pip install spark-bestfit[spark]` could not resolve with this pyspark. "
            f"The constrained install failed, so the code was still tested without project caps: {result}."
        )
        details = f"constrained\n{result}\n" + _pip_error_details(r.pip_error)
        if r.tests_passed is False:
            failed_section, names = _failed_tests_section(r)
            sections.append(failed_section)
            details += f"\nfailed {r.failed_count}\n" + "\n".join(names)
    sections = [reason, _environment(r, run_url), *sections]
    if r.pip_error.strip():
        sections.append(_pip_error_tail(r.pip_error))
    body, fingerprint = _assemble(details, sections)
    return Finding(cannot_install_title(r.spark), (CANARY_LABEL, "dependencies"), body, fingerprint)


def _tests_fail(r: TargetResult, run_url: str) -> Finding:
    failed_section, names = _failed_tests_section(r)
    sections = [f"`pytest -m spark` fails on pyspark {r.spark}.", _environment(r, run_url), failed_section]
    body, fingerprint = _assemble(f"failed {r.failed_count}\n" + "\n".join(names), sections)
    return Finding(tests_fail_title(r.spark), (CANARY_LABEL, "tests"), body, fingerprint)


def _passes_outside(r: TargetResult, run_url: str) -> Finding:
    sections = [
        f"pyspark {r.spark} passes `pytest -m spark` but is outside the declared `pyspark` range, "
        "so the range can probably be widened.",
        _environment(r, run_url),
    ]
    body, fingerprint = _assemble("passes outside the declared range", sections)
    return Finding(passes_outside_title(r.spark), (CANARY_LABEL, "dependencies"), body, fingerprint)


def is_installable(r: TargetResult) -> bool:
    return r.python is not None and r.unconstrained_ok


def findings_for_target(r: TargetResult, run_url: str) -> list[Finding]:
    """At most one finding per target: the first matching rule wins.

    Install trouble outside the declared range files nothing (``report`` logs it instead).
    """
    if r.in_range and not is_installable(r):
        return [_cannot_install(r, run_url)]
    if r.in_range and not r.constrained_ok:
        return [_cannot_install(r, run_url)]
    if not r.in_range and not is_installable(r):
        return []
    if r.unconstrained_ok and r.tests_passed is False:
        return [_tests_fail(r, run_url)]
    if not r.in_range and r.tests_passed is True:
        return [_passes_outside(r, run_url)]
    return []


def _median(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _bench_table(heading: str, rows: list[Comparison]) -> str:
    lines = [f"### {heading} ({len(rows)})", "| benchmark | baseline median (s) | target median (s) | ratio |"]
    lines.append("|---|---|---|---|")
    for row in rows:
        ratio = "n/a" if row.ratio is None else f"{row.ratio:.2f}"
        name = _code(row.name, in_table=True)
        lines.append(f"| {name} | {_median(row.baseline_median)} | {_median(row.target_median)} | {ratio} |")
    return "\n".join(lines)


def _bench_packages(baseline: Mapping[str, str], target: Mapping[str, str]) -> str:
    rows = [
        f"| {name} | {_cell(baseline.get(name, 'not installed'))} | {_cell(target.get(name, 'not installed'))} |"
        for name in TRACKED_PACKAGES
    ]
    return "\n".join(["| package | baseline | target |", "|---|---|---|", *rows])


def findings_for_bench(
    rows: list[Comparison],
    baseline: str,
    target: str,
    run_url: str,
    baseline_packages: Mapping[str, str] | None = None,
    target_packages: Mapping[str, str] | None = None,
    python: str | None = None,
) -> list[Finding]:
    """One regression finding when any benchmark regressed; speedups and missing rows are listed, never filed.

    The fingerprint covers only which benchmarks regressed, so run-to-run ratio jitter doesn't re-comment.
    """
    by_status = {status: [r for r in rows if r.status == status] for status in ("regression", "speedup", "missing")}
    if not by_status["regression"]:
        return []
    same_python = f"both on Python {_cell(str(python))}" if python else "same Python"
    sections = [
        f"Benchmarks on pyspark {target} are more than 10% slower than on pyspark {baseline} "
        f"(median over interleaved passes, {same_python} and runner).",
        f"Run: {run_url}",
    ]
    if baseline_packages or target_packages:
        sections.append(_bench_packages(baseline_packages or {}, target_packages or {}))
    sections.append(_bench_table("Regressions", by_status["regression"]))
    if by_status["speedup"]:
        sections.append(_bench_table("Speedups", by_status["speedup"]))
    if by_status["missing"]:
        sections.append(_bench_table("Missing on one side", by_status["missing"]))
    details = "\n".join(sorted(row.name for row in by_status["regression"]))
    body, fingerprint = _assemble(details, sections)
    return [Finding(regression_title(target, baseline), (CANARY_LABEL, "performance"), body, fingerprint)]


def workflow_failure_finding(needs: Mapping[str, Mapping[str, Any]], run_url: str) -> Finding:
    """The canary itself broke; ``needs`` is the workflow's ``toJSON(needs)``."""
    jobs = sorted((name, str(job.get("result", "unknown"))) for name, job in needs.items())
    rows = [f"| {_cell(name)} | {_cell(result)} |" for name, result in jobs]
    table = "\n".join(["| job | result |", "|---|---|", *rows])
    sections = [f"The Spark canary run did not finish cleanly.\n\nRun: {run_url}", table]
    body, fingerprint = _assemble("\n".join(f"{name}: {result}" for name, result in jobs), sections)
    return Finding(WORKFLOW_FAILING_TITLE, (CANARY_LABEL, "ci"), body, fingerprint)
