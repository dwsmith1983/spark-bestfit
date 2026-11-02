"""Resolve which PySpark releases the canary should test, and with which Python."""

from __future__ import annotations

import json
import re
import tomllib
import urllib.request
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

DEFAULT_FLOOR = Version("3.5")
SETUPTOOLS_SPARK_BELOW = Version("4.0")
SETUPTOOLS_PYTHON_FROM = Version("3.12")
HTTP_OK = 200

_PYTHON_CLASSIFIER = re.compile(r"^Programming Language :: Python :: (3\.\d+)$")


def stable_versions(releases: dict[str, list[dict]]) -> list[Version]:
    """Final releases that have at least one file that is not yanked, ascending."""
    stable = []
    for raw, files in releases.items():
        version = Version(raw)
        if version.is_prerelease or version.is_devrelease:
            continue
        if not any(not f.get("yanked", False) for f in files):
            continue
        stable.append(version)
    return sorted(stable)


def newest_per_minor(versions: list[Version], floor: Version = DEFAULT_FLOOR) -> list[Version]:
    """Highest patch of each major.minor at or above ``floor``, ascending."""
    newest: dict[tuple[int, int], Version] = {}
    for version in versions:
        if version < floor:
            continue
        key = (version.major, version.minor)
        if key not in newest or version > newest[key]:
            newest[key] = version
    return sorted(newest.values())


def benchmark_pair(targets: list[Version]) -> tuple[Version, Version]:
    """Return ``(baseline, target)``: the two newest targets."""
    if len(targets) < 2:
        raise ValueError(f"benchmark needs at least two Spark versions, got {len(targets)}")
    ordered = sorted(targets)
    return ordered[-2], ordered[-1]


def _load_project(pyproject: Path) -> dict[str, Any]:
    with pyproject.open("rb") as handle:
        return tomllib.load(handle)["project"]


def declared_range(pyproject: Path) -> SpecifierSet:
    """The ``pyspark`` specifier from the ``spark`` optional-dependency group."""
    extras = _load_project(pyproject).get("optional-dependencies", {})
    for line in extras.get("spark", []):
        requirement = Requirement(line)
        if canonicalize_name(requirement.name) == "pyspark":
            return requirement.specifier
    raise ValueError(f"no pyspark requirement in [project.optional-dependencies].spark of {pyproject}")


def project_python(pyproject: Path) -> SpecifierSet:
    """The project's ``requires-python`` specifier."""
    requires = _load_project(pyproject).get("requires-python")
    if not requires:
        raise ValueError(f"no requires-python in [project] of {pyproject}")
    return SpecifierSet(requires)


def pick_python(classifiers: list[str], allowed: SpecifierSet) -> str | None:
    """Newest ``Programming Language :: Python :: 3.X`` classifier that ``allowed`` accepts."""
    candidates = []
    for classifier in classifiers:
        match = _PYTHON_CLASSIFIER.match(classifier)
        if match and allowed.contains(match.group(1)):
            candidates.append(Version(match.group(1)))
    if not candidates:
        return None
    return str(max(candidates))


def needs_setuptools(spark: Version, python: str) -> bool:
    """PySpark 3.x imports distutils, which Python 3.12 removed; setuptools supplies it."""
    return spark < SETUPTOOLS_SPARK_BELOW and Version(python) >= SETUPTOOLS_PYTHON_FROM


def canary_targets(releases: dict[str, list[dict]]) -> list[Version]:
    """The Spark versions the canary runs against."""
    return newest_per_minor(stable_versions(releases))


def build_plan(
    releases: dict[str, list[dict]],
    classifiers_by_version: dict[str, list[str]],
    pyproject: Path,
) -> dict[str, Any]:
    """Build the canary plan: one entry per target plus the benchmark pair."""
    spark_range = declared_range(pyproject)
    allowed_python = project_python(pyproject)
    targets = canary_targets(releases)

    entries = []
    for spark in targets:
        python = pick_python(classifiers_by_version[str(spark)], allowed_python)
        entries.append(
            {
                "spark": str(spark),
                "python": python,
                "in_range": spark_range.contains(spark),
                "setuptools": python is not None and needs_setuptools(spark, python),
            }
        )

    baseline, target = benchmark_pair(targets)
    shared = sorted(set(classifiers_by_version[str(baseline)]) & set(classifiers_by_version[str(target)]))
    bench = {"baseline": str(baseline), "target": str(target), "python": pick_python(shared, allowed_python)}
    return {"targets": entries, "bench": bench}


def fetch_json(url: str, timeout: float = 30.0) -> dict:
    """GET ``url`` over HTTPS and decode the JSON body; raise on any non-200 reply."""
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-https URL: {url}")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        if response.status != HTTP_OK:
            raise RuntimeError(f"GET {url} returned HTTP {response.status}")
        return json.load(response)
