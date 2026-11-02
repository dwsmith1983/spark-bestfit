"""Tests for the Spark canary version resolution helpers."""

import json
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version as V

from spark_canary import __main__ as cli
from spark_canary.versions import (
    benchmark_pair,
    build_plan,
    declared_range,
    fetch_json,
    needs_setuptools,
    newest_per_minor,
    pick_python,
    stable_versions,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "spark_canary" / "pyspark_releases.json"


def _pyproject(tmp_path: Path, spark_range: str = ">=3.5.0,<5.0.0") -> Path:
    """A minimal pyproject.toml, so these tests don't follow the real file's caps."""
    path = tmp_path / "pyproject.toml"
    path.write_text(
        "[project]\n"
        'name = "demo"\n'
        'requires-python = ">=3.11,<3.14"\n'
        "[project.optional-dependencies]\n"
        f'spark = ["pyspark{spark_range}", "pyarrow>=12.0.0"]\n'
    )
    return path


def _classifiers(*pythons: str) -> list[str]:
    return ["Development Status :: 5 - Production/Stable"] + [f"Programming Language :: Python :: {p}" for p in pythons]


@pytest.fixture
def releases() -> dict[str, list[dict]]:
    return json.loads(FIXTURE.read_text())


def test_stable_versions_drops_prerelease_dev_empty_and_yanked(releases):
    result = stable_versions(releases)

    for dropped in ("4.0.0.dev1", "4.1.0rc1", "4.4.0.dev1", "4.2.2", "4.1.5"):
        assert V(dropped) not in result
    assert V("3.4.4") in result
    assert V("4.3.0") in result


def test_newest_per_minor_from_floor(releases):
    result = newest_per_minor(stable_versions(releases))

    assert result == [V("3.5.7"), V("4.0.5"), V("4.1.4"), V("4.2.1"), V("4.3.0")]


def test_benchmark_pair_is_last_two():
    targets = [V("3.5.7"), V("4.0.5"), V("4.1.4"), V("4.2.1"), V("4.3.0")]

    assert benchmark_pair(targets) == (V("4.2.1"), V("4.3.0"))
    assert benchmark_pair([V("4.3.0"), V("5.0.0")]) == (V("4.3.0"), V("5.0.0"))


def test_benchmark_pair_needs_two():
    with pytest.raises(ValueError):
        benchmark_pair([V("4.3.0")])


def test_declared_range_reads_spark_extra(tmp_path):
    rng = declared_range(_pyproject(tmp_path))

    assert "4.3.0" in rng
    assert "5.0.0" not in rng


def test_pick_python_newest_allowed():
    allowed = SpecifierSet(">=3.11,<3.14")

    assert pick_python(_classifiers("3.11", "3.12", "3.13", "3.14"), allowed) == "3.13"
    assert pick_python(_classifiers("3.8", "3.9", "3.10"), allowed) is None


def test_pick_python_orders_numerically_not_lexically():
    # PyPI lists "3.10" before "3.8"; a string max would pick "3.9".
    assert pick_python(_classifiers("3.10", "3.8", "3.9"), SpecifierSet(">=3.8")) == "3.10"


def test_needs_setuptools():
    assert needs_setuptools(V("3.5.7"), "3.12") is True
    assert needs_setuptools(V("3.5.7"), "3.11") is False
    assert needs_setuptools(V("4.0.5"), "3.13") is False


def test_build_plan_shape_and_in_range(releases, tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\n"
        'name = "demo"\n'
        'requires-python = ">=3.11,<3.14"\n'
        "[project.optional-dependencies]\n"
        'spark = ["pyspark>=3.5.0,<4.3.0", "pyarrow>=12.0.0"]\n'
    )
    classifiers = {
        "3.5.7": _classifiers("3.8", "3.9", "3.10"),
        "4.0.5": _classifiers("3.9", "3.10", "3.11", "3.12", "3.13"),
        "4.1.4": _classifiers("3.10", "3.11", "3.12", "3.13", "3.14"),
        "4.2.1": _classifiers("3.10", "3.11", "3.12"),
        "4.3.0": _classifiers("3.11", "3.12", "3.13", "3.14"),
    }

    plan = build_plan(releases, classifiers, pyproject)

    by_spark = {t["spark"]: t for t in plan["targets"]}
    assert [t["spark"] for t in plan["targets"]] == ["3.5.7", "4.0.5", "4.1.4", "4.2.1", "4.3.0"]
    assert by_spark["4.3.0"] == {"spark": "4.3.0", "python": "3.13", "in_range": False, "setuptools": False}
    assert by_spark["4.2.1"] == {"spark": "4.2.1", "python": "3.12", "in_range": True, "setuptools": False}
    assert by_spark["3.5.7"]["python"] is None
    assert by_spark["3.5.7"]["setuptools"] is False
    # Bench python comes from the classifiers both releases share.
    assert plan["bench"] == {"baseline": "4.2.1", "target": "4.3.0", "python": "3.12"}


def test_build_plan_flags_setuptools_for_spark35_on_new_python(releases, tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\n"
        'name = "demo"\n'
        'requires-python = ">=3.11,<3.14"\n'
        "[project.optional-dependencies]\n"
        'spark = ["pyspark>=3.5.0,<5.0.0"]\n'
    )
    classifiers = {v: _classifiers("3.11", "3.12") for v in ("3.5.7", "4.0.5", "4.1.4", "4.2.1", "4.3.0")}

    plan = build_plan(releases, classifiers, pyproject)

    assert plan["targets"][0] == {"spark": "3.5.7", "python": "3.12", "in_range": True, "setuptools": True}


def test_fetch_json_rejects_non_https():
    with pytest.raises(ValueError):
        fetch_json("file:///etc/passwd")


def test_resolve_cli_prints_plan(releases, monkeypatch, capsys, tmp_path):
    def fake_fetch(url: str, timeout: float = 30.0) -> dict:
        if url == "https://pypi.org/pypi/pyspark/json":
            return {"releases": releases}
        version = url.split("/")[-2]
        return {"info": {"classifiers": _classifiers("3.11", "3.12", "3.13")}, "version": version}

    monkeypatch.setattr(cli, "fetch_json", fake_fetch)

    exit_code = cli.main(["resolve", "--pyproject", str(_pyproject(tmp_path))])

    plan = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert [t["spark"] for t in plan["targets"]] == ["3.5.7", "4.0.5", "4.1.4", "4.2.1", "4.3.0"]
    assert plan["bench"] == {"baseline": "4.2.1", "target": "4.3.0", "python": "3.13"}
