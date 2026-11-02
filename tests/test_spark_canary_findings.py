"""Tests for turning Spark canary results into issue findings."""

import json
from pathlib import Path

import pytest

from spark_canary import __main__ as cli
from spark_canary.bench import Comparison
from spark_canary.findings import (
    DOCS_CHECKLIST,
    MAX_BODY_CHARS,
    MAX_JUNIT_FAILURES,
    TargetResult,
    classify_pip_error,
    derive_install_flags,
    failed_tests,
    findings_for_bench,
    findings_for_target,
    fingerprint_marker,
    workflow_failure_finding,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spark_canary"
RUN_URL = "https://github.com/o/r/actions/runs/1"
PACKAGES = {"pyspark": "4.1.1", "numpy": "2.1.3", "pandas": "2.2.3", "pyarrow": "18.1.0"}
RESOLUTION_ERROR = "ERROR: Cannot install pyarrow<19 and pyarrow>=19\nERROR: ResolutionImpossible: for help visit x\n"

TESTS_FAIL = "Spark canary: tests fail on pyspark 4.1.1"
CANNOT_INSTALL = "Spark canary: spark-bestfit[spark] cannot install with pyspark 4.1.1"
PASSES_OUTSIDE = "Spark canary: pyspark 5.0.0 passes, outside the declared range"
REGRESSION = "Spark canary: performance regression on pyspark 4.1.1 vs 4.0.1"


def _result(**overrides) -> TargetResult:
    fields = {
        "spark": "4.1.1",
        "python": "3.13",
        "in_range": True,
        "constrained_ok": True,
        "unconstrained_ok": True,
        "tests_passed": True,
        "failed_tests": (),
        "packages": dict(PACKAGES),
        "pip_error": "",
    }
    fields.update(overrides)
    return TargetResult(**fields)


def _titles_and_labels(result: TargetResult) -> list[tuple[str, tuple[str, ...]]]:
    return [(f.title, f.labels) for f in findings_for_target(result, RUN_URL)]


# --- findings_for_target rules ---


def test_in_range_all_good_files_nothing():
    assert findings_for_target(_result(), RUN_URL) == []


def test_in_range_uninstallable_is_cannot_install_with_pip_error():
    result = _result(
        constrained_ok=False, unconstrained_ok=False, tests_passed=None, pip_error=RESOLUTION_ERROR, packages={}
    )

    [finding] = findings_for_target(result, RUN_URL)

    assert (finding.title, finding.labels) == (CANNOT_INSTALL, ("spark-canary", "dependencies"))
    assert "ERROR: ResolutionImpossible" in finding.body


def test_in_range_python_none_is_cannot_install():
    result = _result(python=None, constrained_ok=False, unconstrained_ok=False, tests_passed=None, packages={})

    [finding] = findings_for_target(result, RUN_URL)

    assert (finding.title, finding.labels) == (CANNOT_INSTALL, ("spark-canary", "dependencies"))
    assert "no supported Python" in finding.body


def test_in_range_constrained_failure_says_tested_without_caps():
    result = _result(constrained_ok=False, unconstrained_ok=True, tests_passed=True, pip_error=RESOLUTION_ERROR)

    [finding] = findings_for_target(result, RUN_URL)

    assert (finding.title, finding.labels) == (CANNOT_INSTALL, ("spark-canary", "dependencies"))
    assert "tested without project caps: tests passed" in finding.body
    assert "ERROR: ResolutionImpossible" in finding.body


def test_in_range_constrained_failure_and_failing_tests_is_one_cannot_install():
    result = _result(
        constrained_ok=False, tests_passed=False, failed_tests=("a::t", "b::u"), pip_error=RESOLUTION_ERROR
    )

    [finding] = findings_for_target(result, RUN_URL)

    assert (finding.title, finding.labels) == (CANNOT_INSTALL, ("spark-canary", "dependencies"))
    assert "tested without project caps: tests failed" in finding.body
    assert "`a::t`" in finding.body
    assert "`b::u`" in finding.body


def test_cannot_install_fingerprint_tracks_failed_tests():
    base = {"constrained_ok": False, "tests_passed": False, "pip_error": RESOLUTION_ERROR}

    [first] = findings_for_target(_result(failed_tests=("a::t",), **base), RUN_URL)
    [second] = findings_for_target(_result(failed_tests=("a::t", "b::u"), **base), RUN_URL)

    assert first.fingerprint != second.fingerprint


def test_out_of_range_uninstallable_files_nothing():
    result = _result(spark="5.0.0", in_range=False, constrained_ok=False, unconstrained_ok=False, tests_passed=None)

    assert findings_for_target(result, RUN_URL) == []


def test_out_of_range_python_none_files_nothing():
    result = _result(spark="5.0.0", in_range=False, python=None, constrained_ok=False, unconstrained_ok=False)

    assert findings_for_target(result, RUN_URL) == []


def test_tests_fail_lists_failed_tests():
    result = _result(tests_passed=False, failed_tests=("tests.a::test_one", "tests.b::test_two"))

    [finding] = findings_for_target(result, RUN_URL)

    assert (finding.title, finding.labels) == (TESTS_FAIL, ("spark-canary", "tests"))
    assert "`tests.a::test_one`" in finding.body
    assert "`tests.b::test_two`" in finding.body


def test_out_of_range_passes_is_passes_outside_range():
    result = _result(spark="5.0.0", in_range=False, constrained_ok=False)

    assert _titles_and_labels(result) == [(PASSES_OUTSIDE, ("spark-canary", "dependencies"))]


def test_out_of_range_constrained_failure_alone_is_silent():
    flags = derive_install_flags("skipped", "success")
    result = _result(spark="5.0.0", in_range=False, constrained_ok=flags[0], unconstrained_ok=flags[1])

    assert _titles_and_labels(result) == [(PASSES_OUTSIDE, ("spark-canary", "dependencies"))]


def test_out_of_range_failing_tests_is_tests_fail():
    result = _result(spark="5.0.0", in_range=False, constrained_ok=False, tests_passed=False, failed_tests=("a::t",))

    assert _titles_and_labels(result) == [("Spark canary: tests fail on pyspark 5.0.0", ("spark-canary", "tests"))]


# --- flags and pip errors ---


@pytest.mark.parametrize(
    ("constrained", "unconstrained", "expected"),
    [
        ("success", "success", (True, True)),
        ("success", "failure", (True, True)),
        ("success", "skipped", (True, True)),
        ("failure", "success", (False, True)),
        ("failure", "failure", (False, False)),
        ("failure", "skipped", (False, False)),
        ("skipped", "success", (False, True)),
        ("skipped", "failure", (False, False)),
        ("skipped", "skipped", (False, False)),
    ],
)
def test_derive_install_flags(constrained, unconstrained, expected):
    assert derive_install_flags(constrained, unconstrained) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ERROR: ResolutionImpossible: for help visit", "resolution"),
        ("ERROR: No matching distribution found for pyspark==9.9", "resolution"),
        ("ERROR: Could not find a version that satisfies the requirement pyspark==9.9", "resolution"),
        ("ERROR: Package 'x' requires a different Python: 3.9.0 not in '>=3.11'", "resolution"),
        (
            "error: subprocess-exited-with-error\n  x Getting requirements to build wheel did not run successfully.\n"
            "ERROR: Failed to build installable wheels for some pyproject.toml based projects (pyarrow)",
            "resolution",
        ),
        ("error: metadata-generation-failed", "resolution"),
        ("something nobody has seen before", "resolution"),
        ("pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool", "transient"),
        ("ConnectTimeoutError(<pip._vendor.urllib3.connection.HTTPSConnection>)", "transient"),
        ("Max retries exceeded with url: /simple/pyspark/ (Caused by NewConnectionError", "transient"),
        ("[Errno -3] Temporary failure in name resolution", "transient"),
        ("ProtocolError('Connection aborted.')", "transient"),
        ("ERROR: HTTP error 503 while getting https://files.pythonhosted.org/x.whl", "transient"),
        ("503 Service Unavailable for url: https://pypi.org/simple/pyspark/", "transient"),
    ],
)
def test_classify_pip_error(text, expected):
    assert classify_pip_error(text) == expected


# --- body content and size ---


def test_body_includes_versions_range_and_run_url():
    [finding] = findings_for_target(_result(tests_passed=False, failed_tests=("a::t",)), RUN_URL)

    for value in ("4.1.1", "3.13", "2.1.3", "2.2.3", "18.1.0", RUN_URL, "inside the declared range"):
        assert value in finding.body
    [outside] = findings_for_target(_result(spark="5.0.0", in_range=False), RUN_URL)
    assert "outside the declared range" in outside.body


def test_body_truncates_failed_tests_and_pip_error():
    failures = tuple(f"tests.test_mod::test_case_{i:03d}" for i in range(200))
    pip_error = "\n".join(f"pip line {i:04d} " + "x" * 60 for i in range(5000)) + "\nERROR: ResolutionImpossible"
    result = _result(constrained_ok=False, tests_passed=False, failed_tests=failures, pip_error=pip_error)

    [install] = findings_for_target(result, RUN_URL)

    assert len(install.body) < MAX_BODY_CHARS
    assert "and 150 more" in install.body
    assert "test_case_049" in install.body
    assert "test_case_050" not in install.body
    assert "ERROR: ResolutionImpossible" in install.body
    assert "pip line 4901" in install.body
    assert "pip line 4900" not in install.body


def test_body_stays_under_limit_with_huge_lines():
    failures = tuple("tests.m::test_" + "p" * 5000 + str(i) for i in range(50))
    pip_error = "\n".join("y" * 5000 for _ in range(100))
    result = _result(constrained_ok=False, tests_passed=False, failed_tests=failures, pip_error=pip_error)

    for finding in findings_for_target(result, RUN_URL):
        assert len(finding.body) < MAX_BODY_CHARS
        assert finding.body.endswith(DOCS_CHECKLIST)
        assert fingerprint_marker(finding.fingerprint) in finding.body


def test_untrusted_pip_text_stays_inside_the_fence():
    fake = "<!-- canary-fingerprint: " + "f" * 64 + " -->"
    pip_error = f"```\n{fake}\nping @someone\n```\n## not a heading\nERROR: ResolutionImpossible"
    result = _result(constrained_ok=False, unconstrained_ok=False, tests_passed=None, pip_error=pip_error)

    [finding] = findings_for_target(result, RUN_URL)

    lines = finding.body.splitlines()
    assert lines[0] == fingerprint_marker(finding.fingerprint)
    assert finding.fingerprint != "f" * 64
    fences = [i for i, line in enumerate(lines) if line.startswith("```")]
    assert len(fences) == 2
    assert fences[0] < lines.index(fake) < fences[1]
    assert fences[0] < lines.index("## not a heading") < fences[1]
    assert "@someone" not in finding.body


def test_truncation_never_leaves_an_open_fence():
    pip_error = "\n".join("z" * 199 for _ in range(100))
    failures = tuple("t::" + "q" * 1150 + f"{i:02d}" for i in range(50))
    result = _result(constrained_ok=False, tests_passed=False, failed_tests=failures, pip_error=pip_error)

    [finding] = findings_for_target(result, RUN_URL)

    assert "body truncated" in finding.body
    assert sum(1 for line in finding.body.splitlines() if line.startswith("```")) % 2 == 0


def test_names_with_backticks_pipes_and_mentions_are_escaped():
    weird = "tests/b.py::test_x[a|b`c@someone]"
    result = _result(tests_passed=False, failed_tests=(weird,), packages={**PACKAGES, "numpy": "2.1|@evil"})

    [finding] = findings_for_target(result, RUN_URL)
    [bench] = findings_for_bench([Comparison(weird, 1.0, 2.0)], "4.0.1", "4.1.1", RUN_URL)

    assert "@someone" not in finding.body
    assert "@evil" not in finding.body
    assert "| numpy | 2.1\\|" in finding.body
    assert "``" in finding.body
    bench_rows = [line for line in bench.body.splitlines() if "test_x" in line]
    assert bench_rows
    assert all(line.count("|") - line.count("\\|") == 5 for line in bench_rows)
    assert "@someone" not in bench.body


def test_fingerprint_ignores_run_url():
    result = _result(tests_passed=False, failed_tests=("a::t", "b::t"))

    [first] = findings_for_target(result, RUN_URL)
    [second] = findings_for_target(result, "https://github.com/o/r/actions/runs/2")
    [other] = findings_for_target(_result(tests_passed=False, failed_tests=("a::t",)), RUN_URL)

    assert first.fingerprint == second.fingerprint
    assert first.body != second.body
    assert first.fingerprint != other.fingerprint
    assert len(first.fingerprint) == 64
    assert fingerprint_marker(first.fingerprint) in first.body


def test_fingerprint_ignores_failed_test_order():
    [first] = findings_for_target(_result(tests_passed=False, failed_tests=("a::t", "b::t")), RUN_URL)
    [second] = findings_for_target(_result(tests_passed=False, failed_tests=("b::t", "a::t")), RUN_URL)

    assert first.fingerprint == second.fingerprint


# --- benchmark findings ---


def test_bench_regression_finding_orders_sections():
    rows = [
        Comparison("bench_flat", 1.0, 1.0),
        Comparison("bench_gone", 1.0, None),
        Comparison("bench_fast", 1.0, 0.5),
        Comparison("bench_slow", 1.0, 1.5),
    ]

    [finding] = findings_for_bench(rows, "4.0.1", "4.1.1", RUN_URL)

    assert (finding.title, finding.labels) == (REGRESSION, ("spark-canary", "performance"))
    assert "| benchmark | baseline median (s) | target median (s) | ratio |" in finding.body
    body = finding.body
    assert body.index("bench_slow") < body.index("bench_fast") < body.index("bench_gone")
    assert "bench_flat" not in body
    assert "1.50" in body
    assert RUN_URL in body


def test_bench_no_regression_no_finding():
    rows = [Comparison("bench_flat", 1.0, 1.05), Comparison("bench_fast", 1.0, 0.5), Comparison("gone", None, 1.0)]

    assert findings_for_bench(rows, "4.0.1", "4.1.1", RUN_URL) == []


def test_bench_body_lists_missing():
    rows = [Comparison("bench_slow", 1.0, 2.0), Comparison("only_target", None, 1.0)]

    [finding] = findings_for_bench(rows, "4.0.1", "4.1.1", RUN_URL)

    assert "only_target" in finding.body


def test_bench_fingerprint_rounds_ratios_and_ignores_run_url():
    [first] = findings_for_bench([Comparison("bench_slow", 1.0, 1.501)], "4.0.1", "4.1.1", RUN_URL)
    [second] = findings_for_bench([Comparison("bench_slow", 1.0, 1.499)], "4.0.1", "4.1.1", "https://x/2")
    [third] = findings_for_bench([Comparison("bench_slow", 1.0, 1.6)], "4.0.1", "4.1.1", RUN_URL)

    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != third.fingerprint


# --- checklist ---


def test_every_body_ends_with_docs_checklist():
    results = [
        _result(constrained_ok=False, unconstrained_ok=False, tests_passed=None, pip_error=RESOLUTION_ERROR),
        _result(python=None, constrained_ok=False, unconstrained_ok=False, tests_passed=None),
        _result(constrained_ok=False, tests_passed=False, failed_tests=("a::t",)),
        _result(spark="5.0.0", in_range=False),
    ]
    findings = [f for r in results for f in findings_for_target(r, RUN_URL)]
    findings += findings_for_bench([Comparison("bench_slow", 1.0, 2.0)], "4.0.1", "4.1.1", RUN_URL)
    findings.append(workflow_failure_finding({"compat": {"result": "failure"}}, RUN_URL))

    assert len(findings) == 6
    for finding in findings:
        assert finding.body.endswith(DOCS_CHECKLIST)
    assert DOCS_CHECKLIST == (
        "### When fixing\n"
        "- [ ] Update the `pyspark` range in `pyproject.toml` if support changes\n"
        "- [ ] Update the Compatibility table in `README.md`\n"
        "- [ ] Update the Compatibility Matrix in `docs/quickstart.rst`"
    )


# --- workflow failure ---


def test_workflow_failure_finding_lists_jobs():
    needs = {"resolve": {"result": "success", "outputs": {}}, "compat": {"result": "failure", "outputs": {}}}

    finding = workflow_failure_finding(needs, RUN_URL)
    again = workflow_failure_finding(needs, "https://x/2")

    assert (finding.title, finding.labels) == ("Spark canary: workflow failing", ("spark-canary", "ci"))
    assert "| compat | failure |" in finding.body
    assert "| resolve | success |" in finding.body
    assert RUN_URL in finding.body
    assert finding.fingerprint == again.fingerprint


# --- JUnit and JSON ---


def test_failed_tests_parses_junit():
    assert failed_tests(FIXTURES / "junit_fail.xml") == (
        "tests.test_core.TestFit::test_fit_broken[norm]",
        "tests.test_spark_backend::test_session",
    )
    assert failed_tests(FIXTURES / "junit_pass.xml") == ()


def test_target_result_json_round_trip():
    result = _result(python=None, failed_tests=("a::t",), pip_error="boom")

    assert TargetResult.from_json(json.loads(json.dumps(result.to_json()))) == result


# --- record CLI ---


def _record(tmp_path: Path, **overrides: str) -> tuple[int, Path]:
    out = tmp_path / "result.json"
    options = {
        "spark": "4.1.1",
        "python": "3.13",
        "in-range": "true",
        "constrained": "success",
        "unconstrained": "skipped",
        "tests": "success",
        "junit": str(tmp_path / "junit.xml"),
        "freeze": str(tmp_path / "freeze.txt"),
        "pip-error": str(tmp_path / "pip-error.txt"),
        "out": str(out),
    }
    options.update(overrides)
    argv = ["record"] + [arg for key, value in options.items() for arg in (f"--{key}", value)]
    return cli.main(argv), out


def test_record_writes_result(tmp_path):
    freeze = tmp_path / "freeze.txt"
    freeze.write_text(
        "numpy==2.1.3\nPandas==2.2.3\npyarrow==18.1.0\npyspark==4.1.1\nscipy==1.14.1\n"
        "-e git+https://github.com/o/r@abc#egg=spark_bestfit\n"
    )

    code, out = _record(tmp_path, tests="failure", junit=str(FIXTURES / "junit_fail.xml"))

    assert code == 0
    assert TargetResult.from_json(json.loads(out.read_text())) == _result(
        tests_passed=False,
        failed_tests=("tests.test_core.TestFit::test_fit_broken[norm]", "tests.test_spark_backend::test_session"),
        failed_total=2,
    )


def test_record_keeps_only_the_tail_of_huge_files(tmp_path):
    (tmp_path / "pip-error.txt").write_text("h" * 3_000_000 + "\nERROR: ResolutionImpossible\n")

    code, out = _record(tmp_path, constrained="failure", unconstrained="failure", tests="skipped")

    assert code == 0
    pip_error = TargetResult.from_json(json.loads(out.read_text())).pip_error
    assert len(pip_error.encode()) <= 2_000_000
    assert pip_error.endswith("ERROR: ResolutionImpossible\n")


def test_record_caps_failed_tests_but_keeps_total(tmp_path):
    cases = "".join(f'<testcase classname="t" name="c{i}"><failure>x</failure></testcase>' for i in range(1500))
    junit = tmp_path / "junit.xml"
    junit.write_text(f'<testsuites><testsuite name="pytest">{cases}</testsuite></testsuites>')

    code, out = _record(tmp_path, tests="failure", junit=str(junit))

    assert code == 0
    result = TargetResult.from_json(json.loads(out.read_text()))
    assert len(result.failed_tests) == MAX_JUNIT_FAILURES
    assert result.failed_tests[0] == "t::c0"
    assert result.failed_total == 1500
    [finding] = findings_for_target(result, RUN_URL)
    assert "and 1450 more" in finding.body
    assert "### Failed tests (1500)" in finding.body


def test_record_python_empty_means_none_and_is_not_a_malfunction(tmp_path):
    code, out = _record(tmp_path, python="", constrained="skipped", unconstrained="skipped", tests="skipped")

    assert code == 0
    result = TargetResult.from_json(json.loads(out.read_text()))
    assert result.python is None
    assert (result.constrained_ok, result.unconstrained_ok, result.tests_passed) == (False, False, None)
    assert result.packages == {}


def test_record_resolution_failure_is_recorded(tmp_path):
    (tmp_path / "pip-error.txt").write_text(RESOLUTION_ERROR)

    code, out = _record(tmp_path, constrained="failure", unconstrained="failure", tests="skipped")

    assert code == 0
    assert TargetResult.from_json(json.loads(out.read_text())).pip_error == RESOLUTION_ERROR


def test_record_transient_failure_with_no_install_exits_nonzero(tmp_path):
    (tmp_path / "pip-error.txt").write_text("ReadTimeoutError: HTTPSConnectionPool(host='pypi.org')")

    code, out = _record(tmp_path, constrained="failure", unconstrained="failure", tests="skipped")

    assert code != 0
    assert not out.exists()


def test_record_transient_constrained_failure_in_range_exits_nonzero(tmp_path):
    (tmp_path / "pip-error.txt").write_text("ReadTimeoutError: HTTPSConnectionPool(host='pypi.org')")

    code, out = _record(tmp_path, constrained="failure", unconstrained="success")

    assert code != 0
    assert not out.exists()
