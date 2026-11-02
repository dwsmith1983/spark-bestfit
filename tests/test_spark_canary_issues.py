"""Tests for filing Spark canary findings as GitHub issues (``gh`` is always faked)."""

import json
from pathlib import Path

import pytest

from spark_canary import __main__ as cli
from spark_canary.findings import DOCS_CHECKLIST, Finding, TargetResult, fingerprint_marker
from spark_canary.issues import upsert

TITLE = "Spark canary: tests fail on pyspark 4.1.1"
OUTSIDE = "Spark canary: pyspark 5.0.0 passes, outside the declared range"
NEW_FP = "a" * 64
OLD_FP = "b" * 64
RUN_URL = "https://github.com/o/r/actions/runs/1"
BOT = {"login": "app/github-actions", "is_bot": True}
HUMAN = {"login": "someone", "is_bot": False}


def _bot_comment(body: str) -> dict:
    return {"author": {"login": "github-actions"}, "viewerDidAuthor": True, "body": body}


def _human_comment(body: str) -> dict:
    return {"author": {"login": "someone"}, "viewerDidAuthor": False, "body": body}


def _finding(title: str = TITLE, labels: tuple[str, ...] = ("spark-canary", "tests")) -> Finding:
    return Finding(title=title, labels=labels, body=f"{fingerprint_marker(NEW_FP)}\nnew details", fingerprint=NEW_FP)


class FakeGh:
    """Records ``gh`` argument lists; answers ``issue list`` and ``issue view`` from canned data."""

    def __init__(self, listed: list[dict] | None = None, view: dict | None = None) -> None:
        self.listed = listed or []
        self.view = view or {"body": "", "comments": []}
        self.calls: list[list[str]] = []
        self.bodies: list[str] = []

    def __call__(self, args: list[str]) -> str:
        self.calls.append(list(args))
        if "--body-file" in args:
            self.bodies.append(Path(args[args.index("--body-file") + 1]).read_text())
        if args[:2] == ["issue", "list"]:
            return json.dumps(self.listed)
        if args[:2] == ["issue", "view"]:
            return json.dumps(self.view)
        return ""

    def verbs(self) -> list[tuple[str, str]]:
        return [(call[0], call[1]) for call in self.calls]


def test_creates_when_none_open():
    gh = FakeGh()

    assert upsert(_finding(), gh) == "created"

    assert gh.calls[0] == ["label", "create", "spark-canary", "--color", "FBCA04", "--force"]
    assert gh.calls[1] == [
        "issue",
        "list",
        "--state",
        "open",
        "--label",
        "spark-canary",
        "--search",
        f'"{TITLE}" in:title',
        "--json",
        "number,title,state",
        "--limit",
        "50",
    ]
    create = gh.calls[2]
    assert create[:2] == ["issue", "create"]
    assert create[create.index("--title") + 1] == TITLE
    assert create[create.index("--label") + 1] == "spark-canary,tests"
    assert gh.bodies == [_finding().body]
    assert len(gh.calls) == 3


def test_comments_on_exact_match():
    view = {"body": fingerprint_marker(OLD_FP), "author": BOT, "comments": []}
    gh = FakeGh(listed=[{"number": 7, "title": TITLE, "state": "OPEN"}], view=view)

    assert upsert(_finding(), gh) == "commented"

    assert ["issue", "view", "7", "--json", "body,author,comments"] in gh.calls
    comment = gh.calls[-1]
    assert comment[:3] == ["issue", "comment", "7"]
    assert gh.bodies == [_finding().body]
    assert ("issue", "create") not in gh.verbs()


def test_unchanged_fingerprint_does_nothing():
    view = {
        "body": fingerprint_marker(OLD_FP),
        "author": BOT,
        "comments": [_human_comment("a human note"), _bot_comment(f"{fingerprint_marker(NEW_FP)}\nlast week")],
    }
    gh = FakeGh(listed=[{"number": 7, "title": TITLE, "state": "OPEN"}], view=view)

    assert upsert(_finding(), gh) == "unchanged"

    assert ("issue", "comment") not in gh.verbs()
    assert ("issue", "create") not in gh.verbs()


def test_newest_fingerprint_wins_over_older_match():
    view = {"body": fingerprint_marker(NEW_FP), "author": BOT, "comments": [_bot_comment(fingerprint_marker(OLD_FP))]}
    gh = FakeGh(listed=[{"number": 7, "title": TITLE, "state": "OPEN"}], view=view)

    assert upsert(_finding(), gh) == "commented"


def test_bot_body_with_matching_fingerprint_is_unchanged():
    view = {"body": f"{fingerprint_marker(NEW_FP)}\ndetails", "author": BOT, "comments": []}
    gh = FakeGh(listed=[{"number": 7, "title": TITLE, "state": "OPEN"}], view=view)

    assert upsert(_finding(), gh) == "unchanged"


def test_human_comment_with_fingerprint_is_ignored():
    view = {
        "body": fingerprint_marker(OLD_FP),
        "author": BOT,
        "comments": [_human_comment(f"{fingerprint_marker(NEW_FP)}\nshh")],
    }
    gh = FakeGh(listed=[{"number": 7, "title": TITLE, "state": "OPEN"}], view=view)

    assert upsert(_finding(), gh) == "commented"


def test_human_issue_body_with_fingerprint_is_ignored():
    view = {"body": fingerprint_marker(NEW_FP), "author": HUMAN, "comments": []}
    gh = FakeGh(listed=[{"number": 7, "title": TITLE, "state": "OPEN"}], view=view)

    assert upsert(_finding(), gh) == "commented"


def test_bot_comment_marker_not_on_first_line_is_ignored():
    view = {
        "body": fingerprint_marker(OLD_FP),
        "author": BOT,
        "comments": [_bot_comment(f"quoted:\n{fingerprint_marker(NEW_FP)}")],
    }
    gh = FakeGh(listed=[{"number": 7, "title": TITLE, "state": "OPEN"}], view=view)

    assert upsert(_finding(), gh) == "commented"


def test_passes_outside_range_searches_all_states_and_never_reopens():
    gh = FakeGh(listed=[{"number": 3, "title": OUTSIDE, "state": "CLOSED"}])

    assert upsert(_finding(OUTSIDE, ("spark-canary", "dependencies")), gh) == "unchanged"

    listing = gh.calls[1]
    assert listing[listing.index("--state") + 1] == "all"
    assert gh.verbs() == [("label", "create"), ("issue", "list")]


def test_passes_outside_range_comments_when_open():
    gh = FakeGh(listed=[{"number": 3, "title": OUTSIDE, "state": "OPEN"}], view={"body": "", "comments": []})

    assert upsert(_finding(OUTSIDE, ("spark-canary", "dependencies")), gh) == "commented"


def test_near_match_title_is_not_existing():
    near = "Spark canary: tests fail on pyspark 4.1.10"
    gh = FakeGh(listed=[{"number": 9, "title": near, "state": "OPEN"}])

    assert upsert(_finding(), gh) == "created"

    assert ("issue", "view") not in gh.verbs()


def test_dry_run_calls_nothing(capsys):
    def explode(args: list[str]) -> str:
        raise AssertionError(f"gh called: {args}")

    assert upsert(_finding(), explode, dry_run=True) == "dry-run"

    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert TITLE in out


# --- report and report-failure CLIs ---


def _target(spark: str, in_range: bool = True) -> dict:
    return {"spark": spark, "python": "3.13", "in_range": in_range, "setuptools": False}


def _write_result(results: Path, spark: str, **overrides) -> None:
    fields = {
        "spark": spark,
        "python": "3.13",
        "in_range": True,
        "constrained_ok": True,
        "unconstrained_ok": True,
        "tests_passed": True,
        "failed_tests": (),
        "packages": {"pyspark": spark},
        "pip_error": "",
    }
    fields.update(overrides)
    results.mkdir(exist_ok=True)
    (results / f"result-{spark}.json").write_text(json.dumps(TargetResult(**fields).to_json()))


def _write_comparison(path: Path, skipped: str | None = None) -> Path:
    path.write_text(json.dumps({"baseline": "4.0.1", "target": "4.1.1", "skipped": skipped, "rows": []}))
    return path


def _report(tmp_path: Path, targets: list[dict], comparison: Path, *extra: str) -> int:
    argv = ["report", "--targets", json.dumps(targets), "--results-dir", str(tmp_path / "results")]
    return cli.main(argv + ["--comparison", str(comparison), "--run-url", RUN_URL, *extra])


class FakeUpsert:
    def __init__(self, fail_first: bool = False) -> None:
        self.fail_first = fail_first
        self.titles: list[str] = []
        self.dry_runs: list[bool] = []

    def __call__(self, finding: Finding, run, dry_run: bool = False) -> str:
        self.titles.append(finding.title)
        self.dry_runs.append(dry_run)
        if self.fail_first and len(self.titles) == 1:
            raise RuntimeError("gh: HTTP 502")
        return "created"


def test_report_continues_after_one_upsert_fails_then_exits_nonzero(tmp_path, monkeypatch, capsys):
    fake = FakeUpsert(fail_first=True)
    monkeypatch.setattr(cli, "upsert", fake)
    results = tmp_path / "results"
    _write_result(results, "4.0.1", tests_passed=False, failed_tests=("a::t",))
    _write_result(results, "4.1.1", tests_passed=False, failed_tests=("b::t",))
    comparison = _write_comparison(tmp_path / "comparison.json")

    code = _report(tmp_path, [_target("4.0.1"), _target("4.1.1")], comparison)

    assert code != 0
    assert fake.titles == ["Spark canary: tests fail on pyspark 4.0.1", "Spark canary: tests fail on pyspark 4.1.1"]
    assert "HTTP 502" in capsys.readouterr().err


def test_report_missing_target_result_is_failure(tmp_path, monkeypatch):
    fake = FakeUpsert()
    monkeypatch.setattr(cli, "upsert", fake)
    _write_result(tmp_path / "results", "4.1.1", tests_passed=False, failed_tests=("b::t",))
    comparison = _write_comparison(tmp_path / "comparison.json")

    code = _report(tmp_path, [_target("4.0.1"), _target("4.1.1")], comparison)

    assert code != 0
    assert fake.titles == ["Spark canary: tests fail on pyspark 4.1.1"]


def test_report_missing_comparison_is_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "upsert", FakeUpsert())
    _write_result(tmp_path / "results", "4.1.1")

    assert _report(tmp_path, [_target("4.1.1")], tmp_path / "never_written.json") != 0


def test_report_skipped_comparison_is_not_failure(tmp_path, monkeypatch, capsys):
    fake = FakeUpsert()
    monkeypatch.setattr(cli, "upsert", fake)
    _write_result(tmp_path / "results", "4.1.1")
    comparison = _write_comparison(tmp_path / "comparison.json", skipped="no common Python")

    code = _report(tmp_path, [_target("4.1.1")], comparison)

    assert code == 0
    assert fake.titles == []
    assert "no common Python" in capsys.readouterr().out


def test_report_files_bench_regression_and_passes_dry_run(tmp_path, monkeypatch):
    fake = FakeUpsert()
    monkeypatch.setattr(cli, "upsert", fake)
    _write_result(tmp_path / "results", "4.1.1")
    comparison = tmp_path / "comparison.json"
    rows = [{"name": "bench_slow", "baseline_median": 1.0, "target_median": 2.0}]
    comparison.write_text(json.dumps({"baseline": "4.0.1", "target": "4.1.1", "skipped": None, "rows": rows}))

    code = _report(tmp_path, [_target("4.1.1")], comparison, "--dry-run")

    assert code == 0
    assert fake.titles == ["Spark canary: performance regression on pyspark 4.1.1 vs 4.0.1"]
    assert fake.dry_runs == [True]


def test_report_logs_out_of_range_uninstallable_to_step_summary(tmp_path, monkeypatch, capsys):
    fake = FakeUpsert()
    monkeypatch.setattr(cli, "upsert", fake)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    _write_result(
        tmp_path / "results", "5.0.0", in_range=False, constrained_ok=False, unconstrained_ok=False, tests_passed=None
    )
    comparison = _write_comparison(tmp_path / "comparison.json")

    code = _report(tmp_path, [_target("5.0.0", in_range=False)], comparison)

    assert code == 0
    assert fake.titles == []
    assert "pyspark 5.0.0" in capsys.readouterr().out
    assert "pyspark 5.0.0" in summary.read_text()


def test_report_failure_files_workflow_failing(monkeypatch):
    fake = FakeUpsert()
    captured: list[Finding] = []
    monkeypatch.setattr(cli, "upsert", lambda f, run, dry_run=False: captured.append(f) or fake(f, run, dry_run))
    needs = {"resolve": {"result": "success", "outputs": {}}, "compat": {"result": "failure", "outputs": {}}}

    code = cli.main(["report-failure", "--needs", json.dumps(needs), "--run-url", RUN_URL, "--dry-run"])

    assert code == 0
    assert fake.titles == ["Spark canary: workflow failing"]
    assert fake.dry_runs == [True]
    assert "| compat | failure |" in captured[0].body
    assert captured[0].body.endswith(DOCS_CHECKLIST)


def test_report_rejects_targets_that_are_not_a_list(tmp_path):
    with pytest.raises(SystemExit):
        _report(tmp_path, {"targets": []}, tmp_path / "c.json")  # type: ignore[arg-type]
