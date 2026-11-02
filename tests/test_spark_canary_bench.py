"""Tests for the Spark canary benchmark comparison."""

import json
from pathlib import Path

import pytest

from spark_canary import __main__ as cli
from spark_canary.bench import Comparison, compare, load_rounds, regressions

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spark_canary"
BASELINE_FILES = [FIXTURES / "bench_a1.json", FIXTURES / "bench_a2.json"]
TARGET_FILES = [FIXTURES / "bench_b1.json", FIXTURES / "bench_b2.json"]
PREFIX = "tests/benchmarks/test_benchmark_x.py::TestX::"
TOGETHER = PREFIX + "test_fit_together"
TOGETHER_100K = PREFIX + "test_fit_together_100k"


def _status(baseline: float, target: float) -> str:
    return Comparison("t", baseline, target).status


def test_load_rounds_merges_passes():
    rounds = load_rounds(BASELINE_FILES)

    assert rounds[TOGETHER] == [1.0, 1.1, 0.9]
    assert rounds[TOGETHER_100K] == [2.0, 2.1]
    assert rounds[PREFIX + "test_only_baseline"] == [0.5]


def test_load_rounds_rejects_missing_data(tmp_path):
    bad = tmp_path / "bench_no_data.json"
    bad.write_text(json.dumps({"benchmarks": [{"fullname": TOGETHER, "stats": {"mean": 1.0}}]}))

    with pytest.raises(ValueError, match="bench_no_data.json"):
        load_rounds([FIXTURES / "bench_a1.json", bad])


def test_load_rounds_rejects_empty_data(tmp_path):
    bad = tmp_path / "bench_empty.json"
    bad.write_text(json.dumps({"benchmarks": [{"fullname": TOGETHER, "stats": {"data": []}}]}))

    with pytest.raises(ValueError, match="bench_empty.json"):
        load_rounds([bad])


def test_load_rounds_rejects_unreadable_json(tmp_path):
    bad = tmp_path / "bench_truncated.json"
    bad.write_text('{"benchmarks": [')

    with pytest.raises(ValueError, match="bench_truncated.json"):
        load_rounds([bad])


def test_load_rounds_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rounds([tmp_path / "never_written.json"])


def test_ratio_boundary():
    assert _status(1.0, 1.10) == "ok"
    assert _status(1.0, 1.1001) == "regression"
    assert _status(1.0, 0.90) == "ok"
    assert _status(1.0, 0.8999) == "speedup"


def test_uses_median_not_mean():
    [row] = compare({"t": [1, 1, 1, 10]}, {"t": [1.05] * 4})

    assert row.baseline_median == 1
    assert row.status == "ok"


def test_missing_side_reported():
    rows = compare({"a": [1.0], "only_base": [1.0]}, {"a": [1.0], "only_target": [1.0]})

    by_name = {row.name: row for row in rows}
    assert [row.name for row in rows] == ["a", "only_base", "only_target"]
    assert by_name["only_base"].status == "missing"
    assert by_name["only_base"].ratio is None
    assert by_name["only_base"].target_median is None
    assert by_name["only_target"].status == "missing"
    assert by_name["only_target"].baseline_median is None


def test_regressions_lists_names():
    rows = compare({"slow": [1.0], "same": [1.0], "fast": [1.0]}, {"slow": [2.0], "same": [1.0], "fast": [0.5]})

    assert regressions(rows) == ["slow"]


def test_regressions_returns_full_node_ids():
    rows = compare(load_rounds(BASELINE_FILES), load_rounds(TARGET_FILES))

    assert regressions(rows) == [TOGETHER]


def test_comparison_json_round_trip():
    for row in (Comparison("n", 1.0, 1.5), Comparison("m", None, 2.0), Comparison("o", 3.0, None)):
        data = row.to_json()

        assert data == {"name": row.name, "baseline_median": row.baseline_median, "target_median": row.target_median}
        assert Comparison.from_json(json.loads(json.dumps(data))) == row


def test_compare_cli_writes_rows_and_prints_regressions(tmp_path, capsys):
    out = tmp_path / "comparison.json"
    args = ["compare", "--baseline", *map(str, BASELINE_FILES), "--target", *map(str, TARGET_FILES)]
    args += ["--baseline-version", "4.0.1", "--target-version", "4.1.0", "--out", str(out)]

    exit_code = cli.main(args)

    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == [TOGETHER]
    result = json.loads(out.read_text())
    assert result["baseline"] == "4.0.1"
    assert result["target"] == "4.1.0"
    assert result["skipped"] is None
    rows = [Comparison.from_json(row) for row in result["rows"]]
    assert rows == compare(load_rounds(BASELINE_FILES), load_rounds(TARGET_FILES))


def test_compare_cli_missing_side_file_raises(tmp_path):
    args = ["compare", "--baseline", str(BASELINE_FILES[0]), "--target", str(tmp_path / "crashed.json")]

    with pytest.raises(FileNotFoundError):
        cli.main(args + ["--out", str(tmp_path / "comparison.json")])


def test_compare_skip_writes_marker(tmp_path, capsys):
    out = tmp_path / "comparison.json"

    exit_code = cli.main(["compare", "--skip", "no common Python", "--target-version", "4.1.0", "--out", str(out)])

    assert exit_code == 0
    assert capsys.readouterr().out == ""
    result = json.loads(out.read_text())
    assert result == {"baseline": None, "target": "4.1.0", "skipped": "no common Python", "rows": []}


def test_compare_cli_requires_both_sides_without_skip(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["compare", "--baseline", str(BASELINE_FILES[0]), "--out", str(tmp_path / "c.json")])
