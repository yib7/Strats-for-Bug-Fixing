"""Smoke tests for scripts/figures/ (the analysis figure scripts).

Mirrors tests/test_run_scaling.py: the scripts/figures/ dir is added to sys.path
and the modules imported directly. No network, no GPU -- rendering is forced to
the headless Agg backend by importing `_common` (which every figure script also
imports). Assertions are structural (a non-empty PNG is produced, the right arms
are pending) -- never exact pixel content.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "figures"))

import execution_vs_codebleu  # noqa: E402
import four_arm_comparison  # noqa: E402
import make_all  # noqa: E402
import scaling_curves  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"


def _assert_png(path: Path) -> None:
    assert path.exists(), f"expected figure not written: {path}"
    assert path.suffix == ".png"
    data = path.read_bytes()
    assert len(data) > 1000, f"figure suspiciously small ({len(data)} bytes): {path}"
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}"


def test_make_all_writes_three_pngs(tmp_path):
    """The driver renders all three figures against committed results + fixtures.

    `csv_out_dir` is redirected: `rebuild_derived_csvs` used to always write
    `results/*.csv` at the repo root, so running the test suite mutated tracked files.
    """
    csv_dir = tmp_path / "csv"
    paths = make_all.make_all(out_dir=tmp_path, csv_out_dir=csv_dir)
    assert [p.name for p in paths] == [
        "four_arm_comparison.png",
        "scaling_curves.png",
        "execution_vs_codebleu.png",
    ]
    for p in paths:
        _assert_png(p)


def test_make_all_does_not_write_into_the_committed_results_dir(tmp_path):
    """pytest must never mutate tracked data (and must not trip CI's clean-tree guard)."""
    tracked = {p: p.read_bytes() for p in RESULTS_DIR.glob("*.csv")}
    assert tracked, "expected the committed derived CSVs to exist"

    make_all.make_all(out_dir=tmp_path, csv_out_dir=tmp_path / "csv")

    for path, before in tracked.items():
        assert path.read_bytes() == before, f"test run modified tracked file {path}"
    assert (tmp_path / "csv" / "scaling_data.csv").is_file()
    assert (tmp_path / "csv" / "execbench_agreement.csv").is_file()


def test_rebuild_derived_csvs_defaults_to_the_real_results_dir():
    """The default must stay the real path -- make_all.py is the documented reproduce step."""
    import inspect

    default = inspect.signature(make_all.rebuild_derived_csvs).parameters["csv_out_dir"].default
    assert Path(default) == RESULTS_DIR


def test_four_arm_renders_from_committed_results(tmp_path):
    out = four_arm_comparison.make(results_dir=RESULTS_DIR, out_dir=tmp_path)
    _assert_png(out)


def test_four_arm_all_real_from_committed_results():
    """All four arms come from committed results (cycle-4 GPU batch landed C/D)."""
    arms = four_arm_comparison.collect_arms(RESULTS_DIR)
    assert arms["A"]["pending"] is False
    assert arms["B"]["pending"] is False
    assert arms["C"]["pending"] is False
    assert arms["D"]["pending"] is False
    # Arm A carries the real ep10 CodeBLEU and the epoch trend.
    assert abs(arms["A"]["codebleu"]["point"] - 0.4769922510731756) < 1e-9
    assert len(arms["A"]["codebleu"]["trend"]) == 3
    # Arm B's band is the seed 0/1/2 min..max around the mean.
    b = arms["B"]["codebleu"]
    assert b["lo"] <= b["point"] <= b["hi"]
    assert b["lo"] < b["hi"]
    # Arm C is the best (max-CodeBLEU) RAG config: rag_codebert_k1.
    assert abs(arms["C"]["codebleu"]["point"] - 0.6517430213596048) < 1e-9
    # Arm D is lora_qwen_test.
    assert abs(arms["D"]["codebleu"]["point"] - 0.8543014246338675) < 1e-9


def test_four_arm_no_crash_with_no_results(tmp_path):
    """Every arm pending (empty results dir) still renders a complete-shaped figure."""
    empty = tmp_path / "empty_results"
    empty.mkdir()
    arms = four_arm_comparison.collect_arms(empty)
    assert all(arms[a]["pending"] for a in ("A", "B", "C", "D"))
    _assert_png(four_arm_comparison.make(results_dir=empty, out_dir=tmp_path))


def test_scaling_renders_from_fixture(tmp_path):
    fixture = FIXTURES_DIR / "scaling_data_example.csv"
    path, is_fixture = scaling_curves.resolve_data_path(fixture)
    assert is_fixture
    rows = scaling_curves.load_scaling_rows(fixture)
    assert {r["axis"] for r in rows} == {"data", "ptcompute"}
    assert {r["arm"] for r in rows} == {"A", "B"}
    _assert_png(scaling_curves.make(data_path=fixture, out_dir=tmp_path))


def test_scaling_handles_missing_data(tmp_path):
    """A missing CSV degrades to a labelled placeholder, not a crash."""
    _assert_png(scaling_curves.make(data_path=tmp_path / "does_not_exist.csv", out_dir=tmp_path))


def test_execution_renders_from_fixture(tmp_path):
    fixture = FIXTURES_DIR / "execbench_agreement_example.csv"
    rows = execution_vs_codebleu.load_agreement_rows(fixture)
    assert {r["arm"] for r in rows} == {"A", "B", "C", "D"}
    _assert_png(execution_vs_codebleu.make(data_path=fixture, out_dir=tmp_path))


def test_execution_handles_missing_data(tmp_path):
    _assert_png(execution_vs_codebleu.make(data_path=tmp_path / "nope.csv", out_dir=tmp_path))


def test_committed_figures_present_and_nonempty():
    """The committed docs/figures PNGs exist (guards against accidental deletion)."""
    for name in ("four_arm_comparison", "scaling_curves", "execution_vs_codebleu"):
        _assert_png(FIGURES_DIR / f"{name}.png")


# --- graceful degradation on a present-but-incomplete result -----------------------------


def _write_result(path: Path, metrics: dict) -> None:
    import json

    path.write_text(json.dumps({"config": {}, "metrics": metrics, "n": 1}) + "\n", encoding="utf-8")


def test_arm_with_a_result_file_missing_a_metric_is_pending_not_a_crash(tmp_path):
    """The module promises graceful degradation and delivered it for an *absent* file;
    a file that existed but lacked a metric crashed with TypeError / ZeroDivisionError."""
    for epoch in (1, 3, 10):
        _write_result(tmp_path / f"finetune_A_ep{epoch}_test.json", {"codebleu": 0.5})  # no syntax
    for seed in (0, 1, 2):
        _write_result(tmp_path / f"finetune_B_seed{seed}_test.json", {"em": 0.0})  # neither metric

    arms = four_arm_comparison.collect_arms(tmp_path)
    assert arms["A"]["pending"] is True
    assert arms["B"]["pending"] is True

    out = four_arm_comparison.make(results_dir=tmp_path, out_dir=tmp_path)
    _assert_png(out)


def test_arm_with_an_explicit_null_metric_is_pending(tmp_path):
    _write_result(tmp_path / "lora_qwen_test.json", {"codebleu": None, "syntax_valid_rate": 0.9})
    arms = four_arm_comparison.collect_arms(tmp_path)
    assert arms["D"]["pending"] is True


def test_empty_results_dir_renders_all_four_arms_pending(tmp_path):
    arms = four_arm_comparison.collect_arms(tmp_path)
    assert all(arms[a]["pending"] is True for a in "ABCD")
    _assert_png(four_arm_comparison.make(results_dir=tmp_path, out_dir=tmp_path))
