"""Tests for the CSV aggregators (scripts/build_scaling_csv.py,
scripts/build_execbench_agreement_csv.py).

Mirrors tests/test_run_scaling.py / tests/test_figures.py: the scripts/ and
scripts/figures/ dirs are added to sys.path and the modules imported directly.
No GPU / network -- the aggregators are pure JSON->CSV, and the figure loaders
are reused only to prove the emitted CSV matches the schema the figures read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "figures"))

import build_execbench_agreement_csv as agg_exec  # noqa: E402
import build_scaling_csv as agg_scale  # noqa: E402
import execution_vs_codebleu  # noqa: E402
import scaling_curves  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Values from the committed cycle-2 result JSONs (results/finetune_*_test.json).
A_EP10_CODEBLEU = 0.4769922510731756
A_EP10_SYNTAX = 0.9168831168831169


def _write_result(path: Path, *, codebleu=0.4, syntax=0.8, n=6545, pass_rate=None) -> None:
    """Write a minimal results/<name>.json with the fields the aggregators read."""
    metrics: dict = {"codebleu": codebleu, "syntax_valid_rate": syntax, "n": n}
    if pass_rate is not None:
        metrics = {"n": n, "compile_rate": 1.0, "pass_rate": pass_rate}
    path.write_text(json.dumps({"config": {}, "metrics": metrics, "n": n}), encoding="utf-8")


# --------------------------------------------------------------------------- #
# build_scaling_csv                                                           #
# --------------------------------------------------------------------------- #


def test_scaling_produces_reference_rows_from_committed_results():
    """Deliverable #1 (post cycle-4 GPU batch): against the full committed results
    (52K references + the real data/ptcompute sweep JSONs), the builder emits the
    reference rows plus every sweep point, each with the correct seed."""
    rows = agg_scale.build_rows(RESULTS_DIR)
    keyed = {(r["arm"], r["axis"], r["x"]): r for r in rows}

    # 52K data reference: arm A (finetune_A_ep10) + arm B seeds 0/1.
    a52 = keyed[("A", "data", agg_scale.FULL_TRAIN_N)]
    assert abs(a52["codebleu"] - A_EP10_CODEBLEU) < 1e-12
    assert abs(a52["syntax"] - A_EP10_SYNTAX) < 1e-12
    assert a52["seed"] == 42
    full_n = agg_scale.FULL_TRAIN_N
    b52_seeds = sorted(
        r["seed"] for r in rows if (r["arm"], r["axis"], r["x"]) == ("B", "data", full_n)
    )
    assert b52_seeds == [0, 1]  # seed2 intentionally excluded (matches the data-curve seed set)

    # ep10 ptcompute reference reuses finetune_A_ep10.
    ep10 = keyed[("A", "ptcompute", agg_scale.PTCOMPUTE_FINAL_EPOCHS)]
    assert abs(ep10["codebleu"] - A_EP10_CODEBLEU) < 1e-12

    # Data-scaling sweep points are now committed (cycle-4 GPU batch): both arms,
    # both seeds, at each of the three sub-52K sizes.
    for arm in ("A", "B"):
        for x in (1000, 5000, 15000):
            seeds = sorted(
                r["seed"] for r in rows if (r["arm"], r["axis"], r["x"]) == (arm, "data", x)
            )
            assert seeds == [0, 1], f"arm {arm} x={x} seeds={seeds}"

    # Ptcompute sweep points (arm A only, seed 42) are now committed too. CodeBLEU is
    # roughly flat past epoch 1 (pretrain compute buys ~nothing beyond that), well
    # below the ep10 finetune-converged point.
    pt1 = keyed[("A", "ptcompute", 1)]
    pt3 = keyed[("A", "ptcompute", 3)]
    assert pt1["seed"] == pt3["seed"] == 42
    assert pt1["codebleu"] < ep10["codebleu"]
    assert pt3["codebleu"] < ep10["codebleu"]
    assert abs(pt1["codebleu"] - pt3["codebleu"]) < 0.01

    # 15 data-axis rows (3 sizes x 2 arms x 2 seeds + 2x 52K refs) + 3 ptcompute-axis rows.
    assert sum(r["axis"] == "data" for r in rows) == 15
    assert sum(r["axis"] == "ptcompute" for r in rows) == 3


def test_scaling_picks_up_a_sweep_point(tmp_path):
    """A present sweep result JSON becomes a data-axis row alongside the references."""
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results / "finetune_A_ep10_test.json", codebleu=0.48, syntax=0.9)
    _write_result(results / "finetune_scale_A_n1k_seed0_test.json", codebleu=0.30, syntax=0.55)

    rows = agg_scale.build_rows(results)
    sweep = [r for r in rows if r["axis"] == "data" and r["x"] == 1000]
    assert len(sweep) == 1
    assert sweep[0] == {
        "arm": "A",
        "axis": "data",
        "x": 1000,
        "codebleu": 0.30,
        "syntax": 0.55,
        "seed": 0,
    }


def test_scaling_empty_results_dir_is_empty(tmp_path):
    assert agg_scale.build_rows(tmp_path) == []


def test_scaling_csv_matches_figure_schema(tmp_path):
    """The emitted CSV round-trips through the figure's own loader (schema lock)."""
    out = tmp_path / "scaling_data.csv"
    agg_scale.write_csv(agg_scale.build_rows(RESULTS_DIR), out)
    rows = scaling_curves.load_scaling_rows(out)  # raises on any missing/renamed column
    assert rows
    assert {r["axis"] for r in rows} <= {"data", "ptcompute"}
    assert {r["arm"] for r in rows} <= {"A", "B"}


def test_committed_scaling_csv_in_sync_with_builder(tmp_path):
    """Drift guard: the committed results/scaling_data.csv must equal what the builder emits
    from the committed result JSONs. If this fails, a result JSON changed without rerunning
    `python scripts/figures/make_all.py` (which rebuilds the CSV) -- the figure would render
    stale numbers. Compared via the figure's loader so line-ending differences don't matter."""
    committed = RESULTS_DIR / "scaling_data.csv"
    assert committed.is_file(), "results/scaling_data.csv must be committed (not gitignored)"
    fresh = tmp_path / "scaling_data.csv"
    agg_scale.write_csv(agg_scale.build_rows(RESULTS_DIR), fresh)
    assert scaling_curves.load_scaling_rows(committed) == scaling_curves.load_scaling_rows(fresh)


# --------------------------------------------------------------------------- #
# build_execbench_agreement_csv                                               #
# --------------------------------------------------------------------------- #


def test_execbench_agreement_joins_codebleu_and_pass_at_1(tmp_path):
    """Deliverable #2: a small fixture -- arm A CodeBLEU + arm A execution -> one row."""
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results / "finetune_A_ep10_test.json", codebleu=0.477)
    _write_result(results / "execbench_A.json", pass_rate=0.124, n=201)

    rows = agg_exec.build_rows(results)
    assert len(rows) == 1
    assert rows[0] == {"arm": "A", "codebleu": 0.477, "pass_at_1": 0.124, "n_bugs": 201}


def test_execbench_agreement_best_rag_and_lora(tmp_path):
    """Arm C = best (max CodeBLEU) rag_*; arm D = lora_qwen; each needs its execbench result."""
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results / "rag_bm25_k3_test.json", codebleu=0.42)
    _write_result(results / "rag_codebert_k5_test.json", codebleu=0.40)
    _write_result(results / "execbench_C.json", pass_rate=0.284, n=201)
    _write_result(results / "lora_qwen_test.json", codebleu=0.462)
    _write_result(results / "execbench_D.json", pass_rate=0.331, n=201)

    rows = {r["arm"]: r for r in agg_exec.build_rows(results)}
    assert rows["C"]["codebleu"] == 0.42  # the max, not 0.40
    assert rows["D"]["pass_at_1"] == 0.331


def test_execbench_agreement_needs_both_axes(tmp_path):
    """CodeBLEU without an execution result (or vice versa) yields no row."""
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results / "finetune_A_ep10_test.json", codebleu=0.477)  # no execbench_A.json
    assert agg_exec.build_rows(results) == []
    # And an empty dir is fine too (figure falls back to its placeholder).
    assert agg_exec.build_rows(tmp_path / "nope") == []


def test_execbench_agreement_csv_matches_figure_schema(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results / "finetune_A_ep10_test.json", codebleu=0.477)
    _write_result(results / "execbench_A.json", pass_rate=0.124, n=201)

    out = tmp_path / "execbench_agreement.csv"
    agg_exec.write_csv(agg_exec.build_rows(results), out)
    loaded = execution_vs_codebleu.load_agreement_rows(out)  # raises on schema drift
    assert loaded == [{"arm": "A", "codebleu": 0.477, "pass_at_1": 0.124, "n_bugs": 201}]


def test_committed_execbench_agreement_csv_in_sync_with_builder(tmp_path):
    """Drift guard: the committed results/execbench_agreement.csv must equal the builder's
    output from the committed result JSONs (rebuilt by make_all.py). Guards against the
    figure silently rendering a stale scatter after a result JSON changes. All four arms
    (A/B/C/D) now have committed execbench results, so the builder emits four rows and the
    committed CSV must match. Compared via the figure's loader (line-ending agnostic)."""
    committed = RESULTS_DIR / "execbench_agreement.csv"
    assert committed.is_file(), "results/execbench_agreement.csv must be committed (not gitignored)"
    fresh = tmp_path / "execbench_agreement.csv"
    agg_exec.write_csv(agg_exec.build_rows(RESULTS_DIR), fresh)
    assert execution_vs_codebleu.load_agreement_rows(
        committed
    ) == execution_vs_codebleu.load_agreement_rows(fresh)


# --------------------------------------------------------------------------- #
# scratch runs must never reach a committed CSV                               #
# --------------------------------------------------------------------------- #


def test_scratch_rag_run_is_excluded_from_the_agreement_csv(tmp_path):
    """A gitignored `*_local*` experiment must not change a committed derived CSV.

    `make_all.py` rebuilds results/execbench_agreement.csv from a glob on every documented
    reproduce run, so an unfiltered `rag_*_test.json` glob let a local scratch file silently
    move the arm-C point -- and the contributor just sees an inexplicably dirty tree.
    """
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results / "rag_bm25_k3_test.json", codebleu=0.42)
    _write_result(results / "execbench_C.json", pass_rate=0.284, n=201)
    # A local experiment that scores higher than every published RAG config.
    _write_result(results / "rag_bm25_k3_local_test.json", codebleu=0.99)

    rows = {r["arm"]: r for r in agg_exec.build_rows(results)}
    assert rows["C"]["codebleu"] == 0.42, "a *_local* scratch run leaked into the committed CSV"


def test_scratch_finetune_run_is_excluded_from_the_figure_loader(tmp_path):
    """Same glob hazard in scripts/figures/_common.load_finetune_results."""
    import _common

    results = tmp_path / "results"
    results.mkdir()
    _write_result(results / "finetune_A_ep10_test.json", codebleu=0.48)
    _write_result(results / "finetune_A_ep10_local_test.json", codebleu=0.99)

    loaded = _common.load_finetune_results(results)
    assert set(loaded) == {"finetune_A_ep10"}


def test_scaling_builder_ignores_scratch_sweep_points(tmp_path):
    """build_scaling_csv reads explicit names, not a glob -- pin that it stays that way."""
    results = tmp_path / "results"
    results.mkdir()
    _write_result(results / "finetune_scale_A_n1k_seed0_test.json", codebleu=0.30, syntax=0.55)
    _write_result(results / "finetune_scale_A_n1k_seed0_local_test.json", codebleu=0.99, syntax=0.1)

    rows = [r for r in agg_scale.build_rows(results) if r["axis"] == "data" and r["x"] == 1000]
    assert len(rows) == 1
    assert rows[0]["codebleu"] == 0.30


# --------------------------------------------------------------------------- #
# --results-dir must not write back into the committed results/               #
# --------------------------------------------------------------------------- #


def test_scaling_main_with_results_dir_writes_beside_it_not_into_the_repo(tmp_path, capsys):
    """`--results-dir X` used to read X but still default `--out` to results/scaling_data.csv.

    `build_rows` degrades gracefully to fewer rows, so pointing at a partial directory
    silently truncated the committed CSV. CI's clean-tree check only covers the make_all.py
    path, which always reads the real RESULTS_DIR -- this one slipped past it.
    """
    results = tmp_path / "partial"
    results.mkdir()
    _write_result(results / "finetune_A_ep10_test.json", codebleu=0.48, syntax=0.9)

    rc = agg_scale.main(["--results-dir", str(results)])
    capsys.readouterr()

    assert rc == 0
    # Written next to --results-dir, not into the repo's committed results/.
    assert (results / "scaling_data.csv").is_file()
    assert agg_scale.resolve_out(
        agg_scale.build_parser().parse_args(["--results-dir", str(results)])
    ) == (results / "scaling_data.csv")


def test_execbench_agreement_main_with_results_dir_writes_beside_it(tmp_path, capsys):
    results = tmp_path / "partial"
    results.mkdir()
    _write_result(results / "finetune_A_ep10_test.json", codebleu=0.477)
    _write_result(results / "execbench_A.json", pass_rate=0.124, n=201)

    rc = agg_exec.main(["--results-dir", str(results)])
    capsys.readouterr()

    assert rc == 0
    assert (results / "execbench_agreement.csv").is_file()


def test_explicit_out_still_wins_over_the_derived_default(tmp_path, capsys):
    """`--out` is still honoured verbatim when the caller passes it."""
    results = tmp_path / "partial"
    results.mkdir()
    _write_result(results / "finetune_A_ep10_test.json", codebleu=0.48, syntax=0.9)
    out = tmp_path / "elsewhere" / "mine.csv"

    assert agg_scale.main(["--results-dir", str(results), "--out", str(out)]) == 0
    capsys.readouterr()
    assert out.is_file()
    assert not (results / "scaling_data.csv").exists()


def test_default_invocation_still_targets_the_committed_csv_paths():
    """Without --results-dir, both builders keep writing the committed derived CSVs."""
    scale_args = agg_scale.build_parser().parse_args([])
    exec_args = agg_exec.build_parser().parse_args([])
    assert Path(agg_scale.resolve_out(scale_args)) == agg_scale.OUT_PATH
    assert Path(agg_exec.resolve_out(exec_args)) == agg_exec.OUT_PATH
