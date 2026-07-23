"""Unit tests for scripts/run_training.py (the Phase-2 orchestrator).

Uses the scripts-import convention: the scripts/ directory is added to
sys.path and the module imported directly.
No GPU / network / training subprocesses here -- only the planning, logging,
and gating pieces, plus one trivial echo subprocess.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_training  # noqa: E402


def test_plan_puts_headline_systems_first():
    names = [s.name for s in run_training.plan_steps()]
    assert names[:5] == [
        "tokenizer",
        "pretrain",
        "finetune_A_ep10",
        "generate_A_ep10",
        "eval_A_ep10",
    ]
    assert names[5] == "finetune_B_seed0"
    assert len(names) == 2 + 3 * 6


def test_plan_markers_and_prune_dirs():
    steps = {s.name: s for s in run_training.plan_steps()}
    assert steps["tokenizer"].done_marker == run_training.TOKENIZER_MODEL
    assert steps["pretrain"].done_marker.name == "model.safetensors"
    assert (
        steps["eval_A_ep10"].done_marker == run_training.RESULTS_DIR / "finetune_A_ep10_test.json"
    )
    assert (
        steps["eval_A_ep10"].prune_checkpoints_in
        == run_training.REPO_ROOT / "outputs" / "finetune_A_ep10"
    )
    assert steps["finetune_B_seed2"].cmd[-1] == "configs/finetune_B_seed2.yaml"


def test_plan_limit_eval_caps_generate_only():
    steps = {s.name: s for s in run_training.plan_steps(limit_eval=50)}
    assert steps["generate_A_ep10"].cmd[-2:] == ["--limit", "50"]
    assert "--limit" not in steps["finetune_A_ep10"].cmd


def test_write_status_renders_table(tmp_path):
    rows = [
        {"name": "pretrain", "status": "running", "started": "t", "duration": "-", "result": "-"}
    ]
    run_training.write_status(tmp_path, rows)
    text = (tmp_path / "STATUS.md").read_text(encoding="utf-8")
    assert "| pretrain | running |" in text
    assert not (tmp_path / "STATUS.md.tmp").exists()


def test_read_em(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"metrics": {"em": 0.0125}}), encoding="utf-8")
    assert run_training.read_em(path) == 0.0125


def test_write_summary_lists_only_available_systems(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    metrics = {"em": 0.05, "em_raw": 0.01, "codebleu": 0.7, "syntax_valid_rate": 0.9, "n": 10}
    (results / "finetune_A_ep10_test.json").write_text(
        json.dumps({"metrics": metrics}), encoding="utf-8"
    )
    run_training.write_summary(tmp_path, results_dir=results)
    text = (tmp_path / "SUMMARY.md").read_text(encoding="utf-8")
    assert "| A_ep10 | 0.0500 |" in text
    assert "B_seed0" not in text


def test_write_summary_writes_nothing_without_results(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    run_training.write_summary(tmp_path, results_dir=results)
    assert not (tmp_path / "SUMMARY.md").exists()


def test_run_step_streams_to_log(tmp_path):
    step = run_training.Step(
        "hello", [sys.executable, "-c", "print('hi from step')"], tmp_path / "unused-marker"
    )
    rc = run_training.run_step(step, tmp_path, {**os.environ})
    assert rc == 0
    assert "hi from step" in (tmp_path / "hello.log").read_text(encoding="utf-8")


def test_preflight_accepts_repo_configs():
    run_training.preflight()  # must not raise against the committed configs


def test_fmt_duration():
    assert run_training._fmt_duration(59) == "59s"
    assert run_training._fmt_duration(61) == "1m 01s"
    assert run_training._fmt_duration(3700) == "1h 01m"
