"""Unit tests for scripts/run_scaling.py (the scaling-sweep orchestrator).

Mirrors tests/test_run_rag.py: the scripts/ directory is added to sys.path and the module
imported directly. No GPU / network / training subprocesses here -- only the planning,
preflight, and summary pieces (CPU-only, config-file I/O at most).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import gen_scaling_configs  # noqa: E402
import run_scaling  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def test_plan_lists_42_steps_finetune_generate_eval():
    steps = run_scaling.plan_steps()
    assert len(steps) == 42  # 14 configs x (finetune + generate + eval)
    names = [s.name for s in steps]
    assert names[:3] == [
        "finetune_scale_A_n1k_seed0",
        "generate_scale_A_n1k_seed0",
        "eval_scale_A_n1k_seed0",
    ]
    # Every config contributes finetune -> generate -> eval, in that order.
    for i in range(0, 42, 3):
        curve = names[i].removeprefix("finetune_")
        assert names[i] == f"finetune_{curve}"
        assert names[i + 1] == f"generate_{curve}"
        assert names[i + 2] == f"eval_{curve}"


def test_plan_markers_and_commands():
    steps = {s.name: s for s in run_scaling.plan_steps()}

    ft = steps["finetune_scale_A_n5k_seed1"]
    assert ft.cmd[-3:] == ["finetune", "--config", "configs/finetune_scale_A_n5k_seed1.yaml"]
    assert ft.done_marker == (
        run_scaling.REPO_ROOT
        / "outputs"
        / "finetune_scale_A_n5k_seed1"
        / "best"
        / "model.safetensors"
    )

    gen = steps["generate_scale_A_n5k_seed1"]
    assert gen.done_marker.name == "predictions_test.jsonl"
    assert "--model" in gen.cmd and "--tokenizer" in gen.cmd

    ev = steps["eval_ptcompute_ep3_seed42"]
    assert ev.done_marker == run_scaling.RESULTS_DIR / "finetune_ptcompute_ep3_seed42_test.json"
    assert ev.cmd[-2:] == ["--name", "finetune_ptcompute_ep3_seed42_test"]
    # The eval step prunes its config's checkpoints once metrics land.
    assert (
        ev.prune_checkpoints_in
        == run_scaling.REPO_ROOT / "outputs" / "finetune_ptcompute_ep3_seed42"
    )


def _copy_repo_configs(dest: Path) -> Path:
    """Copy the committed scaling configs into a tmp dir so a copy can be mutated."""
    dest.mkdir()
    for spec in gen_scaling_configs.iter_config_specs():
        name = f"{spec.stem}.yaml"
        shutil.copy(CONFIGS_DIR / name, dest / name)
    return dest


def test_preflight_accepts_repo_configs():
    run_scaling.preflight()  # must not raise against the committed configs


def test_preflight_rejects_wrong_pretrained_path(tmp_path):
    configs = _copy_repo_configs(tmp_path / "configs")
    # Break arm A's pretrained pointer -> preflight must catch it.
    bad = configs / "finetune_scale_A_n1k_seed0.yaml"
    bad.write_text(
        bad.read_text(encoding="utf-8").replace(
            "pretrained_model_path: outputs/pretrain/final",
            "pretrained_model_path: outputs/pretrain/epoch-1",
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        run_scaling.preflight(configs_dir=configs)
    assert "pretrained_model_path" in str(exc.value)


def test_preflight_rejects_duplicate_output_dir(tmp_path):
    configs = _copy_repo_configs(tmp_path / "configs")
    dup = configs / "finetune_scale_B_n1k_seed1.yaml"
    dup.write_text(
        dup.read_text(encoding="utf-8").replace(
            "output_dir: outputs/finetune_scale_B_n1k_seed1",
            "output_dir: outputs/finetune_scale_B_n1k_seed0",
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        run_scaling.preflight(configs_dir=configs)
    assert "output_dir" in str(exc.value)


def test_write_summary_renders_config_table(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    metrics = {"em": 0.01, "em_raw": 0.0, "codebleu": 0.55, "syntax_valid_rate": 0.9, "n": 30}
    (results / "finetune_scale_A_n5k_seed0_test.json").write_text(
        json.dumps({"metrics": metrics}), encoding="utf-8"
    )
    run_scaling.write_summary(tmp_path, results_dir=results)
    text = (tmp_path / "SUMMARY.md").read_text(encoding="utf-8")
    assert "| finetune_scale_A_n5k_seed0 | 0.0100 | 0.5500 | 0.9000 | 30 |" in text
    # Only configs with a results file appear.
    assert "finetune_scale_B_n1k_seed0" not in text


def test_write_summary_writes_nothing_without_results(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    run_scaling.write_summary(tmp_path, results_dir=results)
    assert not (tmp_path / "SUMMARY.md").exists()


def test_reuses_run_training_helpers():
    # DRY: the shared orchestration machinery is imported, not re-implemented.
    import run_training

    assert run_scaling.Step is run_training.Step
    assert run_scaling.run_step is run_training.run_step
    assert run_scaling.write_status is run_training.write_status
    assert run_scaling.prune_checkpoints is run_training.prune_checkpoints
