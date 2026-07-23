"""Unit tests for scripts/run_rag.py (the RAG-sweep orchestrator).

Mirrors tests/test_run_training.py: the scripts/ directory is added to sys.path
and the module imported directly. No GPU / network / model downloads here -- only
the planning, preflight, and summary pieces (all CPU-only, config-file I/O at most).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_rag  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def test_plan_lists_16_steps_rag_then_eval():
    steps = run_rag.plan_steps()
    assert len(steps) == 16  # 8 configs x (rag + eval)
    names = [s.name for s in steps]
    # First config: bm25 k0 -- rag step immediately followed by its eval step.
    assert names[:2] == ["rag_bm25_k0", "eval_bm25_k0"]
    # Every rag step is followed by its matching eval step.
    for i in range(0, 16, 2):
        assert names[i].startswith("rag_")
        assert names[i + 1] == names[i].replace("rag_", "eval_", 1)


def test_plan_markers_and_commands():
    steps = {s.name: s for s in run_rag.plan_steps()}

    rag = steps["rag_bm25_k3"]
    assert rag.done_marker == run_rag.REPO_ROOT / "outputs" / "rag_bm25_k3" / "predictions.jsonl"
    assert rag.cmd[-3:] == ["rag", "--config", "configs/rag_bm25_k3.yaml"]

    ev = steps["eval_codebert_k5"]
    assert ev.done_marker == run_rag.RESULTS_DIR / "rag_codebert_k5_test.json"
    assert ev.cmd[-2:] == ["--name", "rag_codebert_k5_test"]
    assert "--predictions" in ev.cmd
    preds_idx = ev.cmd.index("--predictions") + 1
    assert ev.cmd[preds_idx] == str(
        run_rag.REPO_ROOT / "outputs" / "rag_codebert_k5" / "predictions.jsonl"
    )


def _copy_repo_configs(dest: Path) -> Path:
    """Copy the 8 committed rag configs into a tmp dir so a copy can be mutated."""
    dest.mkdir()
    for retriever in ("bm25", "codebert"):
        for k in (0, 1, 3, 5):
            name = f"rag_{retriever}_k{k}.yaml"
            shutil.copy(CONFIGS_DIR / name, dest / name)
    return dest


def test_preflight_accepts_repo_configs():
    run_rag.preflight()  # must not raise against the committed configs


def test_preflight_rejects_non_train_kb_split(tmp_path):
    configs = _copy_repo_configs(tmp_path / "configs")
    leaky = configs / "rag_bm25_k3.yaml"
    leaky.write_text(
        leaky.read_text(encoding="utf-8").replace("kb_split: train", "kb_split: test"),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        run_rag.preflight(configs_dir=configs)
    assert "kb_split" in str(exc.value)


def test_preflight_rejects_duplicate_output_dir(tmp_path):
    configs = _copy_repo_configs(tmp_path / "configs")
    dup = configs / "rag_bm25_k1.yaml"
    dup.write_text(
        dup.read_text(encoding="utf-8").replace(
            "output_dir: outputs/rag_bm25_k1", "output_dir: outputs/rag_bm25_k0"
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        run_rag.preflight(configs_dir=configs)
    assert "output_dir" in str(exc.value)


def test_write_summary_renders_retriever_k_table(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    metrics = {"em": 0.0, "em_raw": 0.0, "codebleu": 0.42, "syntax_valid_rate": 0.88, "n": 25}
    (results / "rag_bm25_k3_test.json").write_text(
        json.dumps({"metrics": metrics}), encoding="utf-8"
    )
    run_rag.write_summary(tmp_path, results_dir=results)
    text = (tmp_path / "SUMMARY.md").read_text(encoding="utf-8")
    assert "| bm25 | 3 | 0.4200 | 0.8800 | 0.0000 | 25 |" in text
    # Only configs with a results file appear.
    assert "codebert" not in text


def test_write_summary_writes_nothing_without_results(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    run_rag.write_summary(tmp_path, results_dir=results)
    assert not (tmp_path / "SUMMARY.md").exists()


def test_reuses_run_training_helpers():
    # DRY: the shared orchestration machinery is imported, not re-implemented.
    import run_training

    assert run_rag.Step is run_training.Step
    assert run_rag.run_step is run_training.run_step
    assert run_rag.write_status is run_training.write_status
