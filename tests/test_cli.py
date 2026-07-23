"""Tests for pop.cli."""

import subprocess
import sys

import pytest

SUBCOMMANDS = [
    "smoke",
    "tokenizer",
    "pretrain",
    "finetune",
    "generate",
    "eval",
    "rag",
    "lora",
    "lora-generate",
    "execbench",
]
STUB_SUBCOMMANDS: list[str] = []
CONFIG_SUBCOMMANDS = ["pretrain", "finetune", "rag", "lora", "lora-generate"]


def run_pop(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pop.cli", *args],
        capture_output=True,
        text=True,
    )


def test_help_exits_zero():
    result = run_pop("--help")
    assert result.returncode == 0


@pytest.mark.parametrize("subcommand", SUBCOMMANDS)
def test_help_lists_subcommand(subcommand):
    result = run_pop("--help")
    assert subcommand in result.stdout


@pytest.mark.parametrize("subcommand", STUB_SUBCOMMANDS)
def test_stub_subcommand_not_implemented(subcommand):
    result = run_pop(subcommand)
    assert result.returncode == 2
    assert "not yet implemented" in result.stderr.lower()


def test_no_args_exits_nonzero():
    result = run_pop()
    assert result.returncode != 0


@pytest.mark.parametrize("subcommand", CONFIG_SUBCOMMANDS)
def test_implemented_subcommand_requires_config(subcommand):
    result = run_pop(subcommand)
    assert result.returncode == 2
    assert "--config" in result.stderr


@pytest.mark.parametrize("subcommand", CONFIG_SUBCOMMANDS)
def test_implemented_subcommand_help_shows_config_option(subcommand):
    result = run_pop(subcommand, "--help")
    assert result.returncode == 0
    assert "--config" in result.stdout


@pytest.mark.parametrize("subcommand", CONFIG_SUBCOMMANDS)
def test_implemented_subcommand_rejects_missing_config_file(subcommand, tmp_path):
    missing = tmp_path / "does-not-exist.yaml"
    result = run_pop(subcommand, "--config", str(missing))
    assert result.returncode != 0


@pytest.mark.parametrize("subcommand", CONFIG_SUBCOMMANDS)
def test_implemented_subcommand_missing_config_file_is_clean_exit(subcommand, tmp_path):
    # Regression: a missing --config file used to raise a raw FileNotFoundError
    # traceback instead of a clean CLI error (smoke/eval/execbench already
    # guarded against this; pretrain/finetune/rag did not).
    missing = tmp_path / "does-not-exist.yaml"
    result = run_pop(subcommand, "--config", str(missing))
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert str(missing) in result.stderr


def test_rag_refuses_non_train_kb_split_without_override(tmp_path):
    config_path = tmp_path / "rag.yaml"
    config_path.write_text("kb_split: validation\n", encoding="utf-8")
    result = run_pop("rag", "--config", str(config_path))
    assert result.returncode == 1
    assert "train" in result.stderr.lower()


def test_rag_help_shows_allow_non_train_kb_flag():
    result = run_pop("rag", "--help")
    assert result.returncode == 0
    assert "--allow-non-train-kb" in result.stdout


def test_smoke_help_shows_config_option():
    result = run_pop("smoke", "--help")
    assert result.returncode == 0
    assert "--config" in result.stdout


def test_smoke_defaults_to_configs_smoke_yaml_and_rejects_missing_cwd_config(tmp_path):
    # No configs/smoke.yaml in an empty cwd -> the default path doesn't resolve.
    result = subprocess.run(
        [sys.executable, "-m", "pop.cli", "smoke"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 1
    assert "config" in result.stderr.lower()


def test_smoke_rejects_missing_config_file(tmp_path):
    missing = tmp_path / "does-not-exist.yaml"
    result = run_pop("smoke", "--config", str(missing))
    assert result.returncode != 0


def test_generate_help_shows_model_and_tokenizer_options():
    result = run_pop("generate", "--help")
    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--tokenizer" in result.stdout
    assert "--num-beams" in result.stdout


def test_generate_requires_model_and_tokenizer():
    result = run_pop("generate")
    assert result.returncode == 2
    assert "--model" in result.stderr


def test_generate_rejects_missing_model_dir(tmp_path):
    tok = tmp_path / "tok.model"
    tok.write_text("", encoding="utf-8")
    missing = tmp_path / "no-such-model"
    result = run_pop("generate", "--model", str(missing), "--tokenizer", str(tok))
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert str(missing) in result.stderr


def test_tokenizer_help_shows_options():
    result = run_pop("tokenizer", "--help")
    assert result.returncode == 0
    assert "--out" in result.stdout
    assert "--vocab-size" in result.stdout
    assert "--corpus-samples" in result.stdout


def test_eval_requires_predictions():
    result = run_pop("eval")
    assert result.returncode == 2
    assert "--predictions" in result.stderr


def test_eval_help_shows_predictions_option():
    result = run_pop("eval", "--help")
    assert result.returncode == 0
    assert "--predictions" in result.stdout


def test_execbench_help_shows_flags():
    result = run_pop("execbench", "--help")
    assert result.returncode == 0
    assert "--validate-references" in result.stdout
    assert "--predictions" in result.stdout
    assert "--bench" in result.stdout
    assert "--limit" in result.stdout
    assert "--jobs" in result.stdout


def test_execbench_requires_exactly_one_mode():
    result = run_pop("execbench")
    assert result.returncode == 2
    assert "--validate-references" in result.stderr
    assert "--predictions" in result.stderr


def test_execbench_rejects_both_modes(tmp_path):
    predictions = tmp_path / "preds.jsonl"
    predictions.write_text("", encoding="utf-8")
    result = run_pop("execbench", "--validate-references", "--predictions", str(predictions))
    assert result.returncode == 2


def test_execbench_predictions_rejects_missing_file(tmp_path):
    missing = tmp_path / "does-not-exist.jsonl"
    result = run_pop("execbench", "--predictions", str(missing))
    assert result.returncode != 0


def test_eval_rejects_missing_predictions_file(tmp_path):
    missing = tmp_path / "does-not-exist.jsonl"
    result = run_pop("eval", "--predictions", str(missing))
    assert result.returncode != 0


def test_eval_prints_metrics_and_writes_results(tmp_path):
    predictions = tmp_path / "preds.jsonl"
    predictions.write_text(
        '{"prediction": "int  x = 1;", "reference": "int x = 1;"}\n'
        '{"prediction": "int y = 2;", "reference": "int y = 3;"}\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "pop.cli", "eval", "--predictions", str(predictions)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0
    assert '"em"' in result.stdout
    results_dir = tmp_path / "results"
    written = list(results_dir.glob("*.json"))
    assert len(written) == 1
