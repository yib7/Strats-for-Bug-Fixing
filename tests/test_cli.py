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


def test_the_parser_and_the_dispatch_table_describe_the_same_cli():
    """Every subcommand argparse accepts has a handler, and vice versa.

    The parser is written out subcommand by subcommand (each has its own flags) while dispatch
    is a table, so the two halves are only kept in step by this check.
    """
    import argparse

    from pop.cli import COMMANDS, build_parser

    registered = next(
        action.choices
        for action in build_parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert sorted(registered) == sorted(COMMANDS)
    assert sorted(COMMANDS) == sorted(SUBCOMMANDS)


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


# ---------------------------------------------------------------------------
# Malformed input: a clear message, never a traceback (regression -- every one
# of these used to dump a raw yaml/pydantic/json stack).
# ---------------------------------------------------------------------------


def _assert_clean_error(result, *needles: str) -> None:
    assert result.returncode == 2, f"expected rc=2, got {result.returncode}: {result.stderr}"
    assert "Traceback" not in result.stderr, result.stderr
    for needle in needles:
        assert needle in result.stderr, f"{needle!r} not in {result.stderr!r}"


def test_config_that_is_not_valid_yaml_is_a_clean_error(tmp_path):
    config = tmp_path / "malformed.yaml"
    config.write_text("seed: [unclosed\n  bad: yaml\n", encoding="utf-8")
    _assert_clean_error(run_pop("finetune", "--config", str(config)), "not valid YAML")


def test_config_that_is_not_a_mapping_is_a_clean_error(tmp_path):
    config = tmp_path / "list.yaml"
    config.write_text("- a\n- b\n", encoding="utf-8")
    _assert_clean_error(run_pop("finetune", "--config", str(config)), "expected a YAML mapping")


def test_config_with_wrong_types_is_a_clean_error(tmp_path):
    config = tmp_path / "wrongtype.yaml"
    config.write_text('epochs: "not-an-int"\n', encoding="utf-8")
    result = run_pop("finetune", "--config", str(config))
    _assert_clean_error(result, "invalid config", "epochs")
    assert "pydantic" not in result.stderr.lower()


def test_eval_predictions_that_are_not_json_report_the_line(tmp_path):
    preds = tmp_path / "bad.jsonl"
    preds.write_text('{"prediction": "a", "reference": "b"}\nnot json at all\n', encoding="utf-8")
    _assert_clean_error(
        run_pop("eval", "--predictions", str(preds)), "bad.jsonl:2", "not valid JSON"
    )


def test_eval_predictions_missing_a_key_report_the_line(tmp_path):
    preds = tmp_path / "nokey.jsonl"
    preds.write_text('{"prediction": "a"}\n', encoding="utf-8")
    _assert_clean_error(run_pop("eval", "--predictions", str(preds)), "nokey.jsonl:1", "reference")


def test_execbench_predictions_that_are_not_json_report_the_line(tmp_path):
    preds = tmp_path / "bad.jsonl"
    preds.write_text("{oops\n", encoding="utf-8")
    _assert_clean_error(
        run_pop("execbench", "--predictions", str(preds), "--bench", "quixbugs"),
        "bad.jsonl:1",
        "not valid JSON",
    )


def test_execbench_predictions_missing_a_key_report_the_line(tmp_path):
    preds = tmp_path / "nopred.jsonl"
    preds.write_text('{"bug_id": "BITCOUNT"}\n', encoding="utf-8")
    _assert_clean_error(
        run_pop("execbench", "--predictions", str(preds), "--bench", "quixbugs"),
        "nopred.jsonl:1",
        "prediction",
    )


def test_pop_traceback_env_var_restores_the_raw_exception(tmp_path):
    import os

    config = tmp_path / "list.yaml"
    config.write_text("- a\n", encoding="utf-8")
    env = {**os.environ, "POP_TRACEBACK": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "pop.cli", "finetune", "--config", str(config)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "Traceback" in result.stderr  # opt-in debugging escape hatch


# ---------------------------------------------------------------------------
# Exit-code convention: 2 = usage/input error, 1 = the run ran and failed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "make_missing"),
    [
        (("smoke", "--config"), "missing.yaml"),
        (("eval", "--predictions"), "missing.jsonl"),
        (("execbench", "--predictions"), "missing.jsonl"),
    ],
)
def test_missing_input_file_exits_two(args, make_missing, tmp_path):
    missing = tmp_path / make_missing
    result = run_pop(*args, str(missing))
    assert result.returncode == 2, result.stderr
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
    assert result.returncode == 2  # 2 = usage/input error
    assert "config" in result.stderr.lower()


def test_smoke_rejects_missing_config_file(tmp_path):
    missing = tmp_path / "does-not-exist.yaml"
    result = run_pop("smoke", "--config", str(missing))
    assert result.returncode == 2


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
    assert result.returncode == 2


def test_eval_rejects_missing_predictions_file(tmp_path):
    missing = tmp_path / "does-not-exist.jsonl"
    result = run_pop("eval", "--predictions", str(missing))
    assert result.returncode == 2


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
