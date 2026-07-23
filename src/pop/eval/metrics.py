"""Prediction-vs-reference metrics: EM (raw + normalized), CodeBLEU, syntax validity.

`evaluate_predictions` is the aggregate scorer that later phases (pretrain/finetune
eval, RAG eval, execbench) all call. `write_results` persists a run's metrics to
`results/<name>.json` with a stable schema (config, metrics, n, timestamp, git_sha)
so results can be compared across configs/runs.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import tree_sitter
import tree_sitter_java
from codebleu import calc_codebleu

from pop.eval.normalize import exact_match, exact_match_raw

_JAVA_LANGUAGE = tree_sitter.Language(tree_sitter_java.language())
_JAVA_PARSER = tree_sitter.Parser(_JAVA_LANGUAGE)


def _is_valid_java_method(code: str) -> bool:
    """Parse `code` as a Java method body wrapped in a class.

    CodeXGLUE samples are bare method bodies, not full compilation units, so
    they are wrapped in `class _W { ... }` before parsing. Valid means the
    parse tree has no ERROR nodes and is non-empty.
    """
    if not code.strip():
        return False
    wrapped = f"class _W {{ {code} }}"
    tree = _JAVA_PARSER.parse(wrapped.encode("utf-8"))
    root = tree.root_node
    return root.child_count > 0 and not root.has_error


def evaluate_predictions(preds: list[str], refs: list[str]) -> dict:
    """Compute EM (raw + normalized), CodeBLEU, and syntax-valid rate.

    Returns a dict with keys: em, em_raw, codebleu, syntax_valid_rate, n.
    """
    if not preds or not refs:
        raise ValueError("preds and refs must be non-empty")
    if len(preds) != len(refs):
        raise ValueError(f"preds and refs must be the same length: {len(preds)} != {len(refs)}")

    n = len(preds)
    em = sum(exact_match(p, r) for p, r in zip(preds, refs, strict=True)) / n
    em_raw = sum(exact_match_raw(p, r) for p, r in zip(preds, refs, strict=True)) / n
    syntax_valid_rate = sum(_is_valid_java_method(p) for p in preds) / n

    codebleu_result = calc_codebleu(references=refs, predictions=preds, lang="java")
    codebleu = codebleu_result["codebleu"]

    return {
        "em": em,
        "em_raw": em_raw,
        "codebleu": codebleu,
        "syntax_valid_rate": syntax_valid_rate,
        "n": n,
    }


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_results(name: str, metrics: dict, config: dict) -> Path:
    """Write `results/<name>.json` with schema {config, metrics, n, timestamp, git_sha}."""
    results_dir = Path("results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.json"

    payload = {
        "config": config,
        "metrics": metrics,
        "n": metrics.get("n"),
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
