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
    """The current commit sha, or "unknown" -- provenance stamping must never fail a run."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "unknown"


SCRATCH_NAME_MARKER = "_local"


def is_scratch_run_name(name: str) -> bool:
    """True for the ad-hoc `*_local*` run names the repo gitignores (`results/*_local*.json`).

    These are the CLI's own defaults for `pop smoke` and `pop execbench`, so re-running
    either command is idempotent. Every other name is treated as possibly-published data
    and is guarded by `write_results`.
    """
    return SCRATCH_NAME_MARKER in name


def write_results(name: str, metrics: dict, config: dict, *, overwrite: bool | None = None) -> Path:
    """Write `results/<name>.json` with schema {config, metrics, n, timestamp, git_sha}.

    `name` must be a bare filename (no path separators, no `..`, no drive letter): the
    results directory is a flat namespace and `--name` comes straight from the CLI.

    Refuses to replace an existing file, raising `FileExistsError`. The repo's `results/`
    holds **committed, published** measurements that `docs/report.md` cites; silently
    overwriting one with a fresh (possibly truncated) run would falsify the study.
    `overwrite` defaults to `None` = "auto": the gitignored `*_local*` scratch names are
    replaceable (see `is_scratch_run_name`), anything else is guarded. Pass an explicit
    bool to override either way.

    The directory is resolved from the current working directory on purpose, so running
    a command from a scratch dir writes there instead of into the repo's `results/`.
    """
    # `Path(name).name` strips any directory, drive and root, so it round-trips only for a
    # bare filename. "." and ".." are special-cased: pathlib keeps ".." as a whole component.
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(
            f"results name must be a bare filename (no path separators or '..'), got {name!r}"
        )

    if overwrite is None:
        overwrite = is_scratch_run_name(name)

    results_dir = Path("results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.json"

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists and may be a committed result. Pass --name <other> to "
            f"write elsewhere, or delete the file if you intend to replace it."
        )

    payload = {
        "config": config,
        "metrics": metrics,
        "n": metrics.get("n"),
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
    }
    # Trailing newline: POSIX text-file convention, and it keeps `git diff` from reporting
    # "\ No newline at end of file" on every result. Matches scripts/build_benchmark_manifests.py.
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
