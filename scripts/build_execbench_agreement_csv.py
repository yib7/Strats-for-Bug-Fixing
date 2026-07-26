"""Aggregate the per-arm execution results into ``results/execbench_agreement.csv``.

The CSV-builder half of the execution-vs-CodeBLEU agreement figure: it joins each
arm's **execution** result (``results/execbench_<arm>.json`` -- pass@1 over the 201
vendored QuixBugs + HumanEval-Java bugs) with that arm's **CodeBLEU** (from its
CodeXGLUE-test finetune/rag/lora result) into the single tidy CSV that
``scripts/figures/execution_vs_codebleu.py`` renders.

Output schema (EXACTLY what ``execution_vs_codebleu.load_agreement_rows`` reads)::

    arm,codebleu,pass_at_1,n_bugs

Arm -> sources (mirrors ``scripts/figures/four_arm_comparison.collect_arms``):

* **A** pretrain->finetune T5 - CodeBLEU ``finetune_A_ep10``; exec ``execbench_A.json``
* **B** from-scratch T5       - CodeBLEU mean(``finetune_B_seed*``); exec ``execbench_B.json``
* **C** RAG Qwen              - CodeBLEU best ``rag_*_test.json``; exec ``execbench_C.json``
* **D** LoRA Qwen             - CodeBLEU ``lora_qwen_test``; exec ``execbench_D.json``

``pass_at_1`` is the execution result's ``metrics.pass_rate`` -- with one prediction
per bug (n=1 sample/problem) pass@1 == pass_rate -- and ``n_bugs`` is ``metrics.n``.

A row is emitted only when **both** the execution result and the CodeBLEU exist for
that arm (a scatter point needs both axes), so this degrades gracefully before /
partway through the GPU batch. Producing zero rows is fine: the figure then renders
its labelled ``⟨pending GPU batch⟩`` placeholder.

Usage:
    python scripts/build_execbench_agreement_csv.py           # -> results/execbench_agreement.csv
    python scripts/build_execbench_agreement_csv.py --out X
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from pop.eval.metrics import is_scratch_run_name

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_NAME = "execbench_agreement.csv"
OUT_PATH = RESULTS_DIR / OUT_NAME

CSV_FIELDS = ["arm", "codebleu", "pass_at_1", "n_bugs"]
ARMS = ("A", "B", "C", "D")

# `pop eval` writes each CodeXGLUE-test run as results/<config>_test.json.
B_SEED_RUNS = ("finetune_B_seed0_test", "finetune_B_seed1_test", "finetune_B_seed2_test")
A_REF = "finetune_A_ep10_test"
LORA_REF = "lora_qwen_test"


def _load_metrics(name: str, results_dir: Path) -> dict | None:
    path = results_dir / f"{name}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("metrics", {})


def _best_rag_codebleu(results_dir: Path) -> float | None:
    """Highest CodeBLEU across committed ``rag_*_test.json`` (arm C = the best RAG config).

    Gitignored ``*_local*`` scratch runs are skipped. This CSV is a *committed* file that
    ``scripts/figures/make_all.py`` rebuilds on every documented reproduce run, so without
    the filter a contributor's local experiment (say ``rag_bm25_k3_local_test.json``)
    silently moves the published arm-C point and leaves the working tree dirty for reasons
    that read as inexplicable.
    """
    best: float | None = None
    for path in sorted(results_dir.glob("rag_*_test.json")):
        if is_scratch_run_name(path.stem):
            continue
        metrics = json.loads(path.read_text(encoding="utf-8")).get("metrics", {})
        cb = metrics.get("codebleu")
        if cb is None:
            continue
        if best is None or cb > best:
            best = cb
    return best


def arm_codebleu(arm: str, results_dir: Path) -> float | None:
    """The CodeXGLUE-test CodeBLEU for one arm, or None if its result JSON is absent."""
    if arm == "A":
        metrics = _load_metrics(A_REF, results_dir)
        return metrics.get("codebleu") if metrics else None
    if arm == "B":
        vals = [
            metrics["codebleu"]
            for name in B_SEED_RUNS
            if (metrics := _load_metrics(name, results_dir)) and "codebleu" in metrics
        ]
        return sum(vals) / len(vals) if vals else None
    if arm == "C":
        return _best_rag_codebleu(results_dir)
    if arm == "D":
        metrics = _load_metrics(LORA_REF, results_dir)
        return metrics.get("codebleu") if metrics else None
    raise ValueError(f"unknown arm {arm!r}")


def arm_execution(arm: str, results_dir: Path) -> tuple[float, int | None] | None:
    """(pass@1, n_bugs) from ``results/execbench_<arm>.json``, or None if it does not exist."""
    metrics = _load_metrics(f"execbench_{arm}", results_dir)
    if metrics is None or metrics.get("pass_rate") is None:
        return None
    return float(metrics["pass_rate"]), metrics.get("n")


def build_rows(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """One row per arm that has BOTH an execution result and a CodeBLEU (graceful partial)."""
    results_dir = Path(results_dir)
    rows: list[dict] = []
    for arm in ARMS:
        execution = arm_execution(arm, results_dir)
        if execution is None:
            continue
        codebleu = arm_codebleu(arm, results_dir)
        if codebleu is None:
            continue
        pass_at_1, n_bugs = execution
        rows.append(
            {
                "arm": arm,
                "codebleu": codebleu,
                "pass_at_1": pass_at_1,
                "n_bugs": "" if n_bugs is None else n_bugs,
            }
        )
    return rows


def write_csv(rows: list[dict], out_path: Path = OUT_PATH) -> Path:
    """Write ``rows`` in the agreement schema; returns the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # lineterminator="\n": csv defaults to CRLF, which would rewrite this committed file with
    # different bytes on every platform (see .gitattributes).
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help="dir holding the execbench_*/finetune_*/rag_*/lora_* result JSONs (default: results/)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=f"output CSV path (default: <results-dir>/{OUT_NAME})",
    )
    return parser


def resolve_out(args: argparse.Namespace) -> Path:
    """Where to write, defaulting *beside the inputs* rather than into the repo's results/.

    ``--out`` used to default to the committed ``results/execbench_agreement.csv`` no matter
    what ``--results-dir`` said, so reading a partial directory wrote a silently truncated
    version of a published file (``build_rows`` degrades to fewer rows rather than failing).
    Deriving the default from ``--results-dir`` keeps the plain no-flags invocation writing
    exactly where it always did, because ``--results-dir`` itself defaults to ``results/``.
    """
    return Path(args.out) if args.out is not None else Path(args.results_dir) / OUT_NAME


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    rows = build_rows(Path(args.results_dir))
    path = write_csv(rows, resolve_out(args))
    arms = ", ".join(r["arm"] for r in rows) or "none"
    print(f"wrote {path} ({len(rows)} arm rows: {arms})")
    if not rows:
        print("  (no execbench_<arm>.json with a matching CodeBLEU yet -- run the exec-eval first)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
