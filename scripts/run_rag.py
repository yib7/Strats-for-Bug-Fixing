"""RAG-sweep orchestrator: for each of the 8 RAG configs, `pop rag` -> `pop eval`.

Runs the full retrieval-augmented-prompting sweep (BM25/CodeBERT x k=0/1/3/5) as a
one-command, resumable job -- the RAG analogue of `scripts/run_training.py`. Every
step is a `python -m pop.cli ...` subprocess, in order, with:

- **Resumability.** Every step has a durable done-marker (its output artifact):
  a `rag` step's marker is `<output_dir>/predictions.jsonl`, its `eval` step's
  marker is `results/rag_<retriever>_k<k>_test.json`. Steps whose marker already
  exists are skipped, so re-running after a crash / reboot / Colab session limit
  continues where the run left off.
- **File-based progress logging** watchable from outside the process: a live
  ``STATUS.md`` table, an ``orchestrator.log``, and one ``<step>.log`` per step
  (also echoed to stdout, so a Colab cell shows it live). On Colab, ``logs/`` is a
  symlink into Google Drive (the ``notebooks/colab_rag.ipynb`` setup cell makes it
  one), so ``STATUS.md`` is readable from the Drive web/phone app during the run.

This mirrors `scripts/run_training.py` and **reuses** its Step/plan/resume/status/log
machinery (imported, not copy-pasted). Unlike the T5 training orchestrator there is
no EM>0 gate (that was T5-decoding-specific) and no checkpoint pruning (RAG produces
no checkpoints).

Usage:
    python scripts/run_rag.py           # the full 8-config RAG sweep (needs a GPU)
    python scripts/run_rag.py --list    # print the plan + run preflight, CPU-only, run nothing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_training import (  # noqa: E402  (sibling script; path inserted above)
    REPO_ROOT,
    RESULTS_DIR,
    Step,
    _fmt_duration,
    _now,
    _pop,
    run_step,
    write_status,
)

LOGS_ROOT = REPO_ROOT / "logs" / "rag"
CONFIGS_DIR = REPO_ROOT / "configs"

RETRIEVERS = ("bm25", "codebert")
KS = (0, 1, 3, 5)


def _config_id(retriever: str, k: int) -> str:
    return f"{retriever}_k{k}"


def plan_steps(configs_dir: Path = CONFIGS_DIR) -> list[Step]:
    """The 16-step plan: each of the 8 configs contributes a `rag` then an `eval` step."""
    from pop.config import RagConfig

    steps: list[Step] = []
    for retriever in RETRIEVERS:
        for k in KS:
            cfg_id = _config_id(retriever, k)
            config_rel = f"configs/rag_{cfg_id}.yaml"
            cfg = RagConfig.from_yaml(configs_dir / f"rag_{cfg_id}.yaml")
            output_dir = REPO_ROOT / cfg.output_dir
            predictions = output_dir / "predictions.jsonl"
            results_name = f"rag_{cfg_id}_test"
            steps.append(Step(f"rag_{cfg_id}", _pop("rag", "--config", config_rel), predictions))
            steps.append(
                Step(
                    f"eval_{cfg_id}",
                    _pop("eval", "--predictions", str(predictions), "--name", results_name),
                    RESULTS_DIR / f"{results_name}.json",
                )
            )
    return steps


def preflight(configs_dir: Path = CONFIGS_DIR) -> None:
    """Fail fast (before any GPU second) if the config wiring is off.

    Asserts all 8 RAG configs parse, each builds its retriever KB from the ``train``
    split (leakage guard), and each writes to a distinct ``output_dir``.
    """
    if not (REPO_ROOT / "pyproject.toml").is_file():
        raise SystemExit(f"run_rag: repo root not found at {REPO_ROOT}")

    from pop.config import RagConfig

    problems: list[str] = []
    seen_output_dirs: dict[Path, str] = {}
    for retriever in RETRIEVERS:
        for k in KS:
            cfg_id = _config_id(retriever, k)
            path = configs_dir / f"rag_{cfg_id}.yaml"
            if not path.is_file():
                problems.append(f"{cfg_id}: config not found at {path}")
                continue
            try:
                cfg = RagConfig.from_yaml(path)
            except Exception as exc:  # noqa: BLE001 -- surface any parse/validation error
                problems.append(f"{cfg_id}: failed to parse ({exc})")
                continue
            if cfg.retriever != retriever:
                problems.append(f"{cfg_id}: retriever {cfg.retriever!r} != {retriever!r}")
            if cfg.k != k:
                problems.append(f"{cfg_id}: k {cfg.k} != {k}")
            if cfg.kb_split != "train":
                problems.append(
                    f"{cfg_id}: kb_split {cfg.kb_split!r} != 'train' (leakage guard: the "
                    "retriever knowledge base must be the train split)"
                )
            out = Path(cfg.output_dir)
            if out in seen_output_dirs:
                problems.append(f"{cfg_id}: output_dir {out} collides with {seen_output_dirs[out]}")
            else:
                seen_output_dirs[out] = cfg_id

    if problems:
        raise SystemExit("run_rag preflight failed:\n  " + "\n  ".join(problems))


def write_summary(run_dir: Path, results_dir: Path = RESULTS_DIR) -> None:
    """Render a retriever x k table (CodeBLEU / syntax_valid_rate / em) to SUMMARY.md.

    Reads whatever ``results/rag_*_test.json`` exist; writes nothing if none do yet.
    Mirrors ``run_training.write_summary``.
    """
    lines = [
        "# RAG sweep results summary",
        "",
        "| retriever | k | codebleu | syntax_valid_rate | em | n |",
        "|---|---|---|---|---|---|",
    ]
    found = False
    for retriever in RETRIEVERS:
        for k in KS:
            path = results_dir / f"rag_{_config_id(retriever, k)}_test.json"
            if not path.is_file():
                continue
            found = True
            m = json.loads(path.read_text(encoding="utf-8"))["metrics"]
            lines.append(
                f"| {retriever} | {k} | {m['codebleu']:.4f} "
                f"| {m['syntax_valid_rate']:.4f} | {m['em']:.4f} | {m['n']} |"
            )
    if found:
        (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_plan(steps: list[Step]) -> None:
    print(f"RAG sweep plan: {len(steps)} steps ({len(steps) // 2} configs)")
    for i, step in enumerate(steps, 1):
        status = "done" if step.is_done() else "pending"
        print(
            f"  {i:2d}. {step.name:20s} [{status:>7}] -> {step.done_marker.relative_to(REPO_ROOT)}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="print the plan and run preflight WITHOUT executing any step (CPU-only proof; "
        "a real --dry-run is impossible here since even one step needs a model download)",
    )
    args = parser.parse_args(argv)

    # Windows consoles are often cp1252; subprocess output can contain Unicode (tqdm bars,
    # tokenizer logs). Degrade unencodable chars to '?' on the console -- the per-step .log
    # files are written as UTF-8 and stay lossless.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    preflight()
    steps = plan_steps()

    if args.list_only:
        _print_plan(steps)
        print("preflight: OK")
        return 0

    run_dir = LOGS_ROOT / "sweep"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ}
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Local AMD (Windows/ROCm): reduce allocator fragmentation. Ignored on CUDA.
    env.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

    orchestrator_log = run_dir / "orchestrator.log"

    def olog(message: str) -> None:
        line = f"[{_now()}] {message}"
        with orchestrator_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    rows = [
        {"name": s.name, "status": "done (earlier run)" if s.is_done() else "pending"}
        for s in steps
    ]
    done_already = sum(r["status"].startswith("done") for r in rows)
    olog(f"run dir: {run_dir}")
    olog(f"{done_already}/{len(steps)} steps already complete; resuming at the first pending step")
    write_status(run_dir, rows)

    for i, step in enumerate(steps):
        row = rows[i]
        if step.is_done():
            olog(f"skip {step.name} (marker exists: {step.done_marker.name})")
            continue
        row.update(status="running", started=_now())
        write_status(run_dir, rows)
        olog(f"start {step.name}")
        t0 = time.monotonic()
        rc = run_step(step, run_dir, env)
        duration = _fmt_duration(time.monotonic() - t0)
        if rc != 0:
            row.update(status="FAILED", duration=duration, result=f"exit {rc}; see {step.name}.log")
            write_status(run_dir, rows)
            olog(f"FAILED {step.name} (exit {rc}) -- fix the cause and re-run; done steps skip")
            return 1
        if not step.done_marker.exists():
            row.update(
                status="FAILED",
                duration=duration,
                result=f"exit 0, missing {step.done_marker.name}",
            )
            write_status(run_dir, rows)
            olog(f"FAILED {step.name}: exit 0 but done-marker missing: {step.done_marker}")
            return 1
        row.update(status="done", duration=duration, result=step.done_marker.name)
        write_status(run_dir, rows)
        olog(f"done {step.name} in {duration}")
        # Refresh the results table after every eval so a mid-sweep disconnect still
        # leaves a partial SUMMARY.md on Drive.
        if step.name.startswith("eval_"):
            write_summary(run_dir)

    write_summary(run_dir)
    olog("all steps complete; SUMMARY.md written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
