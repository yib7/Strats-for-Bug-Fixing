"""Phase-2 training orchestrator: tokenizer -> pretrain -> 6x (finetune -> generate -> eval).

Runs every Phase-2 step as a `python -m pop.cli ...` subprocess, in order, with:

- **Resumability.** Every step has a durable done-marker (its output artifact);
  steps whose marker already exists are skipped, so re-running this script
  after a crash / reboot / Colab session limit continues where the run left
  off. *Mid-step* resume is handled inside the trainers themselves, which
  reload the latest ``checkpoint-*`` in their output dir (see
  ``pop.train.pretrain`` / ``pop.train.finetune``).
- **File-based progress logging** watchable from outside the process: a live
  ``STATUS.md`` table, an ``orchestrator.log``, and one ``<step>.log`` per
  step capturing subprocess output (also echoed to stdout, so a Colab cell
  shows it live). On Colab, ``logs/`` is a symlink into Google Drive (the
  ``notebooks/colab_phase2.ipynb`` setup cell makes it one), so ``STATUS.md``
  is readable from the Drive web/phone app while the run is going.
- **The EM>0 gate** (docs/gpu-reproduction.md): after the FIRST system's eval, if
  normalized exact match is exactly 0.0, halt instead of spending GPU hours
  on the remaining five systems -- all-zero EM at this scale means a decoding
  bug, and the documented next step is a greedy-vs-beam-5 sweep, not more
  training. ``--skip-gate`` bypasses it after such a diagnosis.
- **Disk pruning:** once a system's eval has landed, its intermediate
  ``checkpoint-*`` dirs are deleted (``best/`` is kept), bounding a Google
  Drive workspace to a few GB.

Step order puts the headline A-vs-B comparison first (A_ep10, then B_seed0)
so the experiment's core answer exists after the first two systems and the
remaining sweeps only refine it.

Usage:
    python scripts/run_training.py                  # the full Phase-2 run
    python scripts/run_training.py --dry-run        # plumbing check via `pop smoke` (CPU, ~1 min)
    python scripts/run_training.py --limit-eval 50  # debug: cap generate/eval to 50 samples
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_ROOT = REPO_ROOT / "logs" / "train"
TOKENIZER_MODEL = REPO_ROOT / "outputs" / "tokenizer" / "tokenizer.model"
PRETRAIN_FINAL = REPO_ROOT / "outputs" / "pretrain" / "final"
RESULTS_DIR = REPO_ROOT / "results"

# Headline A-vs-B systems first for early signal; the rest refine the curves.
SYSTEMS = ("A_ep10", "B_seed0", "A_ep3", "A_ep1", "B_seed1", "B_seed2")
GATE_SYSTEM = "A_ep10"


@dataclass
class Step:
    name: str
    cmd: list[str]
    done_marker: Path
    prune_checkpoints_in: Path | None = None
    always_run: bool = False

    def is_done(self) -> bool:
        return not self.always_run and self.done_marker.exists()


def _pop(*args: str) -> list[str]:
    return [sys.executable, "-m", "pop.cli", *args]


def plan_steps(limit_eval: int | None = None) -> list[Step]:
    steps = [
        Step("tokenizer", _pop("tokenizer"), TOKENIZER_MODEL),
        Step(
            "pretrain",
            _pop("pretrain", "--config", "configs/pretrain_10ep.yaml"),
            PRETRAIN_FINAL / "model.safetensors",
        ),
    ]
    for system in SYSTEMS:
        ft_dir = REPO_ROOT / "outputs" / f"finetune_{system}"
        best = ft_dir / "best"
        predictions = best / "predictions_test.jsonl"
        results_name = f"finetune_{system}_test"
        generate_cmd = _pop("generate", "--model", str(best), "--tokenizer", str(TOKENIZER_MODEL))
        if limit_eval is not None:
            generate_cmd += ["--limit", str(limit_eval)]
        steps += [
            Step(
                f"finetune_{system}",
                _pop("finetune", "--config", f"configs/finetune_{system}.yaml"),
                best / "model.safetensors",
            ),
            Step(f"generate_{system}", generate_cmd, predictions),
            Step(
                f"eval_{system}",
                _pop("eval", "--predictions", str(predictions), "--name", results_name),
                RESULTS_DIR / f"{results_name}.json",
                prune_checkpoints_in=ft_dir,
            ),
        ]
    return steps


def plan_dry_run_steps() -> list[Step]:
    """The one-step `--dry-run` plan: `pop smoke` through the real orchestrator plumbing.

    The done-marker is derived from ``SmokeConfig.results_name`` rather than hardcoded, so
    it names the file `pop smoke` actually writes. Hardcoding it drifted once already: the
    marker said ``results/smoke.json``, which is a *committed* result the smoke command
    deliberately never touches (the clobber guard moved local runs to ``smoke_local``). A
    marker that is always present makes the post-step existence check in ``main`` vacuous,
    so a smoke run that exited 0 without writing anything still reported a successful dry
    run -- the exact plumbing failure --dry-run exists to catch before any GPU time.
    """
    from pop.config import SmokeConfig

    return [
        Step(
            "smoke",
            _pop("smoke"),
            RESULTS_DIR / f"{SmokeConfig().results_name}.json",
            always_run=True,
        )
    ]


def preflight() -> None:
    """Fail fast (before any GPU time) if the repo/config wiring is off."""
    if not (REPO_ROOT / "pyproject.toml").is_file():
        raise SystemExit(f"run_training: repo root not found at {REPO_ROOT}")

    from pop.config import FinetuneConfig, PretrainConfig

    expected_tokenizer = Path("outputs/tokenizer/tokenizer.model")
    problems: list[str] = []

    pre = PretrainConfig.from_yaml(REPO_ROOT / "configs" / "pretrain_10ep.yaml")
    if Path(pre.tokenizer_path) != expected_tokenizer:
        problems.append(f"pretrain: tokenizer_path {pre.tokenizer_path} != {expected_tokenizer}")
    if (REPO_ROOT / pre.output_dir / "final") != PRETRAIN_FINAL:
        problems.append(f"pretrain: output_dir {pre.output_dir} inconsistent with {PRETRAIN_FINAL}")

    for system in SYSTEMS:
        cfg = FinetuneConfig.from_yaml(REPO_ROOT / "configs" / f"finetune_{system}.yaml")
        if Path(cfg.tokenizer_path) != expected_tokenizer:
            problems.append(f"{system}: tokenizer_path {cfg.tokenizer_path}")
        if Path(cfg.output_dir) != Path("outputs") / f"finetune_{system}":
            problems.append(f"{system}: output_dir {cfg.output_dir}")
        pretrained = cfg.pretrained_model_path
        if system.startswith("A"):
            if pretrained is None or (REPO_ROOT / pretrained) != PRETRAIN_FINAL:
                problems.append(f"{system}: pretrained_model_path {pretrained} != {PRETRAIN_FINAL}")
        elif pretrained is not None:
            problems.append(f"{system}: unexpectedly sets pretrained_model_path {pretrained}")

    if problems:
        raise SystemExit("run_training preflight failed:\n  " + "\n  ".join(problems))


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def write_status(run_dir: Path, rows: list[dict]) -> None:
    lines = [
        "# Phase-2 training status",
        "",
        f"_Updated {_now()}. Completed steps are skipped on relaunch; an interrupted",
        "`running` step resumes from its latest checkpoint on the next launch._",
        "",
        "| step | status | started | duration | result |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['status']} | {row.get('started', '-')} "
            f"| {row.get('duration', '-')} | {row.get('result', '-')} |"
        )
    tmp = run_dir / "STATUS.md.tmp"
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(run_dir / "STATUS.md")


def read_em(results_path: Path) -> float:
    return float(json.loads(results_path.read_text(encoding="utf-8"))["metrics"]["em"])


def write_summary(run_dir: Path, results_dir: Path = RESULTS_DIR) -> None:
    lines = [
        "# Phase-2 results summary",
        "",
        "| system | em | em_raw | codebleu | syntax_valid_rate | n |",
        "|---|---|---|---|---|---|",
    ]
    found = False
    for system in SYSTEMS:
        path = results_dir / f"finetune_{system}_test.json"
        if not path.is_file():
            continue
        found = True
        m = json.loads(path.read_text(encoding="utf-8"))["metrics"]
        lines.append(
            f"| {system} | {m['em']:.4f} | {m['em_raw']:.4f} | {m['codebleu']:.4f} "
            f"| {m['syntax_valid_rate']:.4f} | {m['n']} |"
        )
    if found:
        (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_step(step: Step, run_dir: Path, env: dict[str, str]) -> int:
    log_path = run_dir / f"{step.name}.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {step.name} @ {_now()} =====\n$ {' '.join(step.cmd)}\n")
        log.flush()
        proc = subprocess.Popen(
            step.cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        # Not an `assert`: this orchestrator drives multi-hour GPU runs, and under `python -O`
        # a stripped assert would leave the loop below iterating None.
        if proc.stdout is None:  # pragma: no cover -- stdout=PIPE guarantees a stream
            raise RuntimeError(f"{step.name}: subprocess stdout was not captured")
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return proc.wait()


def prune_checkpoints(output_dir: Path, olog) -> None:
    for ckpt in sorted(output_dir.glob("checkpoint-*")):
        shutil.rmtree(ckpt, ignore_errors=True)
        olog(f"pruned {ckpt.relative_to(REPO_ROOT)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run `pop smoke` through the orchestrator plumbing instead of the real steps",
    )
    parser.add_argument(
        "--limit-eval",
        type=int,
        default=None,
        help="debug: cap generate/eval to N samples (done-markers then reflect the capped run)",
    )
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="bypass the EM>0 halt (only after diagnosing a 0-EM result per "
        "docs/gpu-reproduction.md)",
    )
    args = parser.parse_args(argv)

    # Windows consoles are often cp1252; subprocess output can contain Unicode
    # (tqdm bars, tokenizer logs). Degrade unencodable chars to '?' on the console
    # -- the per-step .log files are written as UTF-8 and stay lossless.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    preflight()

    run_dir = LOGS_ROOT / ("dry" if args.dry_run else "phase2")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        steps = plan_dry_run_steps()
    else:
        steps = plan_steps(args.limit_eval)

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
        else:
            row.update(status="running", started=_now())
            write_status(run_dir, rows)
            olog(f"start {step.name}")
            t0 = time.monotonic()
            rc = run_step(step, run_dir, env)
            duration = _fmt_duration(time.monotonic() - t0)
            if rc != 0:
                row.update(
                    status="FAILED",
                    duration=duration,
                    result=f"exit {rc}; see {step.name}.log",
                )
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

        if step.prune_checkpoints_in is not None:
            prune_checkpoints(step.prune_checkpoints_in, olog)

        # EM>0 gate: applies to the first system's eval whether it ran now or earlier.
        if step.name == f"eval_{GATE_SYSTEM}" and not args.dry_run:
            em = read_em(RESULTS_DIR / f"finetune_{GATE_SYSTEM}_test.json")
            if em == 0.0 and not args.skip_gate:
                row["result"] = "em=0.0 -> HALT"
                write_status(run_dir, rows)
                olog(
                    f"EM GATE: finetune_{GATE_SYSTEM} scored exact-match 0.0. Halting instead of "
                    "training the remaining systems: all-zero EM at this scale means a decoding "
                    "bug (docs/gpu-reproduction.md). Next: eyeball a few lines of "
                    f"outputs/finetune_{GATE_SYSTEM}/best/predictions_test.jsonl, re-generate with "
                    "`pop generate --num-beams 5` on the same checkpoint, and only re-run this "
                    "script with --skip-gate once the cause is understood."
                )
                return 3
            verdict = "(gate bypassed)" if args.skip_gate and em == 0.0 else "-- continuing"
            olog(f"EM gate: {GATE_SYSTEM} em={em:.4f} {verdict}")

    if args.dry_run:
        olog("dry run complete")
    else:
        write_summary(run_dir)
        olog("all steps complete; SUMMARY.md written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
