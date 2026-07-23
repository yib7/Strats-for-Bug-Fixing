"""Scaling-sweep orchestrator: per scaling config, `pop finetune` -> `generate` -> `eval`.

Runs both scaling curves -- the data-scaling curve (arms A/B x train_n {1k,5k,15k} x seed {0,1})
and the pretrain-compute curve (arm A x pretrain-epochs {1,3}) -- as one command, resumable, the
scaling analogue of `scripts/run_training.py`. Every step is a `python -m pop.cli ...` subprocess,
in order, with:

- **Resumability.** Every step has a durable done-marker (its output artifact): a `finetune`
  step's marker is ``<output_dir>/best/model.safetensors``, its `generate` step's is
  ``<output_dir>/best/predictions_test.jsonl``, its `eval` step's is
  ``results/<stem>_test.json``. Steps whose marker already exists are skipped, so re-running
  after a crash / reboot / Colab session limit continues where the run left off. *Mid-step*
  resume is handled inside the T5 finetuner (it reloads the latest ``checkpoint-*``).
- **File-based progress logging** watchable from outside the process: a live ``STATUS.md``
  table, an ``orchestrator.log``, and one ``<step>.log`` per step (also echoed to stdout so a
  Colab cell shows it live). On Colab, ``logs/`` is a symlink into Google Drive (the
  ``notebooks/colab_scaling.ipynb`` setup cell makes it one), so ``STATUS.md`` is readable from
  the Drive web/phone app during the run.
- **Disk pruning:** once a config's eval lands, its intermediate ``checkpoint-*`` dirs are
  deleted (``best/`` is kept), bounding a Google Drive workspace to a few GB.

This mirrors `scripts/run_training.py` / `scripts/run_rag.py` and **reuses** the shared
Step/plan/resume/status/log/prune machinery (imported, not copy-pasted). Unlike the headline T5
run there is no EM>0 gate (that was a first-system diagnostic; the scaling sweep is pure curve
fill-in over configs the A-vs-B answer already validated). The config set comes from
``scripts/gen_scaling_configs.iter_config_specs`` -- the single source of truth shared with the
generator, so the plan can never drift from the committed configs.

The **52K (full-data)** data-curve point and the **ep10** pretrain-compute point are NOT run
here: they reuse the existing ``finetune_A_ep10`` / ``finetune_B_seed{0,1}`` results
(``scripts/run_training.py``).

Usage:
    python scripts/run_scaling.py           # both scaling sweeps (needs a GPU)
    python scripts/run_scaling.py --list     # print the plan + run preflight, CPU-only, run nothing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_scaling_configs import iter_config_specs  # noqa: E402  (sibling script; path above)
from run_training import (  # noqa: E402  (sibling script; path inserted above)
    REPO_ROOT,
    RESULTS_DIR,
    TOKENIZER_MODEL,
    Step,
    _fmt_duration,
    _now,
    _pop,
    prune_checkpoints,
    run_step,
    write_status,
)

LOGS_ROOT = REPO_ROOT / "logs" / "scaling"
CONFIGS_DIR = REPO_ROOT / "configs"


def plan_steps(configs_dir: Path = CONFIGS_DIR) -> list[Step]:
    """Per scaling config: a `finetune` -> `generate` -> `eval` triple (mirrors run_training)."""
    from pop.config import FinetuneConfig

    steps: list[Step] = []
    for spec in iter_config_specs():
        cfg = FinetuneConfig.from_yaml(configs_dir / f"{spec.stem}.yaml")
        ft_dir = REPO_ROOT / cfg.output_dir
        best = ft_dir / "best"
        predictions = best / "predictions_test.jsonl"
        results_name = f"{spec.stem}_test"
        curve_id = spec.stem.removeprefix("finetune_")
        steps += [
            Step(
                f"finetune_{curve_id}",
                _pop("finetune", "--config", f"configs/{spec.stem}.yaml"),
                best / "model.safetensors",
            ),
            Step(
                f"generate_{curve_id}",
                _pop("generate", "--model", str(best), "--tokenizer", str(TOKENIZER_MODEL)),
                predictions,
            ),
            Step(
                f"eval_{curve_id}",
                _pop("eval", "--predictions", str(predictions), "--name", results_name),
                RESULTS_DIR / f"{results_name}.json",
                prune_checkpoints_in=ft_dir,
            ),
        ]
    return steps


def preflight(configs_dir: Path = CONFIGS_DIR) -> None:
    """Fail fast (before any GPU second) if the scaling-config wiring is off.

    Asserts every scaling config parses, uses the shared tokenizer, writes to a distinct
    ``output_dir`` consistent with its filename, and carries the pretrained-path / train_n
    that its curve requires (arm A -> ``outputs/pretrain/final``; arm B -> none; ptcompute
    -> ``outputs/pretrain/epoch-{N}``; data configs set ``train_n``, ptcompute leaves it
    unset).
    """
    if not (REPO_ROOT / "pyproject.toml").is_file():
        raise SystemExit(f"run_scaling: repo root not found at {REPO_ROOT}")

    from pop.config import FinetuneConfig

    expected_tokenizer = Path("outputs/tokenizer/tokenizer.model")
    problems: list[str] = []
    seen_output_dirs: dict[Path, str] = {}

    for spec in iter_config_specs():
        path = configs_dir / f"{spec.stem}.yaml"
        if not path.is_file():
            problems.append(f"{spec.stem}: config not found at {path}")
            continue
        try:
            cfg = FinetuneConfig.from_yaml(path)
        except Exception as exc:  # noqa: BLE001 -- surface any parse/validation error
            problems.append(f"{spec.stem}: failed to parse ({exc})")
            continue

        if Path(cfg.tokenizer_path) != expected_tokenizer:
            problems.append(
                f"{spec.stem}: tokenizer_path {cfg.tokenizer_path} != {expected_tokenizer}"
            )
        if Path(cfg.output_dir) != Path(spec.output_dir):
            problems.append(f"{spec.stem}: output_dir {cfg.output_dir} != {spec.output_dir}")

        expected_pretrained = (
            None if spec.pretrained_model_path is None else Path(spec.pretrained_model_path)
        )
        actual_pretrained = (
            None if cfg.pretrained_model_path is None else Path(cfg.pretrained_model_path)
        )
        if actual_pretrained != expected_pretrained:
            problems.append(
                f"{spec.stem}: pretrained_model_path {cfg.pretrained_model_path} != "
                f"{spec.pretrained_model_path}"
            )
        if cfg.train_n != spec.train_n:
            problems.append(f"{spec.stem}: train_n {cfg.train_n} != {spec.train_n}")

        out = Path(cfg.output_dir)
        if out in seen_output_dirs:
            problems.append(f"{spec.stem}: output_dir {out} collides with {seen_output_dirs[out]}")
        else:
            seen_output_dirs[out] = spec.stem

    if problems:
        raise SystemExit("run_scaling preflight failed:\n  " + "\n  ".join(problems))


def write_summary(run_dir: Path, results_dir: Path = RESULTS_DIR) -> None:
    """Render a config x metrics table to SUMMARY.md from whatever results exist yet.

    Mirrors ``run_rag.write_summary``: writes nothing if no ``results/*_test.json`` for the
    scaling configs exist. The reused ``finetune_A_ep10`` / ``finetune_B_seed{0,1}`` (52K) and
    ep10 points live in their own results files and are joined in later (Cycle-7 analysis).
    """
    lines = [
        "# Scaling sweep results summary",
        "",
        "| config | em | codebleu | syntax_valid_rate | n |",
        "|---|---|---|---|---|",
    ]
    found = False
    for spec in iter_config_specs():
        path = results_dir / f"{spec.stem}_test.json"
        if not path.is_file():
            continue
        found = True
        m = json.loads(path.read_text(encoding="utf-8"))["metrics"]
        lines.append(
            f"| {spec.stem} | {m['em']:.4f} | {m['codebleu']:.4f} "
            f"| {m['syntax_valid_rate']:.4f} | {m['n']} |"
        )
    if found:
        (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_plan(steps: list[Step]) -> None:
    print(f"Scaling sweep plan: {len(steps)} steps ({len(steps) // 3} configs)")
    for i, step in enumerate(steps, 1):
        status = "done" if step.is_done() else "pending"
        print(
            f"  {i:2d}. {step.name:28s} [{status:>7}] -> {step.done_marker.relative_to(REPO_ROOT)}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="print the plan and run preflight WITHOUT executing any step (CPU-only proof; "
        "a real --dry-run is impossible here since even one step needs a GPU finetune)",
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
        else:
            row.update(status="running", started=_now())
            write_status(run_dir, rows)
            olog(f"start {step.name}")
            t0 = time.monotonic()
            rc = run_step(step, run_dir, env)
            duration = _fmt_duration(time.monotonic() - t0)
            if rc != 0:
                row.update(
                    status="FAILED", duration=duration, result=f"exit {rc}; see {step.name}.log"
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
        # Refresh the results table after every eval so a mid-sweep disconnect still leaves a
        # partial SUMMARY.md on Drive.
        if step.name.startswith("eval_"):
            write_summary(run_dir)

    write_summary(run_dir)
    olog("all steps complete; SUMMARY.md written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
