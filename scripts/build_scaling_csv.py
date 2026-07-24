"""Aggregate the scaling-sweep eval JSONs into ``results/scaling_data.csv``.

This is the CSV-builder half of the Cycle-7 analysis: it turns the per-config
``results/finetune_scale_*_test.json`` / ``results/finetune_ptcompute_*_test.json``
files produced by ``scripts/run_scaling.py`` (plus the reused cycle-2 full-data
runs) into the single tidy CSV that ``scripts/figures/scaling_curves.py`` renders.

Output schema (EXACTLY what ``scaling_curves.load_scaling_rows`` reads)::

    arm,axis,x,codebleu,syntax,seed

* ``arm``   - ``A`` (pretrain->finetune) or ``B`` (from-scratch).
* ``axis``  - ``data`` (x = finetune ``train_n``) or ``ptcompute`` (x = pretrain epochs).
* ``x``     - the sweep coordinate.
* ``codebleu`` / ``syntax`` - ``metrics.codebleu`` / ``metrics.syntax_valid_rate`` of that run.
* ``seed``  - the run's seed (multiple rows per ``x`` -> the figure's shaded band).

Rows come from three sources, and **only rows whose result JSON exists are
emitted** (graceful partial -- run this after any subset of the GPU batch and it
writes the curve so far):

1. **Data-axis sweep points** -- ``finetune_scale_{A,B}_n{1k,5k,15k}_seed{0,1}``
   (x = ``train_n`` 1000/5000/15000), from ``gen_scaling_configs.iter_config_specs``
   (the single source of truth shared with the generator/orchestrator).
2. **Ptcompute-axis sweep points** -- ``finetune_ptcompute_ep{1,3}_seed42``
   (x = pretrain epochs 1/3).
3. **Reference points reused from the cycle-2 full-data runs** (NOT re-run):
   the 52K data point (arm A = ``finetune_A_ep10``; arm B = ``finetune_B_seed{0,1}``)
   and the ep10 ptcompute point (arm A = ``finetune_A_ep10``, since pretrain-final
   == epoch-10). These are a *different* pretrain instance than the ptcompute
   sweep's fresh pretrain -- a documented minor inconsistency (see
   ``docs/gpu-runbook.md`` Step 0).

Usage:
    python scripts/build_scaling_csv.py            # write results/scaling_data.csv
    python scripts/build_scaling_csv.py --out X    # write elsewhere
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling gen_scaling_configs

from gen_scaling_configs import iter_config_specs  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = RESULTS_DIR / "scaling_data.csv"

CSV_FIELDS = ["arm", "axis", "x", "codebleu", "syntax", "seed"]

# Full CodeXGLUE-medium refinement train split -- the "52K" top data-curve point
# (matches tests/fixtures/scaling_data_example.csv and the frozen roadmap).
FULL_TRAIN_N = 52364
# pretrain-final == the 10th pretrain epoch, so finetune_A_ep10 IS the ep10 ptcompute point.
PTCOMPUTE_FINAL_EPOCHS = 10

# Reference runs reused as the curves' top points (committed cycle-2 results).
A_REF = "finetune_A_ep10"  # arm A full-data (seed 42)
B_REFS = ("finetune_B_seed0", "finetune_B_seed1")  # arm B full-data (seeds 0/1, matching the sweep)
A_REF_SEED = 42


def _load_metrics(name: str, results_dir: Path) -> dict | None:
    """Return the ``metrics`` dict of ``results/<name>.json``, or None if it does not exist."""
    path = results_dir / f"{name}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("metrics", {})


def _row(arm: str, axis: str, x: float, metrics: dict, seed: int) -> dict:
    return {
        "arm": arm,
        "axis": axis,
        "x": x,
        "codebleu": metrics["codebleu"],
        "syntax": metrics["syntax_valid_rate"],
        "seed": seed,
    }


def build_rows(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Assemble the CSV rows from whatever result JSONs currently exist (graceful partial)."""
    results_dir = Path(results_dir)
    rows: list[dict] = []

    # 1 + 2. Sweep points from the generated scaling configs.
    for spec in iter_config_specs():
        metrics = _load_metrics(f"{spec.stem}_test", results_dir)
        if metrics is None:
            continue
        if spec.kind == "data":
            arm = re.search(r"scale_([AB])_", spec.stem).group(1)
            rows.append(_row(arm, "data", spec.train_n, metrics, spec.seed))
        else:  # ptcompute (arm A only)
            epochs = int(re.search(r"ep(\d+)_", spec.stem).group(1))
            rows.append(_row("A", "ptcompute", epochs, metrics, spec.seed))

    # 3. Reference points reused from the cycle-2 full-data runs. `pop eval` writes each run as
    #    results/<config>_test.json, so the reference lookups carry the same `_test` suffix.
    a_ref = _load_metrics(f"{A_REF}_test", results_dir)
    if a_ref is not None:
        rows.append(_row("A", "data", FULL_TRAIN_N, a_ref, A_REF_SEED))
    for name in B_REFS:
        metrics = _load_metrics(f"{name}_test", results_dir)
        if metrics is not None:
            seed = int(re.search(r"seed(\d+)", name).group(1))
            rows.append(_row("B", "data", FULL_TRAIN_N, metrics, seed))
    if a_ref is not None:
        rows.append(_row("A", "ptcompute", PTCOMPUTE_FINAL_EPOCHS, a_ref, A_REF_SEED))

    return rows


def write_csv(rows: list[dict], out_path: Path = OUT_PATH) -> Path:
    """Write ``rows`` to ``out_path`` in the scaling-curve schema; returns the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help="directory holding the finetune_*_test.json result files (default: results/)",
    )
    parser.add_argument(
        "--out",
        default=str(OUT_PATH),
        help="output CSV path (default: results/scaling_data.csv)",
    )
    args = parser.parse_args(argv)

    rows = build_rows(Path(args.results_dir))
    path = write_csv(rows, Path(args.out))
    n_data = sum(r["axis"] == "data" for r in rows)
    n_pt = sum(r["axis"] == "ptcompute" for r in rows)
    print(f"wrote {path} ({len(rows)} rows: {n_data} data-axis, {n_pt} ptcompute-axis)")
    if not rows:
        print("  (no scaling result JSONs found yet -- run the scaling sweep first)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
