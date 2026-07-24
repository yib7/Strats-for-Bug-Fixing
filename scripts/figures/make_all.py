"""Render every analysis figure into ``docs/figures/`` (deterministic, headless).

Drives the three figure scripts against the committed real ``results/*.json``:

* ``four_arm_comparison.py``   -> ``docs/figures/four_arm_comparison.png``
* ``scaling_curves.py``        -> ``docs/figures/scaling_curves.png``
* ``execution_vs_codebleu.py`` -> ``docs/figures/execution_vs_codebleu.png``

Before rendering, the two derived CSVs the scaling / execution figures read
(``results/scaling_data.csv`` and ``results/execbench_agreement.csv``) are
**rebuilt from the committed result JSONs** by the ``build_*_csv`` scripts. That
keeps the JSONs the single source of truth and guarantees the figures reproduce
from a clean checkout instead of silently falling back to the fixtures.

Every script is deterministic (Agg backend, point estimates + committed data),
so re-running this is safe and the outputs are committed to the repo.

Run:  ``python scripts/figures/make_all.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # sibling build_*_csv scripts

import build_execbench_agreement_csv  # noqa: E402
import build_scaling_csv  # noqa: E402
import execution_vs_codebleu  # noqa: E402
import four_arm_comparison  # noqa: E402
import scaling_curves  # noqa: E402
from _common import FIGURES_DIR, RESULTS_DIR  # noqa: E402


def rebuild_derived_csvs(csv_out_dir: Path = RESULTS_DIR) -> list[Path]:
    """Regenerate the derived analysis CSVs from the committed result JSONs.

    Idempotent: reads only committed ``results/*.json`` and overwrites the CSVs
    with byte-identical content on every run. Arms without a result yet (e.g. arm
    B's optional execution point) are simply omitted -- the builders degrade
    gracefully, so this never fabricates rows.

    ``csv_out_dir`` defaults to the real ``results/``; tests redirect it to a tmp dir so
    running the suite never writes into tracked files.
    """
    csv_out_dir = Path(csv_out_dir)
    csv_out_dir.mkdir(parents=True, exist_ok=True)
    return [
        build_scaling_csv.write_csv(
            build_scaling_csv.build_rows(), csv_out_dir / "scaling_data.csv"
        ),
        build_execbench_agreement_csv.write_csv(
            build_execbench_agreement_csv.build_rows(),
            csv_out_dir / "execbench_agreement.csv",
        ),
    ]


def make_all(out_dir: Path = FIGURES_DIR, csv_out_dir: Path = RESULTS_DIR) -> list[Path]:
    """Rebuild the derived CSVs, then render all figures; returns the PNG paths."""
    rebuild_derived_csvs(csv_out_dir)
    return [
        four_arm_comparison.make(out_dir=out_dir),
        scaling_curves.make(out_dir=out_dir),
        execution_vs_codebleu.make(out_dir=out_dir),
    ]


if __name__ == "__main__":
    paths = make_all()
    for p in paths:
        print(f"wrote {p}")
