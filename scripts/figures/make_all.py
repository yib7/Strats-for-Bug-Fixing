"""Render every analysis figure into ``docs/figures/`` (deterministic, headless).

Drives the three figure scripts against the committed real results (arms A & B)
plus the committed fixtures (scaling / execution, pending the GPU batch):

* ``four_arm_comparison.py``   -> ``docs/figures/four_arm_comparison.png``
* ``scaling_curves.py``        -> ``docs/figures/scaling_curves.png``
* ``execution_vs_codebleu.py`` -> ``docs/figures/execution_vs_codebleu.png``

Every script is deterministic (Agg backend, point estimates + committed data),
so re-running this is safe and the outputs are committed to the repo.

Run:  ``python scripts/figures/make_all.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import execution_vs_codebleu  # noqa: E402
import four_arm_comparison  # noqa: E402
import scaling_curves  # noqa: E402
from _common import FIGURES_DIR  # noqa: E402


def make_all(out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Render all figures; returns the list of written PNG paths."""
    return [
        four_arm_comparison.make(out_dir=out_dir),
        scaling_curves.make(out_dir=out_dir),
        execution_vs_codebleu.make(out_dir=out_dir),
    ]


if __name__ == "__main__":
    paths = make_all()
    for p in paths:
        print(f"wrote {p}")
