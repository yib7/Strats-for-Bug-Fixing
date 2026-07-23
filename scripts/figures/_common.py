"""Shared helpers for the analysis figure scripts.

Importing this module fixes matplotlib to the headless ``Agg`` backend *before*
``pyplot`` is imported, so every figure script renders deterministically with no
display and no per-machine backend surprises (this is what makes the figure
smoke tests safe to run in CI). It also centralises the repo paths, the arm
colour palette, and the ``results/*.json`` loader so the individual scripts stay
small.

The four experimental arms of the study (see ``docs/report.md``):

* **A** - pretrain -> finetune T5-small (epochs 1/3/10).
* **B** - finetune-from-scratch T5-small (seeds 0/1/2).
* **C** - RAG / prompt Qwen2.5-Coder-1.5B (best-CodeBLEU sweep config).
* **D** - LoRA-finetune Qwen2.5-Coder-1.5B.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # deterministic + headless; MUST precede the pyplot import

import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

# Colour-blind-safe qualitative palette (Okabe-Ito subset), one per arm. Pending
# arms render in a neutral grey with a hatch so a filled-in figure and a
# placeholder figure are visually distinguishable at a glance.
ARM_COLORS = {
    "A": "#0072b2",  # blue   - pretrain -> finetune T5
    "B": "#e69f00",  # orange - from-scratch T5
    "C": "#009e73",  # green  - RAG-Qwen
    "D": "#cc79a7",  # pink   - LoRA-Qwen
}
PENDING_COLOR = "#c8ccd4"
PENDING_HATCH = "//"

ARM_LABELS = {
    "A": "A: pretrain→finetune T5",
    "B": "B: from-scratch T5",
    "C": "C: RAG Qwen",
    "D": "D: LoRA Qwen",
}


def _apply_rc() -> None:
    """Consistent, restrained styling shared by every figure."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 120,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
        }
    )


def load_result(path: Path) -> dict:
    """Load one ``results/<name>.json`` file (schema documented in metrics.py)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_finetune_results(results_dir: Path = RESULTS_DIR) -> dict[str, dict]:
    """Map ``<name>`` -> metrics dict for every committed ``finetune_*_test.json``.

    Returns just the ``metrics`` sub-dict per run (``em``, ``em_raw``,
    ``codebleu``, ``syntax_valid_rate``, ``n``). Missing directory -> empty map,
    so callers never crash when a results set has not been produced yet.
    """
    results_dir = Path(results_dir)
    out: dict[str, dict] = {}
    if not results_dir.is_dir():
        return out
    for path in sorted(results_dir.glob("finetune_*_test.json")):
        name = path.stem.removesuffix("_test")
        out[name] = load_result(path).get("metrics", {})
    return out


def save_fig(fig, name: str, out_dir: Path = FIGURES_DIR) -> Path:
    """Write ``<out_dir>/<name>.png`` (creating the dir) and return the path.

    ``metadata`` is pinned so re-rendering an unchanged figure produces a stable
    file (no embedded timestamp) and does not churn the committed PNG in git.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", metadata={"Software": "pop.figures"})
    plt.close(fig)
    return path
