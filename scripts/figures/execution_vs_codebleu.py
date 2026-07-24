"""Execution-vs-CodeBLEU agreement scatter (pass@1 on execbench vs CodeBLEU).

Asks whether the "looks right" metric (CodeBLEU) and the "is right" metric
(execution pass@1 over the 201 vendored QuixBugs + HumanEval-Java bugs) rank the
arms the same way. Each arm is one point: x = CodeBLEU, y = pass@1.

CSV schema (see ``tests/fixtures/execbench_agreement_example.csv``)::

    arm,codebleu,pass_at_1,n_bugs

``results/execbench_agreement.csv`` is committed, and
``scripts/figures/make_all.py`` rebuilds it from the committed ``results/*.json``
before every render -- so the committed figure is real measured data. The fixture
(``tests/fixtures/execbench_agreement_example.csv``) survives only as the fallback
for a results directory with no agreement data; the *illustrative fixture data*
title appears in that case and only in that case. A missing file degrades to a
clearly-labelled empty placeholder rather than crashing.

Run standalone:  ``python scripts/figures/execution_vs_codebleu.py``
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    ARM_COLORS,
    ARM_LABELS,
    FIGURES_DIR,
    FIXTURES_DIR,
    RESULTS_DIR,
    _apply_rc,
    plt,
    save_fig,
)


def resolve_data_path(data_path: Path | None) -> tuple[Path, bool]:
    """Pick the CSV to render and whether it is the illustrative fixture."""
    if data_path is not None:
        p = Path(data_path)
        return p, p == FIXTURES_DIR / "execbench_agreement_example.csv"
    real = RESULTS_DIR / "execbench_agreement.csv"
    if real.exists():
        return real, False
    return FIXTURES_DIR / "execbench_agreement_example.csv", True


def load_agreement_rows(path: Path) -> list[dict]:
    """Parse the agreement CSV into typed rows; missing file -> empty list."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(
                {
                    "arm": r["arm"].strip(),
                    "codebleu": float(r["codebleu"]),
                    "pass_at_1": float(r["pass_at_1"]),
                    "n_bugs": int(r["n_bugs"]) if r.get("n_bugs") else None,
                }
            )
    return rows


def make(data_path: Path | None = None, out_dir: Path = FIGURES_DIR) -> Path:
    """Render the execution-vs-CodeBLEU scatter; returns the written PNG path."""
    path, is_fixture = resolve_data_path(data_path)
    rows = load_agreement_rows(path)

    _apply_rc()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    if not rows:
        ax.text(
            0.5,
            0.5,
            "⟨pending GPU batch⟩\nno execution predictions yet",
            ha="center",
            va="center",
            fontsize=11,
            style="italic",
            color="#6b7280",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.suptitle("Execution vs CodeBLEU", fontsize=12)
        return save_fig(fig, "execution_vs_codebleu", out_dir)

    for r in rows:
        arm = r["arm"]
        ax.scatter(
            r["codebleu"],
            r["pass_at_1"],
            s=140,
            color=ARM_COLORS.get(arm, "#666666"),
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.annotate(
            ARM_LABELS.get(arm, arm),
            (r["codebleu"], r["pass_at_1"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8.5,
        )

    ax.set_xlabel("CodeBLEU (test)")
    ax.set_ylabel("Execution pass@1 (QuixBugs + HumanEval-Java, n≈201)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.5, max(r["pass_at_1"] for r in rows) * 1.25))
    suffix = "  (illustrative fixture data)" if is_fixture else ""
    fig.suptitle(f"Do CodeBLEU and execution agree?{suffix}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_fig(fig, "execution_vs_codebleu", out_dir)


if __name__ == "__main__":
    print(make())
