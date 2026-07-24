"""Scaling-curve figures from a tidy ``results/scaling_data.csv``.

CSV schema (one row per measured point; see
``tests/fixtures/scaling_data_example.csv`` for a complete example)::

    arm,axis,x,codebleu,syntax,seed

* ``arm``    - ``A`` (pretrain->finetune) or ``B`` (from-scratch).
* ``axis``   - ``data`` (data-scaling; ``x`` = finetune ``train_n``) or
               ``ptcompute`` (pretrain-compute; ``x`` = pretrain epochs).
* ``x``      - the sweep coordinate (numeric).
* ``codebleu`` / ``syntax`` - the two headline metrics at that point.
* ``seed``   - the seed for this measurement (multiple rows per ``x`` -> band).

One panel is drawn per ``axis`` present. Within a panel, each arm is one line
(CodeBLEU vs ``x``, mean over seeds) with the seed min..max drawn as a shaded
band.

``results/scaling_data.csv`` is committed, and ``scripts/figures/make_all.py``
rebuilds it from the committed ``results/*.json`` before every render -- so the
committed figure is real measured data. The fixture
(``tests/fixtures/scaling_data_example.csv``) survives only as the fallback for a
results directory with no scaling data at all; the *illustrative fixture data*
title appears in that case and only in that case.

Run standalone:  ``python scripts/figures/scaling_curves.py``
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
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

AXIS_TITLES = {
    "data": "Data-scaling curve",
    "ptcompute": "Pretraining-compute curve",
}
AXIS_XLABELS = {
    "data": "finetune train_n (pairs, log scale)",
    "ptcompute": "pretrain epochs",
}


def resolve_data_path(data_path: Path | None) -> tuple[Path, bool]:
    """Pick the CSV to render and whether it is the illustrative fixture.

    Prefers an explicit ``data_path``; else the real ``results/scaling_data.csv``
    if it exists; else the committed fixture. Returns ``(path, is_fixture)``.
    """
    if data_path is not None:
        p = Path(data_path)
        return p, p == FIXTURES_DIR / "scaling_data_example.csv"
    real = RESULTS_DIR / "scaling_data.csv"
    if real.exists():
        return real, False
    return FIXTURES_DIR / "scaling_data_example.csv", True


def load_scaling_rows(path: Path) -> list[dict]:
    """Parse the scaling CSV into typed rows; unknown/missing file -> empty list."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(
                {
                    "arm": r["arm"].strip(),
                    "axis": r["axis"].strip(),
                    "x": float(r["x"]),
                    "codebleu": float(r["codebleu"]),
                    "syntax": float(r["syntax"]),
                    "seed": int(r["seed"]),
                }
            )
    return rows


def _aggregate(rows: list[dict], axis: str, arm: str) -> tuple[list, list, list, list]:
    """Return (xs, means, los, his) for CodeBLEU at each x, averaged over seeds."""
    by_x: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        if r["axis"] == axis and r["arm"] == arm:
            by_x[r["x"]].append(r["codebleu"])
    xs = sorted(by_x)
    means = [sum(by_x[x]) / len(by_x[x]) for x in xs]
    los = [min(by_x[x]) for x in xs]
    his = [max(by_x[x]) for x in xs]
    return xs, means, los, his


def _draw_axis_panel(ax, rows: list[dict], axis: str) -> None:
    _apply_rc()
    arms = [a for a in ("A", "B") if any(r["axis"] == axis and r["arm"] == a for r in rows)]
    for arm in arms:
        xs, means, los, his = _aggregate(rows, axis, arm)
        if not xs:
            continue
        ax.plot(xs, means, marker="o", color=ARM_COLORS[arm], label=ARM_LABELS[arm], zorder=3)
        if any(lo != hi for lo, hi in zip(los, his, strict=True)):
            ax.fill_between(xs, los, his, color=ARM_COLORS[arm], alpha=0.18, zorder=1)
    if axis == "data":
        ax.set_xscale("log")
    ax.set_xlabel(AXIS_XLABELS.get(axis, axis))
    ax.set_ylabel("CodeBLEU")
    ax.set_title(AXIS_TITLES.get(axis, axis))
    ax.legend(fontsize=8, frameon=False)


def _draw_empty(ax) -> None:
    ax.text(
        0.5,
        0.5,
        "⟨pending GPU batch⟩\nno results/scaling_data.csv yet",
        ha="center",
        va="center",
        fontsize=11,
        style="italic",
        color="#6b7280",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def make(data_path: Path | None = None, out_dir: Path = FIGURES_DIR) -> Path:
    """Render the scaling curves; returns the written PNG path.

    ``data_path=None`` auto-resolves (real CSV if present, else fixture).
    """
    path, is_fixture = resolve_data_path(data_path)
    rows = load_scaling_rows(path)
    axes_present = [a for a in ("data", "ptcompute") if any(r["axis"] == a for r in rows)]

    if not axes_present:
        fig, ax = plt.subplots(figsize=(7, 5))
        _draw_empty(ax)
        fig.suptitle("Scaling curves", fontsize=12)
        return save_fig(fig, "scaling_curves", out_dir)

    fig, axes = plt.subplots(
        1, len(axes_present), figsize=(5.5 * len(axes_present), 5), squeeze=False
    )
    for ax, axis in zip(axes[0], axes_present, strict=True):
        _draw_axis_panel(ax, rows, axis)

    suffix = "  (illustrative fixture data)" if is_fixture else ""
    fig.suptitle(f"Scaling curves — CodeBLEU vs data / pretrain compute{suffix}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_fig(fig, "scaling_curves", out_dir)


if __name__ == "__main__":
    print(make())
