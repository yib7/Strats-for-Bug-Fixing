"""Four-arm comparison figure: CodeBLEU + syntax-valid rate across arms A/B/C/D.

Reads every committed ``results/finetune_*_test.json`` (the real arm-A epoch and
arm-B seed runs from the A100 GPU batch) and renders a two-panel bar chart:

* **Arm A** (pretrain->finetune T5) - the final ``A_ep10`` bar, with the epoch
  1/3/10 points overlaid as a connected trend so the finetune-epoch progression
  is visible inside the bar.
* **Arm B** (from-scratch T5) - the mean across seeds 0/1/2 as the bar height,
  with the seed-to-seed **band** (min..max) drawn as an error bar.
* **Arm C** (RAG-Qwen) - the best-CodeBLEU config across the ``results/rag_*_test.json``
  sweep (``rag_codebert_k1``).
* **Arm D** (LoRA-Qwen) - ``results/lora_qwen_test.json``.

If an arm's result JSON is absent the script degrades gracefully, drawing that arm
as a clearly-marked *pending* placeholder (hollow hatched slot) rather than crashing,
so the figure is well-shaped in either state.

Bootstrap CIs (``pop.eval.bootstrap``) are *optional* and only used when a
per-sample predictions jsonl is supplied via ``--ci-predictions ARM=path`` (none
are committed, so the default figure uses point estimates + the arm-B seed band).

Run standalone:  ``python scripts/figures/four_arm_comparison.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    ARM_COLORS,
    ARM_LABELS,
    FIGURES_DIR,
    RESULTS_DIR,
    _apply_rc,
    load_finetune_results,
    load_result,
    plt,
    save_fig,
)

METRICS = [
    ("codebleu", "CodeBLEU"),
    ("syntax_valid_rate", "Syntax-valid rate"),
]
ARMS = ["A", "B", "C", "D"]


def _arm_a(finetune: dict) -> dict | None:
    """Arm A summary: the ep10 point plus the epoch-1/3/10 trend for each metric."""
    epochs = [(1, "finetune_A_ep1"), (3, "finetune_A_ep3"), (10, "finetune_A_ep10")]
    present = [(ep, finetune[name]) for ep, name in epochs if name in finetune]
    if not present or "finetune_A_ep10" not in finetune:
        return None
    top = finetune["finetune_A_ep10"]
    # A result file that exists but is missing a metric is as unusable as an absent one:
    # returning None routes it through the same "pending" placeholder instead of reaching
    # the plot as a None and crashing on `point + 0.015`.
    if any(top.get(key) is None for key, _ in METRICS):
        return None
    out: dict = {"pending": False, "epochs": [ep for ep, _ in present]}
    for key, _ in METRICS:
        out[key] = {
            "point": top.get(key),
            "trend": [m.get(key) for _, m in present],
        }
    return out


def _arm_b(finetune: dict) -> dict | None:
    """Arm B summary: mean across seeds + the seed band (min..max) per metric."""
    seeds = [
        finetune[n]
        for n in ("finetune_B_seed0", "finetune_B_seed1", "finetune_B_seed2")
        if n in finetune
    ]
    if not seeds:
        return None
    out: dict = {"pending": False, "n_seeds": len(seeds)}
    for key, _ in METRICS:
        vals = [m[key] for m in seeds if m.get(key) is not None]
        if not vals:
            return None  # present-but-metric-less files: `sum(vals) / len(vals)` would divide by 0
        out[key] = {
            "point": sum(vals) / len(vals),
            "lo": min(vals),
            "hi": max(vals),
        }
    return out


def _best_rag(results_dir: Path) -> dict | None:
    """Arm C: best (max CodeBLEU) committed RAG result, or None if none exist yet."""
    files = sorted(results_dir.glob("rag_*_test.json"))
    best = None
    for path in files:
        metrics = load_result(path).get("metrics", {})
        if "codebleu" not in metrics:
            continue
        if best is None or metrics["codebleu"] > best["codebleu"]:
            best = metrics
    return _arm_from_single(best)


def _arm_from_single(metrics: dict | None) -> dict | None:
    """Wrap a single committed metrics dict (arm C's best RAG run, arm D's LoRA run).

    Returns None -- i.e. render the "pending" placeholder -- when the file is absent *or*
    present but missing one of the plotted metrics; a None point would crash the plot.
    """
    if not metrics or any(metrics.get(k) is None for k, _ in METRICS):
        return None
    return {"pending": False, **{k: {"point": metrics.get(k)} for k, _ in METRICS}}


def collect_arms(results_dir: Path = RESULTS_DIR) -> dict[str, dict]:
    """Build the per-arm summary used by the plot; pending arms are ``{'pending': True}``."""
    results_dir = Path(results_dir)
    finetune = load_finetune_results(results_dir)

    lora_path = results_dir / "lora_qwen_test.json"
    lora_metrics = load_result(lora_path).get("metrics", {}) if lora_path.exists() else None

    arms = {
        "A": _arm_a(finetune),
        "B": _arm_b(finetune),
        "C": _best_rag(results_dir),
        "D": _arm_from_single(lora_metrics),
    }
    return {arm: (summary or {"pending": True}) for arm, summary in arms.items()}


def _draw_pending(ax, x: float) -> None:
    """A hollow hatched full-height slot + label: a reserved-but-empty arm."""
    ax.bar(
        x, 1.0, width=0.62, color="none", edgecolor="#9aa0aa", hatch="//", linewidth=1.0, zorder=1
    )
    ax.text(
        x,
        0.5,
        "⟨pending\nGPU batch⟩",
        ha="center",
        va="center",
        fontsize=8.5,
        style="italic",
        color="#6b7280",
    )


def _draw_panel(ax, arms: dict, key: str, title: str) -> None:
    _apply_rc()
    xs = range(len(ARMS))
    for x, arm in zip(xs, ARMS, strict=True):
        summary = arms[arm]
        if summary.get("pending"):
            _draw_pending(ax, x)
            continue
        m = summary[key]
        point = m["point"]
        if arm == "B":
            yerr = [[point - m["lo"]], [m["hi"] - point]]
            ax.bar(x, point, width=0.62, color=ARM_COLORS[arm], zorder=2)
            ax.errorbar(
                x, point, yerr=yerr, fmt="none", ecolor="#333333", capsize=5, lw=1.4, zorder=3
            )
        else:
            ax.bar(x, point, width=0.62, color=ARM_COLORS[arm], zorder=2)
        # Overlay arm-A's epoch trend as connected markers inside the bar.
        if arm == "A" and "trend" in m:
            trend = [v for v in m["trend"] if v is not None]
            ax.plot(
                [x] * len(trend),
                trend,
                marker="o",
                ms=5,
                color="#0b3d5c",
                lw=1.2,
                zorder=4,
                label="A: epochs 1/3/10" if key == "codebleu" else None,
            )
        # Value label above each real bar.
        ax.text(x, point + 0.015, f"{point:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([ARM_LABELS[a] for a in ARMS], rotation=20, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(title)
    ax.set_title(title)


def make(results_dir: Path = RESULTS_DIR, out_dir: Path = FIGURES_DIR) -> Path:
    """Render the four-arm comparison figure; returns the written PNG path."""
    arms = collect_arms(results_dir)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (key, title) in zip(axes, METRICS, strict=True):
        _draw_panel(ax, arms, key, title)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper left", fontsize=8, frameon=False)
    pending = [a for a, s in arms.items() if s.get("pending")]
    note = (
        f"arm B error bar = seed 0/1/2 band; arms {'/'.join(pending)} pending the GPU batch"
        if pending
        else "arm A markers = finetune epochs 1/3/10; arm B error bar = seed 0/1/2 band"
    )
    fig.suptitle(
        "Four-arm bug-fix comparison (CodeXGLUE-medium test, n=6545)\n" + note,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_fig(fig, "four_arm_comparison", out_dir)


if __name__ == "__main__":
    print(make())
