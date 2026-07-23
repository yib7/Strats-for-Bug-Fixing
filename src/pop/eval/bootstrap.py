"""Percentile bootstrap confidence intervals.

Two entry points:

- `bootstrap_ci` — fast path for simple per-sample scores where the metric is
  the mean (e.g. exact-match rate). Resampling and aggregation are vectorized
  with numpy.
- `bootstrap_ci_fn` — general path for metrics that are *not* a simple mean
  of per-sample values, notably corpus-level metrics like CodeBLEU that are
  computed by a single call over the whole corpus rather than by averaging
  independent per-sample scores. Resamples the raw items and recomputes the
  metric function on each resample.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

# TypeVar form (not PEP-695 `[T]` syntax) so the package imports on Python 3.11
# runtimes (e.g. Google Colab); see pyproject.toml's requires-python floor.
T = TypeVar("T")


def bootstrap_ci(
    scores: list[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `scores`.

    Resamples `scores` with replacement `n_boot` times, computes the mean of
    each resample, and returns the `(alpha/2, 1 - alpha/2)` percentiles of
    the resulting distribution of means. Deterministic given `seed`.
    """
    if not scores:
        raise ValueError("scores must be non-empty")

    values = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)

    resample_indices = rng.integers(0, n, size=(n_boot, n))
    resample_means = values[resample_indices].mean(axis=1)

    lo = float(np.percentile(resample_means, 100 * (alpha / 2)))
    hi = float(np.percentile(resample_means, 100 * (1 - alpha / 2)))
    return lo, hi


def bootstrap_ci_fn(
    items: Sequence[T],
    metric_fn: Callable[[Sequence[T]], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a corpus-level metric.

    Unlike `bootstrap_ci`, which resamples precomputed per-sample floats and
    averages them, this resamples the raw `items` (e.g. (prediction,
    reference) pairs) with replacement and calls `metric_fn` on each
    resample. This is required for metrics that are *not* decomposable into
    a mean of independent per-sample scores -- e.g. CodeBLEU, which is
    computed by a single `calc_codebleu` call over the whole corpus rather
    than averaged from per-sample scores.

    `n_boot` defaults much lower than `bootstrap_ci` (1000 vs 10000) because
    `metric_fn` may itself be expensive (e.g. invoke a parser/compiler per
    call), so each replicate costs far more than a numpy mean.
    """
    if not items:
        raise ValueError("items must be non-empty")

    items = list(items)
    rng = np.random.default_rng(seed)
    n = len(items)

    resample_indices = rng.integers(0, n, size=(n_boot, n))
    replicate_values = [metric_fn([items[i] for i in indices]) for indices in resample_indices]

    lo = float(np.percentile(replicate_values, 100 * (alpha / 2)))
    hi = float(np.percentile(replicate_values, 100 * (1 - alpha / 2)))
    return lo, hi
