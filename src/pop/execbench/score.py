"""pass@k estimator and result aggregation for execbench."""

from __future__ import annotations

from collections import defaultdict

from pop.execbench.harness import ExecResult


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021, HumanEval Eq. 1).

    `n` = samples drawn per problem, `c` = number of those that passed, `k` = the @k to
    estimate. Numerically stable: works in log-free product form over `n - c` terms rather
    than the naive `1 - C(n-c, k) / C(n, k)` binomial-coefficient ratio (which overflows for
    even moderately large `n`).

    **Nothing in this study calls it.** Track 2 draws exactly one greedy sample per bug, and
    at n=1 the estimator degenerates to `c`, so pass@1 is just `aggregate()`'s `pass_rate` --
    which is what `scripts/build_execbench_agreement_csv.py` and `docs/report.md` read. It is
    kept, tested and documented rather than deleted because it is the piece that would have
    to be right the moment anyone samples k>1 (temperature sampling, best-of-n reranking),
    and re-deriving a numerically stable form of Eq. 1 later is exactly the kind of thing
    that gets done wrong. Treat a call site appearing here as a signal that the sampling
    regime changed and the surrounding aggregation needs revisiting too.
    """
    if n < 0 or c < 0:
        raise ValueError(f"n and c must be >= 0 (got n={n}, c={c})")
    if k < 1:
        raise ValueError(f"k must be >= 1 (got k={k})")
    if c > n:
        raise ValueError(f"c ({c}) cannot exceed n ({n})")
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n})")

    if n - c < k:
        return 1.0

    result = 1.0
    for i in range(n - c + 1, n + 1):
        result *= 1.0 - k / i
    return 1.0 - result


def _error_kind_counts(results: list[ExecResult]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in results:
        counts[r.error_kind] += 1
    return dict(counts)


def aggregate(results: list[ExecResult]) -> dict:
    """Aggregate a batch of `ExecResult`s into pass/compile rates, overall and per-benchmark."""
    n = len(results)
    if n == 0:
        return {"n": 0, "compile_rate": 0.0, "pass_rate": 0.0, "per_benchmark": {}}

    compiled = sum(1 for r in results if r.compiled)
    passed = sum(1 for r in results if r.passed)

    grouped: dict[str, list[ExecResult]] = defaultdict(list)
    for r in results:
        grouped[r.bench or "unknown"].append(r)

    per_benchmark = {}
    for bench, items in grouped.items():
        bn = len(items)
        bc = sum(1 for i in items if i.compiled)
        bp = sum(1 for i in items if i.passed)
        per_benchmark[bench] = {
            "n": bn,
            "compile_rate": bc / bn,
            "pass_rate": bp / bn,
            "error_kind_counts": _error_kind_counts(items),
        }

    return {
        "n": n,
        "compile_rate": compiled / n,
        "pass_rate": passed / n,
        "per_benchmark": per_benchmark,
    }
