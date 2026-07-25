"""Tests for the pop.eval stack (normalize, metrics, bootstrap)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pop.eval.bootstrap import bootstrap_ci, bootstrap_ci_fn
from pop.eval.metrics import evaluate_predictions, write_results
from pop.eval.normalize import exact_match, exact_match_raw, normalize_code

# ---------------------------------------------------------------------------
# normalize_code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  int x = 1;  ", "int x = 1;"),
        ("int\tx\t=\t1;", "int x = 1;"),
        ("int x =\n1;\n", "int x = 1;"),
        ("int   x  =    1;", "int x = 1;"),
        ("int x = 1;\n\n\nreturn x;", "int x = 1; return x;"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize_code_collapses_whitespace(raw, expected):
    assert normalize_code(raw) == expected


def test_normalize_code_does_not_retokenize():
    # normalize_code is whitespace-only: it must not add/remove spaces around
    # tokens that weren't separated by whitespace in the input.
    assert normalize_code("foo ( )") != normalize_code("foo()")
    assert normalize_code("foo ( )") == "foo ( )"
    assert normalize_code("foo()") == "foo()"


# ---------------------------------------------------------------------------
# exact_match / exact_match_raw
# ---------------------------------------------------------------------------


def test_exact_match_raw_is_strict_string_equality_after_strip():
    assert exact_match_raw("int x = 1;", "int x = 1;") is True
    assert exact_match_raw("int  x = 1;", "int x = 1;") is False


def test_exact_match_normalizes_whitespace():
    assert exact_match("int  x = 1;", "int x = 1;") is True
    assert exact_match("int\tx\t=\t1;", "int x = 1;") is True


def test_exact_match_still_false_for_different_code():
    assert exact_match("int x = 2;", "int x = 1;") is False
    assert exact_match_raw("int x = 2;", "int x = 1;") is False


def test_exact_match_raw_vs_normalized_divergence():
    pred = "public void foo() {\n    return;\n}"
    ref = "public void foo() { return; }"
    assert exact_match_raw(pred, ref) is False
    assert exact_match(pred, ref) is True


def test_strict_equality_whitespace_pitfall():
    """The strict-`==` exact-match pitfall this metric guards against.

    A naive `prediction.strip() == reference.strip()` check run on decoded output
    fails when detokenization shifts internal whitespace (e.g. tab -> space run),
    scoring a pred/ref pair that differs only in whitespace as *not* a match even
    though the code is identical. Left unguarded this can drive exact match to 0%
    on predictions that are actually correct. The normalized metric collapses
    whitespace before comparing.
    """
    pred = "public int add(int a, int b) {\treturn a + b;\t}"
    ref = "public int add(int a, int b) { return a + b; }"

    assert exact_match_raw(pred, ref) is False
    assert exact_match(pred, ref) is True


# ---------------------------------------------------------------------------
# evaluate_predictions
# ---------------------------------------------------------------------------


VALID_JAVA = "class _W { int foo() { return 1; } }"
BROKEN_JAVA = "class _W { int foo( { return 1; ] } }"


def test_evaluate_predictions_returns_expected_keys():
    preds = ["int x = 1;", "int y = 2;"]
    refs = ["int x = 1;", "int y = 3;"]
    metrics = evaluate_predictions(preds, refs)
    assert set(metrics) == {"em", "em_raw", "codebleu", "syntax_valid_rate", "n"}
    assert metrics["n"] == 2


def test_evaluate_predictions_em_counts_matches():
    preds = ["int  x = 1;", "int y = 2;"]
    refs = ["int x = 1;", "int y = 3;"]
    metrics = evaluate_predictions(preds, refs)
    assert metrics["em"] == pytest.approx(0.5)
    assert metrics["em_raw"] == pytest.approx(0.0)


def test_evaluate_predictions_syntax_valid_rate():
    preds = [VALID_JAVA, BROKEN_JAVA]
    refs = [VALID_JAVA, VALID_JAVA]
    metrics = evaluate_predictions(preds, refs)
    assert metrics["syntax_valid_rate"] == pytest.approx(0.5)


def test_evaluate_predictions_codebleu_is_high_for_identical_code():
    preds = [VALID_JAVA]
    refs = [VALID_JAVA]
    metrics = evaluate_predictions(preds, refs)
    assert metrics["codebleu"] > 0.9


def test_evaluate_predictions_empty_lists_raises():
    with pytest.raises(ValueError):
        evaluate_predictions([], [])


def test_evaluate_predictions_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        evaluate_predictions(["a"], ["a", "b"])


# ---------------------------------------------------------------------------
# write_results
# ---------------------------------------------------------------------------


def test_write_results_writes_expected_schema(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics = {"em": 1.0, "em_raw": 0.0, "codebleu": 0.9, "syntax_valid_rate": 1.0, "n": 2}
    config = {"model": "t5-small"}

    path = write_results("my-run", metrics, config)

    assert path == tmp_path / "results" / "my-run.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["config"] == config
    assert data["metrics"] == metrics
    assert data["n"] == 2
    assert "timestamp" in data
    assert "git_sha" in data


def test_write_results_refuses_to_clobber_an_existing_file(tmp_path, monkeypatch):
    # results/ holds committed, published measurements; a second run must not replace one.
    monkeypatch.chdir(tmp_path)
    metrics = {"em": 1.0, "em_raw": 0.0, "codebleu": 0.9, "syntax_valid_rate": 1.0, "n": 2}
    path = write_results("published", metrics, {})
    original = path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError) as excinfo:
        write_results("published", {"n": 1}, {})

    assert "--name" in str(excinfo.value)
    assert path.read_text(encoding="utf-8") == original  # untouched


def test_write_results_overwrite_flag_allows_replacement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_results("run", {"n": 2}, {})
    path = write_results("run", {"n": 7}, {}, overwrite=True)
    assert json.loads(path.read_text(encoding="utf-8"))["n"] == 7


@pytest.mark.parametrize("name", ["smoke_local", "execbench_local_validate_references"])
def test_write_results_scratch_names_are_replaceable(name, tmp_path, monkeypatch):
    # The CLI's own gitignored `*_local*` defaults: re-running the documented commands
    # must stay idempotent, so only *non*-scratch names get the clobber guard.
    monkeypatch.chdir(tmp_path)
    write_results(name, {"n": 2}, {})
    path = write_results(name, {"n": 7}, {})
    assert json.loads(path.read_text(encoding="utf-8"))["n"] == 7


def test_scratch_run_name_matches_the_gitignore_pattern():
    from pop.eval.metrics import is_scratch_run_name

    assert is_scratch_run_name("smoke_local")
    assert is_scratch_run_name("execbench_local_predictions")
    assert not is_scratch_run_name("smoke")
    assert not is_scratch_run_name("execbench_validate_references")
    assert not is_scratch_run_name("finetune_A_ep10_test")


# the backslash cases must be rejected on POSIX too, where `\` is a legal filename
# character: the guard's contract is "no path separators", not "none that this OS parses".
@pytest.mark.parametrize("name", ["../oops", "sub/dir", "..\\oops", "sub\\dir", "", ".", ".."])
def test_write_results_rejects_names_that_escape_the_results_dir(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        write_results(name, {"n": 1}, {})
    assert not (tmp_path.parent / "oops.json").exists()


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_syntax_valid_rate_flags_broken_java():
    metrics = evaluate_predictions([VALID_JAVA, BROKEN_JAVA], [VALID_JAVA, VALID_JAVA])
    assert 0.0 <= metrics["syntax_valid_rate"] <= 1.0


def test_syntax_valid_rate_empty_and_whitespace_predictions_are_invalid():
    # An empty/whitespace-only prediction wraps to `class _W {  }`, which
    # tree-sitter parses without ERROR nodes -- but a model that emits nothing
    # should not be scored as syntactically valid.
    metrics = evaluate_predictions(["", "   ", "\n\t"], [VALID_JAVA, VALID_JAVA, VALID_JAVA])
    assert metrics["syntax_valid_rate"] == pytest.approx(0.0)


def test_bootstrap_ci_bounds_sane_for_constant_distribution():
    scores = [1.0] * 50
    lo, hi = bootstrap_ci(scores, n_boot=1000, alpha=0.05, seed=42)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_bootstrap_ci_seeded_is_reproducible():
    rng = np.random.default_rng(0)
    scores = list(rng.uniform(0, 1, size=200))
    lo1, hi1 = bootstrap_ci(scores, n_boot=2000, alpha=0.05, seed=42)
    lo2, hi2 = bootstrap_ci(scores, n_boot=2000, alpha=0.05, seed=42)
    assert (lo1, hi1) == (lo2, hi2)


def test_bootstrap_ci_bounds_contain_mean_for_known_distribution():
    rng = np.random.default_rng(1)
    scores = list(rng.normal(loc=0.5, scale=0.05, size=500))
    lo, hi = bootstrap_ci(scores, n_boot=5000, alpha=0.05, seed=42)
    assert lo < np.mean(scores) < hi
    assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# bootstrap_ci_fn
# ---------------------------------------------------------------------------


def test_bootstrap_ci_fn_matches_bootstrap_ci_for_mean_metric():
    rng = np.random.default_rng(2)
    scores = list(rng.uniform(0, 1, size=300))

    lo_simple, hi_simple = bootstrap_ci(scores, n_boot=5000, alpha=0.05, seed=42)
    lo_fn, hi_fn = bootstrap_ci_fn(
        scores, metric_fn=lambda xs: float(np.mean(xs)), n_boot=5000, alpha=0.05, seed=42
    )

    # Same seed, same resampling scheme (rng.integers(0, n, size=(n_boot, n))),
    # same aggregation -> should match closely (not bit-identical since
    # bootstrap_ci vectorizes the mean while bootstrap_ci_fn calls metric_fn
    # per replicate, but the underlying resample indices and math agree).
    assert lo_fn == pytest.approx(lo_simple, abs=1e-9)
    assert hi_fn == pytest.approx(hi_simple, abs=1e-9)


def test_bootstrap_ci_fn_corpus_level_metric_contains_point_estimate():
    # A non-decomposable, corpus-level metric: ratio of totals rather than a
    # mean of per-item ratios. Mirrors CodeBLEU-style corpus scoring, where
    # the metric must be recomputed over the whole resampled corpus.
    rng = np.random.default_rng(3)
    numerators = rng.integers(1, 10, size=100)
    denominators = rng.integers(10, 20, size=100)
    items = list(zip(numerators, denominators, strict=True))

    def corpus_ratio(pairs):
        nums = sum(p[0] for p in pairs)
        dens = sum(p[1] for p in pairs)
        return nums / dens

    point_estimate = corpus_ratio(items)
    lo, hi = bootstrap_ci_fn(items, metric_fn=corpus_ratio, n_boot=500, alpha=0.05, seed=42)

    assert lo <= point_estimate <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_fn_seeded_is_reproducible():
    items = list(range(50))

    def toy_metric(xs):
        return sum(xs) / len(xs)

    lo1, hi1 = bootstrap_ci_fn(items, metric_fn=toy_metric, n_boot=300, alpha=0.05, seed=42)
    lo2, hi2 = bootstrap_ci_fn(items, metric_fn=toy_metric, n_boot=300, alpha=0.05, seed=42)
    assert (lo1, hi1) == (lo2, hi2)


def test_bootstrap_ci_fn_empty_items_raises():
    with pytest.raises(ValueError):
        bootstrap_ci_fn([], metric_fn=lambda xs: 0.0)
