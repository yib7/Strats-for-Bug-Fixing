"""Tests for pop.data.corpus and pop.data.refinement (no network access)."""

from __future__ import annotations

from pop.data.corpus import load_pretraining_corpus
from pop.data.refinement import load_refinement_pairs, subsample

# ---- corpus ----

LONG_METHOD = " ".join(["token"] * 20)  # 20 whitespace tokens, within [10, 512]
SHORT_METHOD = " ".join(["token"] * 3)  # 3 tokens, below MIN_TOKENS
TOO_LONG_METHOD = " ".join(["token"] * 600)  # above MAX_TOKENS


def _fixture_records():
    return [
        {"whole_func_string": LONG_METHOD + " a"},
        {"whole_func_string": LONG_METHOD + " b"},
        {"whole_func_string": LONG_METHOD + " a"},  # exact duplicate of first -> deduped
        {"whole_func_string": SHORT_METHOD},  # filtered: too short
        {"whole_func_string": TOO_LONG_METHOD},  # filtered: too long
        {"whole_func_string": ""},  # filtered: empty
        {"other_field": "no text field here"},  # filtered: missing text
    ]


def test_load_pretraining_corpus_filters_and_dedupes():
    corpus = load_pretraining_corpus(num_samples=10, seed=1, records=_fixture_records())
    assert len(corpus) == 2
    assert all(10 <= len(text.split()) <= 512 for text in corpus)
    assert len(set(corpus)) == len(corpus)


def test_load_pretraining_corpus_respects_num_samples():
    corpus = load_pretraining_corpus(num_samples=1, seed=1, records=_fixture_records())
    assert len(corpus) == 1


def test_load_pretraining_corpus_deterministic_by_seed():
    records = _fixture_records()
    corpus_a = load_pretraining_corpus(num_samples=10, seed=7, records=list(records))
    corpus_b = load_pretraining_corpus(num_samples=10, seed=7, records=list(records))
    assert corpus_a == corpus_b


def test_load_pretraining_corpus_supports_fallback_field_names():
    records = [{"func_code_string": LONG_METHOD + " x"}, {"code": LONG_METHOD + " y"}]
    corpus = load_pretraining_corpus(num_samples=10, seed=1, records=records)
    assert len(corpus) == 2


# ---- refinement ----


def _refinement_fixture():
    return [
        {"buggy": "int a = 1;", "fixed": "int a = 2;"},
        {"buggy": "int b = 1;", "fixed": "int b = 2;"},
        {"buggy": "int c = 1;", "fixed": "int c = 2;"},
        {"buggy": "int d = 1;", "fixed": "int d = 2;"},
        {"buggy": "int e = 1;", "fixed": "int e = 2;"},
    ]


def test_load_refinement_pairs_shape():
    pairs = load_refinement_pairs("train", records=_refinement_fixture())
    assert len(pairs) == 5
    for pair in pairs:
        assert set(pair.keys()) == {"buggy", "fixed"}


def test_load_refinement_pairs_skips_incomplete_records():
    records = [*_refinement_fixture(), {"buggy": "only buggy"}, {"fixed": "only fixed"}]
    pairs = load_refinement_pairs("train", records=records)
    assert len(pairs) == 5


def test_subsample_returns_requested_count():
    pairs = _refinement_fixture()
    sub = subsample(pairs, 3, seed=42)
    assert len(sub) == 3
    for item in sub:
        assert item in pairs


def test_subsample_returns_all_when_n_exceeds_pool():
    pairs = _refinement_fixture()
    sub = subsample(pairs, 100, seed=42)
    assert len(sub) == len(pairs)


def test_subsample_deterministic_by_seed():
    pairs = _refinement_fixture()
    sub_a = subsample(pairs, 3, seed=5)
    sub_b = subsample(pairs, 3, seed=5)
    assert sub_a == sub_b
