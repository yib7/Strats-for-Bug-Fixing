"""Tests for pop.train.span_corruption (pure logic + dataset + collator)."""

from __future__ import annotations

import random

import pytest

from pop.train.span_corruption import (
    DataCollatorForT5,
    SpanCorruptionDataset,
    corrupt_spans,
    is_sentinel_placeholder,
    placeholder_sentinel_index,
    resolve_sentinels,
    sentinel_placeholder,
)


def _sentinels_in_order(seq: list[int]) -> list[int]:
    return [placeholder_sentinel_index(t) for t in seq if is_sentinel_placeholder(t)]


# ---- corrupt_spans: pure logic ----


def test_corrupt_spans_masks_approximately_target_rate():
    token_ids = list(range(100))
    input_ids, labels = corrupt_spans(
        token_ids, corruption_rate=0.15, mean_span_len=3, rng=random.Random(0)
    )
    num_masked = sum(1 for t in labels if not is_sentinel_placeholder(t))
    assert 10 <= num_masked <= 20  # ~15 +/- slack for span-length rounding


def test_corrupt_spans_sentinels_strictly_increasing_and_paired():
    token_ids = list(range(50))
    input_ids, labels = corrupt_spans(
        token_ids, corruption_rate=0.3, mean_span_len=3, rng=random.Random(1)
    )

    input_sentinels = _sentinels_in_order(input_ids)
    label_sentinels = _sentinels_in_order(labels)

    assert input_sentinels == label_sentinels
    assert input_sentinels == sorted(input_sentinels)
    assert input_sentinels == list(range(len(input_sentinels)))  # 0, 1, 2, ...


def test_corrupt_spans_reconstruction_round_trip():
    token_ids = list(range(40))
    input_ids, labels = corrupt_spans(
        token_ids, corruption_rate=0.25, mean_span_len=2, rng=random.Random(2)
    )

    # Reconstruct the original sequence by substituting each sentinel in
    # input_ids with its corresponding span pulled out of labels.
    span_by_index: dict[int, list[int]] = {}
    i = 0
    while i < len(labels):
        idx = placeholder_sentinel_index(labels[i])
        i += 1
        span = []
        while i < len(labels) and not is_sentinel_placeholder(labels[i]):
            span.append(labels[i])
            i += 1
        span_by_index[idx] = span

    reconstructed = []
    for tok in input_ids:
        if is_sentinel_placeholder(tok):
            reconstructed.extend(span_by_index[placeholder_sentinel_index(tok)])
        else:
            reconstructed.append(tok)

    assert reconstructed == token_ids


def test_corrupt_spans_deterministic_by_seed():
    token_ids = list(range(60))
    out_a = corrupt_spans(token_ids, corruption_rate=0.2, mean_span_len=3, rng=random.Random(42))
    out_b = corrupt_spans(token_ids, corruption_rate=0.2, mean_span_len=3, rng=random.Random(42))
    assert out_a == out_b


def test_corrupt_spans_different_seeds_usually_differ():
    token_ids = list(range(60))
    out_a = corrupt_spans(token_ids, corruption_rate=0.2, mean_span_len=3, rng=random.Random(1))
    out_b = corrupt_spans(token_ids, corruption_rate=0.2, mean_span_len=3, rng=random.Random(2))
    assert out_a != out_b


def test_corrupt_spans_empty_input():
    assert corrupt_spans([], rng=random.Random(0)) == ([], [])


def test_corrupt_spans_single_token_masks_it():
    input_ids, labels = corrupt_spans([7], rng=random.Random(0))
    assert input_ids == [sentinel_placeholder(0)]
    assert labels == [sentinel_placeholder(0), 7]


def test_sentinel_placeholder_helpers_roundtrip():
    for i in range(10):
        ph = sentinel_placeholder(i)
        assert is_sentinel_placeholder(ph)
        assert placeholder_sentinel_index(ph) == i
    assert not is_sentinel_placeholder(0)
    assert not is_sentinel_placeholder(5)


def test_resolve_sentinels_maps_placeholders_only():
    ids = [1, 2, sentinel_placeholder(0), 3, sentinel_placeholder(1)]
    resolved = resolve_sentinels(ids, lambda i: 1000 + i)
    assert resolved == [1, 2, 1000, 3, 1001]


# ---- SpanCorruptionDataset + DataCollatorForT5 (need torch + a fake tokenizer) ----


class _FakeTokenizer:
    """Deterministic word-splitting tokenizer good enough for dataset tests."""

    pad_id = 0
    eos_id = 1
    unk_id = 2

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._next_id = 10  # leave room below for pad/eos/unk/sentinels

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = self._next_id
                self._next_id += 1
            ids.append(self._vocab[word])
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def sentinel_id(self, index: int) -> int:
        return 900 + index


def test_span_corruption_dataset_shapes():
    texts = [
        "public void foo ( ) { int x = 1 ; return x ; }",
        "private int bar ( int y ) { return y + 1 ; }",
    ]
    tok = _FakeTokenizer()
    ds = SpanCorruptionDataset(
        texts, tok, max_length=64, corruption_rate=0.3, mean_span_len=2, seed=0
    )
    assert len(ds) == 2
    item = ds[0]
    assert set(item.keys()) == {"input_ids", "labels"}
    assert item["input_ids"][-1] == tok.eos_id
    assert item["labels"][-1] == tok.eos_id
    # Sentinel ids used are real tokenizer ids (>= 900), not placeholders.
    assert all(t >= 0 for t in item["input_ids"])
    assert all(t >= 0 for t in item["labels"])


def test_span_corruption_dataset_deterministic_by_seed():
    texts = ["public void foo ( ) { int x = 1 ; return x ; }"] * 3
    tok_a, tok_b = _FakeTokenizer(), _FakeTokenizer()
    ds_a = SpanCorruptionDataset(texts, tok_a, max_length=64, seed=123)
    ds_b = SpanCorruptionDataset(texts, tok_b, max_length=64, seed=123)
    assert ds_a[0] == ds_b[0]


def test_data_collator_pads_and_masks_labels():
    pytest.importorskip("torch")
    collator = DataCollatorForT5(pad_id=0, label_pad_id=-100)
    features = [
        {"input_ids": [5, 6, 7], "labels": [8, 9]},
        {"input_ids": [1, 2], "labels": [3, 4, 5, 6]},
    ]
    batch = collator(features)
    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"].shape == (2, 4)
    assert batch["input_ids"][1].tolist() == [1, 2, 0]
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]
    assert batch["labels"][0].tolist() == [8, 9, -100, -100]
