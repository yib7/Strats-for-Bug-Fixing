"""T5 span-corruption: pure masking function + dataset + collator.

``corrupt_spans`` is a pure function of ``(token_ids, corruption_rate,
mean_span_len, rng)`` with no dependency on a tokenizer/vocabulary, so it can
be unit tested without training a SentencePiece model. Sentinels are
represented as small negative integers (``-1, -2, -3, ...`` for sentinel
index ``0, 1, 2, ...``) inside ``corrupt_spans``'s output; callers that need
*real* tokenizer sentinel ids (e.g. :class:`SpanCorruptionDataset`) resolve
them via :func:`resolve_sentinels` against a tokenizer's
``sentinel_id(index)`` method.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from typing import Any


def sentinel_placeholder(index: int) -> int:
    """Placeholder id used by :func:`corrupt_spans` for sentinel ``index``."""
    return -(index + 1)


def placeholder_sentinel_index(token_id: int) -> int:
    """Inverse of :func:`sentinel_placeholder`; only valid for negative ids."""
    return -token_id - 1


def is_sentinel_placeholder(token_id: int) -> bool:
    return token_id < 0


def resolve_sentinels(ids: Iterable[int], sentinel_id_fn: Callable[[int], int]) -> list[int]:
    """Map placeholder sentinel ids (negative ints) to real tokenizer ids."""
    return [
        sentinel_id_fn(placeholder_sentinel_index(tid)) if is_sentinel_placeholder(tid) else tid
        for tid in ids
    ]


def _distribute(total: int, parts: int, rng: random.Random, allow_zero: bool) -> list[int]:
    """Randomly split ``total`` into ``parts`` nonnegative ints summing to total.

    If ``allow_zero`` is False, every part is >= 1 (requires total >= parts).
    """
    if parts <= 0:
        return []
    if parts == 1:
        return [total]
    if allow_zero:
        cuts = sorted(rng.choices(range(0, total + 1), k=parts - 1))
    else:
        cuts = sorted(rng.sample(range(1, total), parts - 1))
    points = [0, *cuts, total]
    return [points[i + 1] - points[i] for i in range(parts)]


def corrupt_spans(
    token_ids: list[int],
    corruption_rate: float = 0.15,
    mean_span_len: int = 3,
    rng: random.Random | None = None,
) -> tuple[list[int], list[int]]:
    """Apply T5-style span corruption to a sequence of token ids.

    Returns ``(input_ids, labels)`` where:
      - ``input_ids`` is the original sequence with each corrupted span
        replaced by a single sentinel placeholder id.
      - ``labels`` is the concatenation, for each span in order, of that
        span's sentinel placeholder id followed by the original span tokens.

    Sentinel placeholder ids are strictly increasing in magnitude (index
    0, 1, 2, ...) and appear in the same relative order in both
    ``input_ids`` and ``labels`` -- i.e. they are paired 1:1.

    Neither an EOS token nor real tokenizer sentinel ids are added here;
    that is the caller's job (see :class:`SpanCorruptionDataset`), keeping
    this function a pure, tokenizer-independent transformation.
    """
    n = len(token_ids)
    if n == 0:
        return [], []
    if rng is None:
        rng = random.Random()

    num_to_mask = max(1, round(n * corruption_rate))
    num_to_mask = min(num_to_mask, n)

    num_spans = max(1, round(num_to_mask / max(1, mean_span_len)))
    num_spans = min(num_spans, num_to_mask)

    span_lengths = _distribute(num_to_mask, num_spans, rng, allow_zero=False)
    gap_lengths = _distribute(n - num_to_mask, num_spans + 1, rng, allow_zero=True)

    input_ids: list[int] = []
    labels: list[int] = []
    pos = 0
    for i in range(num_spans):
        gap = gap_lengths[i]
        input_ids.extend(token_ids[pos : pos + gap])
        pos += gap

        span_len = span_lengths[i]
        span_tokens = token_ids[pos : pos + span_len]
        pos += span_len

        sentinel = sentinel_placeholder(i)
        input_ids.append(sentinel)
        labels.append(sentinel)
        labels.extend(span_tokens)

    trailing_gap = gap_lengths[num_spans]
    input_ids.extend(token_ids[pos : pos + trailing_gap])

    return input_ids, labels


class SpanCorruptionDataset:
    """Torch-compatible dataset yielding span-corrupted (input_ids, labels).

    Sequences are left unpadded (variable length); pad with
    :class:`DataCollatorForT5` at batch-collation time.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: Any,
        max_length: int = 512,
        corruption_rate: float = 0.15,
        mean_span_len: int = 3,
        seed: int = 42,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.corruption_rate = corruption_rate
        self.mean_span_len = mean_span_len
        self._seed = seed

    def __len__(self) -> int:
        return len(self.texts)

    def _encode(self, idx: int) -> list[int]:
        token_ids = self.tokenizer.encode(self.texts[idx], max_length=self.max_length - 1)
        if len(token_ids) < 2:
            # Degenerate/near-empty sample: fall back to the first example,
            # mirroring the notebook's guard against unmaskable inputs.
            token_ids = self.tokenizer.encode(self.texts[0], max_length=self.max_length - 1)
        return token_ids

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        token_ids = self._encode(idx)
        # Deterministic-but-distinct per-item rng derived from the dataset seed.
        rng = random.Random(hash((self._seed, idx)) & 0xFFFFFFFF)
        raw_input, raw_labels = corrupt_spans(
            token_ids, self.corruption_rate, self.mean_span_len, rng
        )
        input_ids = resolve_sentinels(raw_input, self.tokenizer.sentinel_id)
        labels = resolve_sentinels(raw_labels, self.tokenizer.sentinel_id)
        input_ids.append(self.tokenizer.eos_id)
        labels.append(self.tokenizer.eos_id)
        return {"input_ids": input_ids, "labels": labels}


class DataCollatorForT5:
    """Pads a batch of variable-length ``{"input_ids", "labels"}`` dicts.

    ``input_ids`` are padded with ``pad_id`` (and an ``attention_mask`` is
    produced); ``labels`` are padded with ``-100`` so the padding is ignored
    by the loss.
    """

    def __init__(self, pad_id: int, label_pad_id: int = -100):
        self.pad_id = pad_id
        self.label_pad_id = label_pad_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_input_len = max(len(f["input_ids"]) for f in features)
        max_label_len = max(len(f["labels"]) for f in features)

        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            ids = f["input_ids"]
            pad_len = max_input_len - len(ids)
            input_ids.append(ids + [self.pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

            lbl = f["labels"]
            label_pad_len = max_label_len - len(lbl)
            labels.append(lbl + [self.label_pad_id] * label_pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
