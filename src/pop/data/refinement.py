"""CodeXGLUE code-refinement ("medium") pair loading for finetuning.

Source: ``google/code_x_glue_cc_code_refinement``, config name ``"medium"``.
Splits are ``train`` / ``validation`` / ``test``.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path

SOURCE = "google/code_x_glue_cc_code_refinement"
CONFIG_NAME = "medium"


def _default_records(split: str) -> Iterable[dict]:
    """Load a CodeXGLUE code-refinement split from HuggingFace (network required)."""
    from datasets import load_dataset

    return load_dataset(SOURCE, name=CONFIG_NAME, split=split)


def load_refinement_pairs(split: str, records: Iterable[dict] | None = None) -> list[dict]:
    """Load buggy/fixed Java method pairs for a given split.

    Args:
        split: one of "train", "validation", "test".
        records: optional injectable iterable of dicts with "buggy"/"fixed"
            keys (used by tests to avoid network access).

    Returns:
        A list of ``{"buggy": str, "fixed": str}`` dicts.
    """
    if records is None:
        records = _default_records(split)

    pairs: list[dict] = []
    for example in records:
        buggy = example.get("buggy")
        fixed = example.get("fixed")
        if buggy is None or fixed is None:
            continue
        pairs.append({"buggy": buggy, "fixed": fixed})
    return pairs


def subsample(pairs: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Deterministically subsample `n` pairs (or all of them if n >= len(pairs))."""
    if n >= len(pairs):
        return list(pairs)
    rng = random.Random(seed)
    return rng.sample(pairs, n)


def load_pairs_file(path: str | Path) -> list[dict]:
    """Load buggy/fixed pairs from a JSONL fixture file (one `{"buggy": ..., "fixed": ...}`
    object per line), matching `scripts/build_smoke_fixtures.py`'s output format.
    """
    pairs: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        pairs.append({"buggy": record["buggy"], "fixed": record["fixed"]})
    return pairs
