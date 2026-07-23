"""Whitespace normalization and exact-match scoring.

A naive exact-match check compares `prediction.strip() == reference.strip()`
after tokenizer decode. That is a trap for code: decoding changes internal
whitespace (tabs/newlines/multi-space runs collapse or shift) even when the
output is textually identical to the reference apart from whitespace, so strict
comparison can report 0% exact match on predictions that are in fact correct.
`exact_match` here normalizes whitespace before comparing; `exact_match_raw`
keeps the strict behavior so the two can be reported side by side.
"""

from __future__ import annotations

import re

_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_code(s: str) -> str:
    """Collapse every whitespace run to a single space and strip the ends.

    This is whitespace-only normalization -- it does not retokenize, so
    "foo ( )" and "foo()" remain different strings.
    """
    return _WHITESPACE_RUN.sub(" ", s.strip())


def exact_match(pred: str, ref: str) -> bool:
    """Whitespace-normalized exact match (the fixed metric)."""
    return normalize_code(pred) == normalize_code(ref)


def exact_match_raw(pred: str, ref: str) -> bool:
    """Strict exact match after strip only (the naive comparison, kept for contrast)."""
    return pred.strip() == ref.strip()
