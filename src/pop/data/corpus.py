"""CodeSearchNet-Java corpus loading for T5 span-corruption pretraining.

Source selection:

- Primary: ``code-search-net/code_search_net``'s Hugging Face **auto-converted
  parquet export** (``refs/convert/parquet`` ref), ``java/train`` subset,
  addressed by direct ``hf://`` URL (:data:`PARQUET_JAVA_TRAIN`). This loads
  natively -- no loading *script*, no remote code. Script-based loading of the
  dataset's canonical loader is not an option: as of ``datasets>=3.x`` it
  requires ``trust_remote_code=True`` (removed entirely in ``datasets>=4.x``),
  and the upstream script targets an older API regardless. The parquet export
  is addressed by file path rather than ``load_dataset(PRIMARY_SOURCE, "java",
  revision=...)`` because the export exposes a single ``default`` config, not
  per-language configs (that form raises ``BuilderConfig 'java' not found``) --
  but the per-language parquet files themselves do exist under
  ``<lang>/<split>/*.parquet``. Verified 2026-07-17 (datasets 5.0.0).

- Fallback: ``Nan-Do/code-search-net-java``, a Java-only parquet re-upload of
  the same corpus that loads natively under ``datasets>=3.x`` (no custom
  script). Verified working 2026-07-17.

Both sources are untested here (no network access in this environment / in CI
unit tests) -- this is the documented, best-effort default loader. Callers can
bypass network access entirely by passing ``records`` directly (used by tests
with tiny fixtures, and usable in real runs if the caller already has a loaded
HF ``Dataset``/iterable of dicts in hand).

The length filter keeps methods with roughly 10-512 tokens, plus deduplication.
Since no trained tokenizer is available at corpus-load time, token counts are
approximated with whitespace splitting, which is close enough for a coarse
length filter.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

PRIMARY_SOURCE = "code-search-net/code_search_net"
FALLBACK_SOURCE = "Nan-Do/code-search-net-java"

# Direct hf:// URL to the canonical dataset's auto-converted parquet export,
# java/train subset. Addressed by file path because the parquet export exposes a
# single ``default`` config (not per-language configs), so
# ``load_dataset(PRIMARY_SOURCE, "java", revision="refs/convert/parquet")`` raises
# ``BuilderConfig 'java' not found`` -- but the per-language parquet *files* do
# exist under ``<lang>/<split>/*.parquet`` and load natively (no script, no
# remote code).
PARQUET_JAVA_TRAIN = f"hf://datasets/{PRIMARY_SOURCE}@refs/convert/parquet/java/train/*.parquet"

MIN_TOKENS = 10
MAX_TOKENS = 512

# Candidate column names across the primary dataset and known community
# mirrors, tried in order.
_TEXT_FIELDS = ("whole_func_string", "func_code_string", "code", "original_string")


def _extract_text(sample: dict) -> str | None:
    for key in _TEXT_FIELDS:
        value = sample.get(key)
        if value:
            return value
    return None


def _default_records() -> Iterable[dict]:
    """Load CodeSearchNet-Java from HuggingFace `datasets` (network required).

    Primary: the canonical dataset's auto-converted parquet export, ``java/train``
    subset, addressed by direct ``hf://`` URL (:data:`PARQUET_JAVA_TRAIN`) -- no
    loading script, no remote code. Fallback: ``Nan-Do/code-search-net-java``, a
    Java-only parquet re-upload of the same corpus that loads natively under
    ``datasets>=3.x``.
    """
    from datasets import load_dataset

    try:
        return load_dataset("parquet", data_files=PARQUET_JAVA_TRAIN, split="train")
    except Exception as e:
        logger.warning("primary source (canonical java/train parquet) failed: %s", e)
        return load_dataset(FALLBACK_SOURCE, split="train")


def load_pretraining_corpus(
    num_samples: int,
    seed: int = 42,
    records: Iterable[dict] | None = None,
) -> list[str]:
    """Load (and filter/dedupe/shuffle-subsample) a Java method corpus.

    Args:
        num_samples: maximum number of methods to return.
        seed: seed for the shuffle used to pick the returned subset
            (deterministic given the same filtered/deduped pool).
        records: optional injectable iterable of dataset-like dicts. Tests
            (and any caller that already has a loaded dataset/dicts) pass
            this to avoid network access. If ``None``, loads the real
            CodeSearchNet-Java dataset over the network.

    Returns:
        A list of Java method source strings, deduplicated, length-filtered
        (10-512 whitespace tokens), shuffled deterministically by ``seed``,
        and truncated to ``num_samples``.
    """
    if records is None:
        records = _default_records()

    seen: set[str] = set()
    corpus: list[str] = []
    for sample in records:
        text = _extract_text(sample)
        if not text:
            continue
        if text in seen:
            continue
        n_tokens = len(text.split())
        if n_tokens < MIN_TOKENS or n_tokens > MAX_TOKENS:
            continue
        seen.add(text)
        corpus.append(text)

    rng = random.Random(seed)
    rng.shuffle(corpus)
    return corpus[:num_samples]


# Separator between methods in a `load_corpus_file`-format fixture, matching
# `scripts/build_smoke_fixtures.py`'s `CORPUS_SEPARATOR`.
CORPUS_FILE_SEPARATOR = "\n// ===SMOKE_METHOD_SEP===\n"


def load_corpus_file(path: str | Path) -> list[dict]:
    """Load a `CORPUS_FILE_SEPARATOR`-delimited text file of Java methods.

    Returns a list of `{"code": str}` records -- the same dict shape `load_pretraining_corpus`
    expects via its `records` parameter (`"code"` is one of `_TEXT_FIELDS`), so callers can do
    `load_pretraining_corpus(n, records=load_corpus_file(path))` to reuse the existing
    dedup/length-filter/shuffle logic against a committed fixture instead of the network.
    """
    text = Path(path).read_text(encoding="utf-8")
    methods = [m.strip() for m in text.split(CORPUS_FILE_SEPARATOR) if m.strip()]
    return [{"code": method} for method in methods]
