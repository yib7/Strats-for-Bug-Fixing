"""SentencePiece Unigram tokenizer training for the pop T5 stack.

Trains a single SentencePiece model over a code corpus with:
  - fixed special-token ids matching :class:`pop.tokenizer.wrapper.PopTokenizer`
    (``pad=0``, ``eos=1``, ``unk=2``, no separate ``bos``);
  - 100 T5-style sentinel tokens ``<extra_id_0>`` .. ``<extra_id_99>`` registered
    as user-defined symbols so they always occupy stable, addressable ids.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import sentencepiece as spm

NUM_SENTINELS = 100
PAD_ID = 0
EOS_ID = 1
UNK_ID = 2


def sentinel_tokens() -> list[str]:
    return [f"<extra_id_{i}>" for i in range(NUM_SENTINELS)]


def train_tokenizer(
    corpus: Iterable[str],
    out_path: str | Path,
    vocab_size: int = 16384,
    model_type: str = "unigram",
) -> Path:
    """Train a SentencePiece model and write it to ``out_path``.

    Args:
        corpus: iterable of training strings (e.g. Java method sources).
        out_path: destination ``.model`` file path. Parent directories are
            created as needed.
        vocab_size: target vocabulary size (includes the 3 control tokens and
            100 sentinels).
        model_type: SentencePiece model type (default "unigram" per nb cell 11).

    Returns:
        The path to the written ``.model`` file (== ``out_path``).
    """
    out_path = Path(out_path)
    if out_path.suffix != ".model":
        out_path = out_path.with_suffix(".model")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_prefix = str(out_path.with_suffix(""))

    sentences = [text for text in corpus if text and text.strip()]
    if not sentences:
        raise ValueError("train_tokenizer requires a non-empty corpus")

    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(sentences),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        pad_id=PAD_ID,
        eos_id=EOS_ID,
        unk_id=UNK_ID,
        bos_id=-1,
        user_defined_symbols=sentinel_tokens(),
        character_coverage=1.0,
        # Small/toy corpora (unit tests) can't always hit the exact requested
        # vocab size; allow SentencePiece to use the closest feasible size
        # instead of raising.
        hard_vocab_limit=False,
    )

    produced = Path(model_prefix + ".model")
    if produced != out_path:
        produced.replace(out_path)
    return out_path
