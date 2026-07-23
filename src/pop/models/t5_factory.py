"""T5 model factory matching notebook cell 22 (manual config, tied embeddings).

``pad_token_id``/``eos_token_id``/``decoder_start_token_id`` are hardcoded to
0/1/0 rather than threaded through ``cfg`` because :class:`pop.tokenizer.wrapper.PopTokenizer`
(and :func:`pop.tokenizer.train.train_tokenizer`) always fix those ids -- see
their module docstrings.
"""

from __future__ import annotations

from typing import Any

from transformers import T5Config, T5ForConditionalGeneration

PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1


def _get(cfg: Any, name: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def create_t5_model(vocab_size: int, cfg: Any) -> T5ForConditionalGeneration:
    """Build a randomly-initialized T5ForConditionalGeneration.

    Args:
        vocab_size: tokenizer vocabulary size.
        cfg: an object or dict exposing ``d_model``, ``d_ff``, ``d_kv``,
            ``num_heads``, ``num_layers``, ``num_decoder_layers`` (defaults
            match the notebook's hyperparameters if absent).
    """
    t5_config = T5Config(
        vocab_size=vocab_size,
        d_model=_get(cfg, "d_model", 512),
        d_ff=_get(cfg, "d_ff", 2048),
        d_kv=_get(cfg, "d_kv", 64),
        num_heads=_get(cfg, "num_heads", 8),
        num_layers=_get(cfg, "num_layers", 6),
        num_decoder_layers=_get(cfg, "num_decoder_layers", 6),
        decoder_start_token_id=PAD_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=PAD_TOKEN_ID,
        tie_word_embeddings=True,
    )

    model = T5ForConditionalGeneration(config=t5_config)
    model.resize_token_embeddings(vocab_size)
    return model
