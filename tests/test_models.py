"""Tests for pop.models.t5_factory."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pop.config import T5ModelConfig  # noqa: E402
from pop.models.t5_factory import create_t5_model  # noqa: E402

TINY_CFG = T5ModelConfig(
    d_model=16, d_ff=32, d_kv=4, num_heads=2, num_layers=1, num_decoder_layers=1
)


def test_create_t5_model_shapes():
    model = create_t5_model(vocab_size=64, cfg=TINY_CFG)
    assert model.config.vocab_size == 64
    assert model.config.d_model == 16
    assert model.config.num_layers == 1
    assert model.config.num_decoder_layers == 1


def test_create_t5_model_special_token_ids():
    model = create_t5_model(vocab_size=64, cfg=TINY_CFG)
    assert model.config.pad_token_id == 0
    assert model.config.eos_token_id == 1
    assert model.config.decoder_start_token_id == 0


def test_create_t5_model_ties_word_embeddings():
    model = create_t5_model(vocab_size=64, cfg=TINY_CFG)
    assert model.config.tie_word_embeddings is True


def test_create_t5_model_accepts_dict_cfg():
    cfg = {
        "d_model": 16,
        "d_ff": 32,
        "d_kv": 4,
        "num_heads": 2,
        "num_layers": 1,
        "num_decoder_layers": 1,
    }
    model = create_t5_model(vocab_size=64, cfg=cfg)
    assert model.config.d_model == 16


def test_create_t5_model_forward_smoke():
    model = create_t5_model(vocab_size=64, cfg=TINY_CFG)
    input_ids = torch.randint(3, 64, (2, 5))
    labels = torch.randint(3, 64, (2, 4))
    out = model(input_ids=input_ids, labels=labels)
    assert out.loss is not None
    assert out.logits.shape == (2, 4, 64)
