"""Tests for pop.generate.generate_t5_predictions.

Builds a tiny in-test tokenizer + T5, saves it, and checks the batched
generation loop returns one decoded string per input (untrained -> content is
garbage, so only shape/plumbing is asserted). CPU-only, no network.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pop.config import T5ModelConfig  # noqa: E402
from pop.generate import generate_t5_predictions  # noqa: E402
from pop.models.t5_factory import create_t5_model  # noqa: E402
from pop.tokenizer.train import train_tokenizer  # noqa: E402
from pop.tokenizer.wrapper import PopTokenizer  # noqa: E402

TINY_CORPUS = [
    "public void foo ( ) { int x = 1 ; return x ; }",
    "private int bar ( int y ) { return y + 1 ; }",
    "public String baz ( String s ) { return s . trim ( ) ; }",
] * 30

TINY_MODEL_CFG = T5ModelConfig(
    d_model=16, d_ff=32, d_kv=4, num_heads=2, num_layers=1, num_decoder_layers=1
)


@pytest.fixture()
def tiny_model_and_tokenizer(tmp_path):
    tok_path = train_tokenizer(TINY_CORPUS, tmp_path / "tok.model", vocab_size=160)
    tokenizer = PopTokenizer.load(tok_path)
    model = create_t5_model(len(tokenizer), TINY_MODEL_CFG)
    model_dir = tmp_path / "model"
    model.save_pretrained(str(model_dir))
    return model_dir, tok_path


def test_generate_returns_one_prediction_per_input(tiny_model_and_tokenizer):
    model_dir, tok_path = tiny_model_and_tokenizer
    buggy = ["int a = 1 ;", "return x + 1 ;", "public void m ( ) { }"]
    preds = generate_t5_predictions(
        model_dir,
        tok_path,
        buggy,
        max_seq_length=32,
        max_new_tokens=8,
        batch_size=2,  # 3 inputs, batch 2 -> exercises a full batch + a remainder
        device="cpu",
    )
    assert isinstance(preds, list)
    assert len(preds) == len(buggy)
    assert all(isinstance(p, str) for p in preds)


def test_generate_empty_input_returns_empty(tiny_model_and_tokenizer):
    model_dir, tok_path = tiny_model_and_tokenizer
    preds = generate_t5_predictions(model_dir, tok_path, [], device="cpu")
    assert preds == []
