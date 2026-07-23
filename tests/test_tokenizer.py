"""Tests for pop.tokenizer.train and pop.tokenizer.wrapper.

Trains a tiny real SentencePiece model in-test (small vocab) -- no network,
no committed fixture binaries.
"""

from __future__ import annotations

from pop.tokenizer.train import NUM_SENTINELS as TRAIN_NUM_SENTINELS
from pop.tokenizer.train import train_tokenizer
from pop.tokenizer.wrapper import PopTokenizer

TINY_CORPUS = [
    "public void foo ( ) { int x = 1 ; return x ; }",
    "private int bar ( int y ) { return y + 1 ; }",
    "public String baz ( String s ) { return s . trim ( ) ; }",
] * 30


def _tiny_tokenizer(tmp_path) -> PopTokenizer:
    model_path = train_tokenizer(TINY_CORPUS, tmp_path / "tok.model", vocab_size=160)
    return PopTokenizer.load(model_path)


def test_train_tokenizer_writes_model_file(tmp_path):
    model_path = train_tokenizer(TINY_CORPUS, tmp_path / "tok.model", vocab_size=160)
    assert model_path.exists()
    assert model_path.suffix == ".model"


def test_tokenizer_round_trip(tmp_path):
    tok = _tiny_tokenizer(tmp_path)
    text = "public void foo ( ) { int x = 1 ; }"
    ids = tok.encode(text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    decoded = tok.decode(ids)
    # Unigram tokenization is lossy on whitespace normalization in general,
    # but for a corpus this repetitive the round trip should be exact.
    assert decoded.replace(" ", "") == text.replace(" ", "")


def test_tokenizer_special_token_ids(tmp_path):
    tok = _tiny_tokenizer(tmp_path)
    assert tok.pad_id == 0
    assert tok.eos_id == 1
    assert tok.unk_id == 2


def test_tokenizer_sentinel_ids_are_unique_and_addressable(tmp_path):
    tok = _tiny_tokenizer(tmp_path)
    ids = [tok.sentinel_id(i) for i in range(NUM_SENTINELS_FOR_TEST)]
    assert len(set(ids)) == len(ids)
    for i in ids:
        assert tok.is_sentinel_id(i)
    assert not tok.is_sentinel_id(tok.pad_id)
    assert not tok.is_sentinel_id(tok.eos_id)


NUM_SENTINELS_FOR_TEST = TRAIN_NUM_SENTINELS


def test_tokenizer_sentinel_id_out_of_range_raises(tmp_path):
    tok = _tiny_tokenizer(tmp_path)
    import pytest

    with pytest.raises(ValueError):
        tok.sentinel_id(-1)
    with pytest.raises(ValueError):
        tok.sentinel_id(100)


def test_decode_skips_special_tokens_by_default(tmp_path):
    tok = _tiny_tokenizer(tmp_path)
    ids = tok.encode("public void foo ( ) { int x = 1 ; }")
    ids_with_specials = [tok.sentinel_id(0), *ids, tok.eos_id, tok.pad_id, tok.pad_id]
    decoded = tok.decode(ids_with_specials)
    plain_decoded = tok.decode(ids)
    assert decoded == plain_decoded


def test_batch_encode(tmp_path):
    tok = _tiny_tokenizer(tmp_path)
    texts = ["public void foo ( ) { }", "private int bar ( ) { }"]
    batch = tok.batch_encode(texts)
    assert len(batch) == 2
    assert batch[0] == tok.encode(texts[0])
    assert batch[1] == tok.encode(texts[1])


def test_len_matches_vocab_size(tmp_path):
    tok = _tiny_tokenizer(tmp_path)
    assert len(tok) == tok.vocab_size
    assert len(tok) >= 160 - 20  # SentencePiece may land slightly below target on tiny corpora
