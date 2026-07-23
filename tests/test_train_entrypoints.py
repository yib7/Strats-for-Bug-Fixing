"""Thin construction/smoke tests for pop.train.pretrain / pop.train.finetune.

Full end-to-end training runs belong to the smoke pipeline; here we only
verify that `run_pretrain`/`run_finetune` wire a tiny model + tiny data
through a real (CPU, 1-epoch) HF Trainer without crashing, using injected
data/tokenizer fixtures so no network access is needed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pop.config import FinetuneConfig, PretrainConfig, T5ModelConfig  # noqa: E402
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
def tiny_tokenizer_path(tmp_path):
    return train_tokenizer(TINY_CORPUS, tmp_path / "tok.model", vocab_size=160)


def test_run_pretrain_two_steps(tmp_path, tiny_tokenizer_path, monkeypatch):
    from pop.train import pretrain as pretrain_mod

    cfg = PretrainConfig(
        tokenizer_path=tiny_tokenizer_path,
        corpus_num_samples=8,
        max_seq_length=32,
        epochs=1,
        batch_size=4,
        output_dir=tmp_path / "pretrain_out",
        model=TINY_MODEL_CFG,
    )

    # Patch the source of load_pretraining_corpus; run_pretrain imports it
    # lazily *inside the function call*, so the monkeypatch below (applied
    # before run_pretrain executes) is picked up.
    import pop.data.corpus as corpus_mod

    monkeypatch.setattr(
        corpus_mod, "load_pretraining_corpus", lambda num_samples, seed=42: TINY_CORPUS[:8]
    )

    final_dir = pretrain_mod.run_pretrain(cfg)
    assert final_dir.exists()
    assert (final_dir / "config.json").exists()


def test_run_finetune_two_steps(tmp_path, tiny_tokenizer_path, monkeypatch):
    from pop.train import finetune as finetune_mod

    pairs = [{"buggy": "int a = 1 ;", "fixed": "int a = 2 ;"} for _ in range(4)]

    import pop.data.refinement as refinement_mod

    monkeypatch.setattr(refinement_mod, "load_refinement_pairs", lambda split: pairs)

    cfg = FinetuneConfig(
        tokenizer_path=tiny_tokenizer_path,
        max_seq_length=32,
        epochs=1,
        batch_size=2,
        warmup_steps=0,
        output_dir=tmp_path / "finetune_out",
        model=TINY_MODEL_CFG,
    )

    best_dir = finetune_mod.run_finetune(cfg)
    assert best_dir.exists()
    assert (best_dir / "config.json").exists()


def test_tokenizer_fixture_matches_wrapper(tiny_tokenizer_path):
    tok = PopTokenizer.load(tiny_tokenizer_path)
    assert tok.pad_id == 0
    assert tok.eos_id == 1


def test_run_pretrain_keeps_milestone_and_latest_checkpoints(
    tmp_path, tiny_tokenizer_path, monkeypatch, caplog
):
    """3 epochs with milestone [1]: epoch-1 and the latest (epoch-3) checkpoints
    survive, the in-between epoch-2 checkpoint is pruned, and a re-invocation
    resumes (immediately completing) instead of retraining from scratch."""
    import logging

    from pop.train import pretrain as pretrain_mod

    cfg = PretrainConfig(
        tokenizer_path=tiny_tokenizer_path,
        corpus_num_samples=8,
        max_seq_length=32,
        epochs=3,
        batch_size=4,
        save_epochs=[1],
        output_dir=tmp_path / "pretrain_out",
        model=TINY_MODEL_CFG,
    )

    import pop.data.corpus as corpus_mod

    monkeypatch.setattr(
        corpus_mod, "load_pretraining_corpus", lambda num_samples, seed=42: TINY_CORPUS[:8]
    )

    pretrain_mod.run_pretrain(cfg)

    # 8 samples / batch 4 = 2 optimizer steps per epoch -> epochs 1/2/3 save at
    # steps 2/4/6; epoch 2's step-4 checkpoint must have been pruned.
    steps = sorted(
        int(p.name.rsplit("-", 1)[-1]) for p in (tmp_path / "pretrain_out").glob("checkpoint-*")
    )
    assert steps == [2, 6]

    caplog.set_level(logging.INFO, logger="pop.train.pretrain")
    final_dir = pretrain_mod.run_pretrain(cfg)
    assert final_dir.exists()
    assert "Resuming pretraining from checkpoint" in caplog.text


def test_run_pretrain_writes_stable_epoch_dir(tmp_path, tiny_tokenizer_path, monkeypatch):
    """A milestone epoch writes a stable, loadable `epoch-{N}` model dir (so a
    ptcompute finetune config can point `pretrained_model_path` at it) without
    disturbing the step-numbered `checkpoint-*` pruning."""
    from transformers import T5ForConditionalGeneration

    from pop.train import pretrain as pretrain_mod

    cfg = PretrainConfig(
        tokenizer_path=tiny_tokenizer_path,
        corpus_num_samples=8,
        max_seq_length=32,
        epochs=1,
        batch_size=4,
        save_epochs=[1],
        output_dir=tmp_path / "pretrain_out",
        model=TINY_MODEL_CFG,
    )

    import pop.data.corpus as corpus_mod

    monkeypatch.setattr(
        corpus_mod, "load_pretraining_corpus", lambda num_samples, seed=42: TINY_CORPUS[:8]
    )

    pretrain_mod.run_pretrain(cfg)

    epoch_dir = tmp_path / "pretrain_out" / "epoch-1"
    assert (epoch_dir / "config.json").exists()
    # The stable dir must be loadable as a model (this is what a ptcompute config's
    # `pretrained_model_path` feeds to `T5ForConditionalGeneration.from_pretrained`).
    model = T5ForConditionalGeneration.from_pretrained(str(epoch_dir))
    assert model.config.num_layers == TINY_MODEL_CFG.num_layers
    # `epoch-*` is not a `checkpoint-*` dir, so the pruning glob leaves it alone.
    assert not epoch_dir.name.startswith("checkpoint-")


def test_run_finetune_resumes_from_checkpoint(tmp_path, tiny_tokenizer_path, monkeypatch, caplog):
    """A second run over the same output_dir with more epochs picks up from the
    latest checkpoint instead of restarting."""
    import json
    import logging

    from pop.train import finetune as finetune_mod

    pairs = [{"buggy": "int a = 1 ;", "fixed": "int a = 2 ;"} for _ in range(4)]

    import pop.data.refinement as refinement_mod

    monkeypatch.setattr(refinement_mod, "load_refinement_pairs", lambda split: pairs)

    out = tmp_path / "finetune_out"

    def make_cfg(epochs: int) -> FinetuneConfig:
        return FinetuneConfig(
            tokenizer_path=tiny_tokenizer_path,
            max_seq_length=32,
            epochs=epochs,
            batch_size=2,
            warmup_steps=0,
            output_dir=out,
            model=TINY_MODEL_CFG,
        )

    finetune_mod.run_finetune(make_cfg(epochs=1))
    assert (out / "checkpoint-2").exists()  # 4 pairs / batch 2 = 2 steps per epoch

    caplog.set_level(logging.INFO, logger="pop.train.finetune")
    finetune_mod.run_finetune(make_cfg(epochs=2))
    assert "Resuming finetuning from checkpoint" in caplog.text

    state = json.loads((out / "checkpoint-4" / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["global_step"] == 4
