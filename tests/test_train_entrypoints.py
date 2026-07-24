"""Thin construction/smoke tests for pop.train.pretrain / pop.train.finetune.

Full end-to-end training runs belong to the smoke pipeline; here we only
verify that `run_pretrain`/`run_finetune` wire a tiny model + tiny data
through a real (CPU, 1-epoch) HF Trainer without crashing, using injected
data/tokenizer fixtures so no network access is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pop.config import FinetuneConfig, PretrainConfig, T5ModelConfig  # noqa: E402
from pop.tokenizer.train import train_tokenizer  # noqa: E402
from pop.tokenizer.wrapper import PopTokenizer  # noqa: E402
from pop.train import finetune as finetune_mod  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

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


class _StopAfterTrainers(RuntimeError):
    """Sentinel: unwind run_smoke once both trainers have been called."""


# --- report_to plumbing: `pop smoke` must not ship a run to wandb -------------------------


def test_run_smoke_forces_report_to_empty(tmp_path, monkeypatch):
    """Regression: `pop smoke` is documented as network-free in three places, but the
    trainers enabled the wandb integration purely on WANDB_API_KEY being present -- the
    normal state of an ML practitioner's shell -- so a "local sanity check" uploaded a run.
    """
    from pop.config import SmokeConfig
    from pop.train import smoke as smoke_mod

    seen: dict[str, object] = {}

    def fake_pretrain(cfg, *, report_to=None):
        seen["pretrain"] = report_to
        return tmp_path / "pretrain_final"

    def fake_finetune(cfg, *, report_to=None):
        seen["finetune"] = report_to
        # Stop here deterministically: everything after this point needs a real model on
        # disk, and this test is only about how the trainers were invoked.
        raise _StopAfterTrainers

    monkeypatch.setattr("pop.train.pretrain.run_pretrain", fake_pretrain)
    monkeypatch.setattr("pop.train.finetune.run_finetune", fake_finetune)
    monkeypatch.setenv("WANDB_API_KEY", "pretend-this-is-set")
    monkeypatch.chdir(tmp_path)

    cfg = SmokeConfig(
        corpus_file=REPO_ROOT / "tests" / "fixtures" / "smoke_corpus.txt",
        finetune_pairs_file=REPO_ROOT / "tests" / "fixtures" / "smoke_finetune_pairs.jsonl",
        val_pairs_file=REPO_ROOT / "tests" / "fixtures" / "smoke_val_pairs.jsonl",
        eval_pairs_file=REPO_ROOT / "tests" / "fixtures" / "smoke_eval_pairs.jsonl",
        output_dir=tmp_path / "smoke_out",
    )
    with pytest.raises(_StopAfterTrainers):
        smoke_mod.run_smoke(cfg)

    assert seen["pretrain"] == [], "run_smoke must pass report_to=[] to run_pretrain"
    assert seen["finetune"] == [], "run_smoke must pass report_to=[] to run_finetune"


def test_trainers_default_to_wandb_only_when_the_key_is_set():
    """The opt-out must not change the normal GPU path: report_to is still key-driven."""
    import inspect

    from pop.train.finetune import run_finetune
    from pop.train.pretrain import run_pretrain

    for fn in (run_pretrain, run_finetune):
        param = inspect.signature(fn).parameters["report_to"]
        assert param.default is None, f"{fn.__name__}: default must stay 'auto' (None)"
        assert param.kind is inspect.Parameter.KEYWORD_ONLY


def _spy_training_arguments(monkeypatch) -> dict:
    """Capture the TrainingArguments kwargs run_finetune builds (it imports them lazily)."""
    import transformers

    captured: dict = {}
    real = transformers.TrainingArguments

    def spy(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(transformers, "TrainingArguments", spy)
    return captured


def _pairs_file(path: Path, n: int) -> Path:
    import json

    line = json.dumps({"buggy": "int a = 1 ;", "fixed": "int a = 2 ;"}) + "\n"
    path.write_text(line * n, encoding="utf-8")
    return path


def _no_val_cfg(tmp_path, tiny_tokenizer_path, **overrides) -> FinetuneConfig:
    return FinetuneConfig(
        tokenizer_path=tiny_tokenizer_path,
        train_pairs_file=_pairs_file(tmp_path / "train_pairs.jsonl", 4),
        max_seq_length=32,
        epochs=1,
        batch_size=2,
        warmup_steps=0,
        output_dir=tmp_path / "finetune_out_val",
        model=TINY_MODEL_CFG,
        **overrides,
    )


def test_run_finetune_without_a_validation_set_disables_eval_and_best_selection(
    tmp_path, tiny_tokenizer_path, monkeypatch
):
    """`train_pairs_file` without `val_pairs_file` leaves `val_pairs` empty, but the
    arguments still demanded eval_strategy="epoch" + load_best_model_at_end on eval_loss
    over an empty eval dataset. `run_lora` already gates on a has_eval flag; this mirrors it.

    (transformers 5.x happens to tolerate the old combination rather than raising, so this
    asserts the configuration directly instead of relying on a crash.)
    """
    captured = _spy_training_arguments(monkeypatch)

    best_dir = finetune_mod.run_finetune(_no_val_cfg(tmp_path, tiny_tokenizer_path))

    assert captured["eval_strategy"] == "no"
    assert captured["load_best_model_at_end"] is False
    assert captured["metric_for_best_model"] is None
    assert best_dir.is_dir() and (best_dir / "config.json").is_file()


def test_run_finetune_with_a_validation_set_still_selects_the_best_checkpoint(
    tmp_path, tiny_tokenizer_path, monkeypatch
):
    """The gate must not change the real training path, which is validation-selected."""
    captured = _spy_training_arguments(monkeypatch)

    cfg = _no_val_cfg(
        tmp_path,
        tiny_tokenizer_path,
        val_pairs_file=_pairs_file(tmp_path / "val_pairs.jsonl", 2),
    )
    finetune_mod.run_finetune(cfg)

    assert captured["eval_strategy"] == "epoch"
    assert captured["load_best_model_at_end"] is True
    assert captured["metric_for_best_model"] == "eval_loss"
