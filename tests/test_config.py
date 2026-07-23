"""Tests for pop.config (pydantic v2 YAML config models)."""

from __future__ import annotations

from pathlib import Path

from pop.config import FinetuneConfig, LoRAConfig, PretrainConfig, RagConfig, T5ModelConfig


def test_pretrain_config_defaults_match_notebook_hyperparams():
    cfg = PretrainConfig(tokenizer_path="tok.model")
    assert cfg.corruption_rate == 0.15
    assert cfg.max_seq_length == 512
    assert cfg.epochs == 3
    assert cfg.batch_size == 64
    assert cfg.lr == 1e-4
    assert cfg.save_epochs == [1, 3, 10]
    assert cfg.model == T5ModelConfig(
        d_model=512, d_ff=2048, d_kv=64, num_heads=8, num_layers=6, num_decoder_layers=6
    )


def test_finetune_config_defaults_match_notebook_hyperparams():
    cfg = FinetuneConfig(tokenizer_path="tok.model")
    assert cfg.epochs == 3
    assert cfg.batch_size == 64
    assert cfg.lr == 5e-5
    assert cfg.warmup_steps == 500


def test_gradient_accumulation_steps_defaults_to_one_and_is_settable(tmp_path):
    # Default keeps effective batch == batch_size (backward-compatible).
    assert PretrainConfig(tokenizer_path="tok.model").gradient_accumulation_steps == 1
    assert FinetuneConfig(tokenizer_path="tok.model").gradient_accumulation_steps == 1

    yaml_path = tmp_path / "pretrain.yaml"
    yaml_path.write_text(
        "tokenizer_path: tok.model\nbatch_size: 8\ngradient_accumulation_steps: 8\n",
        encoding="utf-8",
    )
    cfg = PretrainConfig.from_yaml(yaml_path)
    assert cfg.batch_size == 8
    assert cfg.gradient_accumulation_steps == 8  # effective batch 64


def test_pretrain_config_from_yaml(tmp_path):
    yaml_path = tmp_path / "pretrain.yaml"
    yaml_path.write_text(
        "tokenizer_path: tok.model\ncorpus_num_samples: 100\nepochs: 1\nmodel:\n  d_model: 32\n",
        encoding="utf-8",
    )
    cfg = PretrainConfig.from_yaml(yaml_path)
    assert cfg.corpus_num_samples == 100
    assert cfg.epochs == 1
    assert cfg.model.d_model == 32
    assert cfg.tokenizer_path == Path("tok.model")


def test_finetune_config_from_yaml(tmp_path):
    yaml_path = tmp_path / "finetune.yaml"
    yaml_path.write_text(
        "tokenizer_path: tok.model\ntrain_n: 1000\nwarmup_steps: 10\n",
        encoding="utf-8",
    )
    cfg = FinetuneConfig.from_yaml(yaml_path)
    assert cfg.train_n == 1000
    assert cfg.warmup_steps == 10


def test_pretrain_config_requires_tokenizer_path():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PretrainConfig()


def test_rag_config_defaults():
    cfg = RagConfig()
    assert cfg.retriever == "bm25"
    assert cfg.k == 3
    assert cfg.model_name == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert cfg.kb_split == "train"
    assert cfg.split == "test"


def test_rag_config_from_yaml(tmp_path):
    yaml_path = tmp_path / "rag.yaml"
    yaml_path.write_text(
        "retriever: codebert\nk: 5\nmodel_name: some/model\n",
        encoding="utf-8",
    )
    cfg = RagConfig.from_yaml(yaml_path)
    assert cfg.retriever == "codebert"
    assert cfg.k == 5
    assert cfg.model_name == "some/model"


def test_rag_config_rejects_invalid_k():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RagConfig(k=2)


def test_rag_config_rejects_invalid_retriever():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RagConfig(retriever="tfidf")


def test_lora_config_from_yaml_parses_committed_draft():
    cfg = LoRAConfig.from_yaml("configs/lora_qwen.yaml")
    assert cfg.seed == 42
    assert cfg.base_model.startswith("Qwen")
    assert cfg.train_split == "train"
    assert cfg.validation_split == "validation"
    assert cfg.max_seq_length == 512
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 32
    assert cfg.lora_dropout == 0.05
    assert cfg.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert cfg.epochs == 3
    # Micro-batch 4 x accum 4 = effective batch 16, sized so the 1.5B Qwen fits a 40 GB A100.
    assert cfg.batch_size == 4
    assert cfg.gradient_accumulation_steps == 4
    assert cfg.lr == 1e-4
    assert cfg.warmup_steps == 100
    assert cfg.output_dir == Path("outputs/lora_qwen")
    assert cfg.train_n is None
    assert cfg.train_pairs_file is None
    assert cfg.val_pairs_file is None


def test_lora_config_defaults():
    cfg = LoRAConfig()
    assert cfg.base_model == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert cfg.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert cfg.output_dir == Path("outputs/lora_qwen")
