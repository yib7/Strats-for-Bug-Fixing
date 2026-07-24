"""Pydantic v2 configuration models for `pop pretrain` / `pop finetune` YAML configs.

Default hyperparameters:
vocab 16384, max_seq 512, T5 d_model 512 / d_ff 2048 / d_kv 64 / 8 heads /
6+6 layers, corruption 0.15, pretrain lr 1e-4, finetune lr 5e-5 / warmup 500.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class T5ModelConfig(BaseModel):
    d_model: int = 512
    d_ff: int = 2048
    d_kv: int = 64
    num_heads: int = 8
    num_layers: int = 6
    num_decoder_layers: int = 6


def _load_yaml(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data or {}


class PretrainConfig(BaseModel):
    seed: int = 42
    tokenizer_path: Path
    corpus_num_samples: int = 50000
    corpus_file: Path | None = None
    """Optional path to a fixture corpus file (see `pop.data.corpus.load_corpus_file`).

    When set, `run_pretrain` loads records from this file instead of the injectable loaders'
    default (network) HuggingFace path. Used by `pop smoke`; also usable for any offline run
    with a pre-materialized corpus.
    """
    max_seq_length: int = 512
    corruption_rate: float = 0.15
    mean_span_length: int = 3
    epochs: int = 3
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    """Micro-batches accumulated per optimizer step. Effective batch =
    ``batch_size * gradient_accumulation_steps``; used to keep the notebook's
    effective batch (64) while fitting a smaller per-device batch in GPU memory."""
    lr: float = 1e-4
    save_epochs: list[int] = Field(default_factory=lambda: [1, 3, 10])
    output_dir: Path = Path("outputs/pretrain")
    model: T5ModelConfig = Field(default_factory=T5ModelConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PretrainConfig:
        return cls(**_load_yaml(path))


class FinetuneConfig(BaseModel):
    seed: int = 42
    tokenizer_path: Path
    pretrained_model_path: Path | None = None
    train_split: str = "train"
    validation_split: str = "validation"
    train_pairs_file: Path | None = None
    """Optional path to a fixture JSONL pairs file (see `pop.data.refinement.load_pairs_file`).

    When set, `run_finetune` loads the training pairs from this file instead of
    `load_refinement_pairs(train_split)`. Used by `pop smoke`.
    """
    val_pairs_file: Path | None = None
    """Same as `train_pairs_file` but for the validation split; only consulted when
    `train_pairs_file` is also set."""
    train_n: int | None = None
    max_seq_length: int = 512
    epochs: int = 3
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    """Micro-batches accumulated per optimizer step. Effective batch =
    ``batch_size * gradient_accumulation_steps``; used to keep the notebook's
    effective batch (64) while fitting a smaller per-device batch in GPU memory."""
    lr: float = 5e-5
    warmup_steps: int = 500
    output_dir: Path = Path("outputs/finetune")
    model: T5ModelConfig = Field(default_factory=T5ModelConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> FinetuneConfig:
        return cls(**_load_yaml(path))


class RagConfig(BaseModel):
    """Config for `pop rag`: retriever choice, k, model, and data splits.

    Leakage guard: `kb_split` should be "train" -- the exemplar knowledge base
    is built from this split and must never see `split` (the evaluation
    split) pairs. The CLI enforces this (refuses to run unless `kb_split ==
    "train"`, overridable via `--allow-non-train-kb`); this config model does
    not enforce it itself.
    """

    seed: int = 42
    retriever: Literal["bm25", "codebert"] = "bm25"
    k: Literal[0, 1, 3, 5] = 3
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    codebert_model_name: str = "microsoft/codebert-base"
    split: str = "test"
    kb_split: str = "train"
    output_dir: Path = Path("outputs/rag")
    gen_kwargs: dict = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RagConfig:
        return cls(**_load_yaml(path))


class LoRAConfig(BaseModel):
    """Config for `pop lora`: PEFT LoRA-finetunes a causal LM (Qwen2.5-Coder) on the CodeXGLUE
    refinement pairs -- the "LoRA bridge" arm between the from-scratch T5 pretrain+finetune arm
    and the zero/few-shot RAG-prompting arm. Field defaults mirror the draft
    `configs/lora_qwen.yaml`.
    """

    seed: int = 42
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    train_split: str = "train"
    validation_split: str = "validation"
    train_pairs_file: Path | None = None
    """Optional path to a fixture JSONL pairs file (see `pop.data.refinement.load_pairs_file`).

    When set, `run_lora` loads the training pairs from this file instead of
    `load_refinement_pairs(train_split)`. Used for a CPU smoke -- no network, no GPU."""
    val_pairs_file: Path | None = None
    """Same as `train_pairs_file` but for the validation split; only consulted when
    `train_pairs_file` is also set."""
    train_n: int | None = None
    max_seq_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    epochs: int = 3
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    """Micro-batches accumulated per optimizer step. Effective batch =
    ``batch_size * gradient_accumulation_steps``; kept for parity with Pretrain/FinetuneConfig."""
    lr: float = 1e-4
    warmup_steps: int = 100
    output_dir: Path = Path("outputs/lora_qwen")

    @classmethod
    def from_yaml(cls, path: str | Path) -> LoRAConfig:
        return cls(**_load_yaml(path))


_SMOKE_MODEL_DEFAULTS = {
    "d_model": 64,
    "d_ff": 128,
    "d_kv": 16,
    "num_heads": 2,
    "num_layers": 2,
    "num_decoder_layers": 2,
}


class SmokeConfig(BaseModel):
    """Config for `pop smoke`: an end-to-end, CPU-minutes, fixture-only dry run of the whole
    tokenizer -> pretrain -> finetune -> eval pipeline. All data comes from the committed
    `tests/fixtures/smoke_*` files (built by `scripts/build_smoke_fixtures.py`); no network
    access.
    """

    seed: int = 42
    corpus_file: Path = Path("tests/fixtures/smoke_corpus.txt")
    finetune_pairs_file: Path = Path("tests/fixtures/smoke_finetune_pairs.jsonl")
    val_pairs_file: Path = Path("tests/fixtures/smoke_val_pairs.jsonl")
    eval_pairs_file: Path = Path("tests/fixtures/smoke_eval_pairs.jsonl")

    vocab_size: int = 512
    max_seq_length: int = 128
    model: T5ModelConfig = Field(default_factory=lambda: T5ModelConfig(**_SMOKE_MODEL_DEFAULTS))

    corpus_num_samples: int = 200
    pretrain_epochs: int = 1
    pretrain_batch_size: int = 8
    pretrain_lr: float = 1e-4

    finetune_epochs: int = 1
    finetune_batch_size: int = 8
    finetune_lr: float = 5e-5
    finetune_warmup_steps: int = 10

    max_new_tokens: int = 64

    output_dir: Path = Path("outputs/smoke")
    results_name: str = "smoke_local"
    """Run name for `results/<results_name>.json`.

    Deliberately NOT "smoke": `results/smoke.json` is a committed, published result that
    `docs/report.md` and `tests/test_smoke.py` read. A local `pop smoke` writes the
    gitignored `results/smoke_local.json` instead, so the documented first command a
    visitor runs can never modify tracked data."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> SmokeConfig:
        return cls(**_load_yaml(path))
