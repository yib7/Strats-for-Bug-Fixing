"""HF Trainer + PEFT LoRA code-refinement finetuning entry point for a causal LM.

The "LoRA bridge" arm: cheaply adapt a pretrained large causal LM (Qwen2.5-Coder)
on the CodeXGLUE refinement pairs, as a middle ground between the from-scratch T5
pretrain+finetune arm and the zero/few-shot RAG-prompting arm.

Mirrors :mod:`pop.train.finetune` in structure: a ``run_lora(cfg) -> Path`` entry
point, injectable/fixture data loading, the shared precision + VRAM-cap machinery
(:mod:`pop.train.precision`), and resume-from-latest-``checkpoint-*``. Heavy imports
(torch/transformers/peft/datasets) are deferred inside :func:`run_lora` so importing
this module (and therefore ``pop --help``) stays fast.

Supervised formatting (SFT): each ``{buggy, fixed}`` pair becomes one causal-LM
example -- an instruction+buggy *prompt* followed by the ``fixed`` *completion*, with
the prompt tokens masked to ``-100`` so loss is computed on the fix only. The prompt
uses the tokenizer's chat template when it has one (real Qwen), else a plain
``buggy -> fixed`` text template (tiny offline test model).

Inference lives here too: :func:`build_lora_prompt` is the single source of truth for
the prompt text (so training and generation cannot drift), and
:func:`build_lora_generator` mirrors :func:`pop.rag.generate.build_generator` --
load the adapted model once, return a reusable batched-greedy callable -- to
produce predictions the ``pop eval`` stack consumes (:func:`generate_lora_fixes`
is its one-shot wrapper).
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pop.config import LoRAConfig

logger = logging.getLogger(__name__)

INSTRUCTION = "Fix the bug in this Java method."
IGNORE_INDEX = -100


def build_lora_prompt(tokenizer: Any, buggy: str) -> str:
    """The instruction+buggy prompt *text* a LoRA model is asked to complete with the fix.

    Single source of truth for the LoRA prompt, shared by training
    (:func:`_build_prompt_ids`, which tokenizes this string) and inference (the
    ``pop lora-generate`` CLI, which feeds this string to
    :func:`generate_lora_fixes`). Because both build the prompt here, the model
    is always asked at inference to complete the exact prefix it was trained on
    -- train and inference cannot drift.

    Uses the tokenizer's chat template with ``add_generation_prompt=True`` when
    it has one (real Qwen2.5-Coder-Instruct), else a plain
    ``### Buggy Java:/### Fixed Java:`` text template (the tiny offline test model).
    """
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": f"{INSTRUCTION}\n\n{buggy}"}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{INSTRUCTION}\n\n### Buggy Java:\n{buggy}\n\n### Fixed Java:\n"


def _build_prompt_ids(tokenizer: Any, buggy: str) -> list[int]:
    """Tokenize the shared LoRA prompt (:func:`build_lora_prompt`) for prompt-masked SFT.

    A chat template already embeds the model's special tokens, so it is
    tokenized with ``add_special_tokens=False`` (re-adding would e.g. double a
    BOS); the plain text template is raw text and takes ``add_special_tokens=True``.
    """
    prompt_text = build_lora_prompt(tokenizer, buggy)
    add_special = not getattr(tokenizer, "chat_template", None)
    return tokenizer(prompt_text, add_special_tokens=add_special)["input_ids"]


class CausalRefinementDataset:
    """Buggy/fixed pairs as prompt-masked causal-LM SFT examples.

    Each item is ``{"input_ids": prompt+completion, "labels": [-100]*prompt + completion}``.
    The completion (``fixed`` code + EOS) is always preserved within ``max_length``; the
    prompt is truncated first if the pair would otherwise overflow, so every example
    keeps at least one supervised token (no all-``-100`` rows -> no NaN loss).
    """

    def __init__(self, pairs: list[dict], tokenizer: Any, max_length: int):
        eos_id = tokenizer.eos_token_id
        self.examples: list[dict[str, list[int]]] = []
        for pair in pairs:
            prompt_ids = _build_prompt_ids(tokenizer, pair["buggy"])
            completion_ids = tokenizer(pair["fixed"], add_special_tokens=False)["input_ids"]
            if eos_id is not None:
                # Reserve room for EOS *before* appending it: truncating after the
                # append would drop the trailing EOS on an over-long `fixed`, and the
                # model would never learn to stop on those examples. Mirrors finetune.py.
                completion_ids = [*completion_ids[: max_length - 1], eos_id]
            else:
                completion_ids = completion_ids[:max_length]
            prompt_budget = max(0, max_length - len(completion_ids))
            prompt_ids = prompt_ids[:prompt_budget]
            input_ids = [*prompt_ids, *completion_ids]
            labels = [IGNORE_INDEX] * len(prompt_ids) + list(completion_ids)
            self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


class CausalCollator:
    """Right-pad ``input_ids`` (with ``pad_id``) and ``labels`` (with ``-100``) and
    build the matching attention mask."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids = f["input_ids"]
            lab = f["labels"]
            pad_len = max_len - len(ids)
            input_ids.append([*ids, *([self.pad_id] * pad_len)])
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append([*lab, *([IGNORE_INDEX] * pad_len)])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def run_lora(cfg: LoRAConfig) -> Path:
    """PEFT LoRA-finetune a causal LM per ``cfg``; returns the saved adapter directory.

    ``cfg.base_model`` may be a local directory (tests point it at a tiny offline model)
    or a HuggingFace hub name (real runs use the Qwen checkpoint). If ``cfg.output_dir``
    already holds a ``checkpoint-*`` from an interrupted run, training resumes from the
    latest one. The trained adapter (``adapter_config.json`` + adapter weights) is written
    to ``<output_dir>/best`` when a validation set is available (best selected on
    ``eval_loss``), else ``<output_dir>/final``.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    from pop.data.refinement import load_refinement_pairs, subsample
    from pop.train.precision import cap_gpu_memory, training_precision

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.train_pairs_file is not None:
        from pop.data.refinement import load_pairs_file

        train_pairs = load_pairs_file(cfg.train_pairs_file)
        val_pairs = load_pairs_file(cfg.val_pairs_file) if cfg.val_pairs_file is not None else []
    else:
        train_pairs = load_refinement_pairs(cfg.train_split)
        val_pairs = load_refinement_pairs(cfg.validation_split)
    if cfg.train_n is not None:
        train_pairs = subsample(train_pairs, cfg.train_n, seed=cfg.seed)
    logger.info(
        "Loaded %d train / %d validation refinement pairs", len(train_pairs), len(val_pairs)
    )

    train_dataset = CausalRefinementDataset(train_pairs, tokenizer, cfg.max_seq_length)
    eval_dataset = (
        CausalRefinementDataset(val_pairs, tokenizer, cfg.max_seq_length) if val_pairs else None
    )
    pad_id = tokenizer.pad_token_id
    collator = CausalCollator(pad_id if pad_id is not None else 0)

    model: Any = AutoModelForCausalLM.from_pretrained(cfg.base_model)
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    bf16, fp16 = training_precision()
    cap_gpu_memory()
    # Unlike the T5 arms, the LoRA arm does NOT auto-scale the micro-batch. scale_micro_batch's
    # VRAM table is calibrated on the 52M-param T5; the 1.5B Qwen (with a ~152k-token vocab) has a
    # far heavier per-sample memory profile at seq 512, so honouring that table OOMs the A100 (the
    # cross-entropy logits tensor alone blows past the 0.85 memory cap on a full-length batch). The
    # config specifies a memory-safe micro-batch x grad-accum directly instead; feed it verbatim.
    micro_batch, grad_accum = cfg.batch_size, cfg.gradient_accumulation_steps
    report_to = ["wandb"] if os.environ.get("WANDB_API_KEY") else []

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    has_eval = eval_dataset is not None
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=micro_batch,
        per_device_eval_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        seed=cfg.seed,
        bf16=bf16,
        fp16=fp16,
        report_to=report_to,
        eval_strategy="epoch" if has_eval else "no",
        save_strategy="epoch",
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    last_checkpoint = get_last_checkpoint(str(output_dir))
    if last_checkpoint:
        logger.info("Resuming LoRA finetuning from checkpoint: %s", last_checkpoint)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    save_dir = output_dir / ("best" if has_eval else "final")
    # PEFT model's save_pretrained writes adapter_config.json + adapter weights only.
    trainer.model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    return save_dir


def _default_lora_generator(base_model: str, adapter_dir: str, gen_kwargs: dict) -> Any:
    """Load base model + PEFT adapter ONCE; return a prompts->texts closure.

    Mirrors :func:`pop.rag.generate._default_transformers_generator`: batched
    decoder-only generation needs a pad token and left padding; Qwen's stale
    ``generation_config.max_length`` default is cleared and HF logging set to
    errors-only so a full-test-split run doesn't print a warning per batch.
    ``device_map="auto"`` places the model on the GPU when there is one --
    passing a pre-built model object to ``pipeline`` skips its own device
    placement, so loading without it would silently generate on CPU.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    pipe.model.generation_config.max_length = None
    tokenizer.clean_up_tokenization_spaces = False

    def generate(prompts: list[str]) -> list[str]:
        # return_full_text=False strips the prompt, so the caller can run
        # pop.rag.prompt.extract_fix on the completion directly.
        raw_results = pipe(prompts, return_full_text=False, **gen_kwargs)
        texts: list[str] = []
        for result in raw_results:
            item = result[0] if isinstance(result, list) else result
            texts.append(item["generated_text"])
        return texts

    return generate


def build_lora_generator(
    base_model: str,
    adapter_dir: str,
    *,
    max_new_tokens: int = 256,
    batch_size: int = 16,
    generator_factory: Any = None,
    **gen_kwargs: object,
) -> Any:
    """Build a reusable prompts->completions callable for a LoRA-adapted causal LM.

    The LoRA twin of :func:`pop.rag.generate.build_generator`'s transformers path:
    the model/adapter are loaded **once** and the returned callable is fed prompt
    chunks (e.g. by ``generate_with_resume``) without reloading per chunk. Greedy
    decoding (``do_sample=False``) keeps predictions reproducible; ``batch_size``
    batches generation so the 6.5k-sample test split isn't one-prompt-at-a-time.
    ``gen_kwargs`` overrides these defaults; ``generator_factory`` is injectable
    so tests exercise dispatch without torch/transformers/peft or a download.
    """
    kwargs: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "do_sample": False,
    }
    kwargs.update(gen_kwargs)
    factory = generator_factory or _default_lora_generator
    return factory(base_model, adapter_dir, kwargs)


def generate_lora_fixes(
    prompts: list[str],
    base_model: str,
    adapter_dir: str,
    *,
    backend_fn: Any = None,
    **gen_kwargs: object,
) -> list[str]:
    """One-shot batch generation for `prompts` with a LoRA-adapted causal LM.

    Thin wrapper over :func:`build_lora_generator` for callers that generate a
    single prompt list (the execbench adapter, tests). An injected
    ``backend_fn(prompts, base_model, adapter_dir, **gen_kwargs)`` overrides it
    entirely (tests pass a fake -- no model load, no download). Long resumable
    runs (``pop lora-generate``) use :func:`build_lora_generator` directly so
    the model loads once across chunks.

    Returns the raw generated text per prompt; the CALLER applies
    :func:`pop.rag.prompt.extract_fix` to turn it into extracted code.
    """
    if backend_fn is not None:
        return backend_fn(prompts, base_model, adapter_dir, **gen_kwargs)
    return build_lora_generator(base_model, adapter_dir, **gen_kwargs)(prompts)
