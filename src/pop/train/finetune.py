"""HF Trainer-based T5 code-refinement finetuning entry point.

Heavy imports (torch/transformers/datasets) are deliberately deferred inside
:func:`run_finetune` so that importing this module (and therefore `pop
--help`) stays fast.

Checkpoint selection is validation-based (``load_best_model_at_end`` on
``eval_loss``) rather than keeping whatever the last epoch produced.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pop.config import FinetuneConfig
    from pop.tokenizer.wrapper import PopTokenizer

logger = logging.getLogger(__name__)


class RefinementDataset:
    """Wraps buggy/fixed pairs as tokenized seq2seq examples (unpadded)."""

    def __init__(self, pairs: list[dict], tokenizer: PopTokenizer, max_length: int):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        pair = self.pairs[idx]
        input_ids = self.tokenizer.encode(pair["buggy"], max_length=self.max_length - 1)
        input_ids = [*input_ids, self.tokenizer.eos_id]
        labels = self.tokenizer.encode(pair["fixed"], max_length=self.max_length - 1)
        labels = [*labels, self.tokenizer.eos_id]
        return {"input_ids": input_ids, "labels": labels}


def run_finetune(cfg: FinetuneConfig, *, report_to: list[str] | None = None) -> Path:
    """Run T5 code-refinement finetuning per ``cfg``; returns the best
    (validation-loss-selected) model directory.

    If ``cfg.output_dir`` already holds a ``checkpoint-*`` from an interrupted
    run, training resumes from the latest one instead of starting over.

    ``report_to`` overrides the experiment-tracking integrations passed to
    ``TrainingArguments``. The default (``None``) enables wandb when
    ``WANDB_API_KEY`` is set; pass ``[]`` to force a fully offline run --
    ``pop smoke`` does, because it is documented as needing no network.
    """
    import torch
    from transformers import Trainer, TrainingArguments

    from pop.data.refinement import load_refinement_pairs, subsample
    from pop.models.t5_factory import create_t5_model
    from pop.tokenizer.wrapper import PopTokenizer
    from pop.train.precision import cap_gpu_memory, scale_micro_batch, training_precision
    from pop.train.span_corruption import DataCollatorForT5

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    tokenizer = PopTokenizer.load(cfg.tokenizer_path)

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

    train_dataset = RefinementDataset(train_pairs, tokenizer, cfg.max_seq_length)
    eval_dataset = RefinementDataset(val_pairs, tokenizer, cfg.max_seq_length)
    collator = DataCollatorForT5(pad_id=tokenizer.pad_id)

    model: Any
    if cfg.pretrained_model_path is not None:
        from transformers import T5ForConditionalGeneration

        model = T5ForConditionalGeneration.from_pretrained(str(cfg.pretrained_model_path))
    else:
        model = create_t5_model(len(tokenizer), cfg.model)

    bf16, fp16 = training_precision()
    cap_gpu_memory()
    micro_batch, grad_accum = scale_micro_batch(cfg.batch_size, cfg.gradient_accumulation_steps)
    if (micro_batch, grad_accum) != (cfg.batch_size, cfg.gradient_accumulation_steps):
        logger.info(
            "GPU batch scaling: micro-batch %d x accum %d (effective batch %d preserved)",
            micro_batch,
            grad_accum,
            micro_batch * grad_accum,
        )
    if report_to is None:
        report_to = ["wandb"] if os.environ.get("WANDB_API_KEY") else []

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validation-based checkpoint selection needs a validation set. `val_pairs` is empty when
    # `train_pairs_file` is set without `val_pairs_file`; asking for eval_strategy="epoch" +
    # load_best_model_at_end on an empty eval dataset is a broken configuration. `run_lora`
    # already gates on the same flag -- this mirrors it.
    has_eval = bool(val_pairs)
    if not has_eval:
        logger.info("No validation pairs: disabling epoch eval and best-checkpoint selection")

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
        eval_dataset=eval_dataset if has_eval else None,
        data_collator=collator,
    )
    from transformers.trainer_utils import get_last_checkpoint

    last_checkpoint = get_last_checkpoint(str(output_dir))
    if last_checkpoint:
        logger.info("Resuming finetuning from checkpoint: %s", last_checkpoint)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    return best_dir
