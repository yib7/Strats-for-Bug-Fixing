"""HF Trainer-based T5 span-corruption pretraining entry point.

Heavy imports (torch/transformers/datasets) are deliberately deferred inside
:func:`run_pretrain` so that importing this module (and therefore `pop
--help`) stays fast.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pop.config import PretrainConfig

logger = logging.getLogger(__name__)


def _save_at_epochs_callback(save_epochs: list[int]):
    from transformers import TrainerCallback

    class SaveAtEpochsCallback(TrainerCallback):
        """Checkpoint at every epoch end, with bounded disk usage.

        Milestone epochs (``save_epochs``, e.g. 1/3/10) are kept permanently
        for later curve analysis. Every other epoch keeps only the *latest*
        checkpoint, which exists so an interrupted run (Colab session limit,
        crash, reboot) can resume from the last completed epoch instead of
        the last milestone; older non-milestone checkpoints are pruned on
        each save so a Google Drive workspace stays a few GB, not tens.

        At each milestone epoch the model is *also* written to a stable,
        loadable ``epoch-{N}`` dir (config.json + weights, like ``final/``),
        so a finetune config can point ``pretrained_model_path`` at a fixed
        path (``outputs/pretrain/epoch-{N}``) instead of a step-numbered
        ``checkpoint-*`` whose step count depends on the corpus/batch size.
        These dirs are named ``epoch-*`` on purpose so the ``checkpoint-*``
        pruning glob above never touches them.
        """

        def on_epoch_end(self, args, state, control, **kwargs):
            control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            import shutil

            epochs = int(args.num_train_epochs)
            if not state.max_steps or epochs <= 0:
                return control
            steps_per_epoch = state.max_steps / epochs
            keep_steps = {round(epoch * steps_per_epoch) for epoch in save_epochs}
            for ckpt in Path(args.output_dir).glob("checkpoint-*"):
                try:
                    step = int(ckpt.name.rsplit("-", 1)[-1])
                except ValueError:
                    continue
                if step != state.global_step and step not in keep_steps:
                    shutil.rmtree(ckpt, ignore_errors=True)

            current_epoch = round(state.global_step / steps_per_epoch)
            model = kwargs.get("model")
            if current_epoch in save_epochs and model is not None:
                epoch_dir = Path(args.output_dir) / f"epoch-{current_epoch}"
                model.save_pretrained(str(epoch_dir))
            return control

    return SaveAtEpochsCallback()


def run_pretrain(cfg: PretrainConfig) -> Path:
    """Run T5 span-corruption pretraining per ``cfg`` and return the final
    model directory.

    If ``cfg.output_dir`` already holds a ``checkpoint-*`` from an interrupted
    run, training resumes from the latest one instead of starting over.
    """
    import random

    import torch
    from transformers import Trainer, TrainingArguments

    from pop.data.corpus import load_pretraining_corpus
    from pop.models.t5_factory import create_t5_model
    from pop.tokenizer.wrapper import PopTokenizer
    from pop.train.precision import cap_gpu_memory, scale_micro_batch, training_precision
    from pop.train.span_corruption import DataCollatorForT5, SpanCorruptionDataset

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    tokenizer = PopTokenizer.load(cfg.tokenizer_path)

    if cfg.corpus_file is not None:
        from pop.data.corpus import load_corpus_file

        records = load_corpus_file(cfg.corpus_file)
        corpus = load_pretraining_corpus(cfg.corpus_num_samples, seed=cfg.seed, records=records)
    else:
        corpus = load_pretraining_corpus(cfg.corpus_num_samples, seed=cfg.seed)
    logger.info("Loaded pretraining corpus: %d methods", len(corpus))

    dataset = SpanCorruptionDataset(
        corpus,
        tokenizer,
        max_length=cfg.max_seq_length,
        corruption_rate=cfg.corruption_rate,
        mean_span_len=cfg.mean_span_length,
        seed=cfg.seed,
    )
    collator = DataCollatorForT5(pad_id=tokenizer.pad_id)

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
    report_to = ["wandb"] if os.environ.get("WANDB_API_KEY") else []

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.lr,
        seed=cfg.seed,
        bf16=bf16,
        fp16=fp16,
        report_to=report_to,
        save_strategy="no",  # checkpointing is driven by SaveAtEpochsCallback below
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[_save_at_epochs_callback(cfg.save_epochs)],
    )
    from transformers.trainer_utils import get_last_checkpoint

    last_checkpoint = get_last_checkpoint(str(output_dir))
    if last_checkpoint:
        logger.info("Resuming pretraining from checkpoint: %s", last_checkpoint)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    return final_dir
