"""`pop smoke`: end-to-end tokenizer -> pretrain -> finetune -> eval dry run on CPU.

Runs entirely against the committed `tests/fixtures/smoke_*` fixtures (see
`scripts/build_smoke_fixtures.py`) -- no network access. Intended to finish in minutes on a
laptop CPU as a pre-launch sanity check before spending real GPU time on the Colab notebooks
(see `docs/handoff.md`).

Heavy imports (torch/transformers) are deferred inside :func:`run_smoke` for the same reason as
`pop.train.pretrain`/`pop.train.finetune`: importing this module (and therefore `pop --help`)
stays fast.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pop.config import SmokeConfig

logger = logging.getLogger(__name__)


def run_smoke(cfg: SmokeConfig) -> dict:
    """Run the full micro tokenizer -> pretrain -> finetune -> eval pipeline per `cfg`.

    Returns the metrics dict written to `results/<cfg.results_name>.json`.
    """
    import torch
    from transformers import T5ForConditionalGeneration

    from pop.config import FinetuneConfig, PretrainConfig
    from pop.data.corpus import load_corpus_file
    from pop.data.refinement import load_pairs_file
    from pop.eval.metrics import evaluate_predictions, write_results
    from pop.tokenizer.train import train_tokenizer
    from pop.tokenizer.wrapper import PopTokenizer
    from pop.train.finetune import run_finetune
    from pop.train.pretrain import run_pretrain

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tokenizer.
    logger.info("pop smoke: training tokenizer (vocab_size=%d)", cfg.vocab_size)
    corpus_records = load_corpus_file(cfg.corpus_file)
    corpus_texts = [record["code"] for record in corpus_records]
    tokenizer_path = train_tokenizer(
        corpus_texts, output_dir / "tokenizer", vocab_size=cfg.vocab_size
    )

    # 2. Micro-pretrain.
    logger.info("pop smoke: pretraining (%d epoch(s))", cfg.pretrain_epochs)
    pretrain_cfg = PretrainConfig(
        seed=cfg.seed,
        tokenizer_path=tokenizer_path,
        corpus_num_samples=cfg.corpus_num_samples,
        corpus_file=cfg.corpus_file,
        max_seq_length=cfg.max_seq_length,
        epochs=cfg.pretrain_epochs,
        batch_size=cfg.pretrain_batch_size,
        lr=cfg.pretrain_lr,
        save_epochs=[cfg.pretrain_epochs],
        output_dir=output_dir / "pretrain",
        model=cfg.model,
    )
    pretrain_final_dir = run_pretrain(pretrain_cfg)

    # 3. Micro-finetune.
    logger.info("pop smoke: finetuning (%d epoch(s))", cfg.finetune_epochs)
    finetune_cfg = FinetuneConfig(
        seed=cfg.seed,
        tokenizer_path=tokenizer_path,
        pretrained_model_path=pretrain_final_dir,
        train_pairs_file=cfg.finetune_pairs_file,
        val_pairs_file=cfg.val_pairs_file,
        max_seq_length=cfg.max_seq_length,
        epochs=cfg.finetune_epochs,
        batch_size=cfg.finetune_batch_size,
        lr=cfg.finetune_lr,
        warmup_steps=cfg.finetune_warmup_steps,
        output_dir=output_dir / "finetune",
        model=cfg.model,
    )
    finetune_best_dir = run_finetune(finetune_cfg)

    # 4. Generate on the held-out eval fixture + full metric stack.
    logger.info("pop smoke: generating + scoring on %s", cfg.eval_pairs_file)
    tokenizer = PopTokenizer.load(tokenizer_path)
    model = T5ForConditionalGeneration.from_pretrained(str(finetune_best_dir))
    model.eval()

    eval_pairs = load_pairs_file(cfg.eval_pairs_file)
    predictions: list[str] = []
    references: list[str] = []
    with torch.no_grad():
        for pair in eval_pairs:
            input_ids = tokenizer.encode(pair["buggy"], max_length=cfg.max_seq_length - 1)
            input_ids = [*input_ids, tokenizer.eos_id]
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            attention_mask = torch.ones_like(input_tensor)
            generated = model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=cfg.max_new_tokens,
                num_beams=1,
            )
            prediction = tokenizer.decode(generated[0].tolist())
            predictions.append(prediction)
            references.append(pair["fixed"])

    metrics = evaluate_predictions(predictions, references)

    results_path = write_results(
        cfg.results_name,
        metrics,
        config={
            "mode": "smoke",
            "vocab_size": cfg.vocab_size,
            "corpus_n": len(corpus_texts),
            "finetune_n": len(load_pairs_file(cfg.finetune_pairs_file)),
            "eval_n": len(eval_pairs),
            "model": cfg.model.model_dump(),
            "pretrain_epochs": cfg.pretrain_epochs,
            "finetune_epochs": cfg.finetune_epochs,
        },
    )
    logger.info("pop smoke: wrote results to %s", results_path)

    print("\npop smoke -- summary")
    print("-" * 40)
    print(f"{'corpus methods':<22}{len(corpus_texts):>10}")
    print(f"{'finetune pairs':<22}{len(load_pairs_file(cfg.finetune_pairs_file)):>10}")
    print(f"{'eval samples':<22}{len(eval_pairs):>10}")
    for key in ("em", "em_raw", "codebleu", "syntax_valid_rate"):
        print(f"{key:<22}{metrics[key]:>10.4f}")
    print("-" * 40)
    print(f"results: {results_path}")

    return metrics
