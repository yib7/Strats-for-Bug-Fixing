"""Batched T5 generation for producing code-refinement predictions.

Mirrors the un-batched generation loop in :func:`pop.train.smoke.run_smoke`
but batches across inputs (right-padded with an attention mask) so it is
usable both by the ``pop generate`` CLI and the training orchestrator to
score a whole eval split. Heavy imports (torch/transformers) are deferred
inside :func:`generate_t5_predictions` so importing this module (and therefore
``pop --help``) stays fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def generate_t5_predictions(
    model_dir: str | Path,
    tokenizer_path: str | Path,
    buggy_texts: Sequence[str],
    *,
    max_seq_length: int = 512,
    max_new_tokens: int = 256,
    num_beams: int = 1,
    batch_size: int = 16,
    device: str | None = None,
) -> list[str]:
    """Generate a fixed-code prediction for each buggy input with a finetuned T5.

    Args:
        model_dir: directory of a saved ``T5ForConditionalGeneration`` (as
            written by ``pop finetune``'s ``best/`` output).
        tokenizer_path: path to the SentencePiece ``.model`` the model was
            trained with.
        buggy_texts: the buggy source strings to fix.
        max_seq_length: max encoder input length (an EOS is appended within it).
        max_new_tokens: max tokens to generate per input.
        num_beams: beam width (1 = greedy).
        batch_size: number of inputs per generation batch.
        device: torch device string; defaults to ``"cuda"`` if available else
            ``"cpu"``.

    Returns:
        One decoded prediction string per input, in input order.
    """
    import torch
    from transformers import T5ForConditionalGeneration

    from pop.tokenizer.wrapper import PopTokenizer

    tokenizer = PopTokenizer.load(tokenizer_path)
    model = T5ForConditionalGeneration.from_pretrained(str(model_dir))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    predictions: list[str] = []
    with torch.no_grad():
        for start in range(0, len(buggy_texts), batch_size):
            batch = buggy_texts[start : start + batch_size]
            encoded = []
            for text in batch:
                ids = tokenizer.encode(text, max_length=max_seq_length - 1)
                encoded.append([*ids, tokenizer.eos_id])

            max_len = max(len(ids) for ids in encoded)
            input_ids = torch.full((len(encoded), max_len), tokenizer.pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
            for i, ids in enumerate(encoded):
                input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention_mask[i, : len(ids)] = 1

            generated = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            predictions.extend(tokenizer.decode(row.tolist()) for row in generated)

    return predictions
