"""GPU precision selection + memory-safety helpers shared by the trainer entry points.

Centralizes two portability concerns for the training/generation subprocesses:

- :func:`training_precision`: bf16 on GPUs that support it (RX 9070, A100, L4),
  else fp16 mixed precision on other CUDA GPUs -- Colab's free-tier T4 has no
  bf16, and full fp32 there forfeits the tensor cores (several times slower).
  ``POP_FORCE_FP32=1`` forces fp32 as an escape hatch if fp16 ever goes
  numerically sideways (NaN losses).
- :func:`cap_gpu_memory`: caps the CUDA caching allocator to a fraction of VRAM
  (default 0.85, override with ``POP_GPU_MEM_FRACTION``). Needed because on
  Windows/WDDM the driver does not fail an over-VRAM allocation -- it silently
  overflows into shared *system* RAM. Measured on the 16 GB RX 9070: a batch-32
  step "allocated" 19.4 GB, ran ~6x slower, and starved the whole desktop. With
  the cap, overflow surfaces as an ordinary CUDA OOM in this process instead.
- :func:`scale_micro_batch`: on large-VRAM GPUs (Colab Pro A100/L4), raises the
  per-device micro-batch while shrinking gradient accumulation so the effective
  batch -- the frozen notebook hyperparameter -- never changes. Its VRAM table is
  calibrated on the 52M-param T5, so only the T5 pretrain/finetune arms use it; the
  1.5B-param LoRA arm (much heavier per-sample memory, ~152k-token vocab) sets a
  memory-safe micro-batch in its config directly rather than through this table.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _env_str(name: str) -> str | None:
    """Read an env var, treating blank as unset.

    `.env.example` documents `set -a; . ./.env; set +a` and ships `POP_GPU_MEM_FRACTION=`
    with an empty value, so sourcing it exports the empty string. `os.environ.get(name,
    default)` then returns `""` -- the default never applies -- and `float("")` raised,
    killing every training entry point on exactly the GPU machine whose owner followed the
    documented setup.
    """
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip() or None


def _env_number(name: str, default: str, cast):
    """`cast(_env_str(name) or default)`, falling back to the default on a malformed value.

    A typo in an env var must not crash a long GPU run hours in.
    """
    raw = _env_str(name) or default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        logger.warning("%s=%r is not a valid %s; using %s", name, raw, cast.__name__, default)
        return cast(default)


def training_precision() -> tuple[bool, bool]:
    """Return ``(bf16, fp16)`` flags for ``transformers.TrainingArguments``.

    Exactly one of the two is True on a CUDA device (bf16 preferred), both are
    False on CPU or when ``POP_FORCE_FP32=1``.
    """
    import torch

    if _env_str("POP_FORCE_FP32") == "1":
        return False, False
    if not torch.cuda.is_available():
        return False, False
    if torch.cuda.is_bf16_supported():
        return True, False
    return False, True


def cap_gpu_memory() -> None:
    """Cap this process's CUDA allocations to a fraction of dedicated VRAM."""
    import torch

    if not torch.cuda.is_available():
        return
    fraction = _env_number("POP_GPU_MEM_FRACTION", "0.85", float)
    torch.cuda.set_per_process_memory_fraction(fraction)


# (min VRAM GiB, per-device micro-batch), largest first. Peaks measured at seq 512
# on the 52M-param T5: micro 16 = 10.1 GB, micro 32 = 19.4 GB -- safe on 24 GB (L4)
# and 40 GB (A100) cards respectively, under the 0.85 memory cap.
_VRAM_MICRO_BATCH: tuple[tuple[int, int], ...] = ((35, 32), (20, 16))


def scale_micro_batch(batch_size: int, grad_accum: int) -> tuple[int, int]:
    """Pick a larger per-device micro-batch on large-VRAM GPUs, preserving the
    effective batch.

    The shipped configs use batch 8 x accum 8, sized for 16 GB cards (see
    configs/pretrain_10ep.yaml); on a 40 GB A100 or 24 GB L4 that underutilizes
    the GPU badly. This bumps the micro-batch (32 / 16 respectively) and shrinks
    accumulation so ``batch_size * grad_accum`` -- the frozen notebook
    hyperparameter -- is unchanged. ``POP_MICRO_BATCH=<n>`` forces a specific
    micro-batch instead. Only exact divisors of the effective batch are accepted;
    anything else (including CPU / small-VRAM GPUs) returns the config unchanged.

    Caveat: a different micro-batch changes optimizer steps-per-epoch, so don't
    switch GPU classes while resuming *mid*-step from its checkpoints (switching
    between steps is fine) -- docs/colab-runbook.md tells the user the same.
    """
    import torch

    effective = batch_size * grad_accum
    forced = _env_str("POP_MICRO_BATCH")
    if forced:
        micro = _env_number("POP_MICRO_BATCH", "0", int)
        if 0 < micro <= effective and effective % micro == 0:
            return micro, effective // micro
        return batch_size, grad_accum
    if not torch.cuda.is_available():
        return batch_size, grad_accum
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    for min_gib, micro in _VRAM_MICRO_BATCH:
        if vram_gib >= min_gib and micro > batch_size and effective % micro == 0:
            return micro, effective // micro
    return batch_size, grad_accum
