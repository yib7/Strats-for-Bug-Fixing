"""Unit tests for pop.train.precision (bf16/fp16 selection + GPU memory cap)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pop.train.precision import (  # noqa: E402
    cap_gpu_memory,
    scale_micro_batch,
    training_precision,
)


def test_cpu_only_is_fp32(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert training_precision() == (False, False)


def test_bf16_gpu_selects_bf16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert training_precision() == (True, False)


def test_non_bf16_gpu_falls_back_to_fp16(monkeypatch):
    # Colab's free-tier T4 case: CUDA available, no bf16 -> fp16 mixed precision.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert training_precision() == (False, True)


def test_force_fp32_env_overrides(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setenv("POP_FORCE_FP32", "1")
    assert training_precision() == (False, False)


def test_cap_gpu_memory_noop_on_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cap_gpu_memory()  # must not raise


def test_cap_gpu_memory_uses_fraction_env(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda fraction: calls.append(fraction)
    )
    monkeypatch.setenv("POP_GPU_MEM_FRACTION", "0.5")
    cap_gpu_memory()
    assert calls == [0.5]


def _capture_fraction(monkeypatch) -> list[float]:
    calls: list[float] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "set_per_process_memory_fraction", lambda fraction: calls.append(fraction)
    )
    return calls


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_cap_gpu_memory_treats_a_blank_env_var_as_unset(blank, monkeypatch):
    """Regression: `.env.example` ships `POP_GPU_MEM_FRACTION=` and documents
    `set -a; . ./.env; set +a`, which exports the empty string. `os.environ.get(name,
    default)` then returns "" -- the default never applied -- and `float("")` raised,
    killing every training entry point on exactly the GPU box that followed the docs."""
    calls = _capture_fraction(monkeypatch)
    monkeypatch.setenv("POP_GPU_MEM_FRACTION", blank)
    cap_gpu_memory()
    assert calls == [0.85]


def test_cap_gpu_memory_falls_back_on_a_malformed_value(monkeypatch):
    # A typo must not crash a multi-hour GPU run.
    calls = _capture_fraction(monkeypatch)
    monkeypatch.setenv("POP_GPU_MEM_FRACTION", "0.85x")
    cap_gpu_memory()
    assert calls == [0.85]


@pytest.mark.parametrize("blank", ["", "  "])
def test_force_fp32_blank_env_is_not_truthy(blank, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setenv("POP_FORCE_FP32", blank)
    assert training_precision() == (True, False)  # blank == unset, so bf16 still wins


@pytest.mark.parametrize("blank", ["", "  "])
def test_scale_micro_batch_ignores_a_blank_override(blank, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("POP_MICRO_BATCH", blank)
    assert scale_micro_batch(8, 8) == (8, 8)


def test_scale_micro_batch_ignores_a_malformed_override(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("POP_MICRO_BATCH", "sixteen")
    assert scale_micro_batch(8, 8) == (8, 8)


def _fake_vram(monkeypatch, gib: float) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda idx: SimpleNamespace(total_memory=int(gib * 1024**3)),
    )


def test_scale_micro_batch_unchanged_on_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert scale_micro_batch(8, 8) == (8, 8)


def test_scale_micro_batch_unchanged_on_16gb(monkeypatch):
    _fake_vram(monkeypatch, 16)
    assert scale_micro_batch(8, 8) == (8, 8)


def test_scale_micro_batch_l4_class(monkeypatch):
    _fake_vram(monkeypatch, 24)
    assert scale_micro_batch(8, 8) == (16, 4)


def test_scale_micro_batch_a100_class(monkeypatch):
    _fake_vram(monkeypatch, 40)
    assert scale_micro_batch(8, 8) == (32, 2)


def test_scale_micro_batch_env_override_wins(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("POP_MICRO_BATCH", "16")
    assert scale_micro_batch(8, 8) == (16, 4)


def test_scale_micro_batch_rejects_non_divisor_override(monkeypatch):
    monkeypatch.setenv("POP_MICRO_BATCH", "7")
    assert scale_micro_batch(8, 8) == (8, 8)
    monkeypatch.setenv("POP_MICRO_BATCH", "128")
    assert scale_micro_batch(8, 8) == (8, 8)
