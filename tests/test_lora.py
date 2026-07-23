"""Tiny-model CPU smoke for pop.train.lora.

Verifies that `run_lora` genuinely wires a tiny causal LM + tiny data through a real
(CPU, 1-epoch) HF Trainer with a PEFT LoRA adapter and writes a loadable adapter dir --
fully offline: a locally-constructed Qwen2 causal LM (whose attention exposes the
q/k/v/o_proj modules the LoRA `target_modules` point at) and a locally-trained BPE
tokenizer are saved to a tmp dir that `LoRAConfig.base_model` points at, so no model or
tokenizer is ever downloaded.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from pop.config import LoRAConfig  # noqa: E402

TINY_PAIRS = [
    {"buggy": "int a = 1 ;", "fixed": "int a = 2 ;"},
    {"buggy": "if (c == 'a') count += 1 ;", "fixed": "if (c == 'A') count += 1 ;"},
    {"buggy": "return s . trim ( ) ;", "fixed": "return s . strip ( ) ;"},
    {
        "buggy": "for ( int i = 0 ; i < n ; i ++ ) x ++ ;",
        "fixed": "for ( int i = 1 ; i < n ; i ++ ) x ++ ;",
    },
]


def _build_tiny_model_dir(dest) -> str:
    """Construct + save a tiny offline Qwen2 causal LM and BPE tokenizer to `dest`."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    corpus = [p["buggy"] for p in TINY_PAIRS] + [p["fixed"] for p in TINY_PAIRS]
    corpus = corpus * 8

    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=["<pad>", "<eos>", "<unk>"])
    backend.train_from_iterator(corpus, trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(dest)

    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    Qwen2ForCausalLM(config).save_pretrained(dest)
    return str(dest)


def test_run_lora_writes_adapter(tmp_path, monkeypatch):
    # Belt-and-braces: nothing in this test may touch the network.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    from pop.train.lora import run_lora

    base_model = _build_tiny_model_dir(tmp_path / "tiny_model")

    train_file = tmp_path / "train.jsonl"
    train_file.write_text("\n".join(json.dumps(p) for p in TINY_PAIRS) + "\n", encoding="utf-8")
    val_file = tmp_path / "val.jsonl"
    val_file.write_text("\n".join(json.dumps(p) for p in TINY_PAIRS[:2]) + "\n", encoding="utf-8")

    cfg = LoRAConfig(
        base_model=base_model,
        train_pairs_file=train_file,
        val_pairs_file=val_file,
        max_seq_length=48,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        epochs=1,
        batch_size=2,
        warmup_steps=0,
        output_dir=tmp_path / "lora_out",
    )

    adapter_dir = run_lora(cfg)

    assert adapter_dir == tmp_path / "lora_out" / "best"
    assert (adapter_dir / "adapter_config.json").exists()
    # Adapter weights are written alongside the config (safetensors or bin).
    weight_files = list(adapter_dir.glob("adapter_model.*"))
    assert weight_files, f"no adapter weights in {adapter_dir}"

    # The saved adapter loads back onto a fresh copy of the base model.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(base_model)
    loaded = PeftModel.from_pretrained(base, str(adapter_dir))
    assert loaded is not None


class _FakeTok:
    """Minimal tokenizer: one id per non-space char; no chat template. Enough to
    exercise CausalRefinementDataset's prompt-masking + EOS-preserving truncation
    without building a real model/tokenizer."""

    chat_template = None
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) % 50 + 1 for c in text if not c.isspace()]}


class _ChatFakeTok:
    """Tokenizer *with* a chat template (like real Qwen2.5-Coder-Instruct): renders a
    tiny Jinja template and tokenizes chars. Exercises build_lora_prompt's chat branch."""

    chat_template = (
        "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
    )
    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        from jinja2 import Template

        assert tokenize is False
        return Template(self.chat_template).render(
            messages=messages, add_generation_prompt=add_generation_prompt
        )

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) % 50 + 1 for c in text if not c.isspace()]}


def test_causal_dataset_preserves_eos_when_completion_is_truncated():
    from pop.train.lora import IGNORE_INDEX, CausalRefinementDataset

    long_fixed = "x" * 100  # completion (100 ids + EOS) far exceeds max_length
    ds = CausalRefinementDataset([{"buggy": "a b", "fixed": long_fixed}], _FakeTok(), max_length=8)
    ex = ds[0]

    assert len(ex["input_ids"]) <= 8
    # EOS must survive truncation (the model must still learn to stop).
    assert ex["input_ids"][-1] == 99
    assert ex["labels"][-1] == 99
    # Completion filled max_length -> no prompt budget -> nothing masked, but no all--100 row.
    assert IGNORE_INDEX not in ex["labels"]


def test_causal_dataset_masks_prompt_and_appends_eos_when_it_fits():
    from pop.train.lora import IGNORE_INDEX, CausalRefinementDataset

    ds = CausalRefinementDataset([{"buggy": "a b c", "fixed": "d e"}], _FakeTok(), max_length=128)
    ex = ds[0]

    assert ex["input_ids"][-1] == 99  # EOS is the final token
    assert ex["labels"][0] == IGNORE_INDEX  # prompt is masked
    assert ex["labels"].count(IGNORE_INDEX) > 0
    # The supervised tail (labels != -100) equals the completion tokens in input_ids.
    supervised = [lab for lab in ex["labels"] if lab != IGNORE_INDEX]
    assert supervised == ex["input_ids"][-len(supervised) :]
    assert supervised[-1] == 99


# ---------------------------------------------------------------------------
# build_lora_prompt: single source of truth guarding train/inference consistency
# ---------------------------------------------------------------------------


def test_build_lora_prompt_is_single_source_for_training_ids():
    # The plain (non-chat) template: generation and training must build/tokenize
    # the exact same prompt string, so a LoRA model is asked to complete at
    # inference the prefix it was trained on.
    from pop.train.lora import INSTRUCTION, _build_prompt_ids, build_lora_prompt

    tok = _FakeTok()
    buggy = "int a = 1 ;"
    prompt = build_lora_prompt(tok, buggy)

    assert INSTRUCTION in prompt
    assert buggy in prompt
    assert prompt.endswith("### Fixed Java:\n")
    assert _build_prompt_ids(tok, buggy) == tok(prompt, add_special_tokens=True)["input_ids"]


def test_build_lora_prompt_uses_chat_template_with_generation_prompt():
    # Real-Qwen path: the chat template is applied with add_generation_prompt=True,
    # and training tokenizes that same string (specials already embedded).
    from pop.train.lora import _build_prompt_ids, build_lora_prompt

    tok = _ChatFakeTok()
    buggy = "int a = 1 ;"
    prompt = build_lora_prompt(tok, buggy)

    assert "<|user|>" in prompt
    assert prompt.rstrip().endswith("<|assistant|>")
    assert buggy in prompt
    assert _build_prompt_ids(tok, buggy) == tok(prompt, add_special_tokens=False)["input_ids"]


# ---------------------------------------------------------------------------
# generate_lora_fixes dispatch (injected fake backend -- no model load/download)
# ---------------------------------------------------------------------------


def test_generate_lora_fixes_uses_injected_backend_without_loading_model(monkeypatch):
    import sys

    from pop.train.lora import generate_lora_fixes

    # The injected-backend path must never import torch/transformers/peft (no download).
    for mod in ("torch", "transformers", "peft"):
        monkeypatch.setitem(sys.modules, mod, None)

    calls = {}

    def fake_backend(prompts, base_model, adapter_dir, **kwargs):
        calls["base_model"] = base_model
        calls["adapter_dir"] = adapter_dir
        calls["kwargs"] = kwargs
        return [f"fixed:{p}" for p in prompts]

    result = generate_lora_fixes(
        ["p1", "p2"], "base-model", "adapter-dir", backend_fn=fake_backend, max_new_tokens=8
    )

    assert result == ["fixed:p1", "fixed:p2"]
    assert calls["base_model"] == "base-model"
    assert calls["adapter_dir"] == "adapter-dir"
    assert calls["kwargs"] == {"max_new_tokens": 8}


def test_build_lora_generator_defaults_to_batched_greedy(monkeypatch):
    import sys

    from pop.train.lora import build_lora_generator

    # Dispatch must not import torch/transformers/peft (no model load, no download).
    for mod in ("torch", "transformers", "peft"):
        monkeypatch.setitem(sys.modules, mod, None)

    calls = {}

    def fake_factory(base_model, adapter_dir, gen_kwargs):
        calls["base_model"] = base_model
        calls["adapter_dir"] = adapter_dir
        calls["gen_kwargs"] = gen_kwargs
        return lambda prompts: [f"gen:{p}" for p in prompts]

    generator = build_lora_generator("base-model", "adapter-dir", generator_factory=fake_factory)

    # The same generator is reusable across chunks without a factory re-call.
    assert generator(["p1"]) == ["gen:p1"]
    assert generator(["p2", "p3"]) == ["gen:p2", "gen:p3"]
    assert calls["base_model"] == "base-model"
    assert calls["adapter_dir"] == "adapter-dir"
    # Reproducible batched defaults: greedy decoding, batch 16, 256 new tokens.
    assert calls["gen_kwargs"] == {"max_new_tokens": 256, "batch_size": 16, "do_sample": False}


def test_build_lora_generator_explicit_kwargs_override_defaults(monkeypatch):
    from pop.train.lora import build_lora_generator

    calls = {}

    def fake_factory(base_model, adapter_dir, gen_kwargs):
        calls["gen_kwargs"] = gen_kwargs
        return lambda prompts: list(prompts)

    build_lora_generator(
        "base-model",
        "adapter-dir",
        max_new_tokens=64,
        batch_size=4,
        generator_factory=fake_factory,
        do_sample=True,
    )

    assert calls["gen_kwargs"] == {"max_new_tokens": 64, "batch_size": 4, "do_sample": True}


# ---------------------------------------------------------------------------
# pop lora-generate CLI write path (generate_lora_fixes monkeypatched -> canned text)
# ---------------------------------------------------------------------------


def test_lora_generate_cli_writes_prediction_reference_jsonl(tmp_path, monkeypatch):
    import argparse
    from pathlib import Path

    from pop import cli

    output_dir = tmp_path / "lora_out"
    (output_dir / "best").mkdir(parents=True)

    config_path = tmp_path / "lora.yaml"
    config_path.write_text(
        f"base_model: some/model\noutput_dir: {output_dir.as_posix()}\n", encoding="utf-8"
    )

    pairs = [
        {"buggy": "int a = 1 ;", "fixed": "int a = 2 ;"},
        {"buggy": "return s . trim ( ) ;", "fixed": "return s . strip ( ) ;"},
    ]
    # Raw model text with chatty preamble -> extract_fix must strip it to bare code.
    raw = [
        "Here is the fixed code:\nint a = 2 ;",
        "Sure, here's the corrected method:\nreturn s . strip ( ) ;",
    ]

    captured = {}

    def fake_load(split, records=None):
        captured["split"] = split
        return list(pairs)

    def fake_build(base_model, adapter_dir, **kwargs):
        # Stands in for build_lora_generator: capture the load args, return a
        # generator closure the CLI feeds prompt chunks to.
        captured["base_model"] = base_model
        captured["adapter_dir"] = adapter_dir
        captured["kwargs"] = kwargs

        def generate(prompts):
            captured.setdefault("prompts", []).extend(prompts)
            return list(raw)[: len(prompts)]

        return generate

    class _NoChatTok:
        chat_template = None

    monkeypatch.setattr("pop.data.refinement.load_refinement_pairs", fake_load)
    monkeypatch.setattr("pop.train.lora.build_lora_generator", fake_build)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: _NoChatTok())

    out_path = tmp_path / "preds.jsonl"
    args = argparse.Namespace(
        config=str(config_path), split="test", out=str(out_path), max_new_tokens=64, limit=None
    )

    assert cli._run_lora_generate(args) == 0

    lines = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line]
    assert lines == [
        {"prediction": "int a = 2 ;", "reference": "int a = 2 ;"},
        {"prediction": "return s . strip ( ) ;", "reference": "return s . strip ( ) ;"},
    ]
    # Generation was pointed at the adapter's best/ dir with the CLI's gen kwargs.
    assert Path(captured["adapter_dir"]) == output_dir / "best"
    assert captured["base_model"] == "some/model"
    assert captured["kwargs"] == {"max_new_tokens": 64}
    assert captured["split"] == "test"
    # Prompts were built via the shared builder (contain the instruction + buggy).
    assert all("Fix the bug" in p for p in captured["prompts"])


def test_lora_generate_cli_missing_adapter_is_clean_error(tmp_path):
    import argparse

    from pop import cli

    output_dir = tmp_path / "lora_out"
    output_dir.mkdir()  # no best/ or final/ adapter inside

    config_path = tmp_path / "lora.yaml"
    config_path.write_text(
        f"base_model: some/model\noutput_dir: {output_dir.as_posix()}\n", encoding="utf-8"
    )

    args = argparse.Namespace(
        config=str(config_path), split="test", out=None, max_new_tokens=64, limit=None
    )
    assert cli._run_lora_generate(args) == 2
