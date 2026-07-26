"""Reusable text generation over rendered prompts: vLLM if usable, else transformers.

`build_generator` loads the model/engine **once** and returns a callable mapping a
list of prompts to a list of completion strings; callers feed prompt chunks to that
callable without reloading the model per chunk. Backends are lazily imported inside
their factory functions so importing this module (and dispatch-only unit tests)
never requires vLLM, torch, or a model download. The factories are injectable so
tests can exercise dispatch/fallback logic with fakes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

Generator = Callable[[list[str]], list[str]]
GeneratorFactory = Callable[[str, dict], Generator]


def _vllm_importable() -> bool:
    return importlib.util.find_spec("vllm") is not None


def _default_vllm_generator(model_name: str, sampling: dict) -> Generator:
    """Build the vLLM engine once; return a prompts->texts closure that reuses it."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_name)
    params = SamplingParams(**sampling)

    def generate(prompts: list[str]) -> list[str]:
        outputs = llm.generate(prompts, params)
        return [output.outputs[0].text for output in outputs]

    return generate


def _default_transformers_generator(model_name: str, gen_kwargs: dict) -> Generator:
    """Build the transformers pipeline once; return a prompts->texts closure.

    Decoder-only batched generation needs a pad token and left padding, so both are
    set on the tokenizer here (Qwen ships no pad token by default).
    """
    from transformers import pipeline
    from transformers.utils import logging as hf_logging

    # Quiet the per-batch generation warnings so a multi-thousand-sample sweep doesn't
    # flood the notebook (errors still surface). The max_length clear below removes the
    # actual cause of the loudest one; this is the backstop.
    hf_logging.set_verbosity_error()

    pipe = pipeline("text-generation", model=model_name)
    tokenizer = pipe.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Qwen's default GenerationConfig leaves max_length at 20; passing max_new_tokens then
    # makes transformers warn "both max_new_tokens and max_length set" on every batch. Clear
    # it so only max_new_tokens governs -- removes the warning without changing behavior.
    pipe.model.generation_config.max_length = None
    # clean_up_tokenization_spaces is a no-op for BPE tokenizers (it only warns that it's being
    # ignored); disabling it silences that notice.
    tokenizer.clean_up_tokenization_spaces = False

    def generate(prompts: list[str]) -> list[str]:
        # return_full_text=False makes the pipeline drop the prompt tokens itself, which is
        # the only reliable way to do it: the pipeline's own `generated_text` is a *decode of
        # the re-tokenized prompt*, not the string that was passed in, so chat-template
        # rendering, special-token round-tripping or whitespace normalization can make the two
        # differ. Recovering the completion by `full_text[len(prompt):] if
        # full_text.startswith(prompt)` therefore had a silent else-branch that returned prompt
        # + completion, and `extract_fix` would then pick the largest code block out of the
        # *prompt* -- for k>=1 that is a retrieved exemplar's already-fixed method, emitted as
        # if the model had produced it. This matches the LoRA arm's twin (train/lora.py).
        raw_results = pipe(prompts, return_full_text=False, **gen_kwargs)
        return [
            (result[0] if isinstance(result, list) else result)["generated_text"]
            for result in raw_results
        ]

    return generate


def build_generator(
    model_name: str,
    *,
    max_new_tokens: int = 256,
    greedy: bool = True,
    batch_size: int = 16,
    backend: str | None = None,
    vllm_generator_factory: GeneratorFactory | None = None,
    transformers_generator_factory: GeneratorFactory | None = None,
    **gen_kwargs: object,
) -> Generator:
    """Build a reusable prompts->completions callable, loading the model once.

    Backend: vLLM when importable, else the transformers pipeline. If vLLM is
    auto-selected but fails to initialize (e.g. a CUDA-mismatched wheel on Colab,
    where `import vllm` succeeds but building the engine raises), this transparently
    falls back to the transformers backend. A *forced* ``backend="vllm"`` re-raises
    instead, so an explicit request fails loudly rather than silently downgrading.

    The two backends take different generation kwargs, so this normalizes
    `max_new_tokens` and greedy decoding to each backend's names (a bare vLLM
    `SamplingParams()` would default to max_tokens=16 + sampling -- truncated,
    non-reproducible). Greedy (temperature 0 / `do_sample=False`) makes results
    deterministic for a fixed seed. `batch_size` batches the transformers path so
    it isn't one-sample-at-a-time. Anything in `gen_kwargs` overrides these defaults.
    """
    chosen = backend or ("vllm" if _vllm_importable() else "transformers")

    if chosen == "vllm":
        sampling: dict[str, object] = {"max_tokens": max_new_tokens}
        if greedy:
            sampling["temperature"] = 0.0
        sampling.update(gen_kwargs)  # caller-supplied kwargs win
        factory = vllm_generator_factory or _default_vllm_generator
        try:
            return factory(model_name, sampling)
        except Exception as exc:  # noqa: BLE001 -- any engine-init failure -> fall back
            if backend == "vllm":
                raise
            print(
                f"[pop] vLLM was selected but failed to initialize ({exc}); "
                "falling back to the transformers backend.",
                file=sys.stderr,
            )
            chosen = "transformers"

    if chosen == "transformers":
        tf_kwargs: dict[str, object] = {"max_new_tokens": max_new_tokens, "batch_size": batch_size}
        if greedy:
            tf_kwargs["do_sample"] = False
        tf_kwargs.update(gen_kwargs)
        factory = transformers_generator_factory or _default_transformers_generator
        return factory(model_name, tf_kwargs)

    raise ValueError(f"Unknown backend: {chosen!r} (expected 'vllm' or 'transformers')")


def _repair_partial(partial: Path) -> int:
    """Count the usable records in a checkpoint file, dropping a torn trailing write.

    A process killed mid-``write`` leaves a partial JSON object with no trailing newline.
    It still counts as a line, so resume would append straight onto it and produce one
    unparseable record in the middle of the finished file -- which ``pop eval`` then
    rejects. Keep only lines that both parse as JSON and are newline-terminated,
    rewriting the file when anything was dropped. Returns the surviving count.
    """
    if not partial.exists():
        return 0
    text = partial.read_text(encoding="utf-8")
    kept: list[str] = []
    # splitlines(keepends=True): a final line without "\n" is exactly the torn-write case.
    for line in text.splitlines(keepends=True):
        if not line.endswith("\n"):
            break
        stripped = line.strip()
        if not stripped:
            continue
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            break
        kept.append(line)
    if len("".join(kept)) != len(text):
        partial.write_text("".join(kept), encoding="utf-8")
    return len(kept)


_PARTIAL_META_VERSION = 1


def _identity_path(partial: Path) -> Path:
    return partial.with_name(partial.name + ".meta")


def _partial_identity(prompts: Sequence[str], references: Sequence[str]) -> dict:
    """Digest the exact task a checkpoint answers: every prompt and every reference.

    A ``.partial`` records only predictions, so nothing in it says which prompts produced
    them. Hashing the full lists (rather than just their length and endpoints) is what makes
    a changed ``k``, retriever, split, prompt template or ``--limit`` all detectable, and it
    is free next to the generation it guards: SHA-256 over the 6,545-pair test split's
    prompts is milliseconds against hours of GPU time.
    """
    digest = hashlib.sha256()
    for seq in (prompts, references):
        digest.update(f"{len(seq)}\n".encode())
        for item in seq:
            # Length-prefixed so ["ab", "c"] and ["a", "bc"] cannot collide.
            encoded = item.encode("utf-8")
            digest.update(f"{len(encoded)}:".encode())
            digest.update(encoded)
    return {
        "version": _PARTIAL_META_VERSION,
        "n": len(prompts),
        "digest": digest.hexdigest(),
    }


def write_partial_identity(
    partial: str | Path, prompts: Sequence[str], references: Sequence[str]
) -> Path:
    """Write the sidecar that stamps ``partial`` with the task it belongs to.

    Public because a checkpoint is only resumable if its identity file exists, so anything
    that fabricates a ``.partial`` (tests, a recovery script) needs to stamp it the same way
    ``generate_with_resume`` does rather than hand-rolling the digest format.
    """
    partial = Path(partial)
    meta_path = _identity_path(partial)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(_partial_identity(prompts, references)) + "\n", encoding="utf-8"
    )
    return meta_path


def _identity_mismatch_reason(
    partial: Path, prompts: Sequence[str], references: Sequence[str]
) -> str | None:
    """``None`` if ``partial`` provably belongs to this task, else why it does not."""
    meta_path = _identity_path(partial)
    if not meta_path.exists():
        return "it carries no identity record"
    try:
        recorded = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "its identity record is unreadable"
    expected = _partial_identity(prompts, references)
    if recorded != expected:
        return (
            f"its identity record does not match this run "
            f"(recorded n={recorded.get('n')}, this run n={expected['n']})"
        )
    return None


def generate_with_resume(
    prompts: Sequence[str],
    references: Sequence[str],
    out_path: str | Path,
    generate_fn: Callable[[list[str]], list[str]],
    *,
    chunk_size: int = 256,
) -> int:
    """Generate predictions in chunks, checkpointing so an interrupted run resumes.

    Each chunk's ``{"prediction", "reference"}`` JSONL lines are appended to a
    sibling ``<out_path>.partial`` file and flushed. Only once every prompt is
    done is the partial atomically renamed to ``out_path`` (via ``os.replace``),
    so ``out_path`` existing always means "this config is fully complete" -- the
    contract the sweep orchestrator's done-marker relies on. On a re-run, the
    already-written partial lines are counted and generation resumes at that
    index, so a mid-config Colab disconnect never re-generates finished work.

    A partial is only resumed when a sibling ``<out_path>.partial.meta`` proves it
    answers *this* prompt/reference list (see `_partial_identity`); otherwise it is
    discarded with a note on stderr and the config restarts. Without that check,
    resuming after changing ``cfg.split``, ``k``, the retriever or ``--limit`` kept
    the first ``n_done`` predictions -- answers to *different* inputs -- and then
    finalized the file as though it were complete, producing a mixed predictions
    file that `pop eval` scores without complaint.

    `generate_fn` maps a chunk of prompts to the same number of final prediction
    strings (retrieval/prompting/extraction are the caller's job). Returns the
    total number of predictions in the finished file.
    """
    out_path = Path(out_path)
    if len(prompts) != len(references):
        raise ValueError(
            f"prompts ({len(prompts)}) and references ({len(references)}) must be equal length"
        )

    partial = out_path.with_name(out_path.name + ".partial")
    n_done = 0
    if partial.exists():
        reason = _identity_mismatch_reason(partial, prompts, references)
        if reason is None:
            n_done = _repair_partial(partial)
        else:
            print(
                f"[pop] discarding the checkpoint {partial.name} because {reason}; "
                "this config restarts from scratch.",
                file=sys.stderr,
            )
            partial.unlink()
    if n_done > len(prompts):
        # Unreachable while the identity matches (the length is part of the digest); kept as
        # a guard against a checkpoint corrupted in some way the digest cannot see.
        raise ValueError(
            f"partial file {partial} has {n_done} lines but only {len(prompts)} prompts; "
            "delete it to restart this config from scratch"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_partial_identity(partial, prompts, references)
    with partial.open("a", encoding="utf-8") as f:
        for start in range(n_done, len(prompts), chunk_size):
            chunk = list(prompts[start : start + chunk_size])
            preds = generate_fn(chunk)
            if len(preds) != len(chunk):
                raise ValueError(
                    f"generate_fn returned {len(preds)} predictions for {len(chunk)} prompts"
                )
            for pred, ref in zip(preds, references[start : start + len(chunk)], strict=True):
                f.write(json.dumps({"prediction": pred, "reference": ref}) + "\n")
            f.flush()

    os.replace(partial, out_path)
    # The identity only describes an in-progress checkpoint; `out_path` existing already
    # means "complete", so leaving the sidecar behind would just be litter that a later
    # run of a *different* config could find next to its own partial.
    _identity_path(partial).unlink(missing_ok=True)
    return len(prompts)
