# Handoff: running the GPU phases

> **Status: complete** — these runs have been executed; the results are in
> [`report.md`](report.md). This document is kept as the historical operational runbook.

> **Current Phase-2 path (2026-07-17): Google Colab via `notebooks/colab_phase2.ipynb` +
> `scripts/run_training.py` -- see `docs/colab-runbook.md`.** The single notebook replaces the
> separate `colab_pretrain.ipynb`/`colab_finetune.ipynb` flow below for Phase 2 (those remain
> as reference; `colab_rag.ipynb`/`colab_lora.ipynb` still cover Phases 4-5). The local RX 9070
> path below still works and uses the same orchestrator.

> **✅ Phase 2 COMPLETE (2026-07-18, Colab A100, ~3 h 40 m, all 6 systems). Full numbers +
> analysis: `results/phase2_summary.md`; raw metrics: `results/finetune_*_test.json`.**
> Headline: within Pipeline A, CodeBLEU/syntax rise monotonically with finetune epochs
> (0.36/0.42/0.48; 46%/79%/92%) — the model learns. A_ep10 sits *inside* Pipeline B's seed band
> on CodeBLEU (0.476–0.484), so **pretraining shows no CodeBLEU benefit over from-scratch, only
> ~+3 pt syntax validity** — a small A-vs-B gap, now pinned down with a seed-variance ablation.
> **The EM>0 gate is resolved:** EM is genuinely ~0 (0–4/6545) but this was *audited* and confirmed
> real, not a decoding/metric artifact — the metric agrees 4 ways, predictions are complete valid
> Java (99.8% end on `}`, none truncated/degenerate), so
> beam-5 is not expected to help and no decoding fix is warranted. EM below even a copy-input
> baseline because the model always edits. Report CodeBLEU + syntax as the headline metrics.

This repo builds and verifies everything that runs without a GPU: the package skeleton,
data/tokenizer/model/eval modules, the RAG pipeline, the execution harness, and the smoke
pipeline. Actually retraining the T5 models and running the RAG/LoRA evaluations needs a GPU and
your own Weights & Biases key -- this document is the exact sequence.

## Local GPU: AMD RX 9070 (the preferred path)

The machine's RX 9070 (RDNA4/gfx1201) works with AMD's official PyTorch-on-Windows ROCm
wheels (torch 2.9.1+rocm7.2.1, Python 3.12; verified 2026-07-17: `torch.cuda.is_available()`
is True, Trainer selects `cuda:0`, bf16 supported, and `pop smoke` runs end-to-end on the GPU
in ~15 s). Set it up or repair it any time with:

```
python scripts/setup_rocm_windows.py
```

That script creates a **separate** env `.venv-rocm` (the stable CPU env `.venv`/uv is
untouched), installs the AMD wheels, and applies two documented workarounds the setup needs
(see the script's docstring): a `sitecustomize.py` shim for the missing `torch.distributed`
extension, and a one-line patch for a transformers 5.14 `DTensor` guard bug. Requires
Adrenalin driver >= 26.2.2 (already satisfied).

Run any phase locally by swapping the runner prefix -- every `pop` command below works as:

```
.venv-rocm\Scripts\pop pretrain --config configs/pretrain_10ep.yaml
```

Caveats of the local path:
- **Preview-grade stack.** ROCm-on-Windows is new; if a run misbehaves (NaNs, crashes,
  wildly-off losses), rerun the same config on Colab before trusting any conclusion. Numbers
  destined for the write-up should be reproduced on Colab (CUDA) at least once.
- **No vLLM** on Windows/ROCm: `pop rag` automatically uses the transformers fallback path.
- **W&B logging** works the same: `wandb login` once inside `.venv-rocm`, or leave it unset
  and metrics stay local-only.

The Colab notebooks below remain the fallback (and the reproduction check) for every phase.

## Before you launch anything: `pop smoke`

```
uv run pop smoke
```

This runs the *entire* pipeline shape (tokenizer train -> pretrain -> finetune -> generate ->
score) end-to-end on your local CPU in well under a minute, against tiny committed fixtures
(`tests/fixtures/smoke_*`, `configs/smoke.yaml`). It proves the code path works before you spend
GPU time/money on it. If `pop smoke` fails, nothing downstream will work either -- fix that
first. A passing run writes `results/smoke.json` with non-null `codebleu`/`syntax_valid_rate`
(exact match is expected to be 0.0 at this scale; that's fine, it's a plumbing check, not a
quality bar).

**Before every notebook launch:** open the notebook's install cell and pin the exact commit you
intend to run (see each notebook's install-cell markdown) -- don't launch against a moving
branch tip.

All runtimes below are **estimates** for a single Colab T4 GPU; actual time depends on which GPU
Colab hands you and current queue/throttling. Treat them as "roughly this much, budget more."

## Phase 2 -- Core retraining (`colab_pretrain.ipynb`, `colab_finetune.ipynb`)

Runs the Pipeline A ("pre-trained + fine-tuned") vs. Pipeline B ("fine-tuned, no pre-training")
comparison. The A-vs-B CodeBLEU difference is small, so this phase adds the seed-variance ablation
needed to tell a real gap from noise.

### Order

1. **`colab_pretrain.ipynb`** + `configs/pretrain_10ep.yaml` -- trains the real vocab-16384
   tokenizer on the full CodeSearchNet-Java corpus, then runs T5 span-corruption pretraining for
   10 epochs, checkpointing at epochs 1/3/10.
   - Rough runtime: tokenizer training a few minutes; pretraining is the long pole, likely
     several hours for 50K methods x 10 epochs on a T5-base-sized model on a T4 -- **estimate,
     confirm against your own first-epoch wall time and extrapolate**.
   - Bring back: `outputs/tokenizer/` (the trained tokenizer) and `outputs/pretrain/final/`
     (plus any epoch-1/epoch-3 checkpoints you want for later curve analysis) -> put them under
     the same paths in your local clone (or wherever Phase 3's compute-curve work expects them).
2. **`colab_finetune.ipynb`** -- re-upload the `outputs/tokenizer/` and `outputs/pretrain/final/`
   directories from step 1 into this Colab session first. Then, editing the notebook's `CONFIG`
   variable and re-running the run cell once per config:
   - `configs/finetune_A_ep1.yaml`, `configs/finetune_A_ep3.yaml`, `configs/finetune_A_ep10.yaml`
     -- Pipeline A (pretrained + finetuned), epochs swept at fixed seed 42.
   - `configs/finetune_B_seed0.yaml`, `configs/finetune_B_seed1.yaml`,
     `configs/finetune_B_seed2.yaml` -- Pipeline B (finetuned only, no pretraining -- these
     configs deliberately omit `pretrained_model_path`, so `pop finetune` builds a fresh random
     init), same 10 epochs, seed swept for variance.
   - Rough runtime per config: finetuning a T5-base on the CodeXGLUE-medium train split, tens of
     minutes to ~1-2 hours depending on epoch count -- **estimate**; the 6 configs together are
     the bulk of this phase's GPU time.
   - Bring back: each config's `outputs/finetune_*/best/` checkpoint directory.

### The Phase-2 EM>0 gate

A strict string `==` exact-match metric on un-normalized decoded text can report **0.00%
everywhere** as a whitespace artifact rather than a real result (see `docs/measurement.md`, fixed
in `pop.eval.normalize`). This repo's `pop.eval` already uses the fixed, normalized EM.

**Gate: after generating predictions from a real finetune_A/B checkpoint and scoring them with
`pop eval`, exact match must come out > 0 on at least some samples.** If it's still exactly 0.0
after these fixes:

1. **Don't accept it as "the models just don't get anything exactly right."** At T5-base scale
   on a same-domain refinement task, some fraction of the test set should be trivial (identity
   fixes, whitespace-only diffs, etc.) and land as exact matches once decoding is sane.
2. **Stop and debug decoding before touching anything else.** Specifically: sweep greedy decoding
   vs. beam search (beam width 5) and inspect a handful of raw generations by eye. Degenerate
   repetition (`.METHOD_5().METHOD_5().METHOD_5()...`) in generated text is a decoding/undertraining
   symptom distinct from the metric artifact. If beam-5 fixes it, freeze on beam-5 for all later
   runs and note the change; if it doesn't, the corruption-rate/epoch-count/corpus-overlap
   hypotheses are the next things to check via the compute/data-scaling curves (Phase 3), *not*
   something to paper over by relaxing the metric.

Only once EM>0 is confirmed on a real run should Phase 3 (compute/data-scaling curves) proceed.

## Phase 4 -- RAG evaluation (`colab_rag.ipynb`)

Retrieval-augmented prompting of Qwen2.5-Coder-1.5B-Instruct, avoiding the naive-RAG pitfall
(exemplars truncated at 200 chars with a literal `"..."`, no chat template -- `docs/measurement.md`
§2). No pretraining checkpoint needed; this phase is independent of Phase 2/3.

- Configs: `configs/rag_bm25_k{0,1,3,5}.yaml`, `configs/rag_codebert_k{0,1,3,5}.yaml` (k=0 is the
  zero-shot baseline for each retriever choice -- 8 configs total). Edit the notebook's `CONFIG`
  variable and re-run once per config.
- Rough runtime per config: dominated by loading Qwen2.5-Coder-1.5B and generating over the test
  split; likely 20-60 minutes per config on a T4, more for higher k (longer prompts) --
  **estimate**.
- Bring back: `results/<config-name>.json` (the notebook's run cell chains `pop rag` ->
  `pop eval` so this is written directly) for each of the 8 configs.

## Phase 5 -- LoRA bridge (`colab_lora.ipynb`)

LoRA-adapts `Qwen2.5-Coder-1.5B-Instruct` on the refinement pairs (`configs/lora_qwen.yaml`),
generates on the test split, and scores with `pop eval` — landing `results/lora_qwen_test.json` in
the same schema as the other arms. Run it end to end with `colab_lora.ipynb` (train → generate →
eval); see `docs/gpu-runbook-final.md` Step 3 for the exact cells.

## Optional: arm-B execution point (`colab_execbench.ipynb`)

The execution-eval notebook runs all four arms over the 201 QuixBugs + HumanEval-Java bugs. Arm B's
execution point **has now been run** (via the **"Arm B — from-scratch T5"** cells in
`colab_execbench.ipynb`: finetune `configs/finetune_B_seed0.yaml` from a random init → generate over
both benches → score → `results/execbench_B.json`). The result: **0.0% compile / 0.0% pass**, identical
to arm A — confirming the whole-file-vs-method mismatch is shared by both T5 arms. `execbench_B.json` is
committed, Figure 3 now shows the fourth point, and the report states the measured number rather than an
expectation. To reproduce, rerun those cells (idempotent — they skip if the artifact is already on
Drive) and `python scripts/figures/make_all.py` locally.

## What to bring back, in general

For every phase: the `results/<name>.json` file(s) (schema: `pop.eval.metrics.write_results` --
config, metrics, n, timestamp, git_sha) go into this repo's `results/` directory. Checkpoint
directories (`outputs/pretrain/final/`, `outputs/finetune_*/best/`) are large and don't need to
live in git; keep them wherever your later analysis/curve scripts expect them (the later
analysis/curve phases), and note the location in a `DECISIONS.md`-style log entry if it's not
obvious from the config's `output_dir`.
