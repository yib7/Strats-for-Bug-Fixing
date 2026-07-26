# GPU reproduction: environments and launch order

Everything in this repo that runs without a GPU is verified on CPU: the `pop` package,
data/tokenizer/model/eval modules, the RAG pipeline, the execution harness, and the smoke
pipeline. Retraining the T5 models and running the RAG/LoRA evaluations needs a GPU.

This document covers the **environments** (Colab and a local AMD card) and the **launch order**
per arm. For the consolidated single-sitting Colab batch, see
[`gpu-runbook.md`](gpu-runbook.md); for the resumable-orchestrator details and the GPU-choice
table, see [`colab-runbook.md`](colab-runbook.md).

W&B is optional throughout: `pop` enables it only when `WANDB_API_KEY` is set (see
`.env.example`) and runs local-only otherwise.

## Local GPU: AMD RX 9070 (ROCm on Windows)

An RX 9070 (RDNA4/gfx1201) works with AMD's PyTorch-on-Windows ROCm wheels (torch
2.9.1+rocm7.2.1, Python 3.12): `torch.cuda.is_available()` is True, Trainer selects `cuda:0`,
bf16 is supported, and `pop smoke` runs end-to-end on the GPU in ~15 s. Set the environment up
or repair it with:

```
python scripts/setup_rocm_windows.py
```

That script creates a **separate** env `.venv-rocm` (the stable CPU env `.venv` is untouched),
installs the AMD wheels, and applies two documented workarounds (see the script's docstring): a
`sitecustomize.py` shim for the missing `torch.distributed` extension, and a one-line patch for a
transformers 5.14 `DTensor` guard bug. Requires Adrenalin driver >= 26.2.2.

Run any phase locally by swapping the runner prefix. Every `pop` command works as:

```
.venv-rocm\Scripts\pop pretrain --config configs/pretrain_10ep.yaml
```

Caveats of the local path:

- **Preview-grade stack.** ROCm-on-Windows is new; if a run misbehaves (NaNs, crashes, wildly-off
  losses), rerun the same config on Colab before trusting any conclusion. Numbers destined for the
  write-up should be reproduced on Colab (CUDA) at least once.
- **No vLLM** on Windows/ROCm: `pop rag` automatically uses the transformers fallback path.
- **W&B logging** works the same: `wandb login` once inside `.venv-rocm`, or leave `WANDB_API_KEY`
  unset and metrics stay local-only.

The Colab notebooks are the primary path and the reproduction check for every arm.

## Before launching anything: `pop smoke`

```
uv run pop smoke
```

This runs the *entire* pipeline shape (tokenizer train → pretrain → finetune → generate → score)
end-to-end on CPU in well under a minute, against tiny committed fixtures (`tests/fixtures/smoke_*`,
`configs/smoke.yaml`). It proves the code path works before spending GPU time on it. If `pop smoke`
fails, nothing downstream will work either. A passing run writes `results/smoke_local.json` with
non-null `codebleu`/`syntax_valid_rate` (exact match is expected to be 0.0 at this scale, since it
is a plumbing check rather than a quality bar). That file is gitignored scratch; the committed
`results/smoke.json` from the verified reference run is never touched by a local `pop smoke`.

**Before every notebook launch:** open the notebook's install cell and pin the exact commit you
intend to run (see each notebook's install-cell markdown) rather than a moving branch tip.

Runtimes below are **estimates** for a single Colab T4; actual time depends on which GPU Colab
allocates and on current queue/throttling. Treat them as "roughly this much, budget more."

## T5 arms A and B: core retraining

Runs the arm A ("pretrained + fine-tuned") vs. arm B ("fine-tuned, no pre-training") comparison.
The A-vs-B CodeBLEU difference is small, so this stage includes the seed-variance ablation needed
to tell a real gap from noise. It all runs from one notebook, `notebooks/colab_phase2.ipynb`,
driving the resumable orchestrator `scripts/run_training.py` (see
[`colab-runbook.md`](colab-runbook.md) for the Drive setup and the disconnect/resume mechanics).
The orchestrator's step order is:

1. **Tokenizer + pretrain.** `pop tokenizer` trains the vocab-16384 SentencePiece model on the
   CodeSearchNet-Java corpus, then `pop pretrain --config configs/pretrain_10ep.yaml` runs T5
   span-corruption pretraining for 10 epochs, checkpointing at epochs 1/3/10.
   - Rough runtime: tokenizer training a few minutes; pretraining is the long pole, likely several
     hours for 50K methods × 10 epochs on a t5-small-sized model on a T4. **Estimate; confirm
     against your own first-epoch wall time and extrapolate.**
   - Produces: `outputs/tokenizer/` and `outputs/pretrain/final/` (plus the epoch-1/epoch-3
     checkpoints the scaling curves consume).
2. **Six finetune → generate → eval cycles**, one per system, headline comparison first
   (`A_ep10`, `B_seed0`, then `A_ep3`, `A_ep1`, `B_seed1`, `B_seed2`):
   - `configs/finetune_A_ep{1,3,10}.yaml` is arm A (pretrained + finetuned), epochs swept at fixed
     seed 42.
   - `configs/finetune_B_seed{0,1,2}.yaml` is arm B (finetuned only; these configs deliberately omit
     `pretrained_model_path`, so `pop finetune` builds a fresh random init), same 10 epochs, seed
     swept for variance.
   - Rough runtime per config: finetuning the t5-small-sized model on the CodeXGLUE-medium train split, tens of
     minutes to ~1–2 hours depending on epoch count (**estimate**); these six configs are the bulk
     of the stage's GPU time.
   - Produces: each config's `outputs/finetune_*/best/` checkpoint directory and
     `results/finetune_{A_ep1,A_ep3,A_ep10,B_seed0,B_seed1,B_seed2}_test.json`. Intermediate
     `checkpoint-*` dirs are pruned after each system's eval to bound Drive usage.

### The EM>0 gate

A strict string `==` exact-match metric on un-normalized decoded text can report **0.00% everywhere**
as a whitespace artifact rather than a real result (see [`measurement.md`](measurement.md); fixed in
`pop.eval.normalize`). `pop.eval` uses the fixed, normalized EM.

`scripts/run_training.py` therefore **halts** (exit code 3) if the first system's exact match comes
out exactly 0.0, rather than burning GPU quota on five more all-zero systems. The intended response
is to sweep greedy vs. beam-5 decoding (`pop generate --num-beams 5`) and inspect raw generations
before continuing: degenerate repetition (`.METHOD_5().METHOD_5()...`) is a decoding/undertraining
symptom distinct from the metric artifact. Re-run with `--skip-gate` once diagnosed.

In this study the gate was exercised and the 0-EM result was audited and confirmed genuine rather
than a decoding artifact: the metric agrees four ways and predictions are complete, valid Java. See
[`report.md`](report.md) for that analysis.

## Arm C: RAG evaluation (`colab_rag.ipynb`)

Retrieval-augmented prompting of Qwen2.5-Coder-1.5B-Instruct, avoiding the naive-RAG pitfall
(exemplars truncated at 200 chars with a literal `"..."`, no chat template; see
[`measurement.md`](measurement.md) §2). Independent of the T5 arms.

- Configs: `configs/rag_bm25_k{0,1,3,5}.yaml`, `configs/rag_codebert_k{0,1,3,5}.yaml` (k=0 is the
  zero-shot baseline for each retriever, 8 configs total).
- Rough runtime per config: dominated by loading Qwen2.5-Coder-1.5B and generating over the test
  split; likely 20–60 minutes per config on a T4, more for higher k (**estimate**).
- Produces: `results/rag_*_test.json` for each of the 8 configs (the run cell chains `pop rag` →
  `pop eval`).

## Arm D: LoRA bridge (`colab_lora.ipynb`)

LoRA-adapts `Qwen2.5-Coder-1.5B-Instruct` on the refinement pairs (`configs/lora_qwen.yaml`),
generates on the test split, and scores with `pop eval`, landing `results/lora_qwen_test.json` in
the same schema as the other arms. Run it end to end with `colab_lora.ipynb` (train → generate →
eval); see [`gpu-runbook.md`](gpu-runbook.md) Step 3 for the exact cells.

## Execution eval (`colab_execbench.ipynb`)

The execution-eval notebook runs all four arms over the 201 QuixBugs + HumanEval-Java bugs. Each arm
generates over both benches, then the JDK harness compiles and runs the candidate fixes. Arm B is
covered by the **"Arm B, from-scratch T5"** cells (finetune `configs/finetune_B_seed0.yaml` from a
random init → generate over both benches → score → `results/execbench_B.json`); it measures 0.0%
compile / 0.0% pass, identical to arm A, which is what pins the whole-file-vs-method mismatch as
shared by both T5 arms. The cells are idempotent: they skip if the artifact is already on Drive.
Re-render figures afterwards with `python scripts/figures/make_all.py`.

## Re-running a config that already has a committed result

Every `results/*.json` in this repo does double duty: it is a published measurement **and**
it is the done-marker the orchestrators (`scripts/run_training.py`, `run_rag.py`,
`run_scaling.py`) use to decide whether a step still needs doing. On a fresh clone all of
them already exist, so a full sweep will happily retrain and regenerate, then **skip every
eval step** and leave the committed numbers in place — unlinked from the models you just
trained. That is the safe default, not an accident, but it is silent, so know the rules:

- **`pop eval` / `pop execbench` refuse to overwrite a committed result.** `write_results`
  raises `FileExistsError` (`pop` exits 1) for any name that is not scratch. Scratch means
  the name contains `_local`, which is gitignored (`results/*_local*.json`) and freely
  replaceable — that is why `pop smoke` writes `results/smoke_local.json` and never touches
  the committed `results/smoke.json`.
- **To genuinely reproduce a number, delete that one file first**, then re-run the step.
  The orchestrator then sees the marker missing and runs it, and `write_results` has nothing
  to refuse. Do this deliberately, one file at a time, and diff the result against what git
  still has for it.
- **To compare without touching the published set, pass a different `--name`.** Anything
  with `_local` in it (`--name rag_bm25_k3_local_test`) is scratch: gitignored, overwritable,
  and excluded from the derived CSVs, which filter scratch names out of their globs so a
  local experiment cannot leak into `results/execbench_agreement.csv` or the figures.
- **An empty run is rejected rather than recorded.** `pop execbench` exits 2 with
  `no bugs selected` if `--bench`/`--limit`/the predictions file select nothing, instead of
  writing a legitimate-looking `pass_rate: 0.0` over `n: 0` — and, in the
  `--validate-references` case, instead of exiting 0 as though 201/201 references had passed.
- **The CSV builders write beside their inputs.** `scripts/build_scaling_csv.py` and
  `scripts/build_execbench_agreement_csv.py` default `--out` to
  `<--results-dir>/<name>.csv`, so pointing `--results-dir` at a scratch directory keeps the
  committed CSVs untouched. With no flags they still write into `results/`, which is what
  `scripts/figures/make_all.py` and the documented reproduce step expect.

A partially generated predictions file is a separate mechanism and needs no cleanup:
`pop rag` / `pop lora-generate` checkpoint into `<output>.partial`, resume onto it only when
its sidecar `.partial.meta` proves it answers the same prompts and references, and discard it
with a note on stderr otherwise. Changing `k`, the retriever, the split or `--limit` and
re-running is therefore safe — it restarts that config rather than mixing old predictions in.

## What each run produces

Every arm writes `results/<name>.json` following the `pop.eval.metrics.write_results` schema
(`{config, metrics, n, timestamp, git_sha}`) into this repo's `results/` directory; those files are
the study's source of truth and are committed. Checkpoint directories
(`outputs/pretrain/final/`, `outputs/finetune_*/best/`) are large and intentionally not in git;
keep them wherever the analysis and curve scripts expect them, matching each config's `output_dir`.
[`results-manifest.md`](results-manifest.md) is the full inventory of what a complete study contains.
