# Colab runbook: the T5 arms in one notebook, resumable

The whole T5 retraining batch (tokenizer -> 10-epoch pretrain -> 6 finetuned systems ->
generate -> eval) runs from **one notebook**, `notebooks/colab_phase2.ipynb`, driving the
resumable orchestrator `scripts/run_training.py`. Every artifact (checkpoints, results,
progress logs) lives on **your Google Drive**, so Colab disconnects cost nothing but time.

No GitHub tokens, no pushes, no API keys are needed anywhere in this flow.

## What you need

- A Google account with **~6 GB free Drive space** (checkpoints are pruned as the run
  progresses; steady-state is ~3-4 GB).
- The two files from this repo (both are in the local working copy):
  - `dist/pop_repo.zip`: the code, built with `git archive --format=zip -o dist/pop_repo.zip HEAD`
  - `notebooks/colab_phase2.ipynb`: the notebook

## One-time setup (~2 minutes)

1. In [Google Drive](https://drive.google.com), create a folder named **`pop_phase2`**
   directly under *My Drive*.
2. Drag **`dist/pop_repo.zip`** into that folder.
3. Drag **`notebooks/colab_phase2.ipynb`** anywhere on Drive (e.g. the same folder).

## Each session (~1 minute of clicking, then hands-off)

1. In Drive, double-click `colab_phase2.ipynb` -> **Open with Google Colaboratory**.
2. Runtime -> Change runtime type -> pick a GPU (see the table below; **A100** if you have
   Colab Pro, **T4** on the free tier) -> Save.
3. Runtime -> **Run all**. Approve the "connect to Google Drive" popup when it appears.
4. Leave the tab open. That's it.

## Which GPU? (Colab Pro)

The trainers auto-scale the per-device micro-batch to the GPU's VRAM (T4: 8x8, L4: 16x4,
A100: 32x2; the effective batch of 64, the frozen hyperparameter, never changes), so any
choice is correct; they differ only in wall-clock and compute-unit burn. Rough, to be
calibrated by the first epoch's STATUS.md timings:

| GPU            | whole-run wall-clock | compute units (~) | sessions needed |
|----------------|----------------------|-------------------|-----------------|
| T4 (free/Pro)  | ~35-50 h             | ~65-90            | many, over days |
| L4 (Pro)       | ~9-14 h              | ~45-65            | 1-2             |
| A100 40GB (Pro)| ~3-5 h               | ~40-60            | usually 1       |

Unit burn rates are roughly T4 ~2/h, L4 ~5/h, A100 ~12/h (Colab shows the exact live rate
under Resources), so the *total unit cost is similar everywhere* and the A100 simply
finishes ~10x sooner. If A100 capacity is unavailable, L4 is the fallback.

One rule when resuming across sessions: **don't switch GPU class while a training step is
mid-run** (its checkpoint's step accounting assumes the same micro-batch). Switching between
steps (i.e. when STATUS.md shows the previous step `done`) is always fine. If you must
switch mid-step, delete that step's `outputs/.../checkpoint-*` dirs on Drive first and let
the step restart.

The run cell streams live training logs. Independently of the notebook, you can check
progress from any device (including your phone) by opening
**`Drive/pop_phase2/logs/train/phase2/STATUS.md`**: a step-by-step table the orchestrator
rewrites continuously. Per-step logs sit next to it.

## When the session ends early (it will, and that's fine)

Free-tier sessions stop after a few hours (usage caps, idle checks, 12 h hard max, weekly
GPU quota of very roughly 15-30 h). Nothing is lost:

- Reopen the notebook -> **Run all** again.
- Finished steps skip instantly (their output already exists on Drive).
- A training step that was interrupted mid-way resumes from its **last completed epoch**
  checkpoint (also on Drive), not from scratch.

Expect the full 6-system matrix to take **several sessions spread over some days** of
free-tier quota. The step order front-loads the headline comparison on purpose: after
`pretrain` + `finetune_A_ep10` + `finetune_B_seed0` (the first ~3 long steps) you already
have the core A-vs-B answer; the remaining systems (A_ep3, A_ep1, B_seed1, B_seed2) only
refine curves and variance. All runtimes are estimates until the first epoch calibrates
them: check STATUS.md durations and extrapolate.

## The EM>0 gate (a deliberate stop, not a crash)

After the first system (`A_ep10`) is generated and scored, the orchestrator **halts if
exact match is exactly 0.0** (exit code 3, clear message in the log). Per
[`gpu-reproduction.md`](gpu-reproduction.md), all-zero EM at this scale points at a decoding bug,
and the next move is a greedy-vs-beam-5 sweep (`pop generate --num-beams 5`) and eyeballing
`outputs/finetune_A_ep10/best/predictions_test.jsonl`, not burning quota on five more
all-zero systems. Once diagnosed, re-run with `--skip-gate` if that's the informed call.

## Done looks like

- `Drive/pop_phase2/results/finetune_{A_ep10,A_ep3,A_ep1,B_seed0,B_seed1,B_seed2}_test.json`
- `Drive/pop_phase2/logs/train/phase2/SUMMARY.md`: the A-vs-B results table

Bring back to the repo: download those 6 JSONs (+ SUMMARY.md) and drop them into the local
`results/` directory (they're tiny). Checkpoints can stay on Drive.

## Troubleshooting

- **`assert ZIP.is_file()` fails**: `pop_repo.zip` isn't at `MyDrive/pop_phase2/`;
  re-check the folder name/location.
- **NaN / non-decreasing loss on a T4**: the T4 has no bf16, so training uses fp16;
  if it ever goes numerically sideways, change the run cell to
  `!POP_FORCE_FP32=1 python scripts/run_training.py` (slower but exact) and re-run.
- **Drive out of space**: free up to ~6 GB; the orchestrator prunes finetune checkpoints
  after each system's eval, and pretrain keeps only milestone epochs (1/3/10) + the latest.
- **Code was updated locally**: rebuild the zip (`git archive --format=zip -o
  dist/pop_repo.zip HEAD`), replace it in `Drive/pop_phase2/`, and Run all again; the
  notebook re-unpacks the zip every session.
- **Python version assert fails**: the Colab runtime is older than 3.11; Runtime ->
  Disconnect and delete runtime, then reconnect (or report back, the floor can be lowered).

## Local-GPU fallback (unchanged)

The same orchestrator runs on an AMD RX 9070: `.venv-rocm\Scripts\python.exe
scripts\run_training.py`. The real configs run at ~5.5 GB of 16 GB VRAM, and the trainers hard-cap
GPU memory (`POP_GPU_MEM_FRACTION`, default 0.85) so the one-time batch-size probes cannot spill
into system RAM. Colab remains the reproduction check for any local number destined for the
write-up (see [`gpu-reproduction.md`](gpu-reproduction.md)).
