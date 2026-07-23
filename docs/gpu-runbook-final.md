# Consolidated GPU runbook — run the whole study in one Colab sitting

> **Status: complete** — these runs have been executed; the results are in
> [`report.md`](report.md). This document is kept as the historical operational runbook.

This is **the** step-by-step to generate every remaining result of the "pretrain, fine-tune,
or prompt?" study in a single Colab GPU session, and to check at the end that nothing is
missing. It sequences the four arms' GPU work (RAG, scaling, LoRA, execution-eval) plus the
analysis aggregation, respecting the one real dependency (the scaling curves need a fresh pretrain).

Everything here is **build-complete and CPU-tested** already. This document is the run *you* press
in *your* Colab account. No secrets: W&B stays optional and is never required by any cell.

- **Inventory of what "done" means:** [`docs/results-manifest.md`](results-manifest.md).
- **Cycle-2 background** (GPU choice table, disconnect/resume mechanics, EM-gate meaning):
  [`docs/colab-runbook.md`](colab-runbook.md).
- **The report the results flow into:** [`docs/report.md`](report.md) — already filled with the
  committed numbers; re-running this batch reproduces them from scratch rather than trusting the
  committed `results/*.json`.

## Before you start — build & upload the Colab payload

The private repo travels to Colab as a **zip on your Drive** (no token-based `git clone`).
Rebuild it from the exact branch HEAD you reviewed, then upload it:

```bash
mkdir -p dist
git archive --format=zip -o dist/pop_repo.zip HEAD
```

Verify locally that it contains this cycle's new code, then upload `dist/pop_repo.zip` to the
**one shared workspace** `Drive/MyDrive/pop_cycle3/pop_repo.zip` — all three cycle-3 notebooks
unzip from this single location (into `/content/repo`), so you upload it just once and every
arm's results co-locate under one `results/` for the Step-5 aggregation:

```bash
python -c "import zipfile; z=zipfile.ZipFile('dist/pop_repo.zip'); \
print('scripts/build_scaling_csv.py' in z.namelist(), \
'scripts/build_execbench_agreement_csv.py' in z.namelist(), \
'notebooks/colab_scaling.ipynb' in z.namelist(), \
sum(n.startswith('configs/finetune_scale_') for n in z.namelist()), 'scale configs')"
```

**GPU:** an **A100** (Colab Pro) runs the whole batch comfortably; L4/T4 also work but slower
(the trainers auto-scale the micro-batch to VRAM and preserve effective batch 64). Pick
Runtime → Change runtime type → GPU before running any notebook.

**Drive layout the notebooks expect:** all three cycle-3 notebooks share **one** Drive
workspace, `Drive/MyDrive/pop_cycle3/`, and each symlinks `outputs/`, `results/`, `logs/` into
it so artifacts survive session resets. Because the `outputs/` subdirs (`rag_*`,
`finetune_scale_*`, `lora_qwen`, …), the `results/*.json` filenames, and each sweep's `logs/`
run-dir are all distinctly named, the shared folder is safe — and it is exactly what lets
Step 5 aggregate every arm's results from a single `results/`. Because Step 0's pretrain and
Step 2's scaling sweep both run in `colab_scaling.ipynb`, **run them in the same notebook
session** (Step 0 first).

---

## Step 0 — Prerequisite: fresh pretrain (produces the scaling checkpoints)

**Why this is mandatory.** The scaling sweep (Step 2) finetunes *from* pretrain checkpoints:
arm-A data configs load `outputs/pretrain/final`, and the pretrain-compute configs load
`outputs/pretrain/epoch-{1,3}`. The stable `epoch-{1,3}` dirs are a **this-cycle** change
(the milestone callback in `src/pop/train/pretrain.py`); **the earlier pretrain run predates it**,
so pretrain must be **re-run with the current code** to produce them. Seed 42, ~1 h on A100.

> Documented minor inconsistency: the 52K data point and the ep10 pretrain-compute point reuse
> the **committed cycle-2** runs (`finetune_A_ep10` / `finetune_B_seed{0,1}`), which came from a
> *different* pretrain instance than this fresh one. The curves are internally consistent within
> each sweep; the reused top points are labelled as reference and this seam is noted in the report.

**Run it inside `colab_scaling.ipynb`** (so it uses the zip-installed cycle-3 `pop` and writes to
the Drive-symlinked `outputs/`), as the first work of that session:

1. Open `notebooks/colab_scaling.ipynb`; run cells **0–4** (Python/GPU check → mount Drive →
   unzip `pop_repo.zip` + `pip install -e .` → symlink `outputs/`/`results/`/`logs/` to
   `Drive/pop_cycle3`). W&B cell (5) is optional — skip it.
2. Insert and run a **Step-0 cell** with these two commands (tokenizer first, then pretrain):

   ```python
   from pop.data.corpus import load_pretraining_corpus
   from pop.tokenizer.train import train_tokenizer
   corpus = load_pretraining_corpus(50000, seed=42)
   train_tokenizer(corpus, "outputs/tokenizer/tokenizer.model", vocab_size=16384)
   !pop pretrain --config configs/pretrain_10ep.yaml
   ```

| | |
|---|---|
| **Cell to run** | the Step-0 cell above, after `colab_scaling.ipynb` cells 0–4 |
| **Expected artifacts** | `outputs/tokenizer/tokenizer.model`; `outputs/pretrain/final`, `outputs/pretrain/epoch-1`, `outputs/pretrain/epoch-3`, `outputs/pretrain/epoch-10` (all on `Drive/pop_cycle3`) |
| **Verify (one line)** | `!ls outputs/pretrain/final outputs/pretrain/epoch-1 outputs/pretrain/epoch-3` all list a `config.json` + weights |
| **Resume after disconnect** | The T5 pretrainer auto-resumes from the latest `outputs/pretrain/checkpoint-*` on Drive; re-run the `pop pretrain` line. If `epoch-{1,3}` already exist and load, Step 0 is done — skip straight to Step 2. |

Do **not** use the old `notebooks/colab_pretrain.ipynb` for this: it is a cycle-1 launcher that
`git clone`s an old branch (no `epoch-{1,3}` dirs) and uses interactive W&B — it will not produce
the artifacts Step 2 needs.

---

## Step 1 — RAG sweep (arm C)

The 8-config zero-/few-shot Qwen sweep (bm25 × codebert, k ∈ {0,1,3,5}). Independent of Step 0
(uses Qwen, not the T5 tokenizer), so it can run in its own session.

1. Open `notebooks/colab_rag.ipynb`; run cells **0–4** (setup; symlinks to `Drive/pop_cycle3`).
2. Run cell **6** — `!python scripts/run_rag.py --list` — the CPU-only preflight/plan (16 steps).
3. Run cell **7** — `!python scripts/run_rag.py` — the long sweep. Cell 8 prints a retriever×k table.

| | |
|---|---|
| **Cells to run** | `colab_rag.ipynb` 0–4 (setup), 6 (`--list` preflight), 7 (sweep), 8 (summary) |
| **Expected artifacts** | `results/rag_{bm25,codebert}_k{0,1,3,5}_test.json` — **8 files** |
| **Verify (one line)** | `!ls results/rag_*_test.json | wc -l` prints **8** |
| **Resume after disconnect** | `run_rag.py` is done-marker resumable — re-run cell 7; each config's `predictions.jsonl` / `results/*.json` marker skips completed steps. Live progress in `logs/rag/**/STATUS.md` on Drive. |

---

## Step 2 — Scaling sweeps (arms A/B curves) — needs Step 0

Both curves in one resumable command (14 configs → 42 steps): the data-scaling curve
(A/B × train_n {1k,5k,15k} × seed {0,1}) and the pretrain-compute curve (A × pretrain-epochs
{1,3}). Continue in the **same `colab_scaling.ipynb` session as Step 0** (checkpoints already on
`Drive/pop_cycle3`).

1. Run cell **6** — `!python scripts/run_scaling.py --list` — the CPU-only plan + a **config**
   preflight. Know what it does and does not check: `preflight()` verifies each config's
   *wiring* (tokenizer path, a distinct `output_dir`, and the declared `pretrained_model_path` /
   `train_n` each curve requires) — it does **not** stat the checkpoints on disk, so a missing
   `outputs/pretrain/{final,epoch-1,epoch-3}` still passes here and would only fail later at the
   first finetune's model load. The real checkpoint-existence check is Step 0's
   `!ls outputs/pretrain/final outputs/pretrain/epoch-1 outputs/pretrain/epoch-3`; confirm that
   listed those dirs before relying on this sweep.
2. Run cell **7** — `!python scripts/run_scaling.py` — the long sweep (finetune → generate → eval
   per config). Cell 8 prints the results table.

| | |
|---|---|
| **Cells to run** | `colab_scaling.ipynb` 6 (`--list` preflight), 7 (sweep), 8 (summary) |
| **Expected artifacts** | `results/finetune_scale_{A,B}_n{1k,5k,15k}_seed{0,1}_test.json` (**12**) + `results/finetune_ptcompute_ep{1,3}_seed42_test.json` (**2**) |
| **Verify (one line)** | `!ls results/finetune_scale_*_test.json results/finetune_ptcompute_*_test.json | wc -l` prints **14** |
| **Resume after disconnect** | `run_scaling.py` is done-marker resumable (finetune/generate/eval markers); the T5 finetuner also mid-step resumes from the latest `checkpoint-*`. Re-run cell 7. Live `logs/scaling/**/STATUS.md`. |

The **52K** data point and the **ep10** pretrain-compute point are intentionally NOT run here —
they reuse the committed `finetune_A_ep10` / `finetune_B_seed{0,1}` results (joined in at Step 5).

---

## Step 3 — LoRA train → generate → eval (arm D)

LoRA-adapt Qwen2.5-Coder-1.5B on the refinement pairs, generate on test, score.

1. Open `notebooks/colab_lora.ipynb`; run cells **0–4** (setup; symlinks to `Drive/pop_cycle3`).
2. Run cell **7** (train adapter), **8** (generate on test), **9** (eval). Cell 10 prints metrics.

| | |
|---|---|
| **Cells to run** | `colab_lora.ipynb` 0–4 (setup), 7 (`pop lora`), 8 (`pop lora-generate`), 9 (`pop eval --name lora_qwen_test`), 10 (summary) |
| **Expected artifacts** | `outputs/lora_qwen/best` (adapter); `outputs/lora_qwen/predictions_test.jsonl`; `results/lora_qwen_test.json` — **1** results file |
| **Verify (one line)** | `!ls results/lora_qwen_test.json` and `!ls outputs/lora_qwen/best/adapter_config.json` both exist |
| **Resume after disconnect** | The LoRA trainer auto-resumes from the latest `outputs/lora_qwen/checkpoint-*` on Drive (re-run cell 7); generate/eval (8/9) are cheap and idempotent — just re-run them. |

**Two LoRA sanity checks to eyeball once on the real Qwen (do NOT block the batch on them):**

- **(a) Chat-template turn boundary.** In cell 7's environment, print one built training prompt and
  confirm the assistant-turn boundary renders where you expect:
  `from pop.train.lora import build_lora_prompt; from transformers import AutoTokenizer;
  print(build_lora_prompt(AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct"), "buggy();"))`
- **(b) Train/inference tokenization parity.** Training tokenizes the chat prompt with
  `add_special_tokens=False`; the inference pipeline adds special tokens itself. For Qwen2.5 (no
  BOS) these agree, but glance at one generated fix to confirm it is not shifted/off-by-a-token.

Both are expected-fine for Qwen2.5; note anything surprising in the report's limitations rather
than halting.

---

## Step 4 — Execution-eval predictions + harness (per arm)

Feed each arm's generator over the 201 vendored bugs → `{bug_id, prediction, bench}` jsonl →
score with the JDK harness (`pop execbench --predictions`, already 201/201 on references).

> **One-Run-all option:** [`notebooks/colab_execbench.ipynb`](../notebooks/colab_execbench.ipynb)
> runs this whole step (arms A/C/D, both benches, JDK install + scoring) in a single Run-all, with
> idempotent skip-guards so a disconnect resumes. It auto-finetunes arm A's T5 if
> `outputs/finetune_A_ep10/best` isn't on Drive, and auto-picks arm C = the highest-CodeBLEU RAG
> config. The manual cells below remain the reference for exactly what it does.

**Generate `--bench quixbugs` and `--bench humaneval_java` as SEPARATE cells**, because
`--bench all` reloads the arm's model once per benchmark; splitting them loads it once per cell.
Then concatenate the two jsonl and score once (each record carries its own `bench`).

**Arm A (pretrained→finetuned T5)** — needs a finetuned arm-A T5 dir. Reuse the cycle-2
`outputs/finetune_A_ep10/best` if it is on Drive; otherwise re-finetune it first (it loads the
Step-0 pretrain): `!pop finetune --config configs/finetune_A_ep10.yaml`.

```bash
# generate (two cells)
python scripts/gen_execbench_predictions.py --arm t5 \
  --model outputs/finetune_A_ep10/best --tokenizer outputs/tokenizer/tokenizer.model \
  --bench quixbugs      --out outputs/execbench/A_quixbugs.jsonl
python scripts/gen_execbench_predictions.py --arm t5 \
  --model outputs/finetune_A_ep10/best --tokenizer outputs/tokenizer/tokenizer.model \
  --bench humaneval_java --out outputs/execbench/A_humaneval_java.jsonl
# score (one cell)
cat outputs/execbench/A_quixbugs.jsonl outputs/execbench/A_humaneval_java.jsonl \
  > outputs/execbench/A_all.jsonl
pop execbench --predictions outputs/execbench/A_all.jsonl --name execbench_A
```

**Arm B (from-scratch T5)** — *optional* (near-identical CodeBLEU to A). Same as arm A but
`--model outputs/finetune_B_seed0/best` and `--name execbench_B`.

**Arm C (RAG Qwen)** — pick the best RAG config from Step 1 (highest CodeBLEU; e.g. `rag_bm25_k3`):

```bash
python scripts/gen_execbench_predictions.py --arm rag --config configs/rag_bm25_k3.yaml \
  --bench quixbugs      --out outputs/execbench/C_quixbugs.jsonl
python scripts/gen_execbench_predictions.py --arm rag --config configs/rag_bm25_k3.yaml \
  --bench humaneval_java --out outputs/execbench/C_humaneval_java.jsonl
cat outputs/execbench/C_*.jsonl > outputs/execbench/C_all.jsonl
pop execbench --predictions outputs/execbench/C_all.jsonl --name execbench_C
```

**Arm D (LoRA Qwen)** — reads the adapter from `outputs/lora_qwen/best` (Step 3):

```bash
python scripts/gen_execbench_predictions.py --arm lora --config configs/lora_qwen.yaml \
  --bench quixbugs      --out outputs/execbench/D_quixbugs.jsonl
python scripts/gen_execbench_predictions.py --arm lora --config configs/lora_qwen.yaml \
  --bench humaneval_java --out outputs/execbench/D_humaneval_java.jsonl
cat outputs/execbench/D_*.jsonl > outputs/execbench/D_all.jsonl
pop execbench --predictions outputs/execbench/D_all.jsonl --name execbench_D
```

| | |
|---|---|
| **Expected artifacts** | `results/execbench_A.json`, `results/execbench_C.json`, `results/execbench_D.json` (+ optional `results/execbench_B.json`) |
| **Verify (one line)** | each prints `n: 201` and a `per_benchmark` block with `quixbugs` (40) + `humaneval_java` (161) |
| **Resume after disconnect** | Prediction jsonl are plain files on Drive — re-run only the arm/bench cell whose `.jsonl` is missing, then re-score. Scoring is a CPU JDK job (no GPU); safe to re-run. |

> Expect **low compile rates** here: every arm was trained on single *methods* but is fed whole
> buggy *files* — that whole-file-vs-method mismatch is itself the Track-2 finding (see the
> header of `scripts/gen_execbench_predictions.py`), not a bug to fix in this runbook.

---

## Step 5 — Aggregate CSVs + render figures

Turn the JSONs into the two analysis CSVs, then render every figure. All CPU, seconds. Because
every arm wrote into the same shared `results/` on `Drive/pop_cycle3`, all the JSONs are already
co-located here, so these builders (each reads that one `results/` dir) see the whole study at
once.

```bash
python scripts/build_scaling_csv.py               # -> results/scaling_data.csv
python scripts/build_execbench_agreement_csv.py   # -> results/execbench_agreement.csv
python scripts/figures/make_all.py                # -> docs/figures/*.png (real data now)
```

| | |
|---|---|
| **Expected artifacts** | `results/scaling_data.csv`, `results/execbench_agreement.csv`; refreshed `docs/figures/{four_arm_comparison,scaling_curves,execution_vs_codebleu}.png` |
| **Verify (one line)** | `scaling_data.csv` has data rows at x∈{1000,5000,15000,52364} for A/B and ptcompute rows at x∈{1,3,10}; the figures render from real data (no "illustrative fixture" label — the committed CSVs already guarantee this, `make_all.py` rebuilds them) |
| **Resume after disconnect** | Fully deterministic and idempotent — just re-run. The builders are graceful-partial, so running them after only some arms are done still writes a valid (partial) CSV + figure. |

Both builders read whatever `results/*.json` exist and emit only rows they can back with data, so
you can run Step 5 mid-batch to watch the study fill in.

---

## Completeness checklist — every expected `results/*.json`

Tick each once its file exists on Drive. Full inventory with status also in
[`docs/results-manifest.md`](results-manifest.md).

**Reference (already committed from cycle 2 — reused, not re-run):**
- [x] `results/finetune_A_ep10_test.json`  (arm A 52K + ep10 references)
- [x] `results/finetune_B_seed0_test.json`, `results/finetune_B_seed1_test.json`  (arm B 52K)
- [x] `results/finetune_B_seed2_test.json`  (extra arm-B seed, for the four-arm figure band)
- [x] `results/execbench_validate_references.json`  (harness sanity, 201/201)

**Step 1 — RAG (8):**
- [ ] `results/rag_bm25_k0_test.json`   - [ ] `results/rag_codebert_k0_test.json`
- [ ] `results/rag_bm25_k1_test.json`   - [ ] `results/rag_codebert_k1_test.json`
- [ ] `results/rag_bm25_k3_test.json`   - [ ] `results/rag_codebert_k3_test.json`
- [ ] `results/rag_bm25_k5_test.json`   - [ ] `results/rag_codebert_k5_test.json`

**Step 2 — Scaling data curve (12):**
- [ ] `results/finetune_scale_A_n1k_seed0_test.json`  - [ ] `results/finetune_scale_A_n1k_seed1_test.json`
- [ ] `results/finetune_scale_A_n5k_seed0_test.json`  - [ ] `results/finetune_scale_A_n5k_seed1_test.json`
- [ ] `results/finetune_scale_A_n15k_seed0_test.json` - [ ] `results/finetune_scale_A_n15k_seed1_test.json`
- [ ] `results/finetune_scale_B_n1k_seed0_test.json`  - [ ] `results/finetune_scale_B_n1k_seed1_test.json`
- [ ] `results/finetune_scale_B_n5k_seed0_test.json`  - [ ] `results/finetune_scale_B_n5k_seed1_test.json`
- [ ] `results/finetune_scale_B_n15k_seed0_test.json` - [ ] `results/finetune_scale_B_n15k_seed1_test.json`

**Step 2 — Pretrain-compute curve (2):**
- [ ] `results/finetune_ptcompute_ep1_seed42_test.json`
- [ ] `results/finetune_ptcompute_ep3_seed42_test.json`

**Step 3 — LoRA (1):**
- [ ] `results/lora_qwen_test.json`

**Step 4 — Execution-eval (3 required + 1 optional):**
- [ ] `results/execbench_A.json`   (arm A T5, required)
- [ ] `results/execbench_C.json`   (arm C RAG, required)
- [ ] `results/execbench_D.json`   (arm D LoRA, required)
- [ ] `results/execbench_B.json`   (arm B T5, optional)

**Step 5 — Derived (2 CSVs, regenerated locally too):**
- [ ] `results/scaling_data.csv`
- [ ] `results/execbench_agreement.csv`

**Total to produce this batch: 8 + 14 + 1 + 3 (+1 optional) = 26–27 JSONs + 2 CSVs.**

When every required box is ticked: zip `results/` (and any `outputs/` you want to keep) back to
your local clone, commit them on a results-ingest branch, re-run `scripts/figures/make_all.py`
locally, and fill the numbers in `docs/report.md`. That closes the study.
