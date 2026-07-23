# Results manifest — the full study's expected `results/` inventory

The authoritative list of every results file the complete four-arm study should contain, so
"is the study intact?" is checkable at a glance. ✅ = committed / done; ⬜ = pending the one
Colab GPU batch (run it via [`docs/gpu-runbook-final.md`](gpu-runbook-final.md)).

The four arms: **A** pretrain→finetune T5 · **B** from-scratch T5 · **C** RAG Qwen ·
**D** LoRA Qwen. Every JSON follows the `pop.eval.metrics.write_results` schema
(`{config, metrics, n, timestamp, git_sha}`); execbench JSONs carry
`metrics.{pass_rate, compile_rate, per_benchmark}`.

## Arms A & B — T5 finetune references (committed, cycle 2) ✅

These real cycle-2 A100 results are the study's anchor and are reused as the scaling curves' top
(52K / ep10) points — not re-run in the batch.

| File | Status | Role |
|---|---|---|
| `results/finetune_A_ep1_test.json`  | ✅ | arm A, 1 finetune epoch (four-arm epoch trend) |
| `results/finetune_A_ep3_test.json`  | ✅ | arm A, 3 finetune epochs (four-arm epoch trend) |
| `results/finetune_A_ep10_test.json` | ✅ | arm A headline; 52K data point + ep10 ptcompute point |
| `results/finetune_B_seed0_test.json`| ✅ | arm B seed 0; 52K data point (seed 0) |
| `results/finetune_B_seed1_test.json`| ✅ | arm B seed 1; 52K data point (seed 1) |
| `results/finetune_B_seed2_test.json`| ✅ | arm B seed 2; four-arm seed band |

## Arm C — RAG sweep (8, pending) ⬜

bm25 × codebert, k ∈ {0,1,3,5}. Best-CodeBLEU config becomes arm C in the four-arm + agreement figures.

| File | Status | | File | Status |
|---|---|---|---|---|
| `results/rag_bm25_k0_test.json` | ⬜ | | `results/rag_codebert_k0_test.json` | ⬜ |
| `results/rag_bm25_k1_test.json` | ⬜ | | `results/rag_codebert_k1_test.json` | ⬜ |
| `results/rag_bm25_k3_test.json` | ⬜ | | `results/rag_codebert_k3_test.json` | ⬜ |
| `results/rag_bm25_k5_test.json` | ⬜ | | `results/rag_codebert_k5_test.json` | ⬜ |

## Scaling curves (14 + reused refs, pending) ⬜

**Data-scaling curve (12)** — arms A/B × train_n {1k,5k,15k} × seed {0,1}:

| File | Status | | File | Status |
|---|---|---|---|---|
| `results/finetune_scale_A_n1k_seed0_test.json`  | ⬜ | | `results/finetune_scale_B_n1k_seed0_test.json`  | ⬜ |
| `results/finetune_scale_A_n1k_seed1_test.json`  | ⬜ | | `results/finetune_scale_B_n1k_seed1_test.json`  | ⬜ |
| `results/finetune_scale_A_n5k_seed0_test.json`  | ⬜ | | `results/finetune_scale_B_n5k_seed0_test.json`  | ⬜ |
| `results/finetune_scale_A_n5k_seed1_test.json`  | ⬜ | | `results/finetune_scale_B_n5k_seed1_test.json`  | ⬜ |
| `results/finetune_scale_A_n15k_seed0_test.json` | ⬜ | | `results/finetune_scale_B_n15k_seed0_test.json` | ⬜ |
| `results/finetune_scale_A_n15k_seed1_test.json` | ⬜ | | `results/finetune_scale_B_n15k_seed1_test.json` | ⬜ |

**Pretrain-compute curve (2)** — arm A × pretrain-epochs {1,3}, seed 42:

| File | Status |
|---|---|
| `results/finetune_ptcompute_ep1_seed42_test.json` | ⬜ |
| `results/finetune_ptcompute_ep3_seed42_test.json` | ⬜ |

**Reused reference points (no new file):** 52K data point = `finetune_A_ep10` (A) /
`finetune_B_seed{0,1}` (B); ep10 ptcompute point = `finetune_A_ep10`. Joined into
`scaling_data.csv` by `scripts/build_scaling_csv.py`.

## Arm D — LoRA (1, pending) ⬜

| File | Status |
|---|---|
| `results/lora_qwen_test.json` | ⬜ |

## Execution-eval (3 required + 1 optional, pending) ⬜

One per arm over the 201 vendored QuixBugs + HumanEval-Java bugs; pass@1 = `metrics.pass_rate`.

| File | Status | Arm |
|---|---|---|
| `results/execbench_validate_references.json` | ✅ | harness sanity (201/201 on reference patches) |
| `results/execbench_A.json` | ⬜ | arm A T5 (required) |
| `results/execbench_C.json` | ⬜ | arm C RAG (required) |
| `results/execbench_D.json` | ⬜ | arm D LoRA (required) |
| `results/execbench_B.json` | ⬜ | arm B T5 (optional — near-identical CodeBLEU to A) |

## Derived analysis CSVs (2, pending) ⬜

Regenerated from the JSONs above by the aggregators; `.gitignore`d (rebuilt in Colab and locally,
not committed). The committed figures render from `tests/fixtures/*.csv` until these exist.

| File | Built by | Read by | Status |
|---|---|---|---|
| `results/scaling_data.csv` | `scripts/build_scaling_csv.py` | `scripts/figures/scaling_curves.py` | ⬜ |
| `results/execbench_agreement.csv` | `scripts/build_execbench_agreement_csv.py` | `scripts/figures/execution_vs_codebleu.py` | ⬜ |

## Not study results (present in `results/`, ignore for completeness)

`smoke.json` (CPU smoke), `phase2_summary.md` (T5 finetune batch write-up).

## Tally

| Group | Files | Done |
|---|---|---|
| A/B finetune references | 6 | 6 ✅ |
| Execbench references | 1 | 1 ✅ |
| RAG (arm C) | 8 | 0 |
| Scaling data curve | 12 | 0 |
| Pretrain-compute curve | 2 | 0 |
| LoRA (arm D) | 1 | 0 |
| Execbench arms (A/C/D + optional B) | 3–4 | 0 |
| Derived CSVs | 2 | 0 |
| **Study total** | **35–36** | **7** |

**26–27 result JSONs + 2 CSVs remain**, all produced by the single Colab batch in
[`docs/gpu-runbook-final.md`](gpu-runbook-final.md).
