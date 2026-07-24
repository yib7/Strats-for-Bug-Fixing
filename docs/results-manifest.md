# Results manifest — the full study's `results/` inventory

The authoritative list of every results file the four-arm study contains, so "is the study intact?"
is checkable at a glance. ✅ = committed. Every artifact is present, including `execbench_B.json`
(arm B execution — see [`gpu-reproduction.md`](gpu-reproduction.md)).

The four arms: **A** pretrain→finetune T5 · **B** from-scratch T5 · **C** RAG Qwen ·
**D** LoRA Qwen. Every JSON follows the `pop.eval.metrics.write_results` schema
(`{config, metrics, n, timestamp, git_sha}`); execbench JSONs carry
`metrics.{pass_rate, compile_rate, per_benchmark}`.

## Arms A & B — T5 finetune references ✅

These real A100 results are the study's anchor and are reused as the scaling curves' top
(52K / ep10) points.

| File | Status | Role |
|---|---|---|
| `results/finetune_A_ep1_test.json`  | ✅ | arm A, 1 finetune epoch (four-arm epoch trend) |
| `results/finetune_A_ep3_test.json`  | ✅ | arm A, 3 finetune epochs (four-arm epoch trend) |
| `results/finetune_A_ep10_test.json` | ✅ | arm A headline; 52K data point + ep10 ptcompute point |
| `results/finetune_B_seed0_test.json`| ✅ | arm B seed 0; 52K data point (seed 0) |
| `results/finetune_B_seed1_test.json`| ✅ | arm B seed 1; 52K data point (seed 1) |
| `results/finetune_B_seed2_test.json`| ✅ | arm B seed 2; four-arm seed band |

## Arm C — RAG sweep (8) ✅

bm25 × codebert, k ∈ {0,1,3,5}. Best-CodeBLEU config (codebert_k1) is arm C in the four-arm +
agreement figures.

| File | Status | | File | Status |
|---|---|---|---|---|
| `results/rag_bm25_k0_test.json` | ✅ | | `results/rag_codebert_k0_test.json` | ✅ |
| `results/rag_bm25_k1_test.json` | ✅ | | `results/rag_codebert_k1_test.json` | ✅ |
| `results/rag_bm25_k3_test.json` | ✅ | | `results/rag_codebert_k3_test.json` | ✅ |
| `results/rag_bm25_k5_test.json` | ✅ | | `results/rag_codebert_k5_test.json` | ✅ |

## Scaling curves (14 + reused refs) ✅

**Data-scaling curve (12)** — arms A/B × train_n {1k,5k,15k} × seed {0,1}:

| File | Status | | File | Status |
|---|---|---|---|---|
| `results/finetune_scale_A_n1k_seed0_test.json`  | ✅ | | `results/finetune_scale_B_n1k_seed0_test.json`  | ✅ |
| `results/finetune_scale_A_n1k_seed1_test.json`  | ✅ | | `results/finetune_scale_B_n1k_seed1_test.json`  | ✅ |
| `results/finetune_scale_A_n5k_seed0_test.json`  | ✅ | | `results/finetune_scale_B_n5k_seed0_test.json`  | ✅ |
| `results/finetune_scale_A_n5k_seed1_test.json`  | ✅ | | `results/finetune_scale_B_n5k_seed1_test.json`  | ✅ |
| `results/finetune_scale_A_n15k_seed0_test.json` | ✅ | | `results/finetune_scale_B_n15k_seed0_test.json` | ✅ |
| `results/finetune_scale_A_n15k_seed1_test.json` | ✅ | | `results/finetune_scale_B_n15k_seed1_test.json` | ✅ |

**Pretrain-compute curve (2)** — arm A × pretrain-epochs {1,3}, seed 42:

| File | Status |
|---|---|
| `results/finetune_ptcompute_ep1_seed42_test.json` | ✅ |
| `results/finetune_ptcompute_ep3_seed42_test.json` | ✅ |

**Reused reference points (no new file):** 52K data point = `finetune_A_ep10` (A) /
`finetune_B_seed{0,1}` (B); ep10 ptcompute point = `finetune_A_ep10`. Joined into
`scaling_data.csv` by `scripts/build_scaling_csv.py`.

## Arm D — LoRA (1) ✅

| File | Status |
|---|---|
| `results/lora_qwen_test.json` | ✅ |

## Execution-eval (3 committed + 1 optional) ✅

One per arm over the 201 vendored QuixBugs + HumanEval-Java bugs; pass@1 = `metrics.pass_rate`.

| File | Status | Arm |
|---|---|---|
| `results/execbench_validate_references.json` | ✅ | harness sanity (201/201 on reference patches) |
| `results/execbench_A.json` | ✅ | arm A T5 (required) |
| `results/execbench_C.json` | ✅ | arm C RAG (required) |
| `results/execbench_D.json` | ✅ | arm D LoRA (required) |
| `results/execbench_B.json` | ✅ | arm B T5 (measured: 0.0% compile, 0.0% pass — matches arm A) |

## Derived analysis CSVs (2) ✅

Aggregated from the JSONs above by the builders and **committed**. `scripts/figures/make_all.py`
rebuilds them from the committed JSONs before rendering, so the figures reproduce from a clean
checkout (no fixture fallback).

| File | Built by | Read by | Status |
|---|---|---|---|
| `results/scaling_data.csv` | `scripts/build_scaling_csv.py` | `scripts/figures/scaling_curves.py` | ✅ |
| `results/execbench_agreement.csv` | `scripts/build_execbench_agreement_csv.py` | `scripts/figures/execution_vs_codebleu.py` | ✅ |

## Not study results (present in `results/`, ignore for completeness)

`smoke.json` (CPU smoke), `phase2_summary.md` (T5 finetune batch write-up).

## Tally

| Group | Files | Done |
|---|---|---|
| A/B finetune references | 6 | 6 ✅ |
| Execbench references | 1 | 1 ✅ |
| RAG (arm C) | 8 | 8 ✅ |
| Scaling data curve | 12 | 12 ✅ |
| Pretrain-compute curve | 2 | 2 ✅ |
| LoRA (arm D) | 1 | 1 ✅ |
| Execbench arms (A/B/C/D) | 4 | 4 ✅ |
| Derived CSVs | 2 | 2 ✅ |
| **Study total** | **36** | **36 ✅** |

The study is **complete**: all 33 study result JSONs + `execbench_validate_references.json` + the
2 derived CSVs are committed (36 artifacts, nothing outstanding). Arm B's execution point
(`execbench_B.json`) is now measured — 0.0% compile / 0.0% pass, identical to arm A, confirming the
whole-file-vs-method mismatch is shared by both T5 arms rather than an arm-A artifact.
