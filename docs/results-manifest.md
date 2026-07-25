# Results manifest: the full study's `results/` inventory

Every results file the four-arm study contains, and what each one is for, so "is the study intact?"
is answerable at a glance. All 36 artifacts listed here are committed; nothing is outstanding.

The four arms: **A** pretrain→finetune T5 · **B** from-scratch T5 · **C** RAG Qwen ·
**D** LoRA Qwen. Every JSON follows the `pop.eval.metrics.write_results` schema
(`{config, metrics, n, timestamp, git_sha}`); execbench JSONs carry
`metrics.{pass_rate, compile_rate, per_benchmark}`.

## Arms A & B: T5 finetune references (6)

These A100 results are the study's anchor and are reused as the scaling curves' top (52K / ep10)
points.

| File | Role |
|---|---|
| `results/finetune_A_ep1_test.json`  | arm A, 1 finetune epoch (four-arm epoch trend) |
| `results/finetune_A_ep3_test.json`  | arm A, 3 finetune epochs (four-arm epoch trend) |
| `results/finetune_A_ep10_test.json` | arm A headline; 52K data point + ep10 ptcompute point |
| `results/finetune_B_seed0_test.json`| arm B seed 0; 52K data point (seed 0) |
| `results/finetune_B_seed1_test.json`| arm B seed 1; 52K data point (seed 1) |
| `results/finetune_B_seed2_test.json`| arm B seed 2; four-arm seed band |

## Arm C: RAG sweep (8)

bm25 × codebert, k ∈ {0,1,3,5}. The best-CodeBLEU config (codebert_k1) is arm C in the four-arm and
agreement figures; the k=0 configs are the zero-shot baseline.

| BM25 | CodeBERT |
|---|---|
| `results/rag_bm25_k0_test.json` | `results/rag_codebert_k0_test.json` |
| `results/rag_bm25_k1_test.json` | `results/rag_codebert_k1_test.json` |
| `results/rag_bm25_k3_test.json` | `results/rag_codebert_k3_test.json` |
| `results/rag_bm25_k5_test.json` | `results/rag_codebert_k5_test.json` |

## Scaling curves (14, plus reused references)

**Data-scaling curve (12).** Arms A/B × train_n {1k,5k,15k} × seed {0,1}:

| Arm A | Arm B |
|---|---|
| `results/finetune_scale_A_n1k_seed0_test.json`  | `results/finetune_scale_B_n1k_seed0_test.json`  |
| `results/finetune_scale_A_n1k_seed1_test.json`  | `results/finetune_scale_B_n1k_seed1_test.json`  |
| `results/finetune_scale_A_n5k_seed0_test.json`  | `results/finetune_scale_B_n5k_seed0_test.json`  |
| `results/finetune_scale_A_n5k_seed1_test.json`  | `results/finetune_scale_B_n5k_seed1_test.json`  |
| `results/finetune_scale_A_n15k_seed0_test.json` | `results/finetune_scale_B_n15k_seed0_test.json` |
| `results/finetune_scale_A_n15k_seed1_test.json` | `results/finetune_scale_B_n15k_seed1_test.json` |

**Pretrain-compute curve (2).** Arm A × pretrain-epochs {1,3}, seed 42:
`results/finetune_ptcompute_ep1_seed42_test.json` and
`results/finetune_ptcompute_ep3_seed42_test.json`.

**Reused reference points (no new file).** The 52K data point is `finetune_A_ep10` for arm A and
`finetune_B_seed{0,1}` for arm B; the ep10 ptcompute point is `finetune_A_ep10`. All of it is joined
into `scaling_data.csv` by `scripts/build_scaling_csv.py`.

## Arm D: LoRA (1)

`results/lora_qwen_test.json`.

## Execution eval (4 arms + harness validation)

One per arm over the 201 vendored QuixBugs + HumanEval-Java bugs; pass@1 = `metrics.pass_rate`.

| File | What it is |
|---|---|
| `results/execbench_validate_references.json` | harness sanity: 201/201 on the benchmarks' reference patches |
| `results/execbench_A.json` | arm A, T5 |
| `results/execbench_B.json` | arm B, T5 (0.0% compile, 0.0% pass, matching arm A) |
| `results/execbench_C.json` | arm C, RAG |
| `results/execbench_D.json` | arm D, LoRA |

## Derived analysis CSVs (2)

Aggregated from the JSONs above and committed. `scripts/figures/make_all.py` rebuilds them from the
committed JSONs before rendering, so the figures reproduce from a clean checkout with no fixture
fallback.

| File | Built by | Read by |
|---|---|---|
| `results/scaling_data.csv` | `scripts/build_scaling_csv.py` | `scripts/figures/scaling_curves.py` |
| `results/execbench_agreement.csv` | `scripts/build_execbench_agreement_csv.py` | `scripts/figures/execution_vs_codebleu.py` |

## Not study results

Two files sit in `results/` without being part of the study: `smoke.json` (the CPU smoke reference)
and `phase2_summary.md` (the T5 finetune batch write-up). Neither is counted in the tally below.

## Tally

| Group | Files |
|---|---|
| A/B finetune references | 6 |
| RAG (arm C) | 8 |
| Scaling data curve | 12 |
| Pretrain-compute curve | 2 |
| LoRA (arm D) | 1 |
| Execbench arms (A/B/C/D) | 4 |
| Execbench harness validation | 1 |
| Derived CSVs | 2 |
| **Study total** | **36** |

Arm B's execution point (`execbench_B.json`) measures 0.0% compile / 0.0% pass, identical to arm A.
Both T5 arms failing the same way is what pins the whole-file-vs-method mismatch on the shared
architecture, since an arm-A-only artifact would not reproduce under a different training run.
