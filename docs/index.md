# pretrain-or-prompt

**Does pretraining a small model beat prompting — or cheaply adapting — a larger one for Java
bug fixing?** A four-arm study on CodeXGLUE Java refinement, built to be honest about what the
numbers do and do not show.

This site is the analysis home for the project. Start with the **[study report](report.md)**; the
**[measurement notes](measurement.md)** document the evaluation pitfalls this study's metrics are
built to avoid.

## The four arms

| Arm | System | CodeBLEU | Exec pass@1 | Status |
|-----|--------|----------|-------------|--------|
| **A** | pretrain → finetune T5-small | 0.477 | 0.0% | real (Phase-2 A100 batch) |
| **B** | finetune-from-scratch T5-small | 0.479 | — (not run) | real (Phase-2 A100 batch) |
| **C** | RAG-prompted Qwen2.5-Coder-1.5B | 0.652 | **35.8%** | real (Colab GPU batch) |
| **D** | LoRA-finetuned Qwen2.5-Coder-1.5B | **0.854** | 26.4% | real (Colab GPU batch) |

## Headline figure

![Four-arm comparison](figures/four_arm_comparison.png)

All four arms carry real, committed numbers (arm B's error bar is its seed 0/1/2 band; arm C's bar
is the best config from its retriever×k sweep). CodeBLEU ranks the arms D > C > A ≈ B, but execution
pass@1 — does the fix actually compile and run against 201 real Java bugs — ranks the arms
C > D > A ≈ B: LoRA (D) wins on surface similarity, RAG (C) fixes the most real bugs, and both T5 arms
(A and B) fix zero (identical 0% compile/pass — the shared whole-file-vs-method mismatch). See the **[study report](report.md)** for the full method, results tables,
cross-arm findings, and limitations.

## Reproduce locally

```bash
./.venv/Scripts/python.exe scripts/figures/make_all.py   # render docs/figures/*.png
./.venv/Scripts/python.exe -m mkdocs build               # build this site into ./site
```
