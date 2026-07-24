# pretrain-or-prompt

**Does pretraining a small T5 on code beat prompting — or cheaply adapting — a much larger code
LLM, for Java bug-fixing?** A four-arm study: pretrain→finetune T5-small, from-scratch T5-small,
RAG-prompted Qwen2.5-Coder-1.5B, and LoRA-finetuned Qwen2.5-Coder-1.5B, compared on both surface
similarity (CodeBLEU) and whether the fix actually compiles and runs.

## TL;DR — the findings

- **Execution inverts the CodeBLEU ranking.** LoRA (arm D) wins CodeBLEU (0.854), but RAG (arm C)
  fixes the most real bugs (**35.8%** vs **26.4%** pass@1) — while both T5 arms (A and B), despite
  competitive CodeBLEU, fix **zero** of 201 real bugs (identical 0% compile/pass — the shared
  whole-file-vs-method mismatch). Surface similarity and functional correctness disagree at the top of
  the ranking.
- **Pretraining buys no CodeBLEU benefit at full data.** Arm A (pretrained) and arm B
  (from-scratch) converge to ≈0.48 CodeBLEU by the full 52K-pair split. Pretraining's head start
  is real at small data budgets (a ~0.08-point gap at 1K pairs) but shrinks steadily and is gone
  by 52K.
- **RAG clearly beats zero-shot — retrieval, done right, helps.** Adding a single retrieved
  exemplar lifts CodeBLEU from 0.37 to 0.65, and the Qwen arms produce real, non-trivial exact
  matches (LoRA: 9.5% EM) where the T5 arms sit at ~0. That gain depends on the prompt being built
  correctly — full exemplars, chat-templated; a naive few-shot prompt (truncated exemplars, no chat
  template) actively *hurts* vs. zero-shot.
- **The measurement stack is built for rigor.** Whitespace-normalized + strict exact match,
  CodeBLEU, tree-sitter syntax-validity, bootstrap CIs, and a JDK execution harness validated
  **201/201** on reference patches — see [the measurement notes](docs/measurement.md) for the
  evaluation pitfalls this stack is designed to avoid.

## Results at a glance

| Arm | Adaptation | CodeBLEU | Syntax-valid | EM | Exec pass@1 (201 bugs) |
|-----|------------|----------|--------------|-----|-------------------------|
| A | pretrain→finetune T5-small | 0.477 | 92% | ~0 | 0.0% |
| B | from-scratch T5-small | 0.479 | 88% | ~0 | — (not run) |
| C | RAG Qwen (CodeBERT retriever, k=1) | 0.652 | 96% | 1.9% | **35.8%** |
| D | LoRA Qwen | **0.854** | 94% | **9.5%** | 26.4% |

*Arm D tops the CodeBLEU column; arm C tops the execution column — that inversion is the
headline finding (see [the report](docs/report.md) §"Execution vs CodeBLEU").*

## Headline figures

![Four-arm comparison](docs/figures/four_arm_comparison.png)

![CodeBLEU vs execution](docs/figures/execution_vs_codebleu.png)

## Read the full study

- **[docs/report.md](docs/report.md)** — method, all four arms' results tables, scaling curves,
  seven cross-arm findings, and limitations.
- **[docs/measurement.md](docs/measurement.md)** — the evaluation pitfalls that make code
  bug-fixing metrics easy to get wrong (strict-equality EM reporting 0%, few-shot prompts that hurt
  instead of help), and the guards in the eval stack that avoid each. Read this for the rigor case
  behind the numbers above.

## Repo map

- **`src/pop/`** — the package: data loading, tokenizer, T5 model, training (`train/`: pretrain,
  finetune, LoRA), evaluation (`eval/`: EM/CodeBLEU/syntax-validity/bootstrap CIs), `rag/`
  (retrieval-augmented prompting), `execbench/` (JDK execution harness).
- **`scripts/`** — orchestrators (`run_training.py`, `run_rag.py`, `run_scaling.py`), figure
  scripts (`figures/make_all.py`), and CSV aggregators (`build_scaling_csv.py`,
  `build_execbench_agreement_csv.py`).
- **`configs/`** — YAML configs for every run (pretrain, finetune sweeps, RAG retriever×k sweep,
  LoRA, scaling).
- **`notebooks/`** — Colab "Run-all" kits used to produce every GPU result in this study.
- **`benchmarks/`** — vendored QuixBugs-Java + HumanEval-Java + the JDK execution harness.
- **`results/`** — the committed `*.json` metrics every number in the report traces back to.
- **`docs/`** — the study report, measurement notes, and reproduction runbooks.
- **`tests/`** — the test suite; run it with `pytest` (or `pytest -m "not jdk"` to skip the tests
  that need a local JDK, which is what CI does).

## Reproduce

Install (this repo uses a `.venv` + editable install):

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

Then, CPU-only, in order:

```bash
./.venv/Scripts/pop smoke                                # end-to-end sanity check, no GPU needed
./.venv/Scripts/python.exe scripts/figures/make_all.py   # render docs/figures/*.png from results/
./.venv/Scripts/python.exe -m mkdocs build                # build the docs site into ./site
```

None of the above retrains anything — every number in the report was produced on a Colab GPU and
is committed under `results/`. For the full GPU reproduction (pretrain → finetune → RAG → LoRA →
execution eval), see [docs/gpu-runbook.md](docs/gpu-runbook.md).
