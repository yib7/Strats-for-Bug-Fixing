# pretrain-or-prompt: study report

> Four arms: **A** pretrain→finetune T5-small, **B** from-scratch T5-small, **C** RAG-prompted
> Qwen2.5-Coder-1.5B-Instruct, **D** LoRA-finetuned Qwen2.5-Coder-1.5B-Instruct. Arms C and D sit on
> the same base model, so C against D compares prompting with adapting. Every number below was
> measured on GPU and is committed under `results/*.json`.
>
> Two lenses. **Track 1** is CodeBLEU / exact-match / syntax-validity on the full 6,545-pair
> CodeXGLUE test split. **Track 2** is execution pass@1 on 201 real Java bugs (QuixBugs-Java +
> HumanEval-Java) through a JDK harness validated 201/201 on reference patches. The two lenses
> **disagree**, and that disagreement is the headline finding (see Cross-arm findings §5).

## The question

Does **pretraining a small model** (a from-scratch T5 given span-corruption pretraining, then
finetuned) beat the alternatives for Java bug-fixing? The alternatives are finetuning the **same
small model from a random init**, **prompting a larger code LLM** with retrieval (RAG), and
**cheaply adapting** that larger LLM with LoRA. Four arms, one held-out benchmark, two lenses:
surface similarity via CodeBLEU, and does-it-actually-run via execution pass@1.

The question is asked carefully because bug-fix evaluation is easy to get wrong. A strict-equality
exact-match metric can report **0.00% on every configuration**, and a carelessly built few-shot
prompt can make RAG score *worse* than zero-shot. Both are measurement artifacts, not real findings
(the pitfalls are documented in [`docs/measurement.md`](measurement.md)). This study builds the
measurement stack first (whitespace-normalized EM + raw EM, CodeBLEU, tree-sitter syntax-validity,
bootstrap CIs, provenance on every `results/*.json`) and only then compares arms.

## Method: the four arms

| Arm | System | Adaptation | Runs behind it |
|-----|--------|------------|----------------|
| **A** | T5-small, built here from a manual config | span-corruption **pretrain → finetune** | finetune epochs 1/3/10, seed 42 |
| **B** | same T5-small | **finetune from random init** (no pretrain) | seeds 0/1/2 |
| **C** | Qwen2.5-Coder-1.5B-Instruct | **RAG** prompt (BM25 / CodeBERT × k∈{0,1,3,5}) | 8-config sweep; best is codebert_k1 |
| **D** | Qwen2.5-Coder-1.5B-Instruct | **LoRA** finetune (r16/α32/drop0.05, q/k/v/o_proj) | one run |

The T5 the study builds is t5-small sized: `d_model` 512, `d_ff` 2048, `d_kv` 64, 8 attention heads,
6 encoder and 6 decoder layers, over a 16,384-token SentencePiece vocabulary trained in this repo,
at sequence length 512. No pretrained checkpoint is downloaded for arms A or B. Span corruption is
T5's usual pretraining objective: mask out random runs of tokens and have the model reconstruct
them, so it learns the shape of Java before it ever sees a bug-fix pair.

- **Data.** CodeXGLUE `code_x_glue_cc_code_refinement` (medium, Java). Test split = **6,545** buggy→fixed
  method pairs. RAG knowledge base is built **strictly from the train split** (leakage guard enforced
  in `pop rag`).
- **Generation.** Greedy (`num_beams=1`), `max_new_tokens=256`, full test split, the same decoding for
  every arm so the comparison is fair.
- **Metrics.** `em` (whitespace-normalized exact match), `em_raw` (strict string equality, the naive
  version kept alongside it for contrast; see [`measurement.md`](measurement.md) §1),
  `codebleu`, `syntax_valid_rate` (tree-sitter Java parse, no ERROR nodes). CIs via percentile
  bootstrap (`pop.eval.bootstrap`) where per-sample data is available; otherwise point estimates with
  the arm-B seed band.
- **Execution lens (Track 2).** Predicted fixes run through the JDK harness over the 201 vendored
  QuixBugs-Java (40) + HumanEval-Java (161) bugs; the harness is validated **201/201 on reference
  patches** (`results/execbench_validate_references.json`). Measured pass@1 / compile-rate per arm:
  **A** 0.0% compile, 0.0% pass (100% `compile_error`, both benches); **B** 0.0% compile, 0.0% pass
  (100% `compile_error`, both benches); **C (RAG)** 70.6% compile, **35.8%** pass (QuixBugs 25.0%,
  HumanEval-Java 38.5%); **D (LoRA)** 69.2% compile, **26.4%** pass (QuixBugs 32.5%, HumanEval-Java
  24.8%). Both T5 arms (A and B, trained on abstracted single methods) fix zero bugs; the two LLM arms
  carry the entire execution signal.

## Results

### Four-arm headline

[![Four-arm comparison: grouped bars of CodeBLEU and syntax-valid rate for arms A, B, C and D, with
arm B's seed band and arm A's finetune-epoch trend shown as error bars and connected
markers.](figures/four_arm_comparison.png)](figures/four_arm_comparison.png)

*Figure 1, `docs/figures/four_arm_comparison.png`. Arm B's error bar is the seed 0/1/2 band; the
connected markers on arm A trace the finetune-epoch trend (1→3→10); arm C's bar is the best config
from the retriever×k sweep (codebert_k1); arm D is the single LoRA run.*

**Arm A, pretrain → finetune T5.** CodeBLEU and syntax-validity both rise monotonically with
finetune epochs; exact match is ~0 throughout (audited, see findings).

| config | em | em (count) | codebleu | syntax_valid_rate | n |
|--------|----|-----------|----------|-------------------|---|
| A_ep1  | 0.0000 | 0/6545 | 0.3553 | 0.4568 | 6545 |
| A_ep3  | 0.0000 | 0/6545 | 0.4163 | 0.7911 | 6545 |
| A_ep10 | 0.0005 | 3/6545 | 0.4770 | 0.9169 | 6545 |

**Arm B, from-scratch T5.** Ten epochs from random init, seed swept {0,1,2}. The seed spread is
tiny, which is what makes the A-vs-B comparison trustworthy rather than a single-seed fluke.

| config | em | em (count) | codebleu | syntax_valid_rate | n |
|--------|----|-----------|----------|-------------------|---|
| B_seed0 | 0.0003 | 2/6545 | 0.4781 | 0.8833 | 6545 |
| B_seed1 | 0.0006 | 4/6545 | 0.4762 | 0.8782 | 6545 |
| B_seed2 | 0.0005 | 3/6545 | 0.4835 | 0.8949 | 6545 |
| **B (mean ± band)** | n/a | n/a | **0.479** (0.476–0.484) | **0.885** (0.878–0.895) | 6545 |

**Arm C, RAG Qwen.** Retriever × k sweep, 8 configs (`results/rag_*_test.json`):

| retriever | k | codebleu | syntax_valid_rate | em |
|-----------|---|----------|--------------------|-----|
| bm25 | 0 | 0.3674 | 0.9325 | 0.0000 |
| bm25 | 1 | 0.6359 | 0.9641 | 0.0159 |
| bm25 | 3 | 0.6262 | 0.9652 | 0.0139 |
| bm25 | 5 | 0.6331 | 0.9668 | 0.0162 |
| codebert | 0 | 0.3671 | 0.9325 | 0.0000 |
| **codebert** | **1** | **0.6517** | **0.9596** | **0.0189** |
| codebert | 3 | 0.6295 | 0.9638 | 0.0124 |
| codebert | 5 | 0.6343 | 0.9656 | 0.0153 |

Best config is **codebert_k1** (CodeBLEU 0.6517, syntax-valid 0.960, EM 0.0189), which is the arm-C bar
in Figure 1. Retrieval clearly helps: zero-shot (k0) sits at ≈0.367 CodeBLEU for both retrievers, and
adding a single retrieved exemplar (k1) jumps CodeBLEU to ≈0.65, a ~0.28-point gain, while
syntax-validity climbs from 0.932 to 0.96+. Past k1, more exemplars do **not** help: k3 and k5 both sit
slightly below k1 for both retrievers (bm25 0.626/0.633 vs 0.636; codebert 0.630/0.634 vs 0.652). One
well-chosen exemplar beats a longer context window here. BM25 and CodeBERT are near-identical at every
k, with CodeBERT marginally ahead at the k1 optimum (0.6517 vs 0.6359). Retrieval helping rather than
hurting is itself a finding, see Cross-arm finding 4.

**Arm D, LoRA Qwen.** A single LoRA finetune (r16/α32/drop0.05 on q/k/v/o_proj) of the same
1.5B Qwen base scores **CodeBLEU 0.8543, syntax-valid 0.9366, EM 0.0947**
(`results/lora_qwen_test.json`), the best arm on both CodeBLEU and EM by a wide margin: +0.20 CodeBLEU
over arm C's best RAG config and +0.38 over the T5 arms, with an EM rate ~5x arm C's best (9.47% vs
1.89%) and far above the T5 arms' ~0%. Cheap parameter-efficient adaptation of a capable base model
dominates every other arm on the surface-similarity lens, though this ranking does **not** survive the
execution lens (Cross-arm findings 5 and 7).

### Scaling curves

[![Scaling curves: two line charts. Left, CodeBLEU against finetune train_n at 1K, 5K, 15K and ~52K
pairs for arms A and B, which track each other closely. Right, CodeBLEU against pretraining epochs
(1, 3, 10) for arm A, essentially flat.](figures/scaling_curves.png)](figures/scaling_curves.png)

*Figure 2, `docs/figures/scaling_curves.png`. Built from `results/scaling_data.csv` (18 rows: 12
data-curve runs at train_n∈{1K,5K,15K}×seed{0,1} for arms A/B, plus the reused 52K/ep10 reference
points, plus 2 pretrain-compute runs at pretrain-epoch∈{1,3}, plus the reused ep10 point).* Two curves:
(left) CodeBLEU vs finetune `train_n` for arms A and B at 1K/5K/15K/~52K pairs; (right) CodeBLEU vs
pretraining epochs for arm A (finetuning from the epoch-1/3/10 pretrain checkpoints).

**Data curve: the arms converge.** Both arms rise monotonically with `train_n`, and the result worth
reading is in the **gap** between them. Arm A (pretrained) leads arm B (scratch) at every budget,
but the lead shrinks as data grows: the A−B CodeBLEU gap runs ≈0.083 at 1K, ≈0.067 at 5K, ≈0.033 at
15K, and **≈0** at the full 52K split (A 0.477 vs B mean 0.479, inside the seed band). Pretraining's
head start is real, but it is a **small-data-budget effect**: it helps when finetune data is scarce
and evaporates once there is enough finetune data to learn the task directly. That *refines* finding
1 below ("no benefit at full data") by locating where the benefit lives: pretraining buys speed of
convergence while leaving the ceiling where it was. Syntax-validity shows the same convergence more
sharply. It is **exactly 0% at 1K for both arms**, a hard floor where every prediction fails to
parse, and climbs to 88–92% by 52K.

**Pretrain-compute curve: flat.** Finetuning arm A from pretrain checkpoints saved at
pretrain-epoch 1/3/10 gives CodeBLEU **0.4688 / 0.4635 / 0.4770** (syntax-valid 0.897 / 0.904 /
0.917), essentially flat across a 10x change in pretraining compute. The epoch-3 point does not even
sit between the other two: it is the lowest of the three, so the small spread reads as noise rather
than a trend. More pretraining compute past epoch 1 buys **approximately nothing** on the downstream
finetune metric.

### Execution vs CodeBLEU

[![Execution pass@1 against CodeBLEU, one point per arm. RAG-prompted Qwen (C) is highest on pass@1
at 35.8% despite a lower CodeBLEU than LoRA (D) at 26.4%, and the two T5 arms sit on top of each
other at pass@1 = 0.](figures/execution_vs_codebleu.png)](figures/execution_vs_codebleu.png)

*Figure 3, `docs/figures/execution_vs_codebleu.png`. Per-arm execution predictions
(`results/execbench_{A,B,C,D}.json`) plotted against each arm's Track-1 CodeBLEU.* Each arm is one
point (x = CodeBLEU, y = execution pass@1). Arms A and B sit atop each other at pass@1 = 0.

**The headline: the two lenses disagree.** CodeBLEU ranks the arms **D (0.854) > C (0.652) > A ≈ B
(≈0.48)**. Execution pass@1 ranks the four arms **C (35.8%) > D (26.4%) > A ≈ B (0.0%)**,
inverted at the top. Arm D "looks most right" by surface similarity; arm C "is most right" by
does-it-actually-run. And arms A and B, CodeBLEU-competitive with C (0.48 vs 0.65, not wildly below),
fix **zero** of 201 real bugs; CodeBLEU parity does not mean functional parity. (Arm B, run through the
same harness, lands at the identical 0.0% compile / 0.0% pass as arm A: the shared
whole-file-vs-method mismatch, now measured rather than assumed.) The takeaway is not
"CodeBLEU is useless". It still separates all four arms from a random baseline and tracks
syntax-validity sensibly within each arm. The takeaway is that CodeBLEU is a **surface proxy** that
does not preserve the functional-correctness ranking across architecturally different systems, and
should not stand in for execution when both are available.

## Cross-arm findings

What the committed four-arm data supports. Findings 1–3 are the A-vs-B story; 4–7 bring in C and D.
Several of these are negative results, and they are reported as such.

1. **Pretraining buys no measurable CodeBLEU benefit over from-scratch, at this scale.** A_ep10's
   CodeBLEU (0.477) sits *inside* arm B's seed-to-seed band (0.476–0.484). This is not a single-seed
   coincidence: B's CodeBLEU spread across seeds 0/1/2 is ≈ ±0.004, so "pretraining ≈ no benefit"
   survives the seed-variance ablation. Pretraining's *only* edge here is a
   modest **~+3 pt syntax-validity** bump (A 0.917 vs B mean 0.885), real but small.

2. **Exact match is genuinely ~zero (0–4 of 6,545), and that is a real property, not a bug.** The
   A_ep10 greedy predictions were audited four ways (normalized EM, raw `==`, whitespace-collapsed,
   whitespace-removed) and **all return exactly 3**. Predictions are complete, well-formed methods
   (99.8% end on `}`, 0 empty, none near the token cap) that make a *valid-but-different* edit. EM even
   sits below a copy-the-input baseline (~3.4% on CodeXGLUE-medium) because the model always edits and
   so forfeits the no-op cases. Crucially, this is not the strict-`==` measurement artifact (a naive
   metric can report 0% EM on correct-modulo-whitespace predictions); the metric here is the fixed
   one, and EM is low for a real reason, which is why **CodeBLEU + syntax-validity are the
   trustworthy headline metrics** for the A-vs-B story.

3. **The learning signal is unambiguous even with EM≈0.** Within arm A, more finetune epochs improve
   both CodeBLEU (0.355 → 0.416 → 0.477) and syntax-validity (0.457 → 0.791 → 0.917) monotonically.
   The models clearly learned; EM is just the wrong yardstick for this abstracted refinement task.

4. **RAG beats zero-shot, and the prompt construction is what makes or breaks it.** With a
   correctly built pipeline (`build_messages` + `apply_chat_template`, no exemplar truncation),
   retrieval **clearly helps**: zero-shot (k0) CodeBLEU ≈0.367 rises to ≈0.65 at k1 (best config
   codebert_k1: 0.6517), and syntax-validity rises from 0.932 to 0.960. This is not automatic. A
   naive few-shot prompt that truncates exemplars at 200 chars and skips the model's chat template
   makes RAG score *below* zero-shot (a pitfall documented in [`measurement.md`](measurement.md) §2).
   Retrieval helps when the retrieved context reaches the model intact and in-format.

5. **CodeBLEU and execution disagree; the surface-similarity lens is not a proxy for correctness
   across architectures.** Figure 3 shows the ranking inversion directly: CodeBLEU ranks D > C > A≈B;
   execution pass@1 ranks C > D > A/B. Arm D wins the surface-similarity lens by a wide margin (0.854
   vs C's 0.652), yet arm C fixes more real bugs (35.8% vs 26.4% pass@1). CodeBLEU rewards token/AST
   overlap with the reference; it does not require the code to compile, let alone pass tests.
   Execution is the ground truth for "does it run," and here it says something CodeBLEU alone would
   get backwards.

6. **The same metric that scores T5 at zero registers 9.47% for LoRA.** A strict-`==` exact-match
   metric can report 0% EM everywhere purely as a whitespace artifact (the pitfall in
   [`measurement.md`](measurement.md) §1); this study uses the fixed, whitespace-normalized metric.
   That fixed metric still returns EM≈0 for T5 (finding 2: 0–4 of 6,545), while returning
   substantial matches for the Qwen arms: RAG's best config gets 1.89% EM, and LoRA gets **9.47%**
   (620 of 6,545 predictions whitespace-normalized identical to the reference). So the metric
   registers a match whenever a model makes one. T5 simply does not make them.

7. **Execution is the discriminating lens: only the LLM arms fix real bugs, and the T5 arms' 0% is a
   domain mismatch, not a null result.** C and D fix 26–36% of the 201 real bugs; the
   CodeBLEU-competitive T5 arms A and B both fix **zero**, at an identical 0.0% compile / 0.0% pass, so
   the failure is the shared architecture, not one seed. The T5 arms' 0.0% pass (100% `compile_error` on
   both QuixBugs and HumanEval-Java) is not evidence the harness is broken, since the same harness
   passes reference patches 201/201 (`results/execbench_validate_references.json`). It is a
   **whole-file-vs-method domain mismatch**: T5 was trained and evaluated on CodeXGLUE's *abstracted
   single-method* snippets, but the execution benchmark feeds *whole concrete Java files* (real
   class/import/field context, real identifiers). T5 never learned to handle that input shape, so its
   output never compiles. The instruction-tuned Qwen arms, by contrast, generalize (zero-shot in C, or
   with light adaptation in D) to whole-file input despite never being finetuned on it. This is itself
   a finding about T5's brittleness to input-distribution shift versus the LLMs' broader
   generalization, not proof that arm A "learned nothing" (see Limitations).

## Limitations

- **One decoding setting.** Greedy only. A beam-5 pass on A_ep10 would let the writeup *show* rather
  than argue that beam search does not recover EM (audit predicts it will not); it is an optional
  follow-up, not run here.
- **CodeBLEU is a surface proxy.** It rewards token/AST overlap, not correctness, which is exactly why
  the execution lens (Figure 3) exists and why finding 5 matters: a valid-but-different edit, or an
  edit that does not even compile, can still score well on CodeBLEU and syntax-validity.
- **The T5 arms' 0% execution pass should not be read as "T5 learned nothing."** Track 1 shows T5 clearly
  learning the CodeXGLUE refinement task (finding 3: CodeBLEU and syntax-validity rise monotonically
  with finetune epochs, arm A converging with arm B at scale). Track 2's 0% pass, identical for both T5
  arms, is specifically a whole-file-vs-method **input-distribution mismatch** (finding 7): the model
  was never shown whole files during training, so it should not be judged on whole files as if it had
  been. A fair execution comparison for the T5 arms would need method-level extraction and splicing back
  into the surrounding file, out of scope here.
- **No 1.5B-tier Qwen3-Coder comparison.** Arms C and D both sit on Qwen2.5-Coder-1.5B. Qwen3-Coder
  exists, but its smallest published size is 30B-A3B, so the 1.5B tier has no Qwen3-Coder equivalent
  to swap in. Comparing against it would change the parameter budget as well as the model generation,
  which is a different experiment.
- **Provenance caveat.** Every `results/*.json` carries `git_sha: unknown`: the Colab runs executed
  from an uploaded code zip with no `.git`, so the exact commit that produced each batch is not
  embedded in the artifacts. The committed `results/*.json` are therefore the source of truth, and
  every number in this report was cross-checked directly against its source file (the A/B finetune
  batch is additionally written up in `results/phase2_summary.md`). Reproducing the runs from
  scratch, rather than trusting the committed JSONs, requires re-executing the notebooks on a GPU
  (`docs/gpu-reproduction.md`).

## Reproducing the figures

```bash
uv sync --frozen                            # build .venv from the committed lockfile
uv run python scripts/figures/make_all.py   # writes docs/figures/*.png
uv run python -m mkdocs build               # builds the static site
```

The figure scripts are deterministic (headless Agg backend, point estimates, committed data), so
re-running them on the same machine reproduces the committed PNGs byte for byte. Across operating
systems the bytes differ while the plot does not: matplotlib rasterises text through whichever
freetype its wheel was built against, so a Linux or macOS re-render of the same numbers leaves the
three figures showing as modified in `git status`. `git checkout docs/figures` restores the
committed copies.
