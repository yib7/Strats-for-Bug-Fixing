# Phase 2 results — pretrain-vs-scratch T5 on CodeXGLUE refinement (medium, Java)

**Run:** Google Colab A100, 2026-07-18 02:35→06:14 UTC (~3 h 40 m end-to-end), all six
systems completed, no gate halt. Produced by `scripts/run_training.py` from the code in
`dist/pop_repo.zip` built at commit `12d555c` (the results JSONs record `git_sha: unknown`
because Colab ran from the zip, which carries no `.git`; provenance is this commit).

Test split: 6,545 CodeXGLUE `code_x_glue_cc_code_refinement` (medium) pairs. Generation:
greedy (`num_beams=1`), `max_new_tokens=256`.

## Systems

- **Pipeline A** — pretrained (10-epoch span-corruption) **then** finetuned; finetune epochs
  swept {1, 3, 10} at fixed seed 42.
- **Pipeline B** — finetuned from a random init (no pretraining); 10 epochs, seed swept {0,1,2}.

| system | em | em (count) | codebleu | syntax_valid_rate | n |
|---|---|---|---|---|---|
| A_ep1  | 0.0000 | 0/6545 | 0.3553 | 0.4568 | 6545 |
| A_ep3  | 0.0000 | 0/6545 | 0.4163 | 0.7911 | 6545 |
| A_ep10 | 0.0005 | 3/6545 | 0.4770 | 0.9169 | 6545 |
| B_seed0 | 0.0003 | 2/6545 | 0.4781 | 0.8833 | 6545 |
| B_seed1 | 0.0006 | 4/6545 | 0.4762 | 0.8782 | 6545 |
| B_seed2 | 0.0005 | 3/6545 | 0.4835 | 0.8949 | 6545 |

## Reading (honest)

1. **Exact match is genuinely ~zero (0–4 of 6,545) — and this is real, not a bug.** The
   A_ep10 greedy predictions were audited directly (2026-07-18, `results/phase2_summary.md`
   diagnosis below). The near-zero EM is a true property of a small T5 on this abstracted
   refinement task, *not* a decoding/metric artifact (see `docs/measurement.md`).

2. **CodeBLEU and syntax validity are healthy and behave sensibly**, which is the evidence the
   models *did* learn: within Pipeline A, more finetune epochs improve both monotonically
   (CodeBLEU 0.355→0.416→0.477; syntax 0.457→0.791→0.917).

3. **A vs B — does pretraining help? A small gap, pinned down with the variance ablation.**
   A_ep10 (CodeBLEU 0.477, syntax 0.917) sits *inside* Pipeline B's seed-to-seed band (CodeBLEU
   0.476–0.484 across seeds 0/1/2). On CodeBLEU, pretraining gives no measurable benefit over
   from-scratch; its only edge is a modest ~+3 pt syntax-validity bump. The key ablation — is B's
   number stable across seeds? — is answered: yes (CodeBLEU spread ≈ ±0.004), so the ~zero benefit
   is not a single-seed fluke.

## EM diagnosis (2026-07-18) — audit of A_ep10's 6,545 greedy predictions

Ruling out decoding/metric-artifact hypotheses before reporting EM:

- **Metric is not masking matches.** EM computed four ways — the repo's normalized metric, raw
  string `==`, whitespace-collapsed, and all-whitespace-removed — **all return exactly 3**. No
  detokenization/whitespace gap hides correct answers (the strict-`==` pitfall in
  `docs/measurement.md`; guarded against here).
- **No degeneracy or truncation.** 99.8% of predictions end on `}`, 99.7% have balanced braces,
  0 are empty, none approach the 256-token cap (max 567 chars). Median pred/ref length ratio
  0.90. Predictions are complete, well-formed methods that differ from the reference in *content*
  (a valid-but-different edit), not the `.METHOD_k()` repetition that signals decoding degeneration.
- **Therefore beam-5 is not expected to recover EM** — beam search fixes degenerate greedy
  decoding, and greedy is already clean here.
- **EM sits below even a copy-the-input baseline** (~3.4% on CodeXGLUE-medium). Mechanism visible
  in the data: the model *always* edits, so it forfeits the no-op cases a copy baseline catches,
  and at this scale rarely lands the exact transform.

**Verdict:** CodeBLEU + syntax-validity are the trustworthy headline metrics for the A-vs-B
story; EM is honestly reported as ~0 with the above explanation. No decoding fix is warranted.
An optional beam-5 pass on A_ep10 would let the writeup *show* (rather than argue) that beam
search doesn't move EM.
