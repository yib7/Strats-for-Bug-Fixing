# Measurement notes: evaluation pitfalls this study guards against

Comparing bug-fixing systems is only as trustworthy as the metrics behind the comparison, and code
metrics are easy to get wrong in ways that look like real findings. This document records two
measurement pitfalls that can silently invert a result on Java bug-fixing, the mechanism behind
each, and the specific guard in this repo's evaluation stack that avoids it — followed by the rigor
choices that back every number in [`report.md`](report.md).

Both pitfalls are pinned by regression tests so they cannot silently reappear.

## Pitfall 1 — strict string equality makes exact match a whitespace test

A natural way to score exact match is strict string equality after a strip:

```python
def exact_match(prediction: str, reference: str) -> bool:
    return prediction.strip() == reference.strip()
```

Run on the decoded output of a code model, this measures detokenization fidelity, not correctness.
CodeXGLUE's Java bug-fix pairs are pre-tokenized with single spaces around every token
(`public java.lang.String METHOD_1 ( )`), and decoding through a from-scratch SentencePiece pipeline
does not perfectly round-trip that spacing: a multi-space run, a tab introduced by generation
formatting, or a stray newline anywhere in a ~200-token method body is enough to make strict `==`
fail — even when every token in the prediction is correct. At that point exact match is a coin flip
on decoder whitespace, and it can read **0.00% across an entire test set** while the predictions are
in fact correct-modulo-whitespace.

### The guard

`src/pop/eval/normalize.py` splits exact match into two metrics:

```python
def normalize_code(s: str) -> str:
    """Collapse every whitespace run to a single space and strip the ends."""
    return _WHITESPACE_RUN.sub(" ", s.strip())

def exact_match(pred: str, ref: str) -> bool:
    """Whitespace-normalized exact match (the metric used for scoring)."""
    return normalize_code(pred) == normalize_code(ref)

def exact_match_raw(pred: str, ref: str) -> bool:
    """Strict exact match after strip only (the naive comparison, kept for contrast)."""
    return pred.strip() == ref.strip()
```

`exact_match` is whitespace-only normalization, **not** retokenization — `foo ( )` and `foo()` still
compare as different strings (`tests/test_eval.py::test_normalize_code_does_not_retokenize`). It
fixes exactly the whitespace-drift failure mode and nothing more: a prediction that is wrong on the
actual tokens still scores 0. `exact_match_raw` is reported alongside it so the strict and
normalized numbers are always visible together.

`tests/test_eval.py::test_strict_equality_whitespace_pitfall` pins the mechanism directly:

```python
pred = "public int add(int a, int b) {\treturn a + b;\t}"
ref  = "public int add(int a, int b) { return a + b; }"

assert exact_match_raw(pred, ref) is False   # naive: fails
assert exact_match(pred, ref) is True        # normalized: passes
```

A caveat this study is careful about (see [`report.md`](report.md) finding 2): the normalized metric
fixing the *artifact* does not mean exact match will be high. The T5 arms genuinely score EM≈0 under
the fixed metric because they always make a valid-but-different edit — a real property of the model,
confirmed by auditing predictions four ways, not a leftover measurement bug. The fixed metric
registers real matches for the stronger Qwen arms (LoRA: 9.5% EM), which is how we know it *can*
register a match when one occurs.

## Pitfall 2 — a naive few-shot prompt can make RAG score worse than zero-shot

Retrieval-augmented prompting should help a capable instruction-tuned model, so a RAG configuration
scoring *below* its own zero-shot baseline is a red flag that usually points at the prompt, not the
retrieval. Two prompt-construction mistakes compound:

1. **Truncating exemplars.** Building the few-shot prompt by concatenating retrieved examples cut to
   a fixed length with a literal `"..."` appended —

   ```python
   prompt += f"Buggy:\n{example['buggy'][:200]}...\n"
   prompt += f"Fixed:\n{example['fixed'][:200]}...\n"
   ```

   — slices mid-statement, sometimes mid-identifier, on any method longer than the cap (CodeXGLUE
   Java methods routinely run 400–500+ chars). The model is handed syntactically broken exemplars
   with a dangling `...` that looks like elided code but is not valid Java. Stack three of those
   before the actual bug and the in-context "examples" are actively misleading.

2. **No chat template.** An instruction-tuned generator (`Qwen2.5-Coder-1.5B-Instruct`) expects
   `<|im_start|>…<|im_end|>` chat-formatted turns. Feeding it raw concatenated text is out of
   distribution for how it was tuned; combined with three malformed exemplars, the model has every
   incentive to continue the broken pattern rather than produce a clean fix. A zero-shot prompt
   avoids the truncation damage entirely, so it can easily beat a broken few-shot prompt — which
   reads as "retrieval hurts" when the real cause is prompt construction.

### The guard

`src/pop/rag/prompt.py`:

- `build_messages(buggy, exemplars)` labels each exemplar `Buggy:`/`Fixed:` in full, with **no
  length cap**, and returns a `[{"role": "system", …}, {"role": "user", …}]` message list rather
  than a flat string.
- `render_prompt` renders that list through `tokenizer.apply_chat_template(messages, tokenize=False,
  add_generation_prompt=True)`, so the model always sees a well-formed chat turn.
- `extract_fix` strips fenced code blocks and chatty preambles from the response before scoring.

`tests/test_rag.py` pins both fixes: a long exemplar is preserved verbatim with no `"..."` inserted,
and `apply_chat_template` is exercised against a real chat-template fixture. With the corrected
pipeline, retrieval clearly helps — CodeBLEU rises from ≈0.37 (zero-shot) to ≈0.65 at k=1
([`report.md`](report.md) finding 4).

## The rigor choices behind every number

Beyond the two guards above, the evaluation stack is built so results are trustworthy and traceable:

- **Two exact-match metrics** — `exact_match` (normalized, primary) and `exact_match_raw` (strict),
  always reported together (`src/pop/eval/normalize.py` + `metrics.py`).
- **Syntax validity** — a tree-sitter Java parse (no ERROR nodes) on every prediction, so "does it
  even parse" is measured, not assumed.
- **Confidence intervals, not point estimates** — percentile bootstrap CIs over per-sample scores
  (`src/pop/eval/bootstrap.py`).
- **Provenance on every run** — `write_results` persists each run to `results/<name>.json` as
  `{config, metrics, n, timestamp, git_sha}`, so a number traces to the config and commit that
  produced it.
- **Two retriever backends** — BM25 (`bm25s`) and dense CodeBERT + FAISS behind one
  `index()`/`retrieve()` interface, with the knowledge base built **strictly from the train split**
  (leakage guard).
- **Execution ground truth** — a JDK harness that compiles and runs each predicted fix against the
  201 vendored QuixBugs-Java + HumanEval-Java bugs, validated **201/201 on the reference patches**
  (`results/execbench_validate_references.json`) before any model was scored. Surface metrics can be
  gamed by valid-but-different edits; execution cannot (see [`report.md`](report.md) finding 5, the
  CodeBLEU↔execution inversion).
- **Regression coverage** — both pitfalls above are pinned by tests (`tests/test_eval.py`,
  `tests/test_rag.py`) so a refactor cannot silently reintroduce them.
