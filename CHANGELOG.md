# Changelog

Versions follow [semantic versioning](https://semver.org/). Each entry uses the same shape:
`## vX.Y.Z - YYYY-MM-DD`, a one-line summary, then grouped bullets.

## v1.0.1 - 2026-07-28

A fix-forward release on top of `v1.0.0`: no new study result, no new arm, no changed number in the
report. This closes gaps a post-publication audit found in the code, the docs, and the repository's
own packaging — including the version number itself, which never got bumped for the first release.

### Fixed

- **License detection.** GitHub read the repo's license as "Other" because `LICENSE` carried a scope
  paragraph appended after the MIT text, which breaks automated license matching. `LICENSE` is now
  the verbatim 21-line MIT text; the scope paragraph moved to `CREDITS.md`, which already said the
  same thing.
- **Test coverage gap.** `requires-python = ">=3.11,<3.13"` claimed 3.11 support that had never
  actually been exercised. CI now runs the full suite on both 3.11 and 3.12 (`fail-fast: false`, so
  a 3.11-only break can't be masked by 3.12 passing).
- **RAG/LoRA resume correctness.** `generate_with_resume` could resume onto a `.partial` file left by
  a *different* run (different k, retriever, split, or template) with no identity check. It now
  stamps a SHA-256 sidecar over every prompt and reference and discards a mismatched or unprovable
  partial instead of silently continuing it.
- **Retriever empty-knowledge-base contract.** `CodeBERTRetriever` and `BM25Retriever` disagreed on
  what happens against an empty corpus — one returned `[]`, the pinned `bm25s` actually raised
  `ValueError`. Both now behave the same way.
- **Benchmark-manifest cache leak across tests.** A hand-rolled process-global cache could carry a
  poisoned manifest across a `BENCHMARKS_DIR` restore. Replaced with `functools.cache` plus an
  explicit, test-cleared reset; the fix is proven with a deliberate poison/check test pair.
- **Harness output race.** The execution harness could read its captured output buffer while a
  timed-out drain thread was still appending to it. Now guarded with a lock.
- **Network hermeticity is now enforced, not just asserted.** Every test now runs under a fixture
  that raises if anything tries to reach a non-loopback host. The full 448-test suite passes with it
  on — the "this suite makes no network calls" claim is a standing CI guarantee, not a one-off audit
  note.
- **RAG prompt truncation.** The RAG generation path stripped the prompt from model output by string
  slicing, which breaks on tokenizer edge cases; it now uses the pipeline's own
  `return_full_text=False`, matching the LoRA arm's approach.
- **Empty-run false success.** Running the execution harness with a predictions file that selects no
  bugs used to exit 0. It now exits 2 with a clear message.
- **`docs/figures/pipeline.mmd` failed to parse** from its own source (a bare `%%` fused two lines)
  even though nothing in the build regenerates the SVG from it today — fixed before it could bite the
  first person who tries.
- **Search box focus ring** on the documentation site never reached the search input due to a CSS
  selector mismatch.
- **W&B integration no longer forwards its API key to Sentry**, and the docs site's same-origin
  script-loading claim is now verified rather than asserted.
- Four factual corrections and a vocabulary clarification across the documentation site; every
  remaining numeric/version/timing claim in the README and docs was individually traced back to the
  file that determines it, with further corrections (test count, GIF frame count matching its alt
  text, a "verified" claim that covered 33 of 35 result files rather than all of them, a stale
  documentation screenshot, a wrong "highest-CodeBLEU config" name, and an unmeasured "96 seconds"
  replaced with the measured 42).
- BSD-3-Clause notices for the CLM and Hamcrest material were paraphrased rather than reproduced
  verbatim, which is what the license conditions actually require; both are now quoted in full, the
  HumanEval-Java licensing chain is settled rather than left as an open question, and a self-
  contradictory "no share-alike component anywhere" claim (written next to two named MPL/EPL
  components) was corrected.

### Changed

- Package version bumped `0.1.0` → `1.0.1` in `pyproject.toml` and `src/pop/__init__.py`, matching
  the already-published `v1.0.0` GitHub Release tag for the first time.
- Dependency bumps: the two GitHub Pages actions, `fastjsonschema` (relocked). No dependency used in
  the measurement path moved.
- `.gitignore` widened so the vendored-jar exception (`benchmarks/lib/*.jar` staying tracked) can't
  be silently defeated by a broader ignore rule added later.

### Test suite

386 tests at `v1.0.0` to 448 at this release, all green, hermetic, and network-free by construction
(see above).

## v1.0.0 - 2026-07-25

First public release. The study is complete and every number in it traces to a committed file.

### The study

- Four arms compared on Java bug fixing: A, a T5-small pretrained on CodeSearchNet then finetuned;
  B, the same T5-small finetuned from scratch; C, Qwen2.5-Coder-1.5B-Instruct prompted with one
  retrieved exemplar; D, the same Qwen with a LoRA adapter.
- Two evaluation tracks. Track 1 scores CodeBLEU, exact match and syntax validity on the 6,545-pair
  CodeXGLUE test split. Track 2 compiles and runs each candidate against 201 real bugs from
  QuixBugs and HumanEval-Java through a JDK harness.
- Headline result: the two tracks disagree at the top of the ranking. LoRA leads CodeBLEU at 0.854
  while RAG fixes more real bugs, 35.8% pass@1 against 26.4%. Both T5 arms score near 0.48 CodeBLEU
  and fix zero of the 201 bugs.
- The execution harness was validated 201/201 on the benchmarks' own reference patches before any
  model was scored.

### The repository

- `pop` command line tool covering data preparation, tokenizer training, the three training modes,
  evaluation, RAG, and the execution harness.
- The CPU path reproduces the study's figures and site from committed results in about 42 seconds
  from a cold clone. Nothing in it retrains a model or needs a GPU. (This entry originally said 96
  seconds, an estimate that was never measured; 42 s is the measured clone-to-first-result time on
  Windows 11 with an empty `uv` cache.)
- 386 tests, hermetic and network-free. CI runs lint, format, the suite, a real JDK harness smoke
  and the documented CPU reproduce path on every push.
- Documentation site with the full report, measurement notes, and an architecture page.
