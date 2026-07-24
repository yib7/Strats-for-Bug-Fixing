# Changelog

Versions follow [semantic versioning](https://semver.org/). Each entry uses the same shape:
`## vX.Y.Z - YYYY-MM-DD`, a one-line summary, then grouped bullets.

## v1.0.0 - 2026-07-24

First public release. The study is complete and every number in it traces to a committed file.

### The study

- Four arms compared on Java bug fixing: A, a T5-small pretrained on CodeSearchNet then finetuned;
  B, the same T5-small finetuned from scratch; C, Qwen2.5-Coder-1.5B-Instruct prompted with one
  retrieved exemplar; D, Qwen2.5-Coder-1.5B with a LoRA adapter.
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
- The CPU path reproduces the study's figures and site from committed results in about 96 seconds
  from a cold clone. Nothing in it retrains a model or needs a GPU.
- 384 tests, hermetic and network-free. CI runs lint, format, the suite and a real JDK harness
  smoke on every push.
- Documentation site with the full report, measurement notes, and an architecture page.
