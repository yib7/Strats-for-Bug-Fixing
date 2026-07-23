# Provenance — HumanEval-Java

- Source repo: https://github.com/ASSERT-KTH/human-eval-java
- Commit: `ed75a3e0e8d0c97a632885d67281b26218a3a57f` (default branch, shallow clone fetched 2026-07-17)
- Upstream lineage per that repo's README: transformed from OpenAI's HumanEval
  (https://github.com/openai/human-eval, MIT) via the ASE'23 CLM replication package
  (https://github.com/lin-tan/clm).
- License: **no `LICENSE` file is present in the source repo at the vendored commit.** No
  `LICENSE.md`/`COPYING` either. The repo's own README does not state redistribution terms.
  The underlying OpenAI HumanEval problem statements are MIT-licensed; the Java transformation
  (buggy-program mutations + JUnit ports) by the CLM/ASSERT-KTH authors carries no explicit
  license grant in this snapshot. Vendored here read-only for non-commercial research benchmark
  use (execution-harness testing), consistent with how the dataset is used by the APR research
  community (ASE'23 CLM paper and follow-ups). No `LICENSE` file is fabricated; this note stands
  in its place. Flagged as a documented gap, not a blocker per this task's benchmark-vendoring
  scope.

## What was vendored

- `src/main/java/humaneval/buggy/*.java` — 163 buggy program files (package `humaneval.buggy`).
- `src/main/java/humaneval/correct/*.java` — 163 reference (fixed) program files (package
  `humaneval.correct`).
- `src/test/java/humaneval/TEST_*.java` — 163 JUnit 4 test classes (package `humaneval`, each
  importing `humaneval.buggy.<NAME>`).

Not vendored: `target/` (Maven build output), `humaneval-java-sf.json`, `HumanEval.jsonl`,
`humaneval_loc.txt`, `mutation_operators.json`, `print_humaneval.py`, `diff_humaneval.py`,
`pom.xml`, and the repo's own vendored `lib/junit4-4.12.jar` + `lib/hamcrest-all-1.3.jar`
(this study instead vendors JUnit 4.13.2 + Hamcrest-core 1.3 from Maven Central at
`benchmarks/lib/`, shared with QuixBugs).

## Bug count

163 programs vendored (`buggy/*.java` == `correct/*.java` == `TEST_*.java` count == 163). The task
brief anticipated 164 (the size of the original OpenAI HumanEval problem set); this ASSERT-KTH
snapshot ports 163 of the 164 problems to Java (one problem was apparently dropped/not portable in
the upstream CLM replication package — not investigated further here, and not fabricated to reach
164). This is documented as an assumption, not a blocker.

Of those 163, **2 are excluded from `manifest.json`** (161 entries actually shipped) because their
*reference* (fixed) solution cannot pass on any JDK newer than 8, independent of correctness —
confirmed by running `--validate-references` and seeing each as the sole remaining failure:

- `DO_ALGEBRA` — calls `javax.script.ScriptEngineManager.getEngineByName("JavaScript")` (Nashorn),
  removed from the JDK in JDK 15 (JEP 372). `getEngineByName` returns `null` on JDK 15+, so the
  reference solution throws `NullPointerException` on every test.
- `STRING_TO_MD5` — imports `javax.xml.bind.DatatypeConverter` (JAXB), removed from the JDK in
  JDK 11 (JEP 320). The reference solution fails to compile on JDK 11+.

Both are upstream benchmark/packaging issues (the CLM replication package targeted JDK 8, per its
`pom.xml`'s `maven.compiler.target=1.8`), not harness bugs — this repo's local JDK is Temurin 21
and CI runs Temurin 17, both post-dating the removed APIs. The buggy/correct/test source files for
both remain vendored on disk (not deleted) for completeness; only `scripts/build_benchmark_manifests.py`
(see `HUMANEVAL_JAVA_EXCLUDED`) skips them when generating `manifest.json`. Effective harness-usable
count: **161**.

## Test framework

JUnit 4 (`org.junit.Test`, `org.junit.Assert`), matching QuixBugs. Same vendored jars apply.
