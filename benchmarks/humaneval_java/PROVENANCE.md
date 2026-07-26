# Provenance — HumanEval-Java

- Source repo: https://github.com/ASSERT-KTH/human-eval-java
- Commit: `ed75a3e0e8d0c97a632885d67281b26218a3a57f` (default branch, shallow clone fetched 2026-07-17)
- Upstream lineage per that repo's README: transformed from OpenAI's HumanEval
  (https://github.com/openai/human-eval, MIT) via the ASE'23 CLM replication package
  (https://github.com/lin-tan/clm).
- License: **no `LICENSE` file is present in the source repo** — not at the vendored commit and
  not at the current default branch (re-checked 2026-07-24). No `LICENSE.md`/`COPYING` either, and
  the repo's own README does not state redistribution terms.

### Upstream licence chain

The two upstreams this snapshot derives from *are* licensed permissively:

| Layer | Source | License |
|---|---|---|
| Original problem set | [openai/human-eval](https://github.com/openai/human-eval) | MIT |
| Java transformation (buggy mutants + JUnit ports), ASE'23 CLM replication package | [lin-tan/clm](https://github.com/lin-tan/clm) | BSD 3-Clause, "Copyright (c) 2023", ASSET research group, Purdue University (verified 2026-07-24) |
| Snapshot vendored here | [ASSERT-KTH/human-eval-java](https://github.com/ASSERT-KTH/human-eval-java) | **none stated** |

Both upstream grants (MIT and BSD 3-Clause) permit redistribution with attribution, and the
ASSERT-KTH repo presents itself as a redistribution of that CLM work rather than as new authorship.
On that reading, redistributing this subset with attribution to all three layers is permitted. That
inference is **not** a substitute for an explicit grant, however: the vendored snapshot itself
carries none, so its status is best described as *permissive by lineage, unstated at the snapshot*.

### Decision: ship it, with all three layers attributed

This was weighed and settled before the v1.0.0 release; it is a closed decision, recorded here so a
reader can follow the reasoning rather than re-litigate it.

The snapshot's silence is an **absent statement, not a refusal**. ASSERT-KTH/human-eval-java
describes itself as a repackaging of the CLM replication package's Java port; it does not claim new
authorship over the ported programs, and it is the CLM package — BSD 3-Clause — that produced them
from OpenAI's MIT-licensed problem set. Redistributing this subset while satisfying the upstream
conditions therefore stays inside the only grants anyone has actually made over this material. What
satisfying them looks like in practice: nothing is modified, nothing is relicensed, no `LICENSE`
file is fabricated for the snapshot, the BSD 3-Clause notice and disclaimer are reproduced verbatim
in [`CREDITS.md`](../../CREDITS.md), and all three layers are named there and here.

Two alternatives were considered and rejected:

- **Drop the benchmark.** It supplies 161 of the 201 bugs the execution harness runs. Losing it
  guts the execution-vs-CodeBLEU comparison that is the study's headline finding, at no gain in
  license certainty for the QuixBugs half that would remain.
- **Re-derive the Java port from lin-tan/clm directly.** This produces the same artifacts under the
  same BSD 3-Clause terms. It removes one attribution layer from the chain but improves nothing
  about the grant, while discarding the ability to state the exact upstream commit vendored.

The residual risk is that ASSERT-KTH turns out to hold a copyright interest in the snapshot that it
has never asserted and would want to restrict. The material is vendored read-only for research
benchmark use, which is how this dataset is used across the APR literature. If ASSERT-KTH later
states terms, this note and `CREDITS.md` get updated to match them.

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
