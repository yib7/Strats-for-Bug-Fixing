# Provenance — QuixBugs (Java subset)

- Source repo: https://github.com/jkoppel/QuixBugs
- Commit: `4257f44b0ff1181dedaedee6a447e133219fcebf` (default branch, shallow clone fetched 2026-07-17)
- License: MIT (see `LICENSE` in this directory, copied verbatim from the source repo)

## What was vendored

Only the Java-language subset (this study does not use the Python programs/tests):

- `java_programs/*.java` — the 40 buggy program files, plus two shared support classes
  (`Node.java`, `WeightedEdge.java`) that several buggy programs depend on for compilation.
  These two are not "bugs" themselves (they have no counterpart in `correct_java_programs/`).
- `correct_java_programs/*.java` — the 40 reference (fixed) program files.
- `java_testcases/junit/*.java` — the 40 JUnit 4 test classes (`java_testcases.junit.<NAME>_TEST`,
  each importing `java_programs.<NAME>`) plus `QuixFixOracleHelper.java`, a shared formatting
  helper used by several tests.

Not vendored: `python_programs/`, `python_testcases/`, `json_testcases/`, the `.class` files
that ship in the source repo, `quixbugs.pdf`, `conftest.py`, `build.gradle`, and the
`java_testcases/junit/crt_program/` variants (identical tests hardcoded to import
`correct_java_programs.*` instead of `java_programs.*` — redundant with this harness's approach
of swapping the candidate source directly into the `java_programs` package location).

## Test framework

JUnit 4 (`org.junit.Test`, `org.junit.Assert`) — see `@org.junit.Test(timeout = 3000)` annotations
in the vendored test sources. Matching jars vendored at `benchmarks/lib/` (JUnit 4.13.2 +
Hamcrest-core 1.3, fetched from Maven Central).

## Bug count

40 programs (`java_programs/*.java` minus `Node.java`/`WeightedEdge.java` == 40, matching
`correct_java_programs/*.java` count of 40 and `java_testcases/junit/*_TEST.java` count of 40).
