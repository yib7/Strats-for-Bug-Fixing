# Provenance — benchmarks/lib

Fetched read-only from Maven Central on 2026-07-17 (no build required to obtain them):

- `junit-4.13.2.jar` — https://repo1.maven.org/maven2/junit/junit/4.13.2/junit-4.13.2.jar
  License: Eclipse Public License 1.0 (text carried inside the jar at `LICENSE-junit.txt`).
  Source, as EPL-1.0 §3 requires a binary distributor to state:
  https://github.com/junit-team/junit4/tree/r4.13.2
  SHA-1 `8ac9e16d933b6fb43bc7f576336b8f4d7eb5ba12`.
- `hamcrest-core-1.3.jar` — https://repo1.maven.org/maven2/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar
  License: BSD 3-Clause (text carried inside the jar at `LICENSE.txt`).
  SHA-1 `42a25dc3219429f0e5d060061f71acb49bf010a0`.

Both jars are unmodified: each SHA-1 above matches the `.sha1` Maven Central publishes next to the
artifact (re-verified 2026-07-25). Reproduce with `sha1sum benchmarks/lib/*.jar`. The notices both
licenses require a redistributor to carry are reproduced in [`CREDITS.md`](../../CREDITS.md).

Used as the compile+run classpath for both `benchmarks/quixbugs` and `benchmarks/humaneval_java`,
both of which use JUnit 4 (`org.junit.Test`) test sources.
