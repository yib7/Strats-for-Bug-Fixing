"""Build `manifest.json` for the vendored execbench benchmarks.

Rerunnable: scans `benchmarks/quixbugs/` and `benchmarks/humaneval_java/` (vendored per
`PROVENANCE.md` in each directory) and writes each benchmark's `manifest.json` — a list of
`{bug_id, buggy_file, fixed_file, test_files, entry_test_class, support_files}` records, with
paths relative to the benchmark directory. `harness.py` and `test_execbench.py` both consume
this file, so keep its schema stable.

Usage:
    uv run python scripts/build_benchmark_manifests.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"

# Bugs excluded from the manifest because their *reference* (fixed) solution cannot pass on a
# modern JDK regardless of correctness -- both depend on JDK APIs removed after JDK 8. Verified
# by running the harness with `--validate-references`: each was the sole remaining failure with
# no code-path that would make it pass. Documented here rather than silently dropped; see
# `benchmarks/humaneval_java/PROVENANCE.md` for the fuller writeup.
HUMANEVAL_JAVA_EXCLUDED = {
    "DO_ALGEBRA": (
        'uses javax.script.ScriptEngineManager.getEngineByName("JavaScript") (Nashorn), '
        "removed from the JDK in JDK 15 (JEP 372) -- getEngineByName returns null, so the "
        "reference solution throws NullPointerException on every test on JDK 15+."
    ),
    "STRING_TO_MD5": (
        "imports javax.xml.bind.DatatypeConverter (JAXB), removed from the JDK in JDK 11 "
        "(JEP 320) -- the reference solution fails to compile on JDK 11+."
    ),
}


def build_quixbugs_manifest() -> list[dict]:
    bench_dir = BENCHMARKS_DIR / "quixbugs"
    buggy_dir = bench_dir / "java_programs"
    fixed_dir = bench_dir / "correct_java_programs"
    test_dir = bench_dir / "java_testcases" / "junit"

    # Shared support classes required to compile several java_programs/*.java files but that
    # are not themselves "bugs" (no counterpart under correct_java_programs/).
    support_files = sorted(
        f"java_programs/{p.name}"
        for p in buggy_dir.glob("*.java")
        if not (fixed_dir / p.name).is_file()
    )

    entries = []
    for fixed_path in sorted(fixed_dir.glob("*.java")):
        bug_id = fixed_path.stem
        buggy_path = buggy_dir / fixed_path.name
        test_path = test_dir / f"{bug_id}_TEST.java"
        helper_path = test_dir / "QuixFixOracleHelper.java"
        if not buggy_path.is_file():
            raise FileNotFoundError(f"quixbugs: missing buggy file for {bug_id}: {buggy_path}")
        if not test_path.is_file():
            raise FileNotFoundError(f"quixbugs: missing test file for {bug_id}: {test_path}")

        test_files = [f"java_testcases/junit/{test_path.name}"]
        if helper_path.is_file():
            test_files.append(f"java_testcases/junit/{helper_path.name}")

        entries.append(
            {
                "bug_id": bug_id,
                "buggy_file": f"java_programs/{buggy_path.name}",
                "fixed_file": f"correct_java_programs/{fixed_path.name}",
                "test_files": test_files,
                "entry_test_class": f"java_testcases.junit.{bug_id}_TEST",
                "support_files": support_files,
            }
        )
    return entries


def build_humaneval_java_manifest() -> list[dict]:
    bench_dir = BENCHMARKS_DIR / "humaneval_java"
    buggy_dir = bench_dir / "src" / "main" / "java" / "humaneval" / "buggy"
    fixed_dir = bench_dir / "src" / "main" / "java" / "humaneval" / "correct"
    test_dir = bench_dir / "src" / "test" / "java" / "humaneval"

    entries = []
    for fixed_path in sorted(fixed_dir.glob("*.java")):
        bug_id = fixed_path.stem
        if bug_id in HUMANEVAL_JAVA_EXCLUDED:
            continue
        buggy_path = buggy_dir / fixed_path.name
        test_path = test_dir / f"TEST_{bug_id}.java"
        if not buggy_path.is_file():
            raise FileNotFoundError(
                f"humaneval_java: missing buggy file for {bug_id}: {buggy_path}"
            )
        if not test_path.is_file():
            raise FileNotFoundError(f"humaneval_java: missing test file for {bug_id}: {test_path}")

        entries.append(
            {
                "bug_id": bug_id,
                "buggy_file": f"src/main/java/humaneval/buggy/{buggy_path.name}",
                "fixed_file": f"src/main/java/humaneval/correct/{fixed_path.name}",
                "test_files": [f"src/test/java/humaneval/{test_path.name}"],
                "entry_test_class": f"humaneval.TEST_{bug_id}",
                "support_files": [],
            }
        )
    return entries


def main() -> int:
    quixbugs = build_quixbugs_manifest()
    (BENCHMARKS_DIR / "quixbugs" / "manifest.json").write_text(
        json.dumps(quixbugs, indent=2) + "\n", encoding="utf-8"
    )
    print(f"quixbugs: wrote {len(quixbugs)} entries")

    humaneval_java = build_humaneval_java_manifest()
    (BENCHMARKS_DIR / "humaneval_java" / "manifest.json").write_text(
        json.dumps(humaneval_java, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"humaneval_java: wrote {len(humaneval_java)} entries "
        f"({len(HUMANEVAL_JAVA_EXCLUDED)} excluded: {sorted(HUMANEVAL_JAVA_EXCLUDED)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
