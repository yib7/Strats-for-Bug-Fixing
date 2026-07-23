"""Tests for pop.execbench: result classification, pass@k, manifest integrity, and CLI wiring.

Most tests here are pure (no JDK needed): `classify_outcome` is unit tested against canned
javac/JUnit stdout+returncode fixtures, `pass_at_k` against hand-computed values, and the
vendored `manifest.json` files are checked for internal + on-disk consistency. One real
end-to-end compile+run test is marked `@pytest.mark.jdk` and skipped if no local JDK is found.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pop.execbench.harness import (
    BENCHMARKS_DIR,
    JdkNotFoundError,
    classify_outcome,
    get_bug_entry,
    load_manifest,
    normalize_package,
    resolve_jdk,
)
from pop.execbench.score import aggregate, pass_at_k

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- canned javac/JUnit output fixtures (no JDK involved) -----------------------------------

CANNED_JAVAC_ERROR = (
    "BITCOUNT.java:5: error: ';' expected\n    int result = 0\n                  ^\n1 error\n"
)

CANNED_JUNIT_SUCCESS = "JUnit version 4.13.2\n.........\n\nTime: 0.012\n\nOK (9 tests)\n\n"

CANNED_JUNIT_FAILURE = (
    "JUnit version 4.13.2\n.F.......\n\nTime: 0.015\nThere was 1 failure:\n"
    "1) test_1(java_testcases.junit.BITCOUNT_TEST)\n"
    "org.junit.ComparisonFailure: expected:<1> but was:<0>\n\n"
    "FAILURES!!!\nTests run: 9,  Failures: 1\n\n"
)


class TestClassifyOutcome:
    def test_compile_timeout(self):
        result = classify_outcome("BITCOUNT", "quixbugs", None, "compiling forever...")
        assert result.compiled is False
        assert result.passed is False
        assert result.error_kind == "timeout"
        assert "compiling forever" in result.stdout_tail

    def test_compile_error(self):
        result = classify_outcome("BITCOUNT", "quixbugs", 1, CANNED_JAVAC_ERROR)
        assert result.compiled is False
        assert result.passed is False
        assert result.error_kind == "compile_error"
        assert "';' expected" in result.stdout_tail

    def test_run_timeout(self):
        result = classify_outcome("BITCOUNT", "quixbugs", 0, "", None, "hung test...")
        assert result.compiled is True
        assert result.passed is False
        assert result.error_kind == "timeout"

    def test_test_failure(self):
        result = classify_outcome("BITCOUNT", "quixbugs", 0, "", 1, CANNED_JUNIT_FAILURE)
        assert result.compiled is True
        assert result.passed is False
        assert result.error_kind == "test_failure"
        assert "FAILURES" in result.stdout_tail

    def test_ok(self):
        result = classify_outcome("BITCOUNT", "quixbugs", 0, "", 0, CANNED_JUNIT_SUCCESS)
        assert result.compiled is True
        assert result.passed is True
        assert result.error_kind == "ok"

    def test_stdout_tail_truncated(self):
        huge = "x" * 10_000
        result = classify_outcome("BITCOUNT", "quixbugs", 1, huge)
        assert len(result.stdout_tail) <= 4000
        assert result.stdout_tail == huge[-4000:]

    def test_bench_field_carried_through(self):
        result = classify_outcome("ADD", "humaneval_java", 0, "", 0, CANNED_JUNIT_SUCCESS)
        assert result.bench == "humaneval_java"
        assert result.bug_id == "ADD"


class TestNormalizePackage:
    def test_replaces_existing_package(self):
        src = "package correct_java_programs;\n\npublic class BITCOUNT {}\n"
        out = normalize_package(src, "package java_programs;")
        assert out.startswith("package java_programs;")
        assert "correct_java_programs" not in out

    def test_inserts_missing_package(self):
        src = "public class ADD {}\n"
        out = normalize_package(src, "package humaneval.buggy;")
        assert out.startswith("package humaneval.buggy;\n")
        assert "public class ADD" in out

    def test_only_first_package_line_replaced(self):
        # A "package" mention inside a comment/string after the real declaration is untouched.
        src = "package correct_java_programs;\n// not a real package statement here\n"
        out = normalize_package(src, "package java_programs;")
        assert out.count("package java_programs;") == 1
        assert "not a real package statement" in out


# --- resolve_jdk (fake JDK homes, no real JDK needed) -----------------------------------------


def _exe_name(name: str) -> str:
    import platform

    return f"{name}.exe" if platform.system() == "Windows" else name


class TestResolveJdk:
    def test_none_resolves_to_path_lookup(self):
        javac, java = resolve_jdk(None)
        assert javac == "javac"
        assert java == "java"

    def test_valid_jdk_home_resolves_bin_paths(self, tmp_path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        javac_path = bin_dir / _exe_name("javac")
        java_path = bin_dir / _exe_name("java")
        javac_path.write_text("stub")
        java_path.write_text("stub")

        javac, java = resolve_jdk(tmp_path)
        assert javac == str(javac_path)
        assert java == str(java_path)

    def test_missing_jdk_home_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        with pytest.raises(JdkNotFoundError):
            resolve_jdk(missing)

    def test_jdk_home_missing_binaries_raises(self, tmp_path):
        (tmp_path / "bin").mkdir()
        with pytest.raises(JdkNotFoundError):
            resolve_jdk(tmp_path)

    def test_jdk_home_missing_java_only_raises(self, tmp_path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / _exe_name("javac")).write_text("stub")
        # java binary intentionally absent
        with pytest.raises(JdkNotFoundError):
            resolve_jdk(tmp_path)


class TestJdkIdentity:
    def test_missing_jdk_home_reports_error_not_raise(self, tmp_path):
        from pop.execbench.harness import jdk_identity

        missing = tmp_path / "nope"
        info = jdk_identity(missing)
        assert info["java"] is None
        assert info["version"] is None
        assert "error" in info
        assert info["jdk"] == str(missing)


# --- pass_at_k --------------------------------------------------------------------------------


class TestPassAtK:
    def test_c_equals_n_always_passes(self):
        assert pass_at_k(n=5, c=5, k=1) == pytest.approx(1.0)
        assert pass_at_k(n=5, c=5, k=5) == pytest.approx(1.0)

    def test_c_equals_zero_never_passes(self):
        assert pass_at_k(n=5, c=0, k=1) == pytest.approx(0.0)
        assert pass_at_k(n=5, c=0, k=5) == pytest.approx(0.0)

    def test_n_equals_one(self):
        assert pass_at_k(n=1, c=1, k=1) == pytest.approx(1.0)
        assert pass_at_k(n=1, c=0, k=1) == pytest.approx(0.0)

    def test_hand_computed_n5_c2_k1(self):
        # pass@1 with 2/5 correct == c/n by definition of the unbiased estimator.
        assert pass_at_k(n=5, c=2, k=1) == pytest.approx(2 / 5)

    def test_hand_computed_n10_c3_k5(self):
        # 1 - C(7,5)/C(10,5) = 1 - 21/252 = 1 - 1/12
        assert pass_at_k(n=10, c=3, k=5) == pytest.approx(1 - 21 / 252)

    def test_k_greater_than_n_raises(self):
        with pytest.raises(ValueError):
            pass_at_k(n=3, c=1, k=5)

    def test_c_greater_than_n_raises(self):
        with pytest.raises(ValueError):
            pass_at_k(n=3, c=5, k=1)

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            pass_at_k(n=3, c=1, k=0)

    def test_negative_n_or_c_raises(self):
        with pytest.raises(ValueError):
            pass_at_k(n=-1, c=0, k=1)
        with pytest.raises(ValueError):
            pass_at_k(n=3, c=-1, k=1)

    def test_result_is_bounded(self):
        for n, c, k in [(20, 7, 3), (100, 50, 10), (1, 1, 1), (1, 0, 1)]:
            val = pass_at_k(n, c, k)
            assert 0.0 <= val <= 1.0


# --- aggregate ----------------------------------------------------------------------------


def _fake_result(bug_id, bench, compiled, passed, error_kind):
    from pop.execbench.harness import ExecResult

    return ExecResult(bug_id, compiled, passed, error_kind, "", bench)


class TestAggregate:
    def test_empty(self):
        assert aggregate([]) == {"n": 0, "compile_rate": 0.0, "pass_rate": 0.0, "per_benchmark": {}}

    def test_mixed_results(self):
        results = [
            _fake_result("A", "quixbugs", True, True, "ok"),
            _fake_result("B", "quixbugs", True, False, "test_failure"),
            _fake_result("C", "humaneval_java", False, False, "compile_error"),
            _fake_result("D", "humaneval_java", True, True, "ok"),
        ]
        metrics = aggregate(results)
        assert metrics["n"] == 4
        assert metrics["compile_rate"] == pytest.approx(3 / 4)
        assert metrics["pass_rate"] == pytest.approx(2 / 4)
        assert metrics["per_benchmark"]["quixbugs"]["pass_rate"] == pytest.approx(0.5)
        assert metrics["per_benchmark"]["humaneval_java"]["pass_rate"] == pytest.approx(0.5)
        assert metrics["per_benchmark"]["quixbugs"]["error_kind_counts"] == {
            "ok": 1,
            "test_failure": 1,
        }


# --- manifest integrity -----------------------------------------------------------------------


class TestManifestIntegrity:
    @pytest.mark.parametrize(
        ("bench", "expected_count"),
        [("quixbugs", 40), ("humaneval_java", 161)],
    )
    def test_manifest_exists_with_expected_count(self, bench, expected_count):
        manifest = load_manifest(bench)
        assert len(manifest) == expected_count

    @pytest.mark.parametrize("bench", ["quixbugs", "humaneval_java"])
    def test_every_referenced_file_exists(self, bench):
        bench_dir = BENCHMARKS_DIR / bench
        for entry in load_manifest(bench):
            for key in ("buggy_file", "fixed_file"):
                path = bench_dir / entry[key]
                assert path.is_file(), f"{bench}/{entry['bug_id']}: missing {key} {path}"
            for rel in entry["test_files"]:
                path = bench_dir / rel
                assert path.is_file(), f"{bench}/{entry['bug_id']}: missing test file {path}"
            for rel in entry.get("support_files", []):
                path = bench_dir / rel
                assert path.is_file(), f"{bench}/{entry['bug_id']}: missing support file {path}"

    @pytest.mark.parametrize("bench", ["quixbugs", "humaneval_java"])
    def test_bug_ids_unique(self, bench):
        ids = [entry["bug_id"] for entry in load_manifest(bench)]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("bench", ["quixbugs", "humaneval_java"])
    def test_required_keys_present(self, bench):
        required = {"bug_id", "buggy_file", "fixed_file", "test_files", "entry_test_class"}
        for entry in load_manifest(bench):
            assert required.issubset(entry.keys())

    def test_get_bug_entry_found(self):
        entry = get_bug_entry("quixbugs", "BITCOUNT")
        assert entry["bug_id"] == "BITCOUNT"

    def test_get_bug_entry_missing_raises(self):
        with pytest.raises(KeyError):
            get_bug_entry("quixbugs", "NOT_A_REAL_BUG")

    def test_lib_jars_present(self):
        jars = list((BENCHMARKS_DIR / "lib").glob("*.jar"))
        assert len(jars) >= 2, "expected at least junit + hamcrest jars in benchmarks/lib"


# --- predictions CLI mode, stub harness (no JDK) -----------------------------------------------


def test_execbench_predictions_mode_with_stub_harness(tmp_path, monkeypatch):
    """--predictions is exercised only with a stubbed run_bug (no real javac/JUnit)."""
    import pop.execbench.harness as harness_mod
    from pop.cli import main as cli_main
    from pop.execbench.harness import ExecResult

    def fake_run_bug(bug_id, candidate_src, bench, jdk=None, timeout_s=30, workdir=None):
        passed = "GOOD" in candidate_src
        return ExecResult(bug_id, True, passed, "ok" if passed else "test_failure", "", bench)

    monkeypatch.setattr(harness_mod, "run_bug", fake_run_bug)

    predictions = tmp_path / "preds.jsonl"
    predictions.write_text(
        json.dumps({"bug_id": "BITCOUNT", "prediction": "GOOD candidate"})
        + "\n"
        + json.dumps({"bug_id": "GCD", "prediction": "BAD candidate"})
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    rc = cli_main(["execbench", "--predictions", str(predictions), "--bench", "quixbugs"])
    assert rc == 0

    results_files = list((tmp_path / "results").glob("*.json"))
    assert len(results_files) == 1
    payload = json.loads(results_files[0].read_text(encoding="utf-8"))
    assert payload["metrics"]["n"] == 2
    assert payload["metrics"]["pass_rate"] == pytest.approx(0.5)


def test_execbench_predictions_mode_requires_bench_when_all(tmp_path, monkeypatch):
    predictions = tmp_path / "preds.jsonl"
    predictions.write_text(json.dumps({"bug_id": "BITCOUNT", "prediction": "x"}) + "\n", "utf-8")
    monkeypatch.chdir(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "pop.cli", "execbench", "--predictions", str(predictions)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 1
    assert "bench" in result.stderr.lower()


# --- real end-to-end (needs a local JDK) --------------------------------------------------


@pytest.mark.jdk
@pytest.mark.skipif(shutil.which("javac") is None, reason="no local JDK (javac not on PATH)")
class TestEndToEndOneBug:
    BUG_ID = "BITCOUNT"
    BENCH = "quixbugs"

    def test_reference_passes(self):
        from pop.execbench.harness import run_bug

        entry = get_bug_entry(self.BENCH, self.BUG_ID)
        fixed_src = (BENCHMARKS_DIR / self.BENCH / entry["fixed_file"]).read_text(encoding="utf-8")
        result = run_bug(self.BUG_ID, fixed_src, self.BENCH, timeout_s=30)
        assert result.compiled is True
        assert result.passed is True
        assert result.error_kind == "ok"

    def test_buggy_source_fails_tests(self):
        from pop.execbench.harness import run_bug

        entry = get_bug_entry(self.BENCH, self.BUG_ID)
        buggy_src = (BENCHMARKS_DIR / self.BENCH / entry["buggy_file"]).read_text(encoding="utf-8")
        result = run_bug(self.BUG_ID, buggy_src, self.BENCH, timeout_s=30)
        assert result.compiled is True
        assert result.passed is False
        assert result.error_kind == "test_failure"

    def test_syntactically_broken_source_is_compile_error(self):
        from pop.execbench.harness import run_bug

        broken_src = "package java_programs;\npublic class BITCOUNT { this is not java "
        result = run_bug(self.BUG_ID, broken_src, self.BENCH, timeout_s=30)
        assert result.compiled is False
        assert result.error_kind == "compile_error"
