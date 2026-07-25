"""Compile-and-test harness for the QuixBugs-Java / HumanEval-Java execution benchmarks.

`run_bug` writes a candidate source in place of a benchmark's buggy file, compiles it
together with the benchmark's other required Java sources (the JUnit test class(es) plus
any shared support classes named in the manifest) using `javac`, then runs the entry JUnit
test class via `java org.junit.runner.JUnitCore <class>` under a subprocess timeout. On
timeout the whole process (tree) is killed -- `taskkill /T /F` on Windows, process-group
SIGKILL on POSIX.

`classify_outcome` is split out as a pure function (no subprocess/JDK) so result-parsing
logic can be unit tested against canned javac/JUnit stdout+returncode fixtures.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
LIB_DIR = BENCHMARKS_DIR / "lib"

_STDOUT_TAIL_CHARS = 4000

# Bound on how much subprocess output is held in memory per stage (see `_run_with_timeout`).
# 1 MiB is ~250x what `_tail` reports, so nothing a human would read is ever lost.
_MAX_CAPTURE_CHARS = 1 << 20
_READ_BLOCK_CHARS = 1 << 16
_MAX_CAPTURE_BLOCKS = _MAX_CAPTURE_CHARS // _READ_BLOCK_CHARS

# Max JVM heap for a candidate's JUnit run: stops a generated patch that allocates in a loop
# from taking the JVM default (25% of physical RAM) -- times `--jobs`.
#
# Sized against the benchmarks, not guessed. The heaviest legitimate test is QuixBugs
# KNAPSACK test_9 (capacity 6_404_180, 24 items), whose `int[25][6404181]` memo table is
# 611 MiB on its own. Measured on the reference sources with JDK 21: -Xmx512m fails it with
# `OutOfMemoryError: Java heap space` (reference validation drops to 200/201), 768m passes.
# 2g is ~3.3x the largest real allocation -- enough headroom for a different JDK's GC while
# still bounding a runaway candidate far below the default.
_JVM_MAX_HEAP = "-Xmx2g"

# Force the JVM to write its diagnostics as UTF-8, matching how `_run_with_timeout` decodes
# them. Without this the JDK picks the *native* encoding for a redirected stream (cp1252 on a
# typical Windows box), so the same bug produces different `stdout_tail` bytes on Windows and
# on Linux CI. `sun.*` is the JDK 17 spelling, the unprefixed pair is JDK 19+; setting an
# unknown system property is harmless, so both are passed.
_JVM_UTF8_PROPS = (
    "-Dstdout.encoding=UTF-8",
    "-Dstderr.encoding=UTF-8",
    "-Dsun.stdout.encoding=UTF-8",
    "-Dsun.stderr.encoding=UTF-8",
)
# `javac` is itself a Java program; `-J` forwards an option to its launcher VM.
_JAVAC_UTF8_PROPS = tuple(f"-J{prop}" for prop in _JVM_UTF8_PROPS)
_PACKAGE_RE = re.compile(r"^\s*package\s+[\w.]+\s*;", re.MULTILINE)

_MANIFEST_CACHE: dict[str, list[dict]] = {}


class JdkNotFoundError(ValueError):
    """Raised when a given JDK home is missing (or doesn't contain) the required binaries."""


@dataclasses.dataclass
class ExecResult:
    """Outcome of running one bug's candidate source through the harness.

    `error_kind` is one of: "ok" | "compile_error" | "test_failure" | "timeout" |
    "harness_error". `bench` is informational (used for per-benchmark aggregation in
    `pop.execbench.score`); it defaults to "" so the dataclass still matches the
    {bug_id, compiled, passed, error_kind, stdout_tail} shape when constructed positionally
    without it.
    """

    bug_id: str
    compiled: bool
    passed: bool
    error_kind: str
    stdout_tail: str
    bench: str = ""


def _tail(text: str, max_chars: int = _STDOUT_TAIL_CHARS) -> str:
    return text[-max_chars:] if len(text) > max_chars else text


def _bench_dir(bench: str) -> Path:
    """`benchmarks/<bench>/`, rejecting a `bench` that is not a bare directory name.

    `bench` is untrusted input: `pop execbench --predictions` takes it from the `bench`
    field of each JSONL record, so a hand-crafted predictions file could otherwise pass
    `../../..` and make the harness read a `manifest.json` (and the Java sources it names)
    from anywhere on disk -- and `bench` also lands in the `tempfile` prefix in `run_bug`.
    `Path(bench).name` strips any directory, drive and root, so it round-trips only for a
    bare name; "." and ".." are special-cased because pathlib keeps ".." whole. `\\` is
    tested explicitly rather than left to pathlib: it separates on Windows but is a legal
    filename character on POSIX, so without it the same predictions file would be rejected
    on one platform and accepted on the other.
    """
    if not bench or bench in {".", ".."} or "\\" in bench or Path(bench).name != bench:
        raise ValueError(
            f"benchmark name must be a bare directory name under benchmarks/ "
            f"(no path separators or '..'), got {bench!r}"
        )
    return BENCHMARKS_DIR / bench


def load_manifest(bench: str) -> list[dict]:
    if bench not in _MANIFEST_CACHE:
        path = _bench_dir(bench) / "manifest.json"
        _MANIFEST_CACHE[bench] = json.loads(path.read_text(encoding="utf-8"))
    return _MANIFEST_CACHE[bench]


def bench_source_path(bench: str, rel: str) -> Path:
    """Resolve a manifest-named source file to an absolute path inside `benchmarks/<bench>/`.

    Every Java path in a `manifest.json` (`buggy_file`, `fixed_file`, `test_files`,
    `support_files`) goes through here. The manifest is vendored, but it is still a data
    file that a contributor or a dropped-in benchmark can supply, and the files it names
    are read and fed to `javac` -- whose diagnostics quote source lines into
    `stdout_tail`, i.e. into `results/*.json`. Confining the resolved path to the
    benchmark directory keeps a `../../..` entry from turning that into an arbitrary-file
    read. Backslashes are normalised to `/` first so that a `..\\..\\` entry is caught on
    POSIX too, where pathlib would otherwise read it as one long filename that never
    leaves the directory -- the check has to hold wherever the manifest is read, not only
    where it was written.
    """
    base = _bench_dir(bench).resolve()
    path = (base / rel.replace("\\", "/")).resolve()
    if path != base and base not in path.parents:
        raise ValueError(f"{bench}: manifest path escapes the benchmark directory: {rel!r}")
    return path


def get_bug_entry(bench: str, bug_id: str) -> dict:
    for entry in load_manifest(bench):
        if entry["bug_id"] == bug_id:
            return entry
    raise KeyError(f"{bench}: unknown bug_id {bug_id!r}")


def classify_outcome(
    bug_id: str,
    bench: str,
    compile_rc: int | None,
    compile_out: str,
    run_rc: int | None = None,
    run_out: str = "",
) -> ExecResult:
    """Pure classification of a (compile, run) subprocess outcome pair.

    `compile_rc`/`run_rc` are `None` to signal a timeout at that stage (mirroring what
    `_run_with_timeout` returns). `run_rc`/`run_out` are ignored when compilation didn't
    succeed. No subprocess or JDK is invoked here -- this is the piece unit tests exercise
    against canned javac/JUnit output fixtures.
    """
    if compile_rc is None:
        return ExecResult(bug_id, False, False, "timeout", _tail(compile_out), bench)
    if compile_rc != 0:
        return ExecResult(bug_id, False, False, "compile_error", _tail(compile_out), bench)
    if run_rc is None:
        return ExecResult(bug_id, True, False, "timeout", _tail(run_out), bench)
    if run_rc != 0:
        return ExecResult(bug_id, True, False, "test_failure", _tail(run_out), bench)
    return ExecResult(bug_id, True, True, "ok", _tail(run_out), bench)


def normalize_package(candidate_src: str, target_package_decl: str) -> str:
    """Rewrite/insert `candidate_src`'s package declaration to `target_package_decl`.

    Both benchmarks' *fixed* reference sources live in a different Java package than the
    *buggy* file they replace (QuixBugs: `correct_java_programs` vs. `java_programs`;
    HumanEval-Java: `humaneval.correct` vs. `humaneval.buggy`) -- the JUnit tests always
    import the buggy package name, so a candidate source has to be dropped in under that
    package regardless of what package (if any) it declares. `target_package_decl` should be
    a full statement like `"package java_programs;"` (as read verbatim off the buggy file).
    """
    if _PACKAGE_RE.search(candidate_src):
        return _PACKAGE_RE.sub(target_package_decl, candidate_src, count=1)
    return f"{target_package_decl}\n{candidate_src}"


def _resolve_binary(jdk_home: Path, name: str) -> str:
    exe = f"{name}.exe" if platform.system() == "Windows" else name
    path = jdk_home / "bin" / exe
    if not path.is_file():
        raise JdkNotFoundError(f"JDK home {jdk_home} is missing {name} (expected at {path})")
    return str(path)


def resolve_jdk(jdk: str | Path | None) -> tuple[str, str]:
    """Resolve (javac, java) executables for `jdk`.

    `jdk` is treated as a JDK home directory: binaries are resolved as
    `<jdk>/bin/javac[.exe]` and `<jdk>/bin/java[.exe]`. `None` (the default) resolves to the
    bare command names `"javac"`/`"java"`, i.e. PATH lookup, same as before this option
    existed. Raises `JdkNotFoundError` if a given home doesn't exist or is missing either
    binary.
    """
    if jdk is None:
        return "javac", "java"
    jdk_home = Path(jdk)
    if not jdk_home.is_dir():
        raise JdkNotFoundError(f"JDK home not found: {jdk_home}")
    javac = _resolve_binary(jdk_home, "javac")
    java = _resolve_binary(jdk_home, "java")
    return javac, java


def jdk_identity(jdk: str | Path | None = None) -> dict:
    """Best-effort identity of the JDK that would be used for `run_bug(..., jdk=jdk)`.

    Returns a dict with the resolved `java` binary path and the captured `java -version`
    output (which includes `java.version`); if resolution or version capture fails, `error`
    is set and `version` is `None`. Meant to be captured once per run and recorded into the
    results summary JSON so every run documents which JDK produced it.
    """
    jdk_str = str(jdk) if jdk is not None else None
    try:
        _, java_bin = resolve_jdk(jdk)
    except JdkNotFoundError as e:
        return {"jdk": jdk_str, "java": None, "version": None, "error": str(e)}

    try:
        proc = subprocess.run(
            [java_bin, "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        version_out = (proc.stdout + proc.stderr).strip()
        return {"jdk": jdk_str, "java": java_bin, "version": version_out}
    except FileNotFoundError:
        # The single most likely failure for a visitor: no JDK installed at all.
        return {
            "jdk": jdk_str,
            "java": java_bin,
            "version": None,
            "error": f"{java_bin!r} not found (no JDK on PATH?)",
        }
    except Exception as e:  # defensive: identity capture must never crash a run
        return {
            "jdk": jdk_str,
            "java": java_bin,
            "version": None,
            "error": f"{e.__class__.__name__}: {e}",
        }


def _classpath_jars() -> list[str]:
    return [str(p) for p in sorted(LIB_DIR.glob("*.jar"))]


def _classpath_sep() -> str:
    return ";" if platform.system() == "Windows" else ":"


def _kill_tree(proc: subprocess.Popen) -> None:
    """Best-effort kill of `proc` and any children it spawned."""
    if platform.system() == "Windows":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        import signal

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.kill()
    except OSError:
        pass


def _run_with_timeout(cmd: list[str], cwd: Path, timeout_s: int) -> tuple[int | None, str]:
    """Run `cmd`; returns (returncode, combined stdout+stderr) or (None, tail) on timeout.

    The output is captured through a **bounded sliding window**: a reader thread keeps only
    the most recent `_MAX_CAPTURE_CHARS` characters and discards the rest. `cmd` here runs
    model-generated Java, so the stream is attacker-influenced and effectively unbounded --
    a candidate patch that prints in a loop was measured buffering 461 MiB in 6 s, which at
    the default 30 s timeout is ~2.7 GB per bug, times `--jobs`. Draining (rather than just
    not reading) is deliberate: a full pipe buffer would block an otherwise-fine child.

    The window keeps the *tail*, because that is what `_tail` reports and where a JUnit
    failure trace lives; when anything was dropped a one-line notice is prefixed.
    """
    popen_kwargs: dict = {}
    if platform.system() == "Windows":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    # encoding is pinned: javac is invoked with `-encoding UTF-8` and the benchmark sources
    # are UTF-8, so decoding with the platform locale (cp1252 on a typical Windows box)
    # turns compiler diagnostics that quote a source line into mojibake -- and bakes it
    # into results/*.json, where the same run then differs between Windows and Linux CI.
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        **popen_kwargs,
    )

    blocks: deque[str] = deque(maxlen=_MAX_CAPTURE_BLOCKS)
    dropped = False

    def _drain() -> None:
        nonlocal dropped
        stream = proc.stdout
        if stream is None:
            return
        try:
            while True:
                block = stream.read(_READ_BLOCK_CHARS)
                if not block:
                    return
                if len(blocks) == _MAX_CAPTURE_BLOCKS:
                    dropped = True
                blocks.append(block)  # deque.append is atomic; no lock needed
        except (OSError, ValueError):
            return  # the pipe was closed under us (kill/cleanup) -- keep what we have

    reader = threading.Thread(target=_drain, name="execbench-drain", daemon=True)
    reader.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    reader.join(5)
    if proc.stdout is not None:
        try:
            proc.stdout.close()
        except OSError:
            pass

    out = "".join(blocks)
    if dropped:
        out = f"[pop: kept only the last ~{_MAX_CAPTURE_CHARS} characters of output]\n{out}"
    return (None if timed_out else proc.returncode), out


def run_bug(
    bug_id: str,
    candidate_src: str,
    bench: str,
    jdk: str | Path | None = None,
    timeout_s: int = 30,
    workdir: Path | None = None,
) -> ExecResult:
    """Compile `candidate_src` in place of `bug_id`'s buggy file and run its JUnit test(s).

    `jdk`, if given, is a JDK home directory whose `bin/javac`/`bin/java` are used instead of
    the PATH-resolved defaults (see `resolve_jdk`) -- useful for benchmark bugs that need a
    specific JDK version (e.g. the Nashorn/JAXB-dependent HumanEval exclusions). `workdir`
    defaults to a fresh temp dir (cleaned up on return); pass one explicitly to inspect
    artifacts after a run.
    """
    try:
        javac_bin, java_bin = resolve_jdk(jdk)
    except JdkNotFoundError as e:
        return ExecResult(bug_id, False, False, "harness_error", str(e), bench)

    try:
        entry = get_bug_entry(bench, bug_id)
    except (KeyError, FileNotFoundError, ValueError) as e:
        # ValueError: `bench` is not a bare benchmark directory name (see `_bench_dir`).
        return ExecResult(bug_id, False, False, "harness_error", str(e), bench)

    tmp_ctx = None
    if workdir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix=f"execbench_{bench}_{bug_id}_")
        work = Path(tmp_ctx.name)
    else:
        work = Path(workdir)
        work.mkdir(parents=True, exist_ok=True)

    try:
        # Start from empty dirs even when an explicit `workdir` is reused: `javac` is handed
        # `src_dir.glob("*.java")`, so a previous bug's leftover sources would be compiled
        # together with this one's and silently corrupt the result.
        src_dir = work / "src"
        classes_dir = work / "classes"
        for d in (src_dir, classes_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

        buggy_name = Path(entry["buggy_file"]).name
        buggy_original = bench_source_path(bench, entry["buggy_file"]).read_text(encoding="utf-8")
        pkg_match = _PACKAGE_RE.search(buggy_original)
        normalized_src = (
            normalize_package(candidate_src, pkg_match.group(0)) if pkg_match else candidate_src
        )
        (src_dir / buggy_name).write_text(normalized_src, encoding="utf-8")

        for rel in (*entry.get("support_files", []), *entry["test_files"]):
            text = bench_source_path(bench, rel).read_text(encoding="utf-8")
            # `Path(rel).name`: the copy always lands directly in `src_dir`, never at a
            # manifest-chosen subpath.
            (src_dir / Path(rel).name).write_text(text, encoding="utf-8")

        java_files = sorted(str(p) for p in src_dir.glob("*.java"))
        jars = _classpath_jars()
        sep = _classpath_sep()
        classpath = sep.join(jars)

        javac_cmd = [javac_bin, *_JAVAC_UTF8_PROPS, "-encoding", "UTF-8", "-d", str(classes_dir)]
        if classpath:
            javac_cmd += ["-cp", classpath]
        javac_cmd += java_files

        try:
            compile_rc, compile_out = _run_with_timeout(javac_cmd, cwd=work, timeout_s=timeout_s)
        except FileNotFoundError as e:
            return ExecResult(bug_id, False, False, "harness_error", str(e), bench)

        if compile_rc != 0:
            return classify_outcome(bug_id, bench, compile_rc, compile_out)

        run_cp = sep.join([str(classes_dir), *jars])
        java_cmd = [
            java_bin,
            _JVM_MAX_HEAP,
            *_JVM_UTF8_PROPS,
            "-cp",
            run_cp,
            "org.junit.runner.JUnitCore",
            entry["entry_test_class"],
        ]

        try:
            run_rc, run_out = _run_with_timeout(java_cmd, cwd=work, timeout_s=timeout_s)
        except FileNotFoundError as e:
            return ExecResult(bug_id, True, False, "harness_error", str(e), bench)

        return classify_outcome(bug_id, bench, compile_rc, compile_out, run_rc, run_out)
    except Exception as e:  # defensive: harness bugs shouldn't crash a batch run
        return ExecResult(bug_id, False, False, "harness_error", repr(e), bench)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
