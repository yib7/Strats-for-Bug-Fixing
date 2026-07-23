"""Offline tests for pop.execbench.predict (the arm-agnostic prediction adapter).

No JDK, no model, no network: the arm generator is an INJECTED fake ``generate_fn`` returning
canned predictions. The tests assert the emitted records/jsonl match the exact
``{bug_id, prediction, bench}`` schema that ``pop execbench --predictions`` reads, using the
same field-access logic ``pop.cli._run_execbench`` uses -- without ever invoking the JDK
compile+test path.
"""

from __future__ import annotations

import json

from pop.execbench.harness import load_manifest
from pop.execbench.predict import build_prediction_records, read_buggy_source, write_records

# A 2-bug slice of a real manifest (like the CI smoke); no JDK is invoked on these.
BENCH = "quixbugs"


def _fake_generate_fn(sources: list[str]) -> list[str]:
    """Canned generator: one prediction per input, echoing that it saw the buggy source."""
    return [f"// fixed candidate {i}\n{src}" for i, src in enumerate(sources)]


def test_records_have_expected_schema_and_bug_ids():
    entries = load_manifest(BENCH)[:2]
    records = build_prediction_records(BENCH, entries, _fake_generate_fn)

    assert len(records) == 2
    for record in records:
        assert set(record.keys()) == {"bug_id", "prediction", "bench"}
        assert record["bench"] == BENCH
        assert isinstance(record["prediction"], str)

    assert [r["bug_id"] for r in records] == [e["bug_id"] for e in entries]


def test_generate_fn_receives_whole_buggy_sources():
    entries = load_manifest(BENCH)[:2]
    captured: list[list[str]] = []

    def capturing_fn(sources: list[str]) -> list[str]:
        captured.append(sources)
        return ["fix" for _ in sources]

    build_prediction_records(BENCH, entries, capturing_fn)

    assert len(captured) == 1  # a single batched call, not one call per bug
    assert captured[0] == [read_buggy_source(BENCH, e) for e in entries]


def test_emitted_jsonl_is_parseable_by_execbench_predictions_reader(tmp_path):
    """The jsonl round-trips through the exact parsing logic in cli._run_execbench.

    We do NOT invoke run_bug/JDK -- only assert the fields that path reads are present, so a
    combined multi-bench file works with ``--bench all`` (single_bench is None).
    """
    entries = load_manifest(BENCH)[:2]
    records = build_prediction_records(BENCH, entries, _fake_generate_fn)

    out_path = tmp_path / "preds.jsonl"
    write_records(records, out_path)

    # Mirror _run_execbench's predictions reader verbatim (with --bench all => single_bench None).
    single_bench = None
    tasks: list[tuple[str, str, str]] = []
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        bench = record.get("bench") or single_bench
        assert bench is not None, "record must carry 'bench' so --bench all can dispatch it"
        tasks.append((record["bug_id"], bench, record["prediction"]))

    assert len(tasks) == 2
    assert [t[0] for t in tasks] == [e["bug_id"] for e in entries]
    assert all(t[1] == BENCH for t in tasks)
    assert all(isinstance(t[2], str) and t[2] for t in tasks)


def test_mismatched_generate_fn_length_raises():
    entries = load_manifest(BENCH)[:2]

    def short_fn(sources: list[str]) -> list[str]:
        return ["only one"]  # wrong count

    try:
        build_prediction_records(BENCH, entries, short_fn)
    except ValueError as exc:
        assert "one candidate per input" in str(exc)
    else:  # pragma: no cover - guard against silent acceptance
        raise AssertionError("expected ValueError on prediction-count mismatch")
