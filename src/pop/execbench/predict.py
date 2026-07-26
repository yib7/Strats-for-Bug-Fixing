"""Arm-agnostic adapter: turn a bug arm's generator into `pop execbench --predictions` jsonl.

The `pop execbench --predictions <jsonl>` mode (see `pop.cli._run_execbench`) consumes one
JSON object per line with exactly the keys ``{"bug_id", "prediction", "bench"}``: it reads
``record["bug_id"]``, ``record.get("bench") or <single --bench>``, and ``record["prediction"]``
and compiles ``prediction`` in place of that bug's buggy file. This module produces those
records for any arm.

:func:`build_prediction_records` is the single, arm-agnostic core: given a benchmark name, its
manifest entries, and an injectable ``generate_fn`` (buggy source text in -> candidate string
out), it reads each bug's buggy source off disk, batches every source through ``generate_fn`` in
one call, and returns the ``{bug_id, prediction, bench}`` records. Because ``generate_fn`` is
injected, tests pass a fake (no model, no download, no GPU) and the real per-arm generators are
wired in ``scripts/gen_execbench_predictions.py``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pop.execbench import harness

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def read_buggy_source(bench: str, entry: dict) -> str:
    """Read the buggy source file text for one manifest ``entry`` in ``bench``.

    Resolves through ``harness.bench_source_path`` -- the same resolver the harness uses
    (``harness.run_bug``) -- rather than composing the path by hand, so ``bench`` is
    validated as a bare directory name and ``buggy_file`` is confined to
    ``benchmarks/<bench>/``. Both are read out of a data file: a manifest a contributor
    or a dropped-in benchmark supplies, and a ``bench`` that ``pop execbench
    --predictions`` takes from an untrusted JSONL. This whole-file text is what gets fed
    to a model as input -- see the modeling note in
    ``scripts/gen_execbench_predictions.py``.
    """
    return harness.bench_source_path(bench, entry["buggy_file"]).read_text(encoding="utf-8")


def build_prediction_records(
    bench: str,
    entries: list[dict],
    generate_fn: Callable[[list[str]], list[str]],
) -> list[dict]:
    """Build ``{bug_id, prediction, bench}`` records for ``entries`` via ``generate_fn``.

    For each entry the buggy source file is read off disk; all sources are then batched
    through ``generate_fn`` in a single call (so an arm can size its own batches), and the
    i-th returned string becomes bug i's ``prediction``. ``generate_fn`` must return one
    candidate string per input source, in order.

    The returned records match exactly the shape ``pop execbench --predictions`` reads
    (``bug_id`` / ``prediction`` / ``bench``); the ``bench`` field lets a combined multi-bench
    jsonl be run with ``--bench all``.
    """
    buggy_sources = [read_buggy_source(bench, entry) for entry in entries]
    predictions = generate_fn(buggy_sources)
    if len(predictions) != len(entries):
        raise ValueError(
            f"generate_fn returned {len(predictions)} predictions for {len(entries)} bugs "
            f"({bench}); it must return exactly one candidate per input source"
        )
    return [
        {"bug_id": entry["bug_id"], "prediction": prediction, "bench": bench}
        for entry, prediction in zip(entries, predictions, strict=True)
    ]


def write_records(records: list[dict], out_path: str | Path) -> None:
    """Write ``records`` as a ``pop execbench --predictions`` jsonl (one JSON object per line).

    The canonical writer for this schema so the CLI script and any caller emit the exact
    format ``pop.cli._run_execbench`` parses.
    """
    from pathlib import Path

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
