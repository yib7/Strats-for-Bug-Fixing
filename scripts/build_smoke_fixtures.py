"""Build the `pop smoke` fixture files from the vendored benchmarks (no network).

Sources: `benchmarks/quixbugs/` (40 bug programs) and `benchmarks/humaneval_java/` (161 bugs),
both already committed for `pop execbench`. This script extracts:

- ``tests/fixtures/smoke_corpus.txt`` -- ~200 small Java methods for the micro-pretrain
  tokenizer + span-corruption corpus, pulled from the *fixed/correct* and *buggy* sources of
  both benchmarks (any syntactically-valid Java method is fine for span-corruption
  pretraining; it doesn't need to be "correct").
- ``tests/fixtures/smoke_finetune_pairs.jsonl`` -- 50 buggy/fixed method pairs for the
  micro-finetune stage.
- ``tests/fixtures/smoke_val_pairs.jsonl`` -- 10 buggy/fixed pairs used as the finetune
  validation split.
- ``tests/fixtures/smoke_eval_pairs.jsonl`` -- 20 buggy/fixed pairs held out for the final
  smoke-pipeline generation + metrics stage.

Extraction heuristic (documented, not a real Java parser): each benchmark's ``manifest.json``
entry points at a ``buggy_file``/``fixed_file`` pair. Both files are simple `package ...; class
NAME { <members> }` files. We locate the outer class body via a regex + brace-depth scan, then
split that body into top-level "members" (fields end at a depth-0 `;`; methods/blocks end at the
depth-0 `}` that closes them). Members that look like methods (contain `(` before their opening
`{`) are matched by name across the buggy/fixed pair; a pair is kept only when the matched
method's text actually differs (i.e. it's the buggy method, not an unrelated field/import diff).
This works well because QuixBugs/HumanEval-Java bug classes are (near-)single-method by
construction; a handful of entries with no single differing method are silently skipped.

Corpus methods are collected the same way (all method-shaped members from every fixed/buggy
source file in both benchmarks), deduplicated, and deterministically shuffled/truncated to 200.

Rerunnable:
    uv run python scripts/build_smoke_fixtures.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

SEED = 42
CORPUS_SIZE = 200
FINETUNE_SIZE = 50
VAL_SIZE = 10
EVAL_SIZE = 20

CORPUS_SEPARATOR = "\n// ===SMOKE_METHOD_SEP===\n"

_CLASS_RE = re.compile(r"\bclass\s+\w+[^{]*\{")


def _class_body(text: str) -> str | None:
    """Return the text inside the first top-level `class ... { ... }` block."""
    match = _CLASS_RE.search(text)
    if match is None:
        return None
    start = match.end()
    depth = 1
    i = start
    n = len(text)
    while i < n and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1]


def _top_level_members(class_body: str) -> list[str]:
    """Split a class body into top-level members (fields end at `;`, blocks at matching `}`)."""
    members: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(class_body):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                members.append(class_body[start : i + 1])
                start = i + 1
        elif ch == ";" and depth == 0:
            members.append(class_body[start : i + 1])
            start = i + 1
    return [m.strip() for m in members if m.strip()]


def _is_method_member(member: str) -> bool:
    """A "method-shaped" member: contains `(...)  {` before its block body opens."""
    if not member.endswith("}"):
        return False
    if member.lstrip().startswith(("class ", "interface ", "enum ", "static {")):
        return False
    open_brace = member.find("{")
    if open_brace == -1:
        return False
    header = member[:open_brace]
    return "(" in header and ")" in header


def _member_name(member: str) -> str | None:
    idx = member.find("(")
    if idx == -1:
        return None
    before = member[:idx].strip()
    tokens = before.replace("<", " ").replace(">", " ").split()
    return tokens[-1] if tokens else None


def extract_methods(source: str) -> dict[str, str]:
    """Return {method_name: method_text} for every method-shaped member in `source`."""
    body = _class_body(source)
    if body is None:
        return {}
    methods: dict[str, str] = {}
    for member in _top_level_members(body):
        if not _is_method_member(member):
            continue
        name = _member_name(member)
        if name is None:
            continue
        methods[name] = member
    return methods


def _normalize(text: str) -> str:
    return " ".join(text.split())


def load_manifest(bench: str) -> list[dict]:
    return json.loads((BENCHMARKS_DIR / bench / "manifest.json").read_text(encoding="utf-8"))


def collect_corpus_methods() -> list[str]:
    methods: list[str] = []
    seen: set[str] = set()

    def _add_file(path: Path) -> None:
        if not path.is_file():
            return
        source = path.read_text(encoding="utf-8")
        for text in extract_methods(source).values():
            key = _normalize(text)
            if not key or key in seen:
                continue
            seen.add(key)
            methods.append(text.strip())

    for bench in ("quixbugs", "humaneval_java"):
        for entry in load_manifest(bench):
            _add_file(BENCHMARKS_DIR / bench / entry["buggy_file"])
            _add_file(BENCHMARKS_DIR / bench / entry["fixed_file"])

    return methods


def collect_bug_pairs() -> list[dict]:
    pairs: list[dict] = []
    seen: set[str] = set()

    for bench in ("quixbugs", "humaneval_java"):
        for entry in load_manifest(bench):
            buggy_path = BENCHMARKS_DIR / bench / entry["buggy_file"]
            fixed_path = BENCHMARKS_DIR / bench / entry["fixed_file"]
            if not buggy_path.is_file() or not fixed_path.is_file():
                continue
            buggy_methods = extract_methods(buggy_path.read_text(encoding="utf-8"))
            fixed_methods = extract_methods(fixed_path.read_text(encoding="utf-8"))
            for name, buggy_text in buggy_methods.items():
                fixed_text = fixed_methods.get(name)
                if fixed_text is None:
                    continue
                if _normalize(buggy_text) == _normalize(fixed_text):
                    continue  # unchanged method (bug lives elsewhere in the file); skip
                key = _normalize(buggy_text) + "||" + _normalize(fixed_text)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({"buggy": buggy_text.strip(), "fixed": fixed_text.strip()})

    return pairs


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    corpus_methods = collect_corpus_methods()
    rng = random.Random(SEED)
    rng.shuffle(corpus_methods)
    corpus_methods = corpus_methods[:CORPUS_SIZE]
    corpus_path = FIXTURES_DIR / "smoke_corpus.txt"
    corpus_path.write_text(CORPUS_SEPARATOR.join(corpus_methods), encoding="utf-8")
    print(f"Wrote {len(corpus_methods)} corpus methods to {corpus_path}")

    pairs = collect_bug_pairs()
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    needed = FINETUNE_SIZE + VAL_SIZE + EVAL_SIZE
    if len(pairs) < needed:
        raise SystemExit(f"only found {len(pairs)} bug pairs, need at least {needed}")

    finetune_pairs = pairs[:FINETUNE_SIZE]
    val_pairs = pairs[FINETUNE_SIZE : FINETUNE_SIZE + VAL_SIZE]
    eval_pairs = pairs[FINETUNE_SIZE + VAL_SIZE : FINETUNE_SIZE + VAL_SIZE + EVAL_SIZE]

    for name, subset in (
        ("smoke_finetune_pairs.jsonl", finetune_pairs),
        ("smoke_val_pairs.jsonl", val_pairs),
        ("smoke_eval_pairs.jsonl", eval_pairs),
    ):
        out_path = FIXTURES_DIR / name
        with out_path.open("w", encoding="utf-8") as f:
            for pair in subset:
                f.write(json.dumps(pair) + "\n")
        print(f"Wrote {len(subset)} pairs to {out_path}")


if __name__ == "__main__":
    main()
