"""Command-line entry point for pop (pretrain-or-prompt).

Subcommands dispatch to each phase's implementation in `pop.*`.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable

EXECBENCH_CHOICES = ("quixbugs", "humaneval_java", "all")

# Default `results/<name>.json` run names for `pop execbench`. Deliberately in the gitignored
# `*_local*` scratch namespace (see `pop.eval.metrics.is_scratch_run_name`): the committed
# results/execbench_validate_references.json is a published measurement docs/report.md cites,
# and an ad-hoc run must never land on top of it.
EXECBENCH_VALIDATE_RESULTS_NAME = "execbench_local_validate_references"
EXECBENCH_PREDICTIONS_RESULTS_NAME = "execbench_local_predictions"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pop",
        description="pretrain-or-prompt: pretrain+finetune a small T5 vs. prompt a "
        "larger LLM with RAG, for Java bug fixing.",
    )
    subparsers = parser.add_subparsers(dest="command")

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="end-to-end micro tokenizer->pretrain->finetune->eval dry run on fixtures (CPU, "
        "minutes; no network)",
    )
    smoke_parser.add_argument(
        "--config",
        default="configs/smoke.yaml",
        help="Path to a smoke YAML config (pop.config.SmokeConfig); default: configs/smoke.yaml",
    )

    pretrain_parser = subparsers.add_parser(
        "pretrain", help="pretrain a T5 model with span corruption"
    )
    pretrain_parser.add_argument(
        "--config", required=True, help="Path to a pretrain YAML config (pop.config.PretrainConfig)"
    )

    finetune_parser = subparsers.add_parser(
        "finetune", help="finetune a T5 model for Java code refinement"
    )
    finetune_parser.add_argument(
        "--config", required=True, help="Path to a finetune YAML config (pop.config.FinetuneConfig)"
    )

    tokenizer_parser = subparsers.add_parser(
        "tokenizer",
        help="train the SentencePiece tokenizer on the CodeSearchNet-Java corpus (network)",
    )
    tokenizer_parser.add_argument(
        "--out",
        default="outputs/tokenizer/tokenizer.model",
        help="Output .model path (default: outputs/tokenizer/tokenizer.model)",
    )
    tokenizer_parser.add_argument(
        "--vocab-size", type=int, default=16384, help="Target vocab size (default: 16384)"
    )
    tokenizer_parser.add_argument(
        "--corpus-samples",
        type=int,
        default=100000,
        help="Number of Java methods to train the tokenizer on (default: 100000)",
    )
    tokenizer_parser.add_argument("--seed", type=int, default=42, help="Corpus shuffle seed")

    generate_parser = subparsers.add_parser(
        "generate",
        help="generate T5 predictions on a CodeXGLUE refinement split (writes a pop-eval jsonl)",
    )
    generate_parser.add_argument(
        "--model", required=True, help="Path to a finetuned T5 model directory (e.g. .../best)"
    )
    generate_parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to the SentencePiece .model the model was trained with",
    )
    generate_parser.add_argument(
        "--split", default="test", help="CodeXGLUE refinement split (default: test)"
    )
    generate_parser.add_argument(
        "--out",
        default=None,
        help="Output jsonl path (default: <model>/predictions_<split>.jsonl)",
    )
    generate_parser.add_argument(
        "--num-beams", type=int, default=1, help="Beam width (1 = greedy; default: 1)"
    )
    generate_parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Max tokens to generate per input"
    )
    generate_parser.add_argument(
        "--batch-size", type=int, default=16, help="Generation batch size (default: 16)"
    )
    generate_parser.add_argument(
        "--limit", type=int, default=None, help="Cap the number of eval samples (default: no cap)"
    )

    eval_parser = subparsers.add_parser(
        "eval", help="evaluate predictions against references (EM, CodeBLEU, syntax validity)"
    )
    eval_parser.add_argument(
        "--predictions",
        required=True,
        help='Path to a jsonl file of {"prediction": ..., "reference": ...} lines',
    )
    eval_parser.add_argument(
        "--name",
        default=None,
        help="Run name for results/<name>.json (defaults to the predictions file stem)",
    )

    rag_parser = subparsers.add_parser(
        "rag", help="run retrieval-augmented prompting (BM25/CodeBERT + Qwen) for Java bug fixing"
    )
    rag_parser.add_argument(
        "--config", required=True, help="Path to a rag YAML config (pop.config.RagConfig)"
    )
    rag_parser.add_argument(
        "--allow-non-train-kb",
        action="store_true",
        help="Allow the retriever knowledge base to be built from a split other than "
        "'train' (leakage guard override; not for real evaluation runs)",
    )

    lora_parser = subparsers.add_parser(
        "lora",
        help="PEFT LoRA-finetune a causal LM (Qwen2.5-Coder) on the CodeXGLUE refinement pairs",
    )
    lora_parser.add_argument(
        "--config", required=True, help="Path to a lora YAML config (pop.config.LoRAConfig)"
    )

    lora_generate_parser = subparsers.add_parser(
        "lora-generate",
        help="generate LoRA-adapted causal-LM predictions on a CodeXGLUE refinement split "
        "(writes a pop-eval jsonl scored by 'pop eval')",
    )
    lora_generate_parser.add_argument(
        "--config", required=True, help="Path to a lora YAML config (pop.config.LoRAConfig)"
    )
    lora_generate_parser.add_argument(
        "--split", default="test", help="CodeXGLUE refinement split (default: test)"
    )
    lora_generate_parser.add_argument(
        "--out",
        default=None,
        help="Output jsonl path (default: <output_dir>/predictions_<split>.jsonl)",
    )
    lora_generate_parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Max tokens to generate per input"
    )
    lora_generate_parser.add_argument(
        "--limit", type=int, default=None, help="Cap the number of eval samples (default: no cap)"
    )

    execbench_parser = subparsers.add_parser(
        "execbench",
        help="run the Java execution harness (QuixBugs-Java / HumanEval-Java): "
        "compile+test reference or predicted fixes",
    )
    execbench_parser.add_argument(
        "--validate-references",
        action="store_true",
        help="run every benchmark bug's FIXED reference source through the harness "
        "(expected: 100%% pass); mutually exclusive with --predictions",
    )
    execbench_parser.add_argument(
        "--predictions",
        default=None,
        help='Path to a jsonl file of {"bug_id": ..., "prediction": ..., "bench": ...} lines '
        "(bench optional if --bench names a single benchmark); mutually exclusive with "
        "--validate-references",
    )
    execbench_parser.add_argument(
        "--bench",
        choices=EXECBENCH_CHOICES,
        default="all",
        help="Which benchmark(s) to run (default: all)",
    )
    execbench_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of bugs run per benchmark (default: no cap)",
    )
    execbench_parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of bugs to run concurrently (thread pool; subprocesses are independent)",
    )
    execbench_parser.add_argument(
        "--timeout-s",
        type=int,
        default=30,
        help="Per-stage (compile, run) subprocess timeout in seconds (default: 30)",
    )
    execbench_parser.add_argument(
        "--name",
        default=None,
        help="Run name for results/<name>.json (default: 'execbench_local_<mode>'). The "
        "default is deliberately distinct from the committed results/execbench_*.json "
        "files so an ad-hoc run can never clobber a published measurement.",
    )
    execbench_parser.add_argument(
        "--jdk",
        default=None,
        help="Path to a JDK home directory (must contain bin/javac, bin/java); "
        "default: resolve javac/java from PATH",
    )

    return parser


def _run_smoke(args: argparse.Namespace) -> int:
    from pathlib import Path

    from pop.config import SmokeConfig
    from pop.train.smoke import run_smoke

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"pop smoke: config file not found: {config_path}", file=sys.stderr)
        return 2

    cfg = SmokeConfig.from_yaml(config_path)
    try:
        run_smoke(cfg)
    except FileExistsError as e:
        print(f"pop smoke: {e}", file=sys.stderr)
        return 1
    return 0


def _run_pretrain(args: argparse.Namespace) -> int:
    from pathlib import Path

    from pop.config import PretrainConfig
    from pop.train.pretrain import run_pretrain

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"pop pretrain: config file not found: {config_path}", file=sys.stderr)
        return 2

    cfg = PretrainConfig.from_yaml(config_path)
    run_pretrain(cfg)
    return 0


def _run_finetune(args: argparse.Namespace) -> int:
    from pathlib import Path

    from pop.config import FinetuneConfig
    from pop.train.finetune import run_finetune

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"pop finetune: config file not found: {config_path}", file=sys.stderr)
        return 2

    cfg = FinetuneConfig.from_yaml(config_path)
    run_finetune(cfg)
    return 0


def _run_lora(args: argparse.Namespace) -> int:
    from pathlib import Path

    from pop.config import LoRAConfig
    from pop.train.lora import run_lora

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"pop lora: config file not found: {config_path}", file=sys.stderr)
        return 2

    cfg = LoRAConfig.from_yaml(config_path)
    run_lora(cfg)
    return 0


def _run_lora_generate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from pop.config import LoRAConfig
    from pop.data.refinement import load_refinement_pairs
    from pop.rag.generate import generate_with_resume
    from pop.rag.prompt import extract_fix
    from pop.train.lora import build_lora_generator, build_lora_prompt

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"pop lora-generate: config file not found: {config_path}", file=sys.stderr)
        return 2

    cfg = LoRAConfig.from_yaml(config_path)

    output_dir = Path(cfg.output_dir)
    # run_lora writes the adapter to <output_dir>/best when a validation set is
    # available, else <output_dir>/final; prefer best, fall back to final.
    adapter_dir = output_dir / "best"
    if not adapter_dir.is_dir():
        final_dir = output_dir / "final"
        if final_dir.is_dir():
            adapter_dir = final_dir
        else:
            print(
                f"pop lora-generate: no trained adapter found at {output_dir / 'best'} or "
                f"{final_dir} (run 'pop lora --config {config_path}' first)",
                file=sys.stderr,
            )
            return 2

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

    pairs = load_refinement_pairs(args.split)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    prompts = [build_lora_prompt(tokenizer, pair["buggy"]) for pair in pairs]
    references = [pair["fixed"] for pair in pairs]

    # Load the adapted model once; generate_with_resume feeds it prompt chunks and
    # checkpoints each chunk to <out_path>.partial, so a Colab disconnect mid-split
    # resumes instead of restarting (mirrors the RAG arm's _run_rag).
    generator = build_lora_generator(
        cfg.base_model, str(adapter_dir), max_new_tokens=args.max_new_tokens
    )

    def _generate(chunk: list[str]) -> list[str]:
        return [extract_fix(text) for text in generator(chunk)]

    out_path = Path(args.out) if args.out else output_dir / f"predictions_{args.split}.jsonl"
    n = generate_with_resume(prompts, references, out_path, _generate)

    print(f"Wrote {n} predictions to {out_path}", file=sys.stderr)
    return 0


def _run_tokenizer(args: argparse.Namespace) -> int:
    from pathlib import Path

    from pop.data.corpus import load_pretraining_corpus
    from pop.tokenizer.train import train_tokenizer

    corpus = load_pretraining_corpus(args.corpus_samples, seed=args.seed)
    out = train_tokenizer(corpus, Path(args.out), vocab_size=args.vocab_size)
    print(
        f"Trained tokenizer (vocab~{args.vocab_size}) on {len(corpus)} methods -> {out}",
        file=sys.stderr,
    )
    return 0


def _run_generate(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from pop.data.refinement import load_refinement_pairs
    from pop.generate import generate_t5_predictions

    model_dir = Path(args.model)
    if not model_dir.is_dir():
        print(f"pop generate: model directory not found: {model_dir}", file=sys.stderr)
        return 2
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.is_file():
        print(f"pop generate: tokenizer not found: {tokenizer_path}", file=sys.stderr)
        return 2

    pairs = load_refinement_pairs(args.split)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    buggy = [pair["buggy"] for pair in pairs]

    predictions = generate_t5_predictions(
        model_dir,
        tokenizer_path,
        buggy,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    out_path = Path(args.out) if args.out else model_dir / f"predictions_{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for pair, prediction in zip(pairs, predictions, strict=True):
            f.write(json.dumps({"prediction": prediction, "reference": pair["fixed"]}) + "\n")

    print(f"Wrote {len(predictions)} predictions to {out_path}", file=sys.stderr)
    return 0


def _read_jsonl(path, required: tuple[str, ...]) -> list[dict]:
    """Parse a JSONL file, reporting `file:line` for a bad record.

    A hand-edited or half-written predictions file used to surface as a bare
    `json.JSONDecodeError` / `KeyError` traceback. `required` names the keys every record
    must carry, so a missing field is reported against the line that lacks it.
    """
    import json

    records: list[dict] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: not valid JSON ({e.msg})") from e
        if not isinstance(record, dict):
            raise ValueError(
                f"{path}:{lineno}: expected a JSON object, got {type(record).__name__}"
            )
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"{path}:{lineno}: missing key(s) {', '.join(missing)}")
        records.append(record)
    return records


def _write_results(command: str, name: str, metrics: dict, config: dict):
    """`write_results`, turning a refusal-to-clobber into a printed message and `None`.

    `results/` holds committed, published measurements; `write_results` refuses to
    replace one. The CLI reports that as a one-line actionable error rather than a
    `FileExistsError` traceback (see `pop.eval.metrics.write_results`).
    """
    from pop.eval.metrics import write_results

    try:
        return write_results(name, metrics, config)
    except FileExistsError as e:
        print(f"pop {command}: {e}", file=sys.stderr)
        return None


def _run_eval(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from pop.eval.metrics import evaluate_predictions

    predictions_path = Path(args.predictions)
    if not predictions_path.is_file():
        print(f"pop eval: predictions file not found: {predictions_path}", file=sys.stderr)
        return 2

    records = _read_jsonl(predictions_path, required=("prediction", "reference"))
    preds = [record["prediction"] for record in records]
    refs = [record["reference"] for record in records]

    metrics = evaluate_predictions(preds, refs)
    name = args.name or predictions_path.stem
    results_path = _write_results(
        "eval", name, metrics, config={"predictions": str(predictions_path)}
    )
    if results_path is None:
        return 1

    print(json.dumps(metrics, indent=2))
    print(f"Wrote results to {results_path}", file=sys.stderr)
    return 0


def _run_rag(args: argparse.Namespace) -> int:
    from pathlib import Path

    from pop.config import RagConfig
    from pop.data.refinement import load_refinement_pairs
    from pop.rag.generate import build_generator, generate_with_resume
    from pop.rag.prompt import build_messages, extract_fix, render_prompt
    from pop.rag.retrievers import BM25Retriever, CodeBERTRetriever

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"pop rag: config file not found: {config_path}", file=sys.stderr)
        return 2

    cfg = RagConfig.from_yaml(config_path)

    if cfg.kb_split != "train" and not args.allow_non_train_kb:
        print(
            f"pop rag: refusing to build the retriever KB from split {cfg.kb_split!r} "
            "(leakage guard: KB must be 'train'); pass --allow-non-train-kb to override",
            file=sys.stderr,
        )
        return 1

    eval_pairs = load_refinement_pairs(cfg.split)

    retriever: BM25Retriever | CodeBERTRetriever | None = None
    if cfg.k > 0:
        kb_pairs = load_refinement_pairs(cfg.kb_split)
        retriever = (
            BM25Retriever()
            if cfg.retriever == "bm25"
            else CodeBERTRetriever(model_name=cfg.codebert_model_name)
        )
        retriever.index(kb_pairs)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # Build every prompt (retrieval + chat-template render). On a resumed run the
    # already-finished prompts are rebuilt but not re-generated -- generate_with_resume
    # skips them; retrieval is cheap next to generation, and the KB index is built once.
    prompts = []
    references = []
    for pair in eval_pairs:
        exemplars = retriever.retrieve(pair["buggy"], cfg.k) if retriever is not None else []
        messages = build_messages(pair["buggy"], exemplars)
        prompts.append(render_prompt(tokenizer, messages))
        references.append(pair["fixed"])

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / "predictions.jsonl"

    # Load the model/engine ONCE (vLLM if usable, else batched transformers), then feed it
    # prompt chunks -- so generate_with_resume's chunking checkpoints without reloading the model.
    generator = build_generator(cfg.model_name, **cfg.gen_kwargs)

    def _generate(chunk: list[str]) -> list[str]:
        return [extract_fix(text) for text in generator(chunk)]

    # Checkpoints to predictions.jsonl.partial and atomically finalizes to predictions.jsonl,
    # so a mid-config Colab disconnect resumes instead of restarting the whole config.
    n = generate_with_resume(prompts, references, out_path, _generate)

    print(f"Wrote {n} predictions to {out_path}", file=sys.stderr)
    return 0


def _execbench_benches(args: argparse.Namespace) -> list[str]:
    if args.bench == "all":
        return ["quixbugs", "humaneval_java"]
    return [args.bench]


def _limit_per_bench(tasks: list[tuple[str, str, str]], limit: int) -> list[tuple[str, str, str]]:
    """Keep the first `limit` tasks *per benchmark*, as `--limit`'s help text promises.

    Regression: the predictions path sliced the merged task list, so
    `--bench all --predictions X --limit 10` yielded 10 bugs *total* -- all from whichever
    benchmark happened to come first in the file -- while `--validate-references` correctly
    applied the cap inside its per-benchmark loop.
    """
    kept: list[tuple[str, str, str]] = []
    counts: dict[str, int] = {}
    for task in tasks:
        bench = task[1]
        if counts.get(bench, 0) >= limit:
            continue
        counts[bench] = counts.get(bench, 0) + 1
        kept.append(task)
    return kept


def _run_execbench(args: argparse.Namespace) -> int:
    import json
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from pop.execbench import harness as harness_mod
    from pop.execbench.score import aggregate

    if bool(args.validate_references) == bool(args.predictions):
        print(
            "pop execbench: exactly one of --validate-references or --predictions is required",
            file=sys.stderr,
        )
        return 2

    benches = _execbench_benches(args)
    jobs = max(1, args.jobs)
    jdk_info = harness_mod.jdk_identity(args.jdk)

    # Fail fast on a missing/unusable JDK. Without this, every bug fails identically with a
    # bare "harness_error" -- 201 opaque lines whose real cause (a FileNotFoundError for
    # java/javac) is already sitting in `jdk_info` and only visible if the user opens the
    # results JSON. Running first also means no partial results file is written.
    if jdk_info.get("version") is None:
        print(
            f"pop execbench: no usable JDK found "
            f"({jdk_info.get('error', 'java -version produced no output')}).\n"
            f"  Install a JDK 17+ and put java/javac on PATH, or pass --jdk <jdk-home>.\n"
            f"  Tried: {jdk_info.get('java') or jdk_info.get('jdk') or 'javac/java via PATH'}",
            file=sys.stderr,
        )
        return 2

    if args.validate_references:
        tasks: list[tuple[str, str, str]] = []  # (bug_id, bench, candidate_src)
        for bench in benches:
            entries = harness_mod.load_manifest(bench)
            if args.limit is not None:
                entries = entries[: args.limit]
            for entry in entries:
                fixed_path = harness_mod.bench_source_path(bench, entry["fixed_file"])
                candidate_src = fixed_path.read_text(encoding="utf-8")
                tasks.append((entry["bug_id"], bench, candidate_src))

        def _run(task: tuple[str, str, str]):
            bug_id, bench, candidate_src = task
            return harness_mod.run_bug(
                bug_id, candidate_src, bench, jdk=args.jdk, timeout_s=args.timeout_s
            )

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_run, tasks))

        metrics = aggregate(results)
        name = args.name or EXECBENCH_VALIDATE_RESULTS_NAME
        results_path = _write_results(
            "execbench",
            name,
            metrics,
            config={
                "mode": "validate-references",
                "bench": args.bench,
                "limit": args.limit,
                "jdk": jdk_info,
            },
        )
        if results_path is None:
            return 1

        print(json.dumps(metrics, indent=2))
        print(f"Wrote results to {results_path}", file=sys.stderr)

        failures = [r for r in results if not r.passed]
        if failures:
            print(
                f"pop execbench: {len(failures)}/{len(results)} reference bug(s) did not pass:",
                file=sys.stderr,
            )
            for r in failures:
                print(f"  {r.bench}/{r.bug_id}: {r.error_kind}", file=sys.stderr)
            return 1
        return 0

    predictions_path = Path(args.predictions)
    if not predictions_path.is_file():
        print(f"pop execbench: predictions file not found: {predictions_path}", file=sys.stderr)
        return 2

    single_bench = args.bench if args.bench != "all" else None
    records = _read_jsonl(predictions_path, required=("bug_id", "prediction"))
    # `--bench` is constrained by argparse `choices`; a per-record `bench` is not, and it is
    # read from an untrusted data file straight into a filesystem path (the manifest to load,
    # the Java sources to compile, the temp-dir prefix). Validate it against the same set.
    known_benches = tuple(b for b in EXECBENCH_CHOICES if b != "all")
    tasks = []
    for index, record in enumerate(records, 1):
        bench = record.get("bench") or single_bench
        if bench is None:
            print(
                "pop execbench: predictions record missing 'bench' and --bench is 'all'; "
                "specify --bench quixbugs|humaneval_java or add a 'bench' field per record",
                file=sys.stderr,
            )
            return 2
        if bench not in known_benches:
            print(
                f"pop execbench: {predictions_path}: record {index}: unknown benchmark "
                f"{bench!r} (expected one of {', '.join(known_benches)})",
                file=sys.stderr,
            )
            return 2
        tasks.append((record["bug_id"], bench, record["prediction"]))

    if args.limit is not None:
        tasks = _limit_per_bench(tasks, args.limit)

    def _run_pred(task: tuple[str, str, str]):
        bug_id, bench, candidate_src = task
        return harness_mod.run_bug(
            bug_id, candidate_src, bench, jdk=args.jdk, timeout_s=args.timeout_s
        )

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(_run_pred, tasks))

    metrics = aggregate(results)
    name = args.name or EXECBENCH_PREDICTIONS_RESULTS_NAME
    results_path = _write_results(
        "execbench",
        name,
        metrics,
        config={
            "mode": "predictions",
            "predictions": str(predictions_path),
            "jdk": jdk_info,
        },
    )
    if results_path is None:
        return 1

    print(json.dumps(metrics, indent=2))
    print(f"Wrote results to {results_path}", file=sys.stderr)
    return 0


def _describe_error(e: Exception) -> str:
    """Render an input error as one readable line (no stack, no pydantic internals)."""
    if e.__class__.__name__ == "ValidationError" and hasattr(e, "errors"):
        problems = []
        for err in e.errors():
            loc = ".".join(str(part) for part in err.get("loc", ())) or "<top level>"
            problems.append(f"{loc}: {err.get('msg', 'invalid value')}")
        return "invalid config -- " + "; ".join(problems)
    if isinstance(e, KeyError):
        return f"missing key {e.args[0]!r}" if e.args else "missing key"
    return str(e) or e.__class__.__name__


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    try:
        return _dispatch(args, parser)
    except (ValueError, OSError, KeyError) as e:
        # Malformed YAML/JSONL, a config that fails validation, an unreadable file: the user
        # can fix all of these, so they get one actionable line rather than a traceback.
        # pydantic's ValidationError subclasses ValueError, so it lands here too.
        # POP_TRACEBACK=1 restores the raw exception for debugging.
        if os.environ.get("POP_TRACEBACK") == "1":
            raise
        print(f"pop {args.command}: {_describe_error(e)}", file=sys.stderr)
        return 2


# The CLI surface, in `pop --help` order. `test_cli.py` pins this against the parser's own
# registered subcommands, so a subcommand can never be given a parser without a handler (or a
# handler without a parser).
COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "smoke": _run_smoke,
    "tokenizer": _run_tokenizer,
    "pretrain": _run_pretrain,
    "finetune": _run_finetune,
    "generate": _run_generate,
    "eval": _run_eval,
    "rag": _run_rag,
    "lora": _run_lora,
    "lora-generate": _run_lora_generate,
    "execbench": _run_execbench,
}


def _dispatch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    handler = COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
