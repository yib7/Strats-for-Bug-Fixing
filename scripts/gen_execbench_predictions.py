"""Generate `pop execbench --predictions` jsonl for one bug-fixing arm over the vendored bugs.

For a chosen arm (T5 / RAG-Qwen / LoRA-Qwen) this reads every benchmark bug's *buggy source
file*, feeds it through that arm's generator, and writes the combined
``{bug_id, prediction, bench}`` jsonl that ``pop execbench --predictions <jsonl>`` consumes.
Build-only: the actual model generation (and the JDK harness that scores it) runs later in the
user's one Colab GPU sitting -- this script just wires each arm's real generator into the
arm-agnostic :func:`pop.execbench.predict.build_prediction_records` core.

Modeling choice (a deliberate Track-2 finding, not a bug to fix here)
---------------------------------------------------------------------
The model input is the **whole buggy FILE text** (``benchmarks/<bench>/<buggy_file>``) and the
**raw model output** is the candidate written in its place (for RAG/LoRA, after
:func:`pop.rag.prompt.extract_fix` strips chat chatter; T5 output is used verbatim). But every
arm was trained on the CodeXGLUE refinement task -- single *methods*, not whole files with
package declarations, imports, and helper classes. Feeding a method-trained model a whole file
is expected to yield low compile rates; the harness's package-normalization
(:func:`pop.execbench.harness.normalize_package`) fixes the package line but nothing splices a
generated method back into the file. That whole-file-input vs. method-trained-model interplay
is itself the Track-2 result we want to measure -- so this adapter feeds the file as-is and does
NOT attempt method extraction / splicing. (A future arm could parse the buggy file, replace only
the target method body, and re-emit the file; that is out of scope here.)

Usage:
    # T5 arm (A/B): a finetuned T5 dir + its SentencePiece tokenizer
    python scripts/gen_execbench_predictions.py --arm t5 \
        --model outputs/finetune_A_ep10/best --tokenizer outputs/tokenizer/tokenizer.model \
        --bench all --out outputs/execbench/t5_A_preds.jsonl

    # RAG-Qwen arm (C): a rag config
    python scripts/gen_execbench_predictions.py --arm rag \
        --config configs/rag_bm25_k3.yaml --out outputs/execbench/rag_bm25_k3_preds.jsonl

    # LoRA-Qwen arm (D): a lora config (adapter read from <output_dir>/best|final)
    python scripts/gen_execbench_predictions.py --arm lora \
        --config configs/lora_qwen.yaml --out outputs/execbench/lora_qwen_preds.jsonl

Every heavy import (torch / transformers / peft) is deferred into the per-arm builder, so
``--help`` and importing this module stay light and need no GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pop.execbench import harness  # noqa: E402  (path inserted above)
from pop.execbench.predict import build_prediction_records, write_records  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

BENCH_CHOICES = ("quixbugs", "humaneval_java", "all")


def _benches(bench: str) -> list[str]:
    if bench == "all":
        return ["quixbugs", "humaneval_java"]
    return [bench]


def build_t5_generate_fn(args: argparse.Namespace) -> Callable[[list[str]], list[str]]:
    """T5 arm (A/B): buggy file text -> finetuned-T5 output, used verbatim as the candidate."""
    from pop.generate import generate_t5_predictions

    model_dir = Path(args.model)
    if not model_dir.is_dir():
        raise SystemExit(f"--arm t5: model directory not found: {model_dir}")
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.is_file():
        raise SystemExit(f"--arm t5: tokenizer not found: {tokenizer_path}")

    def generate_fn(buggy_sources: list[str]) -> list[str]:
        return generate_t5_predictions(
            model_dir,
            tokenizer_path,
            buggy_sources,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )

    return generate_fn


def build_rag_generate_fn(args: argparse.Namespace) -> Callable[[list[str]], list[str]]:
    """RAG-Qwen arm (C): retrieve exemplars, prompt Qwen, extract the fix.

    Mirrors ``pop.cli._run_rag``: the retriever KB is built once from the config's ``kb_split``
    (leakage guard) and reused across benches; each buggy file becomes a chat prompt, and
    :func:`pop.rag.prompt.extract_fix` turns each raw completion into the candidate.
    """
    from pop.config import RagConfig
    from pop.data.refinement import load_refinement_pairs
    from pop.rag.generate import build_generator
    from pop.rag.prompt import build_messages, extract_fix, render_prompt
    from pop.rag.retrievers import BM25Retriever, CodeBERTRetriever

    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"--arm rag: config file not found: {config_path}")
    cfg = RagConfig.from_yaml(config_path)

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

    # Build the model/engine once, on first call, then reuse it across benches (avoids
    # reloading the model for every batch of buggy sources).
    generator: dict[str, object] = {}

    def generate_fn(buggy_sources: list[str]) -> list[str]:
        if "gen" not in generator:
            generator["gen"] = build_generator(cfg.model_name, **cfg.gen_kwargs)
        prompts = []
        for buggy in buggy_sources:
            exemplars = retriever.retrieve(buggy, cfg.k) if retriever is not None else []
            prompts.append(render_prompt(tokenizer, build_messages(buggy, exemplars)))
        generated = generator["gen"](prompts)
        return [extract_fix(text) for text in generated]

    return generate_fn


def build_lora_generate_fn(args: argparse.Namespace) -> Callable[[list[str]], list[str]]:
    """LoRA-Qwen arm (D): prompt the LoRA-adapted causal LM, extract the fix.

    Mirrors ``pop.cli._run_lora_generate``: the adapter is read from ``<output_dir>/best`` (or
    ``final``), the shared :func:`pop.train.lora.build_lora_prompt` builds each prompt (so train
    and inference cannot drift), and :func:`pop.rag.prompt.extract_fix` extracts the candidate.
    """
    from pop.config import LoRAConfig
    from pop.rag.prompt import extract_fix
    from pop.train.lora import build_lora_prompt, generate_lora_fixes

    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"--arm lora: config file not found: {config_path}")
    cfg = LoRAConfig.from_yaml(config_path)

    output_dir = Path(cfg.output_dir)
    adapter_dir = output_dir / "best"
    if not adapter_dir.is_dir():
        final_dir = output_dir / "final"
        if final_dir.is_dir():
            adapter_dir = final_dir
        else:
            raise SystemExit(
                f"--arm lora: no trained adapter at {output_dir / 'best'} or {final_dir} "
                f"(run 'pop lora --config {config_path}' first)"
            )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

    def generate_fn(buggy_sources: list[str]) -> list[str]:
        prompts = [build_lora_prompt(tokenizer, buggy) for buggy in buggy_sources]
        generated = generate_lora_fixes(
            prompts,
            cfg.base_model,
            str(adapter_dir),
            max_new_tokens=args.max_new_tokens,
        )
        return [extract_fix(text) for text in generated]

    return generate_fn


_ARM_BUILDERS: dict[str, Callable[[argparse.Namespace], Callable[[list[str]], list[str]]]] = {
    "t5": build_t5_generate_fn,
    "rag": build_rag_generate_fn,
    "lora": build_lora_generate_fn,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gen_execbench_predictions",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--arm",
        required=True,
        choices=("t5", "rag", "lora"),
        help="which bug-fixing arm generates the candidates",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output jsonl path for the combined {bug_id, prediction, bench} records",
    )
    parser.add_argument(
        "--bench",
        choices=BENCH_CHOICES,
        default="all",
        help="which benchmark(s) to generate over (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the number of bugs per benchmark (default: no cap)",
    )

    t5 = parser.add_argument_group("t5 arm (--arm t5)")
    t5.add_argument("--model", help="path to a finetuned T5 model directory (e.g. .../best)")
    t5.add_argument("--tokenizer", help="path to the SentencePiece .model the T5 was trained with")
    t5.add_argument("--num-beams", type=int, default=1, help="T5 beam width (1 = greedy)")
    t5.add_argument("--batch-size", type=int, default=16, help="T5 generation batch size")

    rl = parser.add_argument_group("rag / lora arms (--arm rag|lora)")
    rl.add_argument("--config", help="path to the arm's YAML config (RagConfig / LoRAConfig)")

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="max tokens generated per input (t5 and lora arms; rag uses config gen_kwargs)",
    )
    return parser


def _validate_arm_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.arm == "t5":
        if not args.model or not args.tokenizer:
            parser.error("--arm t5 requires --model and --tokenizer")
    elif not args.config:  # rag / lora
        parser.error(f"--arm {args.arm} requires --config")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_arm_args(parser, args)

    generate_fn = _ARM_BUILDERS[args.arm](args)

    all_records: list[dict] = []
    for bench in _benches(args.bench):
        entries = harness.load_manifest(bench)
        if args.limit is not None:
            entries = entries[: args.limit]
        records = build_prediction_records(bench, entries, generate_fn)
        all_records.extend(records)
        print(f"{bench}: generated {len(records)} predictions", file=sys.stderr)

    write_records(all_records, args.out)
    print(f"Wrote {len(all_records)} prediction records to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
