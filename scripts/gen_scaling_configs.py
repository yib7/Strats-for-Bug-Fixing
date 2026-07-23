"""Generate the scaling-study finetune configs (data-scaling + pretrain-compute curves).

Emits the committed `configs/finetune_scale_*.yaml` and `configs/finetune_ptcompute_*.yaml`
files, then you commit them. Re-runnable and idempotent: it overwrites the generated configs
in place. GPU-free (pure text emission); the real runs happen later in the user's Colab batch
via `scripts/run_scaling.py`.

Two curves:

* **Data-scaling curve** -- how CodeBLEU / syntax_valid_rate scale with the number of
  finetune pairs. Arms **A** (pretrained: ``outputs/pretrain/final``) and **B** (no
  pretraining), ``train_n`` in {1000, 5000, 15000}, seeds {0, 1}, 10 finetune epochs,
  effective batch 64 (8 x 8). The **52K (full-data)** point is NOT generated here -- it
  reuses the existing ``finetune_A_ep10`` / ``finetune_B_seed{0,1}`` runs as the curve's
  top data point (documented in the Cycle-7 scaling summary).

* **Pretrain-compute curve** -- how downstream quality scales with pretraining epochs.
  Arm **A** only, pretrain epochs {1, 3}, seed 42, full data (``train_n`` unset), 10
  finetune epochs. ``pretrained_model_path`` points at the stable
  ``outputs/pretrain/epoch-{1,3}`` dirs written by ``pop.train.pretrain``'s milestone
  callback. The **ep10** point is NOT generated here -- pretrain-final == epoch-10, so it
  reuses the existing ``finetune_A_ep10`` run.

The two curves share every fixed hyperparameter with ``configs/finetune_A_ep10.yaml`` /
``configs/finetune_B_seed0.yaml`` (seq 512, effective batch 64 = 8 x 8, lr 5e-5, warmup 500,
T5 512/2048/64/8/6+6) -- the generator changes only seed / train_n / pretrained_model_path /
output_dir, so a scaling curve is a clean one-variable sweep against the existing arms.

Usage:
    python scripts/gen_scaling_configs.py           # (re)write the committed configs
    python scripts/gen_scaling_configs.py --check   # verify committed configs match (no write)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"

TOKENIZER_PATH = "outputs/tokenizer/tokenizer.model"
PRETRAIN_FINAL = "outputs/pretrain/final"

# train_n sizes with the short label used in filenames/output dirs.
DATA_SIZES: dict[str, int] = {"1k": 1000, "5k": 5000, "15k": 15000}
DATA_SEEDS: tuple[int, ...] = (0, 1)
PTCOMPUTE_EPOCHS: tuple[int, ...] = (1, 3)
PTCOMPUTE_SEED = 42

_MODEL_BLOCK = """model:
  d_model: 512
  d_ff: 2048
  d_kv: 64
  num_heads: 8
  num_layers: 6
  num_decoder_layers: 6
"""


@dataclass(frozen=True)
class ScalingConfigSpec:
    """A single generated scaling config: which variables differ from the base arm.

    ``stem`` is the config filename without extension and doubles as the output-dir /
    results-name base (``output_dir == outputs/<stem>``), matching the ``finetune_*`` /
    ``finetune_*_test.json`` naming the rest of the pipeline already uses.
    """

    stem: str
    kind: str  # "data" | "ptcompute"
    seed: int
    pretrained_model_path: str | None
    train_n: int | None
    comment: str

    @property
    def output_dir(self) -> str:
        return f"outputs/{self.stem}"


def iter_config_specs() -> list[ScalingConfigSpec]:
    """Return every generated scaling config, in a stable order.

    Single source of truth: both this generator and ``scripts/run_scaling.py``'s planner /
    preflight consume this list, so the config set can never drift between them.
    """
    specs: list[ScalingConfigSpec] = []
    for label, n in DATA_SIZES.items():
        for seed in DATA_SEEDS:
            specs.append(
                ScalingConfigSpec(
                    stem=f"finetune_scale_A_n{label}_seed{seed}",
                    kind="data",
                    seed=seed,
                    pretrained_model_path=PRETRAIN_FINAL,
                    train_n=n,
                    comment=(
                        f"Data-scaling curve, arm A (pretrained-then-finetuned): "
                        f"train_n={n}, seed={seed}.\n"
                        "# The 52K (full-data) point is NOT generated -- it reuses the existing "
                        "finetune_A_ep10 run."
                    ),
                )
            )
            specs.append(
                ScalingConfigSpec(
                    stem=f"finetune_scale_B_n{label}_seed{seed}",
                    kind="data",
                    seed=seed,
                    pretrained_model_path=None,
                    train_n=n,
                    comment=(
                        f"Data-scaling curve, arm B (no pretraining, randomly initialized): "
                        f"train_n={n}, seed={seed}.\n"
                        "# The 52K (full-data) point is NOT generated -- it reuses the existing "
                        "finetune_B_seed{0,1} runs."
                    ),
                )
            )
    for ep in PTCOMPUTE_EPOCHS:
        specs.append(
            ScalingConfigSpec(
                stem=f"finetune_ptcompute_ep{ep}_seed{PTCOMPUTE_SEED}",
                kind="ptcompute",
                seed=PTCOMPUTE_SEED,
                pretrained_model_path=f"outputs/pretrain/epoch-{ep}",
                train_n=None,
                comment=(
                    f"Pretrain-compute curve, arm A: pretrained for {ep} epoch(s), then 10 "
                    "finetune epochs, full data.\n"
                    f"# pretrained_model_path points at the stable outputs/pretrain/epoch-{ep} "
                    "dir written by\n"
                    "# pop.train.pretrain's milestone callback. The ep10 point is NOT generated "
                    "-- pretrain-final\n"
                    "# == epoch-10, so it reuses the existing finetune_A_ep10 run."
                ),
            )
        )
    return specs


def render_config(spec: ScalingConfigSpec) -> str:
    """Render one config file's YAML text (loadable by ``FinetuneConfig.from_yaml``)."""
    lines = [
        "# GENERATED by scripts/gen_scaling_configs.py -- do not edit by hand; edit the generator.",
        f"# {spec.comment}",
        "",
        f"seed: {spec.seed}",
        f"tokenizer_path: {TOKENIZER_PATH}",
    ]
    if spec.pretrained_model_path is not None:
        lines.append(f"pretrained_model_path: {spec.pretrained_model_path}")
    lines += [
        "",
        "train_split: train",
        "validation_split: validation",
        "max_seq_length: 512",
        "",
    ]
    if spec.train_n is not None:
        lines += [
            f"train_n: {spec.train_n}",
            "",
        ]
    lines += [
        "epochs: 10",
        "# effective batch 64 = 8 x 8 (see pretrain_10ep.yaml); 64 per-device OOMs 16 GB at seq 512",  # noqa: E501
        "batch_size: 8",
        "gradient_accumulation_steps: 8",
        "lr: 5.0e-5",
        "warmup_steps: 500",
        "",
        f"output_dir: {spec.output_dir}",
        "",
        _MODEL_BLOCK.rstrip("\n"),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed configs match what would be generated (write nothing); "
        "exit non-zero if any differ or are missing",
    )
    args = parser.parse_args(argv)

    specs = iter_config_specs()
    if args.check:
        drifted: list[str] = []
        for spec in specs:
            path = CONFIGS_DIR / f"{spec.stem}.yaml"
            expected = render_config(spec)
            if not path.is_file():
                drifted.append(f"{path.name}: missing")
            elif path.read_text(encoding="utf-8") != expected:
                drifted.append(f"{path.name}: out of date")
        if drifted:
            print("gen_scaling_configs --check FAILED:", file=sys.stderr)
            for line in drifted:
                print(f"  {line}", file=sys.stderr)
            print("Re-run `python scripts/gen_scaling_configs.py` and commit.", file=sys.stderr)
            return 1
        print(f"gen_scaling_configs --check OK: {len(specs)} configs match")
        return 0

    for spec in specs:
        path = CONFIGS_DIR / f"{spec.stem}.yaml"
        path.write_text(render_config(spec), encoding="utf-8")
        print(f"wrote {path.relative_to(REPO_ROOT)}")
    print(f"generated {len(specs)} scaling configs into {CONFIGS_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
