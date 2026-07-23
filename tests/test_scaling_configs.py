"""Tests for the generated scaling configs (data-scaling + pretrain-compute curves).

Every committed ``configs/finetune_scale_*.yaml`` / ``configs/finetune_ptcompute_*.yaml``
must parse via ``FinetuneConfig.from_yaml`` and carry the per-curve fields its arm requires.
The generator (``scripts/gen_scaling_configs.py``) is the single source of truth for which
configs exist and what they should contain, so these tests drive off its ``iter_config_specs``
and also assert the committed files are in sync (via ``--check``).

No GPU / network here -- pure YAML parse + field assertions.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pop.config import FinetuneConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import gen_scaling_configs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def test_generator_emits_14_configs():
    specs = gen_scaling_configs.iter_config_specs()
    # 12 data-scaling (2 arms x 3 sizes x 2 seeds) + 2 pretrain-compute (ep 1/3).
    assert len(specs) == 14
    stems = {s.stem for s in specs}
    assert len(stems) == 14  # all distinct


def test_every_generated_config_parses():
    for spec in gen_scaling_configs.iter_config_specs():
        path = CONFIGS_DIR / f"{spec.stem}.yaml"
        assert path.is_file(), f"missing committed config {path.name}"
        cfg = FinetuneConfig.from_yaml(path)  # must not raise
        assert Path(cfg.tokenizer_path) == Path("outputs/tokenizer/tokenizer.model")
        assert Path(cfg.output_dir) == Path(f"outputs/{spec.stem}")
        # Shared, preserved hyperparameters (one-variable sweep against the base arms).
        assert cfg.epochs == 10
        assert cfg.batch_size == 8
        assert cfg.gradient_accumulation_steps == 8


def test_committed_configs_match_generator():
    # `--check` re-renders every spec and byte-compares against the committed file.
    assert gen_scaling_configs.main(["--check"]) == 0


def test_data_scaling_arm_A_sets_pretrained_and_train_n():
    arm_a = [
        s
        for s in gen_scaling_configs.iter_config_specs()
        if s.kind == "data" and "_scale_A_" in s.stem
    ]
    assert len(arm_a) == 6  # 3 sizes x 2 seeds
    for spec in arm_a:
        cfg = FinetuneConfig.from_yaml(CONFIGS_DIR / f"{spec.stem}.yaml")
        assert cfg.pretrained_model_path == Path("outputs/pretrain/final")
        assert cfg.train_n in {1000, 5000, 15000}
        assert cfg.train_n == spec.train_n
        assert cfg.seed in {0, 1}


def test_data_scaling_arm_B_omits_pretrained_but_sets_train_n():
    arm_b = [
        s
        for s in gen_scaling_configs.iter_config_specs()
        if s.kind == "data" and "_scale_B_" in s.stem
    ]
    assert len(arm_b) == 6
    for spec in arm_b:
        cfg = FinetuneConfig.from_yaml(CONFIGS_DIR / f"{spec.stem}.yaml")
        assert cfg.pretrained_model_path is None
        assert cfg.train_n in {1000, 5000, 15000}
        assert cfg.train_n == spec.train_n


def test_ptcompute_configs_point_at_epoch_dirs_with_full_data():
    ptcompute = [s for s in gen_scaling_configs.iter_config_specs() if s.kind == "ptcompute"]
    assert {s.stem for s in ptcompute} == {
        "finetune_ptcompute_ep1_seed42",
        "finetune_ptcompute_ep3_seed42",
    }
    for spec in ptcompute:
        cfg = FinetuneConfig.from_yaml(CONFIGS_DIR / f"{spec.stem}.yaml")
        ep = spec.stem.split("_ep")[1].split("_")[0]
        assert cfg.pretrained_model_path == Path(f"outputs/pretrain/epoch-{ep}")
        assert cfg.train_n is None  # full data
        assert cfg.seed == 42


def test_no_ep10_ptcompute_config_generated():
    # The ep10 point reuses the existing finetune_A_ep10 run (pretrain-final == epoch-10);
    # generating a redundant config would double-run it.
    stems = {s.stem for s in gen_scaling_configs.iter_config_specs()}
    assert "finetune_ptcompute_ep10_seed42" not in stems
    assert not (CONFIGS_DIR / "finetune_ptcompute_ep10_seed42.yaml").is_file()
