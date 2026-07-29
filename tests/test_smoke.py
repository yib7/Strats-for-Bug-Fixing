"""Tests for `pop smoke` config/fixtures wiring, the GPU configs, and notebooks.

The smoke *pipeline itself* (tokenizer -> pretrain -> finetune -> generate -> score) is
validated by actually running `uv run pop smoke` once end-to-end (with `results/smoke.json`
committed), not by a unit test that re-runs the whole thing -- that would duplicate real Trainer
runs across every `pytest`
invocation. What's tested here is the static wiring: configs parse, fixtures exist and are
correctly shaped/sized, and the notebooks are valid.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
CONFIGS_DIR = REPO_ROOT / "configs"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

# Every notebook in notebooks/, discovered rather than listed: a hand-maintained list silently
# stopped covering colab_phase2.ipynb and colab_execbench.ipynb when those were added.
NOTEBOOK_PATHS = sorted(NOTEBOOKS_DIR.glob("colab_*.ipynb"))
NOTEBOOK_NAMES = [path.name for path in NOTEBOOK_PATHS]

# The notebooks that offer a Weights & Biases cell. It must be an interactive `wandb.login()` so a
# key is never pasted into a committed cell; colab_phase2 and colab_execbench deliberately have no
# W&B cell at all (Run all must never stall waiting for input).
WANDB_NOTEBOOK_NAMES = ["colab_lora.ipynb", "colab_rag.ipynb", "colab_scaling.ipynb"]


# ---- SmokeConfig ----


def test_smoke_config_defaults():
    from pop.config import SmokeConfig

    cfg = SmokeConfig()
    assert cfg.vocab_size == 512
    assert cfg.model.num_layers == 2
    assert cfg.pretrain_epochs == 1
    assert cfg.finetune_epochs == 1


def test_smoke_config_parses_committed_yaml():
    from pop.config import SmokeConfig

    cfg = SmokeConfig.from_yaml(CONFIGS_DIR / "smoke.yaml")
    assert cfg.corpus_file == Path("tests/fixtures/smoke_corpus.txt")
    assert cfg.finetune_pairs_file == Path("tests/fixtures/smoke_finetune_pairs.jsonl")
    assert cfg.eval_pairs_file == Path("tests/fixtures/smoke_eval_pairs.jsonl")
    assert cfg.vocab_size == 512


# ---- fixture files: exist, correctly shaped, small ----


def test_smoke_corpus_fixture_has_about_200_methods():
    from pop.data.corpus import load_corpus_file

    records = load_corpus_file(FIXTURES_DIR / "smoke_corpus.txt")
    assert 150 <= len(records) <= 200
    assert all(record["code"].strip() for record in records)


def test_smoke_pairs_fixtures_correct_sizes_and_shape():
    from pop.data.refinement import load_pairs_file

    finetune_pairs = load_pairs_file(FIXTURES_DIR / "smoke_finetune_pairs.jsonl")
    val_pairs = load_pairs_file(FIXTURES_DIR / "smoke_val_pairs.jsonl")
    eval_pairs = load_pairs_file(FIXTURES_DIR / "smoke_eval_pairs.jsonl")

    assert len(finetune_pairs) == 50
    assert len(val_pairs) == 10
    assert len(eval_pairs) == 20

    for pairs in (finetune_pairs, val_pairs, eval_pairs):
        for pair in pairs:
            assert set(pair.keys()) == {"buggy", "fixed"}
            assert pair["buggy"].strip()
            assert pair["fixed"].strip()
            assert pair["buggy"] != pair["fixed"]  # each pair is an actual bug/fix diff


def test_smoke_fixtures_are_disjoint():
    from pop.data.refinement import load_pairs_file

    finetune_pairs = load_pairs_file(FIXTURES_DIR / "smoke_finetune_pairs.jsonl")
    val_pairs = load_pairs_file(FIXTURES_DIR / "smoke_val_pairs.jsonl")
    eval_pairs = load_pairs_file(FIXTURES_DIR / "smoke_eval_pairs.jsonl")

    def _keys(pairs):
        return {(p["buggy"], p["fixed"]) for p in pairs}

    finetune_keys, val_keys, eval_keys = _keys(finetune_pairs), _keys(val_pairs), _keys(eval_pairs)
    assert not (finetune_keys & val_keys)
    assert not (finetune_keys & eval_keys)
    assert not (val_keys & eval_keys)


def test_smoke_fixture_files_total_under_300kb():
    fixture_files = [
        FIXTURES_DIR / "smoke_corpus.txt",
        FIXTURES_DIR / "smoke_finetune_pairs.jsonl",
        FIXTURES_DIR / "smoke_val_pairs.jsonl",
        FIXTURES_DIR / "smoke_eval_pairs.jsonl",
    ]
    total_bytes = sum(f.stat().st_size for f in fixture_files)
    assert total_bytes < 300_000, f"fixtures total {total_bytes} bytes, expected < 300KB"


# ---- data-loading helpers (load_corpus_file / load_pairs_file) ----


def test_load_corpus_file_returns_code_records(tmp_path):
    from pop.data.corpus import CORPUS_FILE_SEPARATOR, load_corpus_file

    path = tmp_path / "corpus.txt"
    path.write_text(
        CORPUS_FILE_SEPARATOR.join(["public void a() {}", "public void b() {}"]),
        encoding="utf-8",
    )
    records = load_corpus_file(path)
    assert records == [{"code": "public void a() {}"}, {"code": "public void b() {}"}]


def test_load_pairs_file_round_trips(tmp_path):
    from pop.data.refinement import load_pairs_file

    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps({"buggy": "int a = 1;", "fixed": "int a = 2;"}) + "\n",
        encoding="utf-8",
    )
    pairs = load_pairs_file(path)
    assert pairs == [{"buggy": "int a = 1;", "fixed": "int a = 2;"}]


# ---- next-cycle GPU configs parse ----


def test_pretrain_10ep_config_parses():
    from pop.config import PretrainConfig

    cfg = PretrainConfig.from_yaml(CONFIGS_DIR / "pretrain_10ep.yaml")
    assert cfg.epochs == 10
    assert cfg.model.d_model == 512
    assert cfg.save_epochs == [1, 3, 10]


def test_finetune_a_configs_parse_with_pretrained_model_path():
    from pop.config import FinetuneConfig

    for n in (1, 3, 10):
        cfg = FinetuneConfig.from_yaml(CONFIGS_DIR / f"finetune_A_ep{n}.yaml")
        assert cfg.epochs == n
        assert cfg.pretrained_model_path is not None


def test_finetune_b_configs_parse_without_pretrained_model_path():
    """Pipeline B is "fine-tuned, no pre-training" -- these configs
    must NOT set pretrained_model_path, or `pop finetune` would silently run Pipeline A instead.
    """
    from pop.config import FinetuneConfig

    for seed in (0, 1, 2):
        cfg = FinetuneConfig.from_yaml(CONFIGS_DIR / f"finetune_B_seed{seed}.yaml")
        assert cfg.seed == seed
        assert cfg.pretrained_model_path is None


def test_rag_configs_parse():
    from pop.config import RagConfig

    for retriever in ("bm25", "codebert"):
        for k in (0, 1, 3, 5):
            cfg = RagConfig.from_yaml(CONFIGS_DIR / f"rag_{retriever}_k{k}.yaml")
            assert cfg.retriever == retriever
            assert cfg.k == k


def test_lora_qwen_config_file_exists_and_is_valid_yaml():
    import yaml

    # No pop.config model for LoRA yet (documented next-cycle scope); just verify it's present
    # and parses as YAML with the fields docs/gpu-reproduction.md and the file's own header
    # describe.
    path = CONFIGS_DIR / "lora_qwen.yaml"
    assert path.is_file()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["base_model"] == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert "lora_r" in data


# ---- notebooks: nbformat-valid, no outputs, no secrets ----


# Deliberately excludes bare "api_key"/"WANDB_API_KEY" -- the wandb-login cell's markdown
# legitimately references the *env var name* (never a value) to explain `pop`'s auto-disable
# behavior; see pop.train.pretrain/finetune. These patterns instead target an actual assigned
# secret value.
_SECRET_PATTERNS = ("api_key=", "api_key = ", "api_key: ", "sk-", "AKIA", "-----BEGIN")


def test_the_expected_notebooks_are_present():
    assert NOTEBOOK_NAMES == [
        "colab_execbench.ipynb",
        "colab_lora.ipynb",
        "colab_phase2.ipynb",
        "colab_rag.ipynb",
        "colab_scaling.ipynb",
    ]


def test_notebooks_are_nbformat_valid():
    import nbformat

    for name in NOTEBOOK_NAMES:
        nb = nbformat.read(NOTEBOOKS_DIR / name, as_version=4)
        nbformat.validate(nb)


def test_notebooks_have_no_outputs():
    import nbformat

    for name in NOTEBOOK_NAMES:
        nb = nbformat.read(NOTEBOOKS_DIR / name, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code":
                assert cell.get("outputs", []) == [], f"{name} has a code cell with outputs"
                assert cell.get("execution_count") is None, f"{name} has an executed cell"


def test_notebooks_contain_no_obvious_secrets():
    import nbformat

    for name in NOTEBOOK_NAMES:
        nb = nbformat.read(NOTEBOOKS_DIR / name, as_version=4)
        text = json.dumps(nb).lower()
        for pattern in _SECRET_PATTERNS:
            assert pattern.lower() not in text, f"{name} contains suspicious pattern {pattern!r}"


def test_notebooks_use_interactive_wandb_login():
    import nbformat

    for name in WANDB_NOTEBOOK_NAMES:
        nb = nbformat.read(NOTEBOOKS_DIR / name, as_version=4)
        source = "\n".join(
            "".join(cell.source) if isinstance(cell.source, list) else cell.source
            for cell in nb.cells
        )
        assert "wandb.login()" in source, f"{name} must call wandb.login() interactively"


def test_notebooks_install_from_the_drive_zip_not_a_git_branch():
    """Every notebook must install `pop` from the Drive-uploaded `pop_repo.zip`.

    Two earlier notebooks instead ran `git clone -b <branch>`, and rotted the moment that
    branch stopped existing -- Run all failed at the install cell. The zip is built from the
    reader's own checkout (`git archive HEAD`), so it cannot go stale that way.
    """
    import nbformat

    for name in NOTEBOOK_NAMES:
        nb = nbformat.read(NOTEBOOKS_DIR / name, as_version=4)
        source = "\n".join(
            "".join(cell.source) if isinstance(cell.source, list) else cell.source
            for cell in nb.cells
        )
        assert "pop_repo.zip" in source, f"{name} must install from pop_repo.zip"
        code = "\n".join(
            "".join(cell.source) if isinstance(cell.source, list) else cell.source
            for cell in nb.cells
            if cell.cell_type == "code"
        )
        assert "git clone" not in code, f"{name} must not git clone a branch"
        if "github.com" in source:
            assert "github.com/yib7/pretrain-or-prompt" in source, name


# ---- results/smoke.json (committed from the verified run) ----


def test_smoke_results_json_committed_with_non_null_metrics():
    results_path = REPO_ROOT / "results" / "smoke.json"
    assert results_path.is_file(), "results/smoke.json should be committed from a verified run"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    assert metrics["n"] > 0
    assert metrics["codebleu"] is not None
    assert metrics["syntax_valid_rate"] is not None


def test_smoke_never_writes_over_the_committed_results_file():
    """A local `pop smoke` must not target the committed results/smoke.json.

    Regression: the default `results_name` used to be "smoke", so the first command the
    README tells a visitor to run silently rewrote a published, tracked measurement.
    """
    import yaml

    from pop.config import SmokeConfig

    assert SmokeConfig().results_name == "smoke_local"
    shipped = yaml.safe_load((CONFIGS_DIR / "smoke.yaml").read_text(encoding="utf-8"))
    assert shipped["results_name"] == "smoke_local"
    assert (REPO_ROOT / "results" / "smoke.json").is_file()  # the one it must not touch


def test_execbench_default_run_names_cannot_collide_with_committed_results():
    """`pop execbench`'s defaults must land in the gitignored scratch namespace.

    Regression: the defaults used to be `execbench_validate_references` /
    `execbench_predictions`, so the command in `.github/workflows/ci.yml` overwrote the
    committed 201/201 reference validation with a truncated 2-bug run.
    """
    from pop.cli import EXECBENCH_PREDICTIONS_RESULTS_NAME, EXECBENCH_VALIDATE_RESULTS_NAME
    from pop.eval.metrics import is_scratch_run_name

    assert is_scratch_run_name(EXECBENCH_VALIDATE_RESULTS_NAME)
    assert is_scratch_run_name(EXECBENCH_PREDICTIONS_RESULTS_NAME)
    # ...and the published file those defaults used to collide with is still guarded.
    assert not is_scratch_run_name("execbench_validate_references")
    assert (REPO_ROOT / "results" / "execbench_validate_references.json").is_file()
