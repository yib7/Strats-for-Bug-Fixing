"""Suite-wide guards.

The only thing here is a hermeticity guard: running the tests must not talk to a network
service, whatever happens to be exported in the developer's shell.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_experiment_tracking_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize Weights & Biases (and Hub telemetry) for every test.

    `run_pretrain` / `run_finetune` / `run_lora` enable the `wandb` Trainer integration
    whenever `WANDB_API_KEY` is set -- correct for a real GPU run, but the tests call those
    trainers for real. So on a machine where the key is exported (the normal state of an ML
    practitioner's shell) a bare `pytest` used to authenticate against api.wandb.ai, start a
    run, hand wandb the repo's `remote.origin.url` and HEAD sha, and resolve wandb's Sentry
    endpoint for crash reporting. Measured, not assumed: with the key set, one instrumented
    run of `tests/test_train_entrypoints.py::test_run_pretrain_two_steps` spawned
    `wandb-core`, ran `git config --get remote.origin.url`, and looked up
    `o151352.ingest.sentry.io`.

    `WANDB_MODE=disabled` no-ops the SDK entirely; the key is dropped as well so nothing can
    authenticate, and error reporting is switched off so no crash report leaves the machine.
    This is function-scoped and autouse, so a test that deliberately exercises the
    key-driven code path (`test_run_smoke_forces_report_to_empty`) can still
    `monkeypatch.setenv("WANDB_API_KEY", ...)` on top of it.
    """
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")
    # `pop` sends no telemetry of its own; this trims huggingface_hub's request User-Agent
    # for the (network-free) tests that construct Hub-aware objects.
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
