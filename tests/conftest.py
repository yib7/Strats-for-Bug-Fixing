"""Suite-wide guards.

Everything here exists so that a bare `pytest` is *hermetic and order-independent*: it must
not talk to a network service whatever happens to be exported in the developer's shell, and
no test may leave process-global state behind that changes how a later test behaves.
"""

from __future__ import annotations

import socket

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


# --- socket-level hermeticity -------------------------------------------------------------

# Loopback stays open. Nothing in the suite is expected to use it, but blocking it would make
# the guard itself the cause of a failure in unrelated machinery -- multiprocessing on
# Windows, a debugger attach -- rather than catching an outbound call, which is the point.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "localhost.localdomain", ""})


class NetworkAccessDuringTestError(AssertionError):
    """A test tried to reach the network. See `tests/conftest.py`."""


def _host_of(address: object) -> str:
    """The host part of whatever a socket API was handed, as a string."""
    if isinstance(address, tuple) and address:
        return str(address[0])
    return str(address)


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that opens a non-loopback connection or resolves a hostname.

    The env-var neutralization above is indirect: it disables the two SDKs we know phone
    home, which only holds as long as nobody adds a third. This is the direct form of the
    same claim -- `socket.socket.connect`, `connect_ex` and `getaddrinfo` are the three
    chokepoints every Python HTTP client (requests, httpx, urllib3, huggingface_hub,
    wandb) ultimately funnels through, so patching them turns "we believe the tests are
    offline" into "the suite fails if they are not".

    This is a guard against *accidental* egress by our own code, not a sandbox: a test could
    trivially bypass it, and a C extension calling connect(2) directly would not be seen. It
    catches the realistic case, which is a new dependency or a new test quietly fetching a
    model.

    Opt out with `@pytest.mark.allow_network` for a test that genuinely needs it; there is no
    such test today, and a new one should have to say so in its own source.
    """
    if request.node.get_closest_marker("allow_network"):
        return

    def _blocked(what: str, address: object) -> NetworkAccessDuringTestError:
        return NetworkAccessDuringTestError(
            f"test tried to reach the network: socket.{what}({address!r}). "
            "The suite must run offline. If this test genuinely needs a network, mark it "
            "@pytest.mark.allow_network; otherwise inject a fake."
        )

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self, address, *args, **kwargs):  # noqa: ANN001, ANN202
        if _host_of(address) not in _LOOPBACK_HOSTS:
            raise _blocked("connect", address)
        return real_connect(self, address, *args, **kwargs)

    def guarded_connect_ex(self, address, *args, **kwargs):  # noqa: ANN001, ANN202
        if _host_of(address) not in _LOOPBACK_HOSTS:
            raise _blocked("connect_ex", address)
        return real_connect_ex(self, address, *args, **kwargs)

    def guarded_getaddrinfo(host, *args, **kwargs):  # noqa: ANN001, ANN202
        if host is not None and str(host) not in _LOOPBACK_HOSTS:
            raise _blocked("getaddrinfo", host)
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


# --- order independence -------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_manifest_cache() -> None:
    """Drop `pop.execbench.harness`'s memoised manifests around every test.

    `load_manifest` memoises by bench name for the duration of the process. A test that
    points `BENCHMARKS_DIR` at a fixture directory and loads `"quixbugs"` therefore leaves
    that fixture cached for every test that runs afterwards, and the failure surfaces as an
    unrelated test breaking depending on collection order -- the single hardest kind of test
    failure to diagnose. Clearing on both sides means neither the poisoner nor the poisoned
    has to know the other exists.

    Imported inside the fixture so collecting a suite that never touches execbench does not
    pull the module in.
    """
    from pop.execbench import harness

    harness.clear_manifest_cache()
    yield
    harness.clear_manifest_cache()
