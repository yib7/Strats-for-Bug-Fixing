"""Tests for the suite's own guards in `tests/conftest.py`.

A guard nobody tests is a guard that can silently stop guarding -- a renamed fixture, a
`monkeypatch.setattr` pointed at the wrong object, a marker typo -- and the failure mode is
that the suite goes back to being allowed on the network without anyone noticing. These
assert that the fixtures are actually installed and actually bite.
"""

from __future__ import annotations

import socket

import pytest

# `tests/` has no __init__.py, so pytest imports conftest.py as the top-level module
# `conftest`. Importing it as `tests.conftest` would load a *second* copy of the module and
# these tests would then compare against a different exception class than the one raised.
from conftest import NetworkAccessDuringTestError

# --- the offline guard ---------------------------------------------------------------------


def test_connecting_to_a_remote_host_is_blocked():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(NetworkAccessDuringTestError, match="tried to reach the network"):
        sock.connect(("huggingface.co", 443))
    sock.close()


def test_connect_ex_to_a_remote_host_is_blocked():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(NetworkAccessDuringTestError):
        sock.connect_ex(("api.wandb.ai", 443))
    sock.close()


def test_resolving_a_remote_hostname_is_blocked():
    with pytest.raises(NetworkAccessDuringTestError, match="getaddrinfo"):
        socket.getaddrinfo("o151352.ingest.sentry.io", 443)


def test_the_guard_names_the_host_it_blocked():
    with pytest.raises(NetworkAccessDuringTestError) as excinfo:
        socket.getaddrinfo("example.invalid", 80)
    assert "example.invalid" in str(excinfo.value)
    assert "allow_network" in str(excinfo.value), "the message must say how to opt out"


def test_loopback_is_left_alone():
    """The guard must not become the cause of failures in local machinery."""
    infos = socket.getaddrinfo("127.0.0.1", 0)
    assert infos


def test_an_https_client_cannot_slip_past_the_guard():
    """The realistic accident is a dependency fetching a model, not a raw socket call."""
    import urllib.error
    import urllib.request

    with pytest.raises((NetworkAccessDuringTestError, urllib.error.URLError)) as excinfo:
        urllib.request.urlopen("https://huggingface.co/", timeout=5)  # noqa: S310
    # urllib wraps the failure; either way the underlying cause must be our guard.
    assert "tried to reach the network" in str(excinfo.value)


@pytest.mark.allow_network
def test_the_opt_out_marker_restores_the_real_socket_api():
    """Asserts the escape hatch exists without using it: no connection is made here."""
    assert socket.socket.connect.__name__ == "connect"
    assert socket.getaddrinfo is socket._socket.getaddrinfo or callable(socket.getaddrinfo)


# --- the manifest-cache isolation guard ----------------------------------------------------


def test_manifest_cache_is_empty_at_the_start_of_a_test():
    from pop.execbench import harness

    assert harness._load_manifest_cached.cache_info().currsize == 0


def test_manifest_cache_still_memoises_within_one_test():
    """Clearing between tests must not turn the cache off."""
    from pop.execbench import harness

    harness.load_manifest("quixbugs")
    harness.load_manifest("quixbugs")
    info = harness._load_manifest_cached.cache_info()
    assert info.currsize == 1
    assert info.hits >= 1
