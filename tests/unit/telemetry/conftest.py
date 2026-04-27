# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Shared fixtures for the telemetry unit test package."""

import pytest

from winml.modelkit.telemetry import Telemetry
from winml.modelkit.telemetry import consent as consent_mod
from winml.modelkit.telemetry import telemetry as telemetry_mod


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the Telemetry singleton between tests so each one starts clean.

    Calls ``shutdown()`` on any pre-existing instance so a real
    ``BatchLogRecordProcessor`` thread (created when a test exercises the
    real LoggerProvider path) does not leak across tests.
    """
    if telemetry_mod._INSTANCE is not None:
        try:
            telemetry_mod._INSTANCE.shutdown()
        except Exception:
            # Best-effort cleanup: a half-initialized singleton from a
            # prior test must not block resetting state for this test.
            pass
    telemetry_mod._INSTANCE = None
    yield
    if telemetry_mod._INSTANCE is not None:
        try:
            telemetry_mod._INSTANCE.shutdown()
        except Exception:
            # Same rationale as above; teardown must always reach the
            # _INSTANCE = None reset below.
            pass
    telemetry_mod._INSTANCE = None


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    """Redirect consent storage to a per-test temp config file."""
    path = tmp_path / "config.json"
    monkeypatch.setattr(consent_mod, "_CONFIG_PATH", path)
    return path


@pytest.fixture
def clean_env(monkeypatch):
    """Remove telemetry-relevant env vars so tests see a known-empty env."""
    for var in (
        "CI",
        "TF_BUILD",
        "GITHUB_ACTIONS",
        "JENKINS_URL",
        "CODEBUILD_BUILD_ID",
        "BUILDKITE",
        "SYSTEM_TEAMFOUNDATIONCOLLECTIONURI",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def enabled_telemetry(monkeypatch, isolated_config, clean_env):
    """Set up the environment for a fully-enabled Telemetry singleton.

    The singleton itself is NOT eagerly constructed — tests that want a
    ready instance should use :func:`running_telemetry`, or call
    ``Telemetry.get_or_init()`` from inside the test body.
    """
    monkeypatch.setattr("winml.modelkit.telemetry.constants.INSTRUMENTATION_KEY", "o:test-key")
    consent_mod._write_stored_consent("enabled")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)


@pytest.fixture
def running_telemetry(enabled_telemetry):
    """Eagerly construct and return a fully-enabled Telemetry instance.

    The underlying ``_logger`` / ``_provider`` are real OneCollector
    objects; tests that want to introspect emission should replace
    ``_logger`` (or ``_provider``) with a ``MagicMock`` inline.
    """
    t = Telemetry.get_or_init()
    assert t.disabled is False
    return t
