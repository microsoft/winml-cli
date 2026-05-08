# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Shared fixtures for the telemetry unit test package.

Singleton reset (autouse) lives in :mod:`tests.conftest` so it covers
every test package, not just this one.
"""

import pytest

from winml.modelkit.telemetry import Telemetry
from winml.modelkit.telemetry import consent as consent_mod


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
    monkeypatch.setattr(
        "winml.modelkit.telemetry.constants.INSTRUMENTATION_KEY", "test-tenant-1234"
    )
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
