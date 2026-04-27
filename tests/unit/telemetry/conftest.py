# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Shared fixtures for the telemetry unit test package."""

import pytest


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    """Redirect consent storage to a per-test temp config file.

    Consumed by test_consent.py and the Telemetry singleton tests.
    """
    from winml.modelkit.telemetry import consent as consent_mod

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
