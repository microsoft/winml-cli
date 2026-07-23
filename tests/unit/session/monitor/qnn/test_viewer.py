# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for qnn.viewer SDK-root resolution.

These tests cover the env-only resolution contract for
``find_qnn_sdk`` (no hardcoded developer-machine fallback paths).
"""

from __future__ import annotations

from winml.modelkit.session.monitor.qnn.viewer import find_qnn_sdk


def test_find_qnn_sdk_returns_none_when_env_unset(monkeypatch, tmp_path):
    """No env var set -> None (no fallback to hardcoded paths)."""
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    assert find_qnn_sdk() is None


def test_find_qnn_sdk_returns_path_when_env_points_to_dir(monkeypatch, tmp_path):
    """Env var pointing to an existing directory -> that Path is returned."""
    monkeypatch.setenv("QNN_SDK_ROOT", str(tmp_path))
    assert find_qnn_sdk() == tmp_path


def test_find_qnn_sdk_returns_none_when_env_points_to_nonexistent(monkeypatch, tmp_path):
    """Env var pointing to a non-existent path -> None."""
    monkeypatch.setenv("QNN_SDK_ROOT", str(tmp_path / "does-not-exist"))
    assert find_qnn_sdk() is None
