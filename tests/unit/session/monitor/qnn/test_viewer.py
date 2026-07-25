# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for qnn.viewer SDK-root resolution.

These tests cover the env-only resolution contract for
``find_qnn_sdk`` (no hardcoded developer-machine fallback paths).
"""

from __future__ import annotations

import subprocess

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


def test_run_qhas_viewer_uses_output_stem_config_per_run(monkeypatch, tmp_path):
    """Same output dir runs must not race on a shared optrace_config.json."""
    from pathlib import Path

    from winml.modelkit.session.monitor.qnn.viewer import run_qhas_viewer

    viewer = tmp_path / "sdk" / "bin" / "x64" / "qnn-profile-viewer.exe"
    viewer.parent.mkdir(parents=True)
    viewer.write_text("", encoding="utf-8")

    qnn_log_a = tmp_path / "profiling_output_a_qnn.log"
    qnn_log_b = tmp_path / "profiling_output_b_qnn.log"
    schematic_a = tmp_path / "profiling_output_a_schematic.bin"
    schematic_b = tmp_path / "profiling_output_b_schematic.bin"
    output_a = tmp_path / "profiling_output_a_qhas_output.json"
    output_b = tmp_path / "profiling_output_b_qhas_output.json"
    for path in (qnn_log_a, qnn_log_b, schematic_a, schematic_b):
        path.write_bytes(b"")

    commands: list[list[str]] = []

    def _fake_run(cmd: list[str], **_kwargs):
        commands.append(cmd)
        output = Path(cmd[cmd.index("--output") + 1])
        output.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    assert run_qhas_viewer(qnn_log_a, schematic_a, output_a, sdk_root=tmp_path / "sdk") == output_a
    assert run_qhas_viewer(qnn_log_b, schematic_b, output_b, sdk_root=tmp_path / "sdk") == output_b

    config_paths = [Path(cmd[cmd.index("--config") + 1]) for cmd in commands]
    assert config_paths == [
        tmp_path / "profiling_output_a_qhas_output_optrace_config.json",
        tmp_path / "profiling_output_b_qhas_output_optrace_config.json",
    ]
    assert config_paths[0] != config_paths[1]
    assert all(path.is_file() for path in config_paths)
