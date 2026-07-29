# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for qnn.viewer SDK-root resolution."""

from __future__ import annotations

import subprocess

import pytest

from winml.modelkit.session.monitor.qnn import viewer
from winml.modelkit.session.monitor.qnn.viewer import find_qnn_sdk


def test_find_qnn_sdk_returns_none_when_env_unset(monkeypatch, tmp_path):
    """No valid env or documented common root -> None."""
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    monkeypatch.setattr(viewer, "_COMMON_SDK_PATHS", [tmp_path / "missing"])
    assert find_qnn_sdk() is None


def test_find_qnn_sdk_returns_path_when_env_points_to_dir(monkeypatch, tmp_path):
    """Env var pointing to an existing directory -> that Path is returned."""
    monkeypatch.setenv("QNN_SDK_ROOT", str(tmp_path))
    assert find_qnn_sdk() == tmp_path


def test_find_qnn_sdk_falls_back_to_common_root_for_invalid_env(monkeypatch, tmp_path):
    """An invalid env path falls through to the documented common roots."""
    monkeypatch.setenv("QNN_SDK_ROOT", str(tmp_path / "does-not-exist"))
    sdk_root = tmp_path / "qairt" / "2.0"
    (sdk_root / "bin").mkdir(parents=True)
    monkeypatch.setattr(viewer, "_COMMON_SDK_PATHS", [tmp_path / "qairt"])
    assert find_qnn_sdk() == sdk_root


def test_find_qnn_sdk_finds_versioned_common_root(monkeypatch, tmp_path):
    """The common-root search selects an SDK version containing ``bin``."""
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    sdk_root = tmp_path / "QC" / "2.30.0.250000"
    (sdk_root / "bin").mkdir(parents=True)
    monkeypatch.setattr(viewer, "_COMMON_SDK_PATHS", [tmp_path / "QC"])

    assert find_qnn_sdk() == sdk_root


def test_find_qnn_sdk_accepts_common_path_as_sdk_root(monkeypatch, tmp_path):
    """An unversioned common path containing bin is itself the SDK root."""
    sdk_root = tmp_path / "qairt"
    (sdk_root / "bin").mkdir(parents=True)
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    monkeypatch.setattr(viewer, "_COMMON_SDK_PATHS", [sdk_root])

    assert find_qnn_sdk() == sdk_root


def test_legacy_viewer_shim_delegates_sdk_discovery_with_one_warning(monkeypatch, tmp_path):
    """The compatibility shim exposes the canonical discovery implementation."""
    from winml.modelkit.optracing.qnn import viewer as legacy_viewer

    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    sdk_root = tmp_path / "qairt" / "2.0"
    (sdk_root / "bin").mkdir(parents=True)
    monkeypatch.setattr(viewer, "_COMMON_SDK_PATHS", [tmp_path / "qairt"])

    with pytest.warns(DeprecationWarning) as warnings:
        assert legacy_viewer.find_qnn_sdk() == sdk_root

    assert len(warnings) == 1


def test_run_qhas_viewer_uses_output_stem_config_per_run(monkeypatch, tmp_path):
    """Same output dir runs must not race on a shared optrace_config.json."""
    from pathlib import Path

    from winml.modelkit.session.monitor.qnn.viewer import run_qhas_viewer

    viewer = tmp_path / "sdk" / "bin" / "x64" / "qnn-profile-viewer.exe"
    viewer.parent.mkdir(parents=True)
    viewer.write_text("", encoding="utf-8")
    reader = tmp_path / "sdk" / "lib" / viewer.parent.name / "QnnHtpOptraceProfilingReader.dll"
    reader.parent.mkdir(parents=True)
    reader.write_bytes(b"")

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
    assert all(Path(cmd[cmd.index("--reader") + 1]) == reader for cmd in commands)
    assert config_paths == [
        tmp_path / "profiling_output_a_qhas_output_optrace_config.json",
        tmp_path / "profiling_output_b_qhas_output_optrace_config.json",
    ]
    assert config_paths[0] != config_paths[1]
    assert all(path.is_file() for path in config_paths)


def test_run_qhas_viewer_requires_matching_optrace_reader(monkeypatch, tmp_path, caplog):
    from winml.modelkit.session.monitor.qnn.viewer import run_qhas_viewer

    sdk_root = tmp_path / "sdk"
    viewer_path = sdk_root / "bin" / "x64" / "qnn-profile-viewer.exe"
    viewer_path.parent.mkdir(parents=True)
    viewer_path.write_bytes(b"")
    qnn_log = tmp_path / "profile.log"
    schematic = tmp_path / "schematic.bin"
    qnn_log.write_bytes(b"")
    schematic.write_bytes(b"")
    monkeypatch.setattr(subprocess, "run", pytest.fail)

    assert (
        run_qhas_viewer(
            qnn_log,
            schematic,
            tmp_path / "output.json",
            sdk_root=sdk_root,
        )
        is None
    )
    assert "QnnHtpOptraceProfilingReader.dll not found" in caplog.text
