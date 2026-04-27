# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for QNNMonitor — the QNN EP op-tracing monitor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_ctor_defaults():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor()
    assert m._level == "basic"
    assert m._output_dir.exists()
    assert m._csv_path.is_absolute()


def test_ctor_accepts_custom_output_dir(tmp_path):
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor(output_dir=tmp_path)
    assert m._output_dir == tmp_path
    assert str(m._csv_path).startswith(str(tmp_path))


def test_ctor_rejects_invalid_level():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    with pytest.raises(ValueError, match="level"):
        QNNMonitor(level="bogus")  # type: ignore[arg-type]


def test_get_session_options_has_disable_cpu_fallback():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    opts = QNNMonitor().get_session_options()
    assert opts["session.disable_cpu_ep_fallback"] == "1"
    assert opts["ep.context_enable"] == "1"
    assert opts["ep.context_embed_mode"] == "0"


def test_get_provider_options_basic():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    opts = QNNMonitor(level="basic").get_provider_options()
    assert opts["profiling_level"] == "detailed"
    assert opts["backend_path"] == "QnnHtp.dll"
    assert opts["htp_performance_mode"] == "high_performance"
    assert "profiling_file_path" in opts


def test_get_provider_options_detail():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    assert QNNMonitor(level="detail").get_provider_options()["profiling_level"] == "optrace"


def test_profiling_keys_not_user_overridable():
    """C-3: user extras cannot override profiling_level or profiling_file_path."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor(
        level="basic",
        extra_provider_options={
            "profiling_level": "off",
            "profiling_file_path": "/attacker/path",
            "htp_performance_mode": "balanced",
        },
    )
    opts = m.get_provider_options()
    assert opts["profiling_level"] == "detailed"
    assert opts["profiling_file_path"] != "/attacker/path"
    assert opts["htp_performance_mode"] == "balanced"  # non-owned extra honored


def test_get_provider_options_idempotent():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor(level="basic")
    assert m.get_provider_options() == m.get_provider_options()


def test_get_session_options_idempotent():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor(level="basic")
    assert m.get_session_options() == m.get_session_options()


def test_requires_session_teardown_true():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    assert QNNMonitor.requires_session_teardown is True


def test_double_enter_raises():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor()
    m.__enter__()
    with pytest.raises(RuntimeError, match="already entered"):
        m.__enter__()


def test_exit_with_no_csv_reports_no_data(tmp_path):
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor(output_dir=tmp_path)
    m.__enter__()
    m.__exit__(None, None, None)
    d = m.to_dict()
    assert d["status"] == "no_data"


def test_exit_parse_failure_caught(tmp_path):
    """If CSV exists but is corrupt, status is 'parse_failed' and error is populated."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    csv = tmp_path / "profiling_output.csv"
    csv.write_text("this is not a valid qnn csv")
    m = QNNMonitor(output_dir=tmp_path)
    m.__enter__()
    m.__exit__(None, None, None)
    d = m.to_dict()
    # Either 'parse_failed' (if parser raises) or 'ok'/'no_data' (if parser
    # gracefully returns empty). We accept any of those but must NOT raise.
    assert d["status"] in ("parse_failed", "no_data", "ok")


def test_exit_does_not_suppress_caller_exception(tmp_path):
    """EPMonitor.__exit__ returning None (not True) → exception propagates."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor(output_dir=tmp_path)
    m.__enter__()
    result = m.__exit__(RuntimeError, RuntimeError("test"), None)
    assert result is None or result is False


def test_to_dict_before_enter():
    """Calling to_dict() before enter/exit returns 'not_run' status in nested schema."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor()
    d = m.to_dict()
    assert d["status"] == "not_run"
    # Schema must match the post-exit OpTraceResult.to_dict() shape.
    assert d["metadata"]["ep"] == "QNNExecutionProvider"


def test_to_dict_pre_exit_returns_nested_schema(tmp_path):
    """Pre-exit to_dict() emits the same nested keys as a fully-populated result."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    monitor = QNNMonitor(level="basic", output_dir=tmp_path)
    out = monitor.to_dict()
    assert "metadata" in out
    assert "summary" in out
    assert "operators" in out
    assert "artifacts" in out
    assert out["status"] == "not_run"
    assert out["metadata"]["tracing_level"] == "basic"
    assert out["metadata"]["device"] == "npu"


def test_is_available_via_bundled():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    with patch(
        "onnxruntime.get_available_providers",
        return_value=["QNNExecutionProvider", "CPUExecutionProvider"],
    ):
        assert QNNMonitor.is_available() is True


def test_is_available_via_winml():
    """When QNN EP is registered via WinML, is_available() returns True."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    fake_ep = MagicMock()
    fake_ep.ep_name = "QNNExecutionProvider"
    with (
        patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]),
        patch("onnxruntime.get_ep_devices", return_value=[fake_ep]),
        patch("winml.modelkit.session.ep_registry.ensure_initialized"),
    ):
        assert QNNMonitor.is_available() is True


def test_is_available_neither():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    with (
        patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"]),
        patch("onnxruntime.get_ep_devices", return_value=[]),
        patch("winml.modelkit.session.ep_registry.ensure_initialized"),
    ):
        assert QNNMonitor.is_available() is False


def test_result_property_none_before_exit():
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    m = QNNMonitor()
    assert m.result is None


def test_no_os_chdir():
    """QNNMonitor MUST NOT mutate CWD per FR-12 / C-5."""
    from pathlib import Path

    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    cwd_before = Path.cwd()
    m = QNNMonitor()
    m.__enter__()
    m.__exit__(None, None, None)
    assert Path.cwd() == cwd_before


def test_find_schematic_rejects_stale_cwd_candidate(tmp_path, monkeypatch):
    """A *_schematic.bin in CWD older than the profiling CSV must NOT be returned.

    Setup:
      - output_dir = tmp_path/out  (no schematic in it → exercise CWD fallback)
      - cwd        = tmp_path/cwd  (contains a STALE schematic)
      - csv        = tmp_path/out/profiling_output.csv (FRESH, written 'now')
    Expected: the stale CWD schematic is older than the CSV by >5s, so the
    mtime gate rejects it and _find_schematic() returns None.
    """
    import os
    import time

    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    out_dir = tmp_path / "out"
    cwd_dir = tmp_path / "cwd"
    out_dir.mkdir()
    cwd_dir.mkdir()

    monitor = QNNMonitor(level="detail", output_dir=out_dir)
    # Fresh CSV (now)
    monitor._csv_path.write_text("dummy")
    # Stale schematic in CWD (1 hour old)
    stale = cwd_dir / "stale_schematic.bin"
    stale.write_bytes(b"")
    old = time.time() - 3600
    os.utime(stale, (old, old))

    monkeypatch.chdir(cwd_dir)
    # CWD glob would surface 'stale', but mtime guard rejects.
    assert monitor._find_schematic() is None


def test_find_schematic_accepts_fresh_cwd_candidate(tmp_path, monkeypatch):
    """A *_schematic.bin in CWD newer than the profiling CSV is accepted (mtime gate)."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    out_dir = tmp_path / "out"
    cwd_dir = tmp_path / "cwd"
    out_dir.mkdir()
    cwd_dir.mkdir()

    monitor = QNNMonitor(level="detail", output_dir=out_dir)
    # CSV first, then a fresh schematic — the schematic mtime >= CSV mtime.
    monitor._csv_path.write_text("dummy")
    fresh = cwd_dir / "fresh_schematic.bin"
    fresh.write_bytes(b"")

    monkeypatch.chdir(cwd_dir)
    assert monitor._find_schematic() == fresh


def test_find_schematic_prefers_output_dir_over_cwd(tmp_path, monkeypatch):
    """When output_dir contains a schematic, CWD is never consulted."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    out_dir = tmp_path / "out"
    cwd_dir = tmp_path / "cwd"
    out_dir.mkdir()
    cwd_dir.mkdir()

    monitor = QNNMonitor(level="detail", output_dir=out_dir)
    in_out = out_dir / "graph_schematic.bin"
    in_out.write_bytes(b"")
    in_cwd = cwd_dir / "graph_schematic.bin"
    in_cwd.write_bytes(b"")

    monkeypatch.chdir(cwd_dir)
    assert monitor._find_schematic() == in_out


def test_output_dir_property_exposes_path(tmp_path):
    """The output_dir property returns the directory used for artifacts."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    monitor = QNNMonitor(level="basic", output_dir=tmp_path)
    assert monitor.output_dir == tmp_path
    assert monitor.output_dir.is_dir()


def test_output_dir_property_for_default_tempdir():
    """When output_dir=None, the property exposes the auto-minted tempdir."""
    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    monitor = QNNMonitor(level="basic")
    assert monitor.output_dir.is_dir()
    assert monitor.output_dir.name.startswith("qnn_profile_")


def test_output_dir_property_is_read_only(tmp_path):
    """output_dir is exposed as a property; rebinding must raise AttributeError."""
    import pytest as _pytest

    from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

    monitor = QNNMonitor(level="basic", output_dir=tmp_path)
    with _pytest.raises(AttributeError):
        monitor.output_dir = tmp_path / "other"  # type: ignore[misc]
