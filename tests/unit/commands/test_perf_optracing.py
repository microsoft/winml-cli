# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the --op-tracing CLI option on winml perf and _resolve_ep_monitor."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from click.testing import CliRunner


if TYPE_CHECKING:
    from pathlib import Path

from winml.modelkit.commands.perf import _resolve_ep_monitor, perf


def _invoke_perf(args: list[str]):
    """Invoke perf CLI with PerfBenchmark.run mocked to prevent model loading."""
    runner = CliRunner()
    with patch(
        "winml.modelkit.commands.perf.PerfBenchmark.run",
        side_effect=RuntimeError("mocked — not running benchmark"),
    ):
        return runner.invoke(perf, args, obj={})


class TestOpTracingOptionParsing:
    """Verify --op-tracing is recognized and validates choices."""

    def test_option_is_recognized(self):
        """--op-tracing is accepted as a valid CLI option."""
        result = _invoke_perf(["--op-tracing", "basic", "-m", "nonexistent"])
        assert "no such option" not in (result.output or "").lower()

    def test_basic_choice_accepted(self):
        """--op-tracing basic is a valid choice."""
        result = _invoke_perf(["--op-tracing", "basic", "-m", "nonexistent"])
        assert "no such option" not in (result.output or "").lower()
        assert "invalid choice" not in (result.output or "").lower()

    def test_detail_choice_accepted(self):
        """--op-tracing detail is a valid choice."""
        result = _invoke_perf(["--op-tracing", "detail", "-m", "nonexistent"])
        assert "no such option" not in (result.output or "").lower()
        assert "invalid choice" not in (result.output or "").lower()

    def test_invalid_choice_rejected(self):
        """--op-tracing with an invalid value is rejected by Click."""
        runner = CliRunner()
        result = runner.invoke(perf, ["--op-tracing", "invalid", "-m", "test"])
        assert result.exit_code != 0
        output_lower = (result.output or "").lower()
        assert "invalid" in output_lower or "choice" in output_lower

    def test_case_insensitive(self):
        """--op-tracing accepts mixed-case values (e.g. Basic, DETAIL)."""
        result = _invoke_perf(["--op-tracing", "BASIC", "-m", "nonexistent"])
        assert "invalid choice" not in (result.output or "").lower()

    def test_without_op_tracing_flag(self):
        """Command works without --op-tracing (default is None)."""
        result = _invoke_perf(["-m", "nonexistent"])
        assert "no such option" not in (result.output or "").lower()

    def test_model_required_with_op_tracing(self):
        """--op-tracing alone without -m still requires a model."""
        runner = CliRunner()
        result = runner.invoke(perf, ["--op-tracing", "basic"])
        assert result.exit_code != 0


class TestResolveEpMonitor:
    """Unit tests for the _resolve_ep_monitor dispatch helper."""

    def test_no_op_tracing_no_ep_returns_null(self, tmp_path: Path):
        """With no op_tracing and no matching EP, returns NullEPMonitor."""
        from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor

        monitor = _resolve_ep_monitor(ep=None, op_tracing=None, output_dir=tmp_path)
        assert isinstance(monitor, NullEPMonitor)

    def test_no_op_tracing_cpu_ep_returns_null(self, tmp_path: Path):
        """CPU EP with no op_tracing yields NullEPMonitor."""
        from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor

        monitor = _resolve_ep_monitor(ep="cpu", op_tracing=None, output_dir=tmp_path)
        assert isinstance(monitor, NullEPMonitor)

    def test_vitisai_ep_no_op_tracing_returns_vitisai_when_available(self, tmp_path: Path):
        """vitisai EP with no op_tracing returns VitisAIMonitor when available."""
        from winml.modelkit.session.monitor.vitisai_monitor import VitisAIMonitor

        with patch.object(VitisAIMonitor, "is_available", return_value=True):
            monitor = _resolve_ep_monitor(ep="vitisai", op_tracing=None, output_dir=tmp_path)
        assert isinstance(monitor, VitisAIMonitor)

    def test_vitisai_ep_unavailable_returns_null(self, tmp_path: Path):
        """vitisai EP with no op_tracing returns NullEPMonitor when VitisAI is unavailable."""
        from winml.modelkit.session.monitor.ep_monitor import NullEPMonitor
        from winml.modelkit.session.monitor.vitisai_monitor import VitisAIMonitor

        with patch.object(VitisAIMonitor, "is_available", return_value=False):
            monitor = _resolve_ep_monitor(ep="vitisai", op_tracing=None, output_dir=tmp_path)
        assert isinstance(monitor, NullEPMonitor)

    def test_op_tracing_qnn_available_returns_qnn_monitor(self, tmp_path: Path):
        """qnn EP with op_tracing returns QNNMonitor when QNN is available."""
        from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

        with patch.object(QNNMonitor, "is_available", return_value=True):
            monitor = _resolve_ep_monitor(ep="qnn", op_tracing="basic", output_dir=tmp_path)
        assert isinstance(monitor, QNNMonitor)

    def test_op_tracing_qnn_unavailable_raises(self, tmp_path: Path):
        """qnn EP with op_tracing raises RuntimeError when QNN is not available."""
        from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

        with (
            patch.object(QNNMonitor, "is_available", return_value=False),
            pytest.raises(RuntimeError, match="Op-tracing not available"),
        ):
            _resolve_ep_monitor(ep="qnn", op_tracing="basic", output_dir=tmp_path)

    def test_op_tracing_unsupported_ep_raises(self, tmp_path: Path):
        """Unsupported EP with op_tracing raises RuntimeError (NFR-2 hard-fail)."""
        with pytest.raises(RuntimeError, match="Op-tracing not available for EP 'dml'"):
            _resolve_ep_monitor(ep="dml", op_tracing="basic", output_dir=tmp_path)

    def test_op_tracing_passes_level_to_qnn_monitor(self, tmp_path: Path):
        """QNNMonitor receives the correct level from _resolve_ep_monitor."""
        from winml.modelkit.session.monitor.qnn_monitor import QNNMonitor

        with patch.object(QNNMonitor, "is_available", return_value=True):
            monitor = _resolve_ep_monitor(ep="qnn", op_tracing="detail", output_dir=tmp_path)
        assert isinstance(monitor, QNNMonitor)
        assert monitor._level == "detail"
