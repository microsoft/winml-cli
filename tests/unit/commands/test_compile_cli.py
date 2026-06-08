# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for winml compile CLI -- device + ep flag parsing.

These tests verify that the --ep and --device flags are accepted, optional,
and that resolve_device() is called at the CLI boundary with the correct args.
No actual compilation or hardware EP registration is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.compile import compile
from winml.modelkit.session import EPDeviceTarget


if TYPE_CHECKING:
    from pathlib import Path


def _fake_ep_device(ep: str, device: str) -> EPDeviceTarget:
    """Build a stub EPDeviceTarget for mocking resolve_device()."""
    return EPDeviceTarget(ep=ep, device=device)


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def fake_onnx(tmp_path: Path) -> Path:
    """Write a minimal (fake) ONNX file so the CLI's exists= check passes."""
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake-onnx")
    return model


@pytest.fixture
def mock_compile_onnx():
    """Prevent any real EP discovery or compilation; return a successful result stub."""
    result = MagicMock()
    result.success = True
    result.output_path = None
    result.compile_time = None
    result.total_time = None
    result.errors = []

    # Default: resolve_device returns a QNN/NPU EPDeviceTarget.
    default_ep_device = _fake_ep_device("QNNExecutionProvider", "npu")

    with (
        patch(
            "winml.modelkit.commands.compile.resolve_device",
            return_value=default_ep_device,
        ) as mock_resolve,
        patch("winml.modelkit.commands.compile.is_compiled_onnx", return_value=False),
        patch("winml.modelkit.compiler.compile_onnx", return_value=result),
        patch("winml.modelkit.compiler.WinMLCompileConfig"),
    ):
        yield result, mock_resolve


# =============================================================================
# CLI --help smoke tests
# =============================================================================


class TestCompileCliHelp:
    """Verify that --device and --ep appear in --help output."""

    def test_device_option_in_help(self, runner: CliRunner) -> None:
        result = runner.invoke(compile, ["--help"])
        assert result.exit_code == 0
        assert "--device" in result.output

    def test_ep_option_in_help(self, runner: CliRunner) -> None:
        result = runner.invoke(compile, ["--help"])
        assert result.exit_code == 0
        assert "--ep" in result.output

    def test_device_choices_in_help(self, runner: CliRunner) -> None:
        """Help text must expose the device choices."""
        result = runner.invoke(compile, ["--help"])
        assert result.exit_code == 0
        # At least one of the valid choices must appear in the help text
        assert any(c in result.output for c in ["npu", "gpu", "cpu"])


# =============================================================================
# CLI invocation tests (no real compilation)
# =============================================================================


class TestCompileCliDeviceEpFlags:
    """Verify CLI accepts --ep/--device combinations and deduces correctly."""

    def test_ep_qnn_no_device_accepted(
        self, runner: CliRunner, fake_onnx: Path, mock_compile_onnx
    ) -> None:
        """--ep qnn without --device is accepted (device deduced from ep)."""
        _result, _ = mock_compile_onnx
        r = runner.invoke(compile, ["-m", str(fake_onnx), "--ep", "qnn"])
        # Should not fail on argument validation
        assert "Error: Invalid value" not in (r.output or "")

    def test_device_npu_no_ep_accepted(
        self, runner: CliRunner, fake_onnx: Path, mock_compile_onnx
    ) -> None:
        """--device npu without --ep is accepted (ep deduced from device)."""
        _result, _ = mock_compile_onnx
        r = runner.invoke(compile, ["-m", str(fake_onnx), "--device", "npu"])
        assert "Error: Invalid value" not in (r.output or "")

    def test_neither_ep_nor_device_accepted(
        self, runner: CliRunner, fake_onnx: Path, mock_compile_onnx
    ) -> None:
        """Omitting both --ep and --device is accepted (defaults to qnn/npu)."""
        _result, _ = mock_compile_onnx
        r = runner.invoke(compile, ["-m", str(fake_onnx)])
        assert "Error: Invalid value" not in (r.output or "")

    def test_device_npu_shows_npu_in_output(
        self, runner: CliRunner, fake_onnx: Path, mock_compile_onnx
    ) -> None:
        """--device npu → 'Device: npu' appears in output."""
        _result, mock_resolve = mock_compile_onnx
        mock_resolve.return_value = _fake_ep_device("QNNExecutionProvider", "npu")
        r = runner.invoke(compile, ["-m", str(fake_onnx), "--device", "npu"])
        assert "Device: npu" in r.output

    def test_device_gpu_shows_gpu_in_output(
        self, runner: CliRunner, fake_onnx: Path, mock_compile_onnx
    ) -> None:
        """--device gpu → 'Device: gpu' appears in output."""
        _result, mock_resolve = mock_compile_onnx
        mock_resolve.return_value = _fake_ep_device("DmlExecutionProvider", "gpu")
        r = runner.invoke(compile, ["-m", str(fake_onnx), "--device", "gpu"])
        assert "Device: gpu" in r.output

    def test_ep_dml_shows_gpu_in_output(
        self, runner: CliRunner, fake_onnx: Path, mock_compile_onnx
    ) -> None:
        """--ep dml → 'Device: gpu' because dml is a GPU EP."""
        _result, mock_resolve = mock_compile_onnx
        mock_resolve.return_value = _fake_ep_device("DmlExecutionProvider", "gpu")
        r = runner.invoke(compile, ["-m", str(fake_onnx), "--ep", "dml"])
        assert "Device: gpu" in r.output

    def test_resolve_device_called_at_boundary(
        self, runner: CliRunner, fake_onnx: Path, mock_compile_onnx
    ) -> None:
        """resolve_device() is called exactly once with the CLI args."""
        _result, mock_resolve = mock_compile_onnx
        runner.invoke(compile, ["-m", str(fake_onnx), "--ep", "qnn", "--device", "npu"])
        mock_resolve.assert_called_once_with(ep="qnn", device="npu")

    def test_resolve_device_called_ep_only(
        self, runner: CliRunner, fake_onnx: Path, mock_compile_onnx
    ) -> None:
        """--ep only: resolve_device(ep=..., device=None)."""
        _result, mock_resolve = mock_compile_onnx
        runner.invoke(compile, ["-m", str(fake_onnx), "--ep", "vitisai"])
        mock_resolve.assert_called_once_with(ep="vitisai", device=None)

    def test_resolve_device_called_device_only(
        self, runner: CliRunner, fake_onnx: Path, mock_compile_onnx
    ) -> None:
        """--device only: resolve_device(ep=None, device=...)."""
        _result, mock_resolve = mock_compile_onnx
        runner.invoke(compile, ["-m", str(fake_onnx), "--device", "gpu"])
        mock_resolve.assert_called_once_with(ep=None, device="gpu")

    def test_resolve_device_called_neither(
        self, runner: CliRunner, fake_onnx: Path, mock_compile_onnx
    ) -> None:
        """Neither --ep nor --device: resolve_device(ep=None, device=None)."""
        _result, mock_resolve = mock_compile_onnx
        runner.invoke(compile, ["-m", str(fake_onnx)])
        mock_resolve.assert_called_once_with(ep=None, device=None)

    def test_device_auto_treated_as_none(
        self, runner: CliRunner, fake_onnx: Path, mock_compile_onnx
    ) -> None:
        """--device auto: treated as None, resolve_device(ep=None, device=None)."""
        _result, mock_resolve = mock_compile_onnx
        runner.invoke(compile, ["-m", str(fake_onnx), "--device", "auto"])
        mock_resolve.assert_called_once_with(ep=None, device=None)

    def test_invalid_device_rejected(self, runner: CliRunner, fake_onnx: Path) -> None:
        """Unknown device string is rejected by Click."""
        result = runner.invoke(compile, ["-m", str(fake_onnx), "--device", "tpu"])
        assert result.exit_code != 0

    def test_invalid_ep_rejected(self, runner: CliRunner, fake_onnx: Path) -> None:
        """Unknown EP string is rejected by Click."""
        result = runner.invoke(compile, ["-m", str(fake_onnx), "--ep", "unknown_ep"])
        assert result.exit_code != 0
