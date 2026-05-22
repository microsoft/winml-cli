# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for --precision in quantize and device display label in compile."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from winml.modelkit.commands.quantize import _resolve_quant_types
from winml.modelkit.session import WinMLEPDevice


def _fake_ep_device(ep: str, device: str) -> WinMLEPDevice:
    return WinMLEPDevice(ep=ep, device=device, vendor_id=0, device_id=0, vendor="")


# =============================================================================
# _resolve_quant_types tests
# =============================================================================


class TestResolveQuantTypes:
    """Test quantization type resolution from precision + explicit flags."""

    def test_defaults_without_precision(self):
        """No precision, no explicit types -> defaults (uint8, uint8)."""
        w, a = _resolve_quant_types(None, None, None)
        assert w == "uint8"
        assert a == "uint8"

    def test_precision_int8(self):
        """--precision int8 -> uint8 weights + uint8 activations."""
        w, a = _resolve_quant_types("int8", None, None)
        assert w == "uint8"
        assert a == "uint8"

    def test_precision_int16(self):
        """--precision int16 -> int16 weights + uint16 activations."""
        w, a = _resolve_quant_types("int16", None, None)
        assert w == "int16"
        assert a == "uint16"

    def test_explicit_weight_overrides_precision(self):
        """--precision int16 --weight-type uint8 -> uint8 weight, uint16 activation."""
        w, a = _resolve_quant_types("int16", "uint8", None)
        assert w == "uint8"
        assert a == "uint16"

    def test_explicit_activation_overrides_precision(self):
        """--precision int8 --activation-type int8 -> uint8 weight, int8 activation."""
        w, a = _resolve_quant_types("int8", None, "int8")
        assert w == "uint8"
        assert a == "int8"

    def test_explicit_both_override_precision(self):
        """Both explicit flags override precision entirely."""
        w, a = _resolve_quant_types("int16", "int8", "int8")
        assert w == "int8"
        assert a == "int8"

    def test_explicit_without_precision(self):
        """Explicit flags without precision use their values."""
        w, a = _resolve_quant_types(None, "int16", "uint16")
        assert w == "int16"
        assert a == "uint16"

    def test_precision_case_insensitive(self):
        w, a = _resolve_quant_types("INT8", None, None)
        assert w == "uint8"
        assert a == "uint8"

    def test_unknown_precision_uses_defaults(self):
        """Unknown precision string falls through to defaults."""
        w, a = _resolve_quant_types("fp16", None, None)
        assert w == "uint8"
        assert a == "uint8"


class TestCompileDeviceDisplayLabel:
    """Device label in compile summary must reflect the resolved WinMLEPDevice.device."""

    def test_dml_ep_shows_gpu_device(self, tmp_path):
        from click.testing import CliRunner

        from winml.modelkit.commands.compile import compile

        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake")

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_path = None
        mock_result.compile_time = None
        mock_result.total_time = None

        with (
            patch(
                "winml.modelkit.commands.compile.resolve_device",
                return_value=_fake_ep_device("DmlExecutionProvider", "gpu"),
            ),
            patch("winml.modelkit.commands.compile.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.compiler.compile_onnx", return_value=mock_result),
            patch("winml.modelkit.compiler.WinMLCompileConfig"),
        ):
            result = CliRunner().invoke(compile, ["-m", str(model_file), "--ep", "dml"])

        assert "Device: gpu" in result.output
        assert "Device: npu" not in result.output
