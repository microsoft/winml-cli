# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for --device/--ep flags in compile and --precision in quantize."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.commands.compile import _resolve_compile_provider
from winml.modelkit.commands.quantize import _resolve_quant_types


# =============================================================================
# _resolve_compile_provider tests
# =============================================================================


class TestResolveCompileProvider:
    """Test compile provider resolution from device + ep flags."""

    def test_npu_defaults_to_qnn(self):
        assert _resolve_compile_provider("npu", None) == "qnn"

    def test_gpu_defaults_to_dml(self):
        assert _resolve_compile_provider("gpu", None) == "dml"

    def test_cpu_returns_cpu(self):
        assert _resolve_compile_provider("cpu", None) == "cpu"

    def test_auto_defaults_to_qnn(self):
        # auto maps to qnn (NPU-first, like DEVICE_POLICY_MAP)
        result = _resolve_compile_provider("auto", None)
        # auto is not in _DEVICE_TO_PROVIDER, falls through to "qnn" default
        assert result == "qnn"

    def test_ep_overrides_device(self):
        """ep takes priority over device mapping."""
        assert _resolve_compile_provider("npu", "migraphx") == "migraphx"
        assert _resolve_compile_provider("gpu", "vitisai") == "vitisai"
        assert _resolve_compile_provider("cpu", "tensorrt") == "tensorrt"

    def test_ep_is_lowercased(self):
        assert _resolve_compile_provider("gpu", "MIGraphX") == "migraphx"
        assert _resolve_compile_provider("gpu", "TENSORRT") == "tensorrt"

    @pytest.mark.parametrize(
        "ep",
        ["qnn", "dml", "migraphx", "tensorrt", "vitisai", "openvino", "cpu"],
    )
    def test_all_valid_eps(self, ep):
        """All valid EP names resolve correctly."""
        result = _resolve_compile_provider("npu", ep)
        assert result == ep

    def test_device_case_insensitive(self):
        assert _resolve_compile_provider("NPU", None) == "qnn"
        assert _resolve_compile_provider("GPU", None) == "dml"


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
    """Device label in compile summary must reflect the resolved EP, not the CLI default."""

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
            patch("winml.modelkit.commands.compile.compile_onnx", return_value=mock_result),
            patch("winml.modelkit.commands.compile.WinMLCompileConfig"),
        ):
            result = CliRunner().invoke(compile, ["-m", str(model_file), "--ep", "dml"])

        assert "Device: gpu" in result.output
        assert "Device: npu" not in result.output
