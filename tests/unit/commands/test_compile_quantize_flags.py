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
        assert _resolve_compile_provider("cpu", "nv_tensorrt_rtx") == "nv_tensorrt_rtx"

    def test_ep_is_lowercased(self):
        assert _resolve_compile_provider("gpu", "MIGraphX") == "migraphx"
        assert _resolve_compile_provider("gpu", "NV_TENSORRT_RTX") == "nv_tensorrt_rtx"

    @pytest.mark.parametrize(
        "ep",
        ["qnn", "dml", "migraphx", "nv_tensorrt_rtx", "vitisai", "openvino", "cpu"],
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
    """Device label in compile summary must reflect the user-supplied --device flag."""

    def test_device_flag_shown_in_output(self, tmp_path):
        """--device gpu must appear in the Device line regardless of the EP.

        The old code used _EP_TO_DEVICE.get(provider, device) to infer the
        device from the EP name. The new code always prints the user-supplied
        --device flag directly, so the label is unambiguous.
        """
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
            patch("winml.modelkit.commands.compile.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.compiler.compile_onnx", return_value=mock_result),
            patch("winml.modelkit.compiler.WinMLCompileConfig"),
        ):
            result = CliRunner().invoke(
                compile, ["-m", str(model_file), "--device", "gpu", "--ep", "qnn"]
            )

        assert "Device: gpu" in result.output
        assert "Device: npu" not in result.output


# =============================================================================
# CLI <-> --config precedence (regression tests for Bug 1)
# =============================================================================


class TestQuantizeCliConfigPrecedence:
    """Verify CLI/config-file priority in `winml quantize`.

    Expected priority (well-designed CLI contract):
        CLI explicit option > config-file value > CLI option default

    Regression tests for the bug where ``from_dict`` filled missing JSON keys
    with dataclass defaults, which the precedence block then treated as if
    they came from the file - silently overriding ``--precision``.
    """

    @staticmethod
    def _setup(tmp_path):
        import numpy as np
        import onnx

        rng = np.random.default_rng(0)
        x = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4])
        y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2])
        w = onnx.numpy_helper.from_array(rng.standard_normal((4, 2), dtype=np.float32), "W")
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("MatMul", ["input", "W"], ["output"])],
            "tiny",
            [x],
            [y],
            [w],
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
        model.ir_version = 8
        model_path = tmp_path / "tiny.onnx"
        onnx.save(model, str(model_path))

        config_path = tmp_path / "bc.json"
        config_path.write_text('{"quant": {}}', encoding="utf-8")
        return model_path, config_path

    @staticmethod
    def _captured_config(runner_args, tmp_path):
        from click.testing import CliRunner

        from winml.modelkit.commands.quantize import quantize as quantize_cmd

        captured: dict[str, object] = {}

        def fake_quantize(model_path, output_path=None, config=None, **kwargs):
            captured["config"] = config
            result = MagicMock()
            result.success = True
            result.output_path = output_path
            result.nodes_quantized = 0
            result.total_time_seconds = 0.0
            result.errors = []
            return result

        with patch("winml.modelkit.quant.quantize_onnx", side_effect=fake_quantize):
            r = CliRunner().invoke(quantize_cmd, runner_args, obj={}, catch_exceptions=False)
        assert r.exit_code == 0, r.output
        return captured["config"], r.output

    # ---- Misbehavior A: explicit --precision must win, even with --config ----

    def test_a1_precision_int16_with_empty_config(self, tmp_path):
        model, bc = self._setup(tmp_path)
        cfg, _ = self._captured_config(
            [
                "-m",
                str(model),
                "--config",
                str(bc),
                "--precision",
                "int16",
                "--samples",
                "2",
            ],
            tmp_path,
        )
        assert cfg.weight_type == "int16", f"weight_type={cfg.weight_type}"
        assert cfg.activation_type == "uint16", f"activation_type={cfg.activation_type}"

    def test_a2_precision_w8a16_with_empty_config(self, tmp_path):
        model, bc = self._setup(tmp_path)
        cfg, _ = self._captured_config(
            [
                "-m",
                str(model),
                "--config",
                str(bc),
                "--precision",
                "w8a16",
                "--samples",
                "2",
            ],
            tmp_path,
        )
        assert cfg.weight_type == "uint8"
        assert cfg.activation_type == "uint16"

    def test_a3_precision_w16a16_with_empty_config(self, tmp_path):
        model, bc = self._setup(tmp_path)
        cfg, _ = self._captured_config(
            [
                "-m",
                str(model),
                "--config",
                str(bc),
                "--precision",
                "w16a16",
                "--samples",
                "2",
            ],
            tmp_path,
        )
        assert cfg.weight_type == "int16"
        assert cfg.activation_type == "uint16"

    # ---- Misbehavior B: CLI sentinel must beat dataclass default ----

    def test_b4_partial_config_only_weight_type(self, tmp_path):
        """JSON sets only quant.weight_type=int16; activation_type must come from precision/default."""  # noqa: E501
        model, _ = self._setup(tmp_path)
        bc = tmp_path / "bc_b4.json"
        bc.write_text('{"quant": {"weight_type": "int16"}}', encoding="utf-8")
        cfg, _ = self._captured_config(
            ["-m", str(model), "--config", str(bc), "--samples", "2"],
            tmp_path,
        )
        assert cfg.weight_type == "int16"
        assert cfg.activation_type == "uint8"
        # With --precision unset and JSON not setting activation_type, the
        # resolver falls back to default uint8 for activation. The contract:
        # JSON's silence about activation_type must not be misread as
        # "user wants uint8" - it stays at the CLI-default sentinel which
        # _resolve_quant_types then maps to uint8 (since precision is None).
        # But weight_type comes from JSON unambiguously. This pins the
        # weight_type value.

    def test_b5_partial_config_only_activation_type(self, tmp_path):
        """JSON sets only quant.activation_type=uint16; weight_type must come from precision/default."""  # noqa: E501
        model, _ = self._setup(tmp_path)
        bc = tmp_path / "bc_b5.json"
        bc.write_text('{"quant": {"activation_type": "uint16"}}', encoding="utf-8")
        cfg, _ = self._captured_config(
            ["-m", str(model), "--config", str(bc), "--samples", "2"],
            tmp_path,
        )
        assert cfg.activation_type == "uint16"
        assert cfg.weight_type == "uint8"
        # With JSON not setting weight_type, weight_type stays at CLI sentinel
        # None, _resolve_quant_types falls back to uint8. Pin activation_type.

    def test_explicit_cli_weight_type_beats_config(self, tmp_path):
        """Explicit --weight-type wins over JSON value."""
        model, _ = self._setup(tmp_path)
        bc = tmp_path / "bc_cli_win.json"
        bc.write_text('{"quant": {"weight_type": "uint8"}}', encoding="utf-8")
        cfg, _ = self._captured_config(
            [
                "-m",
                str(model),
                "--config",
                str(bc),
                "--weight-type",
                "int16",
                "--samples",
                "2",
            ],
            tmp_path,
        )
        assert cfg.weight_type == "int16"

    def test_config_value_used_when_no_cli(self, tmp_path):
        """Config value wins over CLI default when user didn't override."""
        model, _ = self._setup(tmp_path)
        bc = tmp_path / "bc_use.json"
        bc.write_text(
            '{"quant": {"calibration_method": "entropy", "samples": 7}}',
            encoding="utf-8",
        )
        cfg, _output = self._captured_config(
            ["-m", str(model), "--config", str(bc)],
            tmp_path,
        )
        assert cfg.calibration_method == "entropy"
        assert cfg.samples == 7


class TestQuantizeConfigValidation:
    """Build-config parse/shape errors surface as friendly click.UsageError."""

    @staticmethod
    def _setup(tmp_path):
        return TestQuantizeCliConfigPrecedence._setup(tmp_path)

    @staticmethod
    def _invoke(args):
        from click.testing import CliRunner

        from winml.modelkit.commands.quantize import quantize as quantize_cmd

        return CliRunner().invoke(quantize_cmd, args, obj={}, catch_exceptions=False)

    def test_malformed_json_raises_usage_error(self, tmp_path):
        model, _ = self._setup(tmp_path)
        bc = tmp_path / "bad.json"
        bc.write_text('{"quant":', encoding="utf-8")
        r = self._invoke(["-m", str(model), "--config", str(bc)])
        assert r.exit_code != 0
        assert "Invalid JSON in build config" in r.output

    def test_empty_config_raises_usage_error(self, tmp_path):
        model, _ = self._setup(tmp_path)
        bc = tmp_path / "empty.json"
        bc.write_text("", encoding="utf-8")
        r = self._invoke(["-m", str(model), "--config", str(bc)])
        assert r.exit_code != 0
        assert "Config file is empty" in r.output

    def test_non_object_top_level_raises_usage_error(self, tmp_path):
        model, _ = self._setup(tmp_path)
        bc = tmp_path / "list.json"
        bc.write_text("[]", encoding="utf-8")
        r = self._invoke(["-m", str(model), "--config", str(bc)])
        assert r.exit_code != 0
        assert "Build config must be a JSON object" in r.output
