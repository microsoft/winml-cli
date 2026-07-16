# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for config CLI command without any network dependency.

Most coverage stays mock-based around the CLI wrapper, while local-only
validation and ONNX-path tests exercise real parsing without contacting
external services.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch


if TYPE_CHECKING:
    from pathlib import Path

import pytest
from click.testing import CliRunner, Result


@pytest.fixture(autouse=True)
def mock_resolve_device():
    """Mock resolve_check_device_ep to avoid hardware detection in CLI tests.

    The config command calls resolve_check_device_ep() (lazy import) for
    device/EP resolution and display. We mock at the source module since the
    import happens at call time.
    """
    with patch(
        "winml.modelkit.sysinfo.resolve_check_device_ep",
        return_value=("npu", ["npu", "gpu", "cpu"], ["QNNExecutionProvider"]),
    ):
        yield


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def onnx_model_path(tmp_path: Path) -> Path:
    """Create a valid minimal ONNX model for local-only CLI tests."""
    from onnx import TensorProto, helper, save

    x_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10])
    y_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 5])
    w_init = helper.make_tensor("weight", TensorProto.FLOAT, [10, 5], [0.1] * 50)
    node = helper.make_node("MatMul", ["input", "weight"], ["output"])
    graph = helper.make_graph([node], "test_graph", [x_info], [y_info], [w_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx_path = tmp_path / "test_model.onnx"
    save(model, str(onnx_path))
    return onnx_path


@pytest.fixture
def mock_generate_config():
    """Mock the config-generation boundary to avoid network/model loading.

    Returns a MagicMock whose to_dict() yields a valid JSON-serializable dict.
    The mock target is the lazy import inside modelkit.commands.config.

    Also mocks ``AutoConfig.from_pretrained``: the command inspects the model's
    HF config to route encoder-decoder models built without ``--task`` through
    the full composite (#850). These CLI tests use a placeholder model id, so the
    model-inspection boundary is mocked to a non-composite ``model_type`` (so the
    composite probe returns ``None``) and stays network-free.
    """
    mock_cfg = MagicMock()
    mock_cfg.loader.task = "image-classification"
    mock_cfg.to_dict.return_value = {
        "loader": {
            "task": "image-classification",
            "model_class": "ResNetForImageClassification",
        },
        "export": {"opset_version": 17},
        "optim": {},
        "quant": None,
        "compile": None,
    }
    from transformers import BertConfig

    # A real (non-composite) config so the no-task composite probe's detect_task
    # runs without erroring; bert has no encoder-decoder composite -> probe returns
    # None and the command falls through to the mocked generate_hf_build_config.
    mock_hf_config = BertConfig()
    with (
        patch("transformers.AutoConfig.from_pretrained", return_value=mock_hf_config),
        patch(
            "winml.modelkit.config.generate_hf_build_config",
            return_value=mock_cfg,
        ) as mock,
    ):
        yield mock


# =============================================================================
# CLI INTERFACE TESTS
# =============================================================================


class TestConfigCliInterface:
    """Test CLI flag parsing and help text."""

    def test_help_shows_all_options(self, runner: CliRunner) -> None:
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, ["--help"])
        assert result.exit_code == 0

        # All documented options must appear in help text
        expected_options = [
            "--model",
            "-m",
            "--task",
            "-t",
            "--model-class",
            "--model-type",
            "--module",
            "--config",
            "-c",
            "--shape-config",
            "--device",
            "-d",
            "--precision",
            "-p",
            "--output",
            "-o",
            "--library",
            "--verbose",
            "-v",
            "--no-quant",
            "--no-compile",
            "--trust-remote-code",
        ]
        for opt in expected_options:
            assert opt in result.output, f"Expected '{opt}' in help output"

    def test_no_entry_point_error(self, runner: CliRunner) -> None:
        """Invoking with no args should fail (need -m/--model-type/--model-class)."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, [])
        assert result.exit_code != 0

    def test_invalid_device_rejected(self, runner: CliRunner) -> None:
        """--device tpu should be rejected by click.Choice validation."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, ["-m", "test", "--device", "tpu"])
        assert result.exit_code != 0

    def test_invalid_precision_rejected(self, runner: CliRunner) -> None:
        """--precision bf16 should be rejected by click.Choice validation."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, ["-m", "test", "--precision", "bf16"])
        assert result.exit_code != 0

    @pytest.mark.parametrize("device", ["auto", "npu", "gpu", "cpu"])
    def test_valid_device_choices(
        self,
        runner: CliRunner,
        device: str,
        mock_generate_config: MagicMock,
    ) -> None:
        """All valid device choices should be accepted without error."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, ["-m", "test", "--device", device])
        assert result.exit_code == 0, (
            f"Device '{device}' should be accepted, got exit_code={result.exit_code}: "
            f"{result.output}"
        )

    @pytest.mark.parametrize("precision", ["auto", "fp32", "fp16", "int8", "int16"])
    def test_valid_precision_choices(
        self,
        runner: CliRunner,
        precision: str,
        mock_generate_config: MagicMock,
    ) -> None:
        """All valid precision choices should be accepted without error."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, ["-m", "test", "--precision", precision])
        assert result.exit_code == 0, (
            f"Precision '{precision}' should be accepted, "
            f"got exit_code={result.exit_code}: {result.output}"
        )

    def test_output_to_file(
        self,
        runner: CliRunner,
        tmp_path: Path,
        mock_generate_config: MagicMock,
    ) -> None:
        """Outputting to a file via -o should not crash."""
        from winml.modelkit.commands.config import config

        output_file = tmp_path / "out.json"
        result = runner.invoke(config, ["-m", "test", "-o", str(output_file)])
        assert result.exit_code == 0, f"Output to file should succeed: {result.output}"

    def test_model_type_without_model(
        self,
        runner: CliRunner,
        mock_generate_config: MagicMock,
    ) -> None:
        """--model-type bert --task fill-mask should be a valid entry point (no -m needed)."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, ["--model-type", "bert", "--task", "fill-mask"])
        assert result.exit_code == 0, f"model-type without model should succeed: {result.output}"

    def test_config_file_override(
        self,
        runner: CliRunner,
        tmp_path: Path,
        mock_generate_config: MagicMock,
    ) -> None:
        """A config override file via -c should be accepted."""
        from winml.modelkit.commands.config import config

        override_file = tmp_path / "override.json"
        override_file.write_text('{"loader": {"task": "text-classification"}}')

        result = runner.invoke(config, ["-m", "test", "-c", str(override_file)])
        assert result.exit_code == 0, f"Config file override should succeed: {result.output}"

    def test_shape_config_file(
        self,
        runner: CliRunner,
        tmp_path: Path,
        mock_generate_config: MagicMock,
    ) -> None:
        """A shape config file via --shape-config should be accepted."""
        from winml.modelkit.commands.config import config

        shapes_file = tmp_path / "shapes.json"
        shapes_file.write_text('{"height": 224, "width": 224}')

        result = runner.invoke(config, ["-m", "test", "--shape-config", str(shapes_file)])
        assert result.exit_code == 0, f"Shape config file should succeed: {result.output}"

    def test_no_quant_sets_quant_none(
        self,
        runner: CliRunner,
        mock_generate_config: MagicMock,
    ) -> None:
        """--no-quant should set quant=None on the generated config."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, ["-m", "test", "--no-quant"])
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert mock_generate_config.return_value.quant is None

    def test_no_compile_sets_compile_none(
        self,
        runner: CliRunner,
        mock_generate_config: MagicMock,
    ) -> None:
        """--no-compile should set compile=None on the generated config."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, ["-m", "test", "--no-compile"])
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert mock_generate_config.return_value.compile is None

    def test_trust_remote_code_passed_to_api(
        self,
        runner: CliRunner,
        mock_generate_config: MagicMock,
    ) -> None:
        """--trust-remote-code should be passed to generate_hf_build_config."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, ["-m", "test", "--trust-remote-code"])
        assert result.exit_code == 0, f"Failed: {result.output}"
        mock_generate_config.assert_called_once()
        call_kwargs = mock_generate_config.call_args.kwargs
        assert call_kwargs.get("trust_remote_code") is True


# =============================================================================
# ONNX PATH OVERRIDE TESTS
# =============================================================================


def _extract_json(output: str) -> dict:
    """Extract JSON object from mixed CLI output (Rich stderr + JSON stdout).

    CliRunner in Click 8.x mixes stderr and stdout. The JSON block starts
    at the first '{' and ends at the last '}'.
    """
    import json

    start = output.index("{")
    end = output.rindex("}") + 1
    return json.loads(output[start:end])


def _assert_onnx_config_structure(data: dict) -> None:
    """Assert the structure for ONNX input config output."""
    assert data.get("export") is None
    assert "optim" in data


def _invoke_config(*args: str) -> Result:
    """Invoke the config command; do NOT raise on non-zero exit."""
    from winml.modelkit.commands.config import config

    runner = CliRunner()
    return runner.invoke(config, list(args))


class TestConfigOnnxOverrides:
    """Test --no-quant and --no-compile work on the ONNX path."""

    def test_onnx_no_quant(self, runner: CliRunner, tmp_path: Path) -> None:
        """--no-quant should set quant=None even for ONNX configs."""
        from winml.modelkit.commands.config import config

        # Create a fake .onnx file so it classifies as an ONNX_FILE input
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
        ):
            result = runner.invoke(config, ["-m", str(onnx_file), "--no-quant"])
        assert result.exit_code == 0, f"Failed: {result.output}"

        data = _extract_json(result.output)
        assert data.get("quant") is None

    def test_onnx_no_compile(self, runner: CliRunner, tmp_path: Path) -> None:
        """--no-compile should set compile=None even for ONNX configs."""
        from winml.modelkit.commands.config import config

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
        ):
            result = runner.invoke(config, ["-m", str(onnx_file), "--no-compile"])
        assert result.exit_code == 0, f"Failed: {result.output}"

        data = _extract_json(result.output)
        assert data.get("compile") is None


# =============================================================================
# LOCAL ONNX PATH TESTS
# =============================================================================


class TestConfigOnnxLocalPath:
    """Test local ONNX-path handling that should run in default CI."""

    def test_onnx_model_path(self, runner: CliRunner, onnx_model_path: Path) -> None:
        """Passing a .onnx file should produce export=None config."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, ["-m", str(onnx_model_path)], catch_exceptions=False)
        assert result.exit_code == 0, f"Failed: {result.output}"

        data = _extract_json(result.output)
        _assert_onnx_config_structure(data)

    def test_onnx_with_no_compile(self, runner: CliRunner, onnx_model_path: Path) -> None:
        """--no-compile on the ONNX path should yield compile=None."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(
            config,
            ["-m", str(onnx_model_path), "--no-compile"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"Failed: {result.output}"

        data = _extract_json(result.output)
        _assert_onnx_config_structure(data)
        assert data.get("compile") is None

    def test_onnx_with_no_quant(self, runner: CliRunner, onnx_model_path: Path) -> None:
        """--no-quant on the ONNX path should yield quant=None."""
        from winml.modelkit.commands.config import config

        result = runner.invoke(
            config,
            ["-m", str(onnx_model_path), "--no-quant"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"Failed: {result.output}"

        data = _extract_json(result.output)
        _assert_onnx_config_structure(data)
        assert data.get("quant") is None

    def test_onnx_output_to_file(
        self,
        runner: CliRunner,
        onnx_model_path: Path,
        tmp_path: Path,
    ) -> None:
        """ONNX-path config should serialize to disk via -o."""
        from winml.modelkit.commands.config import config

        outfile = tmp_path / "onnx_config.json"
        result = runner.invoke(
            config,
            ["-m", str(onnx_model_path), "-o", str(outfile)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"Failed: {result.output}"
        assert outfile.exists()

        _assert_onnx_config_structure(json.loads(outfile.read_text()))


# =============================================================================
# BAD PATH TESTS
# =============================================================================


class TestConfigBadPath:
    """Bad-path coverage: invalid args, invalid JSON, and local-only errors."""

    def test_no_args_is_error(self) -> None:
        """Invoking with no args must fail with a usage error, not a traceback."""
        result = _invoke_config()
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    def test_missing_entry_point_message(self) -> None:
        """The error message should point the user at the required flags."""
        result = _invoke_config()
        assert result.exit_code != 0
        combined = (result.output or "") + (str(result.exception) if result.exception else "")
        assert "--model" in combined or "--model-type" in combined or "--model-class" in combined

    @pytest.mark.parametrize("bad_device", ["tpu", "fpga", "xpu", "DSP"])
    def test_invalid_device_rejected(self, bad_device: str) -> None:
        """Click's Choice validation must reject unknown --device values."""
        result = _invoke_config(
            "--model-type",
            "bert",
            "--task",
            "fill-mask",
            "--device",
            bad_device,
        )
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    @pytest.mark.parametrize("bad_precision", ["bf16", "fp64", "w3a5", "w4a4"])
    def test_invalid_precision_rejected(self, bad_precision: str) -> None:
        """Unknown precision strings must produce a UsageError, not a traceback."""
        result = _invoke_config(
            "--model-type",
            "bert",
            "--task",
            "fill-mask",
            "--precision",
            bad_precision,
        )
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    @pytest.mark.parametrize("bad_ep", ["tflite", "coreml", "not-a-real-ep"])
    def test_invalid_ep_rejected(self, bad_ep: str) -> None:
        """Unknown --ep values must produce a UsageError, not a traceback."""
        result = _invoke_config(
            "--model-type",
            "bert",
            "--task",
            "fill-mask",
            "--ep",
            bad_ep,
        )
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    def test_nonexistent_config_file_rejected(self, tmp_path: Path) -> None:
        """-c pointing at a missing file must be rejected by Click."""
        missing = tmp_path / "does_not_exist.json"
        result = _invoke_config(
            "--model-type",
            "bert",
            "--task",
            "fill-mask",
            "-c",
            str(missing),
        )
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    def test_empty_config_file_rejected(self, tmp_path: Path) -> None:
        """An empty -c file must produce a UsageError."""
        empty = tmp_path / "empty.json"
        empty.write_text("")
        result = _invoke_config(
            "--model-type",
            "bert",
            "--task",
            "fill-mask",
            "-c",
            str(empty),
        )
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    def test_invalid_json_config_file_rejected(self, tmp_path: Path) -> None:
        """Malformed JSON in -c must produce a UsageError."""
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        result = _invoke_config(
            "--model-type",
            "bert",
            "--task",
            "fill-mask",
            "-c",
            str(bad),
        )
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    def test_non_object_json_config_file_rejected(self, tmp_path: Path) -> None:
        """A JSON array in -c must be rejected (must be an object)."""
        arr = tmp_path / "array.json"
        arr.write_text("[1, 2, 3]")
        result = _invoke_config(
            "--model-type",
            "bert",
            "--task",
            "fill-mask",
            "-c",
            str(arr),
        )
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    def test_empty_shape_config_rejected(self, tmp_path: Path) -> None:
        """An empty --shape-config file must produce a UsageError."""
        empty = tmp_path / "shapes.json"
        empty.write_text("")
        result = _invoke_config(
            "--model-type",
            "bert",
            "--task",
            "fill-mask",
            "--shape-config",
            str(empty),
        )
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    def test_invalid_json_shape_config_rejected(self, tmp_path: Path) -> None:
        """Malformed --shape-config JSON must produce a UsageError."""
        bad = tmp_path / "shapes.json"
        bad.write_text("{height: 224")
        result = _invoke_config(
            "--model-type",
            "bert",
            "--task",
            "fill-mask",
            "--shape-config",
            str(bad),
        )
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    def test_non_object_shape_config_rejected(self, tmp_path: Path) -> None:
        """A JSON list in --shape-config must be rejected (must be an object)."""
        bad = tmp_path / "shapes.json"
        bad.write_text("[224, 224]")
        result = _invoke_config(
            "--model-type",
            "bert",
            "--task",
            "fill-mask",
            "--shape-config",
            str(bad),
        )
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output

    def test_module_with_onnx_file_rejected(self, onnx_model_path: Path) -> None:
        """--module is mutually exclusive with .onnx input."""
        result = _invoke_config(
            "-m",
            str(onnx_model_path),
            "--module",
            "ResNetConvLayer",
        )
        assert result.exit_code != 0
        assert "Traceback (most recent call last)" not in result.output
        combined = (result.output or "") + (str(result.exception) if result.exception else "")
        assert "module" in combined.lower()


# =============================================================================
# ONNX QDQ AUTO-DETECTION TESTS
# =============================================================================


class TestConfigOnnxQdqDetection:
    """Test config command auto-detects QDQ ONNX and sets quant=None."""

    def test_qdq_onnx_sets_quant_none(self, runner: CliRunner, tmp_path: Path) -> None:
        """Config for a QDQ ONNX file should have quant=null in output."""
        from winml.modelkit.commands.config import config

        onnx_file = tmp_path / "quantized.onnx"
        onnx_file.write_bytes(b"fake-qdq-onnx")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=True),
        ):
            result = runner.invoke(config, ["-m", str(onnx_file)])

        assert result.exit_code == 0, f"Failed: {result.output}"
        data = _extract_json(result.output)
        assert data.get("quant") is None, (
            f"Expected quant=null for QDQ model, got: {data.get('quant')}"
        )

    def test_qdq_onnx_output_confirms_no_quant(self, runner: CliRunner, tmp_path: Path) -> None:
        """Config for a QDQ ONNX should produce export=null and quant=null."""
        from winml.modelkit.commands.config import config

        onnx_file = tmp_path / "quantized.onnx"
        onnx_file.write_bytes(b"fake-qdq-onnx")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=True),
        ):
            result = runner.invoke(config, ["-m", str(onnx_file)])

        assert result.exit_code == 0, f"Failed: {result.output}"
        data = _extract_json(result.output)
        assert data.get("export") is None, "QDQ ONNX build should have export=null"
        assert data.get("quant") is None, "QDQ ONNX build should have quant=null"

    def test_qdq_overrides_device_precision(self, runner: CliRunner, tmp_path: Path) -> None:
        """QDQ detection should keep quant=null even with -d npu -p int8."""
        from winml.modelkit.commands.config import config

        onnx_file = tmp_path / "quantized.onnx"
        onnx_file.write_bytes(b"fake-qdq-onnx")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=True),
        ):
            result = runner.invoke(config, ["-m", str(onnx_file), "-d", "npu", "-p", "int8"])

        assert result.exit_code == 0, f"Failed: {result.output}"
        data = _extract_json(result.output)
        assert data.get("quant") is None, "QDQ detection should take precedence over -d npu -p int8"

    def test_non_qdq_onnx_has_default_quant(self, runner: CliRunner, tmp_path: Path) -> None:
        """Config for non-QDQ ONNX should have default quant settings (not null)."""
        from winml.modelkit.commands.config import config

        onnx_file = tmp_path / "normal.onnx"
        onnx_file.write_bytes(b"fake-onnx")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
        ):
            result = runner.invoke(config, ["-m", str(onnx_file)])

        assert result.exit_code == 0, f"Failed: {result.output}"
        data = _extract_json(result.output)
        # Default ONNX config should have quant as a dict (not null)
        assert data.get("quant") is not None, (
            f"Non-QDQ model should have default quant settings, got: {data.get('quant')}"
        )


class TestConfigExportControls:
    """Export CLI overrides (--input-specs/--export-config/--dynamic-axes) on config."""

    @staticmethod
    def _real_config():
        """Build a real WinMLBuildConfig with an export section for integration tests."""
        from winml.modelkit.config import WinMLBuildConfig
        from winml.modelkit.export import InputTensorSpec, WinMLExportConfig
        from winml.modelkit.loader import WinMLLoaderConfig

        return WinMLBuildConfig(
            loader=WinMLLoaderConfig(
                task="fill-mask", model_class="BertForMaskedLM", model_type="bert"
            ),
            export=WinMLExportConfig(
                input_tensors=[
                    InputTensorSpec(
                        name="input_ids", dtype="int64", shape=(1, 16), value_range=(0, 30522)
                    ),
                    InputTensorSpec(name="attention_mask", dtype="int64", shape=(1, 16)),
                ],
            ),
        )

    def test_help_shows_export_control_options(self, runner: CliRunner) -> None:
        from winml.modelkit.commands.config import config

        result = runner.invoke(config, ["--help"])
        assert result.exit_code == 0
        for opt in ("--input-specs", "--export-config", "--dynamic-axes"):
            assert opt in result.output, f"Expected '{opt}' in help output"

    def test_input_specs_patch_by_name_and_derive_dynamic_axes(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """--input-specs patches by name; symbolic dims re-derive dynamic_axes.

        Lets merge_export_overrides actually run and asserts on the emitted config:
        the unlisted attention_mask is preserved (with its int64 dtype), and the
        symbolic ["batch", "seq"] shape on input_ids produces dynamic_axes without
        a separate --dynamic-axes file.
        """
        from winml.modelkit.commands.config import config

        input_specs = tmp_path / "inputs.json"
        input_specs.write_text(json.dumps({"input_ids": {"shape": ["batch", "seq"]}}))
        out = tmp_path / "out.json"

        with (
            patch(
                "winml.modelkit.commands.config._resolve_composite_model_components",
                return_value=None,
            ),
            patch(
                "winml.modelkit.config.generate_hf_build_config",
                return_value=self._real_config(),
            ),
        ):
            result = runner.invoke(
                config,
                ["-m", "bert-base-uncased", "--input-specs", str(input_specs), "-o", str(out)],
            )

        assert result.exit_code == 0, result.output
        export = json.loads(out.read_text())["export"]
        names = [t["name"] for t in export["input_tensors"]]
        assert names == ["input_ids", "attention_mask"]  # unlisted input preserved
        ids = next(t for t in export["input_tensors"] if t["name"] == "input_ids")
        assert ids["dtype"] == "int64"  # preserved, not forced to float32
        axes = export["dynamic_axes"]["input_ids"]  # symbolic dims derived dynamic axes
        assert set(axes.values()) == {"batch", "seq"}

    def test_export_config_and_dynamic_axes_applied(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """--export-config and --dynamic-axes are merged onto the generated config."""
        from winml.modelkit.commands.config import config

        export_config = tmp_path / "export.json"
        export_config.write_text(json.dumps({"opset_version": 18}))
        dynamic_axes = tmp_path / "dynamic_axes.json"
        dynamic_axes.write_text(json.dumps({"input_ids": {"0": "batch"}}))
        out = tmp_path / "out.json"

        with (
            patch(
                "winml.modelkit.commands.config._resolve_composite_model_components",
                return_value=None,
            ),
            patch(
                "winml.modelkit.config.generate_hf_build_config",
                return_value=self._real_config(),
            ),
        ):
            result = runner.invoke(
                config,
                [
                    "-m",
                    "bert-base-uncased",
                    "--export-config",
                    str(export_config),
                    "--dynamic-axes",
                    str(dynamic_axes),
                    "-o",
                    str(out),
                ],
            )

        assert result.exit_code == 0, result.output
        export = json.loads(out.read_text())["export"]
        assert export["opset_version"] == 18
        assert export["dynamic_axes"]["input_ids"] == {"0": "batch"}

    def test_no_export_overrides_leaves_export_unchanged(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Without export flags the generated export section is emitted as-is."""
        from winml.modelkit.commands.config import config

        out = tmp_path / "out.json"
        with (
            patch(
                "winml.modelkit.commands.config._resolve_composite_model_components",
                return_value=None,
            ),
            patch(
                "winml.modelkit.config.generate_hf_build_config",
                return_value=self._real_config(),
            ),
        ):
            result = runner.invoke(config, ["-m", "bert-base-uncased", "-o", str(out)])

        assert result.exit_code == 0, result.output
        export = json.loads(out.read_text())["export"]
        # Concrete auto-resolved shapes -> no dynamic axes are invented.
        assert not export.get("dynamic_axes")

    def test_module_path_rejects_export_overrides(self, runner: CliRunner, tmp_path: Path) -> None:
        """--module rejects export overrides instead of fanning them onto each config."""
        from winml.modelkit.commands.config import config

        input_specs = tmp_path / "inputs.json"
        input_specs.write_text(json.dumps({"input_ids": {"shape": ["batch", "seq"]}}))

        with patch(
            "winml.modelkit.config.generate_hf_build_config",
        ) as mock_generate:
            result = runner.invoke(
                config,
                [
                    "-m",
                    "bert-base-uncased",
                    "--module",
                    "BertLayer",
                    "--input-specs",
                    str(input_specs),
                ],
            )

        assert result.exit_code != 0
        assert "--module" in result.output
        # Rejected up front, before any config generation.
        mock_generate.assert_not_called()

    def test_composite_model_rejects_export_overrides(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Composite models reject export overrides instead of fanning them out."""
        from winml.modelkit.commands.config import config

        dynamic_axes = tmp_path / "dynamic_axes.json"
        dynamic_axes.write_text(json.dumps({"input_ids": {"0": "batch"}}))

        with (
            patch(
                "winml.modelkit.commands.config._resolve_composite_model_components",
                return_value={"encoder": "feature-extraction", "decoder": "text-generation"},
            ),
            patch(
                "winml.modelkit.commands.config._generate_pipeline_configs",
            ) as mock_pipeline,
        ):
            result = runner.invoke(
                config, ["-m", "some/seq2seq", "--dynamic-axes", str(dynamic_axes)]
            )

        assert result.exit_code != 0
        assert "composite" in result.output
        mock_pipeline.assert_not_called()

    def test_composite_rejection_precedes_json_validation(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Composite models are rejected before the override JSON is loaded/validated.

        A malformed --dynamic-axes on a composite model should surface the
        composite-specific error, not a downstream "Invalid export configuration"
        from parsing — mirroring the ONNX path.
        """
        from winml.modelkit.commands.config import config

        dynamic_axes = tmp_path / "dynamic_axes.json"
        dynamic_axes.write_text(json.dumps({"input_ids": {"not-an-int": "batch"}}))

        with (
            patch(
                "winml.modelkit.commands.config._resolve_composite_model_components",
                return_value={"encoder": "feature-extraction", "decoder": "text-generation"},
            ),
            patch(
                "winml.modelkit.commands.config._generate_pipeline_configs",
            ) as mock_pipeline,
        ):
            result = runner.invoke(
                config, ["-m", "some/seq2seq", "--dynamic-axes", str(dynamic_axes)]
            )

        assert result.exit_code != 0
        assert "composite" in result.output
        assert "Invalid export configuration" not in result.output
        mock_pipeline.assert_not_called()

    def test_merge_export_overrides_rejects_export_null(self) -> None:
        """_merge_export_overrides raises when the generated config has export=null."""
        import click

        from winml.modelkit.commands.config import _merge_export_overrides
        from winml.modelkit.config import WinMLBuildConfig
        from winml.modelkit.loader import WinMLLoaderConfig

        cfg = WinMLBuildConfig(
            loader=WinMLLoaderConfig(task="fill-mask", model_class="X", model_type="bert"),
            export=None,
        )
        with pytest.raises(click.UsageError, match="export=null"):
            _merge_export_overrides(cfg, {"dynamic_axes": {"input_ids": {"0": "batch"}}})

    def test_merge_export_overrides_noop_when_empty(self) -> None:
        """_merge_export_overrides returns the config untouched when no overrides given."""
        from winml.modelkit.commands.config import _merge_export_overrides

        cfg = self._real_config()
        assert _merge_export_overrides(cfg, {}) is cfg

    def test_export_overrides_rejected_for_onnx_input(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """--dynamic-axes on a pre-exported ONNX input is a usage error."""
        from winml.modelkit.commands.config import config

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake-onnx")
        dynamic_axes = tmp_path / "dynamic_axes.json"
        dynamic_axes.write_text(json.dumps({"input": {"0": "batch"}}))

        result = runner.invoke(config, ["-m", str(onnx_file), "--dynamic-axes", str(dynamic_axes)])

        assert result.exit_code != 0
        assert "pre-exported ONNX" in result.output

    def test_onnx_rejection_precedes_json_validation(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """ONNX inputs are rejected before the override JSON is loaded/validated.

        A malformed --dynamic-axes on an ONNX input should surface the relevant
        ONNX error, not a downstream "Invalid export configuration" from parsing.
        """
        from winml.modelkit.commands.config import config

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake-onnx")
        dynamic_axes = tmp_path / "dynamic_axes.json"
        dynamic_axes.write_text(json.dumps({"input_ids": {"not-an-int": "batch"}}))

        result = runner.invoke(config, ["-m", str(onnx_file), "--dynamic-axes", str(dynamic_axes)])

        assert result.exit_code != 0
        assert "pre-exported ONNX" in result.output
        assert "Invalid export configuration" not in result.output
