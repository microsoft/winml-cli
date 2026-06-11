# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for compile CLI command.

Tests the compile command CLI interface using Click's CliRunner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest


if TYPE_CHECKING:
    from pathlib import Path
from click.testing import CliRunner

from winml.modelkit.cli import main


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


class TestCompileCommand:
    """Test compile command functionality."""

    def test_compile_help_shows_options(self, runner: CliRunner) -> None:
        """Test compile --help shows all expected options."""
        result = runner.invoke(main, ["compile", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--device" in result.output
        assert "--compiler" in result.output
        assert "--qnn-sdk-root" in result.output

    def test_compile_requires_model_unless_list(self, runner: CliRunner) -> None:
        """Test compile requires --model unless --list is provided.

        Key branch: if list: return early; else if model is None: raise UsageError
        """
        # Without --model should fail
        result = runner.invoke(main, ["compile"])
        assert result.exit_code != 0
        assert "model" in result.output.lower() or "missing" in result.output.lower()

        # With --list should succeed without --model
        result = runner.invoke(main, ["compile", "--list"])
        assert result.exit_code == 0

    @patch("winml.modelkit.compiler.compile_onnx")
    def test_compile_compiler_qairt_sets_ep_config(
        self,
        mock_compile_onnx: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test --compiler qairt sets config.ep_config.compiler correctly.

        Key config: config.ep_config.compiler = compiler
        """
        model_path = tmp_path / "model.onnx"
        self._create_simple_onnx(model_path)

        # Create mock SDK root
        sdk_root = tmp_path / "qairt_sdk"
        sdk_root.mkdir()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_path = tmp_path / "output.onnx"
        mock_result.compile_time = 1.0
        mock_result.total_time = 1.5
        mock_compile_onnx.return_value = mock_result

        result = runner.invoke(
            main,
            [
                "compile",
                "--model",
                str(model_path),
                "--compiler",
                "qairt",
                "--qnn-sdk-root",
                str(sdk_root),
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_compile_onnx.call_args.kwargs
        config = call_kwargs["config"]
        assert config.ep_config.compiler == "qairt"
        assert config.ep_config.qnn_sdk_root == sdk_root

    def test_compile_help_shows_output_option(self, runner: CliRunner) -> None:
        """Test compile --help shows -o/--output option."""
        result = runner.invoke(main, ["compile", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output or "-o" in result.output

    @patch("winml.modelkit.compiler.compile_onnx")
    def test_compile_output_passes_file_path(
        self,
        mock_compile_onnx: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test -o passes a file path to compile_onnx as output_path.

        Before the fix, -o was not a recognized option and Click raised an error.
        """
        model_path = tmp_path / "model.onnx"
        self._create_simple_onnx(model_path)
        output_file = tmp_path / "compiled.onnx"

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_path = output_file
        mock_result.compile_time = 1.0
        mock_result.total_time = 1.5
        mock_compile_onnx.return_value = mock_result

        result = runner.invoke(
            main,
            [
                "compile",
                "-m",
                str(model_path),
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code == 0, result.output
        assert mock_compile_onnx.called
        # output_path should be the file path, not a directory
        call_kwargs = mock_compile_onnx.call_args.kwargs
        assert call_kwargs["output_path"] == output_file

    @patch("winml.modelkit.compiler.compile_onnx")
    def test_compile_output_takes_precedence_over_output_dir(
        self,
        mock_compile_onnx: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test -o takes precedence over --output-dir when both are given."""
        model_path = tmp_path / "model.onnx"
        self._create_simple_onnx(model_path)
        output_file = tmp_path / "compiled.onnx"
        output_dir = tmp_path / "some_dir"

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_path = output_file
        mock_result.compile_time = 1.0
        mock_result.total_time = 1.5
        mock_compile_onnx.return_value = mock_result

        result = runner.invoke(
            main,
            [
                "compile",
                "-m",
                str(model_path),
                "-o",
                str(output_file),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        # -o should win over --output-dir
        call_kwargs = mock_compile_onnx.call_args.kwargs
        assert call_kwargs["output_path"] == output_file

    def test_gpu_device_raises_unsupported_error(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test --device gpu raises an unsupported-EPContext error.

        GPU maps to the DML provider which has enable_ep_context=False.
        Before the fix the error message listed 'dml' as a supported example,
        which was misleading because DML never produces EPContext models.
        """
        model_path = tmp_path / "model.onnx"
        self._create_simple_onnx(model_path)

        result = runner.invoke(main, ["compile", "-m", str(model_path), "--device", "gpu"])

        assert result.exit_code != 0
        assert "does not support EPContext compilation" in result.output
        assert "(e.g. qnn, dml, openvino)" not in result.output
        assert "(e.g. qnn, openvino)" in result.output

    def test_ep_dml_raises_unsupported_error(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test --ep dml raises an unsupported-EPContext error.

        DML has enable_ep_context=False so for_provider('dml') returns None,
        which triggers the error regardless of whether dml was reached via
        --device gpu or --ep dml explicitly.
        """
        model_path = tmp_path / "model.onnx"
        self._create_simple_onnx(model_path)

        result = runner.invoke(
            main, ["compile", "-m", str(model_path), "--device", "gpu", "--ep", "dml"]
        )

        assert result.exit_code != 0
        # Error line is "Provider 'DmlExecutionProvider' does not support …"
        assert "DmlExecutionProvider" in result.output
        assert "(e.g. qnn, dml, openvino)" not in result.output
        assert "(e.g. qnn, openvino)" in result.output

    @patch("winml.modelkit.compiler.compile_onnx")
    def test_ep_nvtensorrtrtx_propagates_gpu_device_type(
        self,
        mock_compile_onnx: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test --device gpu --ep nvtensorrtrtx no longer injects provider_options[device_type]."""
        model_path = tmp_path / "model.onnx"
        self._create_simple_onnx(model_path)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_path = tmp_path / "model_gpu_ctx.onnx"
        mock_result.compile_time = 1.0
        mock_result.total_time = 1.5
        mock_compile_onnx.return_value = mock_result

        result = runner.invoke(
            main,
            ["compile", "-m", str(model_path), "--device", "gpu", "--ep", "nvtensorrtrtx"],
        )

        assert result.exit_code == 0, result.output
        config = mock_compile_onnx.call_args.kwargs["config"]
        assert config.ep_config.provider == "nv_tensorrt_rtx"
        assert "device_type" not in config.ep_config.provider_options

    def test_cpu_device_raises_unsupported_error(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test --device cpu raises an unsupported-EPContext error.

        CPU never produces EPContext models, so it is rejected at the same
        config-is-None gate as DML.
        """
        model_path = tmp_path / "model.onnx"
        self._create_simple_onnx(model_path)

        result = runner.invoke(main, ["compile", "-m", str(model_path), "--device", "cpu"])

        assert result.exit_code != 0
        assert "does not support EPContext compilation" in result.output

    @patch("winml.modelkit.compiler.compile_onnx")
    def test_device_label_reflects_device_flag(
        self,
        mock_compile_onnx: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test the Device line in output shows the --device flag, not the EP-inferred device.

        Before the fix the output used an EP-to-device lookup that returned
        'npu' for --device gpu --ep qnn (qnn's canonical device), so the
        displayed device contradicted what the user passed. The fix drops
        the lookup and always prints the user-supplied device.
        """
        model_path = tmp_path / "model.onnx"
        self._create_simple_onnx(model_path)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_path = tmp_path / "model_gpu_ctx.onnx"
        mock_result.compile_time = 1.0
        mock_result.total_time = 1.5
        mock_compile_onnx.return_value = mock_result

        result = runner.invoke(
            main,
            ["compile", "-m", str(model_path), "--device", "gpu", "--ep", "qnn"],
        )

        assert result.exit_code == 0, result.output
        assert "Device:" in result.output
        # Must show "gpu" (the flag value), not "npu" (qnn's canonical device).
        device_line = next(line for line in result.output.splitlines() if "Device:" in line)
        assert "gpu" in device_line
        assert "npu" not in device_line

    @patch("winml.modelkit.compiler.compile_onnx")
    def test_compile_device_propagates_to_provider_options(
        self,
        mock_compile_onnx: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test --device npu --ep qnn stores device in both provider_options and ep_config.

        device_type in provider_options ensures NPU and GPU builds get different
        cache keys. device in ep_config enables compile stage to align EPContext
        filenames with the actual runtime-resolved device (e.g., model_npu_ctx.onnx).
        """
        model_path = tmp_path / "model.onnx"
        self._create_simple_onnx(model_path)

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_path = tmp_path / "model_npu_ctx.onnx"
        mock_result.compile_time = 1.0
        mock_result.total_time = 1.5
        mock_compile_onnx.return_value = mock_result

        result = runner.invoke(
            main,
            ["compile", "-m", str(model_path), "--device", "npu", "--ep", "qnn"],
        )

        assert result.exit_code == 0, result.output
        config = mock_compile_onnx.call_args.kwargs["config"]
        assert config.ep_config.provider_options.get("device_type") == "NPU"
        assert config.ep_config.device == "npu"

    def test_multiple_models_reject_output_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Multiple -m inputs with -o/--output (a file) are rejected: use --output-dir.

        Several models share one EP context and are written by filename into a
        directory, so a single output file path is ambiguous.
        """
        m1 = tmp_path / "m1.onnx"
        m2 = tmp_path / "m2.onnx"
        self._create_simple_onnx(m1)
        self._create_simple_onnx(m2)
        out_file = tmp_path / "out.onnx"

        result = runner.invoke(main, ["compile", "-m", str(m1), "-m", str(m2), "-o", str(out_file)])

        assert result.exit_code != 0
        assert "output-dir" in result.output.lower(), result.output

    def test_multiple_models_require_output_dir(self, runner: CliRunner, tmp_path: Path) -> None:
        """Multiple -m inputs with neither -o nor --output-dir are rejected.

        --output-dir is mandatory for multi-model compiles (the compiled models are
        written by filename into that directory).
        """
        m1 = tmp_path / "m1.onnx"
        m2 = tmp_path / "m2.onnx"
        self._create_simple_onnx(m1)
        self._create_simple_onnx(m2)

        result = runner.invoke(main, ["compile", "-m", str(m1), "-m", str(m2)])

        assert result.exit_code != 0
        assert "output-dir" in result.output.lower(), result.output

    @patch("winml.modelkit.compiler.compile_multiple_onnx")
    def test_multiple_models_with_output_dir_calls_compile_multiple(
        self,
        mock_compile_multiple: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Multiple -m inputs with --output-dir compile via compile_multiple_onnx."""
        m1 = tmp_path / "m1.onnx"
        m2 = tmp_path / "m2.onnx"
        self._create_simple_onnx(m1)
        self._create_simple_onnx(m2)
        out_dir = tmp_path / "out"

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_path = out_dir / "m2_ctx.onnx"
        mock_result.compile_time = 1.0
        mock_result.total_time = 1.5
        mock_compile_multiple.return_value = [mock_result, mock_result]

        result = runner.invoke(
            main,
            [
                "compile",
                "-m",
                str(m1),
                "-m",
                str(m2),
                "--device",
                "npu",
                "--ep",
                "qnn",
                "--output-dir",
                str(out_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        assert mock_compile_multiple.called
        call_args = mock_compile_multiple.call_args
        # First positional arg is the ordered list of input models.
        passed_models = call_args.args[0]
        assert [str(m) for m in passed_models] == [str(m1), str(m2)]
        # Second positional arg is the output target — the --output-dir directory.
        assert call_args.args[1] == out_dir
        # Backend is carried on the config; defaults to False (model_compiler).
        assert call_args.args[2].use_inference_session is False

    @patch("winml.modelkit.compiler.compile_multiple_onnx")
    def test_use_inference_session_flag_overrides_config(
        self,
        mock_compile_multiple: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """--use-inference-session sets config.use_inference_session=True for the compile."""
        m1 = tmp_path / "m1.onnx"
        m2 = tmp_path / "m2.onnx"
        self._create_simple_onnx(m1)
        self._create_simple_onnx(m2)
        out_dir = tmp_path / "out"

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output_path = out_dir / "m2_ctx.onnx"
        mock_result.compile_time = 1.0
        mock_result.total_time = 1.5
        mock_compile_multiple.return_value = [mock_result, mock_result]

        result = runner.invoke(
            main,
            [
                "compile",
                "-m",
                str(m1),
                "-m",
                str(m2),
                "--device",
                "npu",
                "--ep",
                "qnn",
                "--output-dir",
                str(out_dir),
                "--use-inference-session",
            ],
        )

        assert result.exit_code == 0, result.output
        # The CLI flag is applied onto the config that drives compilation.
        assert mock_compile_multiple.call_args.args[2].use_inference_session is True

    def _create_simple_onnx(self, path: Path) -> None:
        """Create a simple ONNX model for testing."""
        import onnx
        from onnx import TensorProto, helper

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])  # noqa: N806
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])  # noqa: N806
        node = helper.make_node("Identity", ["X"], ["Y"])
        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(path))
