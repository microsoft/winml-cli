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
        assert "--quantize" in result.output
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
    def test_compile_no_quantize_is_noop(
        self,
        mock_compile_onnx: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test --no-quantize is accepted but has no effect on compile config.

        Quantization is now handled by quant module, not compiler.
        The flag is kept for backward compat CLI but is a no-op.
        """
        model_path = tmp_path / "model.onnx"
        self._create_simple_onnx(model_path)

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
                "--no-quantize",
            ],
        )

        assert result.exit_code == 0
        assert mock_compile_onnx.called

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
