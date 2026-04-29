# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for build CLI command — mock-based, no network, no actual builds.

Tests the CLI wrapper around _run_single_build() internal pipeline.
NO WinMLAutoModel involvement.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def mock_resolve_device():
    """Mock resolve_device and WinMLEPRegistry to avoid hardware detection.

    The build command calls resolve_device() for I/O and (since #540)
    WinMLEPRegistry.get_instance() for EP auto-selection when --ep is
    not specified. Both must be mocked to avoid slow DLL scanning and
    WinML SDK discovery on CI runners without WinML installed.
    """
    mock_registry = MagicMock()
    mock_registry.is_ep_available.return_value = False

    with (
        patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("npu", ["npu", "gpu", "cpu"]),
        ),
        patch(
            "winml.modelkit.session.ep_registry.WinMLEPRegistry.get_instance",
            return_value=mock_registry,
        ),
    ):
        yield


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_config_file(tmp_path: Path) -> Path:
    """Create a temporary JSON config file."""
    config = {
        "loader": {"task": "image-classification"},
        "export": {"opset_version": 17, "batch_size": 1},
        "optim": {},
        "quant": {
            "mode": "qdq",
            "samples": 10,
            "task": "image-classification",
            "model_name": "test",
        },
        "compile": {"execution_provider": "qnn"},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path


@pytest.fixture
def mock_build_api():
    """Mock _run_single_build to avoid actual pipeline execution."""
    with patch("winml.modelkit.commands.build._run_single_build", return_value=None) as mock:
        yield mock


@pytest.fixture
def mock_build_reused():
    """Mock _run_single_build returning None (reuse is handled internally)."""
    with patch("winml.modelkit.commands.build._run_single_build", return_value=None) as mock:
        yield mock


# =============================================================================
# CLI INTERFACE TESTS
# =============================================================================


class TestBuildCliInterface:
    """Test CLI flag parsing and help text."""

    def test_help_shows_all_options(self, runner: CliRunner) -> None:
        from winml.modelkit.commands.build import build

        result = runner.invoke(build, ["--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "-c" in result.output
        assert "--model" in result.output
        assert "-m" in result.output
        assert "--output-dir" in result.output
        assert "-o" in result.output
        assert "--use-cache" in result.output
        assert "--rebuild" in result.output
        assert "--no-quant" in result.output
        assert "--no-compile" in result.output
        assert "--no-optimize" in result.output
        assert "--verbose" in result.output
        assert "--no-analyze" in result.output
        assert "--max-optim-iterations" in result.output

    def test_config_required(self, runner: CliRunner) -> None:
        from winml.modelkit.commands.build import build

        result = runner.invoke(build, ["-o", "output/"])
        assert result.exit_code != 0
        assert "config" in result.output.lower() or "required" in result.output.lower()

    def test_output_or_cache_required(
        self, runner: CliRunner, sample_config_file: Path, mock_build_api
    ) -> None:
        from winml.modelkit.commands.build import build

        result = runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "test-model"],
            obj={"debug": False},
        )
        assert result.exit_code != 0
        assert "required" in result.output.lower()

    def test_mutual_exclusion(
        self, runner: CliRunner, sample_config_file: Path, mock_build_api
    ) -> None:
        from winml.modelkit.commands.build import build

        result = runner.invoke(
            build,
            [
                "-c",
                str(sample_config_file),
                "-m",
                "microsoft/resnet-50",
                "-o",
                "output/",
                "--use-cache",
            ],
            obj={"debug": False},
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()


# =============================================================================
# BUILD INVOCATION TESTS
# =============================================================================


class TestBuildInvocation:
    """Test that CLI correctly calls build_hf_model."""

    def test_basic_build(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        from winml.modelkit.commands.build import build

        output_dir = tmp_path / "out"
        result = runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "test-model", "-o", str(output_dir)],
            obj={"debug": False},
        )
        assert result.exit_code == 0, f"Build failed: {result.output}"
        assert mock_build_api.called

    def test_model_id_passed(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        from winml.modelkit.commands.build import build

        runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "microsoft/resnet-50", "-o", str(tmp_path)],
            obj={"debug": False},
        )
        call_kwargs = mock_build_api.call_args.kwargs
        assert call_kwargs["model_id"] == "microsoft/resnet-50"

    def test_model_optional_for_random_weight(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Omitting -m/--model is valid — triggers random-weight build."""
        from winml.modelkit.commands.build import build

        result = runner.invoke(
            build,
            ["-c", str(sample_config_file), "-o", str(tmp_path)],
            obj={"debug": False},
        )
        assert result.exit_code == 0
        call_kwargs = mock_build_api.call_args.kwargs
        assert call_kwargs["model_id"] is None

    def test_rebuild_passed(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        from winml.modelkit.commands.build import build

        runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "test", "-o", str(tmp_path), "--rebuild"],
            obj={"debug": False},
        )
        call_kwargs = mock_build_api.call_args.kwargs
        assert call_kwargs["rebuild"] is True

    def test_default_rebuild_false(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        from winml.modelkit.commands.build import build

        runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "test", "-o", str(tmp_path)],
            obj={"debug": False},
        )
        call_kwargs = mock_build_api.call_args.kwargs
        assert call_kwargs["rebuild"] is False


# =============================================================================
# CONFIG OVERRIDE TESTS
# =============================================================================


class TestBuildConfigOverrides:
    """Test --no-quant and --no-compile CLI overrides."""

    def test_no_quant_sets_none(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        from winml.modelkit.commands.build import build

        runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "test", "-o", str(tmp_path), "--no-quant"],
            obj={"debug": False},
        )
        config = mock_build_api.call_args.kwargs["config"]
        assert config.quant is None

    def test_no_compile_sets_none(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        from winml.modelkit.commands.build import build

        runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "test", "-o", str(tmp_path), "--no-compile"],
            obj={"debug": False},
        )
        config = mock_build_api.call_args.kwargs["config"]
        assert config.compile is None

    def test_no_quant_no_compile_together(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        from winml.modelkit.commands.build import build

        runner.invoke(
            build,
            [
                "-c",
                str(sample_config_file),
                "-m",
                "t",
                "-o",
                str(tmp_path),
                "--no-quant",
                "--no-compile",
            ],
            obj={"debug": False},
        )
        config = mock_build_api.call_args.kwargs["config"]
        assert config.quant is None
        assert config.compile is None


# =============================================================================
# REUSE REPORTING TESTS
# =============================================================================


class TestBuildReuse:
    """Test reuse message when artifact already exists."""

    def test_reuse_message(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_reused: MagicMock,
        tmp_path: Path,
    ) -> None:
        from winml.modelkit.commands.build import build

        result = runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "test", "-o", str(tmp_path)],
            obj={"debug": False},
        )
        assert result.exit_code == 0
        # Reuse detection is handled inside _run_single_build; verify it was called
        assert mock_build_reused.called


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestBuildErrors:
    """Test error handling."""

    def test_invalid_config_json(self, runner: CliRunner, tmp_path: Path) -> None:
        from winml.modelkit.commands.build import build

        bad_config = tmp_path / "bad.json"
        bad_config.write_text("{ not valid }")

        result = runner.invoke(
            build,
            ["-c", str(bad_config), "-m", "test", "-o", str(tmp_path / "out")],
            obj={"debug": False},
        )
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output

    def test_empty_config_file(self, runner: CliRunner, tmp_path: Path) -> None:
        from winml.modelkit.commands.build import build

        empty = tmp_path / "empty.json"
        empty.write_text("")

        result = runner.invoke(
            build,
            ["-c", str(empty), "-m", "test", "-o", str(tmp_path / "out")],
            obj={"debug": False},
        )
        assert result.exit_code != 0
        assert "empty" in result.output.lower()

    def test_build_failure_reported(
        self, runner: CliRunner, sample_config_file: Path, tmp_path: Path
    ) -> None:
        from winml.modelkit.commands.build import build

        with patch("winml.modelkit.commands.build._run_single_build") as mock:
            mock.side_effect = RuntimeError("ONNX export failed")

            result = runner.invoke(
                build,
                ["-c", str(sample_config_file), "-m", "test", "-o", str(tmp_path / "out")],
                obj={"debug": False},
            )
            assert result.exit_code != 0
            assert "Build failed" in result.output

    def test_value_error_becomes_usage_error(
        self, runner: CliRunner, sample_config_file: Path, tmp_path: Path
    ) -> None:
        from winml.modelkit.commands.build import build

        with patch("winml.modelkit.commands.build._run_single_build") as mock:
            mock.side_effect = ValueError("Invalid config")

            result = runner.invoke(
                build,
                ["-c", str(sample_config_file), "-m", "test", "-o", str(tmp_path / "out")],
                obj={"debug": False},
            )
            assert result.exit_code != 0
            assert "Invalid config" in result.output


# =============================================================================
# EP / DEVICE FLAG TESTS
# =============================================================================


class TestBuildEpDevice:
    """Test --ep and --device flags are passed to API."""

    def test_ep_flag_passed(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        from winml.modelkit.commands.build import build

        runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "test", "-o", str(tmp_path), "--ep", "qnn"],
            obj={"debug": False},
        )
        call_kwargs = mock_build_api.call_args.kwargs
        assert call_kwargs["ep"] == "qnn"

    def test_device_flag_passed(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        from winml.modelkit.commands.build import build

        runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "test", "-o", str(tmp_path), "--device", "NPU"],
            obj={"debug": False},
        )
        call_kwargs = mock_build_api.call_args.kwargs
        assert call_kwargs["device"] == "NPU"


# =============================================================================
# VERBOSE / DEBUG TESTS
# =============================================================================


class TestBuildVerbose:
    """Test verbose/debug behavior."""

    def test_verbose_flag(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        from winml.modelkit.commands.build import build

        result = runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "test", "-o", str(tmp_path), "-v"],
            obj={"debug": False},
        )
        assert result.exit_code == 0

    def test_debug_inherited(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        from winml.modelkit.commands.build import build

        result = runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "test", "-o", str(tmp_path)],
            obj={"debug": True},
        )
        assert result.exit_code == 0


# =============================================================================
# ONNX AUTO-DETECTION TESTS
# =============================================================================


class TestBuildOnnxAutoDetect:
    """Test auto-detection of ONNX vs HF model input."""

    def test_build_auto_detect_onnx_file(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        tmp_path: Path,
    ) -> None:
        """When -m points to an existing .onnx file, dispatches to _build_onnx_pipeline."""
        from winml.modelkit.commands.build import build

        # Create a fake .onnx file on disk
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake-onnx-data")

        output_dir = tmp_path / "out"

        with patch(
            "winml.modelkit.commands.build._build_onnx_pipeline", return_value=[]
        ) as mock_onnx:
            result = runner.invoke(
                build,
                ["-c", str(sample_config_file), "-m", str(onnx_file), "-o", str(output_dir)],
                obj={"debug": False},
            )
            assert result.exit_code == 0, f"Build failed: {result.output}"
            mock_onnx.assert_called_once()
            call_kwargs = mock_onnx.call_args.kwargs
            assert call_kwargs["onnx_path"] == onnx_file

    def test_build_auto_detect_hf_model(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When -m is a HF model ID (not .onnx), dispatches to _run_single_build."""
        from winml.modelkit.commands.build import build

        output_dir = tmp_path / "out"
        result = runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "microsoft/resnet-50", "-o", str(output_dir)],
            obj={"debug": False},
        )
        assert result.exit_code == 0, f"Build failed: {result.output}"
        assert mock_build_api.called
        call_kwargs = mock_build_api.call_args.kwargs
        assert call_kwargs["model_id"] == "microsoft/resnet-50"

    def test_build_onnx_suffix_but_not_exists_uses_hf(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        """An .onnx path that doesn't exist falls through to HF path."""
        from winml.modelkit.commands.build import build

        output_dir = tmp_path / "out"
        result = runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "nonexistent.onnx", "-o", str(output_dir)],
            obj={"debug": False},
        )
        # _is_onnx_file checks suffix AND exists(); nonexistent.onnx
        # falls through to HF path since the file doesn't exist on disk
        assert result.exit_code == 0, f"Build failed: {result.output}"
        assert mock_build_api.called
        call_kwargs = mock_build_api.call_args.kwargs
        assert call_kwargs["model_id"] == "nonexistent.onnx"


# =============================================================================
# ANALYZER CONTROL TESTS
# =============================================================================


class TestBuildAnalyzerControl:
    """Test --no-analyze and --max-optim-iterations flags."""

    def test_no_analyze_flag_in_help(self, runner: CliRunner) -> None:
        from winml.modelkit.commands.build import build

        result = runner.invoke(build, ["--help"])
        assert "--no-analyze" in result.output

    def test_max_optim_iterations_in_help(self, runner: CliRunner) -> None:
        from winml.modelkit.commands.build import build

        result = runner.invoke(build, ["--help"])
        assert "--max-optim-iterations" in result.output

    def test_no_analyze_sets_zero_iterations(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        from winml.modelkit.commands.build import build

        runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "test", "-o", str(tmp_path), "--no-analyze"],
            obj={"debug": False},
        )
        extra = mock_build_api.call_args.kwargs["extra_kwargs"]
        assert extra.get("hack_max_optim_iterations") == 0

    def test_max_optim_iterations_passed(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        from winml.modelkit.commands.build import build

        runner.invoke(
            build,
            [
                "-c",
                str(sample_config_file),
                "-m",
                "test",
                "-o",
                str(tmp_path),
                "--max-optim-iterations",
                "5",
            ],
            obj={"debug": False},
        )
        extra = mock_build_api.call_args.kwargs["extra_kwargs"]
        assert extra.get("hack_max_optim_iterations") == 5

    def test_no_analyze_takes_precedence_over_max_iterations(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--no-analyze takes precedence when both flags are specified."""
        from winml.modelkit.commands.build import build

        runner.invoke(
            build,
            [
                "-c",
                str(sample_config_file),
                "-m",
                "test",
                "-o",
                str(tmp_path),
                "--no-analyze",
                "--max-optim-iterations",
                "5",
            ],
            obj={"debug": False},
        )
        extra = mock_build_api.call_args.kwargs["extra_kwargs"]
        assert extra.get("hack_max_optim_iterations") == 0

    def test_default_no_analyzer_kwargs(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        from winml.modelkit.commands.build import build

        runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "test", "-o", str(tmp_path)],
            obj={"debug": False},
        )
        extra = mock_build_api.call_args.kwargs["extra_kwargs"]
        assert "hack_max_optim_iterations" not in extra


# =============================================================================
# --no-optimize FLAG TESTS
# =============================================================================


class TestBuildNoOptimizeFlag:
    """Test --no-optimize CLI flag."""

    def test_help_shows_no_optimize(self, runner: CliRunner) -> None:
        from winml.modelkit.commands.build import build

        result = runner.invoke(build, ["--help"])
        assert "--no-optimize" in result.output

    def test_no_optimize_passed_to_onnx_build(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        tmp_path: Path,
    ) -> None:
        """--no-optimize passes skip_optimize=True via extra_kwargs."""
        from winml.modelkit.commands.build import build

        # Create a fake .onnx file for ONNX path detection
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_text("fake")

        with patch(
            "winml.modelkit.commands.build._run_single_build", return_value=None
        ) as mock_build:
            result = runner.invoke(
                build,
                [
                    "-c",
                    str(sample_config_file),
                    "-m",
                    str(onnx_file),
                    "-o",
                    str(tmp_path / "out"),
                    "--no-optimize",
                ],
                obj={"debug": False},
            )

        assert result.exit_code == 0, f"Failed: {result.output}"
        extra = mock_build.call_args.kwargs["extra_kwargs"]
        assert extra.get("skip_optimize") is True

    def test_no_optimize_passed_to_hf_build(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        tmp_path: Path,
        mock_build_api: MagicMock,
    ) -> None:
        """--no-optimize passes skip_optimize=True via extra_kwargs."""
        from winml.modelkit.commands.build import build

        result = runner.invoke(
            build,
            [
                "-c",
                str(sample_config_file),
                "-m",
                "test-model",
                "-o",
                str(tmp_path),
                "--no-optimize",
            ],
            obj={"debug": False},
        )

        assert result.exit_code == 0, f"Failed: {result.output}"
        extra = mock_build_api.call_args.kwargs["extra_kwargs"]
        assert extra.get("skip_optimize") is True

    def test_no_optimize_default_not_present(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        tmp_path: Path,
        mock_build_api: MagicMock,
    ) -> None:
        """Without --no-optimize, skip_optimize is not in extra_kwargs."""
        from winml.modelkit.commands.build import build

        runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "test-model", "-o", str(tmp_path)],
            obj={"debug": False},
        )

        extra = mock_build_api.call_args.kwargs["extra_kwargs"]
        assert "skip_optimize" not in extra


# =============================================================================
# _run_compile_stage UNIT TESTS
# =============================================================================


class TestRunCompileStageNoOutput:
    """Test _run_compile_stage EP-context skipping and output validation."""

    @patch("winml.modelkit.compiler.compile_onnx")
    def test_dml_skips_compile_entirely(
        self,
        mock_compile: MagicMock,
        tmp_path: Path,
    ) -> None:
        """EPs with enable_ep_context=False (DML, CPU) skip compile_onnx entirely."""
        from winml.modelkit.commands.build import _run_compile_stage
        from winml.modelkit.compiler.configs import WinMLCompileConfig
        from winml.modelkit.config import WinMLBuildConfig

        input_path = tmp_path / "quantized.onnx"
        input_path.write_bytes(b"dummy")
        compiled_path = tmp_path / "compiled.onnx"

        config = WinMLBuildConfig(compile=WinMLCompileConfig.for_dml())
        timings: list[tuple[str, float | None]] = []

        result = _run_compile_stage(
            config=config,
            current_path=input_path,
            compiled_path=compiled_path,
            stage_timings=timings,
        )

        mock_compile.assert_not_called()
        assert result == input_path

    @patch("winml.modelkit.utils.console.get_onnx_graph_summary")
    @patch("winml.modelkit.utils.console.StageLive")
    @patch("winml.modelkit.compiler.compile_onnx")
    def test_raises_when_ep_context_expected_but_missing(
        self,
        mock_compile: MagicMock,
        mock_stage_live: MagicMock,
        mock_graph_summary: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When enable_ep_context=True and compile succeeds but file is absent, raise."""
        from winml.modelkit.commands.build import _run_compile_stage
        from winml.modelkit.compiler.configs import WinMLCompileConfig
        from winml.modelkit.compiler.result import CompileResult
        from winml.modelkit.config import WinMLBuildConfig

        mock_compile.return_value = CompileResult(success=True, output_path=None)
        mock_stage_live.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_stage_live.return_value.__exit__ = MagicMock(return_value=False)

        input_path = tmp_path / "quantized.onnx"
        input_path.write_bytes(b"dummy")
        compiled_path = tmp_path / "compiled.onnx"  # Does NOT exist

        config = WinMLBuildConfig(compile=WinMLCompileConfig.for_qnn())
        timings: list[tuple[str, float | None]] = []

        with pytest.raises(RuntimeError, match="output not found"):
            _run_compile_stage(
                config=config,
                current_path=input_path,
                compiled_path=compiled_path,
                stage_timings=timings,
            )

    @patch("winml.modelkit.utils.console.get_onnx_graph_summary")
    @patch("winml.modelkit.utils.console.StageLive")
    @patch("winml.modelkit.compiler.compile_onnx")
    @patch("winml.modelkit.onnx.external_data.copy_onnx_model")
    def test_returns_compiled_path_when_file_exists(
        self,
        mock_copy: MagicMock,
        mock_compile: MagicMock,
        mock_stage_live: MagicMock,
        mock_graph_summary: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When compile produces an output file, current_path should update."""
        from winml.modelkit.commands.build import _run_compile_stage
        from winml.modelkit.compiler.configs import WinMLCompileConfig
        from winml.modelkit.compiler.result import CompileResult
        from winml.modelkit.config import WinMLBuildConfig

        input_path = tmp_path / "quantized.onnx"
        input_path.write_bytes(b"dummy")
        compiled_path = tmp_path / "compiled.onnx"
        compiled_path.write_bytes(b"compiled_model")  # File EXISTS

        mock_compile.return_value = CompileResult(
            success=True,
            output_path=str(compiled_path),
        )
        mock_stage_live.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_stage_live.return_value.__exit__ = MagicMock(return_value=False)
        mock_graph_summary.return_value = {"op_counts": {"EPContext": 1}}

        config = WinMLBuildConfig(compile=WinMLCompileConfig.for_qnn())
        timings: list[tuple[str, float | None]] = []

        result = _run_compile_stage(
            config=config,
            current_path=input_path,
            compiled_path=compiled_path,
            stage_timings=timings,
        )

        # current_path should be updated to compiled_path
        assert result == compiled_path
