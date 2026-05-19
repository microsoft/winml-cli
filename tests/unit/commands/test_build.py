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


_DEVICE_TO_EPS = {
    "npu": ["QNNExecutionProvider"],
    "gpu": ["DmlExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
}


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
            "winml.modelkit.sysinfo.resolve_eps",
            side_effect=lambda device: list(_DEVICE_TO_EPS.get(device, [])),
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


@pytest.fixture
def mock_run_single_build():
    """Mock the heavy ``_run_single_build`` so flag plumbing can be inspected."""
    with patch("winml.modelkit.commands.build._run_single_build", return_value=None) as mock:
        yield mock


def _make_minimal_config_file(
    tmp_path: Path,
    task: str = "image-classification",
    *,
    name: str = "config.json",
    compile_section: dict | None = None,
) -> str:
    """Create a minimal WinMLBuildConfig JSON for CLI validation tests."""
    config: dict = {
        "loader": {"task": task},
        "export": {"opset_version": 17, "batch_size": 1},
        "optim": {},
        "quant": None,
        "compile": compile_section,
    }
    config_path = tmp_path / name
    config_path.write_text(json.dumps(config))
    return str(config_path)


def _invoke(args: list[str], *, debug: bool = False):
    """Invoke the build command with a fresh CliRunner.

    ``catch_exceptions=False`` is intentionally not used here so
    validation failures still come back as normal Click results.
    """
    from winml.modelkit.commands.build import build

    runner = CliRunner()
    return runner.invoke(build, args, obj={"debug": debug})


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
# CLI VALIDATION / FLAG PLUMBING REGRESSIONS
# =============================================================================


class TestBuildArgValidation:
    """Argument-validation errors must surface as UsageError (exit != 0)."""

    def test_missing_config_required(self, tmp_path: Path):
        """``-c/--config`` is required by Click."""
        result = _invoke(["-o", str(tmp_path / "out")])
        assert result.exit_code != 0
        output = result.output.lower()
        assert "config" in output or "required" in output

    def test_config_file_does_not_exist(self, tmp_path: Path):
        """``click.Path(exists=True)`` rejects non-existent config files."""
        missing = tmp_path / "no_such_config.json"
        result = _invoke(["-c", str(missing), "-o", str(tmp_path / "out")])
        assert result.exit_code != 0
        assert "no_such_config.json" in result.output or "exist" in result.output.lower()

    def test_missing_output_and_cache(self, tmp_path: Path):
        """One of ``--output-dir`` or ``--use-cache`` must be provided."""
        cfg = _make_minimal_config_file(tmp_path)
        result = _invoke(["-c", cfg, "-m", "microsoft/resnet-50"])
        assert result.exit_code != 0
        assert "required" in result.output.lower()

    def test_output_dir_and_use_cache_mutually_exclusive(self, tmp_path: Path):
        """``--output-dir`` and ``--use-cache`` are mutually exclusive."""
        cfg = _make_minimal_config_file(tmp_path)
        result = _invoke(
            [
                "-c",
                cfg,
                "-m",
                "microsoft/resnet-50",
                "-o",
                str(tmp_path / "out"),
                "--use-cache",
            ]
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()

    def test_invalid_json_config(self, tmp_path: Path):
        """Malformed JSON config surfaces ``Invalid JSON in config:``."""
        bad = tmp_path / "bad.json"
        bad.write_text("{ not valid }")
        result = _invoke(["-c", str(bad), "-m", "x", "-o", str(tmp_path / "out")])
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output

    def test_empty_config_file(self, tmp_path: Path):
        """An empty config file is rejected with a clear message."""
        empty = tmp_path / "empty.json"
        empty.write_text("")
        result = _invoke(["-c", str(empty), "-m", "x", "-o", str(tmp_path / "out")])
        assert result.exit_code != 0
        assert "empty" in result.output.lower()

    def test_config_must_be_object_or_array(self, tmp_path: Path):
        """A JSON scalar config (e.g. a string) is rejected."""
        scalar = tmp_path / "scalar.json"
        scalar.write_text('"not an object"')
        result = _invoke(["-c", str(scalar), "-m", "x", "-o", str(tmp_path / "out")])
        assert result.exit_code != 0
        assert "object" in result.output.lower() or "array" in result.output.lower()

    def test_compile_flag_without_compile_section(self, tmp_path: Path):
        """``--compile`` on a config without a compile section is a UsageError."""
        cfg = _make_minimal_config_file(tmp_path, compile_section=None)
        result = _invoke(["-c", cfg, "-m", "x", "-o", str(tmp_path / "out"), "--compile"])
        assert result.exit_code != 0
        assert "compile" in result.output.lower()

    def test_use_cache_requires_loader_task(self, tmp_path: Path):
        """``--use-cache`` without a ``loader.task`` in config errors out."""
        cfg_path = tmp_path / "no_task.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "loader": {},
                    "export": {"opset_version": 17, "batch_size": 1},
                    "optim": {},
                    "quant": None,
                    "compile": None,
                }
            )
        )
        result = _invoke(["-c", str(cfg_path), "-m", "microsoft/resnet-50", "--use-cache"])
        assert result.exit_code != 0
        assert "loader.task" in result.output

    def test_module_mode_rejects_use_cache(self, tmp_path: Path):
        """``--use-cache`` on an array config hits the module-mode-specific error."""
        arr_path = tmp_path / "modules.json"
        arr_path.write_text(
            json.dumps(
                [
                    {
                        "loader": {
                            "task": "image-classification",
                            "model_type": "resnet",
                            "module_path": "layer1",
                        },
                        "export": {"opset_version": 17, "batch_size": 1},
                        "optim": {},
                        "quant": None,
                        "compile": None,
                    }
                ]
            )
        )
        result = _invoke(["-c", str(arr_path), "--use-cache"])
        assert result.exit_code != 0
        assert (
            "--use-cache is not supported for module mode (array config). "
            "Use --output-dir instead." in result.output
        )

    def test_module_array_non_object_entry(self, tmp_path: Path):
        """Module config entries must be JSON objects."""
        arr_path = tmp_path / "bad_modules.json"
        arr_path.write_text(json.dumps([{"loader": {"task": "x"}}, "not-an-object"]))
        result = _invoke(["-c", str(arr_path), "-o", str(tmp_path / "out")])
        assert result.exit_code != 0
        assert "object" in result.output.lower()

    def test_incompatible_loader_task_for_model(self, tmp_path: Path):
        """``config.loader.task`` incompatible with --model must fail upfront.

        Verifies the upfront task↔architecture validator: a config with
        ``loader.task='text-generation'`` paired with a vision model
        like ``microsoft/resnet-50`` must surface a one-line UsageError
        that names the offending task, the model id, the resolved
        architecture, and the supported task list — BEFORE any download
        or Setup banner. See issue: "Config loader.task is not validated
        against --model".
        """
        cfg = _make_minimal_config_file(tmp_path, task="text-generation")

        mock_cfg = MagicMock()
        mock_cfg.model_type = "resnet"
        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=mock_cfg),
            patch(
                "winml.modelkit.loader.task.get_supported_tasks",
                return_value=["image-classification", "image-feature-extraction"],
            ),
            patch(
                "winml.modelkit.loader.task.normalize_task",
                side_effect=lambda t: t,
            ),
        ):
            result = _invoke(
                [
                    "-c",
                    cfg,
                    "-m",
                    "microsoft/resnet-50",
                    "-o",
                    str(tmp_path / "out"),
                ]
            )
        assert result.exit_code != 0
        # One-line actionable error naming all relevant pieces:
        assert "text-generation" in result.output
        assert "microsoft/resnet-50" in result.output
        assert "resnet" in result.output
        assert "image-classification" in result.output

    def test_compatible_loader_task_passes_validation(
        self,
        tmp_path: Path,
        mock_build_api: MagicMock,
    ):
        """A ``loader.task`` supported by the model architecture must NOT block the build.

        Regression guard: the upfront validator must never reject a
        legitimate task/model pair.
        """
        cfg = _make_minimal_config_file(tmp_path, task="image-classification")

        mock_cfg = MagicMock()
        mock_cfg.model_type = "resnet"
        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=mock_cfg),
            patch(
                "winml.modelkit.loader.task.get_supported_tasks",
                return_value=["image-classification", "image-feature-extraction"],
            ),
        ):
            result = _invoke(
                [
                    "-c",
                    cfg,
                    "-m",
                    "microsoft/resnet-50",
                    "-o",
                    str(tmp_path / "out"),
                ]
            )
        assert result.exit_code == 0, f"Build failed: {result.output}"
        assert mock_build_api.called

    def test_help_lists_all_options(self):
        """``--help`` must surface every behavior-bearing option."""
        result = _invoke(["--help"])
        assert result.exit_code == 0
        for flag in [
            "--config",
            "-c",
            "--model",
            "-m",
            "--output-dir",
            "-o",
            "--use-cache",
            "--rebuild",
            "--no-quant",
            "--no-compile",
            "--compile",
            "--ep",
            "--device",
            "--no-analyze",
            "--no-optimize",
            "--max-optim-iterations",
            "--trust-remote-code",
            "--verbose",
        ]:
            assert flag in result.output, f"Help text missing flag: {flag}"


class TestBuildFlagPassthrough:
    """Each behavior-bearing flag must propagate to ``_run_single_build``."""

    def _base_args(self, cfg: str, tmp_path: Path) -> list[str]:
        return ["-c", cfg, "-m", "microsoft/resnet-50", "-o", str(tmp_path / "out")]

    def test_defaults_no_flags(self, tmp_path: Path, mock_run_single_build: MagicMock):
        """With no optional flags, defaults are forwarded as-is."""
        cfg = _make_minimal_config_file(tmp_path)
        result = _invoke(self._base_args(cfg, tmp_path))
        assert result.exit_code == 0, result.output
        kwargs = mock_run_single_build.call_args.kwargs
        assert kwargs["rebuild"] is False
        assert kwargs["model_id"] == "microsoft/resnet-50"
        assert kwargs["extra_kwargs"] == {} or "trust_remote_code" not in kwargs["extra_kwargs"]

    def test_rebuild_flag(self, tmp_path: Path, mock_run_single_build: MagicMock):
        """``--rebuild`` sets ``rebuild=True``."""
        cfg = _make_minimal_config_file(tmp_path)
        result = _invoke([*self._base_args(cfg, tmp_path), "--rebuild"])
        assert result.exit_code == 0, result.output
        assert mock_run_single_build.call_args.kwargs["rebuild"] is True

    def test_no_quant_clears_quant(self, tmp_path: Path, mock_run_single_build: MagicMock):
        """``--no-quant`` zeroes ``config.quant`` before dispatching."""
        cfg_path = tmp_path / "with_quant.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "loader": {"task": "image-classification"},
                    "export": {"opset_version": 17, "batch_size": 1},
                    "optim": {},
                    "quant": {
                        "mode": "qdq",
                        "samples": 10,
                        "task": "image-classification",
                        "model_name": "test",
                    },
                    "compile": None,
                }
            )
        )
        result = _invoke(
            ["-c", str(cfg_path), "-m", "x", "-o", str(tmp_path / "out"), "--no-quant"]
        )
        assert result.exit_code == 0, result.output
        assert mock_run_single_build.call_args.kwargs["config"].quant is None

    def test_no_compile_clears_compile(self, tmp_path: Path, mock_run_single_build: MagicMock):
        """``--no-compile`` zeroes ``config.compile``."""
        cfg = _make_minimal_config_file(tmp_path, compile_section={"execution_provider": "qnn"})
        result = _invoke(["-c", cfg, "-m", "x", "-o", str(tmp_path / "out"), "--no-compile"])
        assert result.exit_code == 0, result.output
        assert mock_run_single_build.call_args.kwargs["config"].compile is None

    def test_compile_preserves_compile(self, tmp_path: Path, mock_run_single_build: MagicMock):
        """``--compile`` keeps the compile section from the config file."""
        cfg = _make_minimal_config_file(tmp_path, compile_section={"execution_provider": "qnn"})
        result = _invoke(["-c", cfg, "-m", "x", "-o", str(tmp_path / "out"), "--compile"])
        assert result.exit_code == 0, result.output
        assert mock_run_single_build.call_args.kwargs["config"].compile is not None

    def test_compile_absent_inherits_from_config(
        self, tmp_path: Path, mock_run_single_build: MagicMock
    ):
        """Without ``--compile`` / ``--no-compile``, config compile is preserved."""
        cfg = _make_minimal_config_file(tmp_path, compile_section={"execution_provider": "qnn"})
        result = _invoke(["-c", cfg, "-m", "x", "-o", str(tmp_path / "out")])
        assert result.exit_code == 0, result.output
        assert mock_run_single_build.call_args.kwargs["config"].compile is not None

    def test_no_optimize_sets_extra_kwarg(self, tmp_path: Path, mock_run_single_build: MagicMock):
        """``--no-optimize`` sets ``extra_kwargs['skip_optimize']``."""
        cfg = _make_minimal_config_file(tmp_path)
        result = _invoke([*self._base_args(cfg, tmp_path), "--no-optimize"])
        assert result.exit_code == 0, result.output
        assert mock_run_single_build.call_args.kwargs["extra_kwargs"].get("skip_optimize") is True

    def test_no_analyze_zeros_max_iterations(
        self, tmp_path: Path, mock_run_single_build: MagicMock
    ):
        """``--no-analyze`` forces ``hack_max_optim_iterations=0``."""
        cfg = _make_minimal_config_file(tmp_path)
        result = _invoke([*self._base_args(cfg, tmp_path), "--no-analyze"])
        assert result.exit_code == 0, result.output
        extras = mock_run_single_build.call_args.kwargs["extra_kwargs"]
        assert extras.get("hack_max_optim_iterations") == 0

    def test_max_optim_iterations_explicit(self, tmp_path: Path, mock_run_single_build: MagicMock):
        """``--max-optim-iterations N`` forwards N as ``hack_max_optim_iterations``."""
        cfg = _make_minimal_config_file(tmp_path)
        result = _invoke([*self._base_args(cfg, tmp_path), "--max-optim-iterations", "5"])
        assert result.exit_code == 0, result.output
        assert (
            mock_run_single_build.call_args.kwargs["extra_kwargs"].get("hack_max_optim_iterations")
            == 5
        )

    def test_no_analyze_wins_over_max_iterations(
        self, tmp_path: Path, mock_run_single_build: MagicMock
    ):
        """``--no-analyze`` takes precedence over ``--max-optim-iterations``."""
        cfg = _make_minimal_config_file(tmp_path)
        result = _invoke(
            [
                *self._base_args(cfg, tmp_path),
                "--no-analyze",
                "--max-optim-iterations",
                "7",
            ]
        )
        assert result.exit_code == 0, result.output
        assert (
            mock_run_single_build.call_args.kwargs["extra_kwargs"].get("hack_max_optim_iterations")
            == 0
        )

    def test_ep_flag_forwarded(self, tmp_path: Path, mock_run_single_build: MagicMock):
        """``--ep`` value reaches ``_run_single_build`` verbatim."""
        cfg = _make_minimal_config_file(tmp_path)
        result = _invoke([*self._base_args(cfg, tmp_path), "--ep", "qnn"])
        assert result.exit_code == 0, result.output
        assert mock_run_single_build.call_args.kwargs["ep"] == "qnn"

    def test_device_flag_forwarded(self, tmp_path: Path, mock_run_single_build: MagicMock):
        """``--device`` value reaches ``_run_single_build`` verbatim."""
        cfg = _make_minimal_config_file(tmp_path)
        result = _invoke([*self._base_args(cfg, tmp_path), "--device", "NPU"])
        assert result.exit_code == 0, result.output
        assert mock_run_single_build.call_args.kwargs["device"] == "NPU"

    def test_trust_remote_code_forwarded(self, tmp_path: Path, mock_run_single_build: MagicMock):
        """``--trust-remote-code`` is forwarded via ``extra_kwargs``."""
        cfg = _make_minimal_config_file(tmp_path)
        result = _invoke([*self._base_args(cfg, tmp_path), "--trust-remote-code"])
        assert result.exit_code == 0, result.output
        assert (
            mock_run_single_build.call_args.kwargs["extra_kwargs"].get("trust_remote_code") is True
        )

    def test_verbose_flag_accepted(self, tmp_path: Path, mock_run_single_build: MagicMock):
        """``-v/--verbose`` is parsed and does not affect dispatch."""
        cfg = _make_minimal_config_file(tmp_path)
        result = _invoke([*self._base_args(cfg, tmp_path), "-v"])
        assert result.exit_code == 0, result.output

    def test_debug_inherited_from_parent_ctx(
        self, tmp_path: Path, mock_run_single_build: MagicMock
    ):
        """``ctx.obj['debug']`` from the parent CLI bumps verbosity."""
        cfg = _make_minimal_config_file(tmp_path)
        result = _invoke(self._base_args(cfg, tmp_path), debug=True)
        assert result.exit_code == 0, result.output

    def test_model_omitted_means_random_weights(
        self, tmp_path: Path, mock_run_single_build: MagicMock
    ):
        """Omitting ``-m`` selects the random-weight build path."""
        cfg = _make_minimal_config_file(tmp_path)
        result = _invoke(["-c", cfg, "-o", str(tmp_path / "out")])
        assert result.exit_code == 0, result.output
        assert mock_run_single_build.call_args.kwargs["model_id"] is None


class TestBuildErrorHandling:
    """Pipeline errors must surface with a hint and a non-zero exit code."""

    def test_value_error_becomes_usage_error(self, tmp_path: Path):
        """``ValueError`` from the pipeline becomes a UsageError."""
        cfg = _make_minimal_config_file(tmp_path)
        with patch(
            "winml.modelkit.commands.build._run_single_build",
            side_effect=ValueError("bad config"),
        ):
            result = _invoke(["-c", cfg, "-m", "x", "-o", str(tmp_path / "out")])
        assert result.exit_code != 0
        assert "bad config" in result.output

    def test_generic_failure_is_reported(self, tmp_path: Path):
        """Unhandled exceptions surface as ``Build failed:`` without traceback."""
        cfg = _make_minimal_config_file(tmp_path)
        with patch(
            "winml.modelkit.commands.build._run_single_build",
            side_effect=RuntimeError("ONNX export failed"),
        ):
            result = _invoke(["-c", cfg, "-m", "x", "-o", str(tmp_path / "out")])
        assert result.exit_code != 0
        assert "Build failed" in result.output

    def test_quant_failure_hint(self, tmp_path: Path):
        """Errors mentioning Quantization include the ``--no-quant`` hint."""
        cfg = _make_minimal_config_file(tmp_path)
        with patch(
            "winml.modelkit.commands.build._run_single_build",
            side_effect=RuntimeError("Quantization failed: calibration"),
        ):
            result = _invoke(["-c", cfg, "-m", "x", "-o", str(tmp_path / "out")])
        assert result.exit_code != 0
        assert "--no-quant" in result.output

    def test_compile_failure_hint(self, tmp_path: Path):
        """Errors mentioning Compilation include the ``--no-compile`` hint."""
        cfg = _make_minimal_config_file(tmp_path)
        with patch(
            "winml.modelkit.commands.build._run_single_build",
            side_effect=RuntimeError("Compilation failed: missing EP"),
        ):
            result = _invoke(["-c", cfg, "-m", "x", "-o", str(tmp_path / "out")])
        assert result.exit_code != 0
        assert "--no-compile" in result.output


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

    def test_compile_inherited_from_config_when_no_flag(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        """No --compile/--no-compile → compile section from config is preserved."""
        from winml.modelkit.commands.build import build

        runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "test", "-o", str(tmp_path)],
            obj={"debug": False},
        )
        config = mock_build_api.call_args.kwargs["config"]
        assert config.compile is not None

    def test_compile_flag_on_config_without_compile_raises(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """--compile on a config that has no compile section raises a UsageError."""
        from winml.modelkit.commands.build import build

        cfg = tmp_path / "no_compile.json"
        cfg.write_text(
            json.dumps(
                {
                    "loader": {"task": "image-classification"},
                    "export": {"opset_version": 17, "batch_size": 1},
                    "optim": {},
                    "compile": None,
                }
            )
        )
        result = runner.invoke(
            build,
            ["-c", str(cfg), "-m", "test", "-o", str(tmp_path), "--compile"],
            obj={"debug": False},
        )
        assert result.exit_code != 0
        assert "compile" in result.output.lower()

    def test_compile_flag_on_config_with_compile_succeeds(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--compile on a config that already has a compile section succeeds and preserves it."""
        from winml.modelkit.commands.build import build

        runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "test", "-o", str(tmp_path), "--compile"],
            obj={"debug": False},
        )
        config = mock_build_api.call_args.kwargs["config"]
        assert config.compile is not None


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
    """Test _run_compile_stage output validation."""

    @patch("winml.modelkit.compiler.compile_onnx")
    def test_none_compile_config_skips_stage(
        self,
        mock_compile: MagicMock,
        tmp_path: Path,
    ) -> None:
        """compile=None skips compile_onnx entirely and returns current_path unchanged."""
        from winml.modelkit.commands.build import _run_compile_stage
        from winml.modelkit.config import WinMLBuildConfig

        input_path = tmp_path / "quantized.onnx"
        input_path.write_bytes(b"dummy")
        compiled_path = tmp_path / "compiled.onnx"

        config = WinMLBuildConfig(compile=None)
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
