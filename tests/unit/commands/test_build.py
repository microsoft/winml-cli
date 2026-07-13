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
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


_DEVICE_TO_EPS = {
    "npu": ["QNNExecutionProvider"],
    "gpu": ["DmlExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
}


def _fake_resolve_check_device_ep(*, device: str = "auto", ep: str | None = None):
    """Side effect for resolve_check_device_ep that honours the requested device.

    The build command's --device path calls resolve_quant_compile_config which
    in turn calls resolve_check_device_ep. Tests pass explicit devices like
    "npu", "gpu", "cpu" -- echo them back with a canonical EP so the downstream
    precision policy resolves deterministically.
    """
    resolved = device.lower() if device != "auto" else "npu"
    eps = _DEVICE_TO_EPS.get(resolved, ["CPUExecutionProvider"])
    return resolved, ["npu", "gpu", "cpu"], eps


@pytest.fixture(autouse=True)
def mock_resolve_device():
    """Mock device/EP resolution to avoid hardware detection.

    The build command calls ``resolve_device`` / ``resolve_eps`` to auto-select
    an EP when ``--ep`` is not specified, and ``resolve_check_device_ep`` (via
    ``resolve_quant_compile_config``) when ``--device`` is explicit. All three
    must be mocked to avoid slow DLL scanning and WinML SDK discovery on CI
    runners without WinML installed. ``WinMLEPRegistry.get_instance`` is also
    patched defensively for any downstream code path that may touch it.
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
            "winml.modelkit.sysinfo.resolve_check_device_ep",
            side_effect=_fake_resolve_check_device_ep,
        ),
        patch(
            "winml.modelkit.session.ep_registry.WinMLEPRegistry.get_instance",
            return_value=mock_registry,
        ),
    ):
        yield


@pytest.fixture(autouse=True)
def mock_task_model_compatibility_validator():
    """Default to no-op for preflight task/model compatibility checks.

    Most build command unit tests are CLI plumbing tests and should not hit
    HuggingFace config resolution paths.
    """
    with patch(
        "winml.modelkit.commands.build._validate_task_supported_for_model",
        return_value=None,
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
            "model_id": "test",
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

    def test_rejects_incompatible_config_task_and_model(self, tmp_path: Path):
        """config.loader.task + --model mismatch fails before pipeline starts."""
        cfg = _make_minimal_config_file(tmp_path, task="text-generation")
        msg = (
            "config.loader.task='text-generation' is not supported for "
            "--model microsoft/resnet-50 (architecture: resnet). "
            "Supported tasks: image-classification, image-feature-extraction."
        )

        with (
            patch(
                "winml.modelkit.commands.build._validate_task_supported_for_model",
                side_effect=ValueError(msg),
            ) as mock_validate,
            patch("winml.modelkit.commands.build._run_single_build") as mock_run,
        ):
            result = _invoke(["-c", cfg, "-m", "microsoft/resnet-50", "-o", str(tmp_path / "out")])

        assert result.exit_code != 0
        assert msg in result.output
        mock_validate.assert_called_once_with(
            model_id="microsoft/resnet-50",
            task="text-generation",
            task_field_name="config.loader.task",
            trust_remote_code=False,
            hf_config=None,
        )
        mock_run.assert_not_called()

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
                        "model_id": "test",
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

    def test_allow_unsupported_nodes_sets_extra_kwarg(
        self, tmp_path: Path, mock_run_single_build: MagicMock
    ):
        """``--allow-unsupported-nodes`` sets ``extra_kwargs['allow_unsupported_nodes']``."""
        cfg = _make_minimal_config_file(tmp_path)
        result = _invoke([*self._base_args(cfg, tmp_path), "--allow-unsupported-nodes"])
        assert result.exit_code == 0, result.output
        extras = mock_run_single_build.call_args.kwargs["extra_kwargs"]
        assert extras.get("allow_unsupported_nodes") is True

    def test_allow_unsupported_nodes_false_by_default(
        self, tmp_path: Path, mock_run_single_build: MagicMock
    ):
        """Without the flag, ``allow_unsupported_nodes`` is forwarded as False."""
        cfg = _make_minimal_config_file(tmp_path)
        result = _invoke(self._base_args(cfg, tmp_path))
        assert result.exit_code == 0, result.output
        extras = mock_run_single_build.call_args.kwargs["extra_kwargs"]
        assert extras.get("allow_unsupported_nodes") is False

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
        assert mock_run_single_build.call_args.kwargs["device"] == "npu"

    def test_precision_flag_sets_quant(self, tmp_path: Path, mock_run_single_build: MagicMock):
        """``--precision int8`` populates the quant config for the target device."""
        cfg = _make_minimal_config_file(tmp_path)  # quant=None
        result = _invoke(
            [*self._base_args(cfg, tmp_path), "--device", "gpu", "--precision", "int8"]
        )
        assert result.exit_code == 0, result.output
        passed = mock_run_single_build.call_args.kwargs["config"]
        assert passed.quant is not None
        assert passed.quant.weight_type == "uint8"

    def test_precision_fp16_sets_fp16_algorithm(
        self, tmp_path: Path, mock_run_single_build: MagicMock
    ):
        """``--precision fp16`` sets an fp16 algorithm quant config."""
        cfg = _make_minimal_config_file(tmp_path)
        result = _invoke(
            [*self._base_args(cfg, tmp_path), "--device", "npu", "--precision", "fp16"]
        )
        assert result.exit_code == 0, result.output
        quant = mock_run_single_build.call_args.kwargs["config"].quant
        assert quant is not None
        assert quant.mode == "fp16"

    def test_precision_alone_triggers_quant_patch(
        self, tmp_path: Path, mock_run_single_build: MagicMock
    ):
        """``--precision`` without ``--device`` still patches quant to the fp16 algorithm."""
        # Config ships an explicit quant section; fp16 must switch it to the
        # fp16 algorithm even though
        # --device was not passed (precision alone triggers the patch path).
        config = {
            "loader": {"task": "image-classification"},
            "export": {"opset_version": 17, "batch_size": 1},
            "optim": {},
            "quant": {
                "mode": "qdq",
                "samples": 10,
                "task": "image-classification",
                "model_id": "test",
            },
            "compile": None,
        }
        cfg = tmp_path / "withquant.json"
        cfg.write_text(json.dumps(config))
        result = _invoke(
            [
                "-c",
                str(cfg),
                "-m",
                "microsoft/resnet-50",
                "-o",
                str(tmp_path / "out"),
                "--precision",
                "fp16",
            ]
        )
        assert result.exit_code == 0, result.output
        quant = mock_run_single_build.call_args.kwargs["config"].quant
        assert quant is not None
        assert quant.mode == "fp16"

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
        assert call_kwargs["device"] == "npu"


# =============================================================================
# EP AUTO-SELECTION TESTS
# =============================================================================


class TestBuildEpAutoSelection:
    """Auto-select EP when --ep is omitted: resolve device -> first compatible EP.

    The selection result depends on the host's available devices/EPs at runtime,
    so resolve_device / resolve_eps are mocked to give the test a known surface.
    Regression: hardcoded ``[QNN, OV, VitisAI]`` walk used to pick OpenVINO on a
    GPU box if OV happened to be installed, leaving black nodes that blocked a
    subsequent build for the actual device (issue #663).
    """

    def test_auto_selects_qnn_for_npu(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Default device=auto resolves to npu (per fixture) -> QNNExecutionProvider."""
        from winml.modelkit.commands.build import build

        result = runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "test", "-o", str(tmp_path)],
            obj={"debug": False},
        )
        assert result.exit_code == 0, result.output
        assert mock_build_api.call_args.kwargs["ep"] == "QNNExecutionProvider"

    def test_auto_selects_dml_for_explicit_gpu(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        """``--device gpu`` (no --ep) -> resolve_device returns gpu -> Dml."""
        from winml.modelkit.commands.build import build

        with patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("gpu", ["gpu", "cpu"]),
        ):
            result = runner.invoke(
                build,
                [
                    "-c",
                    str(sample_config_file),
                    "-m",
                    "test",
                    "-o",
                    str(tmp_path),
                    "--device",
                    "gpu",
                ],
                obj={"debug": False},
            )
        assert result.exit_code == 0, result.output
        assert mock_build_api.call_args.kwargs["ep"] == "DmlExecutionProvider"

    def test_auto_selects_cpu_ep_for_explicit_cpu(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        """``--device cpu`` (no --ep) -> CPUExecutionProvider."""
        from winml.modelkit.commands.build import build

        with patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("cpu", ["cpu"]),
        ):
            result = runner.invoke(
                build,
                [
                    "-c",
                    str(sample_config_file),
                    "-m",
                    "test",
                    "-o",
                    str(tmp_path),
                    "--device",
                    "cpu",
                ],
                obj={"debug": False},
            )
        assert result.exit_code == 0, result.output
        assert mock_build_api.call_args.kwargs["ep"] == "CPUExecutionProvider"

    def test_explicit_ep_bypasses_auto_selection(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        """``--ep qnn`` keeps the user's choice even when device resolution would pick another."""
        from winml.modelkit.commands.build import build

        # resolve_device would point at gpu -> Dml, but --ep wins.
        with (
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("gpu", ["gpu", "cpu"]),
            ),
            patch("winml.modelkit.sysinfo.resolve_eps") as mock_resolve_eps,
        ):
            result = runner.invoke(
                build,
                ["-c", str(sample_config_file), "-m", "test", "-o", str(tmp_path), "--ep", "qnn"],
                obj={"debug": False},
            )
        assert result.exit_code == 0, result.output
        assert mock_build_api.call_args.kwargs["ep"] == "qnn"
        mock_resolve_eps.assert_not_called()

    def test_resolve_device_value_error_surfaces_as_usage_error(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        """resolve_check_device_ep raising (explicit device w/ no compatible EP) -> UsageError.

        Uses default ``--device auto`` (no CLI flag) so the downstream
        device-patch path isn't triggered; the only resolution call is the
        ``resolve_check_device_ep`` inside the auto-select block.
        """
        from winml.modelkit.commands.build import build

        with patch(
            "winml.modelkit.sysinfo.resolve_check_device_ep",
            side_effect=ValueError("simulated resolve failure"),
        ):
            result = runner.invoke(
                build,
                ["-c", str(sample_config_file), "-m", "test", "-o", str(tmp_path)],
                obj={"debug": False},
            )
        assert result.exit_code != 0
        assert "simulated resolve failure" in result.output
        mock_build_api.assert_not_called()

    def test_auto_selection_respects_resolve_eps_priority(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        """First element of resolve_check_device_ep's available_eps is selected."""
        from winml.modelkit.commands.build import build

        with patch(
            "winml.modelkit.sysinfo.resolve_check_device_ep",
            return_value=(
                "gpu",
                ["gpu", "cpu"],
                ["DmlExecutionProvider", "OpenVINOExecutionProvider"],
            ),
        ):
            result = runner.invoke(
                build,
                [
                    "-c",
                    str(sample_config_file),
                    "-m",
                    "test",
                    "-o",
                    str(tmp_path),
                    "--device",
                    "gpu",
                ],
                obj={"debug": False},
            )
        assert result.exit_code == 0, result.output
        assert mock_build_api.call_args.kwargs["ep"] == "DmlExecutionProvider"


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

    def test_build_onnx_suffix_but_not_exists_raises(
        self,
        runner: CliRunner,
        sample_config_file: Path,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A non-existent .onnx path raises cleanly instead of HF fallthrough (#553)."""
        from winml.modelkit.commands.build import build

        output_dir = tmp_path / "out"
        result = runner.invoke(
            build,
            ["-c", str(sample_config_file), "-m", "nonexistent.onnx", "-o", str(output_dir)],
            obj={"debug": False},
        )
        # classify_model_input rejects a missing .onnx path up front rather than
        # handing it to the HF loader (which would give a confusing config error).
        assert result.exit_code != 0
        assert "ONNX file not found" in result.output
        assert not mock_build_api.called

    def test_build_configless_missing_onnx_raises(
        self,
        runner: CliRunner,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Config-less path: a missing .onnx path is rejected up front (#553)."""
        from winml.modelkit.commands.build import build

        output_dir = tmp_path / "out"
        result = runner.invoke(
            build,
            ["-m", "nonexistent.onnx", "-o", str(output_dir), "--ep", "cpu"],
            obj={"debug": False},
        )
        assert result.exit_code != 0
        assert "ONNX file not found" in result.output
        assert not mock_build_api.called

    def test_build_configless_invalid_id_raises(
        self,
        runner: CliRunner,
        mock_build_api: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Config-less path: an unparsable id gets the friendly classifier error."""
        from winml.modelkit.commands.build import build

        output_dir = tmp_path / "out"
        result = runner.invoke(
            build,
            ["-m", "has spaces", "-o", str(output_dir), "--ep", "cpu"],
            obj={"debug": False},
        )
        assert result.exit_code != 0
        assert "not a valid HuggingFace" in result.output
        assert not mock_build_api.called


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


class TestBuildHfPipelineModelType:
    """Regression: the CLI HF pipeline must thread loader.model_type into _load_model.

    Without this, a config requesting a derived model_type (e.g.
    ``qwen3_transformer_only``) is silently loaded as its native type, so the
    wrong model class is exported. See _build_hf_pipeline.
    """

    @patch("winml.modelkit.utils.console.StageLive")
    @patch("winml.modelkit.export.export_onnx")
    @patch("winml.modelkit.build.hf._load_model")
    def test_load_model_receives_config_model_type(
        self,
        mock_load_model: MagicMock,
        mock_export_onnx: MagicMock,
        mock_stage_live: MagicMock,
        tmp_path: Path,
    ) -> None:
        from winml.modelkit.commands.build import _build_hf_pipeline

        mock_stage_live.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_stage_live.return_value.__exit__ = MagicMock(return_value=False)

        # Stop the pipeline right after export so we only exercise the load call.
        sentinel = RuntimeError("stop-after-export")
        mock_export_onnx.side_effect = sentinel

        config = MagicMock()
        config.loader.model_type = "qwen3_transformer_only"
        config.loader.task = "feature-extraction"
        config.export = MagicMock()

        with pytest.raises(RuntimeError, match="stop-after-export"):
            _build_hf_pipeline(
                config=config,
                model_id="Qwen/Qwen3-0.6B",
                output_dir=tmp_path / "out",
                rebuild=True,
                cache_key=None,
                ep=None,
                device="cpu",
                extra_kwargs={},
                preloaded_hf_config=None,
            )

        mock_load_model.assert_called_once()
        assert mock_load_model.call_args.kwargs["model_type"] == "qwen3_transformer_only"

    @patch("winml.modelkit.commands.build._run_compile_stage")
    @patch("winml.modelkit.commands.build._run_quantize_stage")
    @patch("winml.modelkit.commands.build._run_optimize_stage")
    @patch("winml.modelkit.commands.build._show_io")
    @patch("winml.modelkit.utils.console.StageLive")
    @patch("winml.modelkit.export.export_onnx")
    @patch("winml.modelkit.build.hf._load_model")
    def test_quant_model_type_carried_into_quantize_stage(
        self,
        mock_load_model: MagicMock,
        mock_export_onnx: MagicMock,
        mock_stage_live: MagicMock,
        mock_show_io: MagicMock,
        mock_optimize: MagicMock,
        mock_quantize: MagicMock,
        mock_compile: MagicMock,
        tmp_path: Path,
    ) -> None:
        """The CLI HF pipeline must hand the model_type to the quantize stage.

        The model-type-specific quant policy is resolved inside ``quantize_onnx``
        from ``config.quant.model_type``; the pipeline is responsible for carrying
        the resolved variant onto the quant config so the policy fires.
        """
        from winml.modelkit.commands.build import _build_hf_pipeline

        mock_stage_live.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_stage_live.return_value.__exit__ = MagicMock(return_value=False)

        pytorch_model = MagicMock()
        pytorch_model.config.model_type = "qwen3_transformer_only"
        mock_load_model.return_value = pytorch_model

        optimized = tmp_path / "optimized.onnx"
        mock_optimize.return_value = (optimized, None)

        # Stop right after the quantize stage so we don't exercise compile.
        mock_quantize.side_effect = RuntimeError("stop-after-quantize")

        config = MagicMock()
        config.loader.model_type = "qwen3_transformer_only"
        config.loader.task = "text2text-generation"
        config.loader.model_class = None
        config.export = MagicMock()
        config.quant = MagicMock(name="quant_config")
        config.quant.model_type = None
        config.to_dict.return_value = {}

        with pytest.raises(RuntimeError, match="stop-after-quantize"):
            _build_hf_pipeline(
                config=config,
                model_id="Qwen/Qwen3-0.6B",
                output_dir=tmp_path / "out",
                rebuild=True,
                cache_key=None,
                ep=None,
                device="cpu",
                extra_kwargs={},
                preloaded_hf_config=None,
            )

        # The resolved variant must be carried onto the quant config so that
        # quantize_onnx can resolve + apply the model-type-specific policy.
        assert config.quant.model_type == "qwen3_transformer_only"
        mock_quantize.assert_called_once()


class TestBuildEpResolution:
    """--ep forwarding into config generation + the compile EP-availability gate."""

    def _base_args(self, cfg: str, tmp_path: Path) -> list[str]:
        return ["-c", cfg, "-m", "microsoft/resnet-50", "-o", str(tmp_path / "out")]

    def test_ep_forwarded_to_generate_build_config(
        self, tmp_path: Path, mock_run_single_build: MagicMock
    ):
        """On the auto-config path (-m, no -c), --ep reaches generate_build_config.

        Regression: the build command dropped --ep when auto-generating a config,
        so the requested EP never influenced the generated config (it failed or
        analyzed/compiled for the wrong EP).
        """
        fake_cfg = MagicMock()
        fake_cfg.compile = None  # no compile -> EP-availability gate is skipped
        with (
            patch("winml.modelkit.config.generate_build_config", return_value=fake_cfg) as mock_gen,
            patch(
                "winml.modelkit.commands.build._validate_loader_tasks_for_model",
                return_value=None,
            ),
        ):
            result = _invoke(
                ["-m", "microsoft/resnet-50", "--ep", "openvino", "-o", str(tmp_path / "out")]
            )
        assert result.exit_code == 0, result.output
        assert mock_gen.call_args.kwargs["ep"] == "openvino"

    def test_auto_config_onnx_model_uses_single_generate_call(
        self, tmp_path: Path, mock_run_single_build: MagicMock
    ):
        """Auto-config ONNX input must call generate_build_config exactly once.

        Regression: build used to assign branch-specific configs and then
        unconditionally overwrite them with a second generate_build_config call.
        """
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake-onnx-data")

        fake_cfg = MagicMock()
        fake_cfg.compile = None
        fake_cfg.validate.return_value = None
        fake_cfg.loader = MagicMock()
        fake_cfg.loader.task = None

        with (
            patch("winml.modelkit.config.generate_build_config", return_value=fake_cfg) as mock_gen,
            patch(
                "winml.modelkit.commands.build._validate_loader_tasks_for_model",
                return_value=None,
            ),
        ):
            result = _invoke(["-m", str(onnx_file), "-o", str(tmp_path / "out")])

        assert result.exit_code == 0, result.output
        assert mock_gen.call_count == 1
        assert mock_gen.call_args.kwargs["onnx_path"] == str(onnx_file)


class TestBuildOnnxPipelineRegressions:
    """Regressions for CLI ONNX pipeline behavior in _build_onnx_pipeline."""

    def test_pre_quantized_stamp_clears_quant_when_already_skip_optimize(
        self, tmp_path: Path
    ) -> None:
        """skip_optimize is the pre-quantized invariant and must imply no quant."""
        from winml.modelkit.build.common import ensure_pre_quantized_stamped
        from winml.modelkit.config import WinMLBuildConfig
        from winml.modelkit.quant.config import WinMLQuantizationConfig

        onnx_file = tmp_path / "input.onnx"
        onnx_file.write_bytes(b"fake-onnx-data")
        config = WinMLBuildConfig(quant=WinMLQuantizationConfig())
        config.skip_optimize = True

        ensure_pre_quantized_stamped(config, onnx_file)

        assert config.skip_optimize is True
        assert config.quant is None

    @patch("winml.modelkit.quant.quantize_onnx")
    def test_quantize_stage_skips_when_config_skip_optimize_is_set(
        self,
        mock_quantize: MagicMock,
        tmp_path: Path,
    ) -> None:
        """The quantize stage must not double-quantize pre-quantized inputs."""
        from winml.modelkit.commands.build import _run_quantize_stage
        from winml.modelkit.config import WinMLBuildConfig
        from winml.modelkit.quant.config import WinMLQuantizationConfig

        input_path = tmp_path / "input.onnx"
        input_path.write_bytes(b"fake-onnx-data")
        config = WinMLBuildConfig(quant=WinMLQuantizationConfig())
        config.skip_optimize = True
        timings: list[tuple[str, float | None]] = []

        result = _run_quantize_stage(
            config=config,
            current_path=input_path,
            quantized_path=tmp_path / "quantized.onnx",
            stage_timings=timings,
        )

        assert result == input_path
        assert config.quant is None
        assert timings == []
        mock_quantize.assert_not_called()

    def test_pre_quantized_stamp_runs_before_optimize(self, tmp_path: Path) -> None:
        """_build_onnx_pipeline must stamp config before optimize/quantize stages.

        This ensures pre-quantized ONNX inputs can set skip_optimize and clear
        quant before stage dispatch, preventing optimize/quantize double-work.
        """
        from winml.modelkit.commands.build import _build_onnx_pipeline

        onnx_file = tmp_path / "input.onnx"
        onnx_file.write_bytes(b"fake-onnx-data")
        output_dir = tmp_path / "out"

        config = MagicMock()
        config.skip_optimize = False
        config.quant = MagicMock(name="quant_config")
        config.validate.return_value = None
        config.to_dict.return_value = {}

        def _stamp(cfg: MagicMock, _path: Path) -> None:
            cfg.skip_optimize = True
            cfg.quant = None

        with (
            patch(
                "winml.modelkit.build.common.ensure_pre_quantized_stamped",
                side_effect=_stamp,
            ) as mock_stamp,
            patch(
                "winml.modelkit.commands.build._run_optimize_stage",
                return_value=(output_dir / onnx_file.name, None),
            ) as mock_opt,
            patch(
                "winml.modelkit.commands.build._run_quantize_stage",
                side_effect=lambda **kwargs: kwargs["current_path"],
            ) as mock_quant,
            patch(
                "winml.modelkit.commands.build._run_compile_stage",
                side_effect=lambda **kwargs: kwargs["current_path"],
            ),
        ):
            result = _build_onnx_pipeline(
                config=config,
                onnx_path=onnx_file,
                output_dir=output_dir,
                rebuild=True,
                ep="cpu",
                device="cpu",
                extra_kwargs={},
            )

        assert result is not None
        mock_stamp.assert_called_once()
        assert mock_opt.call_args.kwargs["skip_optimize"] is True
        assert mock_quant.call_args.kwargs["config"].quant is None


# =============================================================================
# COMPOSITE MODEL TESTS
# =============================================================================


class TestBuildComposite:
    """Test build fans a composite model out with flat _<component> naming."""

    def test_composite_builds_flat_with_cache_key(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """A composite model runs _run_single_build once per sub-component with cache_key."""
        from winml.modelkit.commands.build import build

        components = {
            "decoder_prefill": "feature-extraction",
            "decoder_gen": "text2text-generation",
        }
        output_dir = tmp_path / "out"

        fake_cfg = MagicMock()
        fake_cfg.quant = None
        fake_cfg.compile = None
        fake_cfg.loader = MagicMock(task=None)

        with (
            patch(
                "winml.modelkit.loader.resolution.resolve_composite_components",
                return_value=components,
            ),
            patch(
                "winml.modelkit.config.generate_build_config",
                return_value=fake_cfg,
            ) as mock_gen_cfg,
            patch(
                "winml.modelkit.commands.build._run_single_build",
            ) as mock_single_build,
            patch(
                "winml.modelkit.commands.build._validate_loader_tasks_for_model",
                return_value=None,
            ),
        ):
            result = runner.invoke(
                build,
                ["-m", "Qwen/Qwen3-0.6B", "-o", str(output_dir)],
                obj={"debug": False},
            )

        assert result.exit_code == 0, result.output
        # One build per component.
        assert mock_single_build.call_count == len(components)
        # All components share the same resolved_dir (flat layout).
        built_dirs = {
            Path(call.kwargs["resolved_dir"]) for call in mock_single_build.call_args_list
        }
        assert built_dirs == {output_dir}
        # Each component name is passed as cache_key for flat file naming.
        built_keys = {call.kwargs["cache_key"] for call in mock_single_build.call_args_list}
        assert built_keys == set(components)
        # generate_build_config is called once for the outer auto-gen config,
        # then once per component.
        component_tasks = {
            call.kwargs["task"]
            for call in mock_gen_cfg.call_args_list
            if "task" in call.kwargs and call.kwargs["task"] is not None
        }
        assert component_tasks == set(components.values())

    def test_composite_autogen_passes_task_none_to_resolver(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Auto-generated config (no -c) passes task=None so the seq2seq bridge applies."""
        from winml.modelkit.commands.build import build

        components = {"encoder": "feature-extraction", "decoder": "text2text-generation"}
        output_dir = tmp_path / "out"

        fake_cfg = MagicMock()
        fake_cfg.quant = None
        fake_cfg.compile = None
        # Auto-detected task for a seq2seq model.
        fake_cfg.loader = MagicMock(task="text2text-generation", model_type="t5")

        with (
            patch(
                "winml.modelkit.loader.resolution.resolve_composite_components",
                return_value=components,
            ) as mock_resolve,
            patch(
                "winml.modelkit.config.generate_build_config",
                return_value=fake_cfg,
            ),
            patch("winml.modelkit.commands.build._run_single_build"),
            patch(
                "winml.modelkit.commands.build._validate_loader_tasks_for_model",
                return_value=None,
            ),
        ):
            result = runner.invoke(
                build,
                ["-m", "google-t5/t5-small", "-o", str(output_dir)],
                obj={"debug": False},
            )

        assert result.exit_code == 0, result.output
        # No -c: task must be None (so the seq2seq bridge is applied), but
        # model_type is still forwarded.
        assert mock_resolve.call_args.kwargs["task"] is None
        assert mock_resolve.call_args.kwargs["model_type"] == "t5"

    def test_composite_config_file_forwards_explicit_task(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """A config-file build forwards the explicit loader.task to the resolver."""
        from winml.modelkit.commands.build import build

        components = {"encoder": "feature-extraction", "decoder": "translation"}
        output_dir = tmp_path / "out"

        # Outer config comes from -c via _load_config; control it directly so
        # loader.task / loader.model_type are deterministic.
        outer_cfg = MagicMock()
        outer_cfg.quant = None
        outer_cfg.compile = None
        outer_cfg.loader = MagicMock(task="translation", model_type="marian")

        component_cfg = MagicMock()
        component_cfg.quant = None
        component_cfg.compile = None
        component_cfg.loader = MagicMock(task="translation", model_type="marian")

        config_path = tmp_path / "config.json"
        config_path.write_text("{}")

        with (
            patch(
                "winml.modelkit.commands.build._load_config",
                return_value=outer_cfg,
            ),
            patch(
                "winml.modelkit.loader.resolution.resolve_composite_components",
                return_value=components,
            ) as mock_resolve,
            patch(
                "winml.modelkit.config.generate_build_config",
                return_value=component_cfg,
            ),
            patch("winml.modelkit.commands.build._run_single_build"),
            patch(
                "winml.modelkit.commands.build._validate_loader_tasks_for_model",
                return_value=None,
            ),
        ):
            result = runner.invoke(
                build,
                [
                    "-c",
                    str(config_path),
                    "-m",
                    "Helsinki-NLP/opus-mt-en-de",
                    "-o",
                    str(output_dir),
                ],
                obj={"debug": False},
            )

        assert result.exit_code == 0, result.output
        # -c provided: explicit task and model_type are forwarded.
        assert mock_resolve.call_args.kwargs["task"] == "translation"
        assert mock_resolve.call_args.kwargs["model_type"] == "marian"

    def test_composite_rejects_use_cache(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """--use-cache is ambiguous for a composite and must be a usage error."""
        from winml.modelkit.commands.build import build

        fake_cfg = MagicMock()
        fake_cfg.quant = None
        fake_cfg.compile = None
        fake_cfg.loader = MagicMock(task="text-generation")
        fake_cfg.generate_cache_key.return_value = "key"

        with (
            patch(
                "winml.modelkit.loader.resolution.resolve_composite_components",
                return_value={"enc": "feature-extraction"},
            ),
            patch("winml.modelkit.config.generate_build_config", return_value=fake_cfg),
            patch(
                "winml.modelkit.commands.build._validate_loader_tasks_for_model",
                return_value=None,
            ),
            patch("winml.modelkit.commands.build._run_single_build"),
        ):
            result = runner.invoke(
                build,
                ["-m", "Qwen/Qwen3-0.6B", "--use-cache"],
                obj={"debug": False},
            )

        assert result.exit_code != 0
        assert "composite" in result.output.lower()

    def test_composite_resolution_valueerror_surfaces_as_usage_error(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """A ValueError during composite resolution is surfaced, not swallowed."""
        from winml.modelkit.commands.build import build

        fake_cfg = MagicMock()
        fake_cfg.quant = None
        fake_cfg.compile = None
        fake_cfg.loader = MagicMock(task=None)

        with (
            patch(
                "winml.modelkit.loader.resolution.resolve_composite_components",
                side_effect=ValueError(
                    "qwen3 has multiple composite exports; pass --task explicitly"
                ),
            ),
            patch("winml.modelkit.config.generate_build_config", return_value=fake_cfg),
            patch(
                "winml.modelkit.commands.build._validate_loader_tasks_for_model",
                return_value=None,
            ),
            patch("winml.modelkit.commands.build._run_single_build"),
        ):
            result = runner.invoke(
                build,
                ["-m", "Qwen/Qwen3-0.6B", "-o", str(tmp_path / "out")],
                obj={"debug": False},
            )

        assert result.exit_code != 0
        assert "multiple composite exports" in result.output

    def test_composite_resolution_unexpected_error_surfaces(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """An unexpected error during composite detection is surfaced, not masked."""
        from winml.modelkit.commands.build import build

        fake_cfg = MagicMock()
        fake_cfg.quant = None
        fake_cfg.compile = None
        fake_cfg.loader = MagicMock(task=None)

        with (
            patch(
                "winml.modelkit.loader.resolution.resolve_composite_components",
                side_effect=KeyError("boom"),
            ),
            patch("winml.modelkit.config.generate_build_config", return_value=fake_cfg),
            patch(
                "winml.modelkit.commands.build._validate_loader_tasks_for_model",
                return_value=None,
            ),
            patch("winml.modelkit.commands.build._run_single_build"),
        ):
            result = runner.invoke(
                build,
                ["-m", "Qwen/Qwen3-0.6B", "-o", str(tmp_path / "out")],
                obj={"debug": False},
            )

        assert result.exit_code != 0
        assert "unexpectedly" in result.output.lower()

    def test_composite_partial_build_warns(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """If a later sub-model fails, completed components are warned about."""
        from winml.modelkit.commands.build import build

        components = {
            "decoder_prefill": "feature-extraction",
            "decoder_gen": "text2text-generation",
        }
        output_dir = tmp_path / "out"

        fake_cfg = MagicMock()
        fake_cfg.quant = None
        fake_cfg.compile = None
        fake_cfg.loader = MagicMock(task=None)

        call_count = [0]

        def fake_single_build(**kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("second sub-model blew up")

        with (
            patch(
                "winml.modelkit.loader.resolution.resolve_composite_components",
                return_value=components,
            ),
            patch("winml.modelkit.config.generate_build_config", return_value=fake_cfg),
            patch(
                "winml.modelkit.commands.build._run_single_build",
                side_effect=fake_single_build,
            ),
            patch(
                "winml.modelkit.commands.build._validate_loader_tasks_for_model",
                return_value=None,
            ),
        ):
            result = runner.invoke(
                build,
                ["-m", "Qwen/Qwen3-0.6B", "-o", str(output_dir)],
                obj={"debug": False},
            )

        assert result.exit_code != 0
        # Warning about partial composite build.
        assert "composite build did not finish" in result.output.lower()

    def test_composite_clears_quant_compile_when_outer_is_none(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """When outer config has quant=None/compile=None, component configs are cleared."""
        from winml.modelkit.commands.build import build

        components = {"enc": "feature-extraction"}
        output_dir = tmp_path / "out"
        config_file = _make_minimal_config_file(tmp_path, task="text-generation")

        component_cfg = MagicMock()
        component_cfg.quant = MagicMock(name="component_quant")
        component_cfg.compile = MagicMock(name="component_compile")

        with (
            patch(
                "winml.modelkit.loader.resolution.resolve_composite_components",
                return_value=components,
            ),
            patch(
                "winml.modelkit.config.generate_build_config",
                return_value=component_cfg,
            ),
            patch("winml.modelkit.commands.build._run_single_build") as mock_build,
            patch(
                "winml.modelkit.commands.build._validate_loader_tasks_for_model",
                return_value=None,
            ),
        ):
            result = runner.invoke(
                build,
                [
                    "-c",
                    config_file,
                    "-m",
                    "Qwen/Qwen3-0.6B",
                    "-o",
                    str(output_dir),
                    "--no-quant",
                    "--no-compile",
                ],
                obj={"debug": False},
            )

        assert result.exit_code == 0, result.output
        built_config = mock_build.call_args.kwargs["config"]
        assert built_config is component_cfg
        # Outer config has quant=None / compile=None → component cleared
        assert built_config.quant is None
        assert built_config.compile is None

    def test_composite_carries_outer_quant_when_component_quant_none(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Outer quant/compile are deep-copied even when component config has none."""
        from winml.modelkit.commands.build import build
        from winml.modelkit.compiler.configs import WinMLCompileConfig
        from winml.modelkit.quant.config import WinMLQuantizationConfig

        components = {"enc": "feature-extraction"}
        output_dir = tmp_path / "out"

        # Outer config (from -c) has explicit quant + compile sections. Control
        # it directly via _load_config so it isn't re-validated against a file.
        outer_cfg = MagicMock()
        outer_cfg.quant = WinMLQuantizationConfig(
            mode="static", samples=42, task="text-generation", model_id="outer/model"
        )
        outer_cfg.compile = WinMLCompileConfig()
        outer_cfg.loader = MagicMock(task="text-generation", model_type="qwen3")

        # Component config generated without quant (e.g. device/precision policy
        # produced no quant), but with populated loader metadata.
        component_cfg = MagicMock()
        component_cfg.quant = None
        component_cfg.compile = None
        component_cfg.loader = MagicMock(task="feature-extraction", model_type="qwen3")

        config_path = tmp_path / "config.json"
        config_path.write_text("{}")

        with (
            patch(
                "winml.modelkit.commands.build._load_config",
                return_value=outer_cfg,
            ),
            patch(
                "winml.modelkit.loader.resolution.resolve_composite_components",
                return_value=components,
            ),
            patch(
                "winml.modelkit.config.generate_build_config",
                return_value=component_cfg,
            ),
            patch("winml.modelkit.commands.build._run_single_build") as mock_build,
            patch(
                "winml.modelkit.commands.build._validate_loader_tasks_for_model",
                return_value=None,
            ),
        ):
            result = runner.invoke(
                build,
                [
                    "-c",
                    str(config_path),
                    "-m",
                    "Qwen/Qwen3-0.6B",
                    "-o",
                    str(output_dir),
                ],
                obj={"debug": False},
            )

        assert result.exit_code == 0, result.output
        built_config = mock_build.call_args.kwargs["config"]
        # Outer quant settings carried over (not None), deep-copied (not the same
        # object as the outer config's quant).
        assert built_config.quant is not None
        assert built_config.quant.samples == 42
        assert built_config.quant is not outer_cfg.quant
        # Calibration identity points at *this* sub-model, not the outer model,
        # even though the component config produced no quant of its own.
        assert built_config.quant.task == "feature-extraction"
        assert built_config.quant.model_id == "Qwen/Qwen3-0.6B"
        assert built_config.quant.model_type == "qwen3"
        # Outer compile carried over too, deep-copied.
        assert built_config.compile is not None
        assert built_config.compile is not outer_cfg.compile


class TestBuildSubmodel:
    """Test --submodel narrows a composite build to a single sub-model."""

    def test_submodel_builds_only_requested_component(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """--submodel builds only the named component."""
        from winml.modelkit.commands.build import build

        components = {
            "decoder_prefill": "feature-extraction",
            "decoder_gen": "text2text-generation",
        }
        output_dir = tmp_path / "out"

        fake_cfg = MagicMock()
        fake_cfg.quant = None
        fake_cfg.compile = None
        fake_cfg.loader = MagicMock(task=None)

        with (
            patch(
                "winml.modelkit.loader.resolution.resolve_composite_components",
                return_value=components,
            ),
            patch(
                "winml.modelkit.config.generate_build_config",
                return_value=fake_cfg,
            ),
            patch(
                "winml.modelkit.commands.build._run_single_build",
            ) as mock_single_build,
            patch(
                "winml.modelkit.commands.build._validate_loader_tasks_for_model",
                return_value=None,
            ),
        ):
            result = runner.invoke(
                build,
                [
                    "-m",
                    "Qwen/Qwen3-0.6B",
                    "-o",
                    str(output_dir),
                    "--submodel",
                    "decoder_prefill",
                ],
                obj={"debug": False},
            )

        assert result.exit_code == 0, result.output
        # Only the requested component is built.
        assert mock_single_build.call_count == 1
        assert mock_single_build.call_args.kwargs["cache_key"] == "decoder_prefill"

    def test_submodel_rejects_unknown_name(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """--submodel with an invalid name is a clean error."""
        from winml.modelkit.commands.build import build

        components = {
            "decoder_prefill": "feature-extraction",
            "decoder_gen": "text2text-generation",
        }

        fake_cfg = MagicMock()
        fake_cfg.quant = None
        fake_cfg.compile = None
        fake_cfg.loader = MagicMock(task=None)

        with (
            patch(
                "winml.modelkit.loader.resolution.resolve_composite_components",
                return_value=components,
            ),
            patch(
                "winml.modelkit.config.generate_build_config",
                return_value=fake_cfg,
            ),
            patch(
                "winml.modelkit.commands.build._run_single_build",
            ) as mock_single_build,
            patch(
                "winml.modelkit.commands.build._validate_loader_tasks_for_model",
                return_value=None,
            ),
        ):
            result = runner.invoke(
                build,
                [
                    "-m",
                    "Qwen/Qwen3-0.6B",
                    "-o",
                    str(tmp_path / "out"),
                    "--submodel",
                    "encoder",
                ],
                obj={"debug": False},
            )

        assert result.exit_code != 0
        assert "Unknown sub-model 'encoder'" in result.output
        assert "decoder_prefill" in result.output
        mock_single_build.assert_not_called()

    def test_submodel_rejects_non_composite(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """--submodel on a non-composite model is a clean error."""
        from winml.modelkit.commands.build import build

        fake_cfg = MagicMock()
        fake_cfg.quant = None
        fake_cfg.compile = None
        fake_cfg.loader = MagicMock(task=None)

        with (
            patch(
                "winml.modelkit.loader.resolution.resolve_composite_components",
                return_value=None,
            ),
            patch(
                "winml.modelkit.config.generate_build_config",
                return_value=fake_cfg,
            ),
            patch(
                "winml.modelkit.commands.build._run_single_build",
            ) as mock_single_build,
            patch(
                "winml.modelkit.commands.build._validate_loader_tasks_for_model",
                return_value=None,
            ),
        ):
            result = runner.invoke(
                build,
                [
                    "-m",
                    "prajjwal1/bert-tiny",
                    "-o",
                    str(tmp_path / "out"),
                    "--submodel",
                    "encoder",
                ],
                obj={"debug": False},
            )

        assert result.exit_code != 0
        assert "not a composite model" in result.output
        mock_single_build.assert_not_called()
