# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E tests for the ``winml build`` CLI command.

This module covers three categories of scenarios for ``winml build``:

1. **Happy path** — each input source (HF, ONNX, random-weight) and each
   artifact destination (``--output-dir`` and ``--use-cache``) runs to
   completion with typical flag combinations.
2. **Bad path** — missing required arguments, invalid values, mutually
   exclusive flags, malformed config files. Each must surface as a
   user-facing error with a non-zero exit code, never a bare stack trace.
3. **Flag / option variations** — every behavior-bearing flag of
   ``winml build`` is exercised both present and absent, and every
   value of the ``--no-compile/--compile`` toggle is covered.

See ``tests/e2e/BUILD_E2E_SCENARIOS.md`` for the full scenario
inventory.

Heavy happy-path tests are gated behind ``slow`` and ``network`` so
they download real models from HuggingFace Hub. Bad-path / CLI
validation tests do not require network access and only carry the
``e2e`` marker.

The build command uses ``@click.pass_context`` and requires
``obj={"debug": True}`` (or ``True``) when invoked via ``CliRunner``.

A minimal hand-crafted config is sufficient for ONNX input (export is
skipped) and for argument-validation tests. Full HF pipeline tests use
``generate_build_config()`` (the same API ``winml config`` calls) so
that ``export.input_tensors`` is populated correctly.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.build import build


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
# Module-level marker: every test in this file is an E2E test.
# Individual tests opt into ``slow`` / ``network`` via class-level
# ``pytestmark`` (HF pipeline) or remain bare ``e2e`` (CLI validation).
pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def _mock_resolve_device():
    """Mock hardware detection to avoid failures in test environments.

    Also mocks the EP registry so the auto-EP-selection branch in
    ``build`` never tries to touch a real WinML SDK install.
    """
    mock_registry = MagicMock()
    mock_registry.is_ep_available.return_value = False

    with (
        patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("cpu", ["cpu"]),
        ),
        patch(
            "winml.modelkit.session.ep_registry.WinMLEPRegistry.get_instance",
            return_value=mock_registry,
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_config_file(
    tmp_path,
    model_id: str,
    task: str | None = None,
    *,
    with_compile: bool = False,
) -> str:
    """Generate a proper WinMLBuildConfig JSON file via the config API.

    Produces a complete config with ``export.input_tensors`` populated,
    which the build pipeline requires for dummy input generation.

    By default the quant and compile sections are cleared so the build
    is as fast as possible. Pass ``with_compile=True`` to keep the
    compile section from the generated config (used by ``--compile``
    flag tests).
    """
    from winml.modelkit.config import WinMLBuildConfig, generate_build_config

    cfg = generate_build_config(model_id, task=task, device="cpu", precision="fp32")
    if isinstance(cfg, WinMLBuildConfig):
        cfg.quant = None
        if not with_compile:
            cfg.compile = None
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg.to_dict(), indent=2))
    return str(p)


def _make_minimal_config_file(
    tmp_path,
    task: str = "image-classification",
    *,
    name: str = "config.json",
    compile_section: dict | None = None,
) -> str:
    """Create a minimal WinMLBuildConfig JSON (for ONNX input / CLI tests).

    Such a minimal config is sufficient for argument validation and for
    ONNX-input builds (no export step needed). It is NOT sufficient for
    a full HF build pipeline — use ``_generate_config_file`` for that.
    """
    config: dict = {
        "loader": {"task": task},
        "export": {"opset_version": 17, "batch_size": 1},
        "optim": {},
        "quant": None,
        "compile": compile_section,
    }
    p = tmp_path / name
    p.write_text(json.dumps(config))
    return str(p)


def _invoke(args: list[str], *, debug: bool = False):
    """Invoke the build command with a fresh CliRunner.

    ``catch_exceptions=False`` is intentionally NOT used here so the
    bad-path tests can assert on the exit code without being derailed
    by unhandled exceptions inside Click.
    """
    runner = CliRunner()
    return runner.invoke(build, args, obj={"debug": debug})


# ===========================================================================
# Bad-path tests — CLI / config validation. No network required.
# ===========================================================================


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
        # Build a config explicitly missing loader.task.
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
        assert "loader.task" in result.output or "task" in result.output.lower()

    def test_module_mode_requires_output_dir(self, tmp_path: Path):
        """An array (module-mode) config combined with ``--use-cache`` is invalid."""
        # Module-mode configs are JSON arrays. They aren't valid for cache mode.
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
        # The error may surface as either mutual-exclusion (when -o is also
        # missing) or as the explicit "not supported for module mode" message.
        out = result.output.lower()
        assert "module" in out or "cache" in out or "required" in out

    def test_module_array_non_object_entry(self, tmp_path: Path):
        """Module config entries must be JSON objects."""
        arr_path = tmp_path / "bad_modules.json"
        arr_path.write_text(json.dumps([{"loader": {"task": "x"}}, "not-an-object"]))
        result = _invoke(["-c", str(arr_path), "-o", str(tmp_path / "out")])
        assert result.exit_code != 0
        assert "object" in result.output.lower()

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


# ===========================================================================
# Flag-variation tests — exercise option plumbing via mocked pipeline.
# No network required.
# ===========================================================================


@pytest.fixture
def mock_run_single_build():
    """Mock the heavy ``_run_single_build`` so flag plumbing can be inspected.

    All flag-variation tests below use this to verify the value of each
    option flows through to the build pipeline without needing to run
    the full export/optimize stages.
    """
    with patch("winml.modelkit.commands.build._run_single_build", return_value=None) as mock:
        yield mock


class TestBuildFlagPassthrough:
    """Each behavior-bearing flag must propagate to ``_run_single_build``."""

    def _base_args(self, cfg: str, tmp_path: Path) -> list[str]:
        return ["-c", cfg, "-m", "microsoft/resnet-50", "-o", str(tmp_path / "out")]

    def test_defaults_no_flags(self, tmp_path: Path, mock_run_single_build: MagicMock):
        """With no optional flags, defaults are forwarded as-is."""
        cfg = _make_minimal_config_file(tmp_path)
        result = _invoke(self._base_args(cfg, tmp_path))
        assert result.exit_code == 0, result.output
        kw = mock_run_single_build.call_args.kwargs
        assert kw["rebuild"] is False
        assert kw["model_id"] == "microsoft/resnet-50"
        assert kw["extra_kwargs"] == {} or "trust_remote_code" not in kw["extra_kwargs"]

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


# ===========================================================================
# Happy-path HF builds — heavy, requires network.
# ===========================================================================


@pytest.mark.slow
@pytest.mark.network
class TestBuildHFHappyPath:
    """Build from HuggingFace model with the export+optimize pipeline."""

    def test_bert_text_classification(self, tmp_path: Path):
        """Full pipeline: export + optimize BERT text-classification."""
        config_path = _generate_config_file(
            tmp_path,
            "bert-base-uncased",
            task="text-classification",
        )
        output_dir = tmp_path / "output"

        result = CliRunner().invoke(
            build,
            [
                "-c",
                config_path,
                "-m",
                "bert-base-uncased",
                "-o",
                str(output_dir),
                "--no-quant",
                "--no-compile",
            ],
            obj={"debug": True},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"build failed (exit {result.exit_code}):\n{result.output}"
        assert output_dir.exists()
        onnx_files = list(output_dir.rglob("*.onnx"))
        assert len(onnx_files) >= 1, (
            f"No ONNX files found in {output_dir}. Contents: "
            f"{[str(p) for p in output_dir.rglob('*')]}"
        )

    def test_resnet_image_classification(self, tmp_path: Path):
        """Vision model end-to-end with explicit ``--ep`` and ``--device``."""
        config_path = _generate_config_file(
            tmp_path,
            "microsoft/resnet-50",
            task="image-classification",
        )
        output_dir = tmp_path / "output"

        result = CliRunner().invoke(
            build,
            [
                "-c",
                config_path,
                "-m",
                "microsoft/resnet-50",
                "-o",
                str(output_dir),
                "--no-quant",
                "--no-compile",
                "--no-analyze",
                "--ep",
                "qnn",
                "--device",
                "NPU",
            ],
            obj={"debug": True},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"build failed (exit {result.exit_code}):\n{result.output}"
        assert list(output_dir.rglob("*.onnx"))

    def test_rebuild_overwrites(self, tmp_path: Path):
        """``--rebuild`` re-runs the pipeline over an existing output dir."""
        config_path = _generate_config_file(
            tmp_path,
            "bert-base-uncased",
            task="text-classification",
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        # Drop a sentinel file to ensure --rebuild doesn't trip on the
        # directory already existing.
        (output_dir / "sentinel.txt").write_text("pre-existing")

        result = CliRunner().invoke(
            build,
            [
                "-c",
                config_path,
                "-m",
                "bert-base-uncased",
                "-o",
                str(output_dir),
                "--no-quant",
                "--no-compile",
                "--rebuild",
            ],
            obj={"debug": True},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"build failed (exit {result.exit_code}):\n{result.output}"
        assert list(output_dir.rglob("*.onnx"))


# ===========================================================================
# Happy-path ONNX passthrough — no HF download needed.
# ===========================================================================


class TestBuildONNXHappyPath:
    """Build from a pre-exported ONNX file (export step is skipped)."""

    def test_onnx_passthrough(self, tmp_path: Path, onnx_model_path: Path):
        """ONNX input should skip export and run optimize only."""
        config_path = _make_minimal_config_file(tmp_path)

        output_dir = tmp_path / "output"

        result = CliRunner().invoke(
            build,
            [
                "-c",
                config_path,
                "-m",
                str(onnx_model_path),
                "-o",
                str(output_dir),
                "--no-quant",
                "--no-compile",
            ],
            obj={"debug": True},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"build failed (exit {result.exit_code}):\n{result.output}"
        assert output_dir.exists()

    def test_onnx_passthrough_no_optimize(self, tmp_path: Path, onnx_model_path: Path):
        """``--no-optimize`` skips the optimize stage on an ONNX passthrough build."""
        config_path = _make_minimal_config_file(tmp_path)

        output_dir = tmp_path / "output"

        result = CliRunner().invoke(
            build,
            [
                "-c",
                config_path,
                "-m",
                str(onnx_model_path),
                "-o",
                str(output_dir),
                "--no-quant",
                "--no-compile",
                "--no-optimize",
            ],
            obj={"debug": True},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"build failed (exit {result.exit_code}):\n{result.output}"
        assert output_dir.exists()
