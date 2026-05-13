# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Happy-path E2E tests for the ``winml build`` CLI command.

This module keeps only real pipeline coverage that exercises export,
optimize, and ONNX passthrough behavior. Cheap CLI validation and flag
plumbing tests live under ``tests/unit/commands`` so they run in the
default test suite instead of the opt-in E2E lane.

See ``tests/e2e/BUILD_E2E_SCENARIOS.md`` for the full scenario
inventory.

Heavy HuggingFace happy-path tests are gated behind ``slow`` and
``network`` because they download real models from HuggingFace Hub.

The build command uses ``@click.pass_context`` and requires
``obj={"debug": True}`` (or ``True``) when invoked via ``CliRunner``.

A minimal hand-crafted config is sufficient for ONNX input (export is
skipped). Full HF pipeline tests use ``generate_build_config()`` (the
same API ``winml config`` calls) so that ``export.input_tensors`` is
populated correctly.
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
    compile section when an E2E scenario needs it.
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
    """Create a minimal WinMLBuildConfig JSON for ONNX-input E2E tests.

    Such a minimal config is sufficient for ONNX-input builds (no export
    step needed). It is NOT sufficient for a full HF build pipeline —
    use ``_generate_config_file`` for that.
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
