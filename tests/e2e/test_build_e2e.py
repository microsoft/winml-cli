# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E tests for the build CLI command.

These are heavy tests that exercise the full build pipeline. To keep
them fast, we use --no-quant --no-compile to skip quantization and
compilation stages.

The build command uses @click.pass_context and requires obj={"debug": False}.

We generate a proper config via ``generate_build_config()`` (same API
the ``winml config`` command calls) to ensure export input_tensors are
populated. A minimal hand-crafted config lacks I/O specs and will fail.

Markers:
    e2e: Full end-to-end test with real models
    slow: Tests that take > 30 seconds
    network: Requires network access to HuggingFace Hub
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.build import build


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.network]


@pytest.fixture(autouse=True)
def _mock_resolve_device():
    """Mock hardware detection to avoid failures in test environments."""
    with patch(
        "winml.modelkit.session.resolve_device_category",
        return_value=("cpu", ["cpu"]),
    ):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_config_file(
    tmp_path,
    model_id: str,
    task: str | None = None,
) -> str:
    """Generate a proper WinMLBuildConfig JSON file via the config API.

    This produces a complete config with input_tensors populated,
    which the build pipeline requires for dummy input generation.
    """
    from winml.modelkit.config import WinMLBuildConfig, generate_build_config

    cfg = generate_build_config(model_id, task=task, device="cpu", precision="fp32")
    # Force no quant/compile for fast tests
    if isinstance(cfg, WinMLBuildConfig):
        cfg.quant = None
        cfg.compile = None
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg.to_dict(), indent=2))
    return str(p)


def _make_minimal_config_file(tmp_path, task: str) -> str:
    """Create a minimal WinMLBuildConfig JSON (for ONNX input tests)."""
    config = {
        "loader": {"task": task},
        "export": {"opset_version": 17, "batch_size": 1},
        "optim": {},
        "quant": None,
        "compile": None,
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config))
    return str(p)


# ===========================================================================
# HF model build (export + optimize only)
# ===========================================================================


class TestBuildHF:
    """Build from HuggingFace model with --no-quant --no-compile."""

    def test_bert_text_classification(self, tmp_path: Path):
        """Full pipeline: export + optimize BERT text-classification.

        Uses --no-quant --no-compile so only export + optimize run.
        """
        config_path = _generate_config_file(
            tmp_path,
            "bert-base-uncased",
            task="text-classification",
        )
        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
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
            obj={"debug": False},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"build failed (exit {result.exit_code}):\n{result.output}"
        # Build should produce an output directory
        assert output_dir.exists()
        # Should contain at least one ONNX file
        onnx_files = list(output_dir.rglob("*.onnx"))
        assert len(onnx_files) >= 1, (
            f"No ONNX files found in {output_dir}. Contents: "
            f"{[str(p) for p in output_dir.rglob('*')]}"
        )


# ===========================================================================
# ONNX input build
# ===========================================================================


class TestBuildONNX:
    """Build from pre-exported ONNX file."""

    def test_onnx_passthrough(self, tmp_path: Path, onnx_model_path: Path):
        """ONNX input should skip export and run optimize only."""
        config_path = _make_minimal_config_file(tmp_path, "image-classification")
        output_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
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
            obj={"debug": False},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"build failed (exit {result.exit_code}):\n{result.output}"
        assert output_dir.exists()
