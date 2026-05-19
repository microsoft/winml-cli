# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E tests for the perf CLI command.

Tests the perf CLI on a generated ONNX model fixture. ONNX inputs flow
through the same PerfBenchmark pipeline as HF inputs (issue #596), so this
exercises the optimize/[quantize]/[compile] path on a real artifact.
The perf command uses @click.pass_context and requires obj={}.

Note: HuggingFace model benchmarks are not exercised here because they
additionally require the export stage and HF model download.

Markers:
    e2e: Full end-to-end test
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.perf import perf


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = [pytest.mark.e2e]


# ===========================================================================
# ONNX benchmark (unified pipeline)
# ===========================================================================


class TestPerfONNXDirect:
    """Benchmark a pre-exported ONNX file through the unified PerfBenchmark path."""

    def test_onnx_benchmark_cpu(self, tmp_path: Path, onnx_model_path: Path):
        """ONNX benchmark on CPU with minimal iterations.

        Uses --device cpu --iterations 3 --warmup 1 for speed.
        Verifies JSON output file is created with expected schema.
        """
        output_file = tmp_path / "perf_result.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            [
                "-m",
                str(onnx_model_path),
                "--device",
                "cpu",
                "--iterations",
                "3",
                "--warmup",
                "1",
                "-o",
                str(output_file),
            ],
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"

        # Verify JSON output file exists and has expected structure
        assert output_file.exists(), f"Output file not created: {output_file}"
        data = json.loads(output_file.read_text())

        # Verify top-level schema
        assert "benchmark_info" in data
        assert "model_info" in data
        assert "latency_ms" in data
        assert "throughput" in data
        assert "raw_samples_ms" in data

        # Verify benchmark_info
        binfo = data["benchmark_info"]
        assert binfo["iterations"] == 3
        assert binfo["warmup"] == 1
        assert binfo["device"] == "cpu"

        # Verify latency stats are populated
        latency = data["latency_ms"]
        assert latency["mean"] > 0
        assert latency["min"] > 0
        assert latency["p50"] > 0

        # Verify model_info has input/output names
        minfo = data["model_info"]
        assert isinstance(minfo["input_names"], list)
        assert len(minfo["input_names"]) >= 1
        assert isinstance(minfo["output_names"], list)
        assert len(minfo["output_names"]) >= 1

        # Verify raw samples count matches iterations
        assert len(data["raw_samples_ms"]) == 3

    def test_onnx_benchmark_verbose(self, tmp_path: Path, onnx_model_path: Path):
        """Benchmark with --verbose should succeed and show debug output."""
        output_file = tmp_path / "verbose_result.json"

        runner = CliRunner()
        result = runner.invoke(
            perf,
            [
                "-m",
                str(onnx_model_path),
                "--device",
                "cpu",
                "--iterations",
                "2",
                "--warmup",
                "1",
                "-o",
                str(output_file),
                "--verbose",
            ],
            obj={},
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"perf failed (exit {result.exit_code}):\n{result.output}"
        assert output_file.exists()
        assert "Results saved to" in result.output
