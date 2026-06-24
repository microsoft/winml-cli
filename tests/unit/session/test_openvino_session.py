# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for OpenVINOSession (OpenVINO Runtime backend for perf).

OpenVINO runs the raw ONNX directly on CPU, so these tests gate only on the
``openvino`` package being importable (CPU is always available when it is) --
not on the ORT OpenVINO EP, which is a different component.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from winml.modelkit.session import OpenVINOSession


if TYPE_CHECKING:
    from pathlib import Path


pytest.importorskip("openvino")

# OpenVINO CPU plugin is always present when the package is installed.
pytestmark = pytest.mark.openvino


class TestOpenVINOSession:
    """OpenVINOSession compile/run/perf surface on CPU."""

    def test_io_config_matches_onnx(self, simple_matmul_onnx: Path) -> None:
        session = OpenVINOSession(simple_matmul_onnx, device="cpu")
        io = session.io_config
        assert io["input_names"] == ["A"]
        assert io["output_names"] == ["C"]
        # input_value_ranges is ORT-only enrichment; precision is shared.
        assert "precision" in io

    def test_compile_is_idempotent(self, simple_matmul_onnx: Path) -> None:
        session = OpenVINOSession(simple_matmul_onnx, device="cpu")
        assert not session.is_compiled
        assert session.ep_name is None
        session.compile()
        compiled = session._compiled
        session.compile()  # second call is a no-op
        assert session._compiled is compiled
        assert session.is_compiled
        assert session.ep_name == "OpenVINOExecutionProvider"

    def test_run_produces_correct_output(
        self, simple_matmul_onnx: Path, sample_input: dict[str, np.ndarray]
    ) -> None:
        session = OpenVINOSession(simple_matmul_onnx, device="cpu")
        outputs = session.run(sample_input)
        assert set(outputs) == {"C"}
        assert outputs["C"].shape == (1, 4)
        assert outputs["C"].dtype == np.float32

    def test_run_auto_compiles(self, simple_matmul_onnx: Path) -> None:
        session = OpenVINOSession(simple_matmul_onnx, device="cpu")
        session.run({"A": np.zeros((1, 4), dtype=np.float32)})
        assert session.is_compiled

    def test_run_empty_inputs_raises(self, simple_matmul_onnx: Path) -> None:
        session = OpenVINOSession(simple_matmul_onnx, device="cpu")
        with pytest.raises(ValueError, match="inputs cannot be empty"):
            session.run({})

    def test_run_enforces_input_dtype(self, simple_matmul_onnx: Path) -> None:
        """Float64 input is coerced to the model's float32 without error."""
        session = OpenVINOSession(simple_matmul_onnx, device="cpu")
        outputs = session.run({"A": np.ones((1, 4), dtype=np.float64)})
        assert outputs["C"].dtype == np.float32

    def test_perf_records_samples(self, simple_matmul_onnx: Path) -> None:
        session = OpenVINOSession(simple_matmul_onnx, device="cpu")
        inputs = {"A": np.zeros((1, 4), dtype=np.float32)}
        with session.perf(warmup=2) as stats:
            for _ in range(7):
                session.run(inputs)
        assert stats.total_count == 7
        assert stats.count == 5  # warmup excluded
        assert stats.mean_ms > 0

    def test_running_model_path_is_input(self, simple_matmul_onnx: Path) -> None:
        session = OpenVINOSession(simple_matmul_onnx, device="cpu")
        session.compile()
        assert session.running_model_path == simple_matmul_onnx

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            OpenVINOSession(tmp_path / "does_not_exist.onnx")

    def test_unavailable_device_raises_friendly_error(self, simple_matmul_onnx: Path) -> None:
        """A device absent from Core().available_devices fails fast with a
        readable message instead of a raw backend stack trace. Uses a bogus
        device name so the test is hardware-independent."""
        session = OpenVINOSession(simple_matmul_onnx, device="bogus")
        with pytest.raises(RuntimeError, match=r"not available\. OpenVINO sees"):
            session.compile()
