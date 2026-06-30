# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CPU EP operator profiler using ORT's built-in profiling.

Orchestrates the end-to-end profiling workflow:

1. Build an ORT ``InferenceSession`` on the CPU EP with profiling enabled.
2. Run warmup + measured inference iterations.
3. Flush the profiling JSON via ``end_profiling()``.
4. Parse the Chrome-tracing JSON into per-operator aggregates.
5. Return a structured ``OpTraceResult``.

Unlike the QNN profiler (which relies on QNN-EP-specific provider options and
CSV output), this uses the profiler built into ORT itself, so it works for any
model that runs on the CPU EP.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ...winml import add_ep_for_device
from ..base import OpTracer
from ..result import OperatorMetrics, OpTraceResult
from .profile_parser import build_node_maps, parse_ort_profile


logger = logging.getLogger(__name__)

# Prefix ORT prepends to the profiling JSON it writes. The full file name is
# ``<prefix>_<timestamp>.json``; ``end_profiling()`` returns the resolved path.
_PROFILE_PREFIX = "onnxruntime_profile"


def _ort_type_to_numpy(ort_type: str) -> np.dtype:
    """Map an ORT tensor type string to a NumPy dtype."""
    mapping: dict[str, np.dtype] = {
        "tensor(float)": np.dtype("float32"),
        "tensor(float16)": np.dtype("float16"),
        "tensor(double)": np.dtype("float64"),
        "tensor(int32)": np.dtype("int32"),
        "tensor(int64)": np.dtype("int64"),
        "tensor(int8)": np.dtype("int8"),
        "tensor(uint8)": np.dtype("uint8"),
        "tensor(bool)": np.dtype("bool"),
    }
    return mapping.get(ort_type, np.dtype("float32"))


def _resolve_shape(shape: list, default_dim: int = 1) -> list[int]:
    """Replace symbolic or ``None`` dimensions with concrete values."""
    return [default_dim if not isinstance(d, int) or d <= 0 else d for d in shape]


class CPUProfiler(OpTracer):
    """CPU EP operator profiler using ORT's built-in profiler.

    Parameters
    ----------
    onnx_path:
        Path to the ONNX model.
    output_dir:
        Directory for the profiling JSON artifact.
    level:
        Profiling level. Only ``"basic"`` is meaningful for the CPU EP; the
        argument is accepted for interface parity with other tracers.
    """

    def __init__(
        self,
        onnx_path: Path,
        *,
        output_dir: Path,
        level: str = "basic",
    ) -> None:
        super().__init__(onnx_path, output_dir=output_dir, level=level)

    def is_available(self) -> bool:
        """The CPU EP is always present in any ORT build."""
        try:
            import onnxruntime as ort

            return "CPUExecutionProvider" in ort.get_available_providers()
        except (ImportError, AttributeError):
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, iterations: int = 5, warmup: int = 2) -> OpTraceResult:
        """Run profiling and return structured results.

        Parameters
        ----------
        iterations:
            Number of measured inference iterations.
        warmup:
            Number of un-measured warmup iterations (dropped from aggregates).
        """
        import onnxruntime as ort

        self.output_dir.mkdir(parents=True, exist_ok=True)

        options = self._build_session_options(ort)
        if not add_ep_for_device(options, "CPUExecutionProvider", ort.OrtHardwareDeviceType.CPU):
            raise RuntimeError("Failed to add CPUExecutionProvider for CPU device.")

        session = ort.InferenceSession(str(self.onnx_path), sess_options=options)
        inputs = self._generate_inputs(session)

        # Warmup + measured runs. ORT's profiler records every run; the warmup
        # samples are dropped during parsing.
        for _ in range(warmup + iterations):
            session.run(None, inputs)

        # Flush profiling data and capture the JSON path ORT wrote.
        profile_path = Path(session.end_profiling())
        del session

        return self._collect_results(profile_path, iterations=iterations, warmup=warmup)

    # ------------------------------------------------------------------
    # ORT configuration
    # ------------------------------------------------------------------

    def _build_session_options(self, ort_module: Any) -> Any:
        """Create ``ort.SessionOptions`` with profiling enabled.

        The profile file is written under ``output_dir`` by prefixing the path,
        mirroring how the QNN profiler places its artifacts. Thread spinning is
        enabled so per-op timings reflect steady-state latency rather than
        thread-pool wake-up overhead.
        """
        options = ort_module.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(self.output_dir / _PROFILE_PREFIX)
        options.add_session_config_entry("session.intra_op.allow_spinning", "1")
        options.add_session_config_entry("session.inter_op.allow_spinning", "1")
        return options

    # ------------------------------------------------------------------
    # Input generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_inputs(session: Any) -> dict[str, np.ndarray]:
        """Generate random inputs matching the model's I/O specification."""
        inputs: dict[str, np.ndarray] = {}
        for inp in session.get_inputs():
            shape = _resolve_shape(inp.shape)
            dtype = _ort_type_to_numpy(inp.type)
            inputs[inp.name] = np.random.rand(*shape).astype(dtype)
        return inputs

    # ------------------------------------------------------------------
    # Result collection
    # ------------------------------------------------------------------

    def _collect_results(
        self,
        profile_path: Path,
        *,
        iterations: int,
        warmup: int,
    ) -> OpTraceResult:
        """Parse the ORT profiling JSON into an ``OpTraceResult``."""
        artifacts: dict[str, str] = {}
        if profile_path.is_file():
            artifacts["profile_json"] = str(profile_path)
        else:
            logger.warning("No profiling JSON found at %s", profile_path)
            return OpTraceResult(
                model=self.onnx_path.name,
                device="cpu",
                tracing_level=self.level,
                ep="CPUExecutionProvider",
                tracing_backend="ort",
                num_samples=0,
                artifacts=artifacts,
            )

        name_to_type, output_to_name = build_node_maps(self.onnx_path)
        parsed = parse_ort_profile(
            profile_path,
            name_to_type,
            output_to_name,
            warmup=warmup,
            iterations=iterations,
        )

        operators = [
            OperatorMetrics(
                name=op["name"],
                op_path=op["op_path"],
                duration_us=op["duration_us"],
                percent_of_total=op["percent_of_total"],
            )
            for op in parsed["operators"]
        ]

        total_us = sum(op.duration_us for op in operators)
        return OpTraceResult(
            model=self.onnx_path.name,
            device="cpu",
            tracing_level=self.level,
            ep="CPUExecutionProvider",
            tracing_backend="ort",
            operators=operators,
            num_samples=parsed["num_samples"],
            summary={"total_op_us": total_us, "num_operators": len(operators)},
            artifacts=artifacts,
        )
