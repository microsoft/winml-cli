# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""QNN EP operator profiler using ORT.

Orchestrates the end-to-end profiling workflow:

1. Build an ORT ``InferenceSession`` with QNN EP and profiling options.
2. Run warmup + measured inference iterations.
3. Tear down the session to flush profiling data.
4. Parse the resulting CSV (basic) or run the profile viewer for QHAS
   (detail) post-processing.
5. Return a structured ``OpTraceResult``.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ...winml import add_ep_for_device
from ..base import OpTracer
from ..result import OperatorMetrics, OpTraceResult
from .csv_parser import parse_qnn_profiling_csv
from .viewer import find_qnn_sdk, run_qhas_viewer


if TYPE_CHECKING:
    from collections.abc import Iterator


logger = logging.getLogger(__name__)


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


def _csv_operator_metrics(samples: list[dict[str, Any]]) -> list[OperatorMetrics]:
    """Aggregate per-sample CSV operator records into ``OperatorMetrics``.

    Each operator's duration and percentage are computed against the metadata
    of the *same* sample — the accelerator cycle total and cycle->US factor
    differ slightly between inferences — then averaged across every sample the
    operator appears in. Operators are keyed by ``op_id`` so identically-named
    ops in different positions stay separate. The result is sorted by duration
    descending.
    """
    acc: dict[int, dict[str, Any]] = {}

    for sample in samples:
        meta = sample["metadata"]
        total_cycles = meta.get("accel_execute_cycles", 0)
        accel_us = meta.get("accel_execute_us", 0)
        cycle_to_us = accel_us / total_cycles if total_cycles > 0 else 0.0

        for op in sample["samples"]:
            oid = op["op_id"]
            entry = acc.setdefault(
                oid,
                {"name": op["name"], "op_id": oid, "duration_us": 0.0, "percent": 0.0, "count": 0},
            )
            entry["duration_us"] += op["cycles"] * cycle_to_us
            entry["percent"] += op["cycles"] / total_cycles * 100 if total_cycles > 0 else 0.0
            entry["count"] += 1

    metrics = [
        OperatorMetrics(
            name=entry["name"],
            op_path=entry["name"],
            op_id=entry["op_id"],
            duration_us=entry["duration_us"] / entry["count"],
            percent_of_total=entry["percent"] / entry["count"],
        )
        for entry in acc.values()
    ]
    # duration is the headline metric; percent breaks ties when US timing is
    # absent (so durations collapse to 0 but cycle shares still differ).
    metrics.sort(key=lambda m: (m.duration_us, m.percent_of_total), reverse=True)
    return metrics


def _csv_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline metadata across samples (HVX threads constant; cycles/US averaged)."""
    if not samples:
        return {"hvx_threads": 0, "accel_execute_cycles": 0, "accel_execute_us": 0}

    n = len(samples)
    metas = [s["metadata"] for s in samples]
    hvx_threads = metas[0]["hvx_threads"]
    if any(m["hvx_threads"] != hvx_threads for m in metas):
        logger.warning(
            "HVX thread count varies across samples (%s); using first sample's value %s",
            [m["hvx_threads"] for m in metas],
            hvx_threads,
        )
    return {
        "hvx_threads": hvx_threads,
        "accel_execute_cycles": round(sum(m["accel_execute_cycles"] for m in metas) / n),
        "accel_execute_us": round(sum(m["accel_execute_us"] for m in metas) / n),
    }


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    """Temporarily change CWD and restore on exit.

    QNN EP writes ``*_schematic.bin`` into the process CWD, so we
    change to the output directory before creating the session.
    """
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


class QNNProfiler(OpTracer):
    """QNN EP operator profiler using ORT.

    Parameters
    ----------
    onnx_path:
        Path to the ONNX model (or ``*_ctx.onnx`` context binary).
    output_dir:
        Directory for profiling artifacts (CSV, log, schematic, QHAS).
    level:
        Profiling level: ``"basic"`` (cycle counts per operator) or
        ``"detail"`` (full QHAS with roofline / DMA traffic).
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
        """Check if QNN EP is available for profiling."""
        try:
            import onnxruntime as ort

            return "QNNExecutionProvider" in ort.get_available_providers()
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
            Number of un-measured warmup iterations (session compile /
            JIT overhead).
        """
        import onnxruntime as ort

        self.output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = (self.output_dir / "profiling_output.csv").resolve()
        csv_path.unlink(missing_ok=True)
        options = self._build_session_options(ort)
        provider_options = self._build_provider_options(csv_path)
        if not add_ep_for_device(
            options, "QNNExecutionProvider", ort.OrtHardwareDeviceType.NPU, provider_options
        ):
            raise RuntimeError("Failed to add QNNExecutionProvider for NPU device.")

        # CWD must be output_dir so schematic.bin lands there.
        with _working_directory(self.output_dir):
            session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=options,
            )

            inputs = self._generate_inputs(session)

            # Warmup (not measured).
            for _ in range(warmup):
                session.run(None, inputs)

            # Measured iterations.
            for _ in range(iterations):
                session.run(None, inputs)

            # Tear down session to flush profiling data.
            del session

        # ---- Post-processing ----
        return self._collect_results(csv_path, iterations, warmup)

    # ------------------------------------------------------------------
    # ORT configuration builders
    # ------------------------------------------------------------------

    def _build_session_options(self, ort_module: Any) -> Any:
        """Create ``ort.SessionOptions`` with profiling config entries.

        ``ep.context_*`` (EPContext / cached-context) entries are only needed for
        ``detail`` tracing, so they are gated on the tracing level.
        """
        options = ort_module.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        if self.level == "detail":
            options.add_session_config_entry("ep.context_enable", "1")
            options.add_session_config_entry("ep.context_embed_mode", "0")
        return options

    def _build_provider_options(self, csv_path: Path) -> dict[str, str]:
        """Build QNN EP provider options dict.

        - ``basic`` mode uses ``profiling_level=detailed`` (per-op cycles).
        - ``detail`` mode uses ``profiling_level=optrace`` (full QHAS).
        """
        profiling_level = "optrace" if self.level == "detail" else "detailed"

        return {
            "htp_performance_mode": "high_performance",
            "htp_graph_finalization_optimization_mode": "3",
            "enable_htp_fp16_precision": "1",
            "profiling_level": profiling_level,
            "profiling_file_path": str(csv_path),
        }

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

    def _collect_results(self, csv_path: Path, iterations: int, warmup: int) -> OpTraceResult:
        """Parse profiling artifacts into an ``OpTraceResult``."""
        artifacts: dict[str, str] = {}
        qnn_log = Path(str(csv_path) + "_qnn.log")

        if csv_path.is_file():
            artifacts["csv"] = str(csv_path)
        if qnn_log.is_file():
            artifacts["qnn_log"] = str(qnn_log)

        # Locate schematic if present (detail mode).
        schematic = self._find_schematic()
        if schematic is not None:
            artifacts["schematic"] = str(schematic)

        # --- Detail mode: attempt QHAS post-processing ---
        if self.level == "detail" and qnn_log.is_file():
            qhas_result = self._try_qhas(qnn_log, schematic, artifacts)
            if qhas_result is not None:
                return qhas_result

        # --- Fallback / basic mode: parse CSV ---
        if csv_path.is_file():
            return self._from_csv(csv_path, iterations, warmup, artifacts)

        # No artifacts at all -- return empty result.
        logger.warning("No profiling artifacts found in %s", self.output_dir)
        return OpTraceResult(
            model=self.onnx_path.name,
            device="npu",
            tracing_level=self.level,
            ep="QNNExecutionProvider",
            tracing_backend="qnn",
            num_samples=0,
            artifacts=artifacts,
        )

    def _find_schematic(self) -> Path | None:
        """Find a ``*_schematic.bin`` file in the output directory."""
        schematics = list(self.output_dir.glob("*_schematic.bin"))
        if schematics:
            return schematics[0]
        return None

    def _try_qhas(
        self,
        qnn_log: Path,
        schematic: Path | None,
        artifacts: dict[str, str],
    ) -> OpTraceResult | None:
        """Attempt QHAS post-processing; return result or ``None``."""
        import json as _json

        if schematic is None or not schematic.is_file():
            logger.info("No schematic found; falling back to CSV for detail mode")
            return None

        qhas_output = self.output_dir / "qhas_output.json"
        result_path = run_qhas_viewer(qnn_log, schematic, qhas_output, sdk_root=find_qnn_sdk())

        if result_path is None or not result_path.is_file():
            logger.info("QHAS viewer did not produce output; falling back")
            return None

        artifacts["qhas"] = str(result_path)
        from .qhas_parser import parse_qhas

        qhas_data = _json.loads(result_path.read_text(encoding="utf-8"))
        parsed = parse_qhas(qhas_data)

        operators = [
            OperatorMetrics(
                name=op["name"],
                op_path=op["op_path"],
                duration_us=op["duration_us"],
                percent_of_total=op["percent_of_total"],
                dominant_path_us=op.get("dominant_path_us"),
                num_htp_ops=op.get("num_htp_ops"),
                dram_read_bytes=op.get("dram_read_bytes"),
                dram_write_bytes=op.get("dram_write_bytes"),
                vtcm_read_bytes=op.get("vtcm_read_bytes"),
                vtcm_write_bytes=op.get("vtcm_write_bytes"),
                vtcm_hit_ratio=op.get("vtcm_hit_ratio"),
            )
            for op in parsed["operators"]
        ]

        return OpTraceResult(
            model=self.onnx_path.name,
            device="npu",
            tracing_level="detail",
            ep="QNNExecutionProvider",
            tracing_backend="qnn",
            operators=operators,
            summary=parsed["summary"],
            artifacts=artifacts,
        )

    def _from_csv(
        self,
        csv_path: Path,
        iterations: int,
        warmup: int,
        artifacts: dict[str, str],
    ) -> OpTraceResult:
        """Build an ``OpTraceResult`` from the basic CSV parser.

        The CSV records every execute call, warmup runs included. Warmup
        carries graph-finalization / JIT overhead, so the first ``warmup``
        samples are dropped; the remaining samples — which must number
        ``iterations`` — feed the operator metrics.
        """
        samples = parse_qnn_profiling_csv(csv_path)

        measured = samples[warmup:]
        if len(measured) != iterations:
            raise ValueError(
                f"Expected {iterations} measured sample(s) after skipping {warmup} "
                f"warmup, got {len(measured)} from {len(samples)} total."
            )

        operators = _csv_operator_metrics(measured)

        return OpTraceResult(
            model=self.onnx_path.name,
            device="npu",
            tracing_level=self.level,
            ep="QNNExecutionProvider",
            tracing_backend="qnn",
            operators=operators,
            num_samples=len(measured),
            summary=_csv_summary(measured),
            artifacts=artifacts,
        )
