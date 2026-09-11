# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""TensorRT RTX basic operator profiling from the EP's Chrome trace."""

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from uuid import uuid4

from .ep_monitor import EPMonitor
from .op_metrics import OperatorMetrics, OpTraceResult


if TYPE_CHECKING:
    from typing import Self


logger = logging.getLogger(__name__)

_MINIMUM_TRTRTX_EP_VERSION = "2.30.49"


class NvTensorRTRTXMonitor(EPMonitor):
    """Collect GPU layer timings without renaming fused layers to ONNX nodes."""

    requires_session_teardown: ClassVar[bool] = True
    ep_name: ClassVar[str | None] = "nvtensorrtrtx"

    def __init__(
        self,
        level: Literal["basic"] = "basic",
        output_dir: Path | None = None,
    ) -> None:
        if level != "basic":
            raise ValueError(f"TensorRT RTX profiling only supports level 'basic', got {level!r}")
        self._output_dir = Path(output_dir) if output_dir is not None else Path.cwd()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        # A dedicated artifact prevents stale data and overlapping monitors from mixing.
        self._profile_path = (
            self._output_dir / f"onnxruntime_trtrtx_profile_{uuid4().hex}.json"
        ).resolve()
        self._onnx_model_path: Path | None = None
        self._onnx_op_types: dict[str, str] = {}
        self._warmup_iterations = 0
        self._measured_iterations: int | None = None
        self._entered = False
        self._result: OpTraceResult | None = None

    @property
    def output_dir(self) -> Path:
        """Directory containing the raw TensorRT RTX trace."""
        return self._output_dir

    @classmethod
    def is_available(cls) -> bool:
        """Return whether the TensorRT RTX execution provider is discoverable."""
        try:
            import onnxruntime as ort

            if "NvTensorRTRTXExecutionProvider" in ort.get_available_providers():
                return True

            from .. import WinMLEPRegistry

            WinMLEPRegistry.instance()
            return any(
                getattr(device, "ep_name", None) == "NvTensorRTRTXExecutionProvider"
                for device in ort.get_ep_devices()
            )
        except (ImportError, OSError, RuntimeError) as exc:
            logger.warning("TensorRT RTX provider discovery failed: %s", exc)
            return False

    def get_provider_options(self) -> dict[str, str]:
        """Enable EP profiling and direct its output to this monitor's artifact."""
        return {
            "nv_enable_profiling": "1",
            "nv_profiling_output_file": str(self._profile_path),
        }

    def set_onnx_model_path(self, onnx_model_path: Path) -> None:
        """Store the source graph path for result metadata."""
        self._onnx_model_path = Path(onnx_model_path)

    def set_onnx_op_types(self, onnx_op_types: dict[str, str]) -> None:
        """Store exact node-name matches for optional type enrichment."""
        self._onnx_op_types = dict(onnx_op_types)

    def set_perf_window(self, warmup: int, measured_iterations: int) -> None:
        """Exclude warmups independently for each EP context."""
        if warmup < 0 or measured_iterations < 0:
            raise ValueError("Perf window counts must be non-negative")
        self._warmup_iterations = warmup
        self._measured_iterations = measured_iterations

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("NvTensorRTRTXMonitor already entered")
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self._result = self._parse_profile()

    def _parse_profile(self) -> OpTraceResult:
        result = OpTraceResult(
            model=str(self._onnx_model_path) if self._onnx_model_path is not None else None,
            device="gpu",
            tracing_level="basic",
            ep="nvtensorrtrtx",
            tracing_backend="nv::trt",
        )
        try:
            with self._profile_path.open(encoding="utf-8") as profile_file:
                result.artifacts["profile"] = str(self._profile_path)
                payload = json.load(profile_file)
            events = payload.get("traceEvents") if isinstance(payload, dict) else payload
            if not isinstance(events, list):
                raise TypeError("TensorRT RTX profile must be a list or contain traceEvents")
            result.operators, result.num_samples = self._parse_operator_events(events)
        except FileNotFoundError:
            result.status = "no_data"
            result.error = (
                "No TensorRT RTX profile was written. Operator profiling requires "
                f"TensorRT RTX EP {_MINIMUM_TRTRTX_EP_VERSION} or newer, with support for "
                "nv_enable_profiling and nv_profiling_output_file."
            )
            return result
        except (OSError, ValueError, TypeError) as exc:
            result.status = "parse_failed"
            result.error = str(exc)
            return result

        if not result.operators:
            result.status = "no_data"
            result.error = "No measured nv::trt::layer events were found in the TensorRT RTX trace."
            return result

        total_us = sum(operator.total_us for operator in result.operators)
        for operator in result.operators:
            operator.duration_us = operator.avg_us
            operator.percent_of_total = operator.total_us / total_us * 100.0 if total_us else 0.0
            result.statistics[operator.op_path] = {
                "avg_us": operator.avg_us,
                "p90_us": operator.p90_us,
                "total_us": operator.total_us,
                "count": float(operator.sample_count),
            }
        result.summary["accel_execute_us"] = total_us
        return result

    def _parse_operator_events(self, events: list[Any]) -> tuple[list[OperatorMetrics], int]:
        # The EP uses pid for contexts and tid for runs, in first-seen run order.
        contexts: dict[int, dict[int, dict[str, float]]] = {}
        types_by_path: dict[str, set[str | None]] = defaultdict(set)
        nodes_by_path: dict[str, dict[str, None]] = defaultdict(dict)
        fused_paths: set[str] = set()
        for event in events:
            if not isinstance(event, dict):
                raise TypeError("TensorRT RTX trace events must be objects")
            if event.get("cat") != "nv::trt::layer" or event.get("ph") != "X":
                continue
            name, pid, tid, duration = (
                event.get("name"),
                event.get("pid"),
                event.get("tid"),
                event.get("dur"),
            )
            if (
                not isinstance(name, str)
                or not name
                or type(pid) is not int
                or type(tid) is not int
                or (type(duration) is not int and type(duration) is not float)
                or not math.isfinite(duration)
                or duration < 0
            ):
                raise ValueError("Invalid TensorRT RTX layer name, pid, tid, or duration")
            args = event.get("args", {})
            if not isinstance(args, dict):
                raise TypeError("TensorRT RTX layer args must be an object")
            onnx_nodes = args.get("onnx_nodes", "")
            if not isinstance(onnx_nodes, str):
                raise TypeError("TensorRT RTX onnx_nodes must be a string")
            if onnx_nodes:
                # The EP can emit either a literal escape or a decoded unit separator.
                node_names = list(
                    dict.fromkeys(re.split(r"\](?:\\u001f|\x1f)\[ONNX Layer: ", onnx_nodes))
                )
            else:
                node_names = [name] if name in self._onnx_op_types else []
            nodes_by_path[name].update(dict.fromkeys(node_names))
            if len(node_names) > 1:
                fused_paths.add(name)
            op_type = self._onnx_op_types.get(node_names[0]) if len(node_names) == 1 else None
            types_by_path[name].add(op_type)
            run = contexts.setdefault(pid, {}).setdefault(tid, {})
            run[name] = run.get(name, 0.0) + float(duration)

        samples_by_path: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        num_samples = 0
        stop = (
            self._warmup_iterations + self._measured_iterations
            if self._measured_iterations is not None
            else None
        )
        for pid, runs in contexts.items():
            if stop is not None and len(runs) < stop:
                raise ValueError(
                    f"TensorRT RTX context {pid} has {len(runs)} runs; "
                    f"expected at least {stop} including warmup"
                )
            retained = list(runs.values())[self._warmup_iterations : stop]
            num_samples = max(num_samples, len(retained))
            for index, run in enumerate(retained):
                for name, duration in run.items():
                    # Same native names across contexts contribute to one per-run total.
                    samples_by_path[name][index] += duration

        operators = []
        for name, samples in samples_by_path.items():
            types = types_by_path[name]
            op_type = next(iter(types)) if len(types) == 1 else None
            operators.append(
                OperatorMetrics(
                    name="Fused" if name in fused_paths else op_type or "Unknown",
                    op_path=name,
                    onnx_op_type=op_type,
                    onnx_nodes=[
                        {"name": node_name, "op_type": self._onnx_op_types.get(node_name)}
                        for node_name in nodes_by_path[name]
                    ]
                    or None,
                    samples_us=list(samples.values()),
                )
            )
        return operators, num_samples
