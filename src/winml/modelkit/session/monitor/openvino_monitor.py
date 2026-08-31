# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""OpenVINO basic operator profiling through ONNX Runtime profiling."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from .ep_monitor import EPMonitor
from .op_metrics import OperatorMetrics, OpTraceResult


if TYPE_CHECKING:
    from typing import Self

    import onnxruntime as ort


logger = logging.getLogger(__name__)


class OpenVinoMonitor(EPMonitor):
    """Collect basic OpenVINO per-operator timings from an ORT profile."""

    requires_session_teardown: ClassVar[bool] = True
    configures_session_options: ClassVar[bool] = True
    ep_name: ClassVar[str | None] = "openvino"

    def __init__(
        self,
        level: Literal["basic"] = "basic",
        output_dir: Path | None = None,
        device: Literal["cpu", "npu"] = "npu",
    ) -> None:
        if level != "basic":
            raise ValueError(f"OpenVINO profiling only supports level 'basic', got {level!r}")
        normalized_device = device.lower()
        if normalized_device not in ("cpu", "npu"):
            raise ValueError(
                f"OpenVINO profiling only supports device 'cpu' or 'npu', got {device!r}"
            )

        self._level = level
        self._device = normalized_device
        self._output_dir = Path(output_dir) if output_dir is not None else Path.cwd()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._profile_prefix = (self._output_dir / "onnxruntime_profile").resolve()
        self._initial_profiles = {
            path.resolve(): self._artifact_signature(path) for path in self._profile_candidates()
        }
        self._entered = False
        self._onnx_model_path: Path | None = None
        self._onnx_op_types: dict[str, str] = {}
        self._measured_iterations: int | None = None
        self._result: OpTraceResult | None = None

    @property
    def output_dir(self) -> Path:
        """Directory containing the ONNX Runtime profile."""
        return self._output_dir

    @classmethod
    def is_available(cls) -> bool:
        """Return whether an OpenVINO execution provider is discoverable."""
        try:
            import onnxruntime as ort

            if "OpenVINOExecutionProvider" in ort.get_available_providers():
                return True

            from ..ep_registry import WinMLEPRegistry

            WinMLEPRegistry.instance()
            return any(
                getattr(device, "ep_name", None) == "OpenVINOExecutionProvider"
                for device in ort.get_ep_devices()
            )
        except (ImportError, OSError, RuntimeError) as exc:
            logger.warning(
                "OpenVinoMonitor.is_available failed (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return False

    def configure_session_options(self, session_options: ort.SessionOptions) -> None:
        """Enable ORT profiling on the monitored inference session."""
        session_options.enable_profiling = True
        session_options.profile_file_prefix = str(self._profile_prefix)

    def set_onnx_model_path(self, onnx_model_path: Path) -> None:
        """Store the source model path for trace metadata."""
        self._onnx_model_path = Path(onnx_model_path)

    def set_onnx_op_types(self, onnx_op_types: dict[str, str]) -> None:
        """Store ONNX node types used to enrich profile events."""
        self._onnx_op_types = dict(onnx_op_types)

    def set_perf_window(self, warmup: int, measured_iterations: int) -> None:
        """Store the measured iteration count for result metadata."""
        del warmup
        self._measured_iterations = measured_iterations

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("OpenVinoMonitor already entered")
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        del exc_type, exc_val, exc_tb
        self._result = self._parse_profile()

    def _parse_profile(self) -> OpTraceResult:
        profile_path = self._find_fresh_profile()
        if profile_path is None:
            return self._failure_result(
                "no_data",
                f"No fresh {self._profile_prefix.name}*.json profile was written.",
            )

        try:
            with profile_path.open(encoding="utf-8") as profile_file:
                payload = json.load(profile_file)
            events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
            if not isinstance(events, list):
                raise TypeError("ORT profile root must be a list or contain a traceEvents list")
            operators = self._parse_operator_events(events)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            result = self._failure_result("parse_failed", str(exc))
            result.artifacts["profile"] = str(profile_path)
            return result

        if not operators:
            result = self._failure_result(
                "no_data",
                "The ORT profile did not contain operator timing events.",
            )
            result.artifacts["profile"] = str(profile_path)
            return result

        total_us = sum(operator.total_us for operator in operators)
        for operator in operators:
            operator.duration_us = operator.avg_us
            operator.percent_of_total = operator.total_us / total_us * 100.0 if total_us else 0.0

        statistics_by_op = {
            operator.op_path: {
                "avg_us": operator.avg_us,
                "p90_us": operator.p90_us,
                "total_us": operator.total_us,
                "count": float(operator.sample_count),
            }
            for operator in operators
        }
        return OpTraceResult(
            model=str(self._onnx_model_path) if self._onnx_model_path is not None else None,
            device=self._device,
            tracing_level=self._level,
            operators=operators,
            ep="openvino",
            tracing_backend="onnxruntime",
            num_samples=self._measured_iterations
            or max(operator.sample_count for operator in operators),
            summary={"execute_us": total_us},
            statistics=statistics_by_op,
            artifacts={"profile": str(profile_path)},
        )

    def _parse_operator_events(self, events: list[Any]) -> list[OperatorMetrics]:
        samples_by_path: dict[str, list[float]] = defaultdict(list)
        op_type_by_path: dict[str, str] = {}
        start_by_path: dict[str, float] = {}

        for event in events:
            if not isinstance(event, dict):
                continue
            args = event.get("args")
            duration = event.get("dur")
            if not isinstance(args, dict) or not isinstance(duration, (int, float)):
                continue

            op_type = args.get("op_name")
            if not isinstance(op_type, str) or not op_type:
                continue
            node_name = args.get("node_name")
            event_name = event.get("name")
            op_path = (
                node_name
                if isinstance(node_name, str) and node_name
                else event_name
                if isinstance(event_name, str) and event_name
                else op_type
            )
            resolved_type = self._onnx_op_types.get(op_path, op_type)
            samples_by_path[op_path].append(float(duration))
            op_type_by_path[op_path] = resolved_type
            timestamp = event.get("ts")
            if isinstance(timestamp, (int, float)):
                start_by_path.setdefault(op_path, float(timestamp))

        return [
            OperatorMetrics(
                name=op_type_by_path[op_path],
                op_path=op_path,
                start_time_us=start_by_path.get(op_path),
                samples_us=samples,
            )
            for op_path, samples in samples_by_path.items()
        ]

    def _find_fresh_profile(self) -> Path | None:
        fresh_profiles = [
            path
            for path in self._profile_candidates()
            if self._artifact_signature(path) != self._initial_profiles.get(path.resolve())
        ]
        return max(fresh_profiles, key=lambda path: path.stat().st_mtime_ns, default=None)

    def _profile_candidates(self) -> list[Path]:
        return list(self._output_dir.glob(f"{self._profile_prefix.name}*.json"))

    @staticmethod
    def _artifact_signature(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _failure_result(
        self,
        status: Literal["no_data", "parse_failed"],
        error: str,
    ) -> OpTraceResult:
        return OpTraceResult(
            model=str(self._onnx_model_path) if self._onnx_model_path is not None else None,
            device=self._device,
            tracing_level=self._level,
            ep="openvino",
            tracing_backend="onnxruntime",
            status=status,
            error=error,
        )
