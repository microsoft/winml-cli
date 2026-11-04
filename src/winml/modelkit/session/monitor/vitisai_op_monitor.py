# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""AMD NPU operator profiling from VitisAI kernels in the ORT trace."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict
from configparser import ConfigParser
from configparser import Error as ConfigError
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from uuid import uuid4

from .ep_monitor import EPMonitor
from .op_metrics import OperatorMetrics, OpTraceResult


if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Self

    import onnxruntime as ort


logger = logging.getLogger(__name__)

_MINIMUM_VITISAI_EP_VERSION = "2.1000.81"
_PROVIDER = "VitisAIExecutionProvider"


def _event_span(event: dict[str, Any]) -> tuple[float, float]:
    timestamp, duration = event.get("ts"), event.get("dur")
    if not isinstance(timestamp, (int, float)) or not isinstance(duration, (int, float)):
        raise TypeError("VitisAI profile timestamp and duration must be numbers")
    if (
        isinstance(timestamp, bool)
        or isinstance(duration, bool)
        or not math.isfinite(timestamp)
        or not math.isfinite(duration)
        or timestamp < 0
        or duration < 0
    ):
        raise ValueError("Invalid VitisAI profile timestamp or duration")
    return float(timestamp), float(timestamp + duration)


class VitisAIOpMonitor(EPMonitor):
    """Collect basic operator timings independently of xrt-smi monitoring."""

    requires_session_teardown: ClassVar[bool] = True
    configures_session_options: ClassVar[bool] = True
    ep_name: ClassVar[str | None] = "vitisai"

    def __init__(
        self,
        level: Literal["basic"] = "basic",
        output_dir: Path | None = None,
    ) -> None:
        if level != "basic":
            raise ValueError(f"VitisAI profiling only supports level 'basic', got {level!r}")
        self._output_dir = Path(output_dir) if output_dir is not None else Path.cwd()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._profile_prefix = (
            self._output_dir / f"onnxruntime_vitisai_profile_{uuid4().hex}"
        ).resolve()
        self._onnx_model_path: Path | None = None
        self._onnx_op_types: dict[str, str] = {}
        self._output_to_name: dict[str, str] = {}
        self._warmup_iterations = 0
        self._measured_iterations: int | None = None
        self._entered = False
        self._result: OpTraceResult | None = None

    @property
    def output_dir(self) -> Path:
        """Directory containing the raw ONNX Runtime trace."""
        return self._output_dir

    @classmethod
    def is_available(cls) -> bool:
        """Discover the EP without requiring the optional xrt-smi utility."""
        try:
            import onnxruntime as ort

            if _PROVIDER in ort.get_available_providers():
                return True

            from .. import WinMLEPRegistry

            WinMLEPRegistry.instance()
            return any(
                getattr(device, "ep_name", None) == _PROVIDER for device in ort.get_ep_devices()
            )
        except (ImportError, OSError, RuntimeError) as exc:
            logger.warning("VitisAI provider discovery failed: %s", exc)
            return False

    def configure_session_options(self, session_options: ort.SessionOptions) -> None:
        """Enable profiling on the dedicated inference session."""
        session_options.enable_profiling = True
        session_options.profile_file_prefix = str(self._profile_prefix)

    def get_provider_options(self) -> dict[str, str]:
        """Enable the EP's kernel event collection, not just ORT node profiling."""
        return {"vitisai_profiling": "1"}

    @staticmethod
    @contextmanager
    def profiling_environment(output_dir: Path) -> Iterator[None]:
        """Enable XRT timelines before EP discovery without editing the user's INI.

        XRT caches configuration on first use, so callers must enter this scope
        before loading an AMD EP or creating the baseline inference session.
        """
        previous = os.environ.get("XRT_INI_PATH")
        source = (
            Path(previous)
            if previous
            else next(
                (
                    path
                    for path in (Path(sys.executable).parent / "xrt.ini", Path.cwd() / "xrt.ini")
                    if path.is_file()
                ),
                None,
            )
        )
        config = ConfigParser(interpolation=None)
        if source is not None:
            try:
                with source.open(encoding="utf-8-sig") as file:
                    config.read_file(file)
            except (OSError, UnicodeError, ConfigError) as exc:
                raise RuntimeError(f"Cannot read XRT configuration {source}: {exc}") from exc

        settings = {
            "Runtime": {"verbosity": "5"},
            "Debug": {"ml_timeline": "true", "ml_timeline_buffer_size": "2304k"},
            "ML_timeline_settings": {"buffer_size": "2304k"},
        }
        for section, options in settings.items():
            actual_section = next(
                (name for name in config.sections() if name.casefold() == section.casefold()),
                section,
            )
            if not config.has_section(actual_section):
                config.add_section(actual_section)
            config[actual_section].update(options)

        output_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="vitisai_xrt_", dir=output_dir) as directory:
            ini_path = (Path(directory) / "xrt.ini").resolve()
            with ini_path.open("w", encoding="utf-8") as file:
                config.write(file)
            os.environ["XRT_INI_PATH"] = str(ini_path)
            logger.debug("VitisAI profiling uses XRT configuration %s", ini_path)
            try:
                yield
            finally:
                if previous is None:
                    os.environ.pop("XRT_INI_PATH", None)
                else:
                    os.environ["XRT_INI_PATH"] = previous

    def set_onnx_model_path(self, onnx_model_path: Path) -> None:
        """Read graph output names without loading external tensor data."""
        import onnx
        from google.protobuf.message import DecodeError

        self._onnx_model_path = Path(onnx_model_path)
        self._output_to_name = {}
        try:
            model = onnx.load(self._onnx_model_path, load_external_data=False)
        except (OSError, ValueError, DecodeError) as exc:
            logger.warning("Could not load VitisAI source graph metadata: %s", exc)
            return
        self._output_to_name = {
            output: node.name
            for node in model.graph.node
            if node.name
            for output in node.output
            if output
        }

    def set_onnx_op_types(self, onnx_op_types: dict[str, str]) -> None:
        """Store exact ONNX node types for kernel-name resolution."""
        self._onnx_op_types = dict(onnx_op_types)

    def set_perf_window(self, warmup: int, measured_iterations: int) -> None:
        """Select completed measured runs rather than warmup invocations."""
        if warmup < 0 or measured_iterations < 0:
            raise ValueError("Perf window counts must be non-negative")
        self._warmup_iterations = warmup
        self._measured_iterations = measured_iterations

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("VitisAIOpMonitor already entered")
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self._result = self._parse_profile()

    def _resolve_node_name(self, kernel_name: str) -> str:
        # These are EP-generated wrappers, not model-specific naming conventions.
        match = re.fullmatch(r"subgraph_\((.+)\)_fix", kernel_name)
        if match:
            return match.group(1)
        match = re.fullmatch(r"subgraph_\((.+)\)", kernel_name)
        if match:
            output = re.sub(r"_vaip_\d+(_\d+)?$", "", match.group(1))
            return self._output_to_name.get(output, output)
        return kernel_name

    def _parse_profile(self) -> OpTraceResult:
        result = OpTraceResult(
            model=str(self._onnx_model_path) if self._onnx_model_path is not None else None,
            device="npu",
            tracing_level="basic",
            ep="vitisai",
            tracing_backend="onnxruntime",
        )
        try:
            profiles = list(self._output_dir.glob(f"{self._profile_prefix.name}*.json"))
            if len(profiles) > 1:
                raise ValueError("Multiple ORT profiles were written for one VitisAI perf window")
            if not profiles:
                result.status = "no_data"
                result.error = (
                    "No VitisAI ORT profile was written. Operator profiling requires "
                    f"VitisAI EP {_MINIMUM_VITISAI_EP_VERSION} or newer."
                )
                return result
            result.artifacts["profile"] = str(profiles[0])
            with profiles[0].open(encoding="utf-8") as profile_file:
                payload = json.load(profile_file)
            events = payload.get("traceEvents") if isinstance(payload, dict) else payload
            if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
                raise TypeError("VitisAI ORT profile must contain a list of event objects")
            result.operators, result.num_samples = self._parse_operator_events(events)
        except (OSError, ValueError, TypeError) as exc:
            result.status = "parse_failed"
            result.error = str(exc)
            return result

        if not result.operators:
            result.status = "no_data"
            result.error = (
                "No measured VitisAI Kernel events were found. A fused EP node alone "
                "does not provide operator timings. Check that VitisAI EP {_MINIMUM_VITISAI_EP_VERSION} or newer is installed."
            )
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

    def _parent_runs(
        self, events: list[dict[str, Any]]
    ) -> tuple[list[tuple[dict[str, Any], int]], int]:
        parents: dict[str, list[dict[str, Any]]] = defaultdict(list)
        model_runs = []
        for event in events:
            name = event.get("name")
            if (
                event.get("cat") == "Node"
                and isinstance(name, str)
                and name.startswith(f"{_PROVIDER}_")
                and name.endswith("_kernel_time")
            ):
                _event_span(event)
                parents[name].append(event)
            elif event.get("cat") == "Session" and name == "model_run":
                model_runs.append(_event_span(event))

        total_runs = (
            self._warmup_iterations + self._measured_iterations
            if self._measured_iterations is not None
            else None
        )
        indexed_parents = []
        if model_runs:
            model_runs.sort()
            if total_runs is not None and len(model_runs) != total_runs:
                raise ValueError(
                    f"VitisAI trace has {len(model_runs)} model runs; expected {total_runs}"
                )
            starts = [start for start, _ in model_runs]
            for occurrences in parents.values():
                for parent in occurrences:
                    start, end = _event_span(parent)
                    index = bisect_right(starts, start) - 1
                    if index < 0 or end > model_runs[index][1]:
                        raise ValueError("VitisAI parent node is outside the ORT model-run spans")
                    indexed_parents.append((parent, index))
            return indexed_parents, len(model_runs)

        # Older traces may omit model_run spans. Require one invocation of each
        # partition per run rather than guessing loop boundaries from divisibility.
        counts = {len(occurrences) for occurrences in parents.values()}
        if len(counts) > 1 or (counts and total_runs is not None and counts != {total_runs}):
            raise ValueError("Cannot align VitisAI parent invocations with completed model runs")
        for occurrences in parents.values():
            occurrences.sort(key=lambda event: event["ts"])
            indexed_parents.extend((parent, index) for index, parent in enumerate(occurrences))
        return indexed_parents, next(iter(counts), 0)

    def _parse_operator_events(
        self, events: list[dict[str, Any]]
    ) -> tuple[list[OperatorMetrics], int]:
        parents, run_count = self._parent_runs(events)
        kernels = []
        for event in events:
            if event.get("cat") != "Kernel":
                continue
            args = event.get("args", {})
            if isinstance(args, dict) and any(
                args.get(key) not in (None, "", _PROVIDER) for key in ("ep", "provider")
            ):
                continue
            _event_span(event)
            kernels.append(event)
        kernels.sort(key=lambda event: event["ts"])
        timestamps = [event["ts"] for event in kernels]
        samples: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        ep_types: dict[str, set[str]] = defaultdict(set)
        start_times: dict[str, float] = {}
        claimed: set[int] = set()
        stop = (
            self._warmup_iterations + self._measured_iterations
            if self._measured_iterations is not None
            else run_count
        )
        for parent, run in sorted(parents, key=lambda item: item[0]["ts"]):
            if not self._warmup_iterations <= run < stop:
                continue
            start, end = _event_span(parent)
            lo, hi = bisect_left(timestamps, start), bisect_right(timestamps, end)
            for index in range(lo, hi):
                kernel = kernels[index]
                if "pid" in parent and "pid" in kernel and parent["pid"] != kernel["pid"]:
                    continue
                if index in claimed:
                    raise ValueError("VitisAI kernel belongs to overlapping parent spans")
                claimed.add(index)
                name, args = kernel.get("name"), kernel.get("args", {})
                if not isinstance(name, str) or not name or not isinstance(args, dict):
                    raise ValueError("Invalid VitisAI kernel name or args")
                ep_type = args.get("op_type", "")
                if not isinstance(ep_type, str):
                    raise TypeError("VitisAI kernel op_type must be a string")
                op_path = self._resolve_node_name(name)
                samples[op_path][run] += float(kernel["dur"])
                if ep_type:
                    ep_types[op_path].add(ep_type)
                start_times.setdefault(op_path, float(kernel["ts"]))

        operators = []
        for op_path, runs in samples.items():
            onnx_type = self._onnx_op_types.get(op_path)
            native_types = ep_types[op_path]
            native_type = next(iter(native_types)) if len(native_types) == 1 else None
            operators.append(
                OperatorMetrics(
                    name=native_type or onnx_type or "Unknown",
                    op_path=op_path,
                    onnx_op_type=onnx_type,
                    start_time_us=start_times[op_path],
                    samples_us=[runs[index] for index in sorted(runs)],
                )
            )
        return operators, max(0, min(stop, run_count) - self._warmup_iterations)
