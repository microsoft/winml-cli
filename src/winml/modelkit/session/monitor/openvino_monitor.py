# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""OpenVINOMonitor — Intel OpenVINO EP per-op profiler via ORT.

Produces an :class:`OpTraceResult` with per-operator execution times at
``level="basic"`` (the only level OpenVINO exposes through this surface).
Profiling data is flushed per ``session.run()`` call — one CSV file per
inference — so :attr:`requires_session_teardown` is ``False``.

Two activation mechanisms are required (both contributed here):

1. ``ORT_OPENVINO_PERF_COUNT`` env var set to an absolute output directory
   path *before* the ``ort.InferenceSession`` is created.
   Source: ``onnxruntime/core/providers/openvino/backend_utils.cc ::
   GetPerfCountDumpPath()``.
2. ``load_config`` provider option containing
   ``{"<DEVICE>": {"PERF_COUNT": "YES"}}``.

The ``PERF_COUNT`` key inside ``load_config`` is owner-enforced per C-3
and cannot be overridden via ``extra_provider_options``.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from .ep_monitor import WinMLEPMonitor
from .op_metrics import OperatorMetrics, OpTraceResult, TraceStatus


if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Self


logger = logging.getLogger(__name__)

_VALID_DEVICES: frozenset[str] = frozenset({"CPU", "GPU", "NPU", "AUTO"})
_ENV_KEY: str = "ORT_OPENVINO_PERF_COUNT"


def _parse_openvino_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Parse a single OpenVINO profiling CSV into a list of row dicts.

    Expected columns (whitespace-tolerant): ``Layer Name``, ``Status``,
    ``Layer Type``, ``Real Time (us)``, ``Exec Type``. Rows with an empty
    ``Layer Name`` are skipped. Unparseable ``Real Time (us)`` values
    default to ``0.0``.
    """
    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw_row in reader:
            norm: dict[str, str] = {
                (k or "").strip(): (v or "").strip() for k, v in raw_row.items() if k
            }
            layer_name = norm.get("Layer Name", "")
            if not layer_name:
                continue
            layer_type = norm.get("Layer Type", "")
            real_time_str = norm.get("Real Time (us)", "0")
            try:
                real_time_us = float(real_time_str)
            except (ValueError, TypeError):
                real_time_us = 0.0
            rows.append(
                {
                    "layer_name": layer_name,
                    "layer_type": layer_type,
                    "real_time_us": real_time_us,
                }
            )
    return rows


class OpenVINOMonitor(WinMLEPMonitor):
    """OpenVINO EP per-op profiler.

    Each ``session.run()`` produces one CSV; ``__exit__`` merges all CSVs
    in ``output_dir`` into a single :class:`OpTraceResult` with
    ``OperatorMetrics.samples_us`` accumulating one entry per inference.
    ``output_dir`` (temp when ``None``) is never auto-cleaned.
    """

    requires_session_teardown: ClassVar[bool] = False
    ep_name: ClassVar[str | None] = "openvino"

    def __init__(
        self,
        level: str = "basic",
        output_dir: Path | None = None,
        device: str = "AUTO",
        extra_provider_options: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize the monitor.

        Args:
            level: Only ``"basic"`` is accepted; ``"detail"`` raises.
            output_dir: Where per-inference CSVs land. When ``None`` a
                per-monitor temp directory is minted and NOT auto-cleaned.
            device: One of ``"CPU"``, ``"GPU"``, ``"NPU"``, ``"AUTO"``.
            extra_provider_options: Merged into provider options.
                ``PERF_COUNT`` inside ``load_config`` is owner-enforced.
        """
        if level != "basic":
            raise ValueError(
                f"OpenVINOMonitor only supports level='basic', got {level!r}"
            )
        if device not in _VALID_DEVICES:
            raise ValueError(
                f"device must be one of {sorted(_VALID_DEVICES)}, got {device!r}"
            )
        self._level: str = level
        self._device: str = device
        self._output_dir: Path = (
            Path(output_dir)
            if output_dir is not None
            else Path(tempfile.mkdtemp(prefix="ov_profile_"))
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._extra: dict[str, str] = dict(extra_provider_options or {})
        self._entered: bool = False
        self._result: OpTraceResult | None = None
        self._onnx_op_types: dict[str, str] = {}
        # Saved in __enter__, restored in __exit__.
        self._prev_env: str | None = None
        self._prev_env_was_set: bool = False

    @property
    def output_dir(self) -> Path:
        """Directory where per-inference profiling CSV files are written."""
        return self._output_dir

    @classmethod
    def is_available(cls) -> bool:
        """Whether the OpenVINO EP is usable on this system."""
        try:
            import onnxruntime as ort
        except ImportError:
            return False

        if "OpenVINOExecutionProvider" in ort.get_available_providers():
            return True

        try:
            from ..ep_registry import WinMLEPRegistry
        except ImportError:
            return False

        try:
            WinMLEPRegistry.instance()
            return any(
                getattr(d, "ep_name", None) == "OpenVINOExecutionProvider"
                for d in ort.get_ep_devices()
            )
        except Exception as exc:
            logger.warning(
                "OpenVINOMonitor.is_available: WinML EP probe failed (%s: %s);"
                " reporting unavailable",
                type(exc).__name__,
                exc,
            )
            return False

    def get_provider_options(self) -> dict[str, str]:
        """Provider options for OpenVINO EP with owner-enforced PERF_COUNT.

        Owner-enforces ``PERF_COUNT: YES`` inside the ``load_config`` JSON
        for the target device; callers cannot disable or weaken it.
        """
        opts: dict[str, str] = dict(self._extra)
        existing: dict[str, Any] = {}
        raw_lc = opts.get("load_config")
        if raw_lc:
            try:
                existing = json.loads(raw_lc)
            except (json.JSONDecodeError, TypeError, ValueError):
                existing = {}
        device_cfg: dict[str, Any] = dict(existing.get(self._device, {}))
        device_cfg["PERF_COUNT"] = "YES"
        existing[self._device] = device_cfg
        opts["load_config"] = json.dumps(existing)
        return opts

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("OpenVINOMonitor already entered")
        self._entered = True
        prev = os.environ.get(_ENV_KEY)
        self._prev_env_was_set = prev is not None
        self._prev_env = prev
        os.environ[_ENV_KEY] = str(self._output_dir)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Restore the env var, then parse all CSVs in the output directory.

        Env var is restored before parsing so the parse pass never sees
        the monitor-injected value. Never suppresses caller exceptions.
        """
        if self._prev_env_was_set and self._prev_env is not None:
            os.environ[_ENV_KEY] = self._prev_env
        else:
            os.environ.pop(_ENV_KEY, None)
        self._result = self._parse_artifacts_safe()

    def _parse_artifacts_safe(self) -> OpTraceResult:
        """Wrap :meth:`_parse_artifacts` with the parse-failure contract."""
        try:
            return self._parse_artifacts()
        except Exception as exc:
            logger.warning("OpenVINOMonitor: artifact parse failed: %s", exc)
            return self._make_failure_result(status="parse_failed", error=str(exc))

    def set_onnx_op_types(self, onnx_op_types: dict[str, str]) -> None:
        """Store the ONNX ``node.name -> node.op_type`` map for CSV parsing.

        Called by ``WinMLSession.perf`` before ``__enter__``. Defensively
        copies the input to prevent caller mutation from corrupting the
        lookup table. Empty / no-graph input is a valid no-op.
        """
        self._onnx_op_types = dict(onnx_op_types)

    def _resolve_op_type(self, layer_name: str, layer_type: str | None = None) -> str:
        """Resolve op type: injected ONNX map, then CSV ``Layer Type``, then raw name."""
        mapped = self._onnx_op_types.get(layer_name)
        if mapped:
            return mapped
        if layer_type:
            return layer_type
        return layer_name

    def _parse_artifacts(self) -> OpTraceResult:
        """Merge all CSVs under ``output_dir`` into a single OpTraceResult."""
        csv_files = sorted(self._output_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(
                "OpenVINOMonitor: no CSV files produced at %s", self._output_dir
            )
            return self._make_failure_result(status="no_data", error=None)

        # Key: layer_name -> {layer_type, samples_us}.
        merged: dict[str, dict[str, Any]] = {}
        for csv_path in csv_files:
            for row in _parse_openvino_csv(csv_path):
                name = row["layer_name"]
                if name not in merged:
                    merged[name] = {"layer_type": row["layer_type"], "samples_us": []}
                merged[name]["samples_us"].append(row["real_time_us"])

        if not merged:
            return self._make_failure_result(status="no_data", error=None)

        avgs: dict[str, float] = {
            name: sum(d["samples_us"]) / len(d["samples_us"])
            for name, d in merged.items()
            if d["samples_us"]
        }
        total_avg_us = sum(avgs.values())

        operators: list[OperatorMetrics] = [
            OperatorMetrics(
                name=self._resolve_op_type(name, merged[name]["layer_type"]),
                op_path=name,
                duration_us=avg,
                percent_of_total=(avg / total_avg_us * 100) if total_avg_us > 0 else 0.0,
                samples_us=list(merged[name]["samples_us"]),
            )
            for name, avg in avgs.items()
        ]

        return OpTraceResult(
            model=None,
            device=self._device.lower(),
            tracing_level=self._level,
            ep="OpenVINOExecutionProvider",
            tracing_backend="openvino",
            operators=operators,
            summary={"device": self._device, "csv_count": len(csv_files)},
            num_samples=len(csv_files),
            artifacts={"csv_dir": str(self._output_dir)},
            status="ok",
        )

    def _make_failure_result(self, status: TraceStatus, error: str | None) -> OpTraceResult:
        """Build a minimal ``OpTraceResult`` for parse-time failures."""
        return OpTraceResult(
            model=None,
            device=self._device.lower(),
            tracing_level=self._level,
            ep="OpenVINOExecutionProvider",
            tracing_backend="openvino",
            operators=[],
            summary={},
            artifacts={"csv_dir": str(self._output_dir)},
            status=status,
            error=error,
        )
