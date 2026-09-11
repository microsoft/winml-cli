# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Process checkpoints with explicit unavailable values and byte evidence."""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from typing import Any

import psutil


logger = logging.getLogger(__name__)
MIB = 1024 * 1024


def metric(value: int | None, source: str, reason: str | None = None) -> dict[str, Any]:
    """Describe a measured byte value or an unavailable metric."""
    return {
        "value_bytes": value,
        "status": "valid" if value is not None else "unavailable",
        "source": source,
        "reason": reason,
    }


def get_rss_mb() -> float:
    """Return current process working set/RSS in MiB."""
    return psutil.Process(os.getpid()).memory_info().rss / MIB


def sample_vram(adapter_luid: str | None) -> dict[str, Any]:
    """Read all matching process memory nodes with per-counter diagnostics."""
    source = "PDH GPU Process Memory"
    reason = "unsupported_platform" if sys.platform != "win32" else "adapter_unresolved"
    result = {"local": metric(None, source, reason), "shared": metric(None, source, reason)}
    if sys.platform != "win32" or not adapter_luid:
        return result
    query = None
    try:
        from ._pdh import PdhQuery, memory_instances

        instances = memory_instances(os.getpid(), adapter_luid)
        if not instances:
            return {key: metric(None, source, "no_process_memory_instance") for key in result}
        query = PdhQuery()
        query.open()
        names: dict[str, list[str]] = {"local": [], "shared": []}
        for index, instance in enumerate(instances):
            for key, counter in (("local", "Local Usage"), ("shared", "Shared Usage")):
                name = f"{key}_{index}"
                names[key].append(name)
                query.add_counter(
                    name, chr(92) + f"GPU Process Memory({instance})" + chr(92) + counter
                )
        values = query.collect(timeout=0.25, interval=0.05)
        for key, counters in names.items():
            samples = [values.get(name) for name in counters]
            valid = all(
                isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in samples
            )
            result[key] = metric(
                sum(int(v) for v in samples if v is not None) if valid else None,
                source,
                None if valid else "counter_data_unavailable",
            )
            result[key]["instances"] = instances
            result[key]["diagnostics"] = {name: query.diagnostics.get(name) for name in counters}
    except Exception as exc:
        logger.debug("VRAM query unavailable", exc_info=True)
        result = {
            key: metric(None, source, f"query_failed: {type(exc).__name__}: {exc}")
            for key in result
        }
    finally:
        if query is not None:
            query.close()
    return result


def get_vram_mb(adapter_luid: str | None) -> tuple[float | None, float | None]:
    """Compatibility wrapper; unavailable counters return None, including no adapter."""
    sample = sample_vram(adapter_luid)
    values = [sample[key]["value_bytes"] for key in ("local", "shared")]
    local, shared = values
    return (
        float(local) / MIB if local is not None else None,
        float(shared) / MIB if shared is not None else None,
    )


class MemoryTracker:
    """Legacy checkpoints plus an optional pre-model snapshot; never peak fallback."""

    def __init__(
        self,
        adapter_luid: str | None,
        *,
        baseline: str,
        device: str,
        adapter_reason: str | None = None,
    ) -> None:
        self.adapter_luid = adapter_luid
        self.baseline = baseline
        self.device = device
        self.adapter_reason = adapter_reason
        self.checkpoints: dict[str, dict[str, Any]] = {}
        self.pid = os.getpid()
        self.process_created_at = psutil.Process(self.pid).create_time()

    def capture(self, phase: str) -> None:
        """Capture one live phase boundary, including raw bytes and status."""
        snapshot: dict[str, Any] = {"timestamp_ns": time.monotonic_ns()}
        try:
            info = psutil.Process(self.pid).memory_info()
            snapshot["rss"] = metric(info.rss, "psutil process working set / RSS")
            private = getattr(info, "private", None)
            snapshot["private"] = metric(
                private, "psutil private commit", "unsupported" if private is None else None
            )
        except psutil.Error as exc:
            snapshot["rss"] = metric(None, "psutil", str(exc))
            snapshot["private"] = metric(None, "psutil", str(exc))
        vram = sample_vram(self.adapter_luid)
        for name in ("local", "shared"):
            value = vram[name]
            if self.device == "cpu":
                value = {**metric(None, "PDH", "cpu_target"), "status": "not_applicable"}
            elif self.adapter_reason:
                value = metric(None, "PDH", self.adapter_reason)
            snapshot["vram_" + name] = value
        snapshot["completed_ns"] = time.monotonic_ns()
        self.checkpoints[phase] = snapshot

    def profile(self) -> dict[str, float | None]:
        """Derive compatible MiB fields only from available endpoints."""
        result: dict[str, float | None] = {}
        for key in ("rss", "vram_local", "vram_shared"):
            points = [
                self.checkpoints[phase][key]["value_bytes"]
                for phase in ("baseline", "after_compile", "after_inference")
            ]
            for phase, value in zip(
                ("baseline", "after_compile", "after_inference"), points, strict=True
            ):
                result[f"{key}_{phase}_mb"] = round(value / MIB, 2) if value is not None else None
            result[f"{key}_checkpoint_peak_mb"] = (
                round(max(points) / MIB, 2) if all(v is not None for v in points) else None
            )
            for label, start, end in (("model_load", 0, 1), ("inference", 1, 2), ("total", 0, 2)):
                a, b = points[start], points[end]
                result[f"{key}_{label}_delta_mb"] = (
                    round((b - a) / MIB, 2) if a is not None and b is not None else None
                )
            # Additive fields never change legacy baseline/delta/peak semantics.
            before = self.checkpoints.get("before_model_load", {}).get(key, {}).get("value_bytes")
            result[f"{key}_before_model_load_mb"] = (
                round(before / MIB, 2) if before is not None else None
            )
            for label, end in (
                ("model_factory", points[0]),
                ("total_from_before_model_load", points[2]),
            ):
                result[f"{key}_{label}_delta_mb"] = (
                    round((end - before) / MIB, 2)
                    if before is not None and end is not None
                    else None
                )
        return result

    def evidence(self) -> dict[str, Any]:
        """Return phase provenance without claiming a continuously sampled peak."""
        return {
            "schema_version": 2,
            "source": "winml perf process checkpoints v2",
            "baseline": self.baseline,
            "additional_baseline": (
                "before_model_factory_after_device_resolution; includes factory, inputs and session"
            ),
            "before_model_load_status": "observed"
            if "before_model_load" in self.checkpoints
            else "unavailable",
            "before_model_load_reason": None
            if "before_model_load" in self.checkpoints
            else "preloaded_component; pre-model phase not observed",
            "legacy_peak_phases": ["baseline", "after_compile", "after_inference"],
            "unit": "bytes",
            "legacy_unit": "MiB",
            "scope": "process",
            "pid": self.pid,
            "process_created_at": self.process_created_at,
            "adapter_luid": self.adapter_luid,
            "adapter_reason": self.adapter_reason,
            "peak_kind": "maximum_of_three_checkpoints",
            "checkpoints": self.checkpoints,
        }
