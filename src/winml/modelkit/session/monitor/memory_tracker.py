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
import threading
import time
from typing import TYPE_CHECKING, Any

import psutil


if TYPE_CHECKING:
    from ._gpu_baseline import GpuBaselineDevice

logger = logging.getLogger(__name__)
MIB = 1024 * 1024


def metric(value: int | None, source: str, reason: str | None = None) -> dict[str, Any]:
    """Describe a measured byte value or an unavailable metric."""
    return {
        "value_bytes": value,
        "status": "valid" if value is not None else "unavailable",
        "source": source,
        "reason": reason,
        "value_origin": "measured_zero"
        if value == 0
        else "measured"
        if value is not None
        else "unknown",
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

        # Retry while the caller is paused at this phase; never borrow a later phase.
        deadline = time.monotonic() + 0.25
        while True:
            instances = memory_instances(os.getpid(), adapter_luid)
            if instances or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        if not instances:
            return {
                key: {
                    **metric(None, source, "no_process_memory_instance; zero_not_proven"),
                    "discovery_status": "absent_unconfirmed",
                }
                for key in result
            }
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
            key: {
                **metric(None, source, f"query_failed: {type(exc).__name__}: {exc}"),
                "discovery_status": "error",
            }
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
        self._gpu_baseline_device: GpuBaselineDevice | None = None
        self.gpu_baseline_preparation: dict[str, Any] = {"status": "not_attempted"}

    def _prepare_gpu_baseline(self) -> None:
        """Materialize missing process counters before any model work, never infer zero."""
        if (
            sys.platform != "win32"
            or self.device != "gpu"
            or not self.adapter_luid
            or self.adapter_reason
        ):
            self.gpu_baseline_preparation = {"status": "not_applicable"}
            return
        start = time.monotonic_ns()
        before = sample_vram(self.adapter_luid)
        if not any(v.get("discovery_status") == "absent_unconfirmed" for v in before.values()):
            self.gpu_baseline_preparation = {
                "status": "existing_counters"
                if all(v["value_bytes"] is not None for v in before.values())
                else "counter_unavailable",
                "initial_observation": before,
            }
            return
        try:
            from ._gpu_baseline import GpuBaselineDevice

            self._gpu_baseline_device = GpuBaselineDevice(self.adapter_luid)
            deadline = time.monotonic() + 2.0
            while True:
                observed = sample_vram(self.adapter_luid)
                valid = all(v["value_bytes"] is not None for v in observed.values())
                if valid or time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            self.gpu_baseline_preparation = {
                "status": "initialized" if valid else "counter_unavailable",
                "method": "D3D12 device on selected LUID; no model or GPU work",
                "initial_observation": before,
                "prepared_observation": observed,
                "retained_until": "after final checkpoint or failure cleanup",
            }
        except Exception as exc:
            # Measurement setup must not prevent a provider from running.
            self.gpu_baseline_preparation = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "initial_observation": before,
            }
        self.gpu_baseline_preparation.update(
            started_ns=start, completed_ns=time.monotonic_ns(), adapter_luid=self.adapter_luid
        )

    def close(self) -> None:
        """Keep baseline overhead stable through inference, then release it."""
        device, self._gpu_baseline_device = self._gpu_baseline_device, None
        if device is not None:
            device.close()

    def capture(self, phase: str) -> None:
        """Capture one live phase boundary, including raw bytes and status."""
        if phase == "before_model_load":
            if phase in self.checkpoints:
                return  # Never replace the original baseline with a later observation.
            self._prepare_gpu_baseline()
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
            "gpu_baseline_preparation": self.gpu_baseline_preparation,
            "gpu_delta_scope": (
                "Measured process change from the recorded endpoint; includes model/provider "
                "allocations, excludes retained model-free baseline device. Not minimum model VRAM."
            ),
            "checkpoints": self.checkpoints,
        }


def _device_memory(adapter_luid: str | None) -> tuple[dict[str, float | None], dict[str, str]]:
    """Use discovered process nodes while preserving missing-counter reasons."""
    sample = sample_vram(adapter_luid)
    values: dict[str, float | None] = {}
    errors: dict[str, str] = {}
    for key in ("local", "shared"):
        name = "vram_" + key
        value = sample[key]["value_bytes"]
        values[name] = value / MIB if value is not None else None
        if value is None:
            errors[name] = sample[key]["reason"] or "counter_unavailable_or_invalid"
    return values, errors


def _rounded(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None


def _delta(after: float | None, before: float | None) -> float | None:
    return _rounded(after - before) if after is not None and before is not None else None


def _rss_high_water() -> tuple[float | None, str | None]:
    """Read Windows' process-lifetime peak, independently of Python polling."""
    try:
        peak = getattr(psutil.Process(os.getpid()).memory_info(), "peak_wset", None)
    except psutil.Error as exc:
        return None, f"query_failed:{type(exc).__name__}"
    if not isinstance(peak, (int, float)) or not math.isfinite(peak) or peak < 0:
        return None, "process_high_water_unavailable"
    return peak / (1024 * 1024), None


class ProcessMemoryTracker:
    """Capture checkpoints and sampled peaks over one explicitly named lifecycle.

    The tracker does not resolve devices or read model properties. Binding must
    happen before model construction; a late binding cannot backfill a baseline.
    Sampling records aggregates rather than growing a per-iteration buffer.
    """

    def __init__(
        self,
        *,
        adapter_luid: str | None = None,
        scope: str = "before_model_load_through_inference",
        interval: float = 0.05,
    ) -> None:
        self.adapter_luid = adapter_luid
        self.scope = scope
        self.interval = interval
        self.checkpoints: dict[str, dict[str, float | None]] = {}
        self.availability: dict[str, dict[str, str]] = {}
        self._peaks: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._last_sample_at: float | None = None
        self._sample_interval_count = 0
        self._sample_interval_total = 0.0
        self._sample_interval_max: float | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = time.monotonic()
        self._ended: float | None = None
        self._binding_before_load = False
        self._os_peak_before: float | None = None
        self._os_peak_after: float | None = None
        self._os_peak_error: str | None = None
        self._load_baseline: dict[str, float | None] | None = None
        self._load_ready: dict[str, float | None] | None = None
        self._load_peaks: dict[str, float] = {}
        self._load_counts: dict[str, int] = {}
        self._load_active = False
        self._load_os_before: float | None = None
        self._load_os_ready: float | None = None
        self._baseline_owner: MemoryTracker | None = None

    def _observe(self) -> tuple[dict[str, float | None], dict[str, str]]:
        values, errors = _device_memory(self.adapter_luid)
        try:
            values["rss"] = get_rss_mb()
        except psutil.Error as exc:
            values["rss"] = None
            errors["rss"] = f"query_failed:{type(exc).__name__}"
        return values, errors

    def checkpoint(self, phase: str) -> None:
        """Record an explicit lifecycle boundary without triggering model work."""
        during_load = self._load_active
        values, errors = self._observe()
        self.checkpoints[phase] = values
        self.availability[phase] = errors
        # Explicit observations belong to the load peak, but are not polling samples.
        with self._lock:
            if during_load and self._load_active:
                for key, value in values.items():
                    if value is not None:
                        self._load_peaks[key] = max(value, self._load_peaks.get(key, value))

    def bind_before_load(self, adapter_luid: str | None, *, device: str = "auto") -> None:
        """Capture a GPU baseline after EP resolution but before model creation."""
        if self._binding_before_load:
            return
        self.adapter_luid = adapter_luid
        self._binding_before_load = True
        self._baseline_owner = MemoryTracker(
            adapter_luid, baseline="before_model_load", device=device
        )
        self._baseline_owner.capture("before_model_load")
        values, errors = _device_memory(adapter_luid)
        self.checkpoints["baseline"].update(values)
        for key in values:
            self.availability["baseline"].pop(key, None)
        self.availability["baseline"].update(errors)
        # The model-specific window excludes generic imports/EP discovery.
        self._load_os_before, _ = _rss_high_water()
        self._load_baseline, reasons = self._observe()
        self.availability["load_baseline"] = reasons
        with self._lock:
            self._load_peaks = {k: v for k, v in self._load_baseline.items() if v is not None}
            self._load_active = True

    def record_model_ready(self) -> None:
        """Freeze load-only observations before inputs, warmup and inference."""
        if self._load_baseline is None:
            return
        self._load_ready, reasons = self._observe()
        self.availability["model_ready"] = reasons
        self._load_os_ready, _ = _rss_high_water()
        with self._lock:
            self._load_active = False
            for key, value in self._load_ready.items():
                if value is not None:
                    self._load_peaks[key] = max(value, self._load_peaks.get(key, value))

    def start(self) -> None:
        """Start sampling before the model-loading call."""
        self._os_peak_before, self._os_peak_error = _rss_high_water()
        self.checkpoint("baseline")
        self._thread = threading.Thread(target=self._sample, name="winml-memory", daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.is_set():
            during_load = self._load_active
            values, _ = self._observe()
            sampled_at = time.monotonic()
            with self._lock:
                if self._last_sample_at is not None:
                    elapsed = sampled_at - self._last_sample_at
                    self._sample_interval_count += 1
                    self._sample_interval_total += elapsed
                    self._sample_interval_max = max(self._sample_interval_max or 0.0, elapsed)
                self._last_sample_at = sampled_at
                for key, value in values.items():
                    if value is not None:
                        self._counts[key] = self._counts.get(key, 0) + 1
                        self._peaks[key] = max(value, self._peaks.get(key, value))
                        if during_load and self._load_active:
                            self._load_counts[key] = self._load_counts.get(key, 0) + 1
                            self._load_peaks[key] = max(value, self._load_peaks.get(key, value))
            self._stop.wait(self.interval)

    def stop(self) -> None:
        """Stop sampling on both successful execution and exceptions."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._ended is None:
            self._os_peak_after, self._os_peak_error = _rss_high_water()
            self._ended = time.monotonic()
        self.close()

    def close(self) -> None:
        """Release retained baseline preparation after sampling has stopped."""
        if self._baseline_owner is not None:
            self._baseline_owner.close()

    def to_dict(self) -> dict[str, float | None]:
        """Return legacy checkpoint/delta fields without treating null as zero."""
        result = {}
        for family in ("rss", "vram_local", "vram_shared"):
            values = [
                self.checkpoints.get(p, {}).get(family)
                for p in ("baseline", "after_compile", "after_inference")
            ]
            for phase in ("baseline", "after_load", "after_compile", "after_inference"):
                result[f"{family}_{phase}_mb"] = _rounded(
                    self.checkpoints.get(phase, {}).get(family)
                )
            observed = [value for value in values if value is not None]
            result[f"{family}_checkpoint_peak_mb"] = (
                _rounded(max(observed)) if len(observed) == len(values) else None
            )
            for name, end, start in (("model_load", 1, 0), ("inference", 2, 1), ("total", 2, 0)):
                result[f"{family}_{name}_delta_mb"] = _delta(values[end], values[start])
        return result

    def metadata(self) -> dict[str, Any]:
        """Describe the actual scope and missing checkpoint observations."""
        return {
            "schema_version": 3,
            "gpu_baseline_preparation": (
                self._baseline_owner.gpu_baseline_preparation
                if self._baseline_owner is not None
                else {"status": "not_applicable"}
            ),
            "unit": "MiB",
            "pid": os.getpid(),
            "adapter_luid": self.adapter_luid,
            "scope": self.scope,
            "rss_baseline": self.scope.split("_through_", 1)[0],
            "gpu_baseline": (
                "after_ep_resolution_before_model_load"
                if self._binding_before_load
                else self.scope.split("_through_", 1)[0]
            ),
            "missing_reasons": self.availability,
            "delta_semantics": "signed process change; not model-only allocation",
            "checkpoint_peak_semantics": "maximum of baseline/after_compile/after_inference only",
            "local_shared_semantics": (
                "separate counters; may overlap on unified memory; do not sum"
            ),
        }

    def load_memory(self) -> dict[str, Any]:
        """Describe measured load cost, without claiming a universal minimum."""
        result: dict[str, Any] = {
            "version": 1,
            "unit": "MiB",
            "pid": os.getpid(),
            "adapter_luid": self.adapter_luid,
            "scope": "after_runtime_setup_before_model_construction_to_ready_before_inputs",
            "includes": (
                "loading, weight preparation and compilation required by this artifact/path"
            ),
            "excludes": "generic runtime setup, benchmark input allocation, warmup and inference",
            "interpretation": (
                "observed load footprint for this configuration; "
                "not a certified minimum hardware capacity"
            ),
        }
        if self._load_baseline is None or self._load_ready is None:
            result.update(
                status="unavailable",
                reason="model was already loaded or readiness was not observed",
            )
            return result
        result["status"] = "measured"
        for family in ("rss", "vram_local", "vram_shared"):
            baseline = self._load_baseline.get(family)
            ready = self._load_ready.get(family)
            peak = self._load_peaks.get(family)
            method = "sampled_and_endpoint_maximum"
            if (
                family == "rss"
                and self._load_os_before is not None
                and self._load_os_ready is not None
                and self._load_os_ready > self._load_os_before
            ):
                peak = max(peak or 0, self._load_os_ready)
                method = "new_os_process_high_water_during_load"
            result[family] = {
                "baseline_mb": _rounded(baseline),
                "ready_mb": _rounded(ready),
                "peak_mb": _rounded(peak),
                "peak_extra_mb": _delta(peak, baseline),
                "ready_extra_mb": _delta(ready, baseline),
                "peak_method": method,
                "sample_count": self._load_counts.get(family, 0),
                "missing_reason": (
                    self.availability["load_baseline"].get(family)
                    or self.availability["model_ready"].get(family)
                ),
            }
        result["os_rss_high_water_before_mb"] = self._load_os_before
        result["os_rss_high_water_at_ready_mb"] = self._load_os_ready
        result["limitations"] = [
            "Sampling may miss short allocations; OS high-water is attributable only "
            + "when it increases during the load window.",
            "RSS measures resident pages, not private commit or exact required capacity.",
            "GPU counters are sampled; local/shared counters may overlap on UMA "
            + "and must not be summed.",
        ]
        return result

    def sampled(self) -> dict[str, Any]:
        """Return sampling coverage and absolute peaks separately from deltas."""
        with self._lock:
            return {
                "scope": self.scope,
                "pid": os.getpid(),
                "adapter_luid": self.adapter_luid,
                "unit": "MiB",
                "configured_poll_delay_sec": self.interval,
                "observed_mean_interval_sec": (
                    self._sample_interval_total / self._sample_interval_count
                    if self._sample_interval_count
                    else None
                ),
                "observed_max_interval_sec": self._sample_interval_max,
                "duration_sec": (self._ended or time.monotonic()) - self._started,
                "sample_count": max(self._counts.values(), default=0),
                "rss_samples": self._counts.get("rss", 0),
                "local_samples": self._counts.get("vram_local", 0),
                "shared_samples": self._counts.get("vram_shared", 0),
                "rss_peak_mb": self._peaks.get("rss"),
                "local_peak_mb": self._peaks.get("vram_local"),
                "shared_peak_mb": self._peaks.get("vram_shared"),
                "os_rss_peak_before_mb": self._os_peak_before,
                "os_rss_peak_after_mb": self._os_peak_after,
                "os_rss_peak_scope": (
                    "process lifetime through measurement stop; not reset at baseline"
                ),
                "os_rss_peak_missing_reason": self._os_peak_error,
                "limitations": (
                    "sampled observations, not an exact continuous peak; "
                    "GPU samples begin after adapter binding; Python polling may miss "
                    "short allocations or pause while native code holds the GIL"
                ),
            }
