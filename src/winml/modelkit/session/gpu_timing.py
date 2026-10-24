# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Runtime CGC ETW protocol and strict inference-body analysis.

Transport-independent: an ETW/TDH consumer supplies decoded events, while the
benchmark supplies *measured* Run keys (never inferred from event delivery time).
This module does not enable tracing or change existing CPU latency statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable

GPU_PROVIDER = "{655dde5b-3272-520c-d5aa-6114ae82f74a}"
RUNTIME_PROVIDER = "{3ed09fb9-ab2d-559e-b4cd-0cce5aca3732}"
GPU_KEYWORD = 0x1
RUN_KEYWORD = 0x2
INFORMATION_LEVEL = 4


class GpuTimingError(ValueError):
    """Requested GPU timing cannot produce a trustworthy measurement."""


@dataclass(frozen=True)
class RunKey:
    process_id: int
    run_id: int


@dataclass(frozen=True)
class GpuSubmission:
    key: RunKey
    path: int
    frequency: int
    submit_start: int
    body_start: int
    body_end: int
    submit_end: int

    def body_ms(self) -> float:
        ticks = (self.submit_start, self.body_start, self.body_end, self.submit_end)
        if self.frequency <= 0 or any(t < 0 or t >= 2**64 for t in ticks):
            raise GpuTimingError("Invalid GPU timestamp or frequency.")
        if not self.submit_start <= self.body_start <= self.body_end <= self.submit_end:
            raise GpuTimingError("GPU timestamps are not monotonic.")
        # Subtract integers before converting to float (GPU ticks may exceed 2**53).
        return (self.body_end - self.body_start) * 1000.0 / self.frequency


@dataclass(frozen=True)
class GpuTimingResult:
    samples_ms: tuple[float, ...]
    expected_runs: int
    recorded_runs: int
    replay_miss_runs: int
    replay_hit_runs: int
    outside_submissions: int

    def to_dict(self) -> dict:
        return {
            "source": "runtime-cgc-etw",
            "metric": "inference-body",
            "unit": "ms",
            "median_ms": median(self.samples_ms),
            "samples_ms": list(self.samples_ms),
            "expected_runs": self.expected_runs,
            "valid_runs": len(self.samples_ms),
            "recorded_runs": self.recorded_runs,
            "replay_miss_runs": self.replay_miss_runs,
            "replay_hit_runs": self.replay_hit_runs,
            "outside_submissions": self.outside_submissions,
            "missing_runs": 0,
            "events_lost": 0,
        }


def analyze_gpu_timing(
    submissions: Iterable[GpuSubmission],
    measured_runs: Iterable[RunKey],
    *,
    events_lost: int,
    consumer_drained: bool,
) -> GpuTimingResult:
    """Analyze a single measured window; exclude warmup via explicit Run keys.

    Fail closed on incomplete data and multi-submission Runs. Path 0 is counted
    but excluded because its body contains transfers. Call once per round if
    reproducing the harness's median-of-round-medians protocol.
    """
    if events_lost != 0 or not consumer_drained:
        raise GpuTimingError("ETW data incomplete: events lost or consumer not drained.")
    keys = list(measured_runs)
    if not keys or any(k.process_id <= 0 or k.run_id <= 0 for k in keys):
        raise GpuTimingError("Measured Runs require positive process IDs and RunIds.")
    if len(set(keys)) != len(keys):
        raise GpuTimingError("Duplicate measured Run key.")
    grouped: dict[RunKey, list[GpuSubmission]] = {key: [] for key in keys}
    outside = 0
    for sample in submissions:
        if sample.key in grouped:
            grouped[sample.key].append(sample)
        else:
            outside += 1
    missing = sum(not samples for samples in grouped.values())
    multi = sum(len(samples) > 1 for samples in grouped.values())
    if missing or multi:
        raise GpuTimingError(
            f"Incomplete/unsupported GPU timing: missing_runs={missing}, "
            f"multi_submission_runs={multi}."
        )
    counts = [0, 0, 0]
    bodies: list[float] = []
    for key in keys:
        sample = grouped[key][0]
        if sample.path not in (0, 1, 2):
            raise GpuTimingError(f"Unknown CGC submission Path: {sample.path}.")
        body = sample.body_ms()
        counts[sample.path] += 1
        if sample.path in (1, 2):
            bodies.append(body)
    if not bodies:
        raise GpuTimingError("No replay GPU samples; recorded-path body includes transfers.")
    return GpuTimingResult(tuple(bodies), len(keys), *counts, outside)
