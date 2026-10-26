# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Opt-in ETL capture for Runtime GPU timestamps; requires ETW access."""

from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import uuid
from contextvars import ContextVar
from statistics import mean
from typing import TYPE_CHECKING
from xml.etree import ElementTree

from .gpu_timing import GPU_PROVIDER, RUNTIME_PROVIDER, GpuSubmission, RunKey, analyze_gpu_timing


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


# Runtime/ORT diagnostics from the Windows ML CI profile plus GPU scheduling.
# All keywords at verbose level; no kernel stack sampling or capture-state calls.
DIAGNOSTIC_PROVIDERS = (
    ("RuntimeCore", RUNTIME_PROVIDER),
    ("WindowsMLRuntime", "{1491d42b-0927-55b9-a765-10ec4bb1980f}"),
    ("ONNX", "{929dd115-1ecb-4cb5-b060-ebd4983c421d}"),
    ("OrtTraceLogging", "{3a26b1ff-7484-7484-7484-15261f42614d}"),
    ("WinML", "{d766d9ff-112c-4dac-9247-241cf99d123f}"),
    ("MachineLearning", "{bcad6aee-c08d-4f66-828c-4c43461a033d}"),
    ("WIL", "{6d1b249d-131b-468a-899b-fb0ad9551772}"),
    ("TestInProduction", "{50109fbd-6d85-5815-731e-c907eca1607b}"),
    ("DxgKrnl", "{802ec45a-1e99-4b83-9920-87c98277ba9d}"),
)


def _trace_events(path: Path) -> Iterator[ElementTree.Element]:
    """Stream complete events, releasing each subtree before reading the next."""
    with path.open("rb") as source:
        events = ElementTree.iterparse(source, events=("start", "end"))  # noqa: S314 -- local tracerpt output
        _, root = next(events)
        depth = 0
        for action, element in events:
            if action == "start":
                depth += 1
            else:
                if depth == 1:
                    try:
                        yield element
                    finally:
                        root.remove(element)
                        element.clear()
                depth -= 1


def analyze_trace(directory: Path) -> dict:
    """Runtime ETW analyze_trace."""
    markers = json.loads((directory / "windows.json").read_text(encoding="utf-8"))
    summary = (directory / "summary.xml").read_text(encoding="utf-8-sig")
    loss = re.search(r"Total Events\s+Lost\s+(\d+)", summary)
    if loss is None:
        raise RuntimeError(
            "Cannot verify ETW loss count from tracerpt summary; retain trace for inspection."
        )
    ns = {"e": "http://schemas.microsoft.com/win/2004/08/events/event"}
    starts, stops, samples = [], {}, []
    for event in _trace_events(directory / "events.xml"):
        system = event.find("e:System", ns)
        if system is None:
            raise RuntimeError("ETW event missing System")
        provider_element = system.find("e:Provider", ns)
        if provider_element is None:
            raise RuntimeError("ETW event missing Provider")
        provider = provider_element.get("Guid", "").lower()
        if provider not in (GPU_PROVIDER.lower(), RUNTIME_PROVIDER.lower()):
            continue
        execution = system.find("e:Execution", ns)
        if execution is None:
            raise RuntimeError("ETW event missing Execution")
        pid, tid = int(execution.attrib["ProcessID"]), int(execution.attrib["ThreadID"])
        if pid != markers["pid"]:
            continue
        data = {
            item.get("Name"): (item.text or "").strip()
            for item in event.findall("e:EventData/e:Data", ns)
        }
        name = event.findtext("e:RenderingInfo/e:Task", namespaces=ns)
        if name not in ("CgcGpuTiming", "PipelineRunStart", "PipelineRunStop"):
            continue
        key = RunKey(pid, int(data["RunId"]))
        time_element = system.find("e:TimeCreated", ns)
        if time_element is None:
            raise RuntimeError("ETW event missing TimeCreated")
        stamp = int(time_element.attrib["RawTime"])
        if provider == GPU_PROVIDER.lower() and name == "CgcGpuTiming":
            samples.append(
                GpuSubmission(
                    key,
                    *[
                        int(data[f])
                        for f in (
                            "Path",
                            "TimestampFrequency",
                            "SubmitStart",
                            "BodyStart",
                            "BodyEnd",
                            "SubmitEnd",
                        )
                    ],
                )
            )
        elif name == "PipelineRunStart" and int(data["Kind"]) == 0:
            starts.append((key, tid, stamp))
        elif name == "PipelineRunStop" and int(data["Kind"]) == 0:
            if key in stops:
                raise RuntimeError("Duplicate PipelineRunStop")
            stops[key] = (tid, stamp, int(data["Result"], 0))
    keys = []
    for begin, end, tid in markers["windows"]:
        matched = [
            (key, stamp) for key, thread, stamp in starts if thread == tid and begin <= stamp <= end
        ]
        if len(matched) != 1:
            raise RuntimeError("Cannot unambiguously match measured iteration to PipelineRunStart")
        key, stamp = matched[0]
        stop = stops.get(key)
        if stop is None or stop[0] != tid or not stamp <= stop[1] <= end or stop[2] != 0:
            raise RuntimeError("Measured Run has no successful matching PipelineRunStop")
        keys.append(key)
    result = analyze_gpu_timing(
        samples, keys, events_lost=int(loss[1]), consumer_drained=True
    ).to_dict()
    ordered = sorted(result["samples_ms"])
    position = (len(ordered) - 1) * 0.95
    lo = int(position)
    result["p95_ms"] = ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (
        position - lo
    )
    result["mean_ms"] = mean(ordered)
    result["aggregation"] = "per-run; single-window median"
    result["trace_directory"] = str(directory.resolve())
    result["collection_profile"] = markers.get("collection_profile", "runtime-gpu-timing")
    result["unavailable_providers"] = markers.get("unavailable_providers", [])
    return result


active_capture: ContextVar = ContextVar("runtime_gpu_capture", default=None)


class GpuCapture:
    """Runtime ETW GpuCapture."""
    def __init__(self, directory: Path) -> None:
        """Runtime ETW __init__."""
        self.directory = directory
        self.name = "WinMLCLI-" + uuid.uuid4().hex
        self.windows: list[tuple[int, int, int]] = []
        self.started = False
        self.unavailable_providers: list[dict[str, str]] = []

    def command(
        self, *args: str, optional_provider: bool = False, session_started: bool | None = None
    ) -> bool | None:
        """Runtime ETW command."""
        run = subprocess.run(args, capture_output=True, text=True, errors="replace")  # noqa: S603 -- fixed tracing tools, no shell
        # Track ownership before diagnostic I/O can fail (for example, a full disk).
        if run.returncode == 0 and session_started is not None:
            self.started = session_started
        try:
            with (self.directory / "capture.log").open("a", encoding="utf-8") as log:
                log.write(run.stdout + run.stderr)
        except OSError:
            # Cleanup must succeed even when the capture log is unwritable.
            if session_started is not False:
                raise
        if optional_provider and run.returncode & 0xFFFFFFFF == 0x80071068:
            return False
        if run.returncode:
            raise RuntimeError(
                f"GPU timing command failed ({run.returncode}): {args[0]}. "
                f"Check ETW permissions and {self.directory / 'capture.log'}."
            )
        return None

    def __enter__(self) -> GpuCapture:
        """Runtime ETW __enter__."""
        if os.name != "nt":
            raise RuntimeError("GPU ETW timing requires Windows.")
        self.directory.mkdir(parents=True, exist_ok=False)
        try:
            self.command(
                "logman",
                "create",
                "trace",
                self.name,
                "-ets",
                "-o",
                str(self.directory / "trace.etl"),
                "-bs",
                "256",
                "-nb",
                "64",
                "256",
                "-p",
                GPU_PROVIDER,
                "0xffffffffffffffff",
                "5",
                session_started=True,
            )
            for name, provider in DIAGNOSTIC_PROVIDERS:
                enabled = self.command(
                    "logman",
                    "update",
                    "trace",
                    self.name,
                    "-ets",
                    "-p",
                    provider,
                    "0xffffffffffffffff",
                    "5",
                    optional_provider=provider != RUNTIME_PROVIDER,
                )
                if enabled is False:
                    self.unavailable_providers.append({"name": name, "guid": provider})
        except BaseException:
            self.stop()
            raise
        self.token = active_capture.set(self)
        return self

    def qpc(self) -> int:
        """Runtime ETW qpc."""
        value = ctypes.c_longlong()
        if not ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(value)):
            raise RuntimeError("QueryPerformanceCounter failed")
        return int(value.value)

    def mark(self) -> tuple[int, int]:
        """Runtime ETW mark."""
        return (self.qpc(), ctypes.windll.kernel32.GetCurrentThreadId())

    def record(self, begin: tuple[int, int]) -> None:
        """Runtime ETW record."""
        self.windows.append((begin[0], self.qpc(), begin[1]))

    def stop(self) -> None:
        """Runtime ETW stop."""
        if self.started:
            self.command("logman", "stop", self.name, "-ets", session_started=False)

    def __exit__(self, *exc: object) -> None:
        """Runtime ETW __exit__."""
        active_capture.reset(self.token)
        self.stop()
        (self.directory / "windows.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "windows": self.windows,
                    "collection_profile": "runtime-ort-gpu-diagnostics-verbose",
                    "unavailable_providers": self.unavailable_providers,
                }
            ),
            encoding="utf-8",
        )

    def finish(self) -> dict:
        """Runtime ETW finish."""
        # -rts retains raw QPC timestamps for exact Run/marker matching.
        self.command(
            "tracerpt",
            str(self.directory / "trace.etl"),
            "-o",
            str(self.directory / "events.xml"),
            "-of",
            "XML",
            "-rts",
            "-y",
        )
        # tracerpt forbids combining raw timestamps and a summary in one pass.
        self.command(
            "tracerpt",
            str(self.directory / "trace.etl"),
            "-summary",
            str(self.directory / "summary.xml"),
            "-o",
            "NUL",
            "-y",
        )
        return analyze_trace(self.directory)
