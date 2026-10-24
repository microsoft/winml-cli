# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Opt-in ETL capture for Runtime GPU timestamps; requires ETW access."""
from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
import ctypes
import json
import os
import subprocess
import uuid
import re
import xml.etree.ElementTree as ET
from statistics import mean
from .gpu_timing import GPU_PROVIDER, RUNTIME_PROVIDER, GpuSubmission, RunKey, analyze_gpu_timing


def analyze_trace(directory: Path) -> dict:
    markers = json.loads((directory / "windows.json").read_text(encoding="utf-8"))
    summary = (directory / "summary.xml").read_text(encoding="utf-8-sig")
    loss = re.search(r"Total Events\s+Lost\s+(\d+)", summary)
    if loss is None:
        raise RuntimeError("Cannot verify ETW loss count from tracerpt summary; retain trace for inspection.")
    ns = {"e": "http://schemas.microsoft.com/win/2004/08/events/event"}
    starts, stops, samples = [], {}, []
    for event in ET.parse(directory / "events.xml").getroot():
        system = event.find("e:System", ns)
        provider = system.find("e:Provider", ns).get("Guid", "").lower()
        if provider not in (GPU_PROVIDER.lower(), RUNTIME_PROVIDER.lower()):
            continue
        execution = system.find("e:Execution", ns)
        pid, tid = int(execution.get("ProcessID")), int(execution.get("ThreadID"))
        if pid != markers["pid"]:
            continue
        data = {item.get("Name"): (item.text or "").strip()
                for item in event.findall("e:EventData/e:Data", ns)}
        key = RunKey(pid, int(data["RunId"]))
        name = event.findtext("e:RenderingInfo/e:Task", namespaces=ns)
        stamp = int(system.find("e:TimeCreated", ns).get("RawTime"))
        if provider == GPU_PROVIDER.lower() and name == "CgcGpuTiming":
            samples.append(GpuSubmission(key, *[int(data[f]) for f in
                ("Path", "TimestampFrequency", "SubmitStart", "BodyStart", "BodyEnd", "SubmitEnd")]))
        elif name == "PipelineRunStart" and int(data["Kind"]) == 0:
            starts.append((key, tid, stamp))
        elif name == "PipelineRunStop" and int(data["Kind"]) == 0:
            if key in stops:
                raise RuntimeError("Duplicate PipelineRunStop")
            stops[key] = (tid, stamp, int(data["Result"], 0))
    keys = []
    for begin, end, tid in markers["windows"]:
        matched = [(key, stamp) for key, thread, stamp in starts if thread == tid and begin <= stamp <= end]
        if len(matched) != 1:
            raise RuntimeError("Cannot unambiguously match measured iteration to PipelineRunStart")
        key, stamp = matched[0]
        stop = stops.get(key)
        if stop is None or stop[0] != tid or not stamp <= stop[1] <= end or stop[2] != 0:
            raise RuntimeError("Measured Run has no successful matching PipelineRunStop")
        keys.append(key)
    result = analyze_gpu_timing(samples, keys, events_lost=int(loss[1]), consumer_drained=True).to_dict()
    ordered = sorted(result["samples_ms"])
    position = (len(ordered) - 1) * .95
    lo = int(position)
    result["p95_ms"] = ordered[lo] + (ordered[min(lo+1, len(ordered)-1)] - ordered[lo]) * (position-lo)
    result["mean_ms"] = mean(ordered)
    result["aggregation"] = "per-run; single-window median"
    result["trace_directory"] = str(directory.resolve())
    return result

active_capture: ContextVar = ContextVar("runtime_gpu_capture", default=None)


class GpuCapture:
    def __init__(self, directory: Path):
        self.directory = directory
        self.name = "WinMLCLI-" + uuid.uuid4().hex
        self.windows = []
        self.started = False

    def command(self, *args):
        run = subprocess.run(args, capture_output=True, text=True, errors="replace")
        with (self.directory / "capture.log").open("a", encoding="utf-8") as log:
            log.write(run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError(f"GPU timing command failed ({run.returncode}): {args[0]}. "
                               f"Check ETW permissions and {self.directory / 'capture.log'}.")

    def __enter__(self):
        if os.name != "nt":
            raise RuntimeError("GPU ETW timing requires Windows.")
        self.directory.mkdir(parents=True, exist_ok=False)
        try:
            self.command("logman", "create", "trace", self.name, "-ets", "-o",
                         str(self.directory / "gpu.etl"), "-p", GPU_PROVIDER, "0x1", "4")
            self.started = True
            self.command("logman", "update", "trace", self.name, "-ets",
                         "-p", RUNTIME_PROVIDER, "0x2", "4")
        except BaseException:
            self.stop()
            raise
        self.token = active_capture.set(self)
        return self

    def qpc(self):
        value = ctypes.c_longlong()
        if not ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(value)):
            raise RuntimeError("QueryPerformanceCounter failed")
        return value.value

    def mark(self):
        return (self.qpc(), ctypes.windll.kernel32.GetCurrentThreadId())

    def record(self, begin):
        self.windows.append((begin[0], self.qpc(), begin[1]))

    def stop(self):
        if self.started:
            self.command("logman", "stop", self.name, "-ets")
            self.started = False

    def __exit__(self, *exc):
        active_capture.reset(self.token)
        self.stop()
        (self.directory / "windows.json").write_text(
            json.dumps({"pid": os.getpid(), "windows": self.windows}), encoding="utf-8")

    def finish(self):
        # -rts retains raw QPC timestamps for exact Run/marker matching.
        self.command("tracerpt", str(self.directory / "gpu.etl"), "-o",
                     str(self.directory / "events.xml"), "-of", "XML", "-rts", "-y")
        # tracerpt forbids combining raw timestamps and a summary in one pass.
        self.command("tracerpt", str(self.directory / "gpu.etl"),
                     "-summary", str(self.directory / "summary.xml"),
                     "-o", "NUL", "-y")
        return analyze_trace(self.directory)
