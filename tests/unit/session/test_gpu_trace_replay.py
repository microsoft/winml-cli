# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Generated trace replay and capture lifecycle regressions."""

import errno
import json
import os
import subprocess
import tracemalloc
from pathlib import Path
from unittest.mock import Mock
from xml.etree import ElementTree

import pytest

from winml.modelkit.commands.perf import BenchmarkConfig, BenchmarkResult, display_console_report
from winml.modelkit.session.gpu_capture import GpuCapture, analyze_trace
from winml.modelkit.session.gpu_timing import GPU_PROVIDER, RUNTIME_PROVIDER


def event(name, stamp, data, provider=RUNTIME_PROVIDER, pid=100):
    node = ElementTree.Element("Event")
    system = ElementTree.SubElement(node, "System")
    ElementTree.SubElement(system, "Provider", Guid=provider)
    ElementTree.SubElement(system, "Execution", ProcessID=str(pid), ThreadID="7")
    ElementTree.SubElement(system, "TimeCreated", RawTime=str(stamp))
    fields = ElementTree.SubElement(node, "EventData")
    for key, value in data.items():
        ElementTree.SubElement(fields, "Data", Name=key).text = str(value)
    rendering = ElementTree.SubElement(node, "RenderingInfo")
    ElementTree.SubElement(rendering, "Task").text = name
    return ElementTree.tostring(node, encoding="unicode")


@pytest.fixture
def trace(tmp_path):
    (tmp_path / "windows.json").write_text(
        json.dumps({"pid": 100, "windows": [[10, 20, 7], [30, 40, 7]]}), encoding="utf-8"
    )
    (tmp_path / "summary.xml").write_text("Total Events Lost 0", encoding="utf-8-sig")
    events = []
    for run, start in ((1, 11), (2, 31)):
        events.extend(
            [
                event("PipelineRunStart", start, {"RunId": run, "Kind": 0}),
                event("PipelineRunStop", start + 5, {"RunId": run, "Kind": 0, "Result": "0x0"}),
            ]
        )
    # Samples arrive after both measured windows, in reverse Run order.
    events.extend(
        event(
            "CgcGpuTiming",
            100,
            {
                "RunId": run,
                "Path": run,
                "TimestampFrequency": 1000,
                "SubmitStart": 0,
                "BodyStart": 2,
                "BodyEnd": 2 + run,
                "SubmitEnd": 8,
            },
            provider=GPU_PROVIDER,
        )
        for run in (2, 1)
    )
    events.append(event("PipelineRunStart", 11, {"RunId": 1, "Kind": 0}, pid=200))
    (tmp_path / "events.xml").write_text(
        '<Events xmlns="http://schemas.microsoft.com/win/2004/08/events/event">'
        + "".join(events)
        + "</Events>",
        encoding="utf-8",
    )
    return tmp_path


def test_trace_analysis_and_reporting(trace):
    result = analyze_trace(trace)
    assert result["valid_runs"] == 2
    assert result["events_lost"] == 0
    assert sorted(result["samples_ms"]) == [1.0, 2.0]
    assert result["mean_ms"] == result["median_ms"] == 1.5
    assert result["p95_ms"] == pytest.approx(1.95)
    report = BenchmarkResult(config=BenchmarkConfig(model_id="trace-replay"))
    report.gpu_execution = result
    assert json.loads(json.dumps(report.to_dict()))["gpu_execution"] == result
    console = Mock()
    display_console_report(report, console)
    assert any("GPU execution time" in str(call) for call in console.print.call_args_list)


def test_cgep_replay_preserves_fallback_policy_without_claiming_cpu_usage(trace):
    path = trace / "windows.json"
    markers = json.loads(path.read_text(encoding="utf-8"))
    markers.update(execution_path="winmlcgep", cpu_fallback_policy="allowed")
    path.write_text(json.dumps(markers), encoding="utf-8")
    result = analyze_trace(trace)
    assert result["cpu_fallback_policy"] == "allowed"
    assert result["cpu_fallback_used"] is None
    assert "excludes CPU" in result["timing_scope"]
    assert result["valid_runs"] == 2


def test_tracerpt_uses_separate_event_and_summary_passes(trace, monkeypatch):
    capture = GpuCapture(trace)
    command = Mock()
    monkeypatch.setattr(capture, "command", command)
    assert capture.finish() == analyze_trace(trace)
    assert [call.args for call in command.call_args_list] == [
        (
            "tracerpt",
            str(trace / "trace.etl"),
            "-o",
            str(trace / "events.xml"),
            "-of",
            "XML",
            "-rts",
            "-y",
        ),
        (
            "tracerpt",
            str(trace / "trace.etl"),
            "-summary",
            str(trace / "summary.xml"),
            "-o",
            "NUL",
            "-y",
        ),
    ]


def test_unrelated_diagnostics_have_bounded_memory(trace):
    path = trace / "events.xml"
    original = path.read_text(encoding="utf-8")
    diagnostic = event("Diagnostic", 0, {"Payload": "x" * 800}, provider="{unrelated}")
    with path.open("w", encoding="utf-8") as stream:
        stream.write(original.removesuffix("</Events>"))
        for _ in range(20000):
            stream.write(diagnostic)
        stream.write("</Events>")
    tracemalloc.start()
    try:
        assert analyze_trace(trace)["valid_runs"] == 2
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # A full tree retains tens of MB; streaming needs only a read buffer.
    assert peak < 8 * 1024 * 1024


@pytest.mark.skipif(os.name != "nt", reason="ETW capture requires Windows")
@pytest.mark.parametrize("failure", ["open", "write"])
def test_log_failure_after_create_stops_owned_session(tmp_path, monkeypatch, failure):
    capture = GpuCapture(tmp_path / "capture")
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "ok", ""))
    monkeypatch.setattr(subprocess, "run", run)
    original_open = Path.open

    def open_log(path, *args, **kwargs):
        if path.name != "capture.log":
            return original_open(path, *args, **kwargs)
        if failure == "open":
            raise OSError(errno.ENOSPC, "disk full")
        log = Mock()
        log.__enter__ = Mock(return_value=log)
        log.__exit__ = Mock(return_value=False)
        log.write.side_effect = OSError(errno.ENOSPC, "disk full")
        return log

    monkeypatch.setattr(Path, "open", open_log)
    with pytest.raises(OSError, match="disk full"):
        capture.__enter__()
    assert [call.args[0][1] for call in run.call_args_list] == ["create", "stop"]
    assert not capture.started
    capture.stop()
    assert run.call_count == 2


@pytest.mark.skipif(os.name != "nt", reason="ETW capture requires Windows")
def test_failed_create_does_not_stop_unowned_session(tmp_path, monkeypatch):
    capture = GpuCapture(tmp_path / "capture")
    run = Mock(return_value=subprocess.CompletedProcess([], 1, "", "failed"))
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="GPU timing command failed"):
        capture.__enter__()
    assert run.call_count == 1
    assert not capture.started


def test_failed_stop_retains_ownership_for_retry(tmp_path, monkeypatch):
    capture = GpuCapture(tmp_path)
    capture.started = True
    run = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 1, "", "failed"),
            subprocess.CompletedProcess([], 0, "", ""),
        ]
    )
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="GPU timing command failed"):
        capture.stop()
    assert capture.started
    capture.stop()
    assert not capture.started


@pytest.mark.skipif(os.name != "nt", reason="ETW capture requires Windows")
@pytest.mark.parametrize("verbose", [False, True])
def test_provider_scope_and_profile_are_explicit(tmp_path, monkeypatch, verbose):
    monkeypatch.setenv("WINML_FULL_DIAGNOSTICS", "1" if verbose else "0")
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "ok", ""))
    monkeypatch.setattr(subprocess, "run", run)
    capture = GpuCapture(tmp_path / "capture")
    with capture:
        pass
    commands = [call.args[0] for call in run.call_args_list]
    creation = commands[0]
    assert creation[creation.index("-p") + 1 :] == (
        GPU_PROVIDER,
        "0xffffffffffffffff" if verbose else "0x1",
        "5" if verbose else "4",
    )
    updates = [args for args in commands if args[1] == "update"]
    if verbose:
        assert len(updates) == 9
        assert all(args[-2:] == ("0xffffffffffffffff", "5") for args in updates)
    else:
        assert len(updates) == 1
        assert updates[0][-3:] == (RUNTIME_PROVIDER, "0x2", "4")
    markers = json.loads((capture.directory / "windows.json").read_text())
    assert markers["collection_profile"] == (
        "runtime-ort-gpu-diagnostics-verbose" if verbose else "runtime-gpu-timing-minimal"
    )
    session = json.loads((capture.directory / "session.json").read_text())
    assert session["name"] == capture.name
    assert session["trace"] == str(capture.directory / "trace.etl")


@pytest.mark.parametrize(("tool", "timeout"), [("logman", 30), ("tracerpt", 120)])
def test_trace_tools_have_bounded_deadlines(tmp_path, monkeypatch, tool, timeout):
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(subprocess, "run", run)
    GpuCapture(tmp_path).command(tool, "test")
    assert run.call_args.kwargs["timeout"] == timeout
    record = json.loads((tmp_path / "capture.log").read_text().splitlines()[0])
    assert record["returncode"] == 0


@pytest.mark.skipif(os.name != "nt", reason="ETW capture requires Windows")
def test_timed_out_create_attempts_owned_cleanup(tmp_path, monkeypatch):
    run = Mock(
        side_effect=[
            subprocess.TimeoutExpired("logman", 30),
            subprocess.CompletedProcess([], 0, "", ""),
        ]
    )
    monkeypatch.setattr(subprocess, "run", run)
    capture = GpuCapture(tmp_path / "capture")
    with pytest.raises(RuntimeError, match="timed out"):
        capture.__enter__()
    assert [call.args[0][1] for call in run.call_args_list] == ["create", "stop"]
    assert not capture.started
