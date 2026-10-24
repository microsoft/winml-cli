# Runtime GPU execution timing

`winml perf --gpu-timing-trace <new-directory>` captures Runtime CGC ETW
events and reports GPU inference-body Mean/P50/P95 alongside CPU latency.
The normal benchmark JSON includes an optional `gpu_execution` object.

```powershell
winml perf -m C:\models\yolo11n.onnx --runtime winml-runtime --backend cgc --device gpu --iterations 1000 --warmup 10 --gpu-timing-trace C:\perf\yolo11n-new
```

Requires Windows, ETW permissions (for example an elevated terminal), and a
compatible development Runtime exposing CgcGpuTiming and PipelineRun events.
The CLI does not automatically elevate. Use a fresh trace directory per run.

The collector matches measured Python call windows to successful Runtime
Runs by process ID, thread ID and QPC, then matches GPU events by RunId.
Warmup is excluded. Only replay paths 1 and 2 contribute GPU body time;
recorded path 0 includes transfers and is excluded. Missing samples, event
loss, unknown paths and multiple submissions per Run fail closed.

Artifacts include gpu.etl, raw-QPC events.xml, events-wallclock.xml,
summary.xml (tracerpt text), windows.json and gpu-execution.json. Runtime
owners are released before stopping the trace to flush pending GPU samples.

## Current limitations

- Summary loss parsing currently recognizes English tracerpt output; other
  formats fail explicitly instead of reporting unverified measurements.
- Statistics are for one measurement window, not median of round medians.
- Local ONNX/MLIR Runtime CGC GPU workloads only; tracing cannot be combined
  with monitor, operator tracing, module or device comparison modes.
- The Runtime Python projection must avoid changing shared ctypes pythonapi
  buffer-function signatures, which can break Click console output.

## Validation

Standalone protocol tests:

```powershell
python tests/unit/session/test_gpu_timing.py
python tests/unit/session/check_gpu_trace_replay.py C:\perf\existing-trace
```

Validated on RTX 5090 D with ResNet-50 (20 measured Runs) and YOLO11n
(1000 measured Runs): all expected GPU samples received, zero ETW loss.
Models, traces and binary payloads are not included in this change.
