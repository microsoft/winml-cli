# GPU timing capture profiles

Routine GPU timing captures only the events consumed by the timing analyzer:

- CGC GPU timing provider: keyword 0x1, informational level 4.
- Runtime pipeline markers: keyword 0x2, informational level 4.

windows.json records this as runtime-gpu-timing-minimal. PID/TID/RunId
matching, complete measured windows, GPU sample validation and event-loss checks
are unchanged. The capture still starts before session initialization.

For an explicit verbose diagnostic capture, set
WINML_FULL_DIAGNOSTICS=1 before running the CLI. This retains the broader
Runtime/ORT/WIL/DxgKrnl provider set with all keywords at verbose level and records
runtime-ort-gpu-diagnostics-verbose. These logs can be much larger than the
timing events; keep the profile identity when comparing results.

session.json records the generated logger name, PID, output ETL path and profile.
Trace-control commands have a 30-second deadline; each tracerpt export has a
120-second deadline. Command return codes and timeout details are recorded in
capture.log. A timed-out create attempts cleanup of its generated logger name;
cleanup never targets other logger names.

WinMLCG EP capture preserves the ORT graph-optimization and intra-op thread
settings used by ordinary perf, including session-factory and monitor overrides.
Enabling ETW does not disable graph optimization or force single-threaded execution.
Keep the same model, inputs, lowering target and CPU-fallback policy when comparing
traced GPU timing with an untraced host measurement; their timing domains differ.

WinMLCG EP capture disables CPU fallback by default. Explicitly permit CPU fallback:

```cmd
winml perf -m model.onnx --runtime winml-ort --ep winmlcg --device gpu --etw-tracing trace --allow-cpu-fallback
```

This option requires WinMLCG EP ETW capture. The trace and GPU result record the policy; usage is false when fallback is disabled, and unknown when it is allowed. Like other CGC-specific options, this advanced option is accepted but hidden from public help. GPU samples cover captured CGC work, not end-to-end inference or CPU execution. Existing event-loss and unambiguous Run matching requirements still apply; permitting fallback does not turn missing or ambiguous GPU samples into valid measurements.
