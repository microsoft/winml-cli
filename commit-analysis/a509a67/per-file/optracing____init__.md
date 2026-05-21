# src/winml/modelkit/optracing/__init__.py (DELETED)

## TL;DR
This file is removed. It was the public façade for the legacy `optracing` package, re-exporting `OpTracer`, `OperatorMetrics`, `OpTraceResult`, the registry helpers (`get_tracer`, `register_tracer`), the report helpers (`display_op_trace_report`, `write_op_trace_json`), and a `is_qnn_profiling_available()` probe. The entire package has been replaced by the `session/monitor/` tree, and `optracing` is no longer importable.

## Diff metrics
- Lines deleted: 34
- Status: DELETED

## What this file did (pre-state)
Defined the public surface of `winml.modelkit.optracing`. It:
- Re-exported the abstract `OpTracer` base class.
- Re-exported the registry functions `get_tracer` and `register_tracer` that resolved an `OpTracer` subclass by (EP-name-substring, level).
- Re-exported the dataclasses `OperatorMetrics` and `OpTraceResult`.
- Re-exported the report helpers `display_op_trace_report` and `write_op_trace_json`.
- Provided a module-level helper `is_qnn_profiling_available()` that checked whether `"QNNExecutionProvider"` appeared in `onnxruntime.get_available_providers()`.
- Listed all of the above in `__all__`.

## Public symbols (pre-deletion)
- `OpTracer` (re-export from `.base`)
- `get_tracer`, `register_tracer` (re-exports from `.registry`)
- `display_op_trace_report`, `write_op_trace_json` (re-exports from `.report`)
- `OperatorMetrics`, `OpTraceResult` (re-exports from `.result`)
- `is_qnn_profiling_available()` — module-level function

## Where the functionality moved
| Pre-state symbol | Where it lives now |
|---|---|
| `OpTracer` | **Dropped.** Replaced by the `EPMonitor` ABC at `src/winml/modelkit/session/monitor/ep_monitor.py`. The new contract is a context manager (`__enter__`/`__exit__`) instead of an explicit `run()` method, and `is_available()` is a `classmethod`. |
| `get_tracer`, `register_tracer` | **Dropped entirely.** The new architecture does not use a runtime EP-pattern registry; monitor selection is the caller's responsibility (instantiate `QNNMonitor()` and pass it to `session.perf(monitor=...)`). |
| `display_op_trace_report` | Lives at `src/winml/modelkit/session/monitor/report.py` with the same signature; only the import path changes. `top_n` default went from `15` to `5`. |
| `write_op_trace_json` | Same — `src/winml/modelkit/session/monitor/report.py`, unchanged signature. |
| `OperatorMetrics` | `src/winml/modelkit/session/monitor/op_metrics.py` (extended with `samples_us`, `avg_us`, `total_us`, `p90_us`, `sample_count`). |
| `OpTraceResult` | `src/winml/modelkit/session/monitor/op_metrics.py` (extended with `status: TraceStatus` and `error: str | None` for failure reporting; `model` is now `str | None`). |
| `is_qnn_profiling_available()` | **Dropped as a free function.** Replacement is `QNNMonitor.is_available()` (a classmethod) at `session/monitor/qnn_monitor.py`, which additionally probes the WinML-registered ORT path via `ep_registry.ensure_initialized()` and a `get_ep_devices()` scan. |

The new `session/monitor/__init__.py` is intentionally empty (no re-exports); consumers import concrete classes from their submodules.

## Net behavior change
- The import path `from winml.modelkit.optracing import ...` no longer resolves at all (the whole package is gone). Callers must update imports to `from winml.modelkit.session.monitor.* import ...`.
- The runtime registry indirection is gone: callers explicitly instantiate `QNNMonitor` rather than calling `get_tracer("QNN", "basic")`. There is no more EP-substring matching layer.
- Op-tracing now plugs into `WinMLSession.perf(...)` as an `EPMonitor`, replacing the standalone `tracer.run(iterations, warmup)` workflow.
- The availability probe gained a WinML-EP discovery path; callers who used to get `False` on `onnxruntime-windowsml` will now correctly see `True` when a QNN device is registered there.

## Risks
- `OpTracer`, `get_tracer`, and `register_tracer` are dropped with no replacement. Any external consumer that built a custom tracer subclass and registered it has no migration path other than re-implementing as an `EPMonitor` subclass.
- The `is_qnn_profiling_available()` free function is gone. Out-of-tree callers that imported it as a one-liner availability check must call `QNNMonitor.is_available()` instead.
