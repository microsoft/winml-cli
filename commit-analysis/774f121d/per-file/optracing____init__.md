# src/winml/modelkit/optracing/__init__.py (DELETED)

## TL;DR
This file is removed. It was the public API surface of the old `optracing` package, re-exporting the `OpTracer` ABC, the dataclass results, the registry helpers, and a `is_qnn_profiling_available()` probe. The entire package is gone; the new `session/monitor/` package replaces it but its own `__init__.py` deliberately exposes nothing — callers import from the concrete submodules.

## Diff metrics
- Lines deleted: 34
- Status: DELETED

## What this file did (pre-state)
Acted as the package facade for `winml.modelkit.optracing`. It:
- Re-exported the `OpTracer` ABC, `OpTraceResult`/`OperatorMetrics` dataclasses, the report helpers, and the `get_tracer`/`register_tracer` registry functions.
- Defined a standalone `is_qnn_profiling_available()` function that probed `ort.get_available_providers()` for `"QNNExecutionProvider"`.
- Eagerly triggered `registry._register_defaults()` by importing `from .registry import get_tracer, register_tracer` (the registry module body called it on import).

## Public symbols (pre-deletion)
- Re-exports:
  - `OpTracer` (from `.base`)
  - `OperatorMetrics`, `OpTraceResult` (from `.result`)
  - `get_tracer`, `register_tracer` (from `.registry`)
  - `display_op_trace_report`, `write_op_trace_json` (from `.report`)
- Module-level function: `is_qnn_profiling_available() -> bool`.
- `__all__` listing all eight names.

## Where the functionality moved
| Pre-state symbol | Where it lives now |
|---|---|
| Package itself (`winml.modelkit.optracing`) | **Replaced by `winml.modelkit.session.monitor`.** The new `monitor/__init__.py` is a near-empty docstring stub — there is no re-export surface; callers import directly from submodules. |
| `OpTracer` | `session.monitor.ep_monitor.WinMLEPMonitor` (different contract — see `optracing__base.md`). |
| `OpTraceResult`, `OperatorMetrics` | `session.monitor.op_metrics` (extended with `status` / `error` / `samples_us` fields). |
| `TraceStatus` (new) | `session.monitor.op_metrics` (didn't exist pre-state). |
| `get_tracer`, `register_tracer` | **Dropped entirely.** Substring-pattern registry replaced by explicit `_resolve_ep_monitor()` dispatch in `commands/perf.py`. |
| `display_op_trace_report`, `write_op_trace_json` | `session.monitor.report` (moved verbatim alongside the package relocation). |
| `is_qnn_profiling_available()` | Folded into `session.monitor.qnn_monitor.QNNMonitor.is_available()` (now a classmethod that does a richer WinML EP probe rather than a plain ORT provider-name check). |

## Net behavior change
- There is no longer a single import line that gives you "all op-tracing public types". Callers reach into specific modules — `from ..session.monitor.qnn_monitor import QNNMonitor`, `from ..session.monitor.op_metrics import OpTraceResult` etc.
- The eager `_register_defaults()` side-effect on import is gone; nothing self-registers anymore.
- The standalone `is_qnn_profiling_available()` helper is replaced by `QNNMonitor.is_available()`, which is the canonical availability gate.

## Risks
- Any caller that did `from winml.modelkit.optracing import OpTraceResult` (or any other re-exported name) will get `ModuleNotFoundError`. Tests previously used these top-level imports; the test suite was updated to match but third-party consumers will break.
- The substring registry permitted runtime extension via `register_tracer("MyPattern", "basic", MyTracer)`. The new explicit dispatch in `perf.py` has no extension hook — adding a new EP monitor requires editing `_resolve_ep_monitor`.
