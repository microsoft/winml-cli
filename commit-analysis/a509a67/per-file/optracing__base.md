# src/winml/modelkit/optracing/base.py (DELETED)

## TL;DR
This file is removed. It defined the `OpTracer` ABC — the EP-agnostic operator-profiling interface (`run()` + `is_available()`). The class is dropped entirely; the new architecture uses an `EPMonitor` context-manager ABC at `session/monitor/ep_monitor.py` instead.

## Diff metrics
- Lines deleted: 35
- Status: DELETED

## What this file did (pre-state)
Defined an abstract base class for operator-level profilers. The contract was:
- Subclasses received the model path and output directory at construction time.
- Callers invoked `tracer.run(iterations, warmup)` to produce an `OpTraceResult`.
- A separate `is_available()` instance method advertised whether the runtime dependencies (ORT EP, SDK binaries) were usable.

The class had exactly one production subclass, `QNNProfiler`.

## Public symbols (pre-deletion)
- `OpTracer` — ABC with two `@abstractmethod`s:
  - `run(self, iterations: int = 5, warmup: int = 2) -> OpTraceResult` — execute profiling, return structured result.
  - `is_available(self) -> bool` — runtime-availability check (instance method).

## Where the functionality moved
| Pre-state symbol | Where it lives now |
|---|---|
| `OpTracer` (ABC) | **Replaced by `EPMonitor` at `src/winml/modelkit/session/monitor/ep_monitor.py`**, but the *contract* is fundamentally different — see below. |
| `OpTracer.run(iterations, warmup)` | **Dropped entirely.** There is no `run()` method on `EPMonitor`. The new architecture inverts control: `WinMLSession.perf(monitor=...)` drives the inference loop; the monitor only observes via `__enter__` / `__exit__`. The warmup / iteration count is now owned by `session.perf()` and `PerfStats`, not the monitor. |
| `OpTracer.is_available()` (instance method) | Replaced by `EPMonitor.is_available()` as a `@classmethod`. Concrete implementation lives in each subclass (e.g. `QNNMonitor.is_available()`). |
| (implicit) constructor arg `onnx_path` | Dropped — the monitor no longer owns the model. The session owns it; the monitor only contributes options and parses post-hoc artifacts. |
| (implicit) constructor arg `output_dir` | Retained on subclasses (e.g. `QNNMonitor.__init__(..., output_dir=...)`) but no longer part of the ABC. |

### New abstractions added on the replacement `EPMonitor` (not present pre-state)
- `__enter__` / `__exit__` (abstract) — the context-manager protocol replacing `run()`.
- `requires_session_teardown: ClassVar[bool]` — declares ordering of session destruction vs `__exit__` (QNN needs `True` because it flushes profiling only on `InferenceSession` destruction).
- `ep_name: ClassVar[str | None]` — pins the session to a specific EP.
- `get_session_options() -> dict[str, str]` — contributes ORT `SessionOptions.add_session_config_entry` keys.
- `get_provider_options() -> dict[str, str]` — contributes provider-option keys passed to `add_provider_for_devices`.
- `set_onnx_op_types(map)` — v2.4 hook for injecting the ONNX `node.name -> op_type` map.
- `result` property — typed accessor returning the wrapped `OpTraceResult | None`.

## Net behavior change
- The "tracer owns the run" model is gone. Previously, a `QNNProfiler` instance owned the ORT session, the input generation, the warmup/iteration loop, and the CWD switch. After this commit, all of that is the responsibility of `WinMLSession.perf()` and the monitor only contributes config and parses artifacts.
- `is_available()` switched from instance to classmethod, so callers can probe availability without instantiating.
- `run()` returning an `OpTraceResult` is replaced by `monitor.result` being populated during `__exit__`.

## Risks
- Any out-of-tree subclass of `OpTracer` will fail to import (`from ..optracing.base import OpTracer` resolves to nothing). The migration path is to re-implement as an `EPMonitor` subclass, which is a non-trivial restructure because the inference loop moves out of the subclass.
- Callers that depended on `tracer.run()` returning a result synchronously must now drive a `session.perf(...)` context-manager loop and read `ctx.monitor.result` after exit.
