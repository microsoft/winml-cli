# Op-Tracing Refactor — Product Requirements Document

**Version**: 2.2
**Date**: 2026-04-19
**Status**: Draft
**Module**: session/monitor
**Supersedes**: `docs/design/optracing/1_req.md` v1.0 (consolidated into this PRD per `docs/standards/design-doc-spec.md`)
**Depends-On**: `docs/standards/design-doc-spec.md`

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Scope](#2-scope)
- [3. User Stories](#3-user-stories)
- [4. Functional Requirements](#4-functional-requirements)
- [5. Non-Functional Requirements](#5-non-functional-requirements)
- [6. Technical Design (high-level)](#6-technical-design-high-level)
- [7. Design Constraints](#7-design-constraints)
- [8. Risks and Mitigations](#8-risks-and-mitigations)
- [9. Open Questions](#9-open-questions)
- [10. Appendix](#10-appendix)
  - [10.1 Glossary](#101-glossary)
  - [10.2 References](#102-references)
  - [10.3 Document History](#103-document-history)
  - [10.4 Migration Footprint](#104-migration-footprint)
  - [10.5 Test Migration Footprint](#105-test-migration-footprint)

---

## 1. Executive Summary

### 1.1 Purpose

Replace the current `QNNProfiler` / `OpTracer` hierarchy with an extended `EPMonitor` design so that per-operator profiling works against both `onnxruntime-qnn` and `onnxruntime-windowsml`, eliminates duplicated ORT session-creation logic, and exposes a single per-EP hierarchy for all vendor-specific observation.

### 1.2 Problem Statement

Two defects motivate this refactor:

**D-1. `QNNProfiler` is broken with `onnxruntime-windowsml`.** The profiler creates its ORT session via the explicit-providers API, which searches for the QNN DLL in the pip package's `capi/` directory. `onnxruntime-windowsml` does not bundle that DLL — it lives under `C:\Program Files\WindowsApps\...` and is registered via WinML. The profiler's session silently falls back to CPU; no profiling data is produced.

**D-2. `QNNProfiler` duplicates `WinMLSession`.** It creates its own `ort.InferenceSession`, duplicating device-policy resolution, EPContext handling, and EP-discovery logic that `WinMLSession` already owns correctly (via `add_provider_for_devices`).

The codebase also carries two parallel per-EP hierarchies — `EPMonitor` in `session/monitor/` and `OpTracer` in `optracing/` — each with one QNN class. This duplication is untenable as more per-EP monitors land.

### 1.3 Success Metrics

- **SC-1** `wmk perf -m <model> --device npu --op-tracing basic` produces a valid CSV and per-op cycle report when run against a QNN NPU under `onnxruntime-windowsml`. Currently fails (silent CPU fallback).
- **SC-2** `QNNProfiler`, `OpTracer`, `optracing/base.py`, `optracing/registry.py` are removed from the codebase. No remaining references via grep.
- **SC-3** `QNNMonitor.is_available()` returns `True` on any machine where `wmk perf --device npu` currently runs on QNN, regardless of which ORT distribution is installed.
- **SC-4** The standalone-profile idiom specified in §4.7 works end-to-end in a one-off script without introducing helper classes.
- **SC-5** All `tests/` pass. New tests cover the behaviors in NFR-7.
- **SC-6** `display_op_trace_report` and `write_op_trace_json` consume `OpTraceResult` (not dict) and are not modified by this refactor. `OpTraceResult.to_dict()` — which already exists at `optracing/result.py:79-95` — is preserved in its current nested schema and extended with additive top-level `status` and `error` keys.

---

## 2. Scope

### 2.1 In Scope

- Deletion of `optracing/` package (except post-processing helpers, which move under `session/monitor/qnn/`).
- Extension of `EPMonitor` ABC with two optional hooks.
- Rewrite of `QNNMonitor` from placeholder to full implementation.
- Extension of `WinMLSession.perf()` to accept an EP monitor and yield a `PerfContext`.
- Collapse of the separate op-tracing block in `commands/perf.py` into the main benchmark loop.
- Extend the existing `OpTraceResult.to_dict()` method (at `optracing/result.py:79-95`) with additive top-level `status` and `error` keys; the existing nested schema is preserved.
- Extract WinML EP registry initializer from `WinMLSession._init_winml_eps_once()` to a module-level function to remove reverse coupling.

### 2.2 Out of Scope

- **OOS-1** QNNMonitor without a session (pure xrt-smi-style external telemetry). Possible future work.
- **OOS-2** New EPMonitor implementations for DML, OpenVINO, or TensorRT. This refactor reshapes the base class and reworks QNN only.
- **OOS-3** Changes to `HWMonitor` internals or PDH polling behavior.
- **OOS-4** Modifying `display_op_trace_report` / `write_op_trace_json` report writers. They continue to consume `OpTraceResult`.
- **OOS-5** Changes to the `wmk perf` CLI flags. `--op-tracing {basic|detail}` semantics preserved.
- **OOS-6** Multiple simultaneous EP monitors on one session. The `monitor=` parameter is singular. HWMonitor and EPMonitor coexist as orthogonal context managers.
- **OOS-7** Input generation utilities. No `generate_dummy_inputs` is added. Callers are responsible for their own input tensors.

---

## 3. User Stories

- **US-1** As a CLI user on a Qualcomm NPU running `onnxruntime-windowsml`, I run `wmk perf --op-tracing basic` and get a per-operator cycle report — without needing to install `onnxruntime-qnn`.
- **US-2** As a ModelKit developer, I add a new per-EP monitor (DML, OpenVINO) by subclassing `EPMonitor` — without duplicating ORT session-creation logic.
- **US-3** As a CI regression-check author, I capture per-operator metrics from a short Python script using only `WinMLSession` and `QNNMonitor` primitives.
- **US-4** As a library consumer, I attach EP-specific observation to an existing `WinMLSession` via `session.perf(monitor=...)` — without learning a second hierarchy.
- **US-5** As a QNN developer, I profile my actual benchmark workload rather than a synthetic profiling pass, so latency numbers reflect realistic inputs.

---

## 4. Functional Requirements

### 4.1 FR-1 — Op-tracing MUST work with both ORT distributions

The refactor MUST produce valid profiling artifacts regardless of whether the user has `onnxruntime-qnn` (bundled QNN DLL) or `onnxruntime-windowsml` (WinML-registered QNN DLL) installed. The implementation MUST use `SessionOptions.add_provider_for_devices([ep_device], options)` after WinML EP registration.

### 4.2 FR-2 — Op-tracing MUST attach via `session.perf(monitor=...)`

The user-facing entry point MUST be a monitor attached via `session.perf(warmup, monitor=QNNMonitor(level=...))`. The separate `QNNProfiler.run(...)` entry point is deleted. The monitor contributes session options AND provider options to compile; parses output artifacts on exit.

### 4.3 FR-3 — `EPMonitor` MUST be the single per-EP hierarchy

The separate `OpTracer` hierarchy (`optracing/base.py`, `optracing/registry.py`) MUST be deleted. All per-EP observation and configuration is expressed through `EPMonitor` subclasses.

### 4.4 FR-4 — `QNNMonitor` MUST replace `QNNProfiler`

The current placeholder `QNNMonitor` MUST become the real implementation. It encodes all QNN-specific knowledge: CSV format, QHAS processing, backend DLL selection, `profiling_level` options, and ORT session teardown for CSV flush. `QNNProfiler` MUST be deleted.

### 4.5 FR-5 — Two profiling levels MUST be exposed

- `QNNMonitor(level="basic")` → `profiling_level="detailed"` → CSV with per-op cycle counts.
- `QNNMonitor(level="detail")` → `profiling_level="optrace"` → QHAS post-processing via QNN SDK viewer. If the SDK viewer is unavailable, the monitor MUST fall back to basic CSV parsing with a `WARNING` log and `status="basic_fallback"` in the result.

### 4.6 FR-6 — `QNNMonitor` MUST produce an `OpTraceResult`

`QNNMonitor.result` MUST expose an `OpTraceResult` object (the existing dataclass from `optracing/result.py`, relocated to `session/monitor/op_metrics.py`). `QNNMonitor.to_dict()` MUST delegate to `OpTraceResult.to_dict()`. The existing `OpTraceResult.to_dict()` method (at `optracing/result.py:79-95`) MUST be preserved in its current nested schema (`{metadata: {...}, summary, operators, statistics, artifacts}`) to keep `display_op_trace_report` and `write_op_trace_json` consumers unchanged per OOS-4. The refactor MAY extend `OpTraceResult` and its `to_dict()` output with new top-level keys `status` and `error` for failure reporting (see FR-5 and NFR-2); existing keys MUST NOT be renamed, removed, or restructured. The `model` field on `OpTraceResult` MAY be relaxed from `str` to `str | None` to support cases where the source model path is unknown (e.g., standalone programmatic profiling); this change is additive (`None` serialises cleanly to JSON `null`).

### 4.7 FR-7 — Standalone profiling MUST work via primitive composition

Callers without a benchmarking harness MUST be able to produce op-trace data using only `WinMLSession` + `QNNMonitor` primitives:

```python
session = WinMLSession("model.onnx", device="npu")
with session.perf(monitor=QNNMonitor(level="basic")) as ctx:
    for _ in range(N):
        session.run(my_inputs)              # caller provides inputs
print(ctx.monitor.to_dict())
```

No `generate_dummy_inputs` utility is added. No helper class wraps the loop. If the caller lacks inputs, the caller MUST generate them.

### 4.8 FR-8 — Availability reporting MUST align with actual usability

`QNNMonitor.is_available()` MUST return `True` iff QNN EP is either bundled (`onnxruntime-qnn`) OR registered via WinML (`onnxruntime-windowsml`). The current single-path check (`ort.get_available_providers()` only) is insufficient. The implementation MUST call a module-level registry initializer (extracted from `WinMLSession._init_winml_eps_once`) and check `ort.get_ep_devices()`.

### 4.9 FR-9 — `HWMonitor` MUST remain orthogonal

`HWMonitor` is NOT migrated under `session.perf()`. It remains a standalone context manager, usable with or without a `WinMLSession`. `HWMonitor` and `EPMonitor` are independent context managers; they MAY be combined by the caller in a single `with` statement.

### 4.10 FR-10 — `EPMonitor` MUST gain two optional hooks

The `EPMonitor` base class MUST gain two optional hooks with defaults on the ABC itself (no Protocol, no Mixin):

- `get_session_options(self) -> dict[str, str]` — default `{}`. Contributions to `SessionOptions.add_session_config_entry()` (e.g., `"session.disable_cpu_ep_fallback"`, `"ep.context_enable"`).
- `get_provider_options(self) -> dict[str, str]` — default `{}`. Contributions to `add_provider_for_devices([ep], opts)` (e.g., `"profiling_level"`, `"backend_path"`).

`WinMLSession` MUST merge both into the respective ORT surfaces during compile. VitisAI / OpenVINO monitors inherit defaults and are unchanged.

### 4.11 FR-11 — Monitor instantiation MUST NOT require a factory / registry

`commands/perf.py` MUST resolve the correct `EPMonitor` class via explicit dispatch based on `--ep` and `--op-tracing` flags. No abstract factory, no registry module, no dynamic plugin loading. If op-tracing is requested against an EP that has no matching monitor, the command MUST fail hard with a descriptive error (no silent fallback).

### 4.12 FR-12 — Monitor MUST NOT mutate process-global state

`QNNMonitor` MUST NOT call `os.chdir()` or otherwise mutate the process's working directory. Output paths (CSV, schematic, QHAS) MUST be controlled via absolute paths in configuration. If a QNN SDK artifact (e.g., `*_schematic.bin`) cannot be redirected via configuration, the monitor MUST either (a) locate the artifact post-hoc from a known fallback location, or (b) document the limitation explicitly in its docstring and skip the artifact with a `WARNING` log.

---

## 5. Non-Functional Requirements

### 5.1 Performance

- **NFR-1** CLI-level ergonomics MUST NOT regress. The benchmark command path MUST collapse the current three context managers (stats + hw + ep) to two (perf-with-monitor + hw) and run in comparable wall time.

### 5.2 Reliability

- **NFR-2** No silent failures. If QNN EP cannot be loaded, the session MUST raise a descriptive `CompilationError`. If the CSV is absent or parsing fails, `to_dict()` MUST return `status="no_data"` or `status="parse_failed"` — never an empty structure masquerading as success.
- **NFR-3** Auto-reset MUST be observable. When `session.perf(monitor=...)` auto-resets a previously compiled session to apply the monitor's options, the event MUST log at `WARNING` level: `"auto-resetting compiled session to apply monitor session/provider options"`. Silent mutation is forbidden.
- **NFR-4** Idempotency. `EPMonitor.get_session_options()` and `EPMonitor.get_provider_options()` MUST return the same dict on repeated calls within one monitor instance's lifetime. File paths MUST be produced at `__init__`, not on each call.

### 5.3 Usability

- **NFR-5** Exception transparency. Monitor `__exit__` MUST NOT suppress exceptions raised from the `with` body. Parse failures inside `__exit__` are logged and reflected in `to_dict()`, but any active exception from the `with` body MUST propagate normally.

### 5.4 Compatibility

- **NFR-6** No process-global state. `EPMonitor` instances MUST be stateless between uses (no module-level caches). Importing `QNNMonitor` MUST NOT trigger EP probes, DLL loads, or network activity. `os.chdir` and equivalent global-state mutations are forbidden (see FR-12).
- **NFR-7** Test coverage. All existing tests in `tests/` MUST pass after the refactor. New tests MUST cover: the availability check on both ORT distributions, CSV parsing, session/provider-option merging rules, auto-reset behavior, load-bearing teardown ordering in `perf().__exit__`, and the double-entry guard on `EPMonitor.__enter__`.

---

## 6. Technical Design (high-level)

Detail lives in `2_coreloop.md`. Headline decisions:

- **Architectural pattern**: Hook-based Plugin + Template Method + Observer. `WinMLSession.compile()` is the template method; `EPMonitor` plugs into two hook points (`get_session_options`, `get_provider_options`); the monitor observes inference via context-manager lifecycle.
- **Session-owned ORT creation**: `WinMLSession` is the sole owner of `ort.InferenceSession` construction. Monitors never create ORT sessions.
- **Singular monitor**: `session.perf(monitor=EPMonitor|None)`. No `monitors=[...]` multi-monitor support.
- **Monitor factory by explicit dispatch**: `commands/perf.py` contains a 10-line dispatch function. No plugin registry.
- **Report consumers unchanged**: `OpTraceResult.to_dict()` is extended (not replaced); the existing nested schema is preserved; report writers continue to consume `OpTraceResult`.

---

## 7. Design Constraints

- **C-1** `WinMLSession` is the sole owner of ORT session creation. Monitors never create `ort.InferenceSession` instances directly.
- **C-2** Teardown ordering inside `perf().__exit__` is load-bearing: **session reset first, monitor `__exit__` second**. Reversing or parallelizing breaks QNN CSV parsing because QNN EP flushes CSV only on ORT session destruction.
- **C-3** `profiling_level` and `profiling_file_path` are NOT user-overridable via `extra_provider_options`. The monitor owns these keys; user overrides MUST be rejected by construction (enforced via explicit key assignment after merge, not via duplicate-key dict literals).
- **C-4** Only one `EPMonitor` per session. The `monitor=` parameter is singular. No multi-monitor support today or planned.
- **C-5** `EPMonitor` instances do not mutate process-global state. No `os.chdir()`, no env-var mutation, no module caches.
- **C-6** `requires_session_teardown: ClassVar[bool]` is an ORT-specific hint to the session that this monitor's data flush requires `ort.InferenceSession` destruction. It is the only place on the base ABC where an ORT implementation detail leaks in; this is accepted as a pragmatic tradeoff (YAGNI vs the architect's proposed `prepare_for_exit` callback).

---

## 8. Risks and Mitigations

- **R-1 / M-1**: Load-bearing teardown ordering regressed by a future contributor. → Integration test asserts `session._session is None` during `monitor.__exit__`; test lives in `tests/unit/session/test_perf_monitor.py`.
- **R-2 / M-2**: Windows file-handle lag after `ort.InferenceSession` destruction may leave CSV partially written when the monitor tries to parse it. → Call `gc.collect()` after `session.reset()` inside `perf().__exit__` to force handle release. Add fallback retry (one retry with 50ms delay) if CSV parse fails on first attempt.
- **R-3 / M-3**: Exception propagation through `perf().__exit__` silently swallowed if `session.reset()` raises while a caller exception is active. → Use `contextlib.ExitStack` or a `try/finally` chain that preserves the active exception per NFR-5.
- **R-4 / M-4**: QNN SDK `schematic.bin` file emitted to a location we cannot control (if absolute paths not supported). → Document as a known limitation; locate via glob post-hoc OR skip the artifact with a `WARNING` log (no `os.chdir`).
- **R-5 / M-5**: Auto-reset surprises a user debugging compile times. → `WARNING`-level log message; documented in `session.perf()` docstring.
- **R-6 / M-6**: Concurrent `WinMLSession` instances in one process both attempting op-tracing would race on the CSV output path (if default temp dirs collide). → QNNMonitor generates a unique output dir at `__init__` (`tempfile.mkdtemp(prefix="qnn_profile_")`) to eliminate collisions.

---

## 9. Open Questions

- **OQ-1** Does the QNN SDK accept an absolute path for `*_schematic.bin` output, enabling full elimination of `os.chdir`-style workarounds? If not, which fallback strategy (glob-locate post-hoc vs skip-with-warning) should be canonical? Resolve during implementation by empirical check against QNN SDK 2.42.
- **OQ-2** Should `OpTraceResult.to_dict()` include a schema version field for forward compatibility with future report formats? Currently leaning no (YAGNI), but decide before merge.

---

## 10. Appendix

### 10.1 Glossary

| Term | Meaning |
|------|---------|
| **ORT** | ONNX Runtime |
| **EP** | Execution Provider — ORT's plugin for a specific backend (QNN, DML, TensorRT, ...) |
| **QNN** | Qualcomm Neural Network — AI runtime for Qualcomm NPUs |
| **QHAS** | QNN Hardware Analyzer Schematic — detailed per-op roofline / DMA traffic data |
| **EPContext** | ORT feature that persists a JIT-compiled model for fast reload |
| **PDH** | Windows Performance Data Helper — OS counters used by `HWMonitor` |
| **WinML EP registration** | `ort.register_execution_provider_library(name, dll_path)` populated from the Windows App SDK's `ExecutionProviderCatalog` |
| **HTP** | Hexagon Tensor Processor — Qualcomm NPU backend within QNN |
| **Op-tracing** | Per-operator profiling: capturing per-op execution cycles during inference |

### 10.2 References

- `docs/standards/design-doc-spec.md` — the spec this PRD conforms to.
- `docs/design/session/monitor/2_coreloop.md` — companion core-loop design.
- `docs/design/session/monitor/iterations/01.md` through `11.md` — brainstorming trail.
- `D:\BYOM\ModelKit_PRs\232\docs\design\perf\qnn_ep_profiling_investigation.md` — original QNN EP profiling investigation (three ORT APIs, five tests, `add_provider_for_devices` solution).
- `D:\BYOM\ModelKit_PRs\232\temp\prove_qnn_ep_profiling.py` — proof script validating the fix.

### 10.3 Document History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-04-17 | Initial `1_req.md` (deleted). Captured requirements from iterations 01-11. |
| 2.0 | 2026-04-19 | Consolidated into `1_prd.md` per `docs/standards/design-doc-spec.md` v1.0. The prior `1_req.md` was deleted from disk (not deprecated-in-place) because its content is fully subsumed here; the `Supersedes` field preserves the historical link. Incorporated user directives (dual `get_session_options` + `get_provider_options` hooks; extend existing `OpTraceResult.to_dict()` — not replace; no `generate_dummy_inputs`; no `os.chdir`; no multi-monitor; factory dispatch; reorganized test migration). Incorporated critic and architect review findings. |
| 2.1 | 2026-04-19 | Post-audit fixes: added Table of Contents; renumbered Appendix to match spec §4.1 (Document History at §10.3); clarified that `OpTraceResult.to_dict()` already exists and the refactor preserves its nested schema, only adding `status`/`error` keys; clarified that `ep_registry.py` already exists and only gains a new `ensure_initialized()` function; added `fixtures/` to test migration; documented `commands/perf.py` import-path redirects. |
| 2.2 | 2026-04-24 | Relocated from docs/design/optracing/ to docs/design/session/monitor/ per spec §1.5.1 transitional commitment (implementation complete). Removed Transitional Location note. |

### 10.4 Migration Footprint

| Action | Paths |
|--------|-------|
| Delete | `src/winml/modelkit/optracing/base.py`, `src/winml/modelkit/optracing/registry.py`, `src/winml/modelkit/optracing/__init__.py`, `src/winml/modelkit/optracing/qnn/profiler.py` |
| Delete (entire directory after moves) | `src/winml/modelkit/optracing/` |
| Move | `optracing/qnn/csv_parser.py` → `session/monitor/qnn/csv_parser.py` |
| Move | `optracing/qnn/qhas_parser.py` → `session/monitor/qnn/qhas_parser.py` |
| Move | `optracing/qnn/viewer.py` → `session/monitor/qnn/viewer.py` |
| Move | `optracing/result.py` (`OpTraceResult`, `OperatorMetrics`) → `session/monitor/op_metrics.py` |
| Move | `optracing/report.py` (`display_op_trace_report`, `write_op_trace_json`) → `session/monitor/report.py` |
| Extend | Existing `OpTraceResult.to_dict()` in the relocated `op_metrics.py`: preserve nested schema; add top-level `status` and `error` keys. Add optional `status` / `error` dataclass fields (both default to `"ok"` / `None`). |
| Relax | `OpTraceResult.model: str` → `str \| None` for cases where source path is unknown. |
| Modify | `session/monitor/ep_monitor.py` — add `get_session_options`, `get_provider_options`, `requires_session_teardown` with defaults |
| Rewrite | `session/monitor/qnn_monitor.py` — from placeholder to full implementation |
| Modify | `session/session.py` — `perf()` gains `monitor=` parameter, returns `PerfContext`; compile-time hook integration; `_init_winml_eps_once` extracted to the module-level function described below |
| Modify | `session/ep_registry.py` — existing file gains a new module-level `ensure_initialized()` function that wraps `WinMLEPRegistry.get_instance().register_to_ort()`. The existing class-based API remains. |
| Modify | `commands/perf.py` — collapse separate op-tracing block; add `_resolve_ep_monitor()` dispatch helper. Import paths for `OpTraceResult`, `display_op_trace_report`, `write_op_trace_json` redirect from `..optracing` to `..session.monitor.report` / `..session.monitor.op_metrics`. Remove import of `is_qnn_profiling_available`, `get_tracer` (both deleted). |

### 10.5 Test Migration Footprint

| Existing test file | New location / action |
|---------------------|-----------------------|
| `tests/unit/optracing/test_csv_parser.py` | Move to `tests/unit/session/monitor/qnn/test_csv_parser.py` |
| `tests/unit/optracing/test_qhas_parser.py` | Move to `tests/unit/session/monitor/qnn/test_qhas_parser.py` |
| `tests/unit/optracing/test_detection.py` | Rewrite as `tests/unit/session/monitor/test_qnn_monitor_availability.py` |
| `tests/unit/optracing/test_integration.py` | Rewrite as `tests/unit/session/test_perf_monitor_integration.py` |
| `tests/unit/optracing/test_perf_optracing_cli.py` | Move to `tests/unit/commands/test_perf_optracing.py` |
| `tests/unit/optracing/test_qnn_profiler.py` | **Delete**; replaced by `tests/unit/session/monitor/test_qnn_monitor.py` (new) |
| `tests/unit/optracing/test_registry.py` | **Delete**; registry is removed. |
| `tests/unit/optracing/test_report.py` | Move to `tests/unit/session/monitor/test_report.py` |
| `tests/unit/optracing/test_result.py` | Move to `tests/unit/session/monitor/test_op_metrics.py`; add tests for the extended `OpTraceResult.to_dict()` with `status` / `error`. |
| `tests/unit/optracing/fixtures/` (directory) | Move to `tests/unit/session/monitor/qnn/fixtures/` — parsers use these (`optrace_resnet50.csv`, `qhas_resnet50.json`). |
| `tests/unit/optracing/` (directory) | Delete after all files moved / deleted. |
| — | **New**: `tests/unit/session/test_perf_monitor_integration.py` — asserts load-bearing teardown ordering (session `_session is None` during monitor `__exit__`). |
| — | **New**: `tests/unit/session/test_perf_auto_reset.py` — asserts `WARNING` log on auto-reset and that provider options are re-merged. |
| — | **New**: `tests/unit/session/monitor/test_ep_monitor_base.py` — asserts defaults of `get_session_options`, `get_provider_options`, `requires_session_teardown`, and double-entry guard. |
| — | **New**: `tests/unit/session/test_ep_registry.py` — asserts `ensure_initialized()` is idempotent and logs only on first call. |
