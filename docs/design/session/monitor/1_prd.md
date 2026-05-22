# Op-Tracing Refactor — Product Requirements Document

**Version**: 2.4.1
**Date**: 2026-05-08
**Status**: Draft
**Module**: session/monitor
**Supersedes**: `docs/design/optracing/1_req.md` v1.0 (consolidated into this PRD per `docs/standards/design-doc-spec.md`)
**Depends-On**: `docs/standards/design-doc-spec.md`

> **See also**: `docs/design/perf/2026-05-03-op-trace-parser-interface-spec.md` v2.0 — focused architectural spec for QNN op-type resolution (ONNX-graph lookup + fallback chain) as a QNNMonitor-private detail. Open questions in spec §10.1 (option a vs b for helper placement, architecture-test mechanism, `_build_op_type_map` placement, typed `proof` accessor follow-up) flow back into this doc when resolved.

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

Replace the current `QNNProfiler` / `OpTracer` hierarchy with an extended `WinMLEPMonitor` design so that per-operator profiling works against both `onnxruntime-qnn` and `onnxruntime-windowsml`, eliminates duplicated ORT session-creation logic, and exposes a single per-EP hierarchy for all vendor-specific observation.

v2.3 introduced a separate `OpTraceParser` ABC alongside `WinMLEPMonitor` to formalize op-trace data parsing as a distinct contract. **v2.4 simplifies that design.** After review, the `OpTraceParser` ABC was deemed over-engineering for a single concrete implementer (QNNMonitor); multiple inheritance + MRO ordering + a separate ABC file are too much complexity for one EP. v2.4 collapses to a single-ABC design: extend the existing `WinMLEPMonitor` ABC with two concrete-default members (`set_onnx_op_types(map)` no-op default and `result` property returning `None`); QNNMonitor stays single-inheritance and owns ONNX-graph lookup + the four-layer fallback chain as private internals (`_resolve_op_type`, `_heuristic_op_type`).

v2.4 also drops `to_dict()` from the `WinMLEPMonitor` ABC contract. The polymorphic `to_dict()` was a god-method that conflated op-tracing telemetry (QNN) with proof-of-execution signals (VitisAI/OpenVINO) under one interface. Concrete monitors now expose data via typed accessors instead: `QNNMonitor.result -> OpTraceResult`; `VitisAIMonitor.proof / OpenVINOMonitor.proof -> ProofOfExecution` (typed accessor + new `ProofOfExecution` class flagged as a follow-up PR — out of scope for this lift). `commands/perf.py` switches from a unified `monitor.to_dict()` call to isinstance-based typed accessor dispatch (QNN payload → `op_trace` JSON key; proof-of-execution payload → `ep_proof` JSON key).

The op-type label on each `OperatorMetrics` is sourced primarily from the ONNX graph's `node.op_type` via an injected `dict[node.name, node.op_type]` lookup map built once at session setup; EP-authoritative fields (e.g. `qhas.qnn_op_type`) and EP-specific heuristics serve as fallbacks. This also enforces strict information hiding around each EP — the QNN-specific CSV/QHAS parsing modules (`qnn/csv_parser.py`, `qnn/qhas_parser.py`) are deleted as public modules and their helpers fold into the QNN monitor's private surface.

### 1.2 Problem Statement

Two defects motivate this refactor:

**D-1. `QNNProfiler` is broken with `onnxruntime-windowsml`.** The profiler creates its ORT session via the explicit-providers API, which searches for the QNN DLL in the pip package's `capi/` directory. `onnxruntime-windowsml` does not bundle that DLL — it lives under `C:\Program Files\WindowsApps\...` and is registered via WinML. The profiler's session silently falls back to CPU; no profiling data is produced.

**D-2. `QNNProfiler` duplicates `WinMLSession`.** It creates its own `ort.InferenceSession`, duplicating device-policy resolution, EPContext handling, and EP-discovery logic that `WinMLSession` already owns correctly (via `add_provider_for_devices`).

The codebase also carries two parallel per-EP hierarchies — `WinMLEPMonitor` in `session/monitor/` and `OpTracer` in `optracing/` — each with one QNN class. This duplication is untenable as more per-EP monitors land.

### 1.3 Success Metrics

- **SC-1** `wmk perf -m <model> --device npu --op-tracing basic` produces a valid CSV and per-op cycle report when run against a QNN NPU under `onnxruntime-windowsml`. Currently fails (silent CPU fallback).
- **SC-2** `QNNProfiler`, `OpTracer`, `optracing/base.py`, `optracing/registry.py` are removed from the codebase. No remaining references via grep.
- **SC-3** `QNNMonitor.is_available()` returns `True` on any machine where `wmk perf --device npu` currently runs on QNN, regardless of which ORT distribution is installed.
- **SC-4** The standalone-profile idiom specified in §4.7 works end-to-end in a one-off script without introducing helper classes.
- **SC-5** All `tests/` pass. New tests cover the behaviors in NFR-7.
- **SC-6** `display_op_trace_report` and `write_op_trace_json` consume `OpTraceResult` (not dict) and are not modified by this refactor. `OpTraceResult.to_dict()` — which already exists at `optracing/result.py:79-95` — is preserved in its current nested schema and extended with additive top-level `status` and `error` keys.
- **SC-7** `OperatorMetrics.name` reports ONNX `node.op_type` verbatim (`Conv`, `LayerNormalization`, `Gelu`) when the ONNX graph is available at session setup. ONNX `node.op_type` is the primary source for `OperatorMetrics.name`, sourced via QNNMonitor's private `_resolve_op_type`. No translation tables — the value is sourced directly from `node.op_type` and rendered as-is per Cardinal Rule #1.
- **SC-8** No module outside `qnn_monitor.py` (and its private `qnn/_internal.py` if option b is chosen for helper placement) imports anything from QNN parsing internals. Verifiable via static import-scan; pinned by an architecture regression test (see NFR-8).
- **SC-9** `QNNMonitor` is constructible from a unit test with an injected `dict[str, str]` for `onnx_op_types` — no real ONNX file required for parser unit tests. Constructor accepts the map directly and `set_onnx_op_types(...)` allows late-binding for the live session-setup path.
- **SC-10** (v2.4) `WinMLEPMonitor` ABC drops `to_dict()`. Concrete monitors expose typed accessors (`result` for op-tracing, `proof` for proof-of-execution). `commands/perf.py` JSON output uses isinstance-based dispatch — QNN payloads land under `op_trace`, VitisAI/OpenVINO payloads land under `ep_proof`, NullEPMonitor contributes nothing.

---

## 2. Scope

### 2.1 In Scope

- Deletion of `optracing/` package (except post-processing helpers, which move under `session/monitor/qnn/`).
- Extension of `WinMLEPMonitor` ABC with two optional hooks.
- Rewrite of `QNNMonitor` from placeholder to full implementation.
- Extension of `WinMLSession.perf()` to accept an EP monitor and yield a `PerfContext`.
- Collapse of the separate op-tracing block in `commands/perf.py` into the main benchmark loop.
- Extend the existing `OpTraceResult.to_dict()` method (at `optracing/result.py:79-95`) with additive top-level `status` and `error` keys; the existing nested schema is preserved.
- Extract WinML EP registry initializer from `WinMLSession._init_winml_eps_once()` to a module-level function to remove reverse coupling.
- Extend `WinMLEPMonitor` ABC with two concrete-default members: `set_onnx_op_types(map) -> None` (no-op default; op-tracing monitors override) and `result -> OpTraceResult | None` property (default `None`; concrete monitors populate `self._result` from `__exit__`).
- Remove `to_dict()` from `WinMLEPMonitor` ABC contract. Concrete monitors expose data via typed accessors instead: `result` for op-tracing (QNNMonitor), `proof` for proof-of-execution (VitisAI/OpenVINO — typed `ProofOfExecution` class flagged as follow-up PR, out of scope).
- `QNNMonitor` stays single-inheritance (`class QNNMonitor(WinMLEPMonitor)`); ONNX op-type lookup + four-layer fallback chain live as private internals (`_resolve_op_type`, `_heuristic_op_type`, `_parse_basic`, `_parse_detail`).
- Add `QNNMonitor.parse_existing_artifacts(level, artifacts, onnx_op_types=None)` classmethod — public entry point for offline analysis of pre-existing CSV/QHAS files (replaces the v2.3 abstract `parse_basic`/`parse_detail` interface).
- Deletion of `qnn/csv_parser.py` and `qnn/qhas_parser.py` as public modules. Their helpers fold either into `qnn_monitor.py` directly (option a) or into a private sibling submodule `qnn/_internal.py` (option b, recommended). Final choice per spec §10.1 Q1.
- ONNX op-type map (`dict[node.name, node.op_type]`) built by `WinMLSession` at session setup (method placement per spec §10.1 Q3) and injected unconditionally into every monitor via `monitor.set_onnx_op_types(map)`. Non-op-tracing monitors inherit the WinMLEPMonitor no-op default and silently ignore the call.
- `commands/perf.py` JSON output: switch from unified `ctx.monitor.to_dict()` to isinstance-based typed accessor dispatch (QNN → `op_trace` key, VitisAI/OpenVINO → `ep_proof` key, NullEPMonitor → no key).
- Refactor of QNN parsing tests: existing unit tests of private helpers (`test_csv_parser.py`, `test_csv_parser_samples.py`, `test_event_id_split.py`, `test_qhas_parser.py`) are deleted as architectural debt; coverage migrates to integration tests on `QNNMonitor.parse_existing_artifacts`.

### 2.2 Out of Scope

- **OOS-1** QNNMonitor without a session (pure xrt-smi-style external telemetry). Possible future work.
- **OOS-2** New WinMLEPMonitor implementations for DML, OpenVINO, or TensorRT. This refactor reshapes the base class and reworks QNN only.
- **OOS-3** Changes to `HWMonitor` internals or PDH polling behavior.
- **OOS-4** Modifying `display_op_trace_report` / `write_op_trace_json` report writers. They continue to consume `OpTraceResult`.
- **OOS-5** Changes to the `wmk perf` CLI flags. `--op-tracing {basic|detail}` semantics preserved.
- **OOS-6** Multiple simultaneous EP monitors on one session. The `monitor=` parameter is singular. HWMonitor and WinMLEPMonitor coexist as orthogonal context managers.
- **OOS-7** Input generation utilities. No `generate_dummy_inputs` is added. Callers are responsible for their own input tensors.

---

## 3. User Stories

- **US-1** As a CLI user on a Qualcomm NPU running `onnxruntime-windowsml`, I run `wmk perf --op-tracing basic` and get a per-operator cycle report — without needing to install `onnxruntime-qnn`.
- **US-2** As a ModelKit developer, I add a new per-EP monitor (DML, OpenVINO) by subclassing `WinMLEPMonitor` — without duplicating ORT session-creation logic.
- **US-3** As a CI regression-check author, I capture per-operator metrics from a short Python script using only `WinMLSession` and `QNNMonitor` primitives.
- **US-4** As a library consumer, I attach EP-specific observation to an existing `WinMLSession` via `session.perf(monitor=...)` — without learning a second hierarchy.
- **US-5** As a QNN developer, I profile my actual benchmark workload rather than a synthetic profiling pass, so latency numbers reflect realistic inputs.

---

## 4. Functional Requirements

### 4.1 FR-1 — Op-tracing MUST work with both ORT distributions

The refactor MUST produce valid profiling artifacts regardless of whether the user has `onnxruntime-qnn` (bundled QNN DLL) or `onnxruntime-windowsml` (WinML-registered QNN DLL) installed. The implementation MUST use `SessionOptions.add_provider_for_devices([ep_device], options)` after WinML EP registration.

### 4.2 FR-2 — Op-tracing MUST attach via `session.perf(monitor=...)`

The user-facing entry point MUST be a monitor attached via `session.perf(warmup, monitor=QNNMonitor(level=...))`. The separate `QNNProfiler.run(...)` entry point is deleted. The monitor contributes session options AND provider options to compile; parses output artifacts on exit.

### 4.3 FR-3 — `WinMLEPMonitor` MUST be the single per-EP hierarchy

The separate `OpTracer` hierarchy (`optracing/base.py`, `optracing/registry.py`) MUST be deleted. All per-EP observation and configuration is expressed through `WinMLEPMonitor` subclasses.

### 4.4 FR-4 — `QNNMonitor` MUST replace `QNNProfiler`

The current placeholder `QNNMonitor` MUST become the real implementation. It encodes all QNN-specific knowledge: CSV format, QHAS processing, backend DLL selection, `profiling_level` options, and ORT session teardown for CSV flush. `QNNProfiler` MUST be deleted.

### 4.5 FR-5 — Two profiling levels MUST be exposed

- `QNNMonitor(level="basic")` → `profiling_level="detailed"` → CSV with per-op cycle counts.
- `QNNMonitor(level="detail")` → `profiling_level="optrace"` → QHAS post-processing via QNN SDK viewer. If the SDK viewer is unavailable, the monitor MUST fall back to basic CSV parsing with a `WARNING` log and `status="basic_fallback"` in the result.

### 4.6 FR-6 — `QNNMonitor` MUST produce an `OpTraceResult`

**FR-6**: `QNNMonitor.result` MUST expose an `OpTraceResult` instance (or `None` when not yet parsed) via the typed accessor inherited from `WinMLEPMonitor`. Consumers obtain JSON via `monitor.result.to_dict()` directly. The legacy `QNNMonitor.to_dict()` shim from v2.x is removed in v2.4 (see FR-20).

The existing `OpTraceResult.to_dict()` method (at `optracing/result.py:79-95`) MUST be preserved in its current nested schema (`{metadata: {...}, summary, operators, statistics, artifacts}`) to keep `display_op_trace_report` and `write_op_trace_json` consumers unchanged per OOS-4. The refactor MAY extend `OpTraceResult` and its `to_dict()` output with new top-level keys `status` and `error` for failure reporting (see FR-5 and NFR-2); existing keys MUST NOT be renamed, removed, or restructured. The `model` field on `OpTraceResult` MAY be relaxed from `str` to `str | None` to support cases where the source model path is unknown (e.g., standalone programmatic profiling); this change is additive (`None` serialises cleanly to JSON `null`).

### 4.7 FR-7 — Standalone profiling MUST work via primitive composition

Callers without a benchmarking harness MUST be able to produce op-trace data using only `WinMLSession` + `QNNMonitor` primitives:

```python
session = WinMLSession("model.onnx", device="npu")
with session.perf(monitor=QNNMonitor(level="basic")) as ctx:
    for _ in range(N):
        session.run(my_inputs)              # caller provides inputs
print(ctx.monitor.result.to_dict() if ctx.monitor.result else "(no op-trace data)")
```

No `generate_dummy_inputs` utility is added. No helper class wraps the loop. If the caller lacks inputs, the caller MUST generate them.

### 4.8 FR-8 — Availability reporting MUST align with actual usability

`QNNMonitor.is_available()` MUST return `True` iff QNN EP is either bundled (`onnxruntime-qnn`) OR registered via WinML (`onnxruntime-windowsml`). The current single-path check (`ort.get_available_providers()` only) is insufficient. The implementation MUST call a module-level registry initializer (extracted from `WinMLSession._init_winml_eps_once`) and check `ort.get_ep_devices()`.

### 4.9 FR-9 — `HWMonitor` MUST remain orthogonal

`HWMonitor` is NOT migrated under `session.perf()`. It remains a standalone context manager, usable with or without a `WinMLSession`. `HWMonitor` and `WinMLEPMonitor` are independent context managers; they MAY be combined by the caller in a single `with` statement.

### 4.10 FR-10 — `WinMLEPMonitor` MUST gain two optional hooks

The `WinMLEPMonitor` base class MUST gain two optional hooks with defaults on the ABC itself (no Protocol, no Mixin):

- `get_session_options(self) -> dict[str, str]` — default `{}`. Contributions to `SessionOptions.add_session_config_entry()` (e.g., `"session.disable_cpu_ep_fallback"`, `"ep.context_enable"`).
- `get_provider_options(self) -> dict[str, str]` — default `{}`. Contributions to `add_provider_for_devices([ep], opts)` (e.g., `"profiling_level"`, `"backend_path"`).

`WinMLSession` MUST merge both into the respective ORT surfaces during compile. VitisAI / OpenVINO monitors inherit defaults and are unchanged.

### 4.11 FR-11 — Monitor instantiation MUST NOT require a factory / registry

`commands/perf.py` MUST resolve the correct `WinMLEPMonitor` class via explicit dispatch based on `--ep` and `--op-tracing` flags. No abstract factory, no registry module, no dynamic plugin loading. If op-tracing is requested against an EP that has no matching monitor, the command MUST fail hard with a descriptive error (no silent fallback).

### 4.12 FR-12 — Monitor MUST NOT mutate process-global state

`QNNMonitor` MUST NOT call `os.chdir()` or otherwise mutate the process's working directory. Output paths (CSV, schematic, QHAS) MUST be controlled via absolute paths in configuration. If a QNN SDK artifact (e.g., `*_schematic.bin`) cannot be redirected via configuration, the monitor MUST either (a) locate the artifact post-hoc from a known fallback location, or (b) document the limitation explicitly in its docstring and skip the artifact with a `WARNING` log.

### 4.13 FR-13 — Removed in v2.4

(v2.3 introduced an `OpTraceParser` ABC requirement here. v2.4 drops that ABC entirely — see §1.1 Purpose. This FR slot is retired; WinMLEPMonitor is the only ABC.)

### 4.14 FR-14 — ONNX `node.op_type` MUST be the primary op-type source

The op-type label assigned to `OperatorMetrics.name` MUST be sourced first from a `dict[node.name → node.op_type]` built from the ONNX graph at session setup, looked up via QNNMonitor's private `_resolve_op_type` method. The fallback chain proceeds only when the ONNX lookup misses: EP-authoritative fields (e.g. `qhas.qnn_op_type` from QHAS detail mode) MUST be consulted next, followed by an EP-specific heuristic (e.g. QNN's `_token_N` strip + leaf-split), and finally the raw `op_path` as a last-resort. Each layer absorbs the failure modes of the one above it. The full chain is a QNNMonitor-private detail; there is no ABC-level contract for it (see §1.1 v2.4 simplification).

### 4.15 FR-15 — Token-suffix stripping MUST bridge runtime IDs to ONNX names

Each EP monitor is responsible for stripping its own runtime-instance suffixes from profiling event IDs before performing the ONNX lookup. For QNN, this means stripping the `_token_N_M` suffix (a QNN-compiler artifact tagging runtime instances) so event IDs like `/encoder/conv1/Conv_token_3_1` match the ONNX `node.name = /encoder/conv1/Conv`. The cleaned form MUST be stored as `OperatorMetrics.op_path` so the user-visible Node column matches `node.name` (which is what users see in Netron). The strip-then-lookup mechanics MUST stay private to the EP module — no other EP needs to know `_token_N` exists.

### 4.16 FR-16 — ONNX op-type names MUST be used verbatim

When ONNX `node.op_type` is the source for `OperatorMetrics.name`, the value MUST be used verbatim. No translation tables (e.g. `LayerNormalization → LayerNorm`, `Conv → Conv2d`) are permitted, anywhere in the parser pipeline, render layer, or downstream consumers. This rule extends to EP-authoritative names at Layer 2 (e.g. QHAS-reported `ElementWiseAdd` is not translated to `Add` to match an ONNX symbol). Per Cardinal Rule #1 in `CLAUDE.md`, vocabulary translation is hardcoded model-specific logic and is forbidden by construction. If a future requirement emerges for showing both ONNX and EP-native op types side-by-side, that is a render-layer feature (a separate column), not a parser concern.

### 4.17 FR-17 — Information hiding MUST be enforced around EP parsing internals

All EP-specific parsing internals (CSV/JSON readers, sample-aggregation accumulators, vocabulary-specific helpers like `_token_N` regex or `qnn_op_type` field names, the resolver chain, the heuristic) MUST be private to the monitor's containing module. No code outside `src/winml/modelkit/session/monitor/qnn/` (or the equivalent EP submodule for future EPs) is permitted to import any of these helpers. The only shapes visible to callers are the `WinMLEPMonitor` ABC, the canonical `OperatorMetrics` / `OpTraceResult` dataclasses, and the concrete monitor classes with their typed accessors.

### 4.18 FR-18 — `WinMLEPMonitor` MUST expose a typed `result` property (v2.4)

The `WinMLEPMonitor` ABC MUST expose a concrete-default `result` property returning `OpTraceResult | None`. The default implementation returns `getattr(self, "_result", None)` — `None` for monitors that don't produce op-trace data (NullEPMonitor, VitisAIMonitor, OpenVINOMonitor); the populated value for monitors that set `self._result` from `__exit__` (QNNMonitor). Concrete op-tracing monitors set `self._result` after parsing artifacts; the default getter handles the None-vs-set dispatch transparently.

### 4.19 FR-19 — `WinMLEPMonitor` MUST accept an injected ONNX op-type map (v2.4)

The `WinMLEPMonitor` ABC MUST expose a concrete-default `set_onnx_op_types(onnx_op_types: dict[str, str]) -> None` method. The default implementation is a no-op. Op-tracing monitors override to actually store the map for use during their `__exit__` parsing pass. Non-op-tracing monitors (NullEPMonitor, VitisAIMonitor, OpenVINOMonitor) inherit the no-op default and ignore the call. `WinMLSession.perf().__enter__` MUST call `monitor.set_onnx_op_types(map)` unconditionally on every monitor — the no-op default makes this safe regardless of monitor type. Idempotent; the last value wins.

### 4.20 FR-20 — `WinMLEPMonitor` MUST NOT carry a `to_dict()` requirement (v2.4)

The `WinMLEPMonitor` ABC MUST drop the `to_dict()` abstract method that v2.3 carried. v2.3's `to_dict()` was a god-method conflating op-tracing telemetry (QNN: per-operator cycles) with proof-of-execution signals (VitisAI/OpenVINO: NPU-utilisation deltas) under a single polymorphic interface, which is dishonest. Concrete monitors expose data via typed accessors instead:

- `QNNMonitor.result -> OpTraceResult` (op-tracing payload).
- `VitisAIMonitor.proof -> ProofOfExecution | None` (proof-of-execution payload — typed `ProofOfExecution` class flagged as follow-up PR; see OQ-6).
- `OpenVINOMonitor.proof -> ProofOfExecution | None` (same pattern as VitisAI).
- `NullEPMonitor` exposes neither — both `result` and `proof` return `None`.

`commands/perf.py` JSON output MUST switch from a unified `monitor.to_dict()` call to isinstance-based typed accessor dispatch: QNN payloads route to `op_trace`; VitisAI/OpenVINO payloads route to `ep_proof`; NullEPMonitor contributes nothing.

---

## 5. Non-Functional Requirements

### 5.1 Performance

- **NFR-1** CLI-level ergonomics MUST NOT regress. The benchmark command path MUST collapse the current three context managers (stats + hw + ep) to two (perf-with-monitor + hw) and run in comparable wall time.

### 5.2 Reliability

- **NFR-2** No silent failures. If QNN EP cannot be loaded, the session MUST raise a descriptive `CompilationError`. If the CSV is absent or parsing fails, `to_dict()` MUST return `status="no_data"` or `status="parse_failed"` — never an empty structure masquerading as success.
- **NFR-3** Auto-reset MUST be observable. When `session.perf(monitor=...)` auto-resets a previously compiled session to apply the monitor's options, the event MUST log at `WARNING` level: `"auto-resetting compiled session to apply monitor session/provider options"`. Silent mutation is forbidden.
- **NFR-4** Idempotency. `WinMLEPMonitor.get_session_options()` and `WinMLEPMonitor.get_provider_options()` MUST return the same dict on repeated calls within one monitor instance's lifetime. File paths MUST be produced at `__init__`, not on each call.

### 5.3 Usability

- **NFR-5** Exception transparency. Monitor `__exit__` MUST NOT suppress exceptions raised from the `with` body. Parse failures inside `__exit__` are logged and reflected in `to_dict()`, but any active exception from the `with` body MUST propagate normally.

### 5.4 Compatibility

- **NFR-6** No process-global state. `WinMLEPMonitor` instances MUST be stateless between uses (no module-level caches). Importing `QNNMonitor` MUST NOT trigger EP probes, DLL loads, or network activity. `os.chdir` and equivalent global-state mutations are forbidden (see FR-12).
- **NFR-7** Test coverage. All existing tests in `tests/` MUST pass after the refactor. New tests MUST cover: the availability check on both ORT distributions, CSV parsing, session/provider-option merging rules, auto-reset behavior, load-bearing teardown ordering in `perf().__exit__`, and the double-entry guard on `WinMLEPMonitor.__enter__`.
- **NFR-8** Architecture regression test. A test MUST assert that no module outside `src/winml/modelkit/session/monitor/qnn/` (specifically: outside `qnn_monitor.py` and, if option b is chosen, the private `qnn/_internal.py` submodule) imports anything from QNN parsing helpers (formerly `qnn.csv_parser`, `qnn.qhas_parser`; v2.4 narrows the scope to `qnn/_internal.py` since the parser ABC is removed). The enforcement mechanism (a Python AST scan vs a `ruff` rule vs a `mypy` plugin) is per spec §10.1 Q2; recommendation is the AST scan for self-containment. The test MUST fail any future commit that re-exposes a private QNN parsing helper.

---

## 6. Technical Design (high-level)

Detail lives in `2_coreloop.md`. Headline decisions:

- **Architectural pattern**: Hook-based Plugin + Template Method + Observer. `WinMLSession.compile()` is the template method; `WinMLEPMonitor` plugs into two hook points (`get_session_options`, `get_provider_options`); the monitor observes inference via context-manager lifecycle.
- **Session-owned ORT creation**: `WinMLSession` is the sole owner of `ort.InferenceSession` construction. Monitors never create ORT sessions.
- **Singular monitor**: `session.perf(monitor=WinMLEPMonitor|None)`. No `monitors=[...]` multi-monitor support.
- **Monitor factory by explicit dispatch**: `commands/perf.py` contains a 10-line dispatch function. No plugin registry.
- **Report consumers unchanged**: `OpTraceResult.to_dict()` is extended (not replaced); the existing nested schema is preserved; report writers continue to consume `OpTraceResult`.

**WinMLEPMonitor ABC extension (v2.4).** A single abstract contract — `WinMLEPMonitor` — covers the per-EP surface. Defines benchmark lifecycle (`__enter__`/`__exit__`, session/provider-option hooks, availability probe). v2.4 extends it with two concrete-default members: `set_onnx_op_types(map) -> None` (no-op default; op-tracing monitors override) and `result -> OpTraceResult | None` (default returns `getattr(self, "_result", None)`; concrete monitors populate `self._result` from `__exit__`). v2.4 drops the previous polymorphic `to_dict()` abstract method.

QNNMonitor stays single-inheritance (`class QNNMonitor(WinMLEPMonitor)`); ONNX op-type lookup, the four-layer fallback chain (`_resolve_op_type`), the `_token_N` heuristic (`_heuristic_op_type`), and CSV/QHAS reading all live as private internals. `WinMLSession` calls `monitor.set_onnx_op_types(map)` unconditionally on every monitor — the no-op default makes the call safe for non-op-tracing monitors (NullEPMonitor, VitisAIMonitor, OpenVINOMonitor). The previous v2.3 `OpTraceParser` ABC is removed; if a second op-tracing EP lands in the future, the abstraction can be re-extracted from the two concrete implementers.

---

## 7. Design Constraints

- **C-1** `WinMLSession` is the sole owner of ORT session creation. Monitors never create `ort.InferenceSession` instances directly.
- **C-2** Teardown ordering inside `perf().__exit__` is load-bearing: **session reset first, monitor `__exit__` second**. Reversing or parallelizing breaks QNN CSV parsing because QNN EP flushes CSV only on ORT session destruction.
- **C-3** `profiling_level` and `profiling_file_path` are NOT user-overridable via `extra_provider_options`. The monitor owns these keys; user overrides MUST be rejected by construction (enforced via explicit key assignment after merge, not via duplicate-key dict literals).
- **C-4** Only one `WinMLEPMonitor` per session. The `monitor=` parameter is singular. No multi-monitor support today or planned.
- **C-5** `WinMLEPMonitor` instances do not mutate process-global state. No `os.chdir()`, no env-var mutation, no module caches.
- **C-6** `requires_session_teardown: ClassVar[bool]` is an ORT-specific hint to the session that this monitor's data flush requires `ort.InferenceSession` destruction. It is the only place on the base ABC where an ORT implementation detail leaks in; this is accepted as a pragmatic tradeoff (YAGNI vs the architect's proposed `prepare_for_exit` callback).
- **C-7** No vocabulary translation tables. When ONNX `node.op_type` is the source for `OperatorMetrics.name`, the value is used verbatim. Translation maps (e.g. `LayerNormalization → LayerNorm`, `Conv → Conv2d`, `ElementWiseAdd → Add`) are forbidden in the parser, the monitor, the render layer, and any downstream consumer — they would constitute the hardcoded model-vocabulary translation prohibited by `CLAUDE.md` Cardinal Rule #1. The Type column is a "best-available op-type label" column, not a normalised-vocabulary column. Width problems (e.g. `LayerNormalization` overflowing a narrow column) are render-layer concerns and MUST be solved by widening, ellipsis-truncation, or a render-only alias map — never by a parser-side translation table.

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
- **OQ-3** (inherited from spec §10.1 Q1) Helper-placement: option (a) fold all CSV/QHAS/`_token_N` helpers as private methods on `QNNMonitor` (single file ~1000+ lines, one read), or option (b) move ~378 lines into a private sibling submodule `qnn/_internal.py` imported only by `qnn_monitor.py` (recommended in spec §7.2). Both satisfy the information-hiding rule; the recommendation is option (b), but the implementing engineer may flip if the indirection feels unnecessary.
- **OQ-4** (inherited from spec §10.1 Q2) Architecture-test mechanism: enforce the "no external imports of QNN parsing internals" rule via Python AST import-scan (recommended), `ruff` lint rule, or `mypy` plugin. AST scan recommended for self-containment and easy extension when future EPs land.
- **OQ-5** (inherited from spec §10.1 Q3) `_build_op_type_map` placement: keep as a `@staticmethod` on `WinMLSession` (recommended — no state, trivially testable as `WinMLSession._build_op_type_map(...)`) or break out as a free function in a dedicated module (e.g. `session/perf/op_type_map.py`) for tighter testability boundaries. Recommend status quo.
- **OQ-6** (v2.4, new) Typed `proof` accessor follow-up: VitisAIMonitor and OpenVINOMonitor currently return their proof-of-execution data via `to_dict()`. v2.4 removes `to_dict()` from the WinMLEPMonitor ABC contract but leaves the concrete `to_dict` methods on those two monitors in place as transitional. The follow-up PR introduces a typed `proof` property + a new `ProofOfExecution` dataclass to replace those `to_dict` returns. This is out of scope for the v2.4 lift (which is QNN-focused) but flagged so it doesn't get lost. Until the follow-up lands, `commands/perf.py` continues to consume `VitisAIMonitor.to_dict()` / `OpenVINOMonitor.to_dict()` for the `ep_proof` JSON key.

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
- `docs/design/perf/2026-05-03-op-trace-parser-interface-spec.md` v2.0 — focused architectural spec for QNN op-type resolution (ONNX-graph lookup + four-layer fallback chain) as a QNNMonitor-private detail. The PRD's v2.3 changes (SC-7/-8/-9, FR-14 through FR-17, NFR-8, C-7) and v2.4 changes (SC-10, FR-18 through FR-20, dropped FR-13 stub) directly reflect this spec; open questions in spec §10.1 (option a/b for helper placement, architecture-test mechanism, `_build_op_type_map` placement, typed `proof` accessor follow-up) flow back into this PRD when resolved.

### 10.3 Document History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-04-17 | Initial `1_req.md` (deleted). Captured requirements from iterations 01-11. |
| 2.0 | 2026-04-19 | Consolidated into `1_prd.md` per `docs/standards/design-doc-spec.md` v1.0. The prior `1_req.md` was deleted from disk (not deprecated-in-place) because its content is fully subsumed here; the `Supersedes` field preserves the historical link. Incorporated user directives (dual `get_session_options` + `get_provider_options` hooks; extend existing `OpTraceResult.to_dict()` — not replace; no `generate_dummy_inputs`; no `os.chdir`; no multi-monitor; factory dispatch; reorganized test migration). Incorporated critic and architect review findings. |
| 2.1 | 2026-04-19 | Post-audit fixes: added Table of Contents; renumbered Appendix to match spec §4.1 (Document History at §10.3); clarified that `OpTraceResult.to_dict()` already exists and the refactor preserves its nested schema, only adding `status`/`error` keys; clarified that `ep_registry.py` already exists and only gains a new `ensure_initialized()` function; added `fixtures/` to test migration; documented `commands/perf.py` import-path redirects. |
| 2.2 | 2026-04-24 | Relocated from docs/design/optracing/ to docs/design/session/monitor/ per spec §1.5.1 transitional commitment (implementation complete). Removed Transitional Location note. |
| 2.3 | 2026-05-06 | Reflect v1.2 of op-trace-parser interface spec (`docs/design/perf/2026-05-03-op-trace-parser-interface-spec.md`): introduce `OpTraceParser` ABC; `QNNMonitor` implements both ABCs via multiple inheritance; ONNX `node.op_type` as primary op-type source with fallback chain; delete `qnn/csv_parser.py` and `qnn/qhas_parser.py` as public modules. SC-7/-8/-9, FR-13 through FR-17, NFR-8, C-7 added. Open questions in spec §10.1 (option a/b for helper placement, architecture-test mechanism, `_build_op_type_map` placement) noted as pending. |
| 2.4 | 2026-05-08 | Major design simplification, reflecting spec v2.0. **Drop `OpTraceParser` ABC entirely** — premature abstraction for a single concrete implementer (QNNMonitor); multiple inheritance + MRO ordering + a separate ABC file are too much complexity for one EP. Wait for the second op-tracing EP before extracting the abstraction. Replacement: extend `WinMLEPMonitor` ABC with two concrete-default members (`set_onnx_op_types` no-op default, `result` property returning `None`); QNNMonitor stays single-inheritance and owns ONNX-graph lookup + fallback chain as private internals. **Drop `to_dict()` from WinMLEPMonitor ABC contract** — god-method conflating op-tracing telemetry (QNN) with proof-of-execution signals (VitisAI/OpenVINO); concrete monitors expose typed accessors instead (`result` for op-tracing, `proof` for proof-of-execution). FR-13 retired (parser ABC); FR-14 through FR-17 rephrased to live on WinMLEPMonitor + QNNMonitor; FR-18, FR-19, FR-20 added. SC-10 added. NFR-8 scope narrowed. New OQ-6 flags VitisAI/OpenVINO typed `proof` accessor as a follow-up PR (out of scope). `commands/perf.py:542,549` switches from unified `monitor.to_dict()` to isinstance-based typed accessor dispatch. |
| 2.4.1 | 2026-05-08 | Doc-review fixes: FR-6 rewritten for v2.4 (was contradicting FR-20); FR-7 sample code updated to typed-accessor form; §10.5 augmented with dual-behavior note for existing QHAS-authoritative test. |

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
| Modify (v2.4) | `src/winml/modelkit/session/monitor/ep_monitor.py` — extend `WinMLEPMonitor` ABC with concrete-default `set_onnx_op_types(onnx_op_types) -> None` (no-op) and `result` property (returns `getattr(self, "_result", None)`). |
| Remove (v2.4) | `src/winml/modelkit/session/monitor/ep_monitor.py::WinMLEPMonitor.to_dict` — abstract method removed from ABC contract. |
| Remove (v2.4) | `src/winml/modelkit/session/monitor/ep_monitor.py::NullEPMonitor.to_dict` (returns `{}`) — removed; `result` and `proof` both return `None` by inheritance, which is the honest answer. |
| Remove (v2.4) | `src/winml/modelkit/session/monitor/qnn_monitor.py::QNNMonitor.to_dict` — removed; the existing `result` property already exposes `OpTraceResult`, and consumers go through `monitor.result.to_dict()` directly. |
| Modify (v2.4) | `src/winml/modelkit/session/monitor/qnn_monitor.py` — STAYS single-inheritance (`class QNNMonitor(WinMLEPMonitor)`). Override `set_onnx_op_types` to actually store the map. Add private `_resolve_op_type`, `_heuristic_op_type`, `_parse_basic`, `_parse_detail` methods. Existing inline `_parse_artifacts` mode dispatch stays on the monitor; calls the private parse methods. Add `parse_existing_artifacts(level, artifacts, onnx_op_types=None)` classmethod for offline use. |
| Modify (v2.4) | `src/winml/modelkit/session/session.py` — `WinMLSession.perf()` builds an ONNX op-type map (`dict[node.name, node.op_type]`) and injects it via `monitor.set_onnx_op_types(map)` **unconditionally on every monitor** — the WinMLEPMonitor no-op default makes the call safe for non-op-tracing monitors. Adds a static `_build_op_type_map(onnx_path)` helper (final placement per spec §10.1 Q3 — may instead be a free function in a dedicated module for testability). |
| Modify (v2.4) | `src/winml/modelkit/commands/perf.py:542,549` — switch from unified `ctx.monitor.to_dict()` (used today for the `ep_proof` JSON key) to isinstance-based typed accessor dispatch: `isinstance(ctx.monitor, QNNMonitor)` → `op_trace` JSON key from `monitor.result.to_dict()`; `isinstance(ctx.monitor, (VitisAIMonitor, OpenVINOMonitor))` → `ep_proof` JSON key from the transitional `monitor.to_dict()` (until typed `proof` accessor follow-up). NullEPMonitor contributes no key. |
| Keep (v2.4, transitional) | `VitisAIMonitor.to_dict`, `OpenVINOMonitor.to_dict` — stay in place as transitional surfaces until the typed `proof` accessor + `ProofOfExecution` class follow-up (OQ-6) lands. Out of scope for this lift. |
| Delete (v2.3) | `src/winml/modelkit/session/monitor/qnn/csv_parser.py` — DELETED as public module; helpers move to private `qnn/_internal.py` (option b, recommended) or to `qnn_monitor.py` directly (option a). Final choice per spec §10.1 Q1. |
| Delete (v2.3) | `src/winml/modelkit/session/monitor/qnn/qhas_parser.py` — DELETED as public module; same migration path as `csv_parser.py`. |
| Delete (v2.3) | `src/winml/modelkit/session/monitor/qnn/__init__.py` — DELETED if empty after migration (option a) or kept as a private package marker with no public exports (option b). |
| Add (v2.3, conditional on option b) | `src/winml/modelkit/session/monitor/qnn/_internal.py` — NEW private submodule. Imported only by `qnn_monitor.py`; no public exports. Houses the formerly-public CSV/QHAS reading primitives, sample-aggregation accumulators, `_TOKEN_SUFFIX` regex, and `_split_op_event_id` heuristic. |

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
| `tests/unit/session/monitor/qnn/test_csv_parser.py` (post-v2.2 location) | **Delete (v2.3)** — tests of a now-private helper. Coverage migrates to `tests/unit/session/monitor/test_qnn_monitor.py::test_parse_basic_*` integration tests using the same CSV fixtures (via `parse_existing_artifacts`). |
| `tests/unit/session/monitor/qnn/test_csv_parser_samples.py` | **Delete (v2.3)** — tests of a now-private helper. Coverage migrates to integration tests on `QNNMonitor.parse_existing_artifacts(level="basic", ...)`. |
| `tests/unit/session/monitor/qnn/test_event_id_split.py` | **Delete (v2.3)** — tests of a now-private helper. Coverage migrates to `_heuristic_op_type` unit tests on `QNNMonitor`. |
| `tests/unit/session/monitor/qnn/test_qhas_parser.py` | **Delete (v2.3)** — tests of a now-private helper. Coverage migrates to `tests/unit/session/monitor/test_qnn_monitor.py::test_parse_detail_*` integration tests using the same QHAS fixtures (via `parse_existing_artifacts`). |
| — | **New (v2.4)**: `tests/unit/session/monitor/test_ep_monitor_base.py::test_set_onnx_op_types_default_is_no_op` — calling `WinMLEPMonitor.set_onnx_op_types({...})` on a subclass that doesn't override is a no-op (does not raise; nothing visible stored). |
| — | **New (v2.4)**: `tests/unit/session/monitor/test_ep_monitor_base.py::test_result_default_is_none` — `WinMLEPMonitor.result` returns `None` for any subclass that doesn't set `self._result`; returns the value when set. |
| — | **New (v2.4)**: `tests/unit/session/monitor/test_qnn_monitor.py::test_resolve_op_type_walks_chain` — given a `QNNMonitor` with hand-built `_onnx_op_types`, parametrise across (L1 hit/miss) × (L2 hit/None) × (L3 hit/None) and assert `_resolve_op_type` returns the right value at each combination. |
| — | **New (v2.4)**: `tests/unit/session/monitor/test_qnn_monitor.py::test_parse_basic_uses_onnx_lookup` — given an injected `onnx_op_types` map and a small CSV fixture, asserts the produced `OperatorMetrics.name` matches the ONNX value. |
| — | **New (v2.4)**: `tests/unit/session/monitor/test_qnn_monitor.py::test_parse_detail_falls_back_to_qhas` — when the ONNX map misses but QHAS provides `qnn_op_type`, the QHAS-authoritative value wins. |
| `tests/unit/session/monitor/test_qnn_monitor.py:527` (`test_qhas_path_uses_authoritative_qnn_op_type`) | KEEP — pins L2-wins behavior with empty/no `onnx_op_types` map. With populated map (live `WinMLSession.perf()` injection), ONNX precedence overrides QHAS — that contract is pinned by a separate new test (see ONNX-precedence-with-injected-map test added in this migration). |
| — | **New (v2.4)**: `tests/unit/commands/test_perf_json_dispatch.py` — `commands/perf.py` JSON output uses isinstance dispatch; QNN payload routes to `op_trace` key, VitisAI payload routes to `ep_proof` key, NullEPMonitor contributes no key. |
| — | **New (v2.3)**: `tests/unit/architecture/test_qnn_imports.py` (or equivalent location) — Python AST scan asserts no module outside `src/winml/modelkit/session/monitor/qnn/` imports any symbol from `qnn.csv_parser`, `qnn.qhas_parser`, or `qnn._internal`. Pinned by NFR-8. Mechanism per spec §10.1 Q2. |
