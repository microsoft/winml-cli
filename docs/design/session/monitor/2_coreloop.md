# Op-Tracing Refactor — Core Loop Design

**Version**: 2.4.1
**Date**: 2026-05-08
**Status**: Draft
**Module**: session/monitor
**Supersedes**: `docs/design/optracing/2_coreloop.md` v1.0 (consolidated per `docs/standards/design-doc-spec.md`)
**Depends-On**: `docs/design/session/monitor/1_prd.md` v2.4, `docs/standards/design-doc-spec.md`

> **See also**: `docs/design/perf/2026-05-03-op-trace-parser-interface-spec.md` v2.0 — focused architectural spec for QNN op-type resolution (ONNX-graph lookup + four-layer fallback chain) as a QNNMonitor-private detail. Open questions in spec §10.1 (option a vs b for helper placement, architecture-test mechanism, `_build_op_type_map` placement, typed `proof` accessor follow-up) flow back into this doc when resolved. The class signatures and worked walkthroughs in this coreloop are normative copies of the spec's §3 / §6 — when the two diverge, the spec is authoritative.

---

## Table of Contents

- [0. Related Documents](#0-related-documents)
- [0.5 I/O Dependencies](#05-io-dependencies)
- [1. Design Philosophy](#1-design-philosophy)
- [2. Module Structure](#2-module-structure)
- [3. Core Loop Implementation](#3-core-loop-implementation)
- [4. API Design](#4-api-design)
  - [4.1 WinMLEPMonitor — revised ABC](#41-epmonitor--revised-abc)
  - [4.2 NullEPMonitor](#42-nullepmonitor)
  - [4.3 QNNMonitor](#43-qnnmonitor)
  - [4.4 PerfContext](#44-perfcontext)
  - [4.5 WinMLSession.perf() — revised](#45-winmlsessionperf--revised)
  - [4.6 OpTraceResult — extend existing to_dict()](#46-optraceresult--extend-existing-to_dict)
  - [4.7 Factory helper in commands/perf.py](#47-factory-helper-in-commandsperfpy)
- [5. CLI Integration](#5-cli-integration)
- [6. Configuration / Data Structures](#6-configuration--data-structures)
- [7. Error Handling](#7-error-handling)
- [8. Testing Strategy](#8-testing-strategy)
- [9. Integration Points](#9-integration-points)
- [10. Future Work](#10-future-work)
- [11. Revision History](#11-revision-history)

---

## 0. Related Documents

| Document | Path | Purpose |
|----------|------|---------|
| PRD | `./1_prd.md` | Requirements, scope, constraints, migration footprint |
| Spec | `../../standards/design-doc-spec.md` | Normative doc standard this design conforms to |
| Iterations | `./iterations/01.md` – `11.md` | Brainstorming record (informational) |
| Upstream | `../../../src/winml/modelkit/session/session.py` | `WinMLSession` — existing code extended here |
| Upstream | `../../../src/winml/modelkit/session/monitor/ep_monitor.py` | `WinMLEPMonitor` ABC — existing code extended here |
| Upstream | `../../../src/winml/modelkit/commands/perf.py` | CLI benchmark entry point — existing code modified here |
| Deleted | `../../../src/winml/modelkit/optracing/qnn/profiler.py` | `QNNProfiler` — deleted by this refactor |

## 0.5 I/O Dependencies

This refactor orchestrates four subsystems. Data dependencies MUST be understood before reading the core loop.

### 0.5.1 Key actors

| Actor | Role | Location |
|-------|------|----------|
| `WinMLSession` | Owns `ort.InferenceSession` lifecycle; exposes `perf()`. v2.3+: also builds `dict[node.name, node.op_type]` from the ONNX graph at session setup. v2.4: injects the map unconditionally into every monitor via `monitor.set_onnx_op_types(map)` — the WinMLEPMonitor no-op default makes the call safe for non-op-tracing monitors. | `session/session.py` |
| `WinMLEPMonitor` (ABC) | Per-EP observer with two optional config hooks. v2.4: extended with two concrete-default members — `set_onnx_op_types(map)` (no-op default; op-tracing monitors override) and `result` property (default `None`; concrete monitors populate `self._result` from `__exit__`). v2.4: `to_dict()` removed from ABC contract. | `session/monitor/ep_monitor.py` |
| `QNNMonitor` | Concrete monitor for Qualcomm NPU. v2.4: stays single-inheritance `class QNNMonitor(WinMLEPMonitor)`. Owns ALL QNN-specific parsing internals (CSV reading, QHAS reading, `_token_N` strip, leaf-split heuristic, four-layer resolver chain) as private surface. Overrides `set_onnx_op_types` to actually store the map; populates `self._result` from `__exit__`. Adds `parse_existing_artifacts` classmethod for offline use. | `session/monitor/qnn_monitor.py` |
| `PerfContext` | Dataclass yielded by `session.perf()` | `session/session.py` (new) |
| `OpTraceResult` | Structured profiling output. Exposed via the typed `QNNMonitor.result` accessor. | `session/monitor/op_metrics.py` (relocated) |
| `ort.InferenceSession` | Actual ORT session; writes profiling CSV | ORT runtime |
| `ep_registry.ensure_initialized()` | Registers WinML EPs into ORT | `session/ep_registry.py` — **new module-level function added to the existing file**; wraps `WinMLEPRegistry.get_instance().register_to_ort()`. Replaces direct use of `WinMLSession._init_winml_eps_once`. |

### 0.5.2 Data dependency graph

```
┌───────────────────────────────────────────────────────────────────────┐
│ caller                                                                 │
│   session = WinMLSession("model.onnx", device="npu")                   │
│   mon = QNNMonitor(level="basic", output_dir=Path(...))                │
│     │                                                                  │
│     ▼                                                                  │
│   with session.perf(warmup=5, monitor=mon) as ctx:                     │
│                   │                                                    │
│                   ▼                                                    │
│     ┌─────────────────────────────────────────────────────────┐       │
│     │ perf.__enter__:                                         │       │
│     │   extra_sess = mon.get_session_options()  ← HOOK 1     │       │
│     │   extra_prov = mon.get_provider_options() ← HOOK 2     │       │
│     │   if (extra_sess or extra_prov) and compiled:          │       │
│     │     logger.warning("auto-reset..."); self.reset()      │       │
│     │   merge into self._active_session_option_entries       │       │
│     │   merge into self._provider_options                     │       │
│     │   # v2.4: ONNX op-type map injection (UNCONDITIONAL)    │       │
│     │   if self._onnx_path is not None:                       │       │
│     │     m = self._build_op_type_map(self._onnx_path)        │       │
│     │     mon.set_onnx_op_types(m)            ← HOOK 3        │       │
│     │   # No isinstance check — WinMLEPMonitor has a no-op default │       │
│     │   # so the call is safe for any monitor.                │       │
│     │   mon.__enter__()                                       │       │
│     └─────────────────────────────────────────────────────────┘       │
│                   │                                                    │
│                   ▼                                                    │
│   session.run(inputs) ── triggers lazy compile ──▶ uses merged opts   │
│     │                                                                  │
│     │  ort.InferenceSession created with profiling options             │
│     │  CSV being written in background (inside the run)                │
│     │                                                                  │
│                   ▼                                                    │
│     ┌─────────────────────────────────────────────────────────┐       │
│     │ perf.__exit__:                                          │       │
│     │   self._perf_stats = None                                │       │
│     │   if mon.requires_session_teardown:                     │       │
│     │     self.reset()              # drop ort.InferenceSession│       │
│     │     gc.collect()              # release Windows handles │       │
│     │   mon.__exit__(exc_info)       # parse CSV → OpTraceResult│     │
│     │   restore saved options                                 │       │
│     └─────────────────────────────────────────────────────────┘       │
│                                                                        │
│   ctx.monitor.result  (typed accessor) → OpTraceResult | None          │
└───────────────────────────────────────────────────────────────────────┘
```

### 0.5.3 Module responsibility summary

- **`WinMLSession`**: template-method owner. Merges monitor hook contributions at compile; creates `ort.InferenceSession`; runs inference; handles teardown ordering at `perf().__exit__`. v2.3+: also responsible for building the ONNX `node.name → node.op_type` map at session setup (via the static `_build_op_type_map(onnx_path)` helper). v2.4: injects the map unconditionally into every monitor via `monitor.set_onnx_op_types(map)` — no isinstance check. The injection happens before `monitor.__enter__()` so the monitor's `__exit__` parse pass always sees a fully-populated map. Non-op-tracing monitors (NullEPMonitor, VitisAIMonitor, OpenVINOMonitor) inherit the WinMLEPMonitor no-op default and silently ignore the call.
- **`WinMLEPMonitor` (ABC)**: contract definition for benchmark lifecycle. Two optional config hooks (`get_session_options`, `get_provider_options`) with base-class defaults. `is_available` classmethod. Mandatory `__enter__`/`__exit__`. v2.4: extended with two concrete-default members — `set_onnx_op_types(map)` (no-op default) and `result` property (default returns `getattr(self, "_result", None)`). v2.4: `to_dict()` removed from the ABC contract — concrete monitors expose data via typed accessors instead.
- **`QNNMonitor`**: concrete implementation. v2.4: stays single-inheritance (`class QNNMonitor(WinMLEPMonitor)`). Declares session+provider options. Overrides `set_onnx_op_types` to actually store the map. Owns ALL QNN-specific parsing (CSV format, QHAS JSON, `_token_N` strip regex, leaf-split heuristic, sample aggregation, cycle-to-microsecond conversion, the four-layer fallback chain `_resolve_op_type`, the `_heuristic_op_type` method, the private `_parse_basic` / `_parse_detail` parse methods) as private internals — either as private methods on the class or in a private sibling submodule `qnn/_internal.py` (option b, recommended per spec §7.2). Populates `self._result` from `__exit__`; the typed `result` property (inherited from WinMLEPMonitor) exposes it. Adds `parse_existing_artifacts(level, artifacts, onnx_op_types=None)` classmethod for offline use. External code sees only the WinMLEPMonitor ABC + the canonical `OperatorMetrics` / `OpTraceResult` shapes.
- **`commands/perf.py`**: CLI dispatcher. Resolves the right monitor class by explicit `if/elif` on `--ep` and `--op-tracing` flags; constructs it with the appropriate `level` and `output_dir`. v2.4: switches the JSON-output flow from a unified `ctx.monitor.to_dict()` to isinstance-based typed accessor dispatch — `isinstance(ctx.monitor, QNNMonitor)` routes the payload to the `op_trace` JSON key (sourced from `monitor.result.to_dict()`); `isinstance(..., (VitisAIMonitor, OpenVINOMonitor))` routes to `ep_proof` (transitional `monitor.to_dict()` until typed `proof` accessor follow-up). NullEPMonitor contributes no key.
- **`session/ep_registry.py`**: module-level `ensure_initialized()`. Single shared entry point for WinML EP registration. Eliminates the reverse-coupling `QNNMonitor → WinMLSession._init_winml_eps_once`.

---

## 1. Design Philosophy

### 1.1 Purpose

Collapse the dual per-EP hierarchy (`WinMLEPMonitor` + `OpTracer`) into one; fix the broken `onnxruntime-windowsml` session-creation path; eliminate code duplication by routing all ORT session construction through `WinMLSession`.

### 1.2 Core Principles

- **P1 — Session owns the session; monitor informs the session.** `WinMLSession.compile()` is the sole owner of `ort.InferenceSession` construction. Monitors contribute configuration via two hooks but never create ORT sessions directly.
- **P2 — Delete > refactor.** Where two abstractions exist for the same concept, delete one. `QNNProfiler` and `OpTracer` are deleted rather than patched.
- **P3 — Good primitives > bespoke facades.** A clean pair (`WinMLSession`, `QNNMonitor`) composes cleanly into any caller shape. We do not add helper classes or wrapper utilities.
- **P4 — Extension by hook, not by new abstraction.** New EP monitors are added by subclassing `WinMLEPMonitor` and overriding the two hooks. No registry, no factory, no plugin loader.
- **P5 — Explicit over implicit.** No silent fallbacks. No silent session mutations (auto-reset logs at `WARNING`). No silent "ep unsupported" errors (hard-fail at dispatch time).

### 1.3 Design Pattern

**Hook-based Plugin + Template Method + Observer.**

- `WinMLSession.compile()` is the template method: it owns the algorithm (resolve device → build session options → find EP device → merge provider options → create ORT session).
- It calls the monitor at two hook points: `get_session_options()` (add_session_config_entry contributions) and `get_provider_options()` (add_provider_for_devices contributions).
- The `WinMLEPMonitor` itself is a context-managed observer: `__enter__` prepares for observation, `__exit__` finalizes.
- The monitor never replaces session behavior — only augments specific steps.

---

## 2. Module Structure

### 2.1 File layout after refactor

The v2.2 layout (post-relocation from `optracing/`) is updated by v2.3 (deletion of `qnn/csv_parser.py` + `qnn/qhas_parser.py` as public modules) and by v2.4 (extending `WinMLEPMonitor` with concrete-default members; QNNMonitor stays single-inheritance — no `op_trace_parser.py` is added).

```
src/winml/modelkit/
├── session/
│   ├── session.py                           # modified (see §4.5) — v2.4: adds _build_op_type_map + unconditional injection
│   ├── ep_registry.py                       # modified — existing file; adds ensure_initialized() (see §4.3)
│   └── monitor/
│       ├── ep_monitor.py                    # MODIFIED (v2.4) — adds set_onnx_op_types (no-op default) + result property; drops to_dict from ABC
│       ├── hw_monitor.py                    # unchanged
│       ├── qnn_monitor.py                   # MAJOR REWRITE (v2.4) — single inheritance from WinMLEPMonitor; private resolver + parse methods (see §4.3)
│       ├── vitisai_monitor.py               # transitional (v2.4) — keeps to_dict for now; typed `proof` accessor flagged as follow-up
│       ├── openvino_monitor.py              # transitional (v2.4) — same as vitisai
│       ├── op_metrics.py                    # moved from optracing/result.py + .to_dict() extension
│       ├── report.py                        # moved from optracing/report.py
│       └── qnn/                             # PRIVATE submodule — only qnn_monitor.py imports from here
│           ├── csv_parser.py                # DELETED (v2.3) — folded into _internal.py (or qnn_monitor.py if option a)
│           ├── qhas_parser.py               # DELETED (v2.3) — same
│           ├── _internal.py                 # NEW (v2.3, option b, recommended) — folds csv_parser.py + qhas_parser.py
│           ├── viewer.py                    # status: see §2.1.1 — verify and document during migration
│           └── __init__.py                  # DELETED if option (a); kept private (no exports) if option (b)
├── commands/
│   └── perf.py                              # modified (see §5) — v2.4: isinstance-based typed accessor dispatch for JSON output
└── optracing/                               # DELETED ENTIRELY (v2.2)
```

#### 2.1.1 Status of `qnn/viewer.py`

`viewer.py` is the QHAS-viewer shell-out helper (invokes the QNN SDK's `qnn-profile-viewer` to convert raw profiling output into QHAS JSON). Per spec §7.4 ("What does NOT change"), the viewer shell-out is unchanged — viewer invocation is a device/SDK lifecycle concern owned by the monitor, only artifact parsing moves into the private `_parse_detail` method. The implementing engineer should verify the file's current state during the migration and document its post-refactor location: it stays under `qnn/` as a sibling module imported only by `qnn_monitor.py`, OR moves into `_internal.py` if option b unifies all QNN-private surface there. Either way, no module outside `qnn/` is permitted to import from `viewer.py`.

### 2.2 Key dependencies

- `WinMLSession.compile()` calls `mon.get_session_options()` and `mon.get_provider_options()` on the active monitor.
- **(v2.4)** `WinMLSession.perf().__enter__` builds the ONNX op-type map via `_build_op_type_map(onnx_path)` and injects it via `monitor.set_onnx_op_types(map)` **unconditionally** on every monitor BEFORE calling `monitor.__enter__()`. The WinMLEPMonitor no-op default makes the call safe for non-op-tracing monitors (NullEPMonitor, VitisAIMonitor, OpenVINOMonitor) — they inherit the default and silently ignore the call. QNNMonitor overrides to actually store the map.
- `QNNMonitor.is_available()` calls `session/ep_registry.py::ensure_initialized()` (NOT `WinMLSession._init_winml_eps_once`, which is deleted).
- `QNNMonitor.__exit__` reads the CSV written by QNN EP during `session.run()` and produces an `OpTraceResult`, populating `self._result`. **(v2.4)** Internally, `__exit__` dispatches to private `self._parse_basic(...)` or `self._parse_detail(...)` based on `self._level`; both private parse methods call the private `_resolve_op_type` template method.
- `commands/perf.py` imports `QNNMonitor` and `VitisAIMonitor` directly; no registry. **(v2.4)** Uses isinstance-based typed accessor dispatch for JSON output: `isinstance(ctx.monitor, QNNMonitor)` → `op_trace` JSON key from `monitor.result.to_dict()`; `isinstance(ctx.monitor, (VitisAIMonitor, OpenVINOMonitor))` → `ep_proof` JSON key from transitional `monitor.to_dict()`.
- **(v2.3)** No module outside `src/winml/modelkit/session/monitor/qnn/` imports `qnn.csv_parser`, `qnn.qhas_parser`, or `qnn._internal`. Pinned by the architecture regression test (PRD NFR-8).

---

## 3. Core Loop Implementation

### 3.1 High-level flow

The canonical flow is the CLI benchmark with op-tracing enabled. It exercises every hook point.

```
wmk perf -m resnet50 --device npu --op-tracing basic
    │
    ▼
commands/perf.py
    │   monitor = _resolve_ep_monitor(ep="qnn", op_tracing="basic", output_dir=...)
    │           → QNNMonitor(level="basic", output_dir=...)
    │
    ▼
with session.perf(warmup=warmup, monitor=monitor) as ctx, HWMonitor() as hw:
    │                                                       ▲
    │                                                       │  orthogonal,
    │                                                       │  process-wide counters
    │
    │   perf.__enter__:
    │     extra_sess = monitor.get_session_options()
    │         → {"session.disable_cpu_ep_fallback": "1",
    │            "ep.context_enable": "1",
    │            "ep.context_embed_mode": "0"}
    │     extra_prov = monitor.get_provider_options()
    │         → {"backend_path": "QnnHtp.dll", ...,
    │            "profiling_level": "detailed",
    │            "profiling_file_path": "<abs path>"}
    │     if (extra_sess or extra_prov) and self._session is not None:
    │         logger.warning("auto-resetting compiled session ...")
    │         self.reset()
    │     merge into session state
    │     # v2.4: build + inject ONNX op-type map UNCONDITIONALLY
    │     if self._onnx_path is not None:
    │         onnx_op_types = WinMLSession._build_op_type_map(self._onnx_path)
    │             → dict[node.name, node.op_type]   (or {} if onnx_path missing/corrupt)
    │         monitor.set_onnx_op_types(onnx_op_types)
    │         # No isinstance check — WinMLEPMonitor.set_onnx_op_types has a
    │         # no-op default; non-op-tracing monitors silently ignore.
    │     monitor.__enter__()                     # sets _entered flag
    │     yield PerfContext(stats=PerfStats(...), monitor=monitor)
    │
    │   for _ in range(iterations):
    │       session.run(inputs)
    │         → first call triggers lazy compile:
    │           - SessionOptions.add_session_config_entry(k, v) for each extra_sess
    │           - add_provider_for_devices([qnn_ep_dev], merged_provider_opts)
    │           - ort.InferenceSession(...) created
    │         → subsequent calls run; QNN EP appends to profiling CSV
    │
    ▼
    perf.__exit__:
      self._perf_stats = None
      exc_info = sys.exc_info()                   # may be (None,None,None)
      try:
          if monitor.requires_session_teardown:   # QNN: True
              self.reset()                        # drops ort.InferenceSession → flushes CSV
              gc.collect()                        # release Windows file handles
      finally:
          try:
              monitor.__exit__(*exc_info)         # parses CSV → OpTraceResult
          finally:
              restore saved session/provider options
    │
    ▼
# After the `with` block
if op_tracing:
    display_op_trace_report(ctx.monitor.result, console)   # OpTraceResult (not dict)
    write_op_trace_json(ctx.monitor.result, json_path)
```

### 3.2 Lifecycle walkthrough — benchmark-only (no EP monitor)

```python
with session.perf(warmup=10) as ctx:      # monitor=None → NullEPMonitor
    for _ in range(100):
        session.run(inputs)
print(ctx.stats.mean_ms)
```

- `NullEPMonitor.get_session_options()` → `{}`
- `NullEPMonitor.get_provider_options()` → `{}`
- `needs_recompile = False` → no auto-reset
- `NullEPMonitor.requires_session_teardown = False` → no reset on exit
- `ctx.monitor.result` is `None` (NullEPMonitor inherits the default)

Zero behavior change from today's `session.perf(warmup=10) as stats` — just one extra level of indirection via `ctx.stats`.

### 3.3 Lifecycle walkthrough — benchmark with VitisAI proof-of-execution

```python
with session.perf(warmup=10, monitor=VitisAIMonitor()) as ctx, HWMonitor() as hw:
    session.run(inputs)
```

- `VitisAIMonitor.get_session_options()` → `{}` (inherits default)
- `VitisAIMonitor.get_provider_options()` → `{}` (inherits default)
- `needs_recompile = False`
- `VitisAIMonitor.__enter__` takes xrt-smi snapshot
- `VitisAIMonitor.__exit__` takes xrt-smi snapshot; `ctx.monitor.npu_proven` is True/False
- `VitisAIMonitor.requires_session_teardown = False` → no reset on exit

Same API, different monitor. No QNN-specific code paths activated.

### 3.4 Lifecycle walkthrough — standalone profile (no CLI)

```python
session = WinMLSession("model.onnx", device="npu")
with session.perf(monitor=QNNMonitor(level="basic")) as ctx:
    for _ in range(10):
        session.run(my_inputs)              # caller provides inputs
print(ctx.monitor.result.to_dict() if ctx.monitor.result else "(no op-trace data)")
```

- No helper class. No `generate_dummy_inputs`. The caller generates inputs.
- 6 lines excluding the import.

### 3.5 Teardown ordering — load-bearing invariant

Inside `session.perf().__exit__`, the following order is **load-bearing**:

```
1. Stop perf timing (self._perf_stats = None)
2. Capture sys.exc_info() (for propagation)
3. IF monitor.requires_session_teardown:
      self.reset()          ← drops ort.InferenceSession; QNN flushes CSV
      gc.collect()          ← Windows: release file handle on CSV
4. monitor.__exit__(*exc_info) ← parses CSV → OpTraceResult
5. Restore saved session_options + provider_options
```

Reversing step 3 and step 4 produces an empty CSV file. Running them concurrently produces a race. Explicitly forbidden by **C-2** in the PRD.

An integration test (§8.3) asserts `session._session is None` during `monitor.__exit__` to lock this invariant.

---

## 4. API Design

### 4.1 `WinMLEPMonitor` — revised ABC

```python
# session/monitor/ep_monitor.py

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .op_metrics import OpTraceResult


class WinMLEPMonitor(ABC):
    """Per-EP observer attached to a WinMLSession for an inference window."""

    # ---- Optional hooks: defaults provided; subclasses override as needed ----

    # ORT-specific hint: does this monitor's data flush require ort.InferenceSession destruction?
    # True for QNNMonitor (CSV flushes on session delete). False for passive monitors.
    requires_session_teardown: ClassVar[bool] = False

    def get_session_options(self) -> dict[str, str]:
        """Entries to pass to SessionOptions.add_session_config_entry(). Default: none."""
        return {}

    def get_provider_options(self) -> dict[str, str]:
        """Options to merge into add_provider_for_devices(). Default: none."""
        return {}

    # ---- NEW (v2.4): ONNX op-type map injection ----

    def set_onnx_op_types(self, onnx_op_types: dict[str, str]) -> None:
        """Inject the ONNX node.name -> node.op_type map.

        Default: no-op. Op-tracing monitors override this to store the map
        for use during their __exit__ parsing pass. Non-op-tracing monitors
        (NullEPMonitor, VitisAIMonitor, OpenVINOMonitor) inherit this default
        and silently ignore the call.

        WinMLSession calls this unconditionally on every monitor before
        __enter__. Idempotent; the last value wins.
        """
        pass

    # ---- NEW (v2.4): typed op-trace result accessor ----

    @property
    def result(self) -> "OpTraceResult | None":
        """Wrapped op-trace result. None for monitors that don't produce it.

        The default returns ``getattr(self, "_result", None)``: ``None`` for
        monitors that never set ``self._result`` (NullEPMonitor, VitisAI,
        OpenVINO), the populated value for op-tracing monitors that set
        ``self._result`` from ``__exit__`` (QNNMonitor).

        No subclass override required unless the subclass wants to compute
        the result lazily — the default getattr-based dispatch is sufficient
        for normal usage.
        """
        return getattr(self, "_result", None)

    # ---- Mandatory contract ----

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Whether this monitor's EP and infrastructure are usable on this system."""

    @abstractmethod
    def __enter__(self) -> Self: ...

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """MUST NOT suppress exceptions from the `with` body."""

    # NOTE (v2.4): to_dict() is REMOVED from the ABC contract.
    # Concrete monitors expose data via typed accessors instead:
    #   - QNNMonitor: monitor.result -> OpTraceResult (op-tracing payload)
    #   - VitisAIMonitor / OpenVINOMonitor: monitor.proof -> ProofOfExecution
    #     (proof-of-execution payload — typed `ProofOfExecution` class flagged
    #     as a follow-up PR; out of scope for this lift)
    #   - NullEPMonitor: exposes neither (both `result` and `proof` return None)
    # commands/perf.py routes JSON output by isinstance dispatch.
```

Invariants:

- `get_session_options()` and `get_provider_options()` MUST be idempotent (NFR-4).
- `__enter__` MUST raise `RuntimeError("<Monitor> already entered")` if called twice without intervening `__exit__`.
- `__exit__` MUST NOT return `True` (which would suppress exceptions).
- `set_onnx_op_types(map)` is idempotent. The last value wins. WinMLSession calls it before `__enter__`; tests may call it any number of times. The base no-op default is safe for any subclass.
- `result` returns `None` until a subclass populates `self._result`. The default getter dispatches via `getattr` — no subclass override needed for the typical "set self._result in __exit__" pattern.

### 4.2 `NullEPMonitor`

Largely unchanged from current `ep_monitor.py:62-88`. Inherits the existing defaults (`get_session_options()` / `get_provider_options()` return `{}`; `requires_session_teardown = False`). v2.4: also inherits the new concrete defaults — `set_onnx_op_types(map)` is a no-op (NullEPMonitor doesn't consume the map), and `result` returns `None` (NullEPMonitor never sets `_result`). The `to_dict()` method that previously returned `{}` is removed; `commands/perf.py` skips NullEPMonitor entirely in JSON output (no `op_trace` key, no `ep_proof` key).

### 4.3 `QNNMonitor`

v2.4 keeps single inheritance: `class QNNMonitor(WinMLEPMonitor)`. ALL QNN-specific helpers (CSV reading, sample extraction, QHAS reading, `_token_N` stripping, leaf-split heuristic, four-layer resolver chain) live as private methods on the monitor or in a private sibling submodule `qnn/_internal.py` (option b, recommended) and are invisible to anything outside the QNN module. Nothing about CSV/QHAS parsing leaks out — the only shapes visible to callers are the WinMLEPMonitor ABC and the canonical dataclasses.

```python
# session/monitor/qnn_monitor.py

import re

from .ep_monitor import WinMLEPMonitor
from .op_metrics import OperatorMetrics, OpTraceResult


class QNNMonitor(WinMLEPMonitor):
    """Qualcomm NPU per-op profiler via ORT's QNN EP.

    Single-inheritance WinMLEPMonitor subclass. Owns ALL QNN-specific concerns
    end-to-end: device lifecycle (from WinMLEPMonitor) plus CSV/QHAS reading,
    `_token_N` stripping, leaf-split heuristic, and the four-layer resolver
    chain — all as private internals.

    Produces an OpTraceResult with per-operator cycle counts (level="basic")
    or full QHAS roofline / DMA traffic (level="detail"). The result is
    exposed via the typed ``result`` property (inherited from WinMLEPMonitor;
    populated from ``__exit__``).
    """

    requires_session_teardown: ClassVar[bool] = True
    # QNN EP flushes the profiling CSV only on ort.InferenceSession destruction.

    # _token_N suffix is a QNN-compiler artefact (token-position-tagged
    # repeats of the same op). Stripped before the ONNX-graph lookup so
    # event IDs like "/encoder/conv1/Conv_token_1_2" match ONNX node
    # names like "/encoder/conv1/Conv". This regex stays inside the QNN
    # module — it is not a general-purpose op-name normaliser.
    _TOKEN_SUFFIX = re.compile(r"_token_\d+(?:_\d+)?")

    def __init__(
        self,
        level: Literal["basic", "detail"] = "basic",
        output_dir: Path | None = None,
        extra_provider_options: Mapping[str, str] | None = None,
        onnx_op_types: dict[str, str] | None = None,   # NEW (v2.3+)
    ) -> None:
        if level not in ("basic", "detail"):
            raise ValueError(f"level must be 'basic' or 'detail', got {level!r}")
        super().__init__()
        self._level = level
        # Idempotency: paths produced at __init__, not per-call
        self._output_dir = Path(output_dir) if output_dir else Path(
            tempfile.mkdtemp(prefix="qnn_profile_")
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = (self._output_dir / "profiling_output.csv").resolve()
        self._qhas_path = (self._output_dir / "qhas_output.json").resolve()
        self._extra = dict(extra_provider_options or {})
        self._entered = False
        self._onnx_op_types: dict[str, str] = dict(onnx_op_types or {})
        self._result: OpTraceResult | None = None

    # -- WinMLEPMonitor: availability, lifecycle, options --------------------

    @classmethod
    def is_available(cls) -> bool:
        import onnxruntime as ort
        if "QNNExecutionProvider" in ort.get_available_providers():
            return True
        # WinML-registered path. `ensure_initialized` is a NEW module-level function
        # added to the existing `session/ep_registry.py`; it wraps the existing
        # `WinMLEPRegistry.get_instance().register_to_ort()` as an idempotent entry point.
        from ..ep_registry import ensure_initialized
        ensure_initialized()
        return any(d.ep_name == "QNNExecutionProvider" for d in ort.get_ep_devices())

    def get_session_options(self) -> dict[str, str]:
        return {
            "session.disable_cpu_ep_fallback": "1",
            "ep.context_enable": "1",
            "ep.context_embed_mode": "0",
        }

    def get_provider_options(self) -> dict[str, str]:
        # Build in layers; last writer wins. Owner-enforced keys applied LAST.
        opts: dict[str, str] = {
            "backend_path": "QnnHtp.dll",
            "htp_performance_mode": "high_performance",
            "htp_graph_finalization_optimization_mode": "3",
            "enable_htp_fp16_precision": "1",
        }
        opts.update(self._extra)
        # C-3: these two keys are NEVER user-overridable
        opts["profiling_level"] = "optrace" if self._level == "detail" else "detailed"
        opts["profiling_file_path"] = str(self._csv_path)
        return opts

    def set_onnx_op_types(self, onnx_op_types: dict[str, str]) -> None:
        """Override the WinMLEPMonitor no-op default — QNN consumes the map."""
        self._onnx_op_types = dict(onnx_op_types)

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("QNNMonitor already entered")
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Parse whatever artifacts are on disk; populate self._result.
        # The inherited `result` property exposes it. Never suppress caller
        # exceptions (no `return True`).
        try:
            self._result = self._parse_artifacts()
        except Exception as e:
            logger.warning("QNNMonitor: artifact parse failed: %s", e)
            self._result = OpTraceResult(
                model=None, device="npu", tracing_level=self._level,
                ep="QNNExecutionProvider", tracing_backend="qnn",
                operators=[], summary={}, artifacts={"csv": str(self._csv_path)},
                status="parse_failed", error=str(e),
            )

    # NOTE: `to_dict()` is removed (v2.4). Consumers go through the typed
    # `result` accessor (inherited from WinMLEPMonitor) and call
    # `monitor.result.to_dict()` directly on the OpTraceResult dataclass.

    # -- Private parsing internals (NOT part of any ABC) ----------------

    def _resolve_op_type(
        self, op_path: str, ep_authoritative: str | None = None
    ) -> str:
        """Walk the four-layer fallback chain (QNNMonitor-private template).

        L1: ONNX graph lookup by ``op_path`` (== node.name post-strip).
        L2: ``ep_authoritative`` (e.g. ``qhas.qnn_op_type``).
        L3: ``_heuristic_op_type(op_path)`` (QNN leaf-split).
        L4: Raw ``op_path``.

        Each layer is monotonic in quality: a higher layer's hit always
        wins. Empty/None at any layer falls through.
        """
        onnx_hit = self._onnx_op_types.get(op_path)
        if onnx_hit:
            return onnx_hit
        if ep_authoritative:
            return ep_authoritative
        heuristic = self._heuristic_op_type(op_path)
        if heuristic:
            return heuristic
        return op_path

    def _heuristic_op_type(self, op_path: str) -> str:
        """Heuristic-only fallback: leaf-split with strip safety.

        Preserves the strip semantics from the legacy _split_op_event_id helper:
        outer whitespace is stripped, inner whitespace around the leaf is stripped,
        and trailing-slash inputs fall back to the original (never empty).
        """
        cleaned = self._TOKEN_SUFFIX.sub("", op_path).strip()
        if "/" not in cleaned:
            return cleaned
        leaf = cleaned.rsplit("/", 1)[-1].strip()
        return leaf if leaf else cleaned  # trailing-slash → fall back to full
```

This preserves the strip-safety semantics of the legacy `_split_op_event_id`
helper at `src/winml/modelkit/session/monitor/qnn/csv_parser.py:39-77`,
covered by `tests/unit/session/monitor/qnn/test_event_id_split.py`. Do not
simplify these guards in the production implementation — the strip-safety
tests are part of the migration's preserve-behavior contract.

```python

    def _parse_basic(self, artifacts: dict[str, Path]) -> list[OperatorMetrics]:
        """Parse the CSV emitted by QNN EP at profiling_level='detailed'.

        Resolution: ``name`` is sourced via ``_resolve_op_type(op_path,
        ep_authoritative=None)``. Basic mode has no QHAS-authoritative
        field, so Layer 2 is always None — the chain falls through to
        ONNX (L1), then to the heuristic (L3), then to raw op_path (L4).
        """
        csv_path = artifacts.get("csv")
        if csv_path is None or not csv_path.is_file():
            return []
        rows = self._read_qnn_csv(csv_path)             # private (formerly qnn/csv_parser.py)
        samples = self._extract_samples(rows)            # private
        ops, meta = self._aggregate_operators(samples)   # private
        cycle_to_us = self._compute_cycle_ratio(meta)    # private
        return [
            self._to_operator_metrics(
                op, cycle_to_us,
                name=self._resolve_op_type(op["op_path"], ep_authoritative=None),
            )
            for op in ops
        ]

    def _parse_detail(self, artifacts: dict[str, Path]) -> list[OperatorMetrics]:
        """Parse the QHAS JSON emitted by ``qnn-profile-viewer``.

        Resolution: ``name`` is sourced via ``_resolve_op_type(op_path,
        ep_authoritative=op["qnn_op_type"])``. The QHAS-authoritative
        ``qnn_op_type`` becomes Layer 2 — when ONNX misses (compiler-
        inserted glue ops, fused subgraphs), QHAS wins.
        """
        qhas_path = artifacts.get("qhas")
        if qhas_path is None or not qhas_path.is_file():
            return []
        ops = self._read_qhas(qhas_path)                 # private (formerly qnn/qhas_parser.py)
        return [
            self._to_operator_metrics_detail(
                op,
                name=self._resolve_op_type(
                    op["op_path"],
                    ep_authoritative=op["qnn_op_type"],
                ),
            )
            for op in ops
        ]

    def _parse_artifacts(self) -> OpTraceResult:
        """Dispatch to _parse_basic / _parse_detail based on self._level.

        Wraps the resulting list[OperatorMetrics] into an OpTraceResult
        with summary / status / artifacts. On Windows file-handle lag:
        retries once with 50ms delay (R-2 mitigation).
        """
        if self._level == "detail":
            ops = self._parse_detail({"qhas": self._qhas_path})
            artifacts = {"qhas": str(self._qhas_path), "csv": str(self._csv_path)}
        else:
            ops = self._parse_basic({"csv": self._csv_path})
            artifacts = {"csv": str(self._csv_path)}
        if not ops:
            return OpTraceResult(
                model=None, device="npu", tracing_level=self._level,
                ep="QNNExecutionProvider", tracing_backend="qnn",
                operators=[], summary={}, artifacts=artifacts,
                status="no_data", error=None,
            )
        return OpTraceResult(
            model=None, device="npu", tracing_level=self._level,
            ep="QNNExecutionProvider", tracing_backend="qnn",
            operators=ops, summary=self._build_summary(ops),
            artifacts=artifacts, status="ok", error=None,
        )

    @classmethod
    def parse_existing_artifacts(
        cls,
        level: Literal["basic", "detail"],
        artifacts: dict[str, Path],
        onnx_op_types: dict[str, str] | None = None,
    ) -> OpTraceResult:
        """Public entry point for offline parsing of pre-existing CSV/QHAS files.

        Useful for tests and ad-hoc analysis scripts that don't run a live
        benchmark. Builds a transient QNNMonitor instance, dispatches to
        the appropriate private parse method, and wraps the result.
        """
        instance = cls(
            level=level,
            output_dir=(
                artifacts.get("csv") or artifacts.get("qhas")
            ).parent if (artifacts.get("csv") or artifacts.get("qhas")) else None,
            onnx_op_types=onnx_op_types,
        )
        ops = (
            instance._parse_detail(artifacts) if level == "detail"
            else instance._parse_basic(artifacts)
        )
        return instance._wrap_ops_into_result(ops, artifacts=artifacts)

    # _read_qnn_csv, _extract_samples, _aggregate_operators,
    # _compute_cycle_ratio, _to_operator_metrics,
    # _read_qhas, _to_operator_metrics_detail, _build_summary,
    # _wrap_ops_into_result
    # ... all private. Either methods on this class (option a) or imported
    # from qnn/_internal.py (option b, recommended). See spec §7.2.
```

**On CWD / `*_schematic.bin`**: Per **C-5** and **FR-12**, `QNNMonitor` does NOT call `os.chdir`. If the QNN SDK emits `*_schematic.bin` to the process's CWD rather than to `profiling_file_path`'s directory, `_parse_artifacts` locates it via `glob` from the expected fallback locations and logs a `WARNING` if not found. The `detail`-mode path degrades gracefully to basic CSV parsing in that case (FR-5).

**On token-suffix stripping (v2.3+)**. The `_TOKEN_SUFFIX` regex is applied in two places, and only inside the QNN module:

1. Inside the private CSV-reading helpers when constructing the `op_path` field — the cleaned form is what gets stored on `OperatorMetrics.op_path` and what gets passed as the lookup key to `_resolve_op_type`. This is the load-bearing bridge: without it, every path-style event ID would miss the ONNX lookup because `node.name` does not carry `_token_N`.
2. Inside `_heuristic_op_type` as belt-and-braces, in case a caller passes a still-suffixed string. Idempotent — double-stripping does nothing.

The cleaned form serves a deliberate UX choice: the user-visible Node column matches `node.name` exactly (which is what users see in Netron), so cross-referencing slow ops against the model graph is a one-step lookup. See spec §3.2 and §3.5 for worked walkthroughs across happy-path, glue-op, and pathological cases.

**On naming convention (v2.3+)**. Per FR-16 / C-7 in the PRD, the value returned by `_resolve_op_type` is rendered verbatim. `LayerNormalization` stays `LayerNormalization` (not translated to `LayerNorm`); `ElementWiseAdd` stays `ElementWiseAdd` (not translated to `Add`). No translation tables anywhere in the parser, monitor, or render layer. Width problems are render-layer concerns (see spec §4 for the rationale).

**On `VitisAIMonitor` and `OpenVINOMonitor` (v2.4 transitional)**. v2.4 removes `to_dict()` from the WinMLEPMonitor ABC contract, but VitisAIMonitor and OpenVINOMonitor keep their concrete `to_dict` methods in place as transitional surfaces. The follow-up PR (OQ-6) introduces a typed `proof` property + a new `ProofOfExecution` dataclass to replace those returns. Until that PR lands, `commands/perf.py`'s isinstance dispatch reads `monitor.to_dict()` directly on those two concrete classes for the `ep_proof` JSON key. NullEPMonitor's `to_dict` (which returned `{}`) is removed in v2.4 — NullEPMonitor contributes no JSON key.

### 4.4 `PerfContext`

```python
# session/session.py

@dataclass(frozen=True)
class PerfContext:
    """Yielded by WinMLSession.perf(). Aggregates perf stats and the attached EP monitor."""
    stats: PerfStats
    monitor: WinMLEPMonitor        # NullEPMonitor when caller passed monitor=None
```

Frozen to prevent accidental mutation during the `with` block. Not a replacement for `PerfStats` — both `stats` and `monitor` are addressable by attribute.

### 4.5 `WinMLSession.perf()` — revised

```python
# session/session.py

@contextmanager
def perf(
    self,
    warmup: int = 0,
    monitor: WinMLEPMonitor | None = None,
) -> Generator[PerfContext, None, None]:
    """Run a scoped performance window.

    Yields:
        PerfContext with `stats: PerfStats` and `monitor: WinMLEPMonitor`.

    Behavior:
        - If `monitor` contributes session_options or provider_options and this
          session is already compiled, the ORT session is auto-reset with a
          WARNING log. Future runs within the `with` body trigger recompile
          with the merged options.
        - If `monitor.requires_session_teardown`, `self.reset()` is called at
          exit BEFORE `monitor.__exit__`, so the monitor sees the fully-flushed
          artifacts (e.g., QNN CSV).
        - Nested perf() is forbidden — raises RuntimeError on re-entry.
    """
    if self._perf_stats is not None:
        raise RuntimeError("session.perf() already active (nested perf is forbidden)")

    mon = monitor or NullEPMonitor()

    # Collect hook contributions
    extra_sess = mon.get_session_options()
    extra_prov = mon.get_provider_options()
    needs_recompile = (extra_sess or extra_prov) and self._session is not None
    if needs_recompile:
        logger.warning(
            "session.perf(): auto-resetting compiled session to apply monitor "
            "session/provider options (monitor=%s)", type(mon).__name__
        )
        self.reset()

    # Save + merge
    saved_sess_entries = dict(self._active_session_option_entries)
    saved_prov = dict(self._provider_options)
    self._active_session_option_entries = {**saved_sess_entries, **extra_sess}
    self._provider_options = {**saved_prov, **extra_prov}

    # NEW in v2.4: inject ONNX op-type map UNCONDITIONALLY on every monitor.
    # The WinMLEPMonitor.set_onnx_op_types base implementation is a no-op, so the
    # call is safe for non-op-tracing monitors (NullEPMonitor, VitisAIMonitor,
    # OpenVINOMonitor) — they inherit the default and silently ignore the map.
    # QNNMonitor overrides to actually store the map.
    if self._onnx_path is not None:
        onnx_op_types = WinMLSession._build_op_type_map(self._onnx_path)
        mon.set_onnx_op_types(onnx_op_types)

    stats = PerfStats(warmup=warmup)
    self._perf_stats = stats
    mon.__enter__()

    try:
        yield PerfContext(stats=stats, monitor=mon)
    finally:
        self._perf_stats = None
        exc_info = sys.exc_info()    # propagates caller exception to monitor.__exit__
        try:
            if mon.requires_session_teardown:
                self.reset()
                gc.collect()         # Windows: release CSV file handle (R-2)
        finally:
            try:
                mon.__exit__(*exc_info)
            finally:
                self._active_session_option_entries = saved_sess_entries
                self._provider_options = saved_prov
```

Also in `WinMLSession.__init__`:

```python
self._active_session_option_entries: dict[str, str] = {}   # NEW state
```

And in `_build_session_options(self, device)`, add the application of monitor contributions:

```python
def _build_session_options(self, device: str) -> ort.SessionOptions:
    ...
    # Apply monitor-contributed session config entries (if perf() context is active)
    for key, value in self._active_session_option_entries.items():
        opts.add_session_config_entry(key, value)
    ...
```

Also add a static helper that builds the ONNX op-type map (NEW in v2.3). Final placement is per spec §10.1 Q3 — a `@staticmethod` on `WinMLSession` is recommended (no state, trivially testable as `WinMLSession._build_op_type_map(...)`); a free function in a dedicated module would also work but adds a file with no clear additional benefit.

```python
@staticmethod
def _build_op_type_map(onnx_path: Path | None) -> dict[str, str]:
    """Build a node.name → node.op_type map from an ONNX file.

    Loads only graph metadata (``load_external_data=False``) so the
    cost is ~milliseconds even for multi-GB models with separate
    weight files. Returns an empty dict when the path is None,
    missing, or unparseable — the parser falls through the chain
    in that case (no warning; the empty-map case is well-defined).

    Anonymous nodes (``node.name == ""``) are excluded — they cannot
    key the map (multiple anonymous nodes would collide). Rare in
    production-exported models; quietly skipping them is correct.
    """
    if onnx_path is None:
        return {}
    p = Path(onnx_path)
    if not p.is_file():
        return {}
    try:
        import onnx
        model = onnx.load(str(p), load_external_data=False)
    except Exception:
        return {}
    return {n.name: n.op_type for n in model.graph.node if n.name}
```

The `load_external_data=False` flag is essential. For models like Qwen3-0.6B-Q8_0 (≈600MB of weights in a sidecar `.bin`) we only need the graph topology, never the tensors. With the flag set, loading is roughly proportional to the protobuf graph size — typically tens of milliseconds.

### 4.6 `OpTraceResult` — extend existing `to_dict()`

`OpTraceResult.to_dict()` **already exists** at `optracing/result.py:79-95`. The refactor **preserves its nested schema exactly** (required by OOS-4: report writers are not modified). Two additive changes:

1. Two new dataclass fields (`status`, `error`) with defaults that keep existing callers working.
2. Two new top-level keys in `to_dict()`, placed alongside the existing keys — the existing `metadata`, `summary`, `operators`, `statistics`, `artifacts` structure is untouched.
3. `model` field relaxed from `str` to `str | None` to support standalone programmatic profiling where the source path is unknown.

```python
# session/monitor/op_metrics.py (moved from optracing/result.py, with additive changes)
# OperatorMetrics is co-located here (moved from optracing/result.py) and its
# existing `to_dict()` method is unchanged; no edits required.

@dataclass
class OpTraceResult:
    # ---- Required (unchanged, except `model` type relax to allow None) ----
    model: str | None                            # was: str — relaxed to accept None
    device: str
    tracing_level: str                           # "basic" or "detail"

    # ---- Defaulted fields (defaults preserved verbatim from current source) ----
    operators: list[OperatorMetrics] = field(default_factory=list)
    ep: str = ""                                 # default preserved — do NOT remove
    tracing_backend: str = ""                    # default preserved — do NOT remove
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    num_samples: int = 0
    summary: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, dict[str, float]] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    # ---- NEW fields (additive; defaults keep existing construction compatible) ----
    status: str = "ok"                           # "ok" | "no_data" | "parse_failed" | "basic_fallback"
    error: str | None = None                     # populated when status == "parse_failed"

    def to_dict(self) -> dict[str, Any]:
        """Existing nested schema preserved. New `status` / `error` keys added alongside."""
        return {
            "metadata": {
                "model": self.model,
                "device": self.device,
                "ep": self.ep,
                "tracing_level": self.tracing_level,
                "tracing_backend": self.tracing_backend,
                "timestamp": self.timestamp,
                "num_samples": self.num_samples,
            },
            "summary": self.summary,
            "operators": [op.to_dict() for op in self.operators],
            "statistics": self.statistics,        # PRESERVED — consumers depend on this
            "artifacts": self.artifacts,
            # ---- Additive keys ----
            "status": self.status,
            "error": self.error,
        }

    def to_json(self, indent: int = 2) -> str:   # unchanged — preserved as-is
        return json.dumps(self.to_dict(), indent=indent)
```

Report consumers (`display_op_trace_report`, `write_op_trace_json`) continue to accept `OpTraceResult` (not dict). `ctx.monitor.result` exposes it directly. New consumers that care about `status` read it at the top level; existing consumers that read `metadata["model"]`, `summary`, `operators`, `statistics`, `artifacts` are unaffected.

### 4.7 Factory helper in `commands/perf.py`

```python
# commands/perf.py (new helper, ~15 lines)

def _resolve_ep_monitor(
    ep: str,
    op_tracing: str | None,
    output_dir: Path,
) -> WinMLEPMonitor:
    """Pick the WinMLEPMonitor for the requested EP and optional op-tracing level.

    Raises RuntimeError with a descriptive message when op-tracing is requested
    against an EP that has no op-tracing monitor.
    """
    if op_tracing:
        if ep == "qnn" and QNNMonitor.is_available():
            return QNNMonitor(level=op_tracing, output_dir=output_dir)
        raise RuntimeError(
            f"Op-tracing not available for EP '{ep}'. Supported: 'qnn'."
        )
    # Proof-of-execution monitors (no op-tracing)
    if ep == "vitisai" and VitisAIMonitor.is_available():
        return VitisAIMonitor()
    return NullEPMonitor()
```

No registry. No abstract factory. One function, `if/elif` dispatch on two string args. Extension by adding branches.

---

## 5. CLI Integration

### 5.1 Current code (being replaced)

`commands/perf.py:1334-1386` — a separate op-tracing block after benchmark. Invokes `QNNProfiler` with its own iteration count and dummy-input generation.

### 5.2 Replacement code

The op-tracing block is **deleted**. Op-tracing is integrated into the benchmark's existing `session.perf()` call.

```python
# inside the main benchmark loop in commands/perf.py

output_dir = output.parent if output else Path.cwd()

try:
    monitor = _resolve_ep_monitor(ep=config.ep, op_tracing=op_tracing, output_dir=output_dir)
except RuntimeError as e:
    console.print(f"[red]Error:[/red] {e}")
    raise SystemExit(1) from None

with (
    session.perf(warmup=config.warmup, monitor=monitor) as ctx,
    hw_monitor as hw,
):
    _run_monitored_loop(session, inputs, ctx.stats, hw,
                        total_iterations=total_iterations, ...)
    if hw:
        self._hw_metrics = hw.to_dict()

# Post-benchmark: report
if op_tracing:
    result = ctx.monitor.result                     # OpTraceResult (not dict)
    if result is None or result.status == "no_data":
        console.print("[yellow]Warning:[/yellow] No profiling data produced.")
    else:
        display_op_trace_report(result, console)
        json_path = output_dir / f"{model_slug}_op_trace.json"
        write_op_trace_json(result, json_path)
        console.print(f"[green]Op-trace saved to:[/green] {json_path}")
```

Semantic change: op-tracing now observes the user's actual benchmark iterations rather than a separate synthetic profiling pass. Per **US-5**, this is the preferred behavior.

### 5.3 Hard-fail on unsupported op-tracing

If `--op-tracing basic` is requested against `--ep dml` (which has no op-tracing monitor), `_resolve_ep_monitor` raises `RuntimeError` with a descriptive message and the CLI exits with code 1. **No silent fallback** per NFR-2.

---

## 6. Configuration / Data Structures

### 6.1 Session options merge order

Inside `WinMLSession.perf().__enter__`, session config entries are merged into `self._active_session_option_entries` in this order:

1. **User-configured** (via `WinMLSession(...)` ctor, not mentioned in current code): base.
2. **Monitor contribution** via `mon.get_session_options()`: overrides #1.

Applied in `_build_session_options()` via `opts.add_session_config_entry(k, v)` for each (k, v) pair.

### 6.2 Provider options merge order

Inside `WinMLSession.perf().__enter__`, provider options are merged into `self._provider_options` in this order:

1. **User-configured** (via `WinMLSession(..., ep_config=EPConfig(provider_options=...))`): base.
2. **Monitor contribution** via `mon.get_provider_options()`: overrides #1.
3. **Monitor-enforced keys** (inside `QNNMonitor.get_provider_options` after all merges): specifically `profiling_level` and `profiling_file_path` — owned by the monitor, never user-overridable.

Implementation detail: `QNNMonitor.get_provider_options()` builds the dict in layers with explicit key assignment at the end (not a duplicate-key dict literal, which would trigger Ruff `F601`):

```python
opts = {...defaults...}
opts.update(self._extra)
opts["profiling_level"] = ...       # LAST — cannot be overridden via extra
opts["profiling_file_path"] = ...
return opts
```

### 6.3 Restoration on exit

Both `self._active_session_option_entries` and `self._provider_options` are saved at `perf().__enter__` and restored at `perf().__exit__`, regardless of whether the caller body raised an exception.

---

## 7. Error Handling

### 7.1 Exception types

| Exception | Raised when | Caught by |
|-----------|-------------|-----------|
| `WinMLSession.CompilationError` | ORT session creation fails (existing behavior) | `commands/perf.py` CLI wrapper |
| `RuntimeError("QNNMonitor already entered")` | `monitor.__enter__` called twice | propagates; it's a programmer bug |
| `RuntimeError("session.perf() already active ...")` | Nested `perf()` | propagates; programmer bug |
| `RuntimeError("Op-tracing not available for EP '<ep>'")` | `_resolve_ep_monitor` called with unsupported `(ep, op_tracing)` pair | CLI exits with code 1 |

### 7.2 Failure paths

| Failure | Detected where | Behavior |
|---------|-----------------|----------|
| QNN EP not available (neither ORT variant has it) | `QNNMonitor.is_available()` → False | `_resolve_ep_monitor` raises descriptive `RuntimeError`. CLI exits 1. |
| Session compile fails with QNN options | `ort.InferenceSession(...)` raises inside `compile()` | Translated to `CompilationError` per existing `session.py:303-314`. Monitor `__exit__` still runs, sees no CSV, produces `status="no_data"`. |
| CSV missing after teardown | `QNNMonitor._parse_artifacts()` | `OpTraceResult(status="no_data", artifacts={})`. Logged at WARNING. Not an exception. |
| CSV parse error | `_parse_artifacts()` raises | Caught in `__exit__`; produces `OpTraceResult(status="parse_failed", error=msg)`. Logged at WARNING. Does not suppress caller exception. |
| QHAS viewer not found (detail mode) | `run_qhas_viewer()` raises / returns None | Fall back to basic CSV parsing. `OpTraceResult.status = "basic_fallback"`. Logged at WARNING. |
| Auto-reset fires | `perf().__enter__` | `logger.warning("auto-resetting ...")` (per NFR-3). Proceeds normally. |
| `__enter__` twice | `QNNMonitor.__enter__` | `RuntimeError("QNNMonitor already entered")`. |
| Windows CSV file-handle lag | `_parse_artifacts()` on first attempt | Retry once with `time.sleep(0.05)`. If still fails → `status="parse_failed"`. |

### 7.3 Exception transparency (NFR-5)

`perf().__exit__` uses a nested `try/finally` pattern that:

1. Captures `sys.exc_info()` at entry (may be a live caller exception).
2. Performs session teardown in an inner `try/finally`.
3. Calls `monitor.__exit__(*exc_info)` so the monitor knows about any active exception.
4. Never calls `return True` → never suppresses.
5. Restores saved options even if monitor.__exit__ raises.

---

## 8. Testing Strategy

### 8.1 Test file migration

See PRD §10.5 for the full migration table. Summary:

- `tests/unit/optracing/*.py` → migrate files to `tests/unit/session/monitor/` and `tests/unit/commands/`.
- `tests/unit/optracing/fixtures/` (contains `optrace_resnet50.csv`, `qhas_resnet50.json`) → move to `tests/unit/session/monitor/qnn/fixtures/` so parsers' unit tests remain functional.
- Delete: `test_registry.py` (registry is removed), `test_qnn_profiler.py` (class is deleted; replaced by `test_qnn_monitor.py`).
- Redirect test imports: from `winml.modelkit.optracing.*` → `winml.modelkit.session.monitor.*` / `winml.modelkit.session.monitor.qnn.*`.

### 8.2 Unit tests (per class)

| Test | Asserts |
|------|---------|
| `test_ep_monitor_base.py::test_defaults` | `WinMLEPMonitor` subclasses with no overrides get `get_session_options() == {}`, `get_provider_options() == {}`, `requires_session_teardown == False`. |
| `test_ep_monitor_base.py::test_double_entry_guard` | Calling `monitor.__enter__()` twice on a concrete `QNNMonitor` raises `RuntimeError`. |
| `test_qnn_monitor.py::test_is_available_bundled` | Mocked `onnxruntime-qnn` → `is_available()` returns True. |
| `test_qnn_monitor.py::test_is_available_winml` | Mocked WinML registration + `get_ep_devices` → True. |
| `test_qnn_monitor.py::test_is_available_neither` | Both paths miss → False. |
| `test_qnn_monitor.py::test_get_provider_options_idempotent` | Two calls return equal dicts. |
| `test_qnn_monitor.py::test_profiling_keys_not_overridable` | `extra_provider_options={"profiling_level":"off"}` ignored; owner-enforced value wins. |
| `test_qnn_monitor.py::test_exit_no_csv` | `__exit__` with no CSV produces `OpTraceResult.status == "no_data"`. |
| `test_qnn_monitor.py::test_exit_parse_failure` | Corrupt CSV → `status == "parse_failed"`, `error` populated. |
| `test_op_metrics.py::test_to_dict_schema` | `OpTraceResult.to_dict()` has required keys (`ep`, `device`, `operators`, `summary`, `artifacts`, `num_samples`, `status`). |
| `test_ep_registry.py::test_ensure_initialized_idempotent` | Calling twice no-ops; logs on first call only. |
| `test_ep_monitor_base.py::test_set_onnx_op_types_default_no_op` (v2.4) | Calling `WinMLEPMonitor.set_onnx_op_types({"a": "b"})` on a subclass that doesn't override is a no-op — does not raise; nothing visible stored. Pinned by spec acceptance criterion P-1. |
| `test_ep_monitor_base.py::test_result_default_none` (v2.4) | `WinMLEPMonitor.result` returns `None` for any subclass that doesn't set `self._result`; returns the value when set. Pinned by spec acceptance criterion P-2. |
| `test_qnn_monitor.py::test_resolve_op_type_walks_chain` (v2.4) | Given a `QNNMonitor` with hand-built `_onnx_op_types`, parametrise across (L1 hit/miss) × (L2 hit/None) × (L3 hit/None) and assert `_resolve_op_type` returns the right value at each combination. Pinned by spec acceptance criteria P-3 / P-4 / P-5 / P-6. |
| `test_qnn_monitor.py::test_set_onnx_op_types_overrides_default` (v2.4) | `QNNMonitor` overrides the WinMLEPMonitor no-op default; calling `set_onnx_op_types({"a": "Conv"})` updates `monitor._onnx_op_types`. Two successive calls — last value wins. |
| `test_qnn_monitor.py::test_heuristic_empty_string_treated_as_miss` (v2.4) | `_heuristic_op_type` returning `""` falls through to L4 (raw `op_path`) inside `_resolve_op_type`. |
| `test_qnn_monitor.py::test_parse_basic_uses_onnx_lookup` (v2.4) | Inject `{"_qnn_event": "Conv"}`; parsing a CSV fixture with that event yields `OperatorMetrics(name="Conv")`. |
| `test_qnn_monitor.py::test_parse_basic_falls_back_to_heuristic` (v2.4) | Inject `{}`; parsing a CSV fixture with `/encoder/conv1/Conv_token_1_2` yields `OperatorMetrics(name="Conv", op_path="/encoder/conv1/Conv")` via the leaf-split heuristic on the cleaned form. |
| `test_qnn_monitor.py::test_parse_detail_falls_back_to_qhas` (v2.4) | Inject `{}`; parsing a QHAS fixture where `qnn_op_type="ElementWiseAdd"` yields `OperatorMetrics(name="ElementWiseAdd")` via L2. |
| `test_qnn_monitor.py::test_parse_detail_onnx_wins_over_qhas` (v2.4) | Inject a map with an entry for the op_path; parsing a QHAS fixture where the same node has a different `qnn_op_type` yields the ONNX value. |
| `test_qnn_monitor.py::test_heuristic_strips_token_suffix` (v2.4) | `monitor._heuristic_op_type("/encoder/conv1/Conv_token_1_2")` returns `"Conv"`. |
| `test_qnn_monitor.py::test_constructor_accepts_onnx_op_types` (v2.4) | `QNNMonitor(level="basic", onnx_op_types={"a": "Conv"})._onnx_op_types == {"a": "Conv"}`. |
| `test_qnn_monitor.py::test_parse_existing_artifacts_classmethod` (v2.4) | `QNNMonitor.parse_existing_artifacts(level="basic", artifacts={"csv": <path>}, onnx_op_types={...})` returns a fully-populated `OpTraceResult`. |
| `test_session.py::test_build_op_type_map_resnet50` (v2.3+) | `WinMLSession._build_op_type_map(<resnet50.onnx>)` returns a non-empty dict whose keys include known node names. |
| `test_session.py::test_build_op_type_map_handles_failures` (v2.3+) | `_build_op_type_map(None)`, `_build_op_type_map(Path("/does/not/exist"))`, and `_build_op_type_map(<corrupt.onnx>)` all return `{}` without raising. |

### 8.3 Integration tests

| Test | Asserts |
|------|---------|
| `test_perf_monitor_integration.py::test_teardown_ordering` | With a `FakeMonitor(requires_session_teardown=True)`, during `monitor.__exit__`, `session._session is None`. |
| `test_perf_monitor_integration.py::test_null_monitor_no_reset` | `session.perf()` with no monitor does NOT reset a compiled session. |
| `test_perf_auto_reset.py::test_auto_reset_fires_on_option_diff` | With a compiled session and a monitor that contributes options, `__enter__` logs WARNING and `session._session` becomes None. |
| `test_perf_auto_reset.py::test_auto_reset_restores_on_exit` | After `perf()` exit, `self._provider_options` is restored to pre-entry state. |
| `test_perf_monitor_integration.py::test_exception_transparency` | Caller exception in `with` body propagates; `monitor.__exit__` called with correct `exc_info`. |
| `test_perf_monitor_integration.py::test_nested_perf_forbidden` | Second `session.perf()` inside first raises `RuntimeError`. |
| `test_perf_monitor_integration.py::test_onnx_map_injected_unconditionally` (v2.4) | `session.perf().__enter__` calls `monitor.set_onnx_op_types(non_empty_map)` BEFORE `monitor.__enter__()` on EVERY monitor (verified via spy on QNNMonitor, VitisAIMonitor, and NullEPMonitor mocks). The non-op-tracing monitors silently absorb the call via the WinMLEPMonitor no-op default. |
| `test_perf_monitor_integration.py::test_no_onnx_path_skips_injection` (v2.4) | When `WinMLSession` was constructed without an ONNX path, no `set_onnx_op_types` call is made. |
| `test_perf_monitor_integration.py::test_qnn_monitor_resolves_via_onnx` (v2.3+, hardware-gated) | Real ONNX (resnet50) + real CSV fixtures: ONNX-resolved `name` for nodes that exist in the graph; falls back correctly when nodes don't. |
| `test_perf_json_dispatch.py::test_qnn_payload_routes_to_op_trace_key` (v2.4) | After a QNN run, `commands/perf.py` JSON output has an `op_trace` key sourced from `monitor.result.to_dict()` and NO `ep_proof` key for that monitor. |
| `test_perf_json_dispatch.py::test_vitisai_payload_routes_to_ep_proof_key` (v2.4) | After a VitisAI run, `commands/perf.py` JSON output has an `ep_proof` key (transitional `monitor.to_dict()` until typed `proof` accessor follow-up) and NO `op_trace` key. |
| `test_perf_json_dispatch.py::test_null_monitor_contributes_no_key` (v2.4) | After a benchmark with `NullEPMonitor`, `commands/perf.py` JSON output has neither `op_trace` nor `ep_proof` keys. |

### 8.3.1 Architecture regression test (v2.3)

Pinned by PRD NFR-8 / SC-8. Mechanism per spec §10.1 Q2 (recommendation: AST scan). v2.4 narrows the test scope: there is no `op_trace_parser.py` module to scan against — only the QNN private internals (`qnn/_internal.py`) need to be import-fenced.

| Test | Asserts |
|------|---------|
| `test_architecture_imports.py::test_no_external_imports_of_qnn_internals` (v2.3+) | A Python AST scan over `src/winml/modelkit/` (excluding `session/monitor/qnn/`) and `tests/` asserts no `import` or `from` statement references `winml.modelkit.session.monitor.qnn.csv_parser`, `winml.modelkit.session.monitor.qnn.qhas_parser`, or `winml.modelkit.session.monitor.qnn._internal`. The single permitted importer of `qnn/_internal.py` (option b) is `qnn_monitor.py`. The test fails any future commit that re-exposes a private QNN parsing helper. |

### 8.4 CLI / E2E tests (hardware-gated)

- `test_perf_optracing.py::test_cli_op_tracing_basic_on_qnn` (skip if no QNN NPU): runs `wmk perf -m resnet50 --device npu --op-tracing basic`, asserts CSV produced, `*_op_trace.json` written, at least one operator entry.
- `test_perf_optracing.py::test_cli_op_tracing_unsupported_ep` (no hardware needed): `--ep dml --op-tracing basic` exits with code 1 and descriptive message.

---

## 9. Integration Points

### 9.1 Downstream consumers

- `commands/perf.py` — uses `_resolve_ep_monitor` + `session.perf(monitor=...)`.
- `display_op_trace_report(result: OpTraceResult, console)` — unchanged; consumes `ctx.monitor.result`.
- `write_op_trace_json(result: OpTraceResult, path)` — unchanged; consumes `ctx.monitor.result`.

### 9.2 Upstream dependencies

- `session/ep_registry.py::ensure_initialized()` — **new module-level function** added to existing `ep_registry.py`; wraps `WinMLEPRegistry.get_instance().register_to_ort()` behind an idempotent single-call entry point. Called by `QNNMonitor.is_available()` AND by `WinMLSession.__init__`. Replaces the existing classmethod `WinMLSession._init_winml_eps_once`, which is deleted.
- `WinMLSession.reset()` — called by `perf().__exit__` for `requires_session_teardown` monitors.
- Import redirect in `commands/perf.py`: `from ..optracing import display_op_trace_report, write_op_trace_json, OpTraceResult` → `from ..session.monitor.report import display_op_trace_report, write_op_trace_json` and `from ..session.monitor.op_metrics import OpTraceResult`. Remove imports of `is_qnn_profiling_available` and `get_tracer` (both deleted).

### 9.3 How future EP monitors plug in

Adding DMLMonitor (hypothetical):

1. Create `session/monitor/dml_monitor.py` with `class DMLMonitor(WinMLEPMonitor)`.
2. Override `is_available()` to check for DML EP.
3. Override `get_provider_options()` if DML profiling requires config (e.g., `"dml_profiling_enabled": "1"`).
4. Override `requires_session_teardown` if DML's profiling data flush needs it.
5. Add one `elif` branch in `commands/perf.py::_resolve_ep_monitor`.

No registry changes. No cross-file wiring.

---

## 10. Future Work

- **FW-1** Extract the auto-reset behavior to a reusable policy object when a second monitor type needs different reset semantics (YAGNI now).
- **FW-2** Investigate QNN SDK's support for absolute `*_schematic.bin` paths; if supported, eliminate the glob-fallback path in `_parse_artifacts` (OQ-1 in PRD).
- **FW-3** Multi-monitor support (`monitors=[...]`). Requires redesigning teardown ordering (see architect review). Out of scope per C-4.
- **FW-4** Schema versioning on `OpTraceResult.to_dict()` output. Consider if report writers need forward compatibility (OQ-2 in PRD).

---

## 11. Revision History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-04-17 | Initial `2_coreloop.md`. Captured architecture from iterations 01-11. |
| 2.0 | 2026-04-19 | Restructured per `docs/standards/design-doc-spec.md` v1.0. Added metadata header, §0 Related Documents, §0.5 I/O Dependencies. Applied user directives: dual `get_session_options` + `get_provider_options` hooks; preserve `OpTraceResult.to_dict()` (not plain dict); `os.chdir` removed (use absolute paths + glob fallback); `generate_dummy_inputs` removed entirely; singular `monitor=`; factory dispatch replaces registry. Applied critic/architect findings: no duplicate dict keys (explicit pop-then-set); add `ep_registry.ensure_initialized` to remove reverse coupling; auto-reset at WARNING (not INFO); `gc.collect` + retry for Windows file-handle lag; exception transparency via `sys.exc_info()` capture; load-bearing teardown ordering made explicit with integration test. |
| 2.1 | 2026-04-19 | Post-audit fixes: added Table of Contents; corrected §4.6 to acknowledge `OpTraceResult.to_dict()` already exists at `optracing/result.py:79-95` and the refactor preserves its nested schema (adds `status`/`error` as additive top-level keys, keeps `metadata`/`summary`/`operators`/`statistics`/`artifacts`); clarified in §0.5.1 and §4.3 that `ensure_initialized()` is a NEW function added to the existing `ep_registry.py`; added `fixtures/` migration to §8.1; documented `commands/perf.py` import-path redirects in §9.2. |
| 2.2 | 2026-04-24 | Relocated from docs/design/optracing/ to docs/design/session/monitor/ per spec §1.5.1 transitional commitment (implementation complete). Removed Transitional Location note. |
| 2.3 | 2026-05-06 | Reflect new op-trace-parser interface spec (`docs/design/perf/2026-05-03-op-trace-parser-interface-spec.md` v1.2): introduce `OpTraceParser` ABC at `session/monitor/op_trace_parser.py` (new §4.1.5); `QNNMonitor` implements both `WinMLEPMonitor` and `OpTraceParser` via multiple inheritance (revised §4.3 — class signature, `parse_basic` / `parse_detail` / `supported_levels` / `_heuristic_op_type` implementations); `WinMLSession` builds ONNX `node.name → node.op_type` map via new `_build_op_type_map` static helper and injects it via `set_onnx_op_types` (revised §4.5; revised §3.1 high-level flow; revised §0.5.2 data dependency graph); delete `qnn/csv_parser.py` and `qnn/qhas_parser.py` as public modules — helpers fold into private `qnn/_internal.py` (option b, recommended) or `qnn_monitor.py` directly (option a) (revised §2.1 file layout, new §2.1.1 viewer.py status note); fallback chain ONNX → EP-authoritative → heuristic → raw event ID (documented in §4.1.5 invariants); naming convention (verbatim ONNX `node.op_type`, no translation tables) noted in §4.3. New tests added to §8.2 (parser ABC + QNNMonitor parsing methods + `_build_op_type_map` helper) and new §8.3.1 architecture regression test. Open questions in spec §10.1 (option a/b for helper placement, architecture-test mechanism, `_build_op_type_map` placement) noted as pending; flow back into this doc when resolved. |
| 2.4 | 2026-05-08 | Major design simplification, reflecting spec v2.0. **Drop `OpTraceParser` ABC entirely** — premature abstraction for a single concrete implementer; multiple inheritance + MRO + a separate ABC file are too much complexity for one EP. Wait for the second op-tracing EP before extracting an abstraction. Replacement: extend `WinMLEPMonitor` ABC (§4.1) with two concrete-default members — `set_onnx_op_types(map)` (no-op default) and `result` property (returns `getattr(self, "_result", None)`). §4.1.5 (OpTraceParser ABC) DELETED. **Drop `to_dict()` from WinMLEPMonitor ABC contract** — god-method conflating op-tracing telemetry with proof-of-execution; concrete monitors expose typed accessors instead (`result` for op-tracing, `proof` for proof-of-execution — typed `ProofOfExecution` class flagged as follow-up PR, OQ-6 in PRD, out of scope). §4.3 QNNMonitor revised: single inheritance (`class QNNMonitor(WinMLEPMonitor)`); private `_resolve_op_type`, `_heuristic_op_type`, `_parse_basic`, `_parse_detail` methods; new `parse_existing_artifacts` classmethod; `to_dict` removed. §4.5 WinMLSession.perf revised: ONNX-map injection now UNCONDITIONAL (no isinstance check) — WinMLEPMonitor no-op default makes the call safe for non-op-tracing monitors. §3.1 high-level flow revised. §0.5.1 / §0.5.2 / §0.5.3 / §2.1 / §2.2 updated. §4.2 NullEPMonitor: `to_dict` removed (returned `{}`); `result`/`proof` both inherit None default. §4.X VitisAI / OpenVINO: `to_dict` kept transitionally pending the typed `proof` accessor follow-up. §8 Testing Strategy revised — `test_op_trace_parser.py` removed; new tests added for WinMLEPMonitor concrete defaults and isinstance-based JSON dispatch. §8.3.1 architecture regression test scope narrowed (no `op_trace_parser.py` to fence). |
| 2.4.1 | 2026-05-08 | Doc-review fixes: §0.5.2 ASCII diagram updated (typed accessor); §3.2 NullEPMonitor walkthrough updated; §3.4 standalone-profile example updated; §4.3 _heuristic_op_type pseudocode strip-safety restored. |
