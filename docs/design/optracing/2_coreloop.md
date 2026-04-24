# Op-Tracing Refactor — Core Loop Design

**Version**: 2.1
**Date**: 2026-04-19
**Status**: Draft
**Module**: session/monitor
**Supersedes**: `docs/design/optracing/2_coreloop.md` v1.0 (consolidated per `docs/standards/design-doc-spec.md`)
**Depends-On**: `docs/design/optracing/1_prd.md`, `docs/standards/design-doc-spec.md`

**Transitional Location** (per `docs/standards/design-doc-spec.md` §1.5.1):
- Current doc directory: `docs/design/optracing/` (legacy feature name)
- Target `Module`: `session/monitor` (post-refactor)
- Relocation commitment: when the implementation PR that deletes `src/winml/modelkit/optracing/` lands, these docs MUST be moved under `docs/design/session/monitor/` in that same PR.

---

## Table of Contents

- [0. Related Documents](#0-related-documents)
- [0.5 I/O Dependencies](#05-io-dependencies)
- [1. Design Philosophy](#1-design-philosophy)
- [2. Module Structure](#2-module-structure)
- [3. Core Loop Implementation](#3-core-loop-implementation)
- [4. API Design](#4-api-design)
  - [4.1 EPMonitor — revised ABC](#41-epmonitor--revised-abc)
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
| Upstream | `../../../src/winml/modelkit/session/monitor/ep_monitor.py` | `EPMonitor` ABC — existing code extended here |
| Upstream | `../../../src/winml/modelkit/commands/perf.py` | CLI benchmark entry point — existing code modified here |
| Deleted | `../../../src/winml/modelkit/optracing/qnn/profiler.py` | `QNNProfiler` — deleted by this refactor |

## 0.5 I/O Dependencies

This refactor orchestrates four subsystems. Data dependencies MUST be understood before reading the core loop.

### 0.5.1 Key actors

| Actor | Role | Location |
|-------|------|----------|
| `WinMLSession` | Owns `ort.InferenceSession` lifecycle; exposes `perf()` | `session/session.py` |
| `EPMonitor` (ABC) | Per-EP observer with two optional config hooks | `session/monitor/ep_monitor.py` |
| `QNNMonitor` | Concrete EPMonitor for Qualcomm NPU | `session/monitor/qnn_monitor.py` |
| `PerfContext` | Dataclass yielded by `session.perf()` | `session/session.py` (new) |
| `OpTraceResult` | Structured profiling output | `session/monitor/op_metrics.py` (relocated) |
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
│   ctx.monitor.to_dict()  → delegates to OpTraceResult.to_dict()        │
└───────────────────────────────────────────────────────────────────────┘
```

### 0.5.3 Module responsibility summary

- **`WinMLSession`**: template-method owner. Merges monitor hook contributions at compile; creates `ort.InferenceSession`; runs inference; handles teardown ordering at `perf().__exit__`.
- **`EPMonitor` (ABC)**: contract definition. Two optional config hooks with base-class defaults. `is_available` classmethod. Mandatory `__enter__`/`__exit__`/`to_dict`.
- **`QNNMonitor`**: concrete implementation. Declares session+provider options; parses CSV/QHAS in `__exit__`; produces `OpTraceResult`.
- **`commands/perf.py`**: CLI dispatcher. Resolves the right monitor class by explicit `if/elif` on `--ep` and `--op-tracing` flags; constructs it with the appropriate `level` and `output_dir`.
- **`session/ep_registry.py`**: module-level `ensure_initialized()`. Single shared entry point for WinML EP registration. Eliminates the reverse-coupling `QNNMonitor → WinMLSession._init_winml_eps_once`.

---

## 1. Design Philosophy

### 1.1 Purpose

Collapse the dual per-EP hierarchy (`EPMonitor` + `OpTracer`) into one; fix the broken `onnxruntime-windowsml` session-creation path; eliminate code duplication by routing all ORT session construction through `WinMLSession`.

### 1.2 Core Principles

- **P1 — Session owns the session; monitor informs the session.** `WinMLSession.compile()` is the sole owner of `ort.InferenceSession` construction. Monitors contribute configuration via two hooks but never create ORT sessions directly.
- **P2 — Delete > refactor.** Where two abstractions exist for the same concept, delete one. `QNNProfiler` and `OpTracer` are deleted rather than patched.
- **P3 — Good primitives > bespoke facades.** A clean pair (`WinMLSession`, `QNNMonitor`) composes cleanly into any caller shape. We do not add helper classes or wrapper utilities.
- **P4 — Extension by hook, not by new abstraction.** New EP monitors are added by subclassing `EPMonitor` and overriding the two hooks. No registry, no factory, no plugin loader.
- **P5 — Explicit over implicit.** No silent fallbacks. No silent session mutations (auto-reset logs at `WARNING`). No silent "ep unsupported" errors (hard-fail at dispatch time).

### 1.3 Design Pattern

**Hook-based Plugin + Template Method + Observer.**

- `WinMLSession.compile()` is the template method: it owns the algorithm (resolve device → build session options → find EP device → merge provider options → create ORT session).
- It calls the monitor at two hook points: `get_session_options()` (add_session_config_entry contributions) and `get_provider_options()` (add_provider_for_devices contributions).
- The `EPMonitor` itself is a context-managed observer: `__enter__` prepares for observation, `__exit__` finalizes.
- The monitor never replaces session behavior — only augments specific steps.

---

## 2. Module Structure

### 2.1 File layout after refactor

```
src/winml/modelkit/
├── session/
│   ├── session.py                           # modified (see §4.5)
│   ├── ep_registry.py                       # modified — existing file; adds ensure_initialized() (see §4.3)
│   └── monitor/
│       ├── ep_monitor.py                    # modified (see §4.1)
│       ├── hw_monitor.py                    # unchanged
│       ├── qnn_monitor.py                   # REWRITTEN (see §4.3)
│       ├── vitisai_monitor.py               # unchanged
│       ├── openvino_monitor.py              # unchanged (inherits new defaults)
│       ├── op_metrics.py                    # NEW — moved from optracing/result.py + .to_dict()
│       ├── report.py                        # NEW — moved from optracing/report.py
│       └── qnn/
│           ├── csv_parser.py                # moved from optracing/qnn/
│           ├── qhas_parser.py               # moved from optracing/qnn/
│           └── viewer.py                    # moved from optracing/qnn/
├── commands/
│   └── perf.py                              # modified (see §5)
└── optracing/                               # DELETED ENTIRELY
```

### 2.2 Key dependencies

- `WinMLSession.compile()` calls `mon.get_session_options()` and `mon.get_provider_options()` on the active monitor.
- `QNNMonitor.is_available()` calls `session/ep_registry.py::ensure_initialized()` (NOT `WinMLSession._init_winml_eps_once`, which is deleted).
- `QNNMonitor.__exit__` reads the CSV written by QNN EP during `session.run()` and produces an `OpTraceResult`.
- `commands/perf.py` imports `QNNMonitor` and `VitisAIMonitor` directly; no registry.

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
- `ctx.monitor.to_dict()` → `{}`, `ctx.monitor.result` is `None`

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
print(ctx.monitor.to_dict())
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

### 4.1 `EPMonitor` — revised ABC

```python
# session/monitor/ep_monitor.py

class EPMonitor(ABC):
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

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary. MUST include `ep` key."""
```

Invariants:

- `get_session_options()` and `get_provider_options()` MUST be idempotent (NFR-4).
- `__enter__` MUST raise `RuntimeError("<Monitor> already entered")` if called twice without intervening `__exit__`.
- `__exit__` MUST NOT return `True` (which would suppress exceptions).

### 4.2 `NullEPMonitor`

Unchanged from current `ep_monitor.py:62-88`. Inherits new default `get_session_options()` / `get_provider_options()` (both return `{}`) and `requires_session_teardown = False`. No edit needed; behavior automatic.

### 4.3 `QNNMonitor`

```python
# session/monitor/qnn_monitor.py

class QNNMonitor(EPMonitor):
    """Qualcomm NPU per-op profiler via ORT's QNN EP.

    Produces an OpTraceResult with per-operator cycle counts (level="basic")
    or full QHAS roofline / DMA traffic (level="detail").
    """

    requires_session_teardown: ClassVar[bool] = True
    # QNN EP flushes the profiling CSV only on ort.InferenceSession destruction.

    def __init__(
        self,
        level: Literal["basic", "detail"] = "basic",
        output_dir: Path | None = None,
        extra_provider_options: Mapping[str, str] | None = None,
    ) -> None:
        if level not in ("basic", "detail"):
            raise ValueError(f"level must be 'basic' or 'detail', got {level!r}")
        self._level = level
        # Idempotency: paths produced at __init__, not per-call
        self._output_dir = Path(output_dir) if output_dir else Path(
            tempfile.mkdtemp(prefix="qnn_profile_")
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = (self._output_dir / "profiling_output.csv").resolve()
        self._extra = dict(extra_provider_options or {})
        self._entered = False
        self._result: OpTraceResult | None = None

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

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("QNNMonitor already entered")
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Parse whatever artifacts are on disk. Never suppress caller exceptions.
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
        # Do not return True → does not suppress caller exception.

    def to_dict(self) -> dict[str, Any]:
        if self._result is None:
            return {"ep": "QNN", "device": "NPU", "status": "not_run"}
        return self._result.to_dict()

    @property
    def result(self) -> OpTraceResult | None:
        """Structured result object. Preferred by report writers."""
        return self._result

    def _parse_artifacts(self) -> OpTraceResult:
        """Parse CSV (always) and QHAS (detail mode)."""
        # ... details: try CSV → fall back to no-data; if detail, try QHAS viewer
        # On Windows file-handle lag: retry once with 50ms delay (R-2 mitigation)
        ...
```

**On CWD / `*_schematic.bin`**: Per **C-5** and **FR-12**, `QNNMonitor` does NOT call `os.chdir`. If the QNN SDK emits `*_schematic.bin` to the process's CWD rather than to `profiling_file_path`'s directory, `_parse_artifacts` locates it via `glob` from the expected fallback locations and logs a `WARNING` if not found. The `detail`-mode path degrades gracefully to basic CSV parsing in that case (FR-5).

### 4.4 `PerfContext`

```python
# session/session.py

@dataclass(frozen=True)
class PerfContext:
    """Yielded by WinMLSession.perf(). Aggregates perf stats and the attached EP monitor."""
    stats: PerfStats
    monitor: EPMonitor        # NullEPMonitor when caller passed monitor=None
```

Frozen to prevent accidental mutation during the `with` block. Not a replacement for `PerfStats` — both `stats` and `monitor` are addressable by attribute.

### 4.5 `WinMLSession.perf()` — revised

```python
# session/session.py

@contextmanager
def perf(
    self,
    warmup: int = 0,
    monitor: EPMonitor | None = None,
) -> Generator[PerfContext, None, None]:
    """Run a scoped performance window.

    Yields:
        PerfContext with `stats: PerfStats` and `monitor: EPMonitor`.

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
) -> EPMonitor:
    """Pick the EPMonitor for the requested EP and optional op-tracing level.

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
| `test_ep_monitor_base.py::test_defaults` | `EPMonitor` subclasses with no overrides get `get_session_options() == {}`, `get_provider_options() == {}`, `requires_session_teardown == False`. |
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

### 8.3 Integration tests

| Test | Asserts |
|------|---------|
| `test_perf_monitor_integration.py::test_teardown_ordering` | With a `FakeMonitor(requires_session_teardown=True)`, during `monitor.__exit__`, `session._session is None`. |
| `test_perf_monitor_integration.py::test_null_monitor_no_reset` | `session.perf()` with no monitor does NOT reset a compiled session. |
| `test_perf_auto_reset.py::test_auto_reset_fires_on_option_diff` | With a compiled session and a monitor that contributes options, `__enter__` logs WARNING and `session._session` becomes None. |
| `test_perf_auto_reset.py::test_auto_reset_restores_on_exit` | After `perf()` exit, `self._provider_options` is restored to pre-entry state. |
| `test_perf_monitor_integration.py::test_exception_transparency` | Caller exception in `with` body propagates; `monitor.__exit__` called with correct `exc_info`. |
| `test_perf_monitor_integration.py::test_nested_perf_forbidden` | Second `session.perf()` inside first raises `RuntimeError`. |

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

1. Create `session/monitor/dml_monitor.py` with `class DMLMonitor(EPMonitor)`.
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
