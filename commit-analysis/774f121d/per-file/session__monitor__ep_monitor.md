# src/winml/modelkit/session/monitor/ep_monitor.py

## TL;DR
This is the keystone ABC of the op-tracing refactor: the single-class replacement for the deleted `OpTracer` ABC plus the v2.3-era extended `EPMonitor`. Renamed to `WinMLEPMonitor`. Mandatory contract shrinks to three members (`__enter__`, `__exit__`, `is_available`); five optional concrete-default extensions are added (`requires_session_teardown` ClassVar, `ep_name` ClassVar, `get_session_options`, `get_provider_options`, `set_onnx_op_types`, `result` property) plus an `__init_subclass__` guard. The class no longer owns an inference loop — it is a passive observer the session drives.

## Diff metrics
- Lines added: ~95
- Lines removed: ~9
- File size: 87 → 172 lines (effectively a rewrite of the surface area; `__enter__` / `__exit__` / `is_available` and `NullEPMonitor.__enter__/__exit__/is_available` are the only members carried verbatim)

## Role before vs after

**Before (a509a67-era `EPMonitor`).** Lifecycle ABC for hardware monitoring. Four abstract methods (`__enter__`, `__exit__`, `to_dict`, `is_available`) and a `NullEPMonitor` Null-Object. Monitors exposed all their telemetry through a polymorphic `to_dict()`. There was a parallel hierarchy in `optracing/base.py::OpTracer` with a different shape: `run(iterations, warmup) -> OpTraceResult` plus `is_available()`. Op-tracing owned the inference loop; hardware monitoring was an outer context manager. The two hierarchies never met.

**After (this commit).** `WinMLEPMonitor` is the single per-EP abstraction. Control is inverted: the monitor no longer drives anything — `WinMLSession.perf()` enters/exits it around the session's own run loop, calls option hooks during compile, and injects the ONNX op-type map before `__enter__`. The ABC carries five extension points beyond the lifecycle triplet; QNN's op-tracing data flows through the typed `result` property instead of the dropped `to_dict()`. The companion `OpTracer` ABC in `optracing/base.py` is deleted (the 35-line ABC declared `run` + `is_available` as abstracts; both responsibilities are absorbed — `run` by `WinMLSession.run`, `is_available` by this ABC).

## Symbol-level changes

### `WinMLEPMonitor` (renamed from `EPMonitor`)
- **`requires_session_teardown: ClassVar[bool] = False`** *(new)*. ORT-specific hint that the monitor's data flush requires `ort.InferenceSession` destruction. Read at `session.py:732` by `WinMLSession.perf().__exit__`: if `True`, `self.reset()` fires BEFORE `monitor.__exit__` so the freshly-flushed CSV is on disk when QNN parses it. QNNMonitor sets it; all other monitors inherit the default `False`. Documented in PRD as constraint C-2 / C-6 and called out as "the only place on the base ABC where an ORT implementation detail leaks in" — pragmatic tradeoff vs. an over-abstracted `prepare_for_exit` callback.
- **`ep_name: ClassVar[str | None] = None`** *(new)*. Target EP short name. When set, `WinMLSession.perf()` (lines 647-656) validates that the monitor's intended EP matches the session's bound EP; mismatch raises `WinMLEPMonitorMismatch`. Without this guard, sessions falling through to ORT's policy-based EP selection would silently drop the monitor's provider options (the exact failure mode the field exists to prevent). QNNMonitor sets `"qnn"`; the proof-of-execution monitors and NullEPMonitor inherit `None`, which means "any EP fine".
- **`__init_subclass__(cls, **kwargs)`** *(new)*. Class-definition-time guard that rejects subclasses shadowing `requires_session_teardown` with a non-bool. Catches typos at import time rather than letting them silently corrupt the C-2 invariant. Docstring is honest: instance-level shadow (`self.requires_session_teardown = "yes"` in `__init__`) is NOT catchable here.
- **`get_session_options() -> dict[str, str]`** *(new, concrete default `{}`)*. Entries to pass to `SessionOptions.add_session_config_entry()`. Consumed by `_build_session_options` at `session.py:184`. QNNMonitor returns `{"ep.context_enable": "1", "ep.context_embed_mode": "0"}`; everyone else inherits the empty default.
- **`get_provider_options() -> dict[str, str]`** *(new, concrete default `{}`)*. Options merged into `add_provider_for_devices([ep], opts)`. Consumed by `_build_provider_options` at `session.py:124`. QNNMonitor returns `profiling_level` + `profiling_file_path` plus pass-through extras; everyone else inherits the empty default.
- **`set_onnx_op_types(onnx_op_types: dict[str, str]) -> None`** *(new, no-op default, `# noqa: B027`)*. Injection point for the L1 lookup table used in `QNNMonitor._resolve_op_type`. Called unconditionally by `WinMLSession.perf()` at line 677, BEFORE `__enter__`. Designed for symmetric dispatch — no isinstance check at the call site; the no-op default makes the call safe for monitors that don't care.
- **`result -> OpTraceResult | None`** *(new property)*. Default returns `getattr(self, "_result", None)`. Op-tracing monitors set `self._result` during `__exit__`. QNNMonitor actually overrides this with an identical implementation that returns `self._result` directly — the `getattr` fallback is what makes the default work for monitors that never touch `self._result`.
- **`__enter__`, `__exit__`, `is_available`** — unchanged signatures, still abstract. The mandatory triplet.
- **REMOVED: `to_dict()` abstract method.** Was the v2.3-era polymorphic data accessor. PRD §1.1 calls it a "god-method that conflated op-tracing telemetry (QNN) with proof-of-execution signals (VitisAI/OpenVINO) under one interface." Replaced by typed accessors (`result` here; transitional `to_dict()` survives on the proof-of-execution monitors as concrete methods, NOT on the ABC).

### `NullEPMonitor`
- Carried forward verbatim except its `to_dict()` override is **REMOVED** (it returned `{}`). The default inherited `result -> None` is "the honest answer" per the v2.4 design history.
- `is_available()` still returns `True` (it does nothing — it's always available).

### Imports
- `ClassVar` added to runtime `typing` import.
- `Self` migrated from `typing_extensions` to `typing` (Py 3.11+ minimum implied).
- `OpTraceResult` added to the `TYPE_CHECKING` block as a forward reference for the `result` property annotation.

### Docstring
- Top-level module docstring gains a "v2.4 additions" block pointing at the parser-interface spec and coreloop §4.1.
- Class docstring example rewritten from the nested `with session.perf(...) as stats: with SomeEPMonitor() as hw: ...` pattern to the single `with session.perf(warmup=10, monitor=SomeEPMonitor()) as ctx:` form yielding a `PerfContext(stats, monitor)`.

## Behavior / contract changes

- **Mandatory contract shrinks** from {`__enter__`, `__exit__`, `to_dict`, `is_available`} to {`__enter__`, `__exit__`, `is_available`}. Subclasses no longer have to ship a `to_dict()`.
- **Control inversion vs. `OpTracer`**: the deleted `OpTracer.run(iterations, warmup)` owned the inference loop. `WinMLEPMonitor` does NOT. The session calls `session.run()` between `__enter__` and `__exit__`. Monitors observe, they don't drive. This is the single biggest semantic change of the refactor.
- **Five new optional extension points** layered on the base class via concrete defaults: two ClassVars + three method/property hooks. Subclasses opt-in by overriding.
- **Class-definition-time validation** via `__init_subclass__` for the `requires_session_teardown` ClassVar — non-bool subclass shadow is rejected at import time with `TypeError`.
- **Data accessor is typed**: `result -> OpTraceResult | None` replaces the polymorphic `to_dict() -> dict[str, Any]`. Type checkers can now distinguish op-tracing from non-op-tracing monitors at the call site.

## Cross-file impact

- **Consumers of the new hooks** (all in `session.py` post-refactor):
  - Line 124: `ep_monitor.get_provider_options()` merged into provider opts.
  - Line 184: `ep_monitor.get_session_options()` merged into session config entries.
  - Line 649-655: `monitor.ep_name` validated against `self._ep` → `WinMLEPMonitorMismatch`.
  - Line 677: `effective_monitor.set_onnx_op_types(self._build_op_type_map(self._onnx_path))`.
  - Line 732: `getattr(effective_monitor, "requires_session_teardown", False)` governs C-2 teardown order.
- **Subclasses in-tree** (4 total): `QNNMonitor` (real op-tracer), `VitisAIMonitor` (proof-of-execution via xrt-smi), `OpenVINOMonitor` (placeholder stub, `is_available()` returns `False` literally), `NullEPMonitor` (no-op fallback). `HWMonitor` is deliberately NOT a subclass (PRD FR-9: "HWMonitor and WinMLEPMonitor are independent context managers; they MAY be combined by the caller").
- **commands/perf.py**: consumes `ctx.monitor.result` (typed) and isinstance-dispatches for the transitional `to_dict()` on proof-of-execution monitors.
- **Deletes**: `src/winml/modelkit/optracing/base.py::OpTracer` ABC (`run` + `is_available` — both abstract). Its job is absorbed: `run` by the externalized `session.run()` loop, `is_available` by this ABC. Net abstraction count: -1 ABC.

## Risks / subtleties

- **`requires_session_teardown` is read via `getattr(...., False)` at `session.py:732`**, not via direct attribute access. That's redundant given the `ClassVar` default already exists on the ABC — but it shields against future subclass deletion of the attribute. Harmless but slightly noisy.
- **The `__init_subclass__` guard is narrow**: only catches class-level shadow with a non-bool literal. It cannot catch `self.requires_session_teardown = ...` in `__init__`, nor a class-level bool that's the wrong value (e.g. a subclass sets `False` when it should be `True`). The PRD note in `__init_subclass__`'s docstring is explicit about this.
- **`ep_name` validation is one-way**: `session.py:647-656` validates the monitor's `ep_name` against `self._ep` BUT only when both are non-None. A QNNMonitor attached to a session whose `_ep` was resolved to "qnn" via policy fallback (rather than an explicit `--ep`) would slip through. Acceptable because the WinMLSession constructor now binds `_ep` from `ep_device.device.ep_name` directly (`session.py:231`), so the ambiguity is gone in practice.
- **`set_onnx_op_types` lacks `@abstractmethod`** by design — it's a no-op default so the session can dispatch unconditionally. The cost is that a forgetful op-tracing subclass that fails to override silently degrades to "L1 always misses" (op-type chain falls through to L2/L3/L4). No test enforces the override. NFR-7 in the PRD lists a `test_set_onnx_op_types_default_is_no_op` for the negative direction (default is a no-op); there's no positive test that op-tracing monitors actually store the map.
- **`result` uses `getattr(self, "_result", None)`** — relies on the convention that op-tracing subclasses set `self._result` in `__exit__`. A forgetful subclass simply yields `None` silently. There's no init-time `self._result: OpTraceResult | None = None` on the base class that would let mypy catch the missing assignment.
- **QNNMonitor re-implements `result`** as `@property def result(self) -> OpTraceResult | None: return self._result` (qnn_monitor.py:294-297). Identical behavior to the inherited default. This is dead-equivalent code — kept presumably for explicitness / docstring placement.
- **`HWMonitor` is intentionally outside the hierarchy**, but its lifecycle is also `__enter__`/`__exit__`. The PRD justifies this on FR-9 ("orthogonal"), but it does mean the codebase carries two parallel context-manager protocols that callers may want to combine.

## Open questions / TODOs

- **Open question (still open per PRD OQ-6)**: the proof-of-execution monitors (`VitisAIMonitor`, `OpenVINOMonitor`) still carry their own `to_dict()` methods as a transitional surface. The plan is to replace these with a typed `proof: ProofOfExecution | None` property in a follow-up, mirroring `result: OpTraceResult | None`. Until that lands, the ABC is asymmetric: op-tracing monitors have a typed accessor, proof-of-execution monitors do not.
- **`_result` typing**: should the base class declare `_result: OpTraceResult | None = None` so type checkers can verify subclass writes? Currently the `getattr` fallback hides any subclass that forgets to assign.
- **Should `set_onnx_op_types` be split**? A `OpTracingMixin` (or sub-ABC) carrying `set_onnx_op_types` + `result` would let mypy enforce "you implement both or neither". The current single-ABC design conflates two roles for the sake of dispatch convenience.

## Simplification opportunities

**Headline finding: the ABC is *modestly* over-pulling its weight, but the over-engineering is narrow.** Of the four concrete subclasses, only QNNMonitor is real. VitisAIMonitor produces real data but ignores every new hook. OpenVINOMonitor is a literal stub (`is_available()` returns `False`; `__enter__`/`__exit__` are pass). NullEPMonitor is a Null-Object. So the "five extension points" exist almost entirely for one consumer. That said, the deletion of `OpTracer` (a parallel 35-line ABC for the same problem) is a net win — the refactor halves the number of per-EP base classes even if the surviving one grew. The accidental complexity is concentrated, not pervasive.

Specific opportunities worth considering:

1. **`QNNMonitor.result` is a copy of the inherited default** (qnn_monitor.py:294-297). Delete it — the inherited `getattr(self, "_result", None)` already does the right thing, and the typed annotation is on the base class. One property, zero behavior change.
2. **`__init_subclass__` is high-cost / low-yield.** It catches one narrow misuse (class-level shadow with a non-bool literal) of one attribute. The cost is 15 lines of runtime introspection that fires on every subclass import. A `# type: ClassVar[bool]` annotation + mypy is the same guard at lower cost. The instance-shadow gap that the docstring itself flags is the more dangerous failure mode, and `__init_subclass__` can't help with it.
3. **`ep_name` could be a constructor argument** rather than a ClassVar. The only consumer (`session.py:649`) reads it via `monitor.ep_name`. ClassVar buys nothing — there's no "the QNN-specific monitor class targets the QNN EP" introspection happening at the class level. Instance-level would be just as expressive and would deflate one of the two ClassVars.
4. **`requires_session_teardown` could collapse into `ep_name`** because the only monitor that needs it is also the only one with `ep_name = "qnn"`. A `_NEEDS_SESSION_TEARDOWN_EPS = {"qnn"}` set in `session.py` would express the C-2 invariant equally well, eliminate the ClassVar from the ABC entirely, and remove the entire `__init_subclass__` guard. The downside: third-party monitors couldn't opt in without modifying session.py. Acceptable given there are no third-party monitors and the in-tree set is fixed.
5. **`set_onnx_op_types` with a no-op default is the right call** for the dispatch site, but the design would be cleaner if the *map building* were also pushed into the monitor — `WinMLSession.perf()` calls `monitor.prepare(onnx_path)` and only QNNMonitor's override actually parses the graph. Avoids paying for op-type map construction when the monitor doesn't need it (VitisAI / OpenVINO / Null pay nothing today only because of the no-op default; the `_build_op_type_map(self._onnx_path)` call still fires unconditionally at line 677).
6. **`OpenVINOMonitor` is dead weight.** It is a placeholder whose `is_available()` returns `False` literally — it can never actually be instantiated by the dispatcher. Either delete the file and let the dispatch fall through to a hard error (per FR-11), or strip it down to a one-line `OpenVINOMonitor = NullEPMonitor` alias. Its current shape implies a working monitor exists when none does.
7. **`NullEPMonitor` does pull its weight** — it eliminates `if monitor is not None:` checks in the perf hot path (`session.py:645`, `effective_monitor: WinMLEPMonitor = monitor if monitor is not None else NullEPMonitor()`). Real Null-Object Pattern win, justified.
8. **The class rename `EPMonitor → WinMLEPMonitor`** lands in this commit. Worth confirming the prefix is load-bearing (i.e. there's an unrelated `EPMonitor` somewhere) or just stylistic; if stylistic, the longer name adds noise at every subclass line and import site.
