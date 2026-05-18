# Review: `src/winml/modelkit/session/monitor/ep_monitor.py`

**Status:** modified
**Lines added/removed:** 95+ / 18-
**Diff command:** `git diff 1bea4cf..HEAD -- src/winml/modelkit/session/monitor/ep_monitor.py`

## 1. Purpose of this file

Defines the `EPMonitor` ABC and the `NullEPMonitor` no-op concrete subclass. The v2.4 additions make this the central contract for all per-EP monitors: `get_session_options` and `get_provider_options` configure ORT session creation; `set_onnx_op_types` injects the ONNX graph's node-name-to-op-type map; `result` exposes the typed op-trace output. The `requires_session_teardown` and `ep_name` class-level flags give `WinMLSession.perf()` the information it needs to manage teardown ordering and EP binding without knowing the monitor's concrete type.

## 2. Changes summary

- Added `ClassVar` import.
- Added `TYPE_CHECKING` import of `OpTraceResult` (for the `result` property return type).
- Added `requires_session_teardown: ClassVar[bool] = False` class variable with subclass-rejection guard in `__init_subclass__`.
- Added `ep_name: ClassVar[str | None] = None` class variable.
- Added `get_session_options() -> dict[str, str]` with default `{}`.
- Added `get_provider_options() -> dict[str, str]` with default `{}`.
- Added `set_onnx_op_types(onnx_op_types: dict[str, str]) -> None` as a concrete no-op default.
- Added `result -> OpTraceResult | None` property using `getattr(self, "_result", None)`.
- Removed `to_dict()` abstract method (v2.4 FR-20).
- Updated module docstring and class example.
- Removed `NullEPMonitor.to_dict()`.

## 3. Per-symbol review

### `EPMonitor` (class)

- **Role:** ABC that all EP-specific monitors inherit from.
- **Signature:** `class EPMonitor(ABC):`
- **Behavior:** Provides two mandatory abstract methods (`__enter__`, `__exit__`, `is_available`) and four concrete-default hooks (`get_session_options`, `get_provider_options`, `set_onnx_op_types`, `result`). The class-level flags `requires_session_teardown` and `ep_name` are read by `WinMLSession.perf()` without isinstance checks.
- **Invariants:** `requires_session_teardown` must remain a class-level `bool`; `__init_subclass__` enforces this for subclasses that explicitly shadow it. `ep_name` must be `None` or a short EP string that `expand_ep_name` can canonicalize (e.g. `"qnn"`, `"dml"`, `"vitisai"`).
- **Risks / concerns:** `__init_subclass__` only rejects non-bool shadows declared at class scope in the subclass's own `__dict__`. It does NOT catch instance-level shadowing in `__init__` (e.g. `self.requires_session_teardown = "yes"`). This is documented in the `__init_subclass__` docstring. The gap is acceptable since instance shadowing of a `ClassVar` is already a type-checker error.
- **Tests:** `tests/unit/session/monitor/test_ep_monitor_base.py` (`test_requires_session_teardown_must_be_bool`, `test_null_monitor_default_*`, `test_ep_monitor_is_abstract`); `tests/unit/session/monitor/test_ep_monitor_extensions.py` (all).

---

### `requires_session_teardown: ClassVar[bool] = False`

- **Role:** Tells `WinMLSession.perf().__exit__` whether to destroy the ORT session before calling `monitor.__exit__`. `True` for QNN (CSV flush), `False` everywhere else.
- **Invariants:** Design Constraint C-2: session reset must happen before monitor `__exit__` when this is `True`.
- **Risks / concerns:** The docstring accurately calls this an "ORT-specific hint" that leaks into the ABC (C-6 tradeoff). The risk of a future subclass setting this to a non-bool string is caught at class-definition time by `__init_subclass__`, but instance-level mutation is not.
- **Tests:** `test_ep_monitor_base.py::test_requires_session_teardown_must_be_bool`, `test_qnn_monitor.py::test_requires_session_teardown_true`.

---

### `ep_name: ClassVar[str | None] = None`

- **Role:** Short EP name used by `WinMLSession.perf()` to validate that the monitor's target EP matches the session's `EPDevice`. When `None`, no EP-binding check is performed.
- **Behavior:** `WinMLSession.perf()` (session.py:680-684) calls `expand_ep_name(monitor.ep_name)` and compares against `self._ep_device.ep`. This means `ep_name` must be a short form that `expand_ep_name` understands (e.g. `"qnn"` → `"QNNExecutionProvider"`). The docstring says "short name" but does not specify the vocabulary explicitly.
- **Risks / concerns:** The short-name vocabulary is implicitly defined by `_SHORT_TO_CANONICAL` in `ep_device.py`. If a future subclass uses a casing variant not in that table, `expand_ep_name` will return the value unchanged, and the comparison against the canonical EP name in `ep_device.ep` will silently pass (since both would be the unrecognized string). This is a silent correctness risk, not a crash. The fix is to document that `ep_name` must be a key in `_SHORT_TO_CANONICAL` or a canonical EP name.
- **Tests:** `tests/unit/session/test_winml_session.py:670-721` covers mismatch detection end-to-end.

---

### `set_onnx_op_types(onnx_op_types: dict[str, str]) -> None`

- **Role:** Concrete no-op default. Op-tracing monitors override to store the injected ONNX node-name-to-op-type map.
- **Signature:** `def set_onnx_op_types(self, onnx_op_types: dict[str, str]) -> None:`
- **Behavior:** Empty body; `# noqa: B027` suppresses ruff's "method with no body" lint. Called unconditionally by `WinMLSession.perf().__enter__` on every monitor.
- **Invariants:** FR-19: the no-op default makes the unconditional call safe for non-op-tracing monitors.
- **Risks / concerns:** None. The design is correct. The `noqa` comment is necessary and annotated with an explanatory suffix.
- **Tests:** `test_ep_monitor_extensions.py::test_null_monitor_set_onnx_op_types_is_no_op`, `test_set_onnx_op_types_accepts_empty_dict`, `test_set_onnx_op_types_returns_none`.

---

### `result -> OpTraceResult | None` (property)

- **Role:** Typed accessor for the op-trace output. Returns `None` for monitors that never set `self._result`.
- **Signature:** `@property def result(self) -> OpTraceResult | None:`
- **Behavior:** `getattr(self, "_result", None)`. Op-tracing monitors populate `self._result` in `__exit__`. `NullEPMonitor`, `VitisAIMonitor`, and `OpenVinoMonitor` never set `self._result` so they return `None`.
- **Invariants:** FR-18. The `getattr` sentinel approach means no abstract constraint forces subclasses to define `_result`; it is purely a naming convention.
- **Risks / concerns:** `QNNMonitor` overrides `result` directly (qnn_monitor.py:284-287), so the base-class getter is never used for `QNNMonitor`. The base-class getter is the fallback for future monitors that follow the convention without overriding the property. If a future monitor stores its result in a differently-named attribute (e.g. `self._trace_result`), the base-class getter silently returns `None` instead of raising — a silent failure. The convention should be documented explicitly.
- **Tests:** `test_ep_monitor_extensions.py::test_result_returns_self_dot_result_when_set`, `test_result_falls_back_when_subclass_omits_result_attr`, `test_result_default_is_none`.

---

### `get_session_options() -> dict[str, str]`

- **Role:** Supplies `SessionOptions.add_session_config_entry()` contributions for this monitor.
- **Behavior:** Returns `{}` by default. `QNNMonitor` overrides to supply EPContext caching options.
- **Invariants:** NFR-4 — must return the same dict content on repeated calls (idempotent). The default is trivially idempotent.
- **Tests:** `test_ep_monitor_base.py::test_null_monitor_default_get_session_options`, `test_qnn_monitor.py::test_get_session_options_idempotent`.

---

### `get_provider_options() -> dict[str, str]`

- **Role:** Supplies options merged into `add_provider_for_devices([ep], opts)`.
- **Behavior:** Returns `{}` by default. `QNNMonitor` overrides to supply `profiling_level` and `profiling_file_path`.
- **Tests:** `test_ep_monitor_base.py::test_null_monitor_default_get_provider_options`.

---

### `NullEPMonitor`

- **Role:** No-op concrete monitor for sessions that don't need EP-specific observation.
- **Behavior:** `__enter__`/`__exit__` do nothing. `is_available()` always returns `True`. Inherits all defaults from `EPMonitor` — `result` returns `None`, `set_onnx_op_types` is a no-op.
- **Risks / concerns:** Removal of `to_dict()` (which returned `{}`) is correct per FR-20. Any caller that previously called `monitor.to_dict()` on a `NullEPMonitor` will get an `AttributeError` at runtime. The `impl-status.md` notes the transitional `to_dict()` shim on `VitisAIMonitor`/`OpenVinoMonitor` but does not call out `NullEPMonitor` — its callers in `commands/perf.py` route through `_get_monitor_dict()` which checks `monitor.result` first (not `to_dict()`), so this is safe.
- **Tests:** Covered by `test_ep_monitor_base.py`.

## 4. Cross-cutting concerns

**Spec drift:** Implementation matches PRD §4.10 (FR-10), §4.18 (FR-18), §4.19 (FR-19), §4.20 (FR-20) exactly. The docstring example uses `ctx.stats.mean_ms` which aligns with the v2.4 `PerfContext` shape.

**Information-hiding contract:** `OpTraceResult` is imported under `TYPE_CHECKING` only (ep_monitor.py:32). At runtime, the `result` property returns `getattr(self, "_result", None)` which is untyped. This is correct — the type annotation is for static analysis only.

**Deferred work:** No TODO markers. The `VitisAIMonitor`/`OpenVinoMonitor` `to_dict()` transitional path (OQ-6) is documented in PRD §9 but not in this file.

**EPDevice / ep_name:** `ep_name` is a short-form string (e.g. `"qnn"`), not a canonical `EPDevice.ep` string. `WinMLSession.perf()` applies `expand_ep_name()` before comparing. This indirection is correct but must be maintained carefully when new EPs are added — if `expand_ep_name` does not know the short form, the EP-binding check silently passes.

## 5. Confidence level

**High.** The changes are well-specified (PRD §4.18/19/20) and the `__init_subclass__` guard is a useful defensive addition. The main residual risk is the `ep_name` short-form vocabulary convention being implicit rather than explicitly validated.

## 6. Verbatim risk inventory

| Severity | Location | Description |
|----------|----------|-------------|
| Low | `ep_monitor.py:72` | `ep_name` is documented as "short name" but the valid vocabulary is implicit (must be in `_SHORT_TO_CANONICAL` in `ep_device.py`). An unrecognized short name silently passes the EP-binding mismatch check instead of raising. |
| Low | `ep_monitor.py:118-120` | `__init_subclass__` rejects class-scope non-bool shadows but not instance-level ones (`self.requires_session_teardown = "yes"` in `__init__` would bypass the guard). Documented but not enforced at runtime. |
| Info | `ep_monitor.py:140-142` | `result` property uses `getattr` sentinel convention; if a future monitor stores its result under a different attribute name, `result` silently returns `None`. The convention should be documented in the ABC's class docstring. |
