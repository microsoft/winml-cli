# src/winml/modelkit/session/monitor/ep_monitor.py

## TL;DR
Substantially expands the `EPMonitor` ABC with v2.4 op-tracing hooks: optional `set_onnx_op_types()` injection point, a typed `result` property, `get_session_options()` / `get_provider_options()` for option contribution, class-level `requires_session_teardown` and `ep_name` flags, plus an `__init_subclass__` guard. Drops the `to_dict()` abstract method, narrowing the mandatory contract to `__enter__`/`__exit__`/`is_available`.

## Diff metrics
- Lines added: ~95
- Lines removed: ~9
- Modified (file existed at 87 lines pre-commit; ~173 post-commit)

## Role before vs after
- **Before:** Minimal ABC with three abstract methods (`__enter__`, `__exit__`, `to_dict`, `is_available`) plus the `NullEPMonitor` Null-Object subclass. Monitors were expected to expose their data via `to_dict()`.
- **After:** Richer monitor contract. Op-tracing monitors expose their data via a typed `result: OpTraceResult | None` property; non-op-tracing monitors inherit a default `None` via `getattr(self, "_result", None)`. The base class also acts as the integration point with `WinMLSession.perf()`, providing optional hooks for session-config entries, provider options, EP pinning, ONNX op-type map injection, and a teardown-ordering invariant flag.

## Symbol-level changes
- **`EPMonitor`** — modified (existing ABC, materially extended)
  - New class-level `ClassVar[bool] requires_session_teardown = False` — signals whether the monitor's data flush requires `ort.InferenceSession` destruction (QNN CSV flush case). Governs C-2 teardown ordering invariant in `WinMLSession.perf()`.
  - New class-level `ClassVar[str | None] ep_name = None` — when set, pins the perf session to a specific EP so `get_provider_options()` output actually reaches `add_provider_for_devices`.
  - New `__init_subclass__(cls, **kwargs)` — rejects subclasses that try to shadow `requires_session_teardown` with a non-bool, raising `TypeError` at class-definition time.
  - New concrete `get_session_options() -> dict[str, str]` — default `{}`; subclass override point for `SessionOptions.add_session_config_entry` entries.
  - New concrete `get_provider_options() -> dict[str, str]` — default `{}`; subclass override point for `add_provider_for_devices` options merge.
  - New concrete `set_onnx_op_types(onnx_op_types: dict[str, str]) -> None` — no-op default with `# noqa: B027` (intentional empty body). Op-tracing monitors override; called unconditionally by `WinMLSession.perf()` before `mon.__enter__()`.
  - New `@property result -> OpTraceResult | None` — returns `getattr(self, "_result", None)`. Replaces the removed `to_dict()` as the canonical typed data accessor for op-tracing monitors.
  - **Removed** `@abstractmethod to_dict()` — no longer part of the mandatory contract. Monitors that still need dict serialization either expose it via `result.to_dict()` or carry their own ad-hoc `to_dict()` (proof-of-execution monitors transitionally).
- **`NullEPMonitor`** — modified (the Null-Object subclass)
  - **Removed** override `to_dict() -> dict[str, Any]: return {}` — no longer needed since the base class no longer requires `to_dict`.
  - `is_available()` / `__enter__()` / `__exit__()` unchanged.
- **TYPE_CHECKING imports** — `OpTraceResult` added to the `if TYPE_CHECKING:` block (forward reference for the `result` property annotation).
- **`ClassVar`** — added to runtime `typing` import.

## Behavior / contract changes
- The mandatory ABC contract shrinks from {`__enter__`, `__exit__`, `to_dict`, `is_available`} to {`__enter__`, `__exit__`, `is_available`}.
- New optional contract layer composed of concrete defaults: `result`, `get_session_options`, `get_provider_options`, `set_onnx_op_types`, and the `requires_session_teardown` / `ep_name` class vars. Subclasses opt-in by overriding.
- Class-definition-time validation via `__init_subclass__` for `requires_session_teardown` — non-bool subclass shadow is rejected with `TypeError`. Note: instance-level shadow is explicitly noted as un-catchable here.
- The docstring example block is rewritten to reflect the new `session.perf(warmup=10, monitor=...)` single-context-manager API yielding a `PerfContext(stats, monitor)`, replacing the previous nested `with SomeEPMonitor() as hw:` pattern.

## Cross-file impact
- **Used by which modules:** every concrete monitor (`QNNMonitor`, `VitisAIMonitor`, `OpenVinoMonitor`, `HWMonitor`) subclasses this ABC; `WinMLSession.perf()` consumes the optional hooks (`requires_session_teardown`, `ep_name`, `get_session_options`, `get_provider_options`, `set_onnx_op_types`); `commands/perf.py` and `eval/evaluate.py` consume `monitor.result` for downstream reporting.
- **Depends on which modules:** stdlib `abc`, `typing`; forward ref to `.op_metrics.OpTraceResult` (TYPE_CHECKING only).

## Risks / subtleties
- Removing `to_dict()` from the abstract contract is a hard break — any out-of-tree monitor relying on `EPMonitor.to_dict` will silently lose the abstract enforcement (the docstring notes proof-of-execution monitors retain their own `to_dict()` "transitionally"). Mixed access patterns in callers (`ctx.monitor.result` vs `ctx.monitor.to_dict()`) are coexisting in the codebase.
- `result` accessor uses `getattr(self, "_result", None)` — relying on the convention that op-tracing subclasses set `self._result` during `__exit__`. There's no enforcement that they actually do; a forgetful subclass simply yields `None` silently.
- `__init_subclass__` only catches class-level `requires_session_teardown` shadowing; instance-level shadow (e.g. `self.requires_session_teardown = "yes"` in `__init__`) is not caught and would still corrupt the C-2 invariant in `WinMLSession.perf()`.
- `ep_name = None` semantics intentionally overloaded: `None` means "any EP fine" (NullEP, VitisAI, OpenVino) per the docstring, but if op-tracing monitors forget to set it, the silent drop of provider options is the exact failure mode the field exists to prevent — also un-enforced.

## Open questions / TODOs surfaced
- Docstring TODO: "proof-of-execution monitors (e.g. VitisAI, OpenVINO) currently expose theirs via `to_dict()` transitionally — to be replaced by a typed `proof` accessor in a follow-up."
- Should `set_onnx_op_types` be `@abstractmethod` with default `None` argument, or split into two ABC subclasses (op-tracing vs proof-of-execution)? The current design conflates the two roles in one base.
- Should the `_result` attribute be promoted to a typed `self._result: OpTraceResult | None = None` initialized on the base class to make the `getattr` fallback unnecessary?
