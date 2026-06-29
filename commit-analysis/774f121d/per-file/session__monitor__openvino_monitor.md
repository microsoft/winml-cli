# src/winml/modelkit/session/monitor/openvino_monitor.py

## TL;DR
Cosmetic-only diff (8 lines): class renamed from `OpenVinoMonitor` to `OpenVINOMonitor` (canonical capitalization per naming convention), base class renamed `EPMonitor` → `WinMLEPMonitor`, and `Self` import migrated from `typing_extensions` to `typing`. The class is still a **pure no-op placeholder** — `__enter__` and `__exit__` do nothing, `is_available()` always returns `False`, and `to_dict()` returns a hard-coded stub `{"ep": "OpenVINO", "device": "NPU", "status": "not_implemented"}`. No actual monitoring is performed and the class is never instantiated anywhere in the codebase.

## Diff metrics
- Lines added: 4
- Lines removed: 4
- Modified — file at 49 lines pre- and post-commit.

## Role before vs after
- **Before:** No-op stub class `OpenVinoMonitor(EPMonitor)` with `is_available() -> False`, vacuous `__enter__`/`__exit__`, and a hardcoded "not_implemented" `to_dict()` payload. Reserved for future Intel-specific NPU telemetry.
- **After:** Identical responsibility (still a no-op stub), with canonical naming. The class is now `OpenVINOMonitor(WinMLEPMonitor)` and is exported from `session/__init__.py` (`OpenVINOMonitor` in `__all__`).

## Symbol-level changes
- **Module docstring** — modified
  - `OpenVinoMonitor` → `OpenVINOMonitor` in the leading docstring line.
- **Import** — modified
  - `from .ep_monitor import EPMonitor` → `from .ep_monitor import WinMLEPMonitor`.
- **`TYPE_CHECKING` block** — modified
  - `from typing_extensions import Self` → `from typing import Self`.
- **Class** — renamed
  - `class OpenVinoMonitor(EPMonitor)` → `class OpenVINOMonitor(WinMLEPMonitor)`.
- All methods unchanged:
  - `__enter__` — returns `self`, no side effects.
  - `__exit__` — empty body.
  - `is_available()` — `return False`.
  - `to_dict()` — `return {"ep": "OpenVINO", "device": "NPU", "status": "not_implemented"}`.

## Behavior / contract changes
- None at runtime. Any caller checking `OpenVINOMonitor.is_available()` still gets `False`, so it is never selected by the dispatch path in `commands/perf.py::_resolve_ep_monitor`. The class exists purely as a hierarchy placeholder.
- After the `EPMonitor` → `WinMLEPMonitor` rename and the removal of `to_dict()` from the ABC's abstract contract (per the `ep_monitor.py` companion notes), this class technically no longer needs `to_dict()` at all — `WinMLEPMonitor.result` returns `None` by default for non-op-tracing monitors, and `commands/perf.py::_monitor_to_json_dict` falls through to `monitor.to_dict()` only when `monitor.result is None and hasattr(monitor, "to_dict")`. The stub method is therefore the only thing keeping `OpenVINOMonitor` distinguishable from `NullEPMonitor` in JSON output.

## Cross-file impact
- **Used by which modules:**
  - `session/__init__.py` re-exports the class.
  - `commands/perf.py::_resolve_ep_monitor` does **not** dispatch to it (only `QNNMonitor` and `VitisAIMonitor` have dispatch branches); grep confirms `OpenVINOMonitor` is referenced only by `session/__init__.py` and the `ep_monitor.py` docstring listing it among "proof-of-execution monitors that inherit the default and ignore the call."
  - The class is documented in `ep_monitor.py` as one of three monitors that "inherit the default and ignore the call" for `set_onnx_op_types()`.
- **Depends on which modules:** `.ep_monitor.WinMLEPMonitor` only.

## Risks / subtleties
- The class is a **dead-weight placeholder**: instantiation has no consumer (`_resolve_ep_monitor` never returns it), `is_available()` always returns `False`, and the stub `to_dict()` payload is never read. Its only purpose is to anchor the future implementation; in the meantime, it adds 1 export to `session/__init__.py` and 1 docstring-citation chain through `ep_monitor.py`.
- After the ABC contract loosened (`to_dict()` no longer abstract per `ep_monitor.py`), this class's `to_dict()` override has no enforcement preventing it from drifting away from the rest. If a future implementer forgets to override `to_dict()` after wiring up real telemetry, the stub `{"status": "not_implemented"}` would silently keep appearing in JSON reports.
- Inconsistent capitalization between module name (`openvino_monitor.py`) and class name (`OpenVINOMonitor`) — module path uses lowercase per Python convention, but anyone grepping for `OpenVino` will miss the rename.

## Open questions / TODOs surfaced
- Should this stub class be deleted entirely until a real implementation lands? The placeholder pattern is documented in PRD `1_prd.md` as intentional but the cost (one dead export, one dead docstring chain) accrues forever.
- If kept, should the docstring note explicitly that callers should prefer `HWMonitor` for OpenVINO NPU utilization (already mentioned, but worth promoting from "module docstring" to "RuntimeError when instantiated")?

## Simplification opportunities
- **Dead code in the strictest sense.** The class has no callers, returns `False` from `is_available()`, and emits a hardcoded `"not_implemented"` payload. Per `MEMORY.md` ("hard-break refactors + function consolidation, never compat shims"), the cleanest move is to **delete `OpenVINOMonitor` and `openvino_monitor.py`** along with the `session/__init__.py` export. When real OpenVINO telemetry is needed, recreate the file with the actual implementation. The class adds zero behavior and one indirection layer.
- `to_dict()` override could be deleted now (the base class no longer requires it per the ABC contract change); `NullEPMonitor` already dropped its override for the same reason. Keeping the stub `{"status": "not_implemented"}` payload is the only thing it does.
- `is_available()` could be folded into the base class as `return False` default for all "placeholder" subclasses, but only one such subclass exists, so this is over-abstraction in waiting.
- The whole "EPMonitor ABC + placeholder no-ops" pattern is overbuilt for the current 2-real-monitor (`QNNMonitor`, `VitisAIMonitor`) world. The `_resolve_ep_monitor` dispatch in `commands/perf.py` is an explicit `if/elif` chain — no polymorphism is being exploited; the ABC is being used to share `result`/`set_onnx_op_types`/`requires_session_teardown` defaults across two implementations and one placeholder. A protocol or a `dataclass` with `result: OpTraceResult | None` as a field would be lighter weight.
