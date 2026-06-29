# src/winml/modelkit/session/monitor/vitisai_monitor.py

## TL;DR
Cosmetic-only diff (8 lines): base class rename `EPMonitor` → `WinMLEPMonitor`, and `Self` import migrated from `typing_extensions` to `typing`. The class has real implementation: it captures `xrt-smi` `command_submissions`/`command_completions` deltas around inference as AMD-NPU proof-of-execution. Exposes data via `to_dict()` only (no typed `result` accessor) — confirming the "transitional `to_dict()`" path mentioned in `commands/perf.py`'s `_monitor_to_json_dict` docstring.

## Diff metrics
- Lines added: 4
- Lines removed: 4
- Modified — file at 182 lines pre- and post-commit.

## Role before vs after
- **Before:** `class VitisAIMonitor(EPMonitor)` — real proof-of-execution monitor for AMD NPU. `__enter__` takes an `xrt-smi` snapshot (submissions/completions counters per-PID), `__exit__` takes a second snapshot, deltas are exposed via `command_submissions`, `command_completions`, `hw_context_status`, and `npu_proven` (True if submissions delta > 0). Data is JSON-serialized via `to_dict()` returning `{"ep": "VitisAI", "npu_proven": bool, "xrt_smi": {...}}`.
- **After:** Identical behavior. Only the base class name and `Self` import location changed.

## Symbol-level changes
- **Module docstring** — modified
  - Line 11: `(which always runs alongside EPMonitors)` → `(which always runs alongside WinMLEPMonitors)`.
- **Import** — modified
  - `from .ep_monitor import EPMonitor` → `from .ep_monitor import WinMLEPMonitor`.
- **`TYPE_CHECKING` block** — modified
  - `from typing_extensions import Self` → `from typing import Self`.
- **Class** — base changed
  - `class VitisAIMonitor(EPMonitor)` → `class VitisAIMonitor(WinMLEPMonitor)`.
- All other symbols unchanged:
  - `__init__` (zero-arg) initializing `_xrt_client`, `_submissions_before/after`, `_completions_before/after`, `_last_hw_status`.
  - `__enter__` / `__exit__` calling `_xrt_start()` / `_xrt_stop()`.
  - Properties `command_submissions`, `command_completions`, `hw_context_status`, `npu_proven`.
  - Classmethod `is_available()` (returns `False` off-Windows; otherwise probes `XrtSmiClient().is_available`).
  - `to_dict()` returning the AMD-specific payload.
  - Private helpers `_xrt_start()`, `_xrt_stop()` (lazy-imports `._xrt_smi.XrtSmiClient`, swallows `ImportError`/`OSError`).

## Behavior / contract changes
- None at runtime.
- Notably **does not** declare class-level `ep_name = "VitisAIExecutionProvider"` (the post-rename `WinMLEPMonitor` introduces this `ClassVar` for EP pinning per the `ep_monitor.py` companion). Per the docstring on `WinMLEPMonitor.ep_name`, `None` means "any EP fine" — which is correct for VitisAI (no provider options to merge), but the silent default means a future provider-options addition would be silently dropped.
- Notably **does not** declare `requires_session_teardown` either — defaults to `False`, which is correct (xrt-smi snapshots don't need ORT teardown to flush).
- **Does not** expose a typed `result` accessor — `WinMLEPMonitor.result` property returns `getattr(self, "_result", None)`, and `VitisAIMonitor` never sets `self._result`. Consumers reach the proof-of-execution data only via `to_dict()`. This matches the `commands/perf.py::_monitor_to_json_dict` docstring: *"Proof-of-execution monitors (VitisAI, OpenVINO) still expose theirs via `to_dict()` transitionally — to be replaced by a typed `proof` accessor in a follow-up PR (see PRD OQ-6)."*

## Cross-file impact
- **Used by which modules:**
  - `session/__init__.py` re-exports `VitisAIMonitor`.
  - `commands/perf.py::_resolve_ep_monitor` dispatches to `VitisAIMonitor()` when `ep_norm == "vitisai" and VitisAIMonitor.is_available()`.
  - `commands/perf.py::_monitor_to_json_dict` reads `monitor.to_dict()` after the `monitor.result` fallthrough (since `VitisAIMonitor` never populates `_result`).
  - `ep_monitor.py` docstring cites `VitisAIMonitor` as the canonical example of a "proof-of-execution" monitor that "inherits the default and ignores the call" for `set_onnx_op_types()`.
- **Depends on which modules:**
  - `.ep_monitor.WinMLEPMonitor` (base).
  - `._xrt_smi.XrtSmiClient` (sibling, lazy-imported inside `_xrt_start`/`_xrt_stop`/`is_available` to keep import-time light off-Windows).
  - stdlib `logging`, `sys`, `os`.

## Risks / subtleties
- `_xrt_start` lazy-imports `XrtSmiClient` inside the function body and constructs it before checking `is_available`. If the constructor raises something other than `ImportError`/`OSError` (e.g. a generic `RuntimeError`), it bubbles up uncaught. The classmethod `is_available()` catches `RuntimeError` too — the two error-handling sets are inconsistent.
- `_xrt_stop` early-returns when `_xrt_client is None or not _xrt_client.is_available`, but if `_xrt_start` failed silently (logged at DEBUG), `_xrt_client` may be `None` *or* may be a non-`None` client that nonetheless had `is_available` flip between calls (unlikely but unhandled).
- `to_dict()` returns a hand-crafted payload — when the typed `proof` accessor lands per OQ-6, both code paths need to be migrated simultaneously to avoid double-reporting via `commands/perf.py::_monitor_to_json_dict` (which checks `monitor.result` first, then falls through to `to_dict`).
- `npu_proven` is `command_submissions > 0`, where `command_submissions = max(0, after - before)`. If xrt-smi data is unavailable, both `before` and `after` are 0, so `npu_proven` is `False` — which is the correct conservative answer but indistinguishable from "monitor ran successfully but NPU did nothing." A `data_unavailable` sentinel state would clarify.
- `command_submissions` field name reuses the underlying xrt-smi terminology; consumers reading `to_dict()` get a nested `xrt_smi` sub-dict that mirrors this internal naming.

## Open questions / TODOs surfaced
- PRD OQ-6 (cited in `commands/perf.py`): replace the transitional `to_dict()` with a typed `proof` accessor on `WinMLEPMonitor`. Open for both `VitisAIMonitor` and the no-op `OpenVINOMonitor` placeholder.
- Should `VitisAIMonitor` set `class-level ep_name = "VitisAIExecutionProvider"` to be future-proof when provider options become needed? Currently `None` works because no options are contributed, but the default is silently brittle.
- Should the `_xrt_start` / `_xrt_stop` exception sets unify? `is_available()` catches `ImportError, RuntimeError`; `_xrt_start` catches `ImportError, OSError`.

## Simplification opportunities
- `_xrt_start` and `_xrt_stop` are private helpers with **single callers each** (only `__enter__` calls `_xrt_start`, only `__exit__` calls `_xrt_stop`). Inlining both would lose nothing — the public API is the context manager protocol itself, and the helper-name split adds an indirection without clarifying anything (the docstring on each helper says only "before"/"after"). **Consolidation candidate** per `MEMORY.md`.
- `_last_hw_status` is fetched from `get_hw_contexts(pid)[-1].status` and exposed as the `hw_context_status` property, which is then shoveled into the `xrt_smi` sub-dict by `to_dict()`. The property has one consumer (`to_dict`); folding the lookup directly into `to_dict` would remove an instance attribute and a property.
- `to_dict()` should arguably become `proof()` returning a typed dataclass per OQ-6; once that lands, the `command_submissions`/`command_completions`/`hw_context_status`/`npu_proven` properties all become fields on that dataclass and the class shrinks substantially.
- The class repeats the `import os` statement inside both `_xrt_start` and `_xrt_stop` — this is intentional (lazy import) but `os.getpid()` is also the only `os` call, so caching `pid = os.getpid()` on the instance in `__init__` would remove both inline imports and both calls. Minor nit but tightens the file.
- `is_available()` does `XrtSmiClient().is_available` — constructing an instance just to read a boolean is wasteful if the constructor is non-trivial. A classmethod predicate on `XrtSmiClient` would let `VitisAIMonitor.is_available` skip the construction.
