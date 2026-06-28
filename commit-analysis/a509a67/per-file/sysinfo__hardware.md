# src/winml/modelkit/sysinfo/hardware.py

## TL;DR
Receives one of the three functions evicted from the now-deleted `sysinfo/device.py`: a new public `get_available_devices()` that returns the prioritised list of device categories (NPU > GPU > CPU) present on the host. The new function is a near-verbatim copy of the old private `_get_available_devices()` from `sysinfo/device.py`, minus the lazy intra-module imports (now unnecessary). Adds a module logger. No other changes — the `CPU`/`GPU`/`NPU`/`RAM` hardware classes and `get_vendor_id_device_id_from_pnp_id` are untouched.

## Diff metrics
- Lines: +29 / -0 (net +29)
- Hunks: 1 (single addition near top-of-file, between imports and the existing first function)
- New symbols: 1 function (`get_available_devices`), 1 module-level (`logger`).
- Removed symbols: none.

## Role before vs after
- Before: Pure hardware-inventory module. Defined `CPU`, `GPU`, `NPU`, `RAM` (with `.get_all()` factory methods backed by WMI / CIM / PnP queries), plus the `get_vendor_id_device_id_from_pnp_id` helper. No device-category aggregation or prioritisation — callers had to call `NPU.get_all()`, `GPU.get_all()`, etc. and assemble a list themselves.
- After: Same hardware-inventory module *plus* one aggregator that produces the host's prioritised device-category list. This is the natural home for the function — it operates on the `NPU`/`GPU` classes already defined here, and its previous home (`sysinfo/device.py`) is gone. The aggregator deliberately returns category strings (`"npu" | "gpu" | "cpu"`), not hardware instances; the docstring explicitly directs instance-level callers back to `NPU.get_all() / GPU.get_all()`.

## Symbol-level changes
- **Added** `logger = logging.getLogger(__name__)` at module level (line 12). Previously the module did no logging.
- **Added** `import logging` (line 5).
- **Added** `def get_available_devices() -> list[str]` (lines 15–37). Behaviour:
  1. Try `NPU.get_all()`; if truthy, append `"npu"`. Exceptions are swallowed and logged at `DEBUG`.
  2. Try `GPU.get_all()`; if truthy, append `"gpu"`. Exceptions are swallowed and logged at `DEBUG`.
  3. Unconditionally append `"cpu"`.
  4. Return the list, in NPU-first order.
  Function body is byte-equivalent to the old `_get_available_devices` in `sysinfo/device.py` *except* the lazy `from .hardware import NPU / GPU` imports were dropped (the symbols are now in the same module).
- No changes to `CPU`, `GPU`, `NPU`, `RAM`, `get_vendor_id_device_id_from_pnp_id`, or any `Architecture` / `to_dict` method.

## Behavior / contract changes
- Public surface gains one function. The function name is now public (no leading underscore) — old `_get_available_devices` was explicitly documented as "internal helper for `resolve_device` and should not be called directly by external code." That restriction is lifted; `get_available_devices` is exported in `sysinfo/__init__.py` and listed in `__all__`.
- Return shape unchanged: `list[str]` containing some subset of `["npu", "gpu", "cpu"]` in priority order, with `"cpu"` always last.
- Exception semantics unchanged from the old private function: NPU/GPU detection failures are caught broadly (`except Exception`) and logged at `DEBUG`; the function never raises.
- **EP cross-check removed at this layer.** The old `resolve_device` in `sysinfo/device.py` cross-referenced the device list against available EPs and could downgrade an answer (e.g. NPU present but no QNN/DML/VitisAI EP installed -> would *not* return "npu"). `get_available_devices` does **not** do this — it reports physical hardware only. The EP cross-check now lives one layer up in `session.auto_detect_device()` (and, transitively, inside the auto-detect branch of `session.resolve_device`).

## Cross-file impact
- `sysinfo/__init__.py` re-exports `get_available_devices` and adds it to `__all__` (see companion analysis).
- `session/ep_device.py` imports `get_available_devices` inside `auto_detect_device()` and uses its result as the device-priority walk order:
  ```python
  from ..sysinfo.hardware import get_available_devices
  ...
  available_devices = get_available_devices()
  ```
  This is the *only* consumer in `src/` (verified via grep). `session.resolve_device(ep=None, device="auto")` reaches `get_available_devices` transitively by delegating to `auto_detect_device()`.
- `commands/sys.py:379` still has a stale comment referencing the old private name `_get_available_devices()`; no functional code touches the new symbol from here.

## Risks / subtleties
- The lazy-import idiom in the old code (`from .hardware import NPU; from .hardware import GPU`) defended against a (now-non-existent) import-time cycle between `sysinfo.device` and `sysinfo.hardware`. The new in-module code references `NPU` and `GPU` directly *but* the function definition appears at line 15 — **before** the `NPU` (line 208) and `GPU` (line 142) classes are defined. Python is fine with this because name resolution happens at call time, not at def time, but a reader scanning top-to-bottom may briefly wonder whether this works. Consider moving the function below the class definitions, or keeping a comment noting the call-time-resolution dependency.
- `except Exception` is a very broad catch; the docstring promises priority NPU > GPU > CPU, but if `NPU.get_all()` raises *anything* (including a `KeyboardInterrupt` subclass — unlikely but technically `BaseException` covers that), it'll be swallowed silently. The old code had the same flaw; this commit didn't tighten it.
- The function returns *device categories*, not hardware instance counts. A host with 0 NPUs *and* 0 GPUs (e.g. a CPU-only VM) returns `["cpu"]`, which is correct. A host with 2 GPUs returns `["gpu", "cpu"]` regardless of how many GPUs there are. Callers that need "how many devices of category X" must still go to `NPU.get_all()` / `GPU.get_all()` — the docstring calls this out explicitly.
- The module logger is new but only emits at `DEBUG`. In default logging configurations this is silent; production diagnostics of "why isn't my NPU detected?" require setting `WINML_LOG_LEVEL=DEBUG` (or equivalent) and re-running. This was the same situation in the old `sysinfo/device.py`; no regression.

## Open questions / TODOs surfaced
- Should the function move *below* the `CPU`/`GPU`/`NPU`/`RAM` class definitions to match top-to-bottom readability? Functionally equivalent, but cleaner.
- Is there value in returning *counts* alongside categories (e.g. `dict[str, int]` instead of `list[str]`) for callers like `commands/sys.py`? Out of scope for this commit, but the constrained `list[str]` shape may push future callers to re-query `*.get_all()`, duplicating work.
- The stale `_get_available_devices` reference in `commands/sys.py:379` should be updated in a follow-up.
