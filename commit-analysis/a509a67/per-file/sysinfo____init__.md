# src/winml/modelkit/sysinfo/__init__.py

## TL;DR
Public API of `sysinfo` package retracted from two device-routing helpers (`get_ep_device_map`, `resolve_device`) to a single hardware-detection helper (`get_available_devices`). The package is now narrowed to its original responsibility — *what hardware is present* — and is no longer a router for ONNX Runtime EPs. All EP/device routing has moved to `session/ep_device.py`.

## Diff metrics
- Lines: +2 / -3 (net -1)
- Hunks: 2
- Re-exports changed: removed `get_ep_device_map`, removed `resolve_device`, added `get_available_devices`.

## Role before vs after
- Before: `sysinfo` exported hardware classes (`CPU`/`GPU`/`NPU`/`OS`/`SysInfo`) **and** two EP-aware routing helpers (`get_ep_device_map` returning the manual EP→device dict, `resolve_device` returning `(chosen_device, available_list)` after cross-checking EP availability). Effectively a dual-purpose facade.
- After: `sysinfo` is purely hardware introspection. The only non-hardware-class symbol it surfaces is `get_available_devices`, which returns the *prioritised list of device categories present on the host* (no EP cross-check). All EP-related deduction now lives in `winml.modelkit.session` — `resolve_device(ep, device)` returns a typed `EPDevice`, and `auto_detect_device()` returns just the auto-picked category string.

## Symbol-level changes
- Removed re-export: `from .device import get_ep_device_map, resolve_device` (entire line; `sysinfo/device.py` is deleted).
- Added re-export: `get_available_devices` from `.hardware`.
- `__all__`: removed `"get_ep_device_map"`, `"resolve_device"`; added `"get_available_devices"`. `CPU`, `GPU`, `NPU`, `OS`, `SysInfo` unchanged.

## Behavior / contract changes
- `sysinfo.resolve_device` is **gone from this package**. Any `from winml.modelkit.sysinfo import resolve_device` now raises `ImportError`. The replacement is `winml.modelkit.session.resolve_device(ep, device) -> EPDevice` — a fully different function with a different signature (2 optionals: `ep` and `device`), a different return type (`EPDevice` dataclass instead of `tuple[str, list[str]]`), and a richer failure taxonomy (`ValueError | EPNotDiscovered | EPRegistrationFailed | DeviceNotFound | AmbiguousMatch`). Callers that only need the auto-picked category string can use `winml.modelkit.session.auto_detect_device()` instead.
- `get_ep_device_map` is gone with no public replacement at this layer. EP↔device routing now goes through the `EPDeviceSpec` catalog inside `session/ep_device.py`; consumers should use `default_device_for_ep` / `eps_for_device` from `..session` rather than reading a dict.
- `get_available_devices` keeps the *signature and priority order* of the old private `_get_available_devices` (NPU > GPU > CPU, CPU always last), but **no EP cross-check** — it reports physical presence only. The EP cross-check now lives one layer up in `session.auto_detect_device()`.

## Cross-file impact
- Consumers of `sysinfo.resolve_device` (old) had to migrate to either `session.resolve_device(ep, device)` (when an `EPDevice` is needed) or `session.auto_detect_device()` (when only the category string is needed). Migrated call sites live in `commands/perf.py`, `commands/eval.py`, `commands/config.py`, `config/build.py`, and `config/precision.py`.
- Consumers of `get_ep_device_map`: none remain — the manual dict is replaced by the `EPDeviceSpec` catalog (single source of truth).
- Consumers of `get_available_devices`: `session/ep_device.py` (used inside `auto_detect_device`); `commands/sys.py:379` references the old internal name in a comment.

## Risks / subtleties
- The bare name `resolve_device` survives in the codebase but means something different from the pre-branch sysinfo function. `from ..sysinfo import resolve_device` is the buggy legacy pattern (now `ImportError`); the typed replacement is `from ..session import resolve_device` (returns `EPDevice`); the string-only auto-pick path is `from ..session import auto_detect_device`. Reviewers fixing old import sites must pick the correct one based on whether the caller consumes an `EPDevice` or a category string.
- `__all__` no longer exposes any EP-aware symbol from `sysinfo`. Any tool that introspected `sysinfo.__all__` to discover EP mappings now sees nothing.
- The package no longer logs an EP-discovery warning ("No execution providers detected, falling back to CPU") — that warning moved into `session.auto_detect_device()`. If a caller historically relied on importing `sysinfo` to surface that warning at startup, the warning is now deferred until a session resolution call.

## Open questions / TODOs surfaced
- No back-compat shim from `sysinfo` for the old `resolve_device` name; was that considered? A deprecation alias would absorb the import-path migration cost for downstream users, but the commit deliberately chose a hard break (Option A in the commit message).
- `commands/sys.py:379` still references `_get_available_devices` (the old private name) in a comment — stale doc-comment.
