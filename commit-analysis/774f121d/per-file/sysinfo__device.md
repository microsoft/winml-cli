# src/winml/modelkit/sysinfo/device.py

## TL;DR
**File deleted entirely (191 lines).** Same eviction as a509a67 — its three responsibilities are split: (1) hardware-detection (`_get_available_devices`) lifts up to `sysinfo/hardware.py` as the public `get_available_devices`. (2) Live EP discovery (`_get_available_eps`) moves to `session/ep_registry.py::available_eps()`. (3) The static EP↔device map (`_EP_DEVICE_MAP`) becomes the `EPDeviceSpec` catalog in `session/ep_device.py`. The string-only `resolve_device("auto")` entry point splits into `session.auto_detect_device()` (returns a category string) and `session.resolve_device(ep, device)` (returns a fully typed `EPDevice`).

## Diff metrics
- Lines: +0 / -191 (file deleted)
- Pre-state symbols: 1 module dict (`_EP_DEVICE_MAP`), 1 derived dict (`_DEVICE_EP_MAP`), 1 frozenset (`_VALID_DEVICES`), 4 functions (`get_ep_device_map`, `_get_available_devices`, `_get_available_eps`, `resolve_device`).
- Post-state symbols: none (file gone).

## Role before vs after
- Before: a 191-line module sitting *inside* `sysinfo` but doing EP-vs-device routing. It owned a hardcoded `_EP_DEVICE_MAP` (NVIDIA/AMD/Qualcomm/MS/Intel + CPU), a derived inverse map, a private hardware-presence detector, a cached EP discovery (combining `WinMLEPRegistry` + `ort.get_available_providers()`), and the public `resolve_device("auto"|...)`. Architecturally this conflated three concerns: (1) static EP↔device knowledge, (2) live hardware introspection, (3) live EP-plugin introspection.
- After: the three concerns are separated:
  1. Static EP↔device knowledge → `EPDeviceSpec` catalog (ordered tuple of dataclasses) in `session/ep_device.py`. The dict-shape API is *not* preserved; consumers use catalog helpers.
  2. Live hardware introspection → `sysinfo/hardware.py::get_available_devices` (public, was private).
  3. Live EP-plugin introspection → `session/ep_registry.py::available_eps()` (public, was private; `functools.lru_cache(maxsize=1)` semantics carried over).
  Plus: the public entrypoint is reshaped. `session.resolve_device(ep, device) -> EPDevice` is a typed resolver; the string-only auto-pick path lives in `session.auto_detect_device() -> str`.

## Symbol-level changes (mapping old → new)
| Old (deleted)                                            | New home                                                                                                | Visibility change                       | Notes |
|----------------------------------------------------------|---------------------------------------------------------------------------------------------------------|-----------------------------------------|-------|
| `_EP_DEVICE_MAP: dict[str, str]` (8 entries, hardcoded) | `EP_DEVICE_SPECS` catalog in `session/ep_device.py` (~13 `EPDeviceSpec` entries)                        | private → public, structural            | OpenVINO `"npu/gpu/cpu"` string parsing is gone; OpenVINO now has separate catalog entries per device. |
| `_DEVICE_EP_MAP` (derived inverse)                       | Derived on demand from the catalog via `eps_for_device(category)`                                       | private → public helper                 | No longer materialised as a module-level dict. |
| `_VALID_DEVICES = frozenset({"npu","gpu","cpu"})`        | `VALID_DEVICES` in `session/ep_device.py` (derived from catalog)                                        | private → public                        | Same content; sourced from catalog. |
| `get_ep_device_map() -> dict[str, str]`                  | **No direct replacement.** Use `default_device_for_ep(ep_full)` or iterate `eps_for_device(category)`   | removed                                 | Dict shape intentionally not preserved; multi-device EPs (OpenVINO) couldn't fit cleanly. |
| `_get_available_devices() -> list[str]`                  | `sysinfo/hardware.py::get_available_devices()` (public, exported in `sysinfo.__init__`)                 | private → public, same module name      | Body is byte-equivalent (try/except NPU.get_all → GPU.get_all → append "cpu"). The lazy `from .hardware import NPU/GPU` imports are no longer needed since the function lives in `hardware.py` itself. |
| `_get_available_eps() -> frozenset[str]`                 | `session/ep_registry.py::available_eps()` (public)                                                      | private → public                        | Body merges `WinMLEPRegistry.instance().get_available_eps()` with `ort.get_available_providers()`; same caching. |
| `resolve_device(device="auto") -> tuple[str, list[str]]` | Split: `session/ep_device.py::auto_detect_device() -> str` for the string-only path; `session/ep_device.py::resolve_device(ep, device) -> EPDevice` for the typed path | hard break, no signature-preserving shim | Callers must pick the right replacement per use case. |

## Behavior / contract changes
- **The bare name `resolve_device` is reused for a fundamentally different function.** Before: `sysinfo.resolve_device(device: str = "auto") -> tuple[str, list[str]]` — pure resolution, no DLL loads, only `ValueError` on failure. After: `session.resolve_device(ep=None, device=None) -> EPDevice` — typed resolver, performs deduction, **registers an EP plugin** via `WinMLEPRegistry` as a side effect, raises one of `EPNotDiscovered | EPRegistrationFailed | DeviceNotFound | ValueError` (the a509a67 `AmbiguousMatch` is removed in this commit).
- **OpenVINO multi-device handling changed.** Old code stored `"npu/gpu/cpu"` as a single value and excluded it from the inverse map. The new catalog encodes OpenVINO as separate entries per category, so OpenVINO *is* a candidate EP for all three device categories. For auto-resolution on a host with only OpenVINO installed, the resolved device may differ.
- **Hardcoded EP list expanded.** Old `_EP_DEVICE_MAP` had 8 entries; new catalog has ~13 (cuda, tensorrt added per the commit body; NvTensorRtRtx casing fixed; bundled CPU/Dml/Azure built-ins added).
- **No `get_ep_device_map` replacement exposed.** Tools/tests that called the public function must traverse the catalog. Commit body: "do not import private symbols (`_EP_TO_DEVICE`, `_DEVICE_TO_PROVIDER`, `_SHORT_TO_FULL`) outside `session/ep_device.py` — use the session facade and public helpers."
- **Module-level FIXME removed.** The long comment explaining *why* `_EP_DEVICE_MAP` had to be hardcoded (PyPI ORT lacks `get_ep_devices()` until Windows ML build lands on Win 11 25H2+) is gone. The constraint applies to the new catalog but the rationale is no longer in the source tree.

## Cross-file impact
- Every importer of `sysinfo.device` is broken. Migrated call sites live in `commands/perf.py`, `commands/eval.py`, `commands/config.py`, `commands/sys.py`, `config/build.py`, `config/precision.py`, `utils/constants.py`, `utils/cli.py`.
- `utils/constants.py::_get_supported_eps` used to call `from ..sysinfo.device import get_ep_device_map`. That function is rewritten in this commit to delegate to `session.expand_ep_name` (see `utils__constants.md`).
- Any commit-period grep `git grep "from ..sysinfo.device"` should return zero hits post-commit.

## Risks / subtleties
- A reviewer doing a blind regex `s/sysinfo.resolve_device/session.resolve_device/` will get code that imports successfully but call sites that pass `resolve_device("auto")` will be interpreted as `resolve_device(ep="auto")` — hitting `expand_ep_name("auto")` and misbehaving. Such sites must route to `session.auto_detect_device()` instead.
- Loss of the FIXME comment about ORT API limitations is a documentation regression — re-stating on the `EPDeviceSpec` catalog would preserve institutional memory.
- `lru_cache(maxsize=1)` on `_get_available_eps` was important because EP discovery touches DLL load paths. Carried over to `session/ep_registry.py::available_eps()`; if a test or runtime path re-binds `WinMLEPRegistry`, the stale cache could mask the patch. This was implicit before; risk surface unchanged, location moved.
- The transitive removal of the `"/"`-encoded multi-device value for OpenVINO is a *behavior* change, not a pure refactor. Any caller that filtered EPs by `if "/" not in device` is silently incorrect.

## Simplification opportunities
- The file is gone; the simplification was the deletion itself. The follow-up would be to restore the FIXME context on the new catalog as a docstring on `EP_DEVICE_SPECS`.
- Confirm `session/ep_registry.py::available_eps()` shares the same cache contract; if the LRU cache misses on legitimate test resets, expose an `available_eps.cache_clear()` shortcut in the session facade.

## Open questions / TODOs surfaced
- Is `get_ep_device_map` truly unused externally? If any out-of-tree code (docs, examples, downstream consumers) called it, the absence of a replacement is a public-API break with no migration path. Worth a CHANGELOG callout.
- Should the FIXME (hardcoded EP list, blocked on ORT PyPI exposing `get_ep_devices()`) be restored as a docstring on `EPDeviceSpec` or `EP_DEVICE_SPECS` in `session/ep_device.py`?
