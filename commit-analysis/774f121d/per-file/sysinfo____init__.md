# src/winml/modelkit/sysinfo/__init__.py

## TL;DR
Identical change to the a509a67 commit: the public API of `sysinfo/` retracts from EP-aware routing (`get_ep_device_map`, `resolve_device`) to hardware-only inspection (`get_available_devices`). EP/device routing has migrated to `session/ep_device.py` — the `sysinfo` package no longer knows about ORT execution providers. Net `-1` line.

## Diff metrics
- Lines: +2 / -3 (net -1)
- Hunks: 2 (one import line; one `__all__` block)
- Re-exports: removed `get_ep_device_map` and `resolve_device`; added `get_available_devices`. `CPU`, `GPU`, `NPU`, `OS`, `SysInfo` unchanged.

## Role before vs after
- Before: dual-purpose facade — hardware inventory **plus** EP-aware routing. `get_ep_device_map()` returned the manual EP→device dict; `resolve_device(device)` returned `(chosen_device, available_list)` after cross-referencing EP availability.
- After: pure hardware inventory. The only non-hardware-class symbol surfaced is `get_available_devices`, which returns the prioritized list of device categories present on the host (NPU > GPU > CPU). No EP cross-check, no ORT touch.

## Symbol-level changes
- Removed: `from .device import get_ep_device_map, resolve_device` (the file `sysinfo/device.py` is deleted in this commit).
- Added: `get_available_devices` to the existing `from .hardware import CPU, GPU, NPU` line.
- `__all__`: removed `"get_ep_device_map"`, `"resolve_device"`; added `"get_available_devices"`. Order of remaining entries unchanged.

## Behavior / contract changes
- `from winml.modelkit.sysinfo import resolve_device` → `ImportError`. The bare name `resolve_device` is reused in `winml.modelkit.session` but with a fundamentally different signature: `resolve_device(ep, device) -> EPDevice` (typed, runs deduction, side-effecting EP registration) vs. the old `resolve_device(device="auto") -> tuple[str, list[str]]` (string-only, pure). Callers who only want the auto-picked category string use `winml.modelkit.session.auto_detect_device()`.
- `from winml.modelkit.sysinfo import get_ep_device_map` → `ImportError`. No public replacement at this layer. EP↔device routing now goes through the `EPDeviceSpec` catalog inside `session/ep_device.py`; consumers should use `default_device_for_ep(ep_full)` or iterate `eps_for_device(category)`.
- `get_available_devices()` keeps the priority order (NPU > GPU > CPU, "cpu" always last) of the old private `_get_available_devices`, but **no EP cross-check** — it reports physical hardware presence only. The EP cross-check now lives in `session.auto_detect_device()`.

## Cross-file impact
- Consumers of `sysinfo.resolve_device` (old): all migrated. Verified no remaining `from ..sysinfo import resolve_device` in `src/`.
- Consumers of `get_ep_device_map`: none remain — the manual dict is replaced by the `EPDeviceSpec` catalog.
- Consumers of `get_available_devices`: only `session/ep_device.py` (used inside `auto_detect_device`).
- The package no longer logs the "No execution providers detected" warning — that was tied to `_get_available_eps`, now in `session/ep_registry.py`.

## Risks / subtleties
- **Bare-name `resolve_device` ambiguity persists across packages.** `from ..sysinfo import resolve_device` is the legacy (now broken) import; `from ..session import resolve_device` returns `EPDevice`; `from ..session import auto_detect_device` returns the category string. A reviewer doing a regex import-fix could pick the wrong replacement. The migration is documented in the commit body but requires careful per-site choice.
- `__all__` no longer exposes any EP-aware symbol from `sysinfo`. Tools that introspected `sysinfo.__all__` to discover EP mappings see nothing useful now.
- Hard break, no deprecation shim. Per the commit body's "Option A" stance, that's deliberate. Downstream out-of-tree consumers (if any) will need to update imports manually.

## Simplification opportunities
- The pre-existing 3-line file header copyright comment could be moved to a single-line `# SPDX-License-Identifier: MIT` reference — but that's a project-wide policy question, not specific to this file.
- `get_available_devices` could be re-exported as `sysinfo.devices` (property) for ergonomic `sysinfo.devices == ["npu","cpu"]` syntax. Marginal; current shape is fine.

## Open questions / TODOs surfaced
- Should the package re-export `WindowsAppRuntimeVersion` from `sysinfo.sysinfo` ever again? It was deleted in this commit (see `sysinfo__sysinfo.md`), so the question is moot — but worth confirming no external consumer needed it.
- `commands/sys.py` and other call sites should be audited for stale comments referencing `_get_available_devices` (the old private name).
