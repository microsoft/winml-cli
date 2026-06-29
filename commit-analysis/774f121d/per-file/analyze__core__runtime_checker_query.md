# src/winml/modelkit/analyze/core/runtime_checker_query.py

## TL;DR
The substantive change in this batch. `_is_ep_available_locally` is rewritten to probe
availability through the new `WinMLEPRegistry.auto_device(resolved)` path instead of the old
`winml.register_execution_providers(ort=True)` + `ort.get_ep_devices()` direct walk. The
module-top `import onnxruntime as ort` is deleted because the function no longer needs raw ORT
access; the new path imports six public symbols from `..session` lazily — `DeviceNotFound`,
`EPDeviceTarget`, `WinMLEPNotDiscovered`, `WinMLEPRegistrationFailed`, `WinMLEPRegistry`,
`resolve_device`, `short_ep_name`. The behavioural contract is preserved: the method still
returns `bool`, still caches the answer in `self._ep_available_locally`, still swallows
"unavailable" outcomes as `False`. What changes is which APIs we walk to get the answer, and
how the negative cases are taxonomised (typed session exceptions replace a broad `except
Exception`).

## Diff metrics
- 28 insertions / 18 deletions (net +10) — the bulk of this batch's churn.
- Single hunk in `_is_ep_available_locally` (lines 1285–1328) plus the module-top
  `import onnxruntime as ort` deletion (line 18).
- New lazy import block of 7 symbols from `...session`.
- Two deleted import lines (`from ... import winml`, `from ...utils.constants import
  DEVICE_TO_DEVICE_TYPE`) — the latter still appears in the file at line 1337 in a
  *different* method (`_get_ep_checker`), so the import is only removed from this one
  call site, not from the file.
- A new docstring paragraph added to `_is_ep_available_locally` explaining the new probe
  strategy.

## Role before vs after
**Before.** The probe asked ORT directly:
1. Force-register every plugin via `winml.register_execution_providers(ort=True)` (a
   side-effecting global call that triggered the analyze-loop double-registration crash
   addressed by the L1/L2 split elsewhere in the codebase).
2. Convert `self.device_type` to an ORT `OrtHardwareDeviceType` enum via the legacy
   `DEVICE_TO_DEVICE_TYPE` mapping in `utils.constants`.
3. Walk `ort.get_ep_devices()` and return `True` iff any tuple matched `(ep_name,
   device.type)`.
4. Any exception → log at DEBUG + return `False`.

**After.** The probe asks the session registry through its canonical entry point:
1. Build an `EPDeviceTarget(ep=short_ep_name(self.ep_name), device=self.device_type.lower())`.
2. Run it through `resolve_device(target)` to fill any `"auto"` axes (defensive — call sites
   typically pass concrete `ep`/`device_type`, so this is a no-op in practice).
3. Call `WinMLEPRegistry.instance().auto_device(resolved)` — which lazily discovers, registers,
   and selects the WinMLEP/device pair. Success → set `_ep_available_locally = True`.
4. Catch the four typed failure modes (`WinMLEPNotDiscovered`, `WinMLEPRegistrationFailed`,
   `DeviceNotFound`, `ValueError`) → log + set `_ep_available_locally = False`.

The rewrite aligns this site with the v2.9 architecture's "Registry is the only entry point;
plugin discovery + DLL load is lazy and idempotent" invariant. The old `register_execution_providers(ort=True)`
side-effect call is exactly the path that produced the analyze-crash diagnostic; eliminating
it removes one of the two callers of that pre-register pass.

## Symbol-level changes
- Module-top: `import onnxruntime as ort` (line 18, pre-commit) — **deleted**. No other
  `ort.` reference survives in the file; the import is now truly orphan-free.
- `RuntimeCheckerQuery._is_ep_available_locally`:
  - Docstring expanded with a paragraph describing the targeted probe and the negative-outcome
    contract.
  - Lazy import block changed from `from ... import winml` + `from ...utils.constants import
    DEVICE_TO_DEVICE_TYPE` to a 7-symbol tuple import from `...session`.
  - `winml.register_execution_providers(ort=True)` call → **removed**.
  - `device_type_enum = DEVICE_TO_DEVICE_TYPE.get(self.device_type)` early-out → **removed**
    (replaced by `device=self.device_type.lower()` inside the `EPDeviceTarget` constructor,
    which validates via the catalog and raises if the device is unknown).
  - `ort.get_ep_devices()` + `any(...)` membership test → **removed**.
  - New construction: `EPDeviceTarget(ep=short_ep_name(self.ep_name),
    device=self.device_type.lower())` followed by `resolve_device(target)` and
    `WinMLEPRegistry.instance().auto_device(resolved)`.
  - `except Exception as e:` → narrowed to `except (WinMLEPNotDiscovered,
    WinMLEPRegistrationFailed, DeviceNotFound, ValueError) as e:`.
  - Log message string changed from "Failed to query EP devices: %s" to "EP %s on %s not
    available locally: %s" — now reports the specific EP/device pair that failed instead of a
    generic ORT failure.

No other symbol in the file is touched. `_get_ep_checker` (which still uses
`DEVICE_TO_DEVICE_TYPE` at line 1337) is intentionally left on the legacy mapping — the
migration here is surgical to the availability-probe path only.

## Behavior / contract changes
- **Side-effect surface shrinks dramatically.** The pre-commit path force-registered every
  plugin EP (`winml.register_execution_providers(ort=True)`) as a probe side-effect. The new
  path only registers the *one* EP the caller asked about, via the registry's lazy load. A
  probe for QNN no longer drags OpenVINO/VitisAI DLLs into the process.
- **Exception taxonomy is now typed.** Callers that subclass or monkey-patch the probe to
  surface "why is the EP unavailable?" now get a typed exception name (`WinMLEPNotDiscovered`
  vs `WinMLEPRegistrationFailed` vs `DeviceNotFound`) in the DEBUG log line instead of a bare
  ORT `Exception`. `ValueError` is included in the catch tuple because `EPDeviceTarget`'s
  `__post_init__` validates `ep` against `VALID_EPS` and `device` against `VALID_DEVICES` and
  raises `ValueError` on miss — i.e. an unknown EP/device string is treated as "not available
  locally" (same outcome as before, where the pre-commit `DEVICE_TO_DEVICE_TYPE.get` returning
  `None` short-circuited to `False`).
- **device_type.lower() casing dependency.** The pre-commit lookup `DEVICE_TO_DEVICE_TYPE.get(self.device_type)`
  was case-sensitive against the exact keys in `utils.constants`. The new path passes
  `self.device_type.lower()` to `EPDeviceTarget`, which expects the canonical lowercase
  `"cpu"|"gpu"|"npu"`. Any caller storing `self.device_type` as `"CPU"` or `"NPU"` (uppercase)
  is now silently mapped to the catalog form — a subtle robustness improvement, consistent
  with the commit body's claim that `auto.py:411 passes device_type.lower() to match the
  other 3 call sites`.
- **Single-EP probe registers a real session-style EP load.** Where the old probe just
  iterated `ort.get_ep_devices()` (read-only), the new probe calls `auto_device`, which
  registers the EP through the registry. This is *not* a heavyweight `ort.InferenceSession`
  construction — it goes through `WinMLEPRegistry.register_ep` which the v2.9 commit made
  idempotent (dll_path cache hits return the cached `WinMLEP` instead of raising). So
  repeated probes on the same RuntimeCheckerQuery instance are still cheap (early return on
  cached `_ep_available_locally`), and even across instances the second register_ep is a
  cache hit.
- **Caller-observable contract on return value unchanged.** Still `bool`; still memoised in
  `self._ep_available_locally`; still consumed by `_get_ep_checker` via the `if not
  self._is_ep_available_locally(): ...` guard at line 1512.

## Cross-file impact
- New import edge: `analyze/core/runtime_checker_query.py` → `modelkit.session` for 7 public
  symbols. All seven are exported in `session/__init__.py`'s `__all__` (verified: lines 11,
  13, 17, 18, 26, 27, 44, 61, 62, 66, 73, 74). The "no private session symbol imports" rule
  from the design doc is satisfied.
- Removed import edges (in this method, but still elsewhere):
  - `from ... import winml` no longer needed here. Other call sites in the package
    (`analyze/runtime_checker/ep_checker.py`, `analyze/runtime_checker/check_ops.py`,
    `analyze/pattern/check_patterns.py`) still call `winml.register_execution_providers(ort=True)`
    or `winml.add_ep_for_device(...)`; the `winml` module dependency survives at the package
    level.
  - `from ...utils.constants import DEVICE_TO_DEVICE_TYPE` is only removed at this call site;
    `_get_ep_checker` (line 1337) still imports and uses it. The `DEVICE_TO_DEVICE_TYPE`
    table is therefore *not* dead — a future cleanup should consider whether
    `_get_ep_checker` could also migrate to the session catalog, but that's outside this
    commit's scope.
- The `_is_ep_available_locally()` callsite (line 1512, inside `_get_ep_checker`) is
  unchanged; the contract `() -> bool` is preserved.

## Risks / subtleties
- **`WinMLEPRegistry.auto_device(resolved)` has side effects.** Probing "is EP X available?"
  now registers EP X into the global registry on success. This means a successful probe
  changes process state (the registry's `_registered` dict). If the caller probes multiple
  EPs in sequence (e.g. across an analyze loop's `eps_to_analyze` fan-out), each probe leaks
  one EP registration. That's the intended new model per the design doc ("registration is
  cheap and idempotent; cache hits return the cached `WinMLEP`"), but it changes the
  process-startup observable footprint of a single analyze call.
- **`ValueError` is broad.** The catch clause includes `ValueError` to absorb
  `EPDeviceTarget.__post_init__` validation failures. If `short_ep_name(self.ep_name)`
  itself raises `ValueError` (e.g. caller passes a malformed EP string), that's also caught
  and reported as "not available locally" — which is the right user-facing outcome but masks
  what is arguably a programmer-error precondition. Consider asserting `self.ep_name in
  VALID_EPS` separately if the distinction matters.
- **Exception list omits `WinMLEPMonitorMismatch`.** The session package raises that during
  monitor binding. `_is_ep_available_locally` doesn't go through the monitor path
  (`auto_device` only does discovery + registration + device selection), so the omission is
  almost certainly correct — but if a future `auto_device` refactor were to grow a monitor
  step, this catch tuple would need updating.
- **The new docstring claims `auto_device` does "DLL load lazily"** — verify against the
  v2.9 `WinMLEPRegistry.register_ep` implementation. The commit body says "register_ep is
  idempotent: dll_path cache hits return the cached WinMLEP instead of raising", consistent
  with the docstring's claim. The probe is therefore safe to repeat.
- **`logger.debug` only.** Failures are still DEBUG-level (matching the pre-commit log
  level). If end-users are confused about "why is my EP marked unavailable?", they need to
  enable DEBUG to see which of the four exception types fired. Worth promoting to INFO for
  `WinMLEPRegistrationFailed` specifically — that's the "your DLL didn't load" case and
  arguably is user-actionable.

## Open questions / TODOs surfaced
- Should `_get_ep_checker` (line 1337) also migrate off `DEVICE_TO_DEVICE_TYPE` onto the
  session catalog? That would let the file drop the legacy mapping import entirely. The
  commit chose not to do this — likely because `_get_ep_checker` builds an EPChecker
  subprocess that still expects raw ORT enums.
- `EPChecker` itself (constructed by `_get_ep_checker`) still does its own
  `winml.register_execution_providers(ort=True)` call at module-import time
  (`runtime_checker/check_ops.py:41`). The probe and the actual check now use two different
  registration paths — probe goes through the new registry; check goes through the old direct
  registration. Worth reconciling in a follow-on.
- The lazy import block of 7 symbols is verbose. A `from ...session import EPDeviceTarget,
  WinMLEPRegistry, resolve_device, short_ep_name` plus a separate `(WinMLEPNotDiscovered,
  WinMLEPRegistrationFailed, DeviceNotFound)` tuple import is a candidate for module-top
  imports given that `session` is already loaded in any realistic analyze run.

## Simplification opportunities
- **Module-top the session imports.** The lazy import inside `_is_ep_available_locally` adds
  no real value — `..session` is already imported across the package (analyzer.py also imports
  `eps_for_device` from it). Moving the import to the top of the file is one line shorter
  and removes the per-call import overhead.
- **`resolve_device(target)` may be unnecessary.** Call sites always pass concrete
  `ep_name`/`device_type` strings into `RuntimeCheckerQuery`; an `EPDeviceTarget` built from
  them is already concrete, so `resolve_device` is a no-op. The defensive call is harmless
  but adds noise. Drop it if you can show `self.ep_name` and `self.device_type` are never
  `"auto"`.
- **The 4-element exception tuple can be widened to `WinMLSessionError`** if such a base class
  exists or is added. Currently each call site that probes EP availability has to know the
  full taxonomy; a base class would let new failure modes land without touching every catch.
- **The `ValueError` catch could be tightened** by validating `self.ep_name` against
  `VALID_EPS` before constructing the target — that converts a "bad input" silent miss into
  a loud programmer-error.
- **The memoisation could move to `functools.lru_cache`** on a module-level helper if
  `RuntimeCheckerQuery` instances are short-lived — though the existing
  `self._ep_available_locally` slot is fine and matches the rest of the file.
- **DEBUG → INFO for `WinMLEPRegistrationFailed`.** This failure mode is user-actionable
  ("install the EP DLL"); the current `logger.debug` hides it from typical CLI output.
