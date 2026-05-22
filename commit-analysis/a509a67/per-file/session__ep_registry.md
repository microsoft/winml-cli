# src/winml/modelkit/session/ep_registry.py

## TL;DR
`WinMLEPRegistry` gains a new `register_ep(ep_name)` method that performs **additive, selective** EP registration with two key defenses — short-circuit if another singleton already loaded the same DLL (avoiding ORT's non-idempotent native `exit(127)` crash), and fallback to `ort.get_ep_devices()` for bundled EPs like CPU/DML that aren't in the WinML catalog. The module also gains a process-cached `available_eps()` aggregator and an `ensure_initialized()` cycle-breaker entry point used by `QNNMonitor.is_available`. Log noise on registration failures is upgraded from DEBUG to WARNING (NFR-2).

## Diff metrics
- Lines added: 153
- Lines removed: 4
- Modified

## Role before vs after
- **Before:** Singleton catalog of WinML-discovered EPs with a single bulk `register_to_ort()` entry point and helpers `get_available_eps`/`get_registered_eps`/`get_ep_library_path`/`is_ep_available`. Failure modes were logged at DEBUG and not surfaced. No per-EP introspection of why registration failed. No module-level idempotent initializer.
- **After:** Same singleton plus three new capabilities: (1) per-EP selective `register_ep()` returning the claimed `OrtEpDevice` list, with defensive guard against ORT's non-idempotent native registration; (2) `available_eps()` lru-cached union of WinML + ORT-known providers; (3) `ensure_initialized()` import-cycle-safe entry point. Per-EP failure messages are exposed via the `registration_failures` property and logs are uniformly WARNING with exception-class prefix.

## Symbol-level changes
- **`WinMLEPRegistry.__init__`** — signature-unchanged; gains `self._registration_failures: dict[str, str] = {}` (line 57).
- **`WinMLEPRegistry._discover_eps`** — log level on the generic `except Exception` branch upgraded from previous wording to `"WinML EP discovery failed (%s: %s)"` including exception class name (lines 74–78).
- **`WinMLEPRegistry._fix_winrt_runtime`** — DEBUG log on failure upgraded to WARNING with exception-class prefix (lines 90–93). Comment cites NFR-2.
- **`WinMLEPRegistry.register_to_ort`** — minor changes: on success, clears any prior entry from `self._registration_failures` for that EP (line 140); on failure, records `f"{type(e).__name__}: {e}"` into `self._registration_failures[name]` and logs at WARNING (lines 142–147). Returned list shape unchanged.
- **`WinMLEPRegistry.register_ep(ep_name) -> list[ort.OrtEpDevice]`** — **added** (lines 151–211). New selective-registration entry. Three behavior paths:
  1. **Catalog hit, not yet registered** (lines 173–197): defensively probe `ort.get_ep_devices()` for the EP name; if found, treat as "already loaded by another singleton", append to `_registered_eps` without re-loading the DLL. Otherwise call `ort.register_execution_provider_library(ep_name, dll_path)`. Wrap any exception in `EPRegistrationFailed(...) from exc`.
  2. **Catalog hit, already registered** (line 198): return current device list filtered by `ep_name`.
  3. **Catalog miss** (lines 200–211): query `ort.get_ep_devices()` for bundled EPs (CPU, DML) and return if present; otherwise raise `EPNotDiscovered` with full diagnostic (catalog contents, MODELKIT_EP_PATH hint).
- **`WinMLEPRegistry.registration_failures`** — added `@property` (lines 234–242). Returns a defensive copy of the failure dict.
- **`WinMLEPRegistry.__del__`** — unchanged.
- **`WinMLEPRegistry.get_instance`** — unchanged.
- **`available_eps() -> frozenset[str]`** — **added** module-level function (lines 258–287), `@functools.lru_cache(maxsize=1)`. Unions WinML registry's `get_available_eps().keys()` with `ort.get_available_providers()`. Catches `ImportError` and `RuntimeError` silently for known-absent backends; logs at WARNING for any other exception via `exc_info=True`.
- **`get_ort_available_providers`** — body change at lines 313–317: the `except Exception` branch's log was upgraded from DEBUG to WARNING with exception-class prefix (the original `logger.debug("WinML discovery skipped: %s", e)` became `logger.warning("WinML discovery skipped (%s: %s)", type(e).__name__, e)`).
- **`ensure_initialized()`** — **added** module-level function (lines 322–347). Idempotent wrapper around `WinMLEPRegistry.get_instance().register_to_ort()`. Used by `QNNMonitor.is_available` per its docstring (line 326) to break a latent import cycle (`monitor → session.WinMLSession → ep_registry`). No latch on failure — retries on subsequent calls.

## Behavior / contract changes
- **`register_ep()` is the new public selective-registration path** used by `resolve_device(ep, device)` in `ep_device.py` (line 434 in that file). Bundled EPs (CPU, DML) flow through the catalog-miss branch and are still returned successfully — they don't need DLL registration.
- **Defensive guard against ORT native crash** (lines 175–187): if any other code (e.g. `winml.py:WinML` singleton, per commit body) has already called `ort.register_execution_provider_library` for an EP, this method now detects that via `ort.get_ep_devices()` and skips the second DLL load. The commit body confirms the bug: "a second registration of the same DLL calls `exit(127)` with no Python traceback" / `STATUS_DLL_NOT_FOUND`. Per commit body, this is described as "the patch fix"; singleton consolidation is deferred (issue I1).
- **`EPRegistrationFailed` is raised with chained cause** (lines 192–196). Callers can use `__cause__` to inspect the original ORT exception.
- **`EPNotDiscovered` message contains the full catalog list** (line 207–211). The hint mentions `MODELKIT_EP_PATH` — this env var is referenced but its handling is not in this file.
- **`available_eps()` is process-cached.** No invalidation API. After the first call, dynamic plugin install (e.g. via `set_provider_path`) is invisible to this aggregator until the process restarts. Hardware/EP set is assumed stable per the docstring.
- **`register_to_ort()` now records per-EP failures** in `self._registration_failures`. The dict is cleared per-EP on subsequent successful re-registration but never bulk-cleared. The `registration_failures` property returns a defensive `.copy()`.
- **Log-level uplift to WARNING** in three places (`_discover_eps`, `_fix_winrt_runtime`, `get_ort_available_providers`) and one new (`register_to_ort` failures) per NFR-2 — environment failures are no longer silent. CI logs will now show these as warnings.
- **`ensure_initialized()` swallows all exceptions** after logging at WARNING. Callers cannot distinguish "WinML registration failed" from "WinML not available on this OS" by return value — only by the WARNING log being present.

## Cross-file impact
- **Imports added:** `functools`, `import onnxruntime as ort` at module level (was previously local imports inside methods), `from .ep_device import EPNotDiscovered, EPRegistrationFailed`.
- **Imports removed:** none.
- **Public API exported via `__init__.py`:** `WinMLEPRegistry` (existed before), `available_eps` (new this commit). `ensure_initialized` is **not** re-exported from `__init__.py` per the post-state of that file (lines 29 of `__init__.py`).
- **Modules that now depend on this file:** `ep_device.py` consumes `WinMLEPRegistry` via lazy import (its `_get_ep_registry()` helper) and calls `available_eps` from inside `auto_detect_device()` / the `resolve_device()` auto-detect branch. `QNNMonitor` calls `ensure_initialized` per its docstring (`monitor/qnn_monitor.py` per commit body).
- **Modules this file now depends on:** `ep_device` (for the two exception types). This creates a directed edge `ep_registry → ep_device`, which is why `ep_device.py` uses lazy import in the reverse direction (its module-level `WinMLEPRegistry: Any = None` sentinel + `_get_ep_registry()` helper).

## Risks / subtleties
- **The `already_loaded` probe (line 181) is the documented defense against a process-killing native bug**, not a stylistic redundancy. Removing it re-introduces the `exit(127)` / `STATUS_DLL_NOT_FOUND` crash. The comment at lines 175–180 is load-bearing.
- **`register_ep` does NOT canonicalize the input name** (docstring line 154–156): "Callers must pass `canonicalize_ep_name(...)` on user-supplied names first." This is enforced by convention — there's no assertion or normalization. A caller passing `"nvtensorrtrtxexecutionprovider"` (lowercase) bypasses the alias table and gets `EPNotDiscovered`.
- **`available_eps` and `ensure_initialized` both swallow exceptions** but log at different levels: `available_eps` catches `ImportError`/`RuntimeError` silently and only WARNs on other exceptions; `ensure_initialized` always WARNs. Inconsistent — `ensure_initialized` may produce false-positive WARNings on non-Windows hosts.
- **The new `import onnxruntime as ort` at line 17** (module level) replaces the previous-state's local imports inside `register_to_ort` and `get_ort_available_providers`. If ORT is uninstallable but the package is imported anyway, the entire module now fails at import — previously this was deferred. Verify this is acceptable.
- **`_registration_failures.pop(name, None)` is called only inside `register_to_ort`** (line 140), not in the new `register_ep` (whose success path doesn't touch the dict). Re-registering an EP via `register_ep` does not clear a previously-recorded failure from `register_to_ort`. Callers reading `registration_failures` may see stale entries.
- **`available_eps()` returns short OR full names depending on source:** WinML's `get_available_eps().keys()` returns whatever names WinML reports (typically full canonical e.g. `QNNExecutionProvider`); `ort.get_available_providers()` returns canonical full names too. So the cache yields full names — but this contract isn't documented in the function and a future change to either source could change name form silently.
- **`functools.lru_cache(maxsize=1)` with `available_eps()` has no args** — the cache is shared across all callers but indexed by zero arguments. The single cached value persists for the process lifetime; `available_eps.cache_clear()` is the only way to invalidate.

## Open questions / TODOs surfaced
- **Singleton consolidation deferred** (commit body issue I1). Both `WinMLEPRegistry` and `winml.py:WinML` call `ort.register_execution_provider_library` against the same DLLs. The defensive guard here is the patch fix. The proper fix — collapsing the two singletons into one — is explicitly out of scope for this commit.
- **`MODELKIT_EP_PATH` env var is mentioned in error text** (line 210) but its discovery/registration is not in this file. Whether it's implemented or aspirational is not visible from this diff.
- **`available_eps()` cache invalidation policy is undocumented.** Test code that wants to probe both pre- and post-registration states must call `available_eps.cache_clear()` between assertions; the docstring at lines 260–263 says hardware/EPs don't change during a process lifetime but doesn't acknowledge that registration *does* change the set.
- **`register_ep` does not surface `EPRegistrationFailed` into `self._registration_failures`** — the failure flows up as a raised exception, not as a recorded dict entry. The asymmetry with `register_to_ort` (which records failures into the dict) may confuse callers reading `registration_failures` for diagnostics.
