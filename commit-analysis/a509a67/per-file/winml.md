# src/winml/modelkit/winml.py

## TL;DR

Adds a defensive idempotency guard inside `WinML.register_execution_providers`: before calling ORT's `register_execution_provider_library(name, path)`, query the live ORT/genai device list and skip the registration call if any device already advertises that `ep_name`. Mitigates a hard-to-debug native crash (`exit(127)` / `STATUS_DLL_NOT_FOUND` / `0xC000026F`) that triggered when both this singleton and the new `WinMLEPRegistry` in `session/ep_registry.py` raced to register the same DLL twice in the same process. Also a one-character casing fix in `add_ep_for_device`'s docstring (`NvTensorRTRTXExecutionProvider` → `NvTensorRtRtxExecutionProvider`). The underlying double-registration problem is **not** fixed (commit body labels it I1, "Singleton consolidation deferred") — only the symptom is.

## Diff metrics

`+13 / -1`. Two hunks: 12 added lines inside `register_execution_providers` (the guard block); 1 line changed in `add_ep_for_device`'s docstring (EP-name casing fix).

## Role before vs after

Role of the module is unchanged: the top-level `WinML` singleton wraps the Windows AppSDK / WinML execution-provider catalog and exposes `register_execution_providers(ort=, ort_genai=)`. It is one of two EP-registration entry points in the codebase. The other — added in this same commit — is `session/ep_registry.py:WinMLEPRegistry`, which carries the equivalent defensive guard (per the commit body, "symmetric defensive guards across both register_execution_provider_library singletons"). Both must coexist until the deferred consolidation lands; the guard makes that coexistence safe.

The free `register_execution_providers(...)` function and the WinML-internal `_fix_winrt_runtime` helper are untouched. `add_ep_for_device`'s body is untouched ("NEVER modify this function" comment preserved); only its docstring's NvTensorRtRtx casing is fixed.

## Symbol-level changes

`WinML.register_execution_providers(self, ort, ort_genai)`:

The pre-existing loop already had two layers of guarding:

1. An in-memory tracker `self._registered_eps[module.__name__]` (list of `name`s previously registered through this singleton in this process).
2. A `try/except` around the actual `register_execution_provider_library` call, printing the traceback on failure.

The new guard sits between them — *before* the try/except but *after* the in-memory check fails:

```python
for name, path in self._ep_paths.items():
    for module in modules:
        if name not in self._registered_eps[module.__name__]:
            # Defensive guard: ORT's register_execution_provider_library is NOT
            # idempotent — a second call for the same DLL calls C++ exit(127) with
            # no Python traceback (surfaces as STATUS_DLL_NOT_FOUND / 0xC000026F).
            # WinMLEPRegistry (session/ep_registry.py) may have already registered
            # this EP in the same process.  Consult the live ORT device list first.
            try:
                already_loaded = any(d.ep_name == name for d in module.get_ep_devices())
            except Exception:
                already_loaded = False  # conservative: attempt the load
            if already_loaded:
                self._registered_eps[module.__name__].append(name)
                continue
            try:
                module.register_execution_provider_library(name, path)
                self._registered_eps[module.__name__].append(name)
            except Exception as e:
                ...
```

Key points:

- Live-state probe uses `module.get_ep_devices()` (i.e. the ORT module's own device enumeration), not a Python-side cache. This is the only way to see registrations done by another in-process singleton.
- Match key is `d.ep_name == name`, where `name` is the canonical full EP name from the WinML AppSDK catalog (`provider.name`).
- The probe is wrapped in `try/except Exception` with the comment "conservative: attempt the load". So if `get_ep_devices()` itself raises (older ORT lacking the symbol, mid-init state, etc.), the code falls through to the existing registration attempt — same behaviour as before the patch. This means the guard cannot regress callers on older runtimes that lack the device-enumeration API.
- On a positive "already loaded" detection, the EP is appended to the local tracker `self._registered_eps[module.__name__]` even though *this* singleton didn't call `register_execution_provider_library`. This is intentional: keeps the singleton's view consistent with the live state, so a third call in the same process is also a no-op.

`add_ep_for_device(...)`:

- Pure docstring edit. The "NEVER modify this function" comment is preserved.
- Docstring's `ep_name` enumeration line changed `"NvTensorRTRTXExecutionProvider"` → `"NvTensorRtRtxExecutionProvider"`. This brings the documentation into agreement with `ort.get_all_providers()` (per commit body: "NvTensorRtRtx casing bug fixed (verified via ort.get_all_providers())") and with `session/ep_device.py:_EP_NAME_ALIASES` / `canonicalize_ep_name`.

## Behavior / contract changes

- Second call to `WinML().register_execution_providers(...)` in the same process is now safe even when another component (e.g. `WinMLEPRegistry`) has already registered the same EP DLL. Before this patch, the call would terminate the entire Python process with `exit(127)` from C++ — no traceback, no Python-level catchable exception.
- The semantics of the `_registered_eps` tracker subtly changes: it now records "this EP is registered in the process" (by *some* registrant), not "this singleton registered this EP". For external consumers that read the returned `dict[str, list[str]]` to know what *this* call did, that distinction matters — the dict can now include EPs that were registered by a different singleton.
- No public signature changes. The return type, kwargs, and singleton-construction semantics of `WinML.__new__` / `__init__` are unchanged.

## Cross-file impact

- The guard's correctness depends on `session/ep_registry.py:WinMLEPRegistry` carrying the *symmetric* guard (commit body: "Symmetric defensive guards across both register_execution_provider_library singletons"). If the symmetric guard there were missing or different, the second registrant could still crash before the live-state probe could see anything.
- The guard implicitly depends on the ORT module's `get_ep_devices()` returning newly-registered EPs *immediately* after the first registrant's `register_execution_provider_library` call returns. Both singletons assume this; if ORT defers device enumeration to the next session-build, the guard would still race. Not contradicted by the commit body's verification runs.
- The NvTensorRtRtx docstring fix is the one place in this file that touches the canonical EP-name taxonomy. The catalog's `_EP_NAME_ALIASES` (in `session/ep_device.py`) is the single source for casing; this file's docstring is hand-maintained and now aligned.

## Risks / subtleties

- **Symptom-not-cause fix.** Commit body explicitly defers the singleton consolidation (issue "I1"). Two singletons still exist, with two independent in-memory caches, two independent registration codepaths, and two `register_execution_provider_library` call sites. Any third code path that bypasses both (e.g. external code that calls `ort.register_execution_provider_library(...)` directly) and then triggers `WinML().register_execution_providers(...)` is now safe — but any third path that bypasses the live-state probe (e.g. `get_ep_devices()` returns stale data due to lazy enumeration) could still crash the process natively.
- **The probe is loop-inner.** `module.get_ep_devices()` is called once per `(name, module)` pair inside a double loop. For the current `_ep_paths` cardinality (a handful) this is negligible, but if the catalog grows to dozens of EPs the cost is `O(n × m × |devices|)`. Easy follow-up: hoist the call outside the inner loop and reuse the set per module.
- **Silent fallthrough on `get_ep_devices()` failure.** `except Exception: already_loaded = False` is justified by the comment ("conservative: attempt the load"), but it means a buggy ORT module that always raises on `get_ep_devices()` would defeat the guard and re-introduce the native crash. Reasonable trade-off; worth a logger.debug at minimum.
- **`__del__` ordering.** The pre-existing `__del__` calls `self._win_app_sdk_handle.__exit__(None, None, None)`. The new guard does not interact with that, but: if process teardown invokes `__del__` while ORT is still holding registered library handles, the AppSDK exit could free a DLL that ORT thinks it owns. Not a new bug, but the guard makes double-registration safer without addressing teardown ordering.
- **Tracker drift on guarded-skip.** Appending to `self._registered_eps[module.__name__]` when the *other* singleton did the registration means the WinML singleton now reports EPs it didn't register. Any debugging code that diffs the singletons' trackers will see overlap that previously would have indicated a bug.

## Open questions / TODOs surfaced

- Singleton consolidation (commit body's "I1") — the two `register_execution_provider_library` entry points should be unified. The guard is explicitly the patch fix, not the design fix.
- The `add_ep_for_device` "NEVER modify this function" comment lists the same EP names that the catalog already enumerates structurally. Worth a `# Source: EP_DEVICE_SPECS` cross-reference comment so future maintainers know where to look when adding a new EP — otherwise this docstring will drift the next time an EP joins the catalog.
- The guard relies on `ep_name` matching between AppSDK's `provider.name` and ORT's `OrtEpDevice.ep_name`. If these ever differ in casing (e.g. AppSDK returns `nvtensorrtrtxexecutionprovider` and ORT returns `NvTensorRtRtxExecutionProvider`), the guard misses and the crash returns. Adding a `canonicalize_ep_name(...)` normalisation on both sides of the `d.ep_name == name` comparison would harden this.
