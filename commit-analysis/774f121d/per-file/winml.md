# src/winml/modelkit/winml.py

## TL;DR
The legacy `WinML` singleton and its two helpers (`register_execution_providers`, `add_ep_for_device`) are formally **deprecated** in this commit. The architectural change is twofold: (1) `WinML.__init__` no longer talks to the Windows AppSDK / `winui3.microsoft.windows.ai.machinelearning.ExecutionProviderCatalog` directly — it now walks `discover_all_eps()` from `ep_path.py` and caches one entry per EP. (2) The whole module sprouts `DeprecationWarning`s, the `_fix_winrt_runtime` hack and the `__del__`-driven AppSDK lifecycle are gone, and the docstrings now point callers at `winml.modelkit.session` (`WinMLEPRegistry`, `EPDeviceTarget`, `resolve_device`). The defensive `get_ep_devices()` idempotency guard (introduced for the a509a67 patch) is preserved. A new `extra_sources: list[EPSource] | None = None` kwarg threads through both registration paths and bypasses the in-process EP cache when set, so override registrations win. Net `+248 / -135`. The file is now positioned as an explicit "thin compatibility surface for the in-tree `analyze/*` callers that have not yet migrated".

## Diff metrics
- Lines: +248 / -135 (net +113)
- Hunks: 7 (module docstring, imports, `WinML` class doc, `__init__`, removal of `__del__` + `_fix_winrt_runtime`, `register_execution_providers` rewrite, `add_ep_for_device` docstring + warning, new `__all__`)
- Symbols added: module-level `_DEPRECATION_MSG` constant, module-level `logger`, `__all__` list.
- Symbols removed: `WinML.__del__`, `WinML._fix_winrt_runtime`, `WinML._providers`, `WinML._win_app_sdk_handle`.

## Role before vs after
- Before: this module was the only entry point for "wire up Windows AppSDK execution providers into ORT/ORTGenAI". It owned the AppSDK initialization handle, called `catalog.find_all_providers()`, walked the providers, and registered each DLL into ORT. It was the original (pre-`session/`) location for plugin-EP knowledge.
- After: it's a deprecation surface that defers all discovery to `ep_path.discover_all_eps()` and explicitly delegates to the `session/` package for the new path. The module docstring now reads "DEPRECATED legacy entry points for plugin-EP bulk registration" and emits `DeprecationWarning` on every entry point. The AppSDK lifecycle (`initialize(...)`, `__exit__` on teardown) is no longer here — that's now `session/winml_handle.py` (per the design docs). The `_fix_winrt_runtime` hack that deleted `msvcp140.dll` from the `winrt-runtime` site-packages is gone — also moved into the session package (or removed entirely; verify in `session/winml_handle.py` if it still applies).

## Symbol-level changes
- **Module docstring (lines 5–24)**: new. Documents the deprecation, lists the migration mapping for each public symbol, and points at `docs/design/session/2_coreloop.md` for the Path A / Path B canonical flows.
- **Imports**:
  - Added: `from __future__ import annotations`, `import logging`, `import warnings`, `TYPE_CHECKING`, the `Path` import is now under `TYPE_CHECKING` (no longer needed at runtime), and `from .ep_path import EPSource, discover_all_eps`.
  - Removed: `import traceback`, top-level `from pathlib import Path`.
  - Newly relocated: `Path` to a `TYPE_CHECKING` guard (only used in type hints now).
- **Module constants**:
  - Added: `logger = logging.getLogger(__name__)` (unused but conventional).
  - Added: `_DEPRECATION_MSG` constant centralizing the deprecation text (used by all three `warnings.warn(...)` call sites).
- **`WinML.__init__`** rewritten:
  - Old: called `_fix_winrt_runtime()`, imported `winui3.microsoft.windows.ai.machinelearning`, called `winml.ExecutionProviderCatalog.get_default()`, `find_all_providers()`, walked providers calling `provider.ensure_ready_async().get()`, captured `provider.library_path` per provider.
  - New: emits a `DeprecationWarning`, then builds `self._resolved: dict[str, tuple[Path, EPSource]]` and `self._ep_paths: dict[str, str]` from `discover_all_eps()` filtered by `status == "primary"`. No AppSDK calls. The `_resolved` dict is new — keeps `(path, source)` for debugging — but only `_ep_paths` (path string only) is consumed by `register_execution_providers`.
- **`WinML.__del__`**: deleted. The AppSDK handle lifetime is no longer owned by this module.
- **`WinML._fix_winrt_runtime`**: deleted. The msvcp140.dll-deletion hack is gone from here; per the session refactor, the AppSDK initialization moved.
- **`WinML.register_execution_providers(ort, ort_genai, extra_sources=None)`** rewritten:
  - New `extra_sources: list[EPSource] | None = None` kwarg. When set, calls `discover_all_eps(extra_sources=extra_sources)` to rebuild `ep_paths` for this call only; otherwise uses the cached `self._ep_paths` from `__init__`.
  - New `skip_cache = extra_sources is not None` boolean. When True, the per-process `self._registered_eps` check is bypassed so a second call with new `extra_sources` actually re-registers. Comment explicitly states ORT's `register_execution_provider_library` is idempotent for the same `(name, path)` pair and returns the existing handle.
  - Preserves the **defensive guard** from a509a67: `module.get_ep_devices()` is consulted before each registration; if the EP is already in the live device list, append to the local tracker and skip. Guard wrapped in `try/except Exception` returning `already_loaded = False` (comment: "conservative: attempt the load").
  - Error path: kept `print(f"Failed to register execution provider {name}: {e}", file=sys.stderr)` but **removed** the trailing `traceback.print_exc()` call (and the `import traceback` it depended on). Failures are now reported as one line, no stack.
- **`register_execution_providers(ort, ort_genai, extra_sources=None)`** (module-level free function): added the new kwarg, added `warnings.warn(_DEPRECATION_MSG, ...)`, and forwards everything to `WinML().register_execution_providers(...)`.
- **`add_ep_for_device(...)`** body unchanged but docstring fully rewritten and a `warnings.warn(_DEPRECATION_MSG, ...)` added at the top. Docstring now shows the typed migration:
  ```
  target = EPDeviceTarget(ep=short_ep_name(ep_name), device=device_type.name.lower())
  resolved = resolve_device(target)
  ep_device = WinMLEPRegistry.instance().auto_device(resolved)
  session_options.add_provider_for_devices([ep_device.device._ort], ep_options or {})
  ```
- **`__all__`** added: explicit `["WinML", "add_ep_for_device", "register_execution_providers"]`. Previously the module had no `__all__` and exposed everything via `from .winml import …`.

## Behavior / contract changes
- **Source of EP knowledge changed.** Before, the AppSDK `ExecutionProviderCatalog` provided the EP list at runtime — meaning `WinML()` reflected whatever the *user's installed AppSDK build* knew about. After, the EP list comes from `discover_all_eps()` — meaning it reflects the in-tree `_DEFAULT_EP_SOURCES` plus the `WINMLCLI_EP_PATH` env var. On hosts where the AppSDK exposes an EP not in the in-tree list (or vice versa), the two paths produce different sets. The commit body frames this as the desired refactor.
- **AppSDK lifecycle no longer owned here.** Before, `WinML.__init__` called `initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI)` and `_win_app_sdk_handle.__enter__()`, then `__del__` called the matching `__exit__`. Now neither happens in this file. Callers that relied on `import winml.modelkit.winml` to *initialize the AppSDK* (e.g. for the side effect of the dynamic-dependency dialog) get nothing. The AppSDK init moved to a different module — verify which one owns it now.
- **Three `DeprecationWarning`s emitted** — at `WinML.__init__`, at the module-level `register_execution_providers`, and at `add_ep_for_device`. Each goes to whatever warning filter `_warnings.py` configures. Tests that didn't expect deprecation warnings may now error out under `-W error::DeprecationWarning`.
- **`add_ep_for_device` behavior unchanged** but docstring now describes *both* the deprecated signature *and* the silent-no-op when no `OrtEpDevice` matches. Previously the silent-no-op was undocumented.
- **Failure reporting weaker.** Old code printed traceback on registration failure; new code only prints `f"Failed to register execution provider {name}: {e}"` to stderr. Debugging a registration failure now requires re-running with logging enabled (and the file has a `logger` but doesn't use it on the error path — see Simplification).
- **New `extra_sources` kwarg behavior**: when set, the call **does** re-register even if the EP-name is already in `self._registered_eps[module.__name__]`. The comment justifies this as "ORT's register_execution_provider_library is idempotent for the same (name, path) pair and returns the existing handle; re-calling with a different path replaces the registration". External callers passing `extra_sources` must accept this re-binding behavior.
- **Idempotency / replacement asymmetry.** With `extra_sources=None` (default), the in-process cache prevents double-registration. With `extra_sources=[...]`, the cache is bypassed *and* the live-state guard still applies. So calling once with `extra_sources=[A]` then once with `extra_sources=[B]` will: (a) bypass cache, (b) probe live state — and if A registered the EP under the same name, the live-state check returns True and skips B. This is silently broken for the "replace-with-different-path" use case the comment claims to support. Confirm against a unit test.

## Cross-file impact
- New dependency on `ep_path.py` (`EPSource`, `discover_all_eps`) — moves `winml.py` downstream of the ep_path module in the import DAG.
- The AppSDK init code (`InitializeOptions.ON_NO_MATCH_SHOW_UI`, `_fix_winrt_runtime`, etc.) was moved out of this file. The destination is the session package — likely `session/winml_handle.py` or similar (verify; not in this batch).
- Direct importers of `winml.modelkit.winml`:
  - `src/winml/modelkit/inference/engine.py:922` — does `from ..models.winml import get_winml_class`. Different module (`models/winml`), unaffected.
  - `src/winml/modelkit/models/__init__.py:35` and `models/auto.py:36` and `models/hf/qwen.py:106` — all import from `models/winml/`, not the top-level `winml.py`. Unaffected.
  - Tests directly exercising this file: `tests/unit/winml/test_winml.py`, `tests/unit/winml/test_winml_deprecation.py`, `tests/unit/ep_path/test_add_ep_for_device.py`, `tests/unit/ep_path/test_register_execution_providers.py`.
- The `__all__` list is now load-bearing for `from winml.modelkit.winml import *` consumers — three names, no more.
- The `winml.modelkit.winml` deprecation pushes external consumers to `winml.modelkit.session` — the `analyze/*` subtree is the one in-tree consumer that hasn't migrated per the module docstring.

## Risks / subtleties
- **AppSDK lifecycle handoff.** The pre-existing `__del__` on `WinML` called `_win_app_sdk_handle.__exit__(None, None, None)`. Removing it means the handle's `__exit__` must now be invoked by whoever owns the handle in the session package. If the handoff missed a cleanup call, process teardown could leak the WinAppSDK initialization. Not a functional bug in steady state but visible under leak-tracking tools.
- **`_fix_winrt_runtime` deletion.** That helper deleted `winrt-runtime/winrt/msvcp140.dll` to avoid a DLL conflict. If the session package didn't carry over the equivalent fix, the conflict re-surfaces on hosts where another package also ships `msvcp140.dll`. The commit body doesn't mention this — worth a follow-up to confirm the new location handles it (or that the underlying conflict is no longer reachable).
- **Deprecation noise leakage.** Three `warnings.warn(_DEPRECATION_MSG)` calls fire on every entry point. The `analyze/*` codepath still imports this — meaning users running `winml-modelkit analyze ...` see deprecation warnings repeatedly per process. If `_warnings.py`'s filter doesn't dedupe these (DeprecationWarning is dedup'd by default at the simplefilter level but not at `warnings.warn` when stacklevel changes), output may be noisy.
- **`extra_sources` cache-bypass and live-state guard interact.** As noted above, the live-state guard skips re-registration even with `skip_cache=True` if the EP is already loaded — defeating the purpose of `extra_sources` when the second source's DLL has the same `ep_name` but a different path. There's no `force=True` escape hatch. Probably acceptable today (no in-tree call site overrides paths) but worth a docstring caveat.
- **`logger` is defined but unused.** All log surfaces use `print(..., file=sys.stderr)` or implicit silence. Replacing the print with `logger.error(...)` would be a one-line fix that gives diagnostics a proper toggle.
- **Stale comment in the live-state probe.** The comment refers to "STATUS_DLL_NOT_FOUND / 0xC000026F" — but the underlying register_execution_provider_library is now wrapped by both this module and `session/ep_registry.py`, both of which carry symmetric guards. If a future refactor consolidates them, this comment will read as a reference to a no-longer-relevant failure mode.
- **`status == "primary"` filter is implicit.** `__init__` only captures primary entries from `discover_all_eps()` — shadowed entries (e.g. a `--ep-path` override of a built-in) are silently dropped. The old code processed every provider the AppSDK returned without status filtering. If a downstream caller depended on registering shadowed-but-not-primary plugins, they'd need to set `extra_sources` explicitly.

## Simplification opportunities
- **Module is now thin enough to deprecate the file outright.** Three deprecated symbols + a singleton wrapper. Migrating the in-tree `analyze/*` callers (the only stated reason this module exists) would let the whole file be deleted. Worth opening a follow-up issue: "delete `winml.py` once `analyze/*` migrates".
- **The `_resolved: dict[str, tuple[Path, EPSource]]` cache is built but only `_ep_paths` is read.** Drop `_resolved` — it's dead state on the singleton. (Comment claims it's for debugging; if so, lift to a `@property` so the cost is paid only when introspected.)
- **`logger` is unused.** Either drop it or replace the stderr print with `logger.exception(...)` — the latter actually carries the traceback that was lost when `traceback.print_exc()` was removed.
- **Three `warnings.warn(_DEPRECATION_MSG, ...)` call sites duplicate the same pattern.** A `@deprecated(_DEPRECATION_MSG)` decorator (Python 3.13 has `typing.deprecated`; can be backported) would compress the boilerplate.
- **`__all__` is duplicated by the migration-mapping in the module docstring.** Could use `__all__` to auto-generate the docstring list at build time, but probably not worth it.
- **The `skip_cache` logic in `register_execution_providers` could be a helper:** `def _should_skip_for_module(name, module, skip_cache)` returning bool. Currently inlined and entangled with the live-state probe.
- **`_DEPRECATION_MSG` could include a `since="…"` field** (e.g. version string) so tools that parse deprecation warnings know when migration began. Today it's free-form text.

## Open questions / TODOs surfaced
- Where did `_fix_winrt_runtime` go? If the msvcp140.dll conflict is still reachable, it needs a home. Trace through `session/` to confirm.
- Where does the AppSDK `initialize(InitializeOptions.ON_NO_MATCH_SHOW_UI)` lifecycle live now? The lack of an `__exit__` call here suggests another module owns it, but if no module owns it, the bootstrap dialog is gone and CI hosts without AppSDK installed silently fail somewhere downstream.
- Should `extra_sources` retire the live-state probe (or force a real `register_execution_provider_library` call regardless)? The current "skip_cache=True but still respect live-state" is half-broken for the "replace registration" use case. Pick one semantic and document it.
- The `analyze/*` migration is the gating item for deleting this file. What's the planned timeline? The commit body doesn't say.
- Should the test suite assert `WinML.__init__` no longer touches the AppSDK? Today the asymmetry between "this module deprecated" and "AppSDK still touched somewhere" is implicit.
