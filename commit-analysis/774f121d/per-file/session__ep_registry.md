# src/winml/modelkit/session/ep_registry.py

## TL;DR

Process-wide singleton registry: discovers EP plugins via `discover_all_eps()` at construction time, registers them with ORT on demand, caches successful registrations, and exposes two compound entry points (`auto_device` for Path A, `available_eps`/`all_discovered` for Path B). v2.9 brings BuiltinSource synthesis into `_discovered` so built-ins (CPU/Dml/Azure) flow through the same `register_ep` pipeline as plugin EPs. Three module-private helpers (`_dedup_ort_devices`, `_ort_get_ep_devices_or_fail`, `_entry_source_tag`) plus two dataclasses (`WinMLEP`, `WinMLEPDevice`) round out the file.

## Diff metrics

- 601 lines (parent had ~371; net +230 inferred from commit stat).
- Top-level: 3 module-private helpers, 2 frozen dataclasses, 1 class with 7 methods + 1 classmethod + 6 instance fields.

## Role before vs after

**Before.** The registry held two separate caches: a plugin-EP `_registered` keyed by EP name (not path) AND a built-in EP enumeration. `register_ep` could raise on cache hit ("library already registered"), breaking precedence retry loops. `WinMLEPDevice` (the old meaning) was an intent-style descriptor.

**After.** Unified pipeline:
- `_discovered: list[EPEntry]` is built once at `__init__` by concatenating `discover_all_eps()` (plugin discovery) with synthesized `EPEntry(ep_name=builtin_name, dll_path=Path(), source=BuiltinSource(...))` rows for every name in `ort.get_available_providers() ∩ {d.ep_name for d in ort.get_ep_devices()}` not already covered by filesystem discovery.
- `register_ep` is idempotent by `entry.dll_path`. Cache hits return the cached `WinMLEP`. BuiltinSource entries get their own `_builtin_registered` cache keyed by `ep_name` (because the `dll_path` sentinel `Path("")` would collide).
- `WinMLEPDevice` is now the flat `(WinMLEP, WinMLDevice)` pair (matches `3_design_classes.md` §3.6).
- The singleton is exposed exclusively via `instance()` classmethod; the old `__new__ + _initialized` pattern is gone.

## Symbol-level changes

### `_dedup_ort_devices(devices)` (lines 36-54)

Collapses duplicate `OrtEpDevice` handles sharing `(vendor_id, device_id, type.name)`. Handles `AttributeError` (un-keyable devices fall through to `out` un-deduplicated). Used by both `register_ep` branches.

### `_ort_get_ep_devices_or_fail(entry)` (lines 57-72)

Wraps `ort.get_ep_devices()` in a try/except converting any exception to `WinMLEPRegistrationFailed`. Lets `auto_device`'s precedence retry handle ORT-side failures (driver reset, native init failures) the same way as DLL load failures.

### `_entry_source_tag(entry)` (lines 75-109)

Dispatches `entry.source` (`EPSource` instance) to one of 7 canonical tags: `pypi`, `nuget`, `winml-catalog`, `directory`, `bundled`, `msix-microsoft`, `msix-workload`, plus `"unknown"` fallback. The MSIX dispatcher reads `family_name_prefix` and returns `"msix-workload"` if it starts with `"WindowsWorkload.EP."`, else `"msix-microsoft"`. Used by `auto_device`'s source-tag filter.

The lazy import inside the function (lines 85-91) imports 5 EPSource subclasses from `..ep_path` to avoid the registry's construction-time import cost. `BuiltinSource` is already imported at the top of this file (line 21) because `__init__`'s synthesis loop uses it directly.

### `WinMLEP` (lines 112-139)

Frozen dataclass: `source: EPEntry, devices: tuple[WinMLDevice, ...]`. `__post_init__` enforces `len(devices) >= 1`. `ep_devices()` returns a tuple of `WinMLEPDevice` pairs — one per device.

### `WinMLEPDevice` (lines 142-162)

Frozen dataclass: `ep: WinMLEP, device: WinMLDevice`. `__post_init__` enforces `any(d is self.device for d in self.ep.devices)` — same-object identity, not equality. Reasonable: composing the pair from registry-built objects always preserves identity.

### `WinMLEPRegistry` (lines 165-489)

#### Class-level

- `_instance: ClassVar[WinMLEPRegistry | None] = None`.

#### `__init__` (lines 182-248)

1. `self._registered: dict[Path, WinMLEP] = {}` — plugin cache.
2. `self._registration_count: dict[str, int] = {}` — tracks suffix counts for ORT's `arg0` (so a second registration of the same `ep_name` becomes `"...ExecutionProvider_1"`).
3. ORT built-in name detection: `provider_names ∩ {d.ep_name for d in ep_devices}` — defensive against an EP listed in `get_available_providers()` but absent from `get_ep_devices()` (F-07 in the comment).
4. `plugin_entries = list(discover_all_eps())`.
5. `discovered_names = {e.ep_name for e in plugin_entries}`.
6. `self._discovered = plugin_entries + [synthesized BuiltinSource entries for builtin_names - discovered_names]`. **Sorted** by `builtin_name` (line 239: `for builtin_name in sorted(builtin_names - discovered_names)`).
7. `self._builtin_registered: dict[str, WinMLEP] = {}`.
8. `self._available_eps_cache: frozenset[str] | None = None`.

#### `_entries_for(ep_full_name)` (lines 250-257)

Linear-scan filter over `self._discovered`. Documented as registry-internal.

#### `register_ep(entry)` (lines 259-355)

Branches on `isinstance(entry.source, BuiltinSource)`:

**Built-in path** (lines 298-313):
1. Cache lookup in `_builtin_registered[entry.ep_name]`.
2. Call `_ort_get_ep_devices_or_fail`.
3. Filter by `d.ep_name == entry.ep_name`.
4. Dedup, raise `WinMLEPRegistrationFailed` on empty.
5. Build `WinMLEP`, cache by `ep_name`, return.

**Plugin path** (lines 318-355):
1. Cache lookup in `_registered[entry.dll_path]`.
2. Compute `arg0 = entry.ep_name if n == 0 else f"{entry.ep_name}_{n}"` where `n = _registration_count.get(entry.ep_name, 0)`.
3. Call `ort.register_execution_provider_library(arg0, str(entry.dll_path))`, convert any exception to `WinMLEPRegistrationFailed`.
4. Increment `_registration_count[entry.ep_name]`.
5. Re-query `ort.get_ep_devices()` via `_ort_get_ep_devices_or_fail`, filter by `d.ep_metadata.get("library_path") == str(entry.dll_path)`.
6. Dedup, raise on empty.
7. Build `WinMLEP`, cache by `dll_path`, return.

#### `auto_device(target)` (lines 357-418)

1. Validate `target.ep != "auto"` and `target.device != "auto"` — raises `ValueError`.
2. `candidates = self._entries_for(expand_ep_name(target.ep))`. Raise `WinMLEPNotDiscovered` if empty.
3. If `target.source is not None`: filter via `_entry_source_tag(e) == target.source`. Raise `UnknownListingPick(target.ep, target.source)` if empty.
4. Precedence retry loop: for each candidate, try `register_ep`. Catch `WinMLEPRegistrationFailed` (stash `last_error`, continue). On success, scan `winml_ep.devices` for `device_type == target.device.upper()` match; return `WinMLEPDevice(ep=winml_ep, device=device)` immediately.
5. After loop: if `last_error is not None`, raise `WinMLEPRegistrationFailed(...)` chained from `last_error`. Else raise `DeviceNotFound(...)`.

#### `all_discovered()` (lines 420-431)

`tuple(self._discovered)` — snapshot accessor. No filtering.

#### `available_eps()` (lines 433-477)

L0-only: every `e.ep_name` in `_discovered`. Memoized on `_available_eps_cache`. Catches `ImportError`/`RuntimeError` (when "WinML / sysinfo not available") → empty frozenset. Catches bare `Exception` → log WARN + empty.

#### `instance()` (lines 479-489)

Classmethod singleton getter. Constructs on first call, returns cached handle thereafter.

## Behavior / contract changes

1. **`register_ep` is idempotent.** Cache hits short-circuit without re-calling ORT. Documented thoroughly in the docstring.
2. **Built-in EPs do not call `register_execution_provider_library`.** They wrap pre-registered ORT handles directly.
3. **Multiple DLLs reporting the same `ep_name` get suffixed registration keys** (`_1`, `_2`, ...) but their device handles still self-report the canonical name. Verified via `temp/probe_double_register.py` per the docstring (lines 281-285). Filtering on `d.ep_metadata.get("library_path") == str(entry.dll_path)` is the load-bearing post-registration filter — without it, multiple registrations of the same `ep_name` would collapse to one device set.
4. **Compatible with the `_get_detected_vendors RuntimeError` headless-server case** via `available_eps`'s try/except. `auto_device`'s precedence retry chains through `WinMLEPRegistrationFailed` so an unwrapped `Exception` from `register_execution_provider_library` becomes the documented exception.
5. **`auto_device` raises `WinMLEPNotDiscovered` for empty-candidate even if `target.source is None`.** A user with no OpenVINO plugin installed who runs `winml perf --ep openvino` gets `WinMLEPNotDiscovered` BEFORE the source filter ever runs. Match `1_req.md` §R1.
6. **`UnknownListingPick` is raised inside the registry**, not in `resolve_device`. Match `2_coreloop.md` §5.3 v2.6.
7. **`DeviceNotFound` vs `WinMLEPRegistrationFailed` distinction:** `auto_device` raises `WinMLEPRegistrationFailed` if at least one candidate's registration raised; raises `DeviceNotFound` only when every candidate registered cleanly but none exposed the target device class. Specifically, the `if last_error is not None` branch (line 410) FIRES even when one candidate succeeded but didn't match the device class — because `last_error` set during an earlier failing candidate's exception still survives. This is a **bug**: if candidate A fails with `WinMLEPRegistrationFailed` and candidate B succeeds but doesn't expose the target device, the user sees `WinMLEPRegistrationFailed` (with A's traceback) when they should see `DeviceNotFound`. See Risks #1.
8. **The `_available_eps_cache` is never invalidated.** Tests that reset `WinMLEPRegistry._instance = None` and re-construct get a fresh cache. But within one process, the discovery cache is genuinely immutable post-init, so this is correct.

## Cross-file impact

- 8+ files import `WinMLEPRegistry.instance()` for Path A flow.
- `commands/sys.py` calls `instance()` and walks `all_discovered()` for the `--list-ep` render.
- `commands/perf.py` and `commands/compile.py` call `auto_device(resolved_target)` after `resolve_device`.
- The qairt `WinMLQairtSession.__init__` falls back to `auto_device(resolve_device(EPDeviceTarget(ep="qnn", device="npu")))` when `ep_device is None`.
- Tests at `tests/unit/session/test_ep_registry.py` (+429 lines), `test_winml_ep.py` (+208 lines), `test_entry_source_tag.py` (+146 lines), `test_dedup_ort_devices.py` (+63 lines) cover this module.

## Risks / subtleties

1. **The `auto_device` last-error logic is fragile.** As noted in Behavior #7: if candidate A raises `WinMLEPRegistrationFailed` and candidate B registers cleanly but exposes no matching device, the user sees A's failure attributed as "no compatible source." Should reset `last_error` after a successful registration (whether or not the device matched), so a registration-success-then-device-miss scenario falls through to `DeviceNotFound`. Bug surface depends on EP availability; mostly latent today.
2. **`_registration_count` increments unconditionally on successful library-load**, but the cache-hit branch (line 318) returns before incrementing. So the count tracks DISTINCT registrations only, which is correct for the suffix logic. The subtle point: a registry that hits the cache once and then has its `_registered` dict mutated externally (impossible in production, but in tests) would have `_registration_count` out of sync. Not really risky.
3. **Built-in entries are sorted by `builtin_name`** for deterministic ordering. Plugin entries preserve discovery order (precedence). The two sub-lists are concatenated in order. So an EP that exists as both a plugin and a built-in (e.g., a future scenario) lands as plugin-primary, built-in-shadowed — which is correct per the doc-comment ("built-ins are lowest priority").
4. **`_ort_get_ep_devices_or_fail` raises `WinMLEPRegistrationFailed` even though the failure is in `get_ep_devices`, not in `register_execution_provider_library`.** Semantically debatable — the exception name suggests "DLL failed to register," but the function raises it from `get_ep_devices()` failures too. The docstring (lines 58-66) defends this as "auto_device's WinMLEPRegistrationFailed retry loop can fall through to the next candidate instead of crashing." Acceptable; just worth understanding.
5. **`_entries_for` is a linear scan over `_discovered`.** For 8-15 entries this is sub-microsecond. Acceptable.
6. **`auto_device` returns IMMEDIATELY on first device-match.** So if a registered EP has multiple devices and the user specified `--device npu`, the loop returns the first NPU device the EP exposes. This works correctly when `_dedup_ort_devices` has already collapsed duplicates. Edge case: an EP that exposes two distinct NPU devices (e.g., dual Snapdragon) would always pick the same one. Determinism comes from `ort.get_ep_devices()` order, which is platform-dependent.
7. **`available_eps`'s `(ImportError, RuntimeError)` catch** suggests prior behavior where `discover_all_eps` could fail with these. Post-refactor, `discover_all_eps` is called at `__init__` time, not at `available_eps` call time — so the try/except in `available_eps` is dead (the discovery has already happened or has already raised at instance construction). See Simplification #2.
8. **`available_eps`'s WARN-logged `Exception` catch** swallows real bugs. If `frozenset(e.ep_name for e in self._discovered)` raises (which it shouldn't), the call silently returns empty. Defensive but covers a bug the caller can't see.
9. **The synthesized `EPEntry` for built-ins gives `dll_path=Path()`** — this is `Path(".")` (the current directory) by Python convention. The doc and code both treat it as a sentinel via `is_filesystem_backed`. Fine, but `Path()` is technically a directory, and `Path("").is_file()` returns False, so the `discover_all_eps` filter coincidentally also drops it. Just worth knowing.

## Simplification opportunities

1. **`auto_device`'s `last_error` should reset to `None` after a successful registration that didn't match the device class.** Three-line fix:
   ```python
   for entry in candidates:
       try:
           winml_ep = self.register_ep(entry)
       except WinMLEPRegistrationFailed as e:
           last_error = e
           continue
       last_error = None  # successful registration; reset
       for device in winml_ep.devices:
           if device.device_type == target_device_upper:
               return WinMLEPDevice(ep=winml_ep, device=device)
   ```
2. **`available_eps`'s try/except is dead.** Discovery happens at `__init__`; by the time `available_eps` runs, `self._discovered` is already populated. The try/except around `frozenset(...)` is defensive but covers no real failure mode. Simplify to:
   ```python
   if self._available_eps_cache is None:
       self._available_eps_cache = frozenset(e.ep_name for e in self._discovered)
   return self._available_eps_cache
   ```
3. **`_entry_source_tag`'s lazy import is fine** but the function's dispatch could use a dict of `type → str`:
   ```python
   _SOURCE_TAGS = {PyPISource: "pypi", NuGetSource: "nuget", ...}
   tag = next((t for cls, t in _SOURCE_TAGS.items() if isinstance(s, cls)), "unknown")
   if tag == "msix-microsoft" and s.family_name_prefix.startswith("WindowsWorkload.EP."):
       tag = "msix-workload"
   return tag
   ```
   Marginal. The current cascading `isinstance` checks are explicit but verbose.
4. **The `BuiltinSource` synthesis in `__init__`** is 10 lines including the comment. Could be a private method `_synthesize_builtins(plugin_entries) -> list[EPEntry]` for testability. As-is the logic is inline with the other `__init__` setup.
5. **`WinMLEPDevice.__post_init__` uses `any(d is self.device for d in self.ep.devices)`** — `self.device in self.ep.devices` would also work because `WinMLDevice` uses default identity equality (no `__eq__`). The current form is more explicit about identity vs. value. Stylistic.
6. **`WinMLEPRegistry.instance` is a classmethod** but the singleton state is on the class itself. Could be a module-level `_get_registry()` function — but the design doc (`3_design_classes.md` §6) explicitly settled on `.instance()`. Don't change.
7. **The `_registered` and `_builtin_registered` caches could merge if keyed by a tagged union**: `dict[tuple[str, str], WinMLEP]` where the key is `("dll", str(dll_path))` or `("builtin", ep_name)`. Would unify the two `register_ep` branches' cache lookup. As-is the two-cache shape mirrors the two-type shape and is arguably more readable.
8. **The synthesized BuiltinSource entries always have `eps=(builtin_name,)`** — same single-name shape. The data is duplicated in `entry.ep_name` and `entry.source.eps[0]`. Could `EPEntry.from_builtin_name(name)` factory it. Minor.

## Open questions / TODOs surfaced

- The `auto_device` last-error tracking bug (Risk #1) should be flagged as a real defect in this batch's headline finding.
- The `available_eps` defensive try/except (Simplification #2) suggests a refactor never cleaned up after `discover_all_eps` moved into `__init__`. Worth a sweep.
- Does ORT's `register_execution_provider_library` actually exit(127) on duplicate registration (mentioned in `3_design_ep.md` §5.2)? The new idempotency cache means the codebase never tests this — defensive plumbing only. If we trust ORT 1.24+ to handle re-registration via the `arg0` suffix, the `_registered`-by-`dll_path` cache could become a true idempotency invariant rather than a defensive guard.
- The 7th source tag `unknown` returned by `_entry_source_tag` is unreachable in production (every `EPSource` subclass is one of the 6 covered). It's a "should not happen" fallback. Could be `assert False` with a `# pragma: no cover` to make the intent explicit.
- The cardinality split between `WinMLEP.ep_devices()` (returns ALL pairs) and `auto_device`'s precedence loop (returns FIRST matching pair) is asymmetric. Users wanting "all NPU devices for this EP" must filter the `ep_devices()` output themselves. Documented in `3_design_classes.md` §3.5 but worth re-noting.
