# src/winml/modelkit/session/ep_device.py

## TL;DR

New file (+748 lines): houses the **intent layer** (`EPDeviceTarget`), the **catalog layer** (`EPDeviceSpec` + `EP_DEVICE_SPECS`), the **runtime adapter** (`WinMLDevice`, single concrete class with internal dispatch on `self._ort.ep_name`), the **pure-deduction resolver** (`resolve_device`) and a fan of helper functions (`auto_detect_device`, `default_ep_for_device`, `default_device_for_ep`, `eps_for_device`, `ep_to_device`, `expand_ep_name`, `short_ep_name`, `lookup_device_spec`). All five session-layer exceptions live here (`WinMLEPNotDiscovered`, `WinMLEPRegistrationFailed`, `DeviceNotFound`, `WinMLEPMonitorMismatch`, `UnknownListingPick`).

## Diff metrics

- Mode: NEW FILE (+748 / -0)
- Major sections: exceptions (5 classes), name helpers (`_SHORT_TO_FULL`, `_FULL_TO_SHORT`, `expand_ep_name`, `short_ep_name`), validation closed-sets (`VALID_DEVICES`, `VALID_SOURCE_TAGS`, `known_ep_short_names`), `EPDeviceTarget` dataclass, `EPDeviceSpec` + 12 catalog rows, lookup/deduction helpers (5 functions), `resolve_device`, `WinMLDevice` class (15+ properties/methods), `_format_bytes` private util.

## Role before vs after

**Before.** EP-device logic was split across `ep_device.py` (typed `WinMLEPDevice` descriptor) and `ep_path.py` (vendor compat + EP name lookups). `WinMLDevice` was an ABC with per-EP subclasses. `resolve_device(ep: str | None, device: str | None, source: str | None = None)` took strings.

**After.** Three layers in one module:
1. **`EPDeviceTarget`** — pure-data user intent. Frozen, JSON-serializable, no ORT dependency. Construction-time validation via `__post_init__`.
2. **`EPDeviceSpec` + `EP_DEVICE_SPECS`** — single authoritative catalog.
3. **`WinMLDevice`** — single concrete class wrapping `ort.OrtEpDevice`. Per-EP metadata via internal dispatch on `self._ort.ep_name`. No ABC, no subclasses.

`resolve_device(target: EPDeviceTarget) -> EPDeviceTarget` is now pure deduction — no DLL load, no registry I/O, no filesystem scan. Source-tag validation against discovered EPEntries moves to `WinMLEPRegistry.auto_device`.

## Symbol-level changes

### Exceptions (lines 47-77)

5 classes, all `# noqa: N818` annotated (PascalCase non-`Error` suffix accepted):
- `WinMLEPNotDiscovered` — "EP plugin is not in the catalog or WINMLCLI_EP_PATH."
- `WinMLEPRegistrationFailed` — "ort.register_execution_provider_library raised."
- `DeviceNotFound` — "EP registered, but no OrtEpDevice matches the descriptor."
- `WinMLEPMonitorMismatch` — "Monitor.ep_name does not agree with EPDeviceTarget.ep."
- `UnknownListingPick` — Constructor takes `ep_name` + `source_tag`, stores them as attributes, formats message with hint to "Run 'winml sys --list-ep' to see available sources."

### Name helpers (lines 85-127)

- `_SHORT_TO_FULL: Final[dict[str, str]]` — 9 entries (qnn, openvino, vitisai, migraphx, nvtensorrtrtx, cuda, tensorrt, dml, cpu). Includes `CUDAExecutionProvider` and `TensorrtExecutionProvider` mappings even though those EPs are not in the catalog.
- `expand_ep_name(name)` — case-folds for lookup; passthrough if not in dict.
- `_FULL_TO_SHORT: Final[dict[str, str]]` — inverse mapping built at module-import time. Comment claims "built lazily so any future additions to _SHORT_TO_FULL are picked up automatically" — this is **incorrect**; it's an eager dict literal at module load, no lazy semantic.
- `short_ep_name(full)` — returns dict lookup or `full.removesuffix("ExecutionProvider").lower()` fallback (so `"AzureExecutionProvider" -> "azure"` works even though `azure` isn't in `_SHORT_TO_FULL`).

### Validation closed sets (lines 139-160)

- `VALID_DEVICES = frozenset({"npu", "gpu", "cpu"})`.
- `VALID_SOURCE_TAGS = frozenset({"bundled", "pypi", "nuget", "msix-microsoft", "msix-workload", "winml-catalog", "directory"})` — the 7 canonical tags per `1_req.md` §R4.
- `known_ep_short_names() -> frozenset[str]` — `frozenset(_SHORT_TO_FULL.keys())`. Function not lru_cached.

### `EPDeviceTarget` (lines 167-235)

Frozen dataclass: `ep: str`, `device: str`, `source: str | None = None`. `__post_init__` runs three checks: device normalization + closed-set membership; ep validation against `known_ep_short_names()` and `_FULL_TO_SHORT`; source tag against `VALID_SOURCE_TAGS`. `to_dict()` uses `dataclasses.asdict`. `from_dict()` reads only `ep`/`device`/`source` keys ignoring legacy `vendor_id`/`device_id`/`vendor` "for forward-compat for persisted JSON written before the Batch C strip."

### `EPDeviceSpec` + `EP_DEVICE_SPECS` (lines 244-303)

`EPDeviceSpec(ep, device, default_provider_options)` with `kw_only=True, slots=True`. The catalog has **12 entries** (not 8 as claimed in the prior commit notes; matches the doc):
- QNN/NPU with `htp_performance_mode="burst"` + `htp_graph_finalization_optimization_mode="3"`.
- DML/GPU, CPU/CPU (positions 1-2 — primary per-device).
- QNN/GPU, QNN/CPU (positions 3-4, secondary, no provider_options).
- OpenVINO/NPU, OpenVINO/GPU, OpenVINO/CPU (positions 5-7).
- VitisAI/NPU (position 8), MIGraphX/GPU (9), Tensorrt/GPU (10), NvTensorRtRtx/GPU (11).

`_BY_KEY: Final[dict[tuple[str, str], EPDeviceSpec]] = {(s.ep, s.device): s for s in EP_DEVICE_SPECS}` — O(1) lookup cache.

`VALID_EPS = frozenset({short_ep_name(s.ep) for s in EP_DEVICE_SPECS})` — 9 short names. **Note:** built-ins CPU/DML come through `short_ep_name` (returning `"cpu"` / `"dml"`); `azure` is NOT in `VALID_EPS` because no `EP_DEVICE_SPECS` row mentions `AzureExecutionProvider`. Yet `_SHORT_TO_FULL` has no `azure` entry either, so users typing `--ep azure` hit the `EPDeviceTarget.__post_init__` validation error before ever reaching the catalog. Consistent but worth noting.

### Lookup / deduction helpers (lines 306-423)

- `lookup_device_spec(ep, device) -> EPDeviceSpec | None` — `_BY_KEY.get((ep, device))`.
- `default_device_for_ep(ep) -> str | None` — `next((s.device for s in EP_DEVICE_SPECS if s.ep == ep), None)`. **Linear scan**, not using `_BY_KEY`.
- `default_ep_for_device(device) -> str | None` — walks `EP_DEVICE_SPECS` filtered by device match AND `s.ep in available_eps()` AND `EP_CATALOG.is_compatible(s.ep)`. Catches `RuntimeError` (from `_get_detected_vendors` on headless servers), logs WARN, returns `None`.
- `eps_for_device(device) -> frozenset[str]` — `frozenset(s.ep for s in EP_DEVICE_SPECS if s.device == device.lower())`. Returns full EP names. Empty frozenset for unknown devices.
- `ep_to_device(ep) -> str` — short→short via `default_device_for_ep(expand_ep_name(ep))`. Raises `ValueError` on miss.

### `auto_detect_device()` (lines 427-466)

Walks `get_available_devices()` from `sysinfo`, checks each device class against `eps_for_device(dev) ∩ available_eps()`, requires at least one ep to also satisfy `EP_CATALOG.is_compatible`. Catches `RuntimeError` from `is_compatible`, logs WARN, returns `"cpu"`. Final fallback is also `"cpu"` if no device matches.

### `resolve_device(target)` (lines 470-549)

```python
def resolve_device(target: EPDeviceTarget) -> EPDeviceTarget:
```

Pure deduction:
- If `device == "auto"`: branch on `ep == "auto"` (call `auto_detect_device()`) vs ep given (call `default_device_for_ep(expand_ep_name(ep))`, raise on None).
- Else: lowercase + validate against `VALID_DEVICES`.
- If `ep == "auto"`: call `default_ep_for_device(device)`, raise on None.
- Build resolved `EPDeviceTarget(ep=expand_ep_name(ep), device=device, source=target.source)`. Logs INFO with before/after representation.

`target.source` passes through unchanged — source-tag validation lives in `auto_device`.

### `WinMLDevice` class (lines 555-737)

`__init__(self, ort_device)` stores `self._ort`. **Common properties** (no dispatch):
- `ep_name`, `device_type` (uppercased), `hardware_name` (cascading FULL_DEVICE_NAME → Description → "<unknown>"), `vendor`, `ep_vendor`, `library_path`, `ort_handle` (public read-only accessor for `self._ort`).

**Vendor-specific properties** (dispatch inline on `ep_name`):
- `memory_bytes` — OpenVINO via `NPU_DEVICE_TOTAL_MEM_SIZE` / `GPU_DEVICE_TOTAL_MEM_SIZE`; DML via parsing `device.metadata['DxgiVideoMemory']` as `"<N> <unit>"`. Returns `None` for everything else.
- `architecture` — OpenVINO via `DEVICE_ARCHITECTURE` (parses `arch=` suffix), passthrough otherwise. `None` elsewhere.
- `capabilities` — OpenVINO via `OPTIMIZATION_CAPABILITIES` token-split with rewrites dict (`GPU_HW_MATMUL → MatMul`, `GPU_USM_MEMORY → USM`, `EXPORT_IMPORT → ""` drop). Returns `()` elsewhere.
- `driver_version` — OpenVINO NPU only (`NPU_DRIVER_VERSION`). `None` elsewhere.
- `compiler_version` — OpenVINO NPU only (`NPU_COMPILER_VERSION`). `None` elsewhere.

**Introspection**: `available_metadata() -> dict`. **Display**: `device_facts() -> tuple[str, ...]` returns `("Architecture: ...", "Driver: ...")` filtered for non-None. `ep_facts() -> tuple[str, ...]` returns `("Memory: ...", "Capabilities: ...")` filtered. Both per the §4.1 attribute-attribution split.

### `_format_bytes` (lines 740-748)

Renders `int` bytes as `"<float> {GiB,MiB,KiB,B}"`. Best-effort.

## Behavior / contract changes

1. **`resolve_device` is now purely deductive.** No filesystem, no registry, no DLL load. The doc-comment matches the implementation. v2.6 explicit choice.
2. **`source` passes through unchanged.** No validation happens in `resolve_device`; that fires in `auto_device` (`ep_registry.py`).
3. **`default_ep_for_device`** does TWO filters: L0 (in `available_eps()`) AND L2 (`EP_CATALOG.is_compatible`). The docstring explicitly does NOT include L1 (registration success) — that fires later in `auto_device`'s precedence loop.
4. **`auto_detect_device` falls back to "cpu" on `RuntimeError` from `is_compatible`** (the headless-server case where `_get_detected_vendors` raises). Doc-comment notes this is a graceful CPU fallback for click commands.
5. **`EPDeviceTarget.from_dict` silently drops legacy keys.** No warning if `vendor_id` / `device_id` / `vendor` come in from old JSON. Forward-compat by design.
6. **`WinMLDevice.ort_handle` is a property returning `self._ort`.** Documented as "public read-only accessor for external callers (analyze/, future plugins) that need to pass the raw OrtEpDevice to APIs like `SessionOptions.add_provider_for_devices` or `ort.ModelCompiler`." But the actual session.py implementation reads `ep_device.device._ort` (the private attribute, line 187) not `ep_device.device.ort_handle`. The property is documented but unused. See Simplification #1 below.
7. **`compiler_version` exists as a property but is excluded from `ep_facts()`.** Documented as `--verbose`-only via `available_metadata()`. Match `4_winml_device.md` §4.1 expectation.
8. **`expand_ep_name` does NOT validate the result against the catalog.** `expand_ep_name("foobar")` returns `"foobar"` silently. Callers (`EPDeviceTarget.__post_init__`) do the validation separately, but a stray `expand_ep_name` consumer that skips validation passes garbage through.

## Cross-file impact

- `ep_registry.py` imports `DeviceNotFound`, `EPDeviceTarget`, `UnknownListingPick`, `WinMLDevice`, `WinMLEPNotDiscovered`, `WinMLEPRegistrationFailed`, `expand_ep_name` — every public exception is sourced from here.
- `session/__init__.py` re-exports the 17 names listed in its `__all__` from here.
- `session/session.py` imports `WinMLEPMonitorMismatch`, `expand_ep_name`, `lookup_device_spec`.
- `commands/_ep_arg.py` imports `VALID_SOURCE_TAGS` — cross-package coupling but minimal.
- `compiler/configs.py` imports `EPDeviceTarget`, `resolve_device` (via facade).
- 10+ files import `resolve_device` and `EPDeviceTarget` for CLI / SDK boundaries.

## Risks / subtleties

1. **`_FULL_TO_SHORT` is built eagerly** but the inline docstring (line 113) claims it's lazy. Comment lie; actual semantic is eager dict comprehension at module load. If `_SHORT_TO_FULL` is mutated post-import (unlikely but possible in tests), the inverse stays stale.
2. **The unused `ort_handle` property** (lines 597-606) suggests an API mismatch between the doc-stated public API and the actual session-layer implementation. `session.py` line 187 reaches `ep_device.device._ort` directly. The property exists but no in-tree consumer uses it.
3. **`default_device_for_ep` does a linear scan** (line 332: `next(s.device for s in EP_DEVICE_SPECS if s.ep == ep)`). Could use `_BY_KEY` if it built a `(_BY_EP: dict[str, str])` index — but with 12 catalog entries this is sub-microsecond. Trade-off: O(n) walk vs index storage.
4. **`auto_detect_device` returns `"cpu"` rather than raising** on no-matching-device or on RuntimeError. The fallback is documented but means a user with no hardware-EP DLLs still gets a session built — bound to `cpu` even if they may not expect it. Consistent with `1_req.md` §R1 ("If no plugin EPs are installed, fall back to bundled CPU.")
5. **The `Any` return type annotation on `WinMLDevice.ort_handle`** would let callers pass anything through. Currently typed as `ort.OrtEpDevice` (line 598). Good.
6. **`EPDeviceTarget` validation matches against `known_ep_short_names()` OR `_FULL_TO_SHORT`** — but `_FULL_TO_SHORT` keys are full names like `"QNNExecutionProvider"`. The check `self.ep not in _FULL_TO_SHORT` (line 204) needs an exact full-name match. `OpenVINOExecutionProvider.AUTO` (the variant the WinMLDevice memory check at line 615 handles via `"OpenVINO" in ep`) would fail `EPDeviceTarget` validation — fine because `.AUTO` is a runtime ORT-reported name, not a user-typeable EP, but the cross-table inconsistency between EPDeviceTarget (rejects `.AUTO`) and WinMLDevice (accepts it via substring match) is worth flagging.
7. **`_BY_KEY` is built at module load** but `EP_DEVICE_SPECS` has duplicate `s.ep` rows (QNN/NPU, QNN/GPU, QNN/CPU; OpenVINO/NPU, .../GPU, .../CPU). The dict-comprehension `{(s.ep, s.device): s for s in EP_DEVICE_SPECS}` correctly keys on the pair, so duplicates don't collide. Just worth knowing.
8. **The OpenVINO `device_type` check inside `memory_bytes`** (line 615: `"OpenVINO" in ep`) uses substring match — would silently match a hypothetical `"OpenVINOFooExecutionProvider"`. Acceptable best-effort.

## Simplification opportunities

1. **The `ort_handle` property on `WinMLDevice` is unused.** Either delete it (and update session.py to drop the leading underscore from `_ort`) or commit to it (make `session.py` use `ep_device.device.ort_handle`). Currently we have private attribute access from inside the package and a public-but-unused accessor.
2. **`expand_ep_name` and `short_ep_name` could be `@functools.cache`'d** — they're called multiple times per session boundary. Trade-off: caching string→string is rarely worth it, but here they're hot.
3. **`default_device_for_ep`'s linear scan could share a single `_BY_EP: dict[str, str]`** built from `EP_DEVICE_SPECS` at module init. Not a perf issue but reduces the asymmetry between `_BY_KEY` and the linear scan helpers.
4. **`auto_detect_device` and `default_ep_for_device` both wrap `EP_CATALOG.is_compatible` in try/except RuntimeError.** Could push the try/except inside `EP_CATALOG.is_compatible` (or a wrapper helper) and let the callers stay clean. Currently both call sites repeat the same defensive pattern with slightly different log messages.
5. **The `EPDeviceTarget.__post_init__` device case-normalization** (line 190-191) uses `object.__setattr__` to mutate a frozen dataclass. The pattern is correct but verbose for the one case (device casing). Could use `dataclasses.replace` at validation time, or normalize at the CLI parser. Minor.
6. **The `_format_bytes` private utility** lives in this module but is called only from `WinMLDevice.ep_facts`. Could be a `staticmethod` on `WinMLDevice` to keep it close to its consumer.
7. **The dispatch in `memory_bytes` could be table-driven.** Currently it's a sequence of `if "OpenVINO" in ep:` then `if ep == "DmlExecutionProvider":` — fine for 2 EPs but the 4_winml_device.md design calls out future QNN/VitisAI/MIGraphX/Tensorrt entries. A `_MEMORY_KEY_TABLES: dict[str, Callable[[OrtEpDevice], int | None]]` keyed on `ep_name` would be cleaner. Same for `architecture`/`capabilities`/`driver_version`/`compiler_version`. As-is the per-EP branches grow with EP count.
8. **`UnknownListingPick.__init__` stores `ep_name` and `source_tag` as instance attributes** and also passes them to `super().__init__(message)`. The base `Exception` already accepts the formatted message as `args[0]`. The extra attributes ARE useful (callers can read `e.ep_name` programmatically) but the doc-comment says "Carries `ep` and `source_tag` in `args`" — `args` is `(message,)`, NOT `(ep_name, source_tag)`. Either update the docstring or change the call to `super().__init__(ep_name, source_tag)` and re-format `__str__`.
9. **`VALID_EPS` is the only `*_EPS` validation constant** but `_SHORT_TO_FULL` has 9 short names while `VALID_EPS` is derived from `EP_DEVICE_SPECS` (12 rows). So `VALID_EPS` contains `{qnn, dml, cpu, openvino, vitisai, migraphx, tensorrt, nvtensorrtrtx}` (8 names — DML and CPU and 6 plugins). The 9th entry in `_SHORT_TO_FULL` is `cuda` which is NOT in `VALID_EPS`. Validation of `EPDeviceTarget(ep="cuda")` passes (because `cuda` is in `known_ep_short_names()`) but the resulting target won't match anything in `EP_DEVICE_SPECS`. This is a latent issue. Either drop `cuda`/`tensorrt` from `_SHORT_TO_FULL` (since they're not catalog rows), or add catalog rows for them, or document that `VALID_EPS` and `known_ep_short_names` answer different questions.

## Open questions / TODOs surfaced

- Comment on lines 282-283 says `# TODO: verify whether device_type is needed under add_provider_for_devices, or auto-derived from OrtEpDevice handle (like QNN's backend_type).` The OpenVINO/NPU/GPU/CPU entries have NO `default_provider_options` — the TODO documents the uncertainty about whether OpenVINO needs `device_type` passed explicitly.
- Comment on line 279: `# QNN/GPU TODO: measure` — secondary QNN entries have no benchmarked options.
- The unused `ort_handle` property + the docstring claim of public usage suggests a documentation drift. The plan was likely "session.py should use the public accessor" but session.py wasn't updated to match. Either bring session.py in line or delete the accessor.
- Comment on line 113 ("built lazily") is wrong. Either rephrase to "built eagerly at module import" or actually lazy-build via a `@functools.cache`'d `_full_to_short()` getter.
- The `VALID_EPS` vs `known_ep_short_names` mismatch (cuda accepted in EPDeviceTarget but not actionable downstream) should be resolved one way or the other.
