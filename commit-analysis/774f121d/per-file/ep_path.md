# src/winml/modelkit/ep_path.py

## TL;DR

New file (+1518 lines): the unified EP discovery layer. Five concrete `EPSource` subclasses (PyPISource, NuGetSource, DirectorySource, WinMLCatalogSource, MSIXPackageSource) plus a sixth marker class `BuiltinSource` for ORT-bundled EPs, an `EPCatalog` immutable metadata registry replacing the legacy `EP_DLL_NAMES` / `_DLL_TO_EP_NAME` / `_EP_VENDOR_REQUIREMENT` dicts, an `EPEntry` discovery record dataclass, and the single entry point `discover_all_eps()`. The module pulls the prior monolithic plugin-discovery into a small ABC + dataclass-implementation hierarchy whose entries are ordered like `PATH` (first-match-wins, with later duplicates surfacing as `status="shadowed"`).

## Diff metrics

- Mode: NEW FILE (+1518 / -0)
- Classes added: `EPCatalog`, `EPCatalog.Row`, `EPEntry`, `EPSource` (ABC), `BuiltinSource`, `PyPISource`, `NuGetSource`, `DirectorySource`, `WinMLCatalogSource`, `MSIXPackageSource` (10 total)
- Free functions added: `_get_detected_vendors`, `_qnn_arch_resolver`, `_nuget_packages_root`, `_release_winml_handle`, `_get_catalog`, `_winml_warn_once`, `_get_pkg_manager`, `_pkg_version_tuple`, `_pkg_version_str`, `_list_msix_eps`, `_default_ep_sources`, `_parse_winmlcli_ep_path`, `discover_all_eps`
- Module constants: `EP_CATALOG`, `_winml_catalog_warned_keys` (module-mutable set), `logger`, `__all__` (10 public names)

## Role before vs after

**Before.** Plugin discovery lived in `winml.py` and the legacy `EP_PLUGIN_REGISTRY` dict, modeling only PyPI-installed plugins. EP DLL filenames, vendor compat, and reverse-lookup tables were three separate module-level dicts. Built-in (CPU/Dml/Azure) EPs were a special case handled in the registry's `__init__`, never represented as `EPSource`-shaped objects.

**After.** A single ordered `EPSource` list owns the precedence chain (PyPI → NuGet → WinMLCatalog → DirectorySource[env-var] → live `_list_msix_eps()`). `discover_all_eps()` flattens it, dedup-keyed on `(ep_name, normcased dll_path)`. The `EPCatalog` singleton freezes the EP metadata (name + dll_name + vendor_requirements) immutably, enforcing both `MappingProxyType` wrapping AND `__setattr__` lockdown after `__init__`. Built-ins are now a first-class `BuiltinSource` subclass whose `resolve()` is a no-op — instances are synthesized at registry init and dispatched through the same `register_ep` path.

## Symbol-level changes

### Top-level: `EPCatalog` (lines 68-186)

- `__slots__ = ("_by_dll", "_by_name", "_initialized")`. Construction signature: `__init__(self, entries: Iterable[EPCatalog.Row])`. Builds two `MappingProxyType`-wrapped dicts: `_by_name` (forward) and `_by_dll` (reverse — only for rows with non-empty `dll_name`).
- `__setattr__` raises `AttributeError("EPCatalog is immutable; cannot set X")` once `_initialized=True`.
- Methods: `dll_name_for(ep) -> str | None`, `ep_for_dll(dll) -> str | None`, `vendor_requirements_for(ep) -> frozenset[str]`, `is_compatible(ep) -> bool`, `all_eps() -> tuple[str, ...]`, `all_dlls() -> tuple[str, ...]`.
- `Row` is a nested frozen dataclass with three fields: `name`, `dll_name` (empty string for bundled EPs), `vendor_requirements: frozenset[str]`.
- Module-level `EP_CATALOG` instance with 8 hardcoded `Row` entries: OpenVINO/QNN/VitisAI/MIGraphX/NvTensorRtRtx (with vendor_requirements) and DML/CPU/Azure (bundled, dll_name="" and frozenset()).

### Top-level: `_get_detected_vendors` (lines 189-220)

`@functools.cache`'d helper. Returns the union of `manufacturer` and `name` strings across `GPU.get_all()` and `NPU.get_all()` from `sysinfo.hardware`. Raises `RuntimeError` on missing `sysinfo.hardware` import OR when `cls.get_all()` raises — defends against `functools.cache` pinning an empty set after a transient failure (which would silently mark every hardware-gated EP as incompatible).

### Top-level: `_qnn_arch_resolver(rel_template) -> str` (lines 228-236)

Picks `arm64ec` vs `amd64` from `platform.machine().lower()`. Used as a `PyPISource.arch_resolver` for the `onnxruntime-qnn` wheel layout.

### Top-level: `_nuget_packages_root() -> Path` (lines 244-260)

Returns `%USERPROFILE%/.nuget/packages` on Windows (with `Path.home()` fallback) or `~/.nuget/packages` on POSIX. Does **not** honor the `NUGET_PACKAGES` env var or the user's `NuGet.Config` `globalPackagesFolder`. Returns the path even when it does not exist.

### `EPEntry` dataclass (lines 268-301)

Frozen dataclass: `ep_name: str, dll_path: Path, source: EPSource, status: str = "primary", version: str | None = None`. One method: `is_filesystem_backed() -> bool` — returns `False` only when `isinstance(self.source, BuiltinSource)`. The method exists solely to gate the `dll_path.is_file()` check inside `discover_all_eps` so the `Path("")` sentinel used for built-ins doesn't get silently dropped.

### `EPSource` ABC (lines 309-344)

Two abstract methods: `resolve(self) -> Iterator[EPEntry]` and `iter_eps(self) -> Iterable[str]`. One concrete: `is_compatible(self) -> bool` which `all(EP_CATALOG.is_compatible(ep) for ep in self.iter_eps())`. Subclasses are frozen dataclasses.

### `BuiltinSource(EPSource)` (lines 347-372)

Frozen dataclass with one field: `eps: tuple[str, ...] = ()`. `resolve()` returns `iter(())` (always empty). `iter_eps()` returns `self.eps`. The class is a marker: instances are only constructed via the `register_ep` synthesis path in `WinMLEPRegistry.__init__`.

### `PyPISource(EPSource)` (lines 375-451)

Fields: `distribution: str`, `relative_dll: str`, `eps: tuple[str, ...]`, `arch_resolver: Callable[[str], str] | None`. `resolve()` uses `importlib.metadata.distribution(self.distribution).locate_file(rel)` (substitutes `arch_resolver` first), then `metadata.version(self.distribution)`. Yields one `EPEntry` per ep in `self.eps`. Failure modes: `PackageNotFoundError` → silent skip (DEBUG log); installed but missing file → WARN log + skip; `metadata.version` raising → DEBUG log + `version=None`.

### `NuGetSource(EPSource)` (lines 454-591)

Same field shape as PyPISource but resolves against `~/.nuget/packages/<lowercase-id>/<version>/<relative_dll>`. `relative_dll` validated for backslashes (raises `ValueError`). Picks the highest `packaging.Version` subdir, preferring stable over prerelease only when no stable exists. Tries each version in order until one's DLL exists. Yields one `EPEntry` per ep with `version=version_dir.name` (verbatim folder name, not the parsed normalized form).

### `DirectorySource(EPSource)` (lines 594-680)

Fields: `root: Path`, `dll_patterns: dict[str, str]`, `env_var: str | None = None`, `required_marker: str | None = None`. Resolve algorithm: (1) env-var gate (silent skip if unset/empty); (2) base path resolution (env_var value + root if root not absolute); (3) `base.exists()` check (WARN + skip on miss); (4) `required_marker` check (WARN + skip); (5) glob each `dll_patterns` value under `base`, yield `EPEntry(version=None)` for first hit. The dict-keys ordering decides yield order.

### WinAppSDK catalog plumbing (lines 683-784)

- `_winml_catalog_warned_keys: set[str]` — module-mutable WARN dedup set.
- `_release_winml_handle(handle)` — `atexit` callback, calls `handle.__exit__(None, None, None)`.
- `_get_catalog()` — `@functools.cache`'d singleton getter; returns `ExecutionProviderCatalog.get_default()` or `None` on three failure modes (ImportError, `initialize()` raises, `get_default()` raises). The `initialize().__enter__()` and `atexit.register` happen here.
- `_winml_warn_once(key, msg, *args)` — emit WARN first time, DEBUG thereafter.

### `WinMLCatalogSource(EPSource)` (lines 787-954)

Fields: `catalog_name: str`, `eps: tuple[str, ...]`, `auto_download: bool = False`. `resolve()` walks `catalog.find_all_providers()`, dispatches to `_resolve_provider`. The provider filter is `getattr(provider, "name", None) == self.catalog_name`. Skips `NotPresent` providers when `auto_download=False`. Calls `provider.ensure_ready_async().get()`, checks `result.status == Success`, reads `provider.library_path`, yields `EPEntry(version=None)` for each ep. Two static helpers `_is_not_present` and `_is_success` use casing-insensitive substring suffix matches (`name.replace("_", "").lower().endswith("notpresent" / "success")`) to handle WinAppSDK 2.0's UPPER_SNAKE vs PascalCase enum names.

### WinRT PackageManager plumbing (lines 957-997)

- `_get_pkg_manager()` — `@functools.cache`'d singleton, `PackageManager()` from `winrt.windows.management.deployment`.
- `_pkg_version_tuple(version) -> tuple[int, int, int, int]` — converts `PackageVersion` to comparable tuple via `getattr`.
- `_pkg_version_str(version) -> str` — `"M.m.b.r"` rendering.

### `MSIXPackageSource(EPSource)` (lines 1000-1113)

Fields: `family_name_prefix: str`, `relative_dll: str`, `eps: tuple[str, ...]`, `version: str | None = None`. Same `\` validation as `NuGetSource.resolve`. Algorithm: `manager.find_packages_by_user_security_id("")` → filter by `id.family_name.startswith(prefix)` → if `self.version` set, filter to exact version string match → `max(matching, key=_pkg_version_tuple)` picks winner → check DLL existence → yield with `version=_pkg_version_str(selected.id.version)`.

### `_list_msix_eps(family_name_prefixes=(...))` (lines 1116-1228)

Live MSIX enumeration. Returns one `MSIXPackageSource` per matched (family, version) found. Auto-detects EP name from the DLL filename inside each package's `ExecutionProvider/` subdir (or shallow scan if absent), mapped through `EP_CATALOG.ep_for_dll`. Returns sources with `family_name_prefix=str(p.id.family_name)` (full PackageFamilyName, no trailing separator) AND `version=_pkg_version_str(p.id.version)` — pinning each enumerated row exactly. Default prefix tuple covers both `MicrosoftCorporationII.WinML.` and `WindowsWorkload.EP.` publishing channels.

### `_default_ep_sources() -> list[EPSource]` (lines 1234-1351)

Returns the canonical 8-source-prefix-plus-MSIX-spread (2 PyPI + 2 NuGet + 5 WinMLCatalog + 2 DirectorySource[env_var] + `*_list_msix_eps()`). Order encodes precedence: PyPI > NuGet > WinMLCatalog > DirectorySource > MSIXPackageSource live enumeration.

### `_parse_winmlcli_ep_path() -> list[EPSource]` (lines 1358-1392)

Parses the `WINMLCLI_EP_PATH` env var (using `os.pathsep`), constructing one `DirectorySource` per `EP_CATALOG.all_eps()` ⊗ entry — i.e., for each path entry it produces N `DirectorySource` instances, one per known EP with `dll_patterns={ep: dll_name}`. Skips entries whose path is not a directory (with WARN). Skips bundled EPs in the catalog loop (those have `dll_name == ""`).

### `discover_all_eps()` (lines 1398-1503)

```python
def discover_all_eps(
    extra_sources: list[EPSource] | None = None,
    *,
    extra_sources_after: list[EPSource] | None = None,
) -> list[EPEntry]:
```

Composition order: `extra_sources` → `_parse_winmlcli_ep_path()` → `_default_ep_sources()` → `extra_sources_after`. Per-source defensive try/except wraps; per-entry defensive try/except wraps for `NotImplementedError` (DEBUG) and bare `Exception` (ERROR). Dedup is `(ep_name, os.path.normcase(os.path.normpath(str(dll_path))))` — the first hit per pair wins, later hits silently dropped at DEBUG. Status reassignment: first occurrence of `ep_name` gets `status="primary"` (or its passed-through primary), subsequent hits get `status="shadowed"` via `dataclasses.replace`.

## Behavior / contract changes

1. **`EPCatalog` is immutable post-init.** Setting any attribute after `__init__` raises `AttributeError`. Tests that need to swap catalog state must construct a fresh `EPCatalog` and patch the module-level `EP_CATALOG` binding (documented in the class docstring).
2. **Built-in EPs flow through the same discovery pipeline as plugins** via the synthesized `BuiltinSource` entries injected by `WinMLEPRegistry.__init__`. The `BuiltinSource.resolve()` deliberately yields nothing — discovery in this module is filesystem-only; built-ins live as out-of-band `EPEntry` constructions.
3. **The `version` field on `EPEntry` has source-specific semantics:** PyPI uses `metadata.version`, NuGet uses the folder name verbatim, MSIXPackage uses the `Package.Id.Version` rendered as `M.m.b.r`. `DirectorySource` and `WinMLCatalogSource` always produce `version=None` (deferred per OQ-2).
4. **`EPEntry.is_filesystem_backed()` is the only protection against the BuiltinSource's `Path("")` sentinel** being silently dropped by `dll_path.is_file()` in `discover_all_eps`. Callers that themselves stat the path (`commands/sys.py`'s renderer) must duplicate this check or share the helper.
5. **Dedup is path-canonicalized but not symlink-resolved.** `os.path.normpath` + `os.path.normcase` collapse `C:\Foo` vs `c:\foo` (Windows) and trailing-slash variants. Two sources legitimately pointing at the same DLL via different mount points or junctions still count as distinct — would need `Path.resolve()` to fix.
6. **`WINMLCLI_EP_PATH` produces one DirectorySource per (path × EP) cross-product.** A single path with 5 known plugin EPs in the catalog becomes 5 sources, each globbing for one filename. Inefficient but harmless because `DirectorySource.resolve()` returns empty on a no-match glob.
7. **`_winml_catalog_warned_keys` is module-mutable.** The dedup set is not reset between tests; long test runs accumulate keys monotonically. This is internal but worth knowing.

## Cross-file impact

- `src/winml/modelkit/session/ep_registry.py` imports `BuiltinSource`, `EPEntry`, `discover_all_eps` directly; the `_entry_source_tag` helper inside `ep_registry.py` `isinstance`-checks against every concrete subclass.
- `src/winml/modelkit/commands/sys.py` consumes `discover_all_eps()` and `EP_CATALOG.is_compatible` for the inventory render.
- `src/winml/modelkit/winml.py` legacy shim is documented in §1 of the module docstring as a consumer.
- The 8 `EP_CATALOG` rows are hardcoded but the data is loadable from configs.py via existing channels — nothing here pulls externally.
- `EP_CATALOG.is_compatible` transitively calls `_get_detected_vendors` which imports `from .sysinfo.hardware import GPU, NPU` lazily. Headless servers raise `RuntimeError`; callers (`default_ep_for_device`, `auto_detect_device` in `ep_device.py`) must catch.

## Risks / subtleties

1. **The catalog's MIGraphX `dll_name` is TODO/unverified** (line 170-172 in-file comment). The catalog row is built with the guessed name `onnxruntime_providers_migraphx.dll`. If wrong, `_list_msix_eps` will skip valid MIGraphX MSIX packages because `EP_CATALOG.ep_for_dll(dll.name)` returns `None`.
2. **No `arch_resolver` for `NuGetSource`** despite the field existing on the dataclass: the default sources hardcode `win-x64` (`Intel.ML.OnnxRuntime.EP.OpenVINO`) and `win-arm64` (`Qualcomm.ML.OnnxRuntime.QNN`) into `relative_dll`. On an arm64 host the OpenVINO NuGet source silently produces no entry; on x64 the QNN source silently produces no entry. The arch detection that exists for PyPI is not wired into NuGet defaults.
3. **`_get_catalog()`'s `atexit` cleanup may run during interpreter shutdown** when the WinAppSDK runtime is already torn down. The `except Exception` swallows the failure at DEBUG (the comment notes `# pragma: no cover`). Acceptable but invisible.
4. **`WinMLCatalogSource._resolve_provider` calls `ensure_ready_async().get()` synchronously.** That blocks the discovery walk if a network download is in progress. With `auto_download=False` (default) this is rare but possible (Windows Update may be staging the package mid-discovery). No timeout.
5. **`_list_msix_eps` shallow-scans `installed_path / "ExecutionProvider"` first, then does `installed_path.glob("**/onnxruntime_providers_*.dll")` as a fallback.** The recursive glob could hit network mount points or symlinks if MSIX packages ever ship that way. Default packages stay flat, but the fallback is unbounded.
6. **`EPCatalog.__setattr__` only protects against attribute *rebinding*, not against mutating the proxied dicts** (which `MappingProxyType` already handles) — together they're tight, but a `del` of `_by_name` is undefined behavior because `__delattr__` is not overridden. Probably fine because `__slots__` prevents arbitrary attribute creation.
7. **The `extra_sources_after` parameter in `discover_all_eps` is keyword-only, but `extra_sources` is positional.** Asymmetric API. The intent seems to be "tests inject high-precedence overrides; CLI inventory inject low-precedence reveals" — fine, but documentable only via the docstring.
8. **`logger.error("Source ... failed mid-iteration: %s", ...)`** in the outer try/except continues to the next source, silently losing entries that *would* have followed in the same iterator. Could mask real bugs (an MSIX source that emits 4 valid entries then raises on the 5th drops the prior 4 from the walk? No — they're already in `result` by then because the inner `for entry in it` appends as it goes. The drop is only the unconsumed tail). OK.

## Simplification opportunities

1. **`_winml_catalog_warned_keys` is a module-level mutable set.** Could be a `_winml_warn_once` closure variable, since nothing else reads it. As-is it's testable via direct mutation; encapsulating is a trade-off.
2. **`PyPISource.resolve()`'s `metadata.version(self.distribution)` call** (line 432) is a second metadata lookup after the `metadata.distribution(self.distribution)` on line 411. The latter returns a `Distribution` object whose `.version` attribute is the same string. Replacing `metadata.version(self.distribution)` with `dist.version` removes one redundant lookup and one try/except.
3. **`EPCatalog._by_dll` could be computed lazily** (via `cached_property` once on a non-frozen container, or via a fresh-set property on the immutable class). As-is the eager build at `__init__` runs once and stays cheap.
4. **`_default_ep_sources()` constructs the same `tuple[str, ...]` literal for every WinMLCatalogSource and PyPISource's `eps` field.** Could share a module-level `_OPENVINO_EPS = ("OpenVINOExecutionProvider",)` constant if extracted. Marginal.
5. **`NuGetSource.resolve` builds `candidates` (sorted by version desc) AND a `stable` filter, then chooses one or the other.** Could collapse to one loop (filter+sort once). The current shape mirrors human reasoning ("first prefer stable, then fall back") but the algorithmic redundancy could be simplified.
6. **`_parse_winmlcli_ep_path` constructs a new `DirectorySource` per EP per path entry** rather than one `DirectorySource` with `dll_patterns={ep1: dll1, ep2: dll2, ...}`. The latter is exactly what `DirectorySource.dll_patterns` was designed for. Net: N×M sources instead of N, no correctness issue but more redundant `is_dir()` checks.
7. **`discover_all_eps` has two separate inner exception handlers** — one wrapping `source.resolve()` (line 1438) and one wrapping `for entry in it` (line 1448). Both catch `NotImplementedError` separately. Single combined `with contextlib.suppress(...)`-style wrap would be cleaner; current shape is verbose but explicit.
8. **`_qnn_arch_resolver` is a top-level free function** used by exactly one `PyPISource` row in `_default_ep_sources`. Could be a lambda inline. Currently named for testability.
9. **`_get_detected_vendors`'s nested try/except wrapping `cls.get_all()`** (line 211) raises a fresh `RuntimeError` for each cls. If GPU works but NPU raises, the function still raises — and `functools.cache` doesn't cache an exception, so subsequent calls re-attempt. Acceptable but the doc-comment about "preventing functools.cache from pinning an empty-set fallback" applies only to the *success path* — exceptions are not cached by `functools.cache` regardless.
10. **`EPEntry.is_filesystem_backed()` is a one-line `isinstance` check.** Could be a property. Could also be inlined at the two call sites (`discover_all_eps` and `commands/sys.py`'s renderer) — both check it once. The method exists for self-documentation; not strictly necessary.

## Open questions / TODOs surfaced

- TODO `ep_path #4` (lines 170-172): "MIGraphX DLL leaf is unverified; mirrors the VitisAI naming convention. Confirm by inspecting an installed MSIX." — must be resolved before MIGraphX MSIX packages will be discovered.
- TODO `ep_path` (lines 282-283): "OQ-2 deferred — provider.version probing." `WinMLCatalogSource` always emits `version=None`. The MSIX `_list_msix_eps` enumeration does carry version; the catalog API path does not.
- No NuGet `arch_resolver` for the default sources — implicit "win-x64 / win-arm64" hardcode in the relative paths. Should be parameterized for cross-arch hosts.
- The `WinMLCatalogSource.auto_download` field is on the dataclass but every default source uses `auto_download=False`. There's no CLI / env to flip it. Acceptable v1 posture, but mark as not-yet-wired.
- `EPCatalog`'s vendor compat is binary (substring match). No notion of "compatible-but-degraded" hardware (e.g., an old Intel iGPU that OpenVINO supports but with reduced ops).
- `EP_CATALOG`'s 8 rows are hardcoded constants — the docstring on `EPCatalog` says "tests swap by constructing a fresh `EPCatalog` and patching the module-level `EP_CATALOG` binding," which is the pragmatic out, but the canonical catalog row set has no externalization path (TOML, JSON, etc.) and would need a code change to add an EP.
