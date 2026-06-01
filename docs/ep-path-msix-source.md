# MsixPackageSource: Version-pinned MSIX EP Discovery

Status: implemented in commit `f79e484e` (and accompanying ABC refactor `ef9a5bdd`, CLI inventory `35dd39a6`, dep switch `eab52093`), as of 2026-05-08. Companion to [`docs/ep-path-design.md`](ep-path-design.md) (the unified `EP_PATH` design) and [`docs/winml-ep-empirical-findings.md`](winml-ep-empirical-findings.md) (live-machine evidence for WinML EP behavior). This document specifies a fourth `EpSource` variant — `MsixPackageSource` — that bypasses the WinAppSDK `ExecutionProviderCatalog` to load a specific installed MSIX EP package version, plus a sibling helper `list_msix_eps()` for inventory/diagnostics. Also covers the `WINMLCLI_EP_PATH` env-var rename.

## Executive summary

- `WinMLCatalogSource` (the existing MSIX-via-catalog source) cannot reach more than one version of any given EP. The IDL contract specifies `ExecutionProvider.PackageId` as a scalar (one package per provider object) and `IExecutionProviderCatalog.FindAllProviders` returns one entry per **provider name**, not per installed package. Empirical proof on the dev box: two QNN EP MSIX packages installed (`...QNN.EP.1.8` v1.8.30.0, `...QNN.EP.2` v2.2420.44.0), `find_all_providers()` returned a single `QNNExecutionProvider` entry resolving to v2 only.
- This is fine for the typical user — Windows curates which version is "current" — but blocks regression isolation, multi-tenant pinning, and any workflow that needs a non-current version.
- This document specifies `MsixPackageSource`, a new `EpSource` that goes directly to `Windows.Management.Deployment.PackageManager` and selects packages by **family-name prefix** (the natural granularity at which Microsoft partitions major version lines). The same data structure is returned by a sibling helper `list_msix_eps()`, so a CLI inventory output can be copy-pasted into `EP_PATH` configuration.
- **Single CLI surface**: the existing `winml sys --list-ep` command becomes a comprehensive inventory of every EP discoverable on the system (PyPI, MSIX-via-catalog, MSIX-via-package-manager, filesystem) with `[primary]` / `[shadowed]` / `[incompatible]` status tags. **There is no separate `--list-msix-eps` flag** — `--list-ep` is the single discovery command. This uses `discover_all_eps()` to retrieve all matches per EP, with the first entry in each group marked as primary and the rest as shadowed.
- Renames `WINML_EP_PATH` (added in this branch, not yet shipped) to `WINMLCLI_EP_PATH` to match the tool's identity. No back-compat alias.

## Background: why the catalog is insufficient

Live evidence on the dev machine (Snapdragon X Elite, wasdk 2.0.1, Windows App Runtime 2.0):

```text
$ Get-AppxPackage | Where-Object { $_.Name -like '*QNN.EP*' } | Select Name, PackageFamilyName, Version

Name              : MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.1.8
PackageFamilyName : MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.1.8_8wekyb3d8bbwe
Version           : 1.8.30.0

Name              : MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.2
PackageFamilyName : MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.2_8wekyb3d8bbwe
Version           : 2.2420.44.0
```

```text
$ uv run python -c "..."  # via cat.find_all_providers()
len(find_all_providers()) = 1
[0] name='QNNExecutionProvider'
    library_path = C:\Program Files\WindowsApps\
                   MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.2_2.2420.44.0_arm64__8wekyb3d8bbwe\
                   ExecutionProvider\onnxruntime_providers_qnn.dll
    package_id.version = PackageVersion(major=2, minor=2420, build=44, revision=0)
```

The IDL ([`microsoft/dynwinrt` API metadata](https://github.com/microsoft/dynwinrt/blob/main/tools/dynwinrt-codegen/api-docs/Microsoft.Windows.AI.Machinelearning.xml), [`m417z/WindowsAppSDK-Index` 2.0.300 IDL](https://github.com/m417z/WindowsAppSDK-Index/blob/main/Microsoft.WindowsAppSDK/deps/Microsoft.Windows.AI.MachineLearning/2.0.300/metadata/Microsoft.Windows.AI.MachineLearning.winmd_winmdidl/Microsoft.Windows.AI.MachineLearning.idl)):

```idl
HRESULT FindAllProviders([out] UINT32* __resultSize,
  [out] [retval] [size_is(, *__resultSize)]
  Microsoft.Windows.AI.MachineLearning.ExecutionProvider*** result);

runtimeclass ExecutionProvider {
    HSTRING                                Name;
    HSTRING                                LibraryPath;
    Windows.ApplicationModel.PackageId     PackageId;        // scalar, not collection
    ExecutionProviderReadyState            ReadyState;
    ExecutionProviderCertification         Certification;
}
```

`PackageId` is scalar; the catalog has no method to enumerate versions. Going below the catalog to the package manager is the only path. The actual `ExecutionProviderCatalog` runtimeclass implementation is closed-source (not in the public `microsoft/WindowsAppSDK` repo, which carries only IDL specs and bootstrap infrastructure), but the IDL contract is binding regardless of the impl.

## Microsoft's MSIX naming: family name encodes the major version line

Verified empirically and documented in [`docs/winml-ep-empirical-findings.md`](winml-ep-empirical-findings.md):

```text
Family name format: <Name>_<PublisherId>
Full name format:   <Name>_<Version>_<Architecture>__<PublisherId>

  Family: MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.1.8_8wekyb3d8bbwe
  Full:   MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.1.8_1.8.30.0_arm64__8wekyb3d8bbwe

  Family: MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.2_8wekyb3d8bbwe
  Full:   MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.2_2.2420.44.0_arm64__8wekyb3d8bbwe
```

Two observations that drive the design:

1. **The family name embeds the major version line.** v1.8 and v2 are two separate package families. Pinning at family-name granularity = pinning a major-version line. This is Microsoft's convention, not ours.
2. **The DLL relative path is identical across versions.** Both `...QNN.EP.1.8` and `...QNN.EP.2` ship the EP at `<InstalledPath>\ExecutionProvider\onnxruntime_providers_qnn.dll`. So `relative_dll` is constant within an EP, not version-specific.

## Design

### `MsixPackageSource` dataclass

```python
@dataclass(frozen=True)
class MsixPackageSource:
    """An MSIX-delivered EP, identified by package-family-name prefix.

    Bypasses the WinAppSDK ExecutionProviderCatalog (which exposes only
    one version per EP-name) to load a specific installed MSIX package
    version. Use when you need to pin a non-current EP version
    (compat testing, regression isolation, multi-tenant scenarios).

    Args:
        family_name_prefix: Prefix matched against installed-package
            PackageFamilyName. Granularity decides what gets pinned —
            "MicrosoftCorporationII.WinML.Qualcomm.QNN.EP." spans both
            v1.8 and v2 families; "...QNN.EP.1.8_" pins to the v1.8 line
            (any build); "...QNN.EP.1.8_8wekyb3d8bbwe" pins to one
            family exactly. The trailing character ('.' or '_') is the
            user's disambiguator against future name collisions
            (e.g., a hypothetical "EP.10_" family).
        relative_dll: POSIX-style relative path inside the package's
            InstalledPath. For QNN EP MSIX (verified):
            "ExecutionProvider/onnxruntime_providers_qnn.dll".
        eps: Canonical EP names this package provides. Always declared
            explicitly — auto-detection is for the listing helper only.
        version: Optional secondary pin to one exact installed Version
            (e.g. "1.8.30.0"). When None (typical), the highest installed
            version within any family matched by family_name_prefix wins.
    """
    family_name_prefix: str
    relative_dll: str
    eps: tuple[str, ...]
    version: str | None = None

    def resolve(self) -> Iterator[tuple[str, Path]]: ...
```

### Same dataclass returned by the inventory helper

```python
def list_msix_eps(
    family_name_prefix: str = "MicrosoftCorporationII.WinML.",
) -> list[MsixPackageSource]:
    """Enumerate installed MSIX EP packages.

    Returns one fully-pinned MsixPackageSource per (family, version)
    found. Each returned value is EP_PATH-ready (drop into the list)
    and resolvable via .resolve().

    EP names are auto-detected from the DLL filename inside each package,
    using the inverse of EP_DLL_NAMES. Packages with no recognizable EP
    DLL are skipped silently.

    Args:
        family_name_prefix: Default catches all WinML-catalog EP MSIXes
            published by Microsoft. Override with a narrower prefix to
            filter (e.g., "MicrosoftCorporationII.WinML.Qualcomm." for
            QNN-only listings, or "MicrosoftCorporationII.WinML.Intel."
            for OpenVINO-only).
    """
```

The result is a list of fully-pinned `MsixPackageSource` instances. Each one has `family_name_prefix` set to the exact PackageFamilyName (with trailing `_8wekyb3d8bbwe` included) and `version` set to the exact installed `Package.Id.Version`. So copy-pasting from CLI output into `EP_PATH` config yields a deterministic, version-locked source.

### Resolution algorithm

`MsixPackageSource.resolve()`:

```text
1. Get the cached PackageManager (functools.cache, same pattern as _get_catalog).
2. packages = manager.find_packages_for_user("")   # current user
3. matching = [p for p in packages if p.id.family_name.startswith(family_name_prefix)]
4. If self.version is set: matching = [p for p in matching if str(p.id.version) == self.version]
5. If matching is empty: log DEBUG, return (yield nothing).
6. selected = max(matching, key=lambda p: tuple(p.id.version))
7. dll = Path(selected.installed_path) / self.relative_dll
8. If not dll.is_file(): log WARN once per (prefix, version), return.
9. for ep in self.eps: yield (ep, dll)
```

`list_msix_eps()`:

```text
1. Get the cached PackageManager.
2. packages = manager.find_packages_for_user("")
3. matching = [p for p in packages if p.id.family_name.startswith(family_name_prefix)]
4. results: list[MsixPackageSource] = []
5. For each p in matching (sorted by family_name then descending version):
     scan p.installed_path for any onnxruntime_providers_*.dll
     map dll.name -> ep_name via inverse of EP_DLL_NAMES
     if no match: skip (log DEBUG)
     results.append(MsixPackageSource(
         family_name_prefix=p.id.family_name + "_",  # trailing _ for round-trip safety  TODO see below
         relative_dll=str(dll.relative_to(p.installed_path)).replace("\\", "/"),
         eps=(ep_name,),
         version=str(p.id.version),
     ))
6. Return results.
```

`★` **Round-trip representation note**: when listing, we have an exact full PackageFamilyName (e.g., `...QNN.EP.2_8wekyb3d8bbwe`). To make the returned `MsixPackageSource` *exactly* pinned, the `family_name_prefix` field is set to `<full_family_name>_` so `startswith()` matches only that family. Documented; not magic. Alternative considered (separate `family_name_exact` field) was rejected on YAGNI grounds — one field, one prefix-matching rule.

### EP-name auto-detection table

The inverse of the existing `EP_DLL_NAMES` (in [`src/winml/modelkit/ep_path.py`](../src/winml/modelkit/ep_path.py:95)):

```python
# Reverse lookup: dll_filename -> canonical_ep_name
_DLL_TO_EP_NAME: dict[str, str] = {
    dll: ep
    for ep, dll_list in EP_DLL_NAMES.items()
    for dll in dll_list
}
```

Lives in `ep_path.py` next to `EP_DLL_NAMES`, derived once at module load. `list_msix_eps()` consults it to map a found DLL filename back to its canonical EP name. Future EPs only need to be added to `EP_DLL_NAMES`; the reverse map updates automatically.

### Caching

`PackageManager` is a WinRT object — instantiation has nontrivial cost (COM initialization). Same singleton pattern as `_get_catalog()`:

```python
@functools.cache
def _get_pkg_manager() -> Any | None:
    """Return cached PackageManager or None if WinRT binding unavailable."""
    try:
        from winrt.windows.management.deployment import PackageManager
    except ImportError as e:
        logger.debug("MsixPackageSource: WinRT PackageManager not available (%s)", e)
        return None
    try:
        return PackageManager()
    except Exception as e:
        logger.warning("MsixPackageSource: PackageManager() failed: %s", e)
        return None
```

Tests reset via `_get_pkg_manager.cache_clear()`, parallel to the existing `_get_catalog` fixture pattern.

### Error modes

| Condition | Severity | Action |
|---|---|---|
| WinRT binding not importable (no `winrt-windows.management.deployment`) | DEBUG, once | Yield nothing. Same opt-in semantics as `WinMLCatalogSource` — install via `[winml-catalog]` extra. |
| `PackageManager()` instantiation raises | WARN, once | Yield nothing. |
| `find_packages_for_user("")` raises | WARN, once per (prefix) | Yield nothing. |
| No packages match prefix | DEBUG | Yield nothing. (Not an error — user may have written speculative `EP_PATH` covering EPs not installed on this machine.) |
| Version pin set but no matching version | DEBUG, with the available versions logged | Yield nothing. |
| Selected package has no DLL at `relative_dll` | WARN, once per (prefix, version) | Yield nothing. (Suggests a wrong `relative_dll` for this vendor.) |
| `installed_path` not readable | WARN, once per (prefix, version) | Yield nothing. (Should not happen for user's own packages on modern Windows; flag for diagnosis.) |

### Default `EP_PATH` is unchanged

`MsixPackageSource` is **not** added to the default `EP_PATH`. The default keeps `[PyPiSource, WinMLCatalogSource, FilesystemSource]` — Microsoft's catalog opinion still wins by default. `MsixPackageSource` is **opt-in only**: users who want pinning add their own rows via `extra_sources=` or by appending to `EP_PATH` directly.

Reasoning:
1. **Reproducibility**: defaulting `MsixPackageSource` would make builds non-reproducible across machines (different MSIX versions installed → different EP picked). The catalog's "follow Windows Update's choice" is a more predictable default for non-pinning users.
2. **Backward compatibility**: existing users get exactly the same behavior. Pinning is an explicit choice.
3. **Discoverability**: `winml sys --list-ep` (see CLI section below) makes the inventory visible; users discover what's available, then opt in.

## CLI integration: `winml sys --list-ep` is the single inventory command

**Single CLI surface, comprehensive inventory.** `winml sys --list-ep` is the *only* command for listing execution providers, and it lists **every EP discoverable on the system** — across every source (PyPI wheels, MSIX catalog, MSIX package manager scan, filesystem drops). No separate `--list-msix-eps` flag.

Today's `--list-ep` (pre–this design) shows only the resolved set: one entry per EP name after `discover_eps()` first-hit-wins. That's misleading — a user with both PyPI `onnxruntime-qnn` and MSIX QNN EP installed sees only the PyPI hit, with no indication that one or two MSIX QNN packages are also available on the box. This design fixes that: `--list-ep` becomes a comprehensive inventory.

### Behavior

`--list-ep` walks **every** source of EP DLLs the system knows about and reports them all, grouped by canonical EP name, annotated with status. **Two orthogonal axes**:

**Axis 1 — resolution rank** (always shown):

| Tag | Meaning |
|---|---|
| `[primary]` | This is the entry `discover_eps()` would load (won first-hit-wins). |
| `[shadowed]` | Resolvable, but a higher-precedence source provides the same EP. |
| `[not-installed]` | Source is configured but the underlying package/wheel/dir is absent. Shown only with `--verbose`. |

**Axis 2 — hardware compatibility** (negative case only):

| Tag | Meaning |
|---|---|
| _(no tag)_ | Default. Compatible vendor hardware detected, or EP has no vendor requirement (e.g., CPU, DML, Azure). |
| `[incompatible]` | EP requires hardware from a vendor not present on this machine (e.g., OpenVINO on a Snapdragon-only box). |

Compatible status is the assumed default — only the *negative* case is tagged, to keep the output uncluttered.

### Output

Output on a Snapdragon X Elite dev box (QNN + OpenVINO PyPI + 2× QNN MSIX installed):

```text
$ winml sys --list-ep

Available Execution Providers

  QNNExecutionProvider                              -> Qualcomm NPU
    [primary]    PyPI    onnxruntime-qnn 2.1.0
                         Path: .venv\...\onnxruntime_qnn\libs\amd64\onnxruntime_providers_qnn.dll
    [shadowed]   MSIX    MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.2  v2.2420.44.0  (catalog default)
                         Path: ...QNN.EP.2_..\ExecutionProvider\onnxruntime_providers_qnn.dll
    [shadowed]   MSIX    MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.1.8  v1.8.30.0
                         Path: ...QNN.EP.1.8_..\ExecutionProvider\onnxruntime_providers_qnn.dll

  OpenVINOExecutionProvider  [incompatible]         -> Intel NPU/GPU/CPU
    [primary]    PyPI    onnxruntime-ep-openvino 1.4.0
                         Path: .venv\...\onnxruntime_ep_openvino\onnxruntime_providers_openvino_plugin.dll

  CPUExecutionProvider                              -> CPU
    [primary]    built-in
```

Notes on the output:

- **One section per EP name**, sorted by precedence of the primary source (most-likely-used EPs lead).
- **`[incompatible]` is a section-level tag** (next to the EP name) because it depends only on the EP and the machine, not on which source provides it.
- **Within a section**, entries are sorted by precedence: `[primary]` first, then `[shadowed]` in `EP_PATH` order, then `[not-installed]` (with `--verbose`).
- **Vendor-qualified device-types** (`Qualcomm NPU`, `Intel NPU/GPU/CPU`) replace bare `NPU/GPU/CPU` to remove cross-vendor ambiguity.
- **Source-kind column** (`PyPI` / `MSIX` / `Filesystem` / `built-in`) at fixed offset for visual scanning.
- **`(catalog default)`** annotation on the MSIX entry tells the user *that specific MSIX is what `WinMLCatalogSource` would pick* — useful for understanding why the catalog chose v2 over v1.8.
- **JSON output** via `--format json` returns a structured shape:

  ```json
  {
    "QNNExecutionProvider": {
      "compatible": true,
      "device_types": "Qualcomm NPU",
      "entries": [
        {
          "status": "primary",
          "source_kind": "PyPiSource",
          "distribution": "onnxruntime-qnn",
          "distribution_version": "2.1.0",
          "dll_path": "..."
        },
        {
          "status": "shadowed",
          "source_kind": "MsixPackageSource",
          "family_name": "MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.2_8wekyb3d8bbwe",
          "version": "2.2420.44.0",
          "installed_path": "...",
          "dll_path": "...",
          "is_catalog_default": true
        }
      ]
    },
    "OpenVINOExecutionProvider": {
      "compatible": false,
      "device_types": "Intel NPU/GPU/CPU",
      "entries": [...]
    }
  }
  ```

### Hardware-compatibility check: `is_compatible()` on each `EpSource`

Each `EpSource` subclass exposes a one-line `is_compatible()` method that delegates to a shared `_ep_is_compatible(ep_name)` helper. The actual compat rule is centralized:

```python
# Single source of truth — simple name-matching to detected hardware vendors.
_EP_VENDOR_REQUIREMENT: dict[str, set[str]] = {
    "QNNExecutionProvider":           {"Qualcomm"},
    "OpenVINOExecutionProvider":      {"Intel"},
    "VitisAIExecutionProvider":       {"AMD"},
    "MIGraphXExecutionProvider":      {"AMD"},
    "NvTensorRtRtxExecutionProvider": {"NVIDIA"},
    "DmlExecutionProvider":           set(),  # any GPU
    "CPUExecutionProvider":           set(),  # always present
    "AzureExecutionProvider":         set(),  # cloud, no local hw req
}


@functools.cache
def _get_detected_vendors() -> frozenset[str]:
    """Cached query of vendor strings reported by sysinfo hardware enumeration.
    Returns the union of GPU + NPU + CPU vendor strings (e.g.
    {'Qualcomm', 'Microsoft'} on a Snapdragon X Elite)."""
    ...


def _ep_is_compatible(ep_name: str) -> bool:
    """Case-insensitive substring match: any required vendor present in any
    detected vendor string. Unknown EP names are assumed compatible (graceful
    forward-compat for new EPs not yet in the table)."""
    required = _EP_VENDOR_REQUIREMENT.get(ep_name, set())
    if not required:
        return True
    detected = _get_detected_vendors()
    return any(req.lower() in v.lower() for req in required for v in detected)


# On every EpSource subclass — same one-line method:
@dataclass(frozen=True)
class PyPiSource:
    ...
    def is_compatible(self) -> bool:
        """True iff every EP in self.eps is compatible with this machine."""
        return all(_ep_is_compatible(ep) for ep in self.eps)
```

`FilesystemSource`, `WinMLCatalogSource`, and `MsixPackageSource` all get the identical method body. The CLI handler calls `source.is_compatible()` per resolved source; the result becomes the `[incompatible]` section-level tag (true iff *all* sources for an EP-name agree). Adding a new EP requires only one line in `_EP_VENDOR_REQUIREMENT` — no per-source-class changes.

**Matching is intentionally substring-based and case-insensitive.** Vendor strings reported by Windows differ across systems (`"Intel(R) Corporation"`, `"Intel"`, `"Intel Corp"`, etc.). Substring on `"Intel"` matches all variants. Misclassification risk is low because vendor name overlaps are rare (no IHV's name is a substring of another's in practice).

**Tests** (`tests/unit/ep_path/test_compat.py`):
- Each entry in `_EP_VENDOR_REQUIREMENT` round-trips: vendor in detected → compatible; not in detected → incompatible; empty requirement → always compatible.
- Substring case-insensitivity (`"intel"` matches `"Intel(R) Corporation"`).
- Unknown EP names default to compatible (forward-compat).
- `_get_detected_vendors()` cache reset for hermetic tests.

### Implementation: how `--list-ep` reaches every source

Two companion functions handle discovery:

- **`discover_eps(extra_sources=...)`** walks the EP_PATH and returns one `ResolvedEp` per EP name (first-hit-wins precedence). Default behavior for routine EP registration.

- **`discover_all_eps(extra_sources=...)`** walks the EP_PATH and returns all matching `ResolvedEp` entries per EP name, grouped as `dict[str, list[ResolvedEp]]`. Each list is sorted by precedence: the first entry is the primary (winner), and any additional entries are shadowed. Used by the CLI inventory command.

`ResolvedEp` is a small dataclass:

```python
@dataclass(frozen=True)
class ResolvedEp:
    ep_name: str
    dll_path: Path
    source: EpSource         # for kind-tagging in CLI output
    status: str              # "primary" | "shadowed"
```

(The CLI may surface a third ``"incompatible"`` status when the EP
section's ``is_compatible()`` is False — that override is applied
in ``commands/sys.py``, not in ``discover_all_eps`` itself.)

To cover **non-current MSIX versions** (which `WinMLCatalogSource` collapses to one), `--list-ep` injects `list_msix_eps()` results as additional sources for the discovery walk. Note: appended via `extra_sources_after`, *not* prepended — MSIX entries should appear `[shadowed]`, not artificially override PyPI/Catalog precedence:

```python
# Inside `winml sys --list-ep` handler:
discovered_msix = list_msix_eps()  # one fully-pinned MsixPackageSource per (family, version)
shadowed_view = discover_all_eps(extra_sources_after=discovered_msix)
```

`list_msix_eps()` is exposed as a Python helper for use from the CLI; it is NOT plumbed into the default `EP_PATH` (per "Default `EP_PATH` is unchanged" above). That keeps day-to-day `discover_eps()` behavior identical to today — only the CLI's inventory view sees the expanded set.

The `(catalog default)` annotation is computed by separately calling `WinMLCatalogSource.resolve()` for each EP name and matching the resulting `library_path` against the `MsixPackageSource` entries.

### What `--list-msix-eps` would have been

The earlier draft of this design proposed a separate `--list-msix-eps` flag for MSIX-specific inventory. Rejected: a single `--list-ep` covering every source is more discoverable, and a user looking for "what EPs do I have?" should not have to know in advance whether the answer involves MSIX. `list_msix_eps()` remains as a *Python-level* helper used by the CLI handler, but it is not a separate CLI surface.

## `WINMLCLI_EP_PATH` env-var rename

The `WINML_EP_PATH` env var (added in this branch, not yet shipped externally) was renamed first to `MODELKIT_EP_PATH`, then to `WINMLCLI_EP_PATH`. Same semantics:

- Path-list using OS-conventional separator (`;` on Windows, `:` on POSIX).
- Each entry treated as a directory; instantiated as a `FilesystemSource` with `dll_patterns` covering every entry in `EP_DLL_NAMES`.
- Resolved before the default `EP_PATH` (i.e., user override beats default).
- Empty/unset: no override.
- Non-existent entries log a WARN and are skipped.

Rationale: `WINMLCLI_*` unambiguously identifies the winml-cli tool. Since the var was never shipped externally, the rename has no migration cost.

Implementation: rename `_parse_modelkit_ep_path()` to `_parse_winmlcli_ep_path()`, change the `os.environ.get` key, update the docstring, update tests. No deprecation warning (nothing to deprecate — the var was never shipped).

## Test plan

### Unit tests for `MsixPackageSource.resolve()`

Mock `_get_pkg_manager()` to return a fake `PackageManager` whose `find_packages_for_user` returns synthetic `Package` objects with controlled `id.family_name`, `id.version`, and `installed_path`. No live PackageManager calls in unit tests — same isolation pattern as `WinMLCatalogSource` uses for its fake binding.

Required cases:

- Single matching package, no version pin → yields the DLL.
- Multiple matching packages, no version pin → yields highest version.
- Version pin matching one package → yields that one.
- Version pin matching no package → yields nothing, DEBUG logged.
- No matching packages → yields nothing, DEBUG logged.
- DLL missing inside matched package → yields nothing, WARN logged.
- WinRT binding unavailable (`_get_pkg_manager` returns None) → yields nothing, DEBUG once.
- `family_name_prefix` exact match (full family name + `_`) → matches only that family.
- `family_name_prefix` major-line prefix (`...QNN.EP.1.8_`) → matches any v1.8.x build.
- `family_name_prefix` cross-line prefix (`...QNN.EP.`) → matches both lines, highest wins.

### Unit tests for `list_msix_eps()`

Same fake-PackageManager pattern. Cases:

- Multiple packages across multiple EPs → returns one entry per (family, version) pair, with auto-detected `eps`.
- Package with no recognizable DLL → skipped silently (DEBUG log).
- Empty install state → returns empty list.
- Returned `MsixPackageSource` instances are usable: `.resolve()` on them yields the same DLL `list_msix_eps` reported.

### Integration test (Windows-only, opt-in)

One `pytest.mark.windows_msix` test that hits the real `PackageManager`:

- Asserts ≥1 package returned for the default prefix on a machine with WinML EP MSIXes installed.
- Each returned `MsixPackageSource` round-trips: `.resolve()` returns a DLL path that exists.
- Skip on machines without `winrt-windows.management.deployment` installed.

### Test for `WINMLCLI_EP_PATH` rename

Update existing `WINML_EP_PATH` test cases in `tests/unit/ep_path/test_ep_path.py` to use `WINMLCLI_EP_PATH`. Add one negative case: setting `WINML_EP_PATH` is a no-op (no warning, no parse) so users who try the old name see "nothing happened" cleanly.

### Tests for `discover_all_eps()`

In `tests/unit/ep_path/test_discover_all_eps.py`:

- **Single source per EP, no shadowing** → returns one `ResolvedEp` per name in the list, with `status="primary"`.
- **PyPI active + filesystem shadowed for same EP** → returns 2 entries in the same list, PyPI first with `status="primary"`, filesystem second with `status="shadowed"`.
- **Three sources stack** (PyPI + MSIX + filesystem) → returns 3 entries in `EP_PATH` order; only the first is `primary`.
- **`extra_sources` injection** (used by CLI to inject `list_msix_eps()` results) → injected sources participate in shadow detection at their precedence position.
- **Multiple EPs, mixed shadow patterns** → grouping is correct; each EP-name list is sorted by precedence.
- **Empty EP_PATH + empty extra_sources** → returns `{}`.

### Tests for the CLI: `winml sys --list-ep`

Golden-test the rendered output for a synthetic scenario reproducing the dev-machine state:
- Active: PyPI `onnxruntime-qnn 2.1.0`
- Shadowed: MSIX `MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.2 v2.2420.44.0` (catalog default)
- Shadowed: MSIX `MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.1.8 v1.8.30.0`
- Active: PyPI `onnxruntime-ep-openvino 1.4.0`
- Built-in: CPU

Mock `_get_pkg_manager`, `_get_catalog`, and `importlib.metadata.distribution` so the test is hermetic. Assert both human-readable text output and `--format json` shape.

## Implementation checklist

In dependency order:

1. **`pyproject.toml`** — add `winrt-windows-management-deployment` to the `[winml-catalog]` extra. Verify version compatibility with the existing `wasdk-*` pin.
2. **`src/winml/modelkit/ep_path.py`**:
   - Add `_get_pkg_manager()` with `@functools.cache`.
   - Add `_DLL_TO_EP_NAME` reverse-lookup dict (inverse of `EP_DLL_NAMES`).
   - Add `_EP_VENDOR_REQUIREMENT` table and `_ep_is_compatible(ep_name)` helper.
   - Add `_get_detected_vendors()` with `@functools.cache` (queries sysinfo hardware).
   - Add `MsixPackageSource` dataclass with `resolve()` and `is_compatible()`.
   - Add `is_compatible()` method to `PyPiSource`, `FilesystemSource`, `WinMLCatalogSource` (one line each, delegates to `_ep_is_compatible`).
   - Add `list_msix_eps()` helper.
   - Add `ResolvedEp` dataclass (ep_name, dll_path, source, status).
   - Add `discover_all_eps()` function. Returns `dict[str, list[ResolvedEp]]` with all matches per EP-name, sorted by precedence (primary first).
   - Update `EpSource` tagged union to include `MsixPackageSource`.
   - Update `__all__`.
   - Rename `_parse_winml_ep_path` → `_parse_winmlcli_ep_path`, update env-var key from `WINML_EP_PATH` to `WINMLCLI_EP_PATH`.
3. **`tests/unit/ep_path/test_msix_package_source.py`** — new file with unit tests above.
4. **`tests/unit/ep_path/test_discover_all_eps.py`** — new file covering `discover_all_eps()` semantics and the `[primary]`/`[shadowed]` ordering rules.
5. **`tests/unit/ep_path/test_compat.py`** — new file covering `_ep_is_compatible`, `_get_detected_vendors`, and `EpSource.is_compatible()` on all four source types.
6. **Update `tests/unit/ep_path/test_ep_path.py`** — rename `WINML_EP_PATH` references to `WINMLCLI_EP_PATH`. Add negative case (setting `WINML_EP_PATH` is a silent no-op).
7. **`src/winml/modelkit/cli/sys.py`** (or wherever `winml sys` is implemented) — extend the existing `--list-ep` flag to use `discover_all_eps(extra_sources_after=list_msix_eps())`, compute the `[incompatible]` section tag from `EpSource.is_compatible()`, and render the comprehensive inventory output described above. Update the `--list-ep --format json` shape.
8. **`tests/unit/cli/test_sys.py`** (if it exists; otherwise create) — golden-test the `--list-ep` output for a synthetic mixed scenario (compatible PyPI primary + 2 shadowed MSIX, plus an incompatible PyPI source on a Snapdragon-only mock). Mock `_get_pkg_manager`, `_get_catalog`, `_get_detected_vendors`.
9. **`docs/ep-path-design.md`** — add a "MsixPackageSource" subsection cross-referencing this doc, plus notes on the `discover_all_eps()` function and the `is_compatible()` API.

## Open TODOs

1. **DLL relative-path layout for non-QNN EP MSIX packages.** Verified for QNN: `ExecutionProvider/onnxruntime_providers_qnn.dll`. Unverified for OpenVINO, VitisAI, MIGraphX, NV TRT-RTX MSIX packages — none installed on the dev box. The `MsixPackageSource.relative_dll` field is intentionally required (no default) to force users to specify per vendor; `list_msix_eps()` discovers the actual layout by scanning. Once we have access to a machine with OpenVINO/VitisAI/etc. MSIX installed, populate a known-paths table in this doc.

2. **`winrt-windows-management-deployment` package availability.** The existing `[winml-catalog]` extra ships `wasdk-microsoft-windows-ai-machinelearning` which transitively pulls in some `winrt-*` packages — need to verify whether `winrt-windows-management-deployment` is already among them, or if we need to add it explicitly. Check via `uv run python -c "from winrt.windows.management.deployment import PackageManager; print('ok')"` after a fresh install.

3. **`PackageManager.find_packages_for_user("")` exact semantics.** Empty string passes "current user." Need to confirm this works for Python under regular (non-elevated) user context — should not require elevation, but worth a live probe before relying on it. The catalog already works without elevation, which is positive evidence.

4. **`Package.Id.Version` ordering.** `tuple(version)` sorting works if `Version` exposes `(major, minor, build, revision)` — confirm via the WinRT binding. Empirical observation: `<class 'PackageVersion(major=2, minor=2420, build=44, revision=0)'>`. Tuple ordering matches semver-ish lex order for these fields, which is what we want.

5. **Multi-arch packages.** Snapdragon X Elite has only `arm64` packages. On x64 machines with both x64 and arm64 EP MSIXes (unusual but possible — Windows on ARM with x64 emulation), `find_packages_for_user("")` may return both architectures. Architecture filter not yet in the design — for now we let the highest-version-wins rule pick deterministically. Revisit if a real cross-arch scenario surfaces.

6. **Catalog-default detection for CLI display.** The `(catalog default)` annotation in `winml sys --list-ep` requires querying `WinMLCatalogSource` once and matching its `library_path` against each `MsixPackageSource`'s computed DLL path. This means the CLI command pulls in the WinAppSDK ML binding — fine, since both features live behind the `[winml-catalog]` extra. If the binding is not installed, the annotation is suppressed; still print the list.

## Sources

- [`docs/ep-path-design.md`](ep-path-design.md) — original `EP_PATH` design, defines `EpSource` and `EP_DLL_NAMES`.
- [`docs/winml-ep-empirical-findings.md`](winml-ep-empirical-findings.md) — live-machine evidence for catalog and MSIX behavior.
- [Microsoft Learn — Initialize execution providers with Windows ML](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/initialize-execution-providers).
- [Microsoft Learn — Register execution providers with Windows ML](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/register-execution-providers).
- [`microsoft/WindowsAppSDK-Samples` — `Samples/WindowsML/cpp-abi/ExecutionProviderCatalog.cpp`](https://github.com/microsoft/WindowsAppSDK-Samples/blob/main/Samples/WindowsML/cpp-abi/ExecutionProviderCatalog.cpp) — C++ ABI sample wrapping `IExecutionProviderCatalog`.
- [`m417z/WindowsAppSDK-Index` — Microsoft.Windows.AI.MachineLearning IDL 2.0.300](https://github.com/m417z/WindowsAppSDK-Index/blob/main/Microsoft.WindowsAppSDK/deps/Microsoft.Windows.AI.MachineLearning/2.0.300/metadata/Microsoft.Windows.AI.MachineLearning.winmd_winmdidl/Microsoft.Windows.AI.MachineLearning.idl) — third-party mirror of the WinAppSDK IDL.
- [Microsoft Learn — Windows.Management.Deployment.PackageManager](https://learn.microsoft.com/en-us/uwp/api/windows.management.deployment.packagemanager) — WinRT API used by `_get_pkg_manager()`.
