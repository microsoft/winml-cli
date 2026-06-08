# Session / EP Module — Canonical Class Reference

**Version**: 1.1
**Date**: 2026-06-07
**Status**: Draft — v1.1 renames `WinMLEPRegistry.auto_ep` → `auto_device` (it returns a `WinMLEPDevice` pair, not a `WinMLEP`); collapses `WinMLDevice` ABC + six subclasses into a single concrete class with internal dispatch tables; pins the `WinMLEPDevice` composition invariant (`.device` is one of `.ep.devices`, same object); drops the `_find_entry` helper (absorbed into `auto_device`).
**Module**: session
**Companion-To**:
- [`1_req.md`](1_req.md) — user-facing requirements
- [`2_coreloop.md`](2_coreloop.md) — type taxonomy, core APIs, the two paths
- [`3_design_ep.md`](3_design_ep.md) — Tier 1/2/3 model and registration internals
- [`4_winml_device.md`](4_winml_device.md) — `WinMLDevice` single class + dispatch tables
- [`5_type_taxonomy.md`](5_type_taxonomy.md) — superseded stub (pointers only)
- [`console_mockup.py`](console_mockup.py) — `winml sys --list-ep` render mockup

---

## Table of Contents

- [1. Purpose](#1-purpose)
- [2. Naming Principles](#2-naming-principles)
- [3. The Six Core Data Classes](#3-the-six-core-data-classes)
  - [3.1 `EPDeviceTarget`](#31-epdevicetarget)
  - [3.2 `EPDeviceSpec`](#32-winmlepdevicespec)
  - [3.3 `EPEntry`](#33-epentry)
  - [3.4 `WinMLDevice`](#34-winmldevice)
  - [3.5 `WinMLEP`](#35-winmlep)
  - [3.6 `WinMLEPDevice` (the flat pair)](#36-winmlepdevice-the-flat-pair)
- [4. Supporting Hierarchy — `EPSource` and Subclasses](#4-supporting-hierarchy--epsource-and-subclasses)
- [5. Catalog — `EPCatalog`](#5-catalog--epcatalog)
- [6. The Registry — `WinMLEPRegistry`](#6-the-registry--winmlepregistry)
- [7. The Session — `WinMLSession`](#7-the-session--winmlsession)
- [8. Exception Taxonomy](#8-exception-taxonomy)
- [9. Class Relationship Diagram](#9-class-relationship-diagram)
- [10. Quick-Reference Cheat Sheet](#10-quick-reference-cheat-sheet)

---

## 1. Purpose

Single-source class reference for the session/EP module. Every class with a one-paragraph explanation, lifecycle, usage example, and pointer to the more detailed design doc. New readers landing on the session/EP module should read this doc first — it fixes the class vocabulary that the rest of the design docs assume.

The session/EP module's job is to turn "the user wants an EP and a device" into "an `ort.InferenceSession` bound to the right `OrtEpDevice` handle." That work decomposes into six core data classes plus a small supporting cast (catalog, registry, session wrapper, exception trio). The six data classes are mutually disjoint by role — there is exactly one class per concern — and the naming distinguishes user-constructible from system-generated at the prefix.

For the path-level flows that compose these classes, see [`2_coreloop.md`](2_coreloop.md) §4 (Path A) and §5 (Path B). This doc is class-shaped; that doc is flow-shaped.

## 2. Naming Principles

Two rules apply uniformly across the module:

**Rule 1 — `WinML*` prefix means predefined or system-generated.** A `WinML*` class cannot be crafted from CLI strings or JSON; constructing one requires a system API — the static `EP_DEVICE_SPECS` catalog (for `EPDeviceSpec`), an ORT registration (for `WinMLEP`, `WinMLEPDevice`), or the device factory (`wrap_ort_device` for `WinMLDevice`). The prefix is a "you can't make me, the system did" marker.

Examples: `WinMLDevice`, `WinMLEP`, `WinMLEPDevice` (the new flat pair), `EPDeviceSpec`, `WinMLEPRegistry`, `WinMLSession`.

**Rule 2 — No prefix means user-craftable.** A non-prefixed class is constructible from strings or paths so tests, configs, and the CLI parser can build them directly. The user (or test code) is the canonical author; system code may also produce them but does not own the creation contract.

Examples: `EPDeviceTarget` (CLI parser builds it from `--ep`/`--device`), `EPEntry` (tests construct it for registry tests; production builds it via `discover_all_eps()`).

**Rule 3 — `EP` is the canonical acronym, not `Ep`.** Per [`docs/naming-convention.md`](../../naming-convention.md) §1, `EP` (Execution Provider) is an all-caps acronym in code symbols. `EPDeviceTarget` is correct; `EpDeviceTarget` is broken. The current `src/` codebase has stale `Ep` casings (`EpCatalog`, `EpSource`, `ResolvedEp`) queued for a one-shot rename PR — see [`2_coreloop.md`](2_coreloop.md) §8.

**Rule 4 — `WinML*` DOES imply OS-bound.** A `WinML*` class can only be constructed via the Windows ML runtime — either by ORT registration (`WinMLDevice`, `WinMLEP`, `WinMLEPDevice` flat pair, `WinMLEPRegistry`, `WinMLSession`) or by Windows-specific discovery (`WinMLCatalogSource` via the WinAppSDK `ExecutionProviderCatalog` API). It cannot be hand-built from primitive strings/paths. Classes without the prefix (`EPDeviceTarget`, `EPDeviceSpec`, `EPEntry`) are pure data — JSON-portable, OS-agnostic, hand-constructible in tests, persist in `compiler/configs.py` cleanly.

## 3. The Six Core Data Classes

The six classes that carry the session/EP module's state. Each has exactly one role; the table below summarizes, and the subsections elaborate.

| Class | Role | Created by | Prefix rule |
|---|---|---|---|
| [`EPDeviceTarget`](#31-epdevicetarget) | Pure intent (`ep`, `device`, optional `source`). | CLI parser, JSON config loader, tests. | No prefix — user-craftable. |
| [`EPDeviceSpec`](#32-winmlepdevicespec) | Catalog row — what *could* exist for an EP. | Static `EP_DEVICE_SPECS` table. | `WinML*` — predefined. |
| [`EPEntry`](#33-epentry) | Filesystem-discovery record. | `discover_all_eps()`; tests construct directly. | No prefix — user-craftable (for tests). |
| [`WinMLDevice`](#34-winmldevice) | Vendor-normalized adapter over `ort.OrtEpDevice` — single concrete class; dispatches per-EP metadata internally. | `wrap_ort_device(handle)`. | `WinML*` — system-generated. |
| [`WinMLEP`](#35-winmlep) | Successful per-source registration aggregate. | `WinMLEPRegistry.register_ep(entry)`. | `WinML*` — system-generated. |
| [`WinMLEPDevice`](#36-winmlepdevice-the-flat-pair) | Flat `(ep, device)` pair — the mirror of `ort.OrtEpDevice`. Invariant: `.device in .ep.devices`. | `WinMLEPRegistry.auto_device(target)` (Path A) and `WinMLEP.ep_devices()` (enumerator). | `WinML*` — system-generated. |

### 3.1 `EPDeviceTarget`

**Role.** Pure intent. The user's pick, before resolution. Either axis (`ep`, `device`) may carry the literal string `"auto"`; the optional `source` field carries the Scenario B disambiguator (e.g., `"pypi"` from `--ep openvino@pypi`).

**Who creates it.** The CLI parser (`commands/perf.py`, `commands/compile.py`, `commands/sys.py`) constructs `EPDeviceTarget` from `--ep` and `--device` args. The JSON config loader (`compiler/configs.py:288 from_dict`) rehydrates it from persisted compile configs. Tests construct it directly.

**Lifecycle.** Built at CLI parse time (or config-load time); consumed by `resolve(target) -> EPDeviceTarget` (§3 of `2_coreloop.md`); after that the resolved `EPDeviceTarget` flows into `registry.auto_device(target)` and is dropped once the `WinMLEPDevice` pair is in hand. Lives microseconds at most.

**Typical usage.**

```python
# CLI parse — Scenario A (no @)
target = EPDeviceTarget(ep="openvino", device="auto", source=None)

# CLI parse — Scenario B (with @-tag)
target = EPDeviceTarget(ep="openvino", device="npu", source="pypi")

# JSON config rehydrate — old config (no source field)
target = EPDeviceTarget.from_dict({"ep": "qnn", "device": "npu"})  # source defaults to None

# Test fixture
target = EPDeviceTarget(ep="cpu", device="cpu", source=None)
```

**More detail.** Path-level walkthroughs at [`2_coreloop.md`](2_coreloop.md) §4.1 (Scenarios A.1-A.4) and §4.2 (Scenarios A.5-A.6); persisted-config round-trip at [`2_coreloop.md`](2_coreloop.md) §4.5.

### 3.2 `EPDeviceSpec`

**Role.** Catalog row — what `(ep, device)` combinations *could* exist for each EP, independent of installation state. Carries `default_provider_options` (the tuning-key defaults like `{"htp_performance_mode": "burst"}` for QNN+NPU). Process-constant: never mutated, never instantiated by callers.

**Who creates it.** The static `EP_DEVICE_SPECS` table in [`session/ep_device.py`](../../../src/winml/modelkit/session/ep_device.py). Each table entry is a module-level constant constructed at import time.

**Lifecycle.** Module-level constants; live for the process lifetime. Callers never instantiate.

**Typical usage.**

```python
# Lookup (internal):
spec = lookup_device_spec("openvino", "npu")
provider_options = dict(spec.default_provider_options)   # mutable copy for merge

# Default-device deduction:
device = default_device_for_ep("openvino")               # consults the spec set
```

**More detail.** Catalog scope and the catalog-vs-registration distinction at [`3_design_ep.md`](3_design_ep.md) §6.4 and §9.

### 3.3 `EPEntry`

**Role.** Filesystem-discovery record. One per `(ep_name, on-disk-source)` pair the discovery walk turns up. No DLL has been loaded; the entry only records "I found `openvino_plugin.dll` at this path, from this source-kind, at this version." Carries `ep_name`, `dll_path`, `source` (the source-kind tag like `"pypi"`), `status` (`"primary"` or `"shadowed"`, derived from precedence position), and `version`.

**Who creates it.** `discover_all_eps()` (the discovery walker; built on top of each `EPSource.resolve()`). Tests may construct `EPEntry` directly when bypassing the discovery layer.

**Lifecycle.** Built once per discovery walk; lives until the registry consumes it (`register_ep(entry) -> WinMLEP`) or the renderer turns it into an incompatible-row DTO. Independent across CLI invocations — `discover_all_eps()` re-runs on every call.

**Typical usage.**

```python
# Production:
for entry in discover_all_eps():
    print(entry.ep_name, entry.source, entry.dll_path)

# Test:
fake_entry = EPEntry(
    ep_name="OpenVINOExecutionProvider",
    dll_path=Path("C:/fake/openvino_plugin.dll"),
    source="pypi",
    status="primary",
    version="1.4.1",
)
ep = WinMLEPRegistry.instance().register_ep(fake_entry)   # in test, with ORT patched
```

**Current state.** Today this record exists as `ResolvedEp` in `ep_path.py:1189`; the rename to `EPEntry` is queued — see [`2_coreloop.md`](2_coreloop.md) §8 and §9.3.

**More detail.** Discovery layer in [`../../ep-path-design.md`](../../ep-path-design.md); registration consumption in [`2_coreloop.md`](2_coreloop.md) §3.1.

### 3.4 `WinMLDevice`

**Role.** Vendor-normalized adapter over `ort.OrtEpDevice`. A **single concrete class** whose vendor-specific properties (`memory_bytes`, `architecture`, `capabilities`, `driver_version`, `compiler_version`) dispatch internally on `self._ort.ep_name` via module-level dispatch tables. Common properties (`ep_name`, `device_type`, `hardware_name`, `vendor`, `ep_vendor`, `library_path`) need no dispatch — they read straight from `self._ort`.

Why a single class (and not an ABC + per-EP subclasses): zero current `ep_metadata` consumers exist in `src/`; the one near-future consumer is the `--list-ep` renderer; one speculative is `EPDoctor`. The six-subclass design carried weight (test surface, import structure, factory dispatch) for a per-EP-method-override contract that nothing in the codebase actually consumes. A single concrete class with module-level dispatch tables preserves the empirical schemas where they were going to live anyway and removes the polymorphism overhead.

**Who creates it.** `wrap_ort_device(handle: ort.OrtEpDevice) -> WinMLDevice` — the module-level factory; a trivial wrapper over the `WinMLDevice(handle)` constructor under the single-class design. Not constructed by callers directly; tests may instantiate via `WinMLDevice(handle)` when the `ort.OrtEpDevice` handle is mocked.

**Lifecycle.** One `WinMLDevice` per `OrtEpDevice` handle. Constructed when the registry calls `wrap_ort_device(...)` inside `register_ep`; lives as long as the containing `WinMLEP` (which is cached in the registry by `entry.dll_path`).

**Typical usage.** The renderer (`commands/sys.py`'s `--list-ep`) is the primary consumer:

```python
for ort_dev in handles_from_register_ep:
    dev = wrap_ort_device(ort_dev)
    facts = dev.facts()                  # tuple[str, ...] for pipe-join display
    metadata_dump = dev.available_metadata()    # raw ep_metadata for --verbose
```

**More detail.** Single-class spec, internal dispatch tables (per-EP schemas), and the trivial `wrap_ort_device` factory in [`4_winml_device.md`](4_winml_device.md).

### 3.5 `WinMLEP`

**Role.** Successful per-source registration aggregate. One `WinMLEP` instance represents "one EP DLL that loaded plus the tuple of `WinMLDevice` handles it contributed." The class is **success-only by design**: the invariant `len(devices) >= 1` means a `WinMLEP` cannot represent a failed registration. Failures live separately as `(EPEntry, Exception)` pairs in the broad-loop output.

**Who creates it.** `WinMLEPRegistry.register_ep(entry: EPEntry) -> WinMLEP`. Tests may patch the registry to return synthetic instances; production callers never call `WinMLEP(...)` directly.

**Lifecycle.** Cached in `WinMLEPRegistry._registered: dict[Path, WinMLEP]` keyed by `entry.dll_path`. Lives for the process lifetime once registered (the registry never unregisters in v1). Idempotent — re-registering the same `entry.dll_path` returns the same `WinMLEP` object identity-wise.

**Flattening helper — `WinMLEP.ep_devices()`.** Convenience accessor that turns the `tuple[WinMLDevice, ...]` field into a tuple of `WinMLEPDevice(ep, device)` pairs:

```python
ep = WinMLEPRegistry.instance().register_ep(entry)
# ep.devices         -> tuple[WinMLDevice, ...]
# ep.ep_devices()    -> tuple[WinMLEPDevice, ...]   one pair per device row

# Pick the (source, device) pair for a specific class:
npu_pair = next(epd for epd in ep.ep_devices() if epd.device.device_type == "NPU")
```

**Typical usage.**

```python
# Path A — compound resolve + retry-shadowed (returns the pair directly):
pair = WinMLEPRegistry.instance().auto_device(resolved_target)
session = WinMLSession(onnx_path, pair, ep_config=..., ep_monitor=...)

# Direct registration (one entry at a time — used by Path B inline loop
# and by tests that bypass the auto_device retry):
ep = WinMLEPRegistry.instance().register_ep(entry)

# Path B — broad-loop:
results: list[WinMLEP] = []
failures: list[tuple[EPEntry, Exception]] = []
for entry in discover_all_eps():
    try:
        results.append(WinMLEPRegistry.instance().register_ep(entry))
    except Exception as e:
        failures.append((entry, e))
```

**More detail.** Class shape and the success-only invariant in [`2_coreloop.md`](2_coreloop.md) §2; registration mechanics in [`2_coreloop.md`](2_coreloop.md) §3.1.

### 3.6 `WinMLEPDevice` (the flat pair)

**Role.** Flat `(ep: WinMLEP, device: WinMLDevice)` pair — the project's typed mirror of `ort.OrtEpDevice`. Whereas a `WinMLEP` carries a tuple of devices that one DLL produced, `WinMLEPDevice` is the one-`(source, device)`-pair shape that the `WinMLSession(...)` constructor consumes. It is the user-facing input to session construction.

**Shape and invariant.**

```python
@dataclass(frozen=True)
class WinMLEPDevice:
    """Project mirror of ort.OrtEpDevice — the (source, device) pair Path A targets.

    Invariant: .device is always one of .ep.devices (same object, not a copy).
    Constructed only by WinMLEPRegistry.auto_device() and by tests; never by
    direct user code."""
    ep:     WinMLEP        # which source registered (carries source attribution + sibling devices)
    device: WinMLDevice    # the specific device picked (one of ep.devices)
```

Stated explicitly: `assert pair.device in pair.ep.devices` — same object, not duplicate data. The two members earn their keep because they answer different questions: "which DLL produced this?" is `.ep.source`; "which raw `OrtEpDevice` handle?" is `.device._ort`. Both are reachable from one pair without re-querying the registry.

**Naming note — same name, new meaning.** In the prior design, `WinMLEPDevice` meant "pure intent `(ep, device)` strings." That role is now `EPDeviceTarget`. The name `WinMLEPDevice` is **reassigned** to the new flat pair (see [`1_req.md`](1_req.md) §3 C3 — no back-compat shims). Old call sites are rewritten in the same PR that lands the new types.

**Who creates it.** `WinMLEPRegistry.auto_device(target)` returns one `WinMLEPDevice` directly (the matched pair, with retry-shadowed fallback handled inside). The lower-level `WinMLEP.ep_devices()` accessor still returns `tuple[WinMLEPDevice, ...]` (one pair per device in `ep.devices`) for callers that need every device row a successful registration produced — the Path B render path is one such caller. Tests may construct directly when patching the registry; production user code never calls `WinMLEPDevice(...)` directly.

**Lifecycle.** Built on demand by `WinMLEPRegistry.auto_device()` (Path A) or `WinMLEP.ep_devices()` (lower-level enumerator). The session holds a reference for its lifetime (the pair is the session's binding target). Otherwise transient.

**Typical usage.**

```python
# Standard Path A — auto_device returns the pair directly:
ep_device: WinMLEPDevice = WinMLEPRegistry.instance().auto_device(resolved_target)

# Direct field access:
ep_device.ep                # WinMLEP — the registration aggregate (.ep.source = EPEntry)
ep_device.device            # WinMLDevice — the per-device adapter; in ep.devices
ep_device.device._ort       # ort.OrtEpDevice — the raw handle for add_provider_for_devices

# Session construction:
session = WinMLSession("model.onnx", ep_device, ep_config=..., ep_monitor=...)
```

**More detail.** Naming-concern discussion (the three-way confusion between `EPDeviceTarget`, `WinMLDevice`, and `WinMLEPDevice`-the-pair) in [`4_winml_device.md`](4_winml_device.md) §9.

## 4. Supporting Hierarchy — `EPSource` and Subclasses

`EPSource` (the discovery ABC) and its five subclasses are the source-kind layer of the discovery walk. Each subclass knows how to look up `(ep_name, dll_path)` pairs for one origin (PyPI, NuGet, MSIX, etc.). They live in [`ep_path.py`](../../../src/winml/modelkit/ep_path.py) — outside the session module but consumed by it.

**`EPSource` (ABC).** Defines the `resolve(self) -> Iterator[EPEntry]` contract. Errors from a single source are logged and swallowed — discovery does not abort because one subclass raised. The current `src/` signature is `Iterator[tuple[str, Path]]`; the rename to `Iterator[EPEntry]` is queued in the casing-sweep PR.

**`PyPISource`.** Walks the active Python environment's `site-packages` for plugin EP wheels (`onnxruntime-ep-openvino`, `onnxruntime-qnn`, etc.). Default first row in the `EP_PATH` precedence list.

**`NuGetSource`.** Walks NuGet caches for plugin EP packages. Default second row.

**`MSIXPackageSource`.** Enumerates installed MSIX packages via WinRT `PackageManager`, filtered by `family_name_prefix`. Produces both `msix-microsoft` (for `MicrosoftCorporationII.WinML.*` families) and `msix-workload` (for `WindowsWorkload.EP.*` families) tags depending on which prefix matches. The two tags may split into two source classes in a future PR — see [`2_coreloop.md`](2_coreloop.md) §8.

**`WinMLCatalogSource`.** Reads MSIX EPs delivered via the WinML EP Catalog API (`Microsoft.Windows.AI.MachineLearning.ExecutionProviderCatalog` + `EnsureReadyAsync`/`FindAllProviders`). Available on Windows 11 24H2+ with WinAppSDK 1.8.1+.

**`DirectorySource`.** Filesystem directory drops — vendor installers, dev builds, `WINMLCLI_EP_PATH` glob hits. Constructed dynamically by `_parse_winmlcli_ep_path()` at discovery time.

The five subclasses together produce one `EPEntry` per discovered DLL. `discover_all_eps()` flattens their iterators into a single `list[EPEntry]` in `EP_PATH` precedence order.

## 5. Catalog — `EPCatalog`

**Role.** Single source of truth for EP metadata: `name`, `dll_name`, `vendor_requirements`. Used by `MSIXPackageSource.list_installed` to map a DLL filename back to its EP name; by `discover_eps` for vendor compatibility; by `_parse_winmlcli_ep_path` for the DLL pattern table. All methods are classmethods; the class is used as a namespace, never instantiated by callers.

**Who creates it.** Module-level; the `EPCatalog._ENTRIES` table is populated at import time. Instantiation is rejected by convention.

**Nested type — `EPCatalog.Row`.** The per-EP metadata record (`name`, `dll_name`, `vendor_requirements`). Today this is a top-level `EpEntry` in `ep_path.py:64`; the nesting into `EPCatalog.Row` is queued in the casing-sweep PR. The rename frees the top-level `EPEntry` name for the §3.3 discovery record. External callers reference `EPCatalog.Row` only for type annotations — which is rare since lookups return scalar fields.

**Current state.** Still spelled `EpCatalog` in `ep_path.py:76`; rename queued — see [`2_coreloop.md`](2_coreloop.md) §9.2.

## 6. The Registry — `WinMLEPRegistry`

**Role.** Process-wide singleton that registers EP DLLs with ORT. Public surface is **two methods** — one atomic (`register_ep`) and one compound (`auto_device`):

```python
class WinMLEPRegistry:
    def register_ep(self, entry: EPEntry) -> WinMLEP: ...
    def auto_device(self, target: EPDeviceTarget) -> WinMLEPDevice: ...
```

`register_ep` is atomic per-source: one DLL load, one `WinMLEP` out, or raise. No discovery, no broad listing, no candidate fallback, no device-class filter — those concerns live elsewhere (discovery in `discover_all_eps()`, broad listing in the inline caller-side loop, fallback handled by `auto_device` or by the caller's loop).

`auto_device` is the compound Path A entry point. The "auto" prefix conveys "resolve + retry-shadowed"; "device" conveys the return shape (a `WinMLEPDevice` pair, not a `WinMLEP`). It rejects `target.ep == "auto"` or `target.device == "auto"` (the target must be resolved first), filters `discover_all_eps()` by `target.ep` and optional `target.source`, tries each candidate in precedence order, and returns the first `WinMLEPDevice` whose device class matches. Raises `WinMLEPRegistrationFailed` (with the last error chained) if all candidates fail.

**State.** `_registered: dict[Path, WinMLEP]` — idempotency cache keyed by `entry.dll_path`. Re-registering the same DLL is a no-op that returns the same `WinMLEP` object identity-wise. No discovery cache, no failure cache.

**Singleton access.**

```python
registry = WinMLEPRegistry.instance()        # classmethod; constructs on first call
pair = registry.auto_device(resolved_target) # compound; Path A entry
ep   = registry.register_ep(entry)           # atomic; Path B / tests
```

Tests that need a fresh instance reset `WinMLEPRegistry._instance = None` and re-call `instance()`.

**Idempotency contract.** Re-registering the same `entry.dll_path` returns the *same* `WinMLEP` instance. Tests, repeated CLI calls within one process, the Path B inline loop, and the inner retry inside `auto_device` all rely on this.

**Lockdown reference.** Full per-step spec for `register_ep` (DLL load step, handle re-read, `wrap_ort_device` wrap, success-only invariant) and for `auto_device` (filter, retry-shadowed loop, device-class match) lives in [`2_coreloop.md`](2_coreloop.md) §3.1. The "what migrates out of the current `register_ep` body" subsection (candidate fallback to `auto_device`, bundled-EP branch to synthetic `EPEntry`, device-class filter to `auto_device` / caller) is at [`2_coreloop.md`](2_coreloop.md) §3.1.

## 7. The Session — `WinMLSession`

**Role.** ONNX Runtime session wrapper bound to one `(EP, device)` target. The Path A user-facing tail.

**Constructor — direct, no `build()` classmethod.**

```python
class WinMLSession:
    def __init__(
        self,
        onnx_path: str | Path,
        ep_device: WinMLEPDevice,
        *,
        ep_config: EPConfig | None = None,
        ep_monitor: WinMLEPMonitor | None = None,
        base_session_options: ort.SessionOptions | None = None,
    ) -> None: ...
```

The constructor consumes a pre-resolved `WinMLEPDevice` (the §3.6 flat pair), runs the three-layer `provider_options` merge (catalog default → user config → monitor overrides), and constructs `ort.InferenceSession` eagerly. `ep_monitor` is optional — `None` is the default and existing call sites (`commands/perf.py`, `commands/compile.py`, `models/auto.py`) work unchanged; monitor-aware sites pass `ep_monitor=monitor`.

The current `__init__` at [`session/session.py:200`](../../../src/winml/modelkit/session/session.py) already has this shape; the v2.2 lock-in adds the new optional `ep_monitor` kwarg. The QAIRT subclass at [`qairt_session.py:44`](../../../src/winml/modelkit/session/qairt/qairt_session.py) extends `__init__` and continues to work — the new kwarg propagates via `**kwargs` in subclasses that don't name it explicitly.

**Subclasses.** `WinMLQairtSession` (QAIRT SDK pipeline; overrides `compile()` to run the QAIRT subprocess instead of `ort.ModelCompiler`).

**Typical usage.**

```python
# Standard SDK pattern:
session = WinMLSession("model.onnx", ep_device,
                       ep_config=EPConfig(provider_options={"htp_performance_mode": "burst"}))

# With monitor:
monitor = QNNMonitor(level="basic", output_dir=Path("./trace"))
session = WinMLSession("model.onnx", ep_device, ep_monitor=monitor)

# Subclass for QAIRT:
session = WinMLQairtSession("model.onnx")    # defaults ep_device to resolve('qnn', 'npu')
session.compile()
```

**More detail.** Path A walkthrough and the constructor flow at [`2_coreloop.md`](2_coreloop.md) §4.1 Step 5; `_build_session_options` post-refactor body at [`2_coreloop.md`](2_coreloop.md) §3.3.

## 8. Exception Taxonomy

Every exception raised by the public APIs in the session/EP module.

| Exception | Where raised | Meaning |
|---|---|---|
| `WinMLEPNotDiscovered` | `resolve()` (Scenario A) | `--ep <name>` doesn't match any discovered `EPEntry`. |
| `WinMLEPRegistrationFailed` | `register_ep()` | ORT's `register_execution_provider_library` raised, or the loaded DLL contributed zero devices. |
| `DeviceNotFound` | Caller-side helper (post-refactor) | A registered EP yielded no row for the requested device class. Demoted from `register_ep`'s contract; surfaces inside `auto_device`'s device-class match or at caller-side enumeration. |
| `AmbiguousMatch` | `register_ep()` (logged, not raised) | Multiple `(vendor_id, device_id)`-distinct handles survived dedup. Registry-bug signal. |
| `WinMLEPMonitorMismatch` | `WinMLSession.perf()` | Monitor `ep_name` does not agree with `WinMLEPDevice.ep`. |
| `UnknownListingPick` | `resolve()` (Scenario B), `auto_device()` | `--ep <name>@<tag>` doesn't match any discovered `EPEntry`. Carries `ep` and `source_tag` in `args`. |
| `IncompatibleListingPick` | `resolve()` (Scenario B) | Matched entry's broad-loop registration raised; original `WinMLEPRegistrationFailed` chained as `__cause__`. Path A refuses to silently fall back. |
| `AmbiguousListingPick` | `resolve()` (Scenario B), `auto_device()` (defensive) | `device="auto"` with an `@<tag>` whose entry contributes multiple device classes without a clean catalog-default fallback. Also raised defensively by `auto_device` if more than one entry matches a tag (tag-algorithm bug signal). |
| `WinMLSessionError`, `CompilationError`, `DeviceNotAvailableError`, `InferenceError`, `NotCompiledError` | `WinMLSession` lifecycle | Session-level exceptions (compile/run state machine). See `session/session.py:129-166`. |
| `ValueError` | `resolve()` | Unknown EP short-name; unknown device class; `--device` only with no installed EP claiming that class. Generic Python error — no custom class. |

The Scenario B trio (`UnknownListingPick`, `IncompatibleListingPick`, `AmbiguousListingPick`) is new in v2.2 of [`2_coreloop.md`](2_coreloop.md). See §4.2 for the raised-by spec and §9.4 for the inventory entry.

## 9. Class Relationship Diagram

```mermaid
classDiagram
    class EPDeviceTarget {
        +ep: str
        +device: str
        +source: str | None
        +to_dict()
        +from_dict()
    }

    class EPDeviceSpec {
        +ep: str
        +device: str
        +default_provider_options: Mapping
    }

    class EPEntry {
        +ep_name: str
        +dll_path: Path
        +source: str
        +status: str
        +version: str
    }

    class WinMLDevice {
        +ep_name: str
        +device_type: str
        +hardware_name: str
        +memory_bytes
        +architecture
        +capabilities
        +driver_version
        +compiler_version
        +facts()
        +available_metadata()
    }

    class WinMLEP {
        +source: EPEntry
        +devices: tuple~WinMLDevice~
        +ep_devices() tuple~WinMLEPDevice~
    }

    class WinMLEPDevice {
        +ep: WinMLEP
        +device: WinMLDevice
    }

    class WinMLEPRegistry {
        -_registered: dict~Path, WinMLEP~
        +instance()$
        +register_ep(entry) WinMLEP
        +auto_device(target) WinMLEPDevice
    }

    class WinMLSession {
        +__init__(onnx_path, ep_device, *, ep_config, ep_monitor, base_session_options)
        +compile()
        +perf()
    }

    class WinMLQairtSession

    class EPSource {
        <<abstract>>
        +resolve() Iterator~EPEntry~
    }

    class PyPISource
    class NuGetSource
    class MSIXPackageSource
    class WinMLCatalogSource
    class DirectorySource

    class EPCatalog {
        <<namespace>>
        +lookup(name)$
        +is_compatible(name)$
    }

    EPSource <|-- PyPISource
    EPSource <|-- NuGetSource
    EPSource <|-- MSIXPackageSource
    EPSource <|-- WinMLCatalogSource
    EPSource <|-- DirectorySource

    WinMLSession <|-- WinMLQairtSession

    EPDeviceTarget ..> WinMLEPDevice : resolves via registry.auto_device
    EPSource ..> EPEntry : yields
    EPCatalog ..> EPEntry : (filename-to-ep lookup during discovery)

    WinMLEPRegistry --> WinMLEP : produces (cached by dll_path)
    WinMLEPRegistry --> WinMLEPDevice : produces via auto_device
    WinMLEP *-- EPEntry : .source
    WinMLEP *-- "1..*" WinMLDevice : .devices
    WinMLEPDevice o-- WinMLEP : .ep
    WinMLEPDevice o-- WinMLDevice : .device (one of .ep.devices)
    WinMLEP ..> WinMLEPDevice : .ep_devices()

    WinMLSession --> WinMLEPDevice : binds to (ctor arg)
    WinMLSession ..> WinMLEPRegistry : (does NOT call directly; caller pre-resolves)
```

**Reading the diagram:**

- Solid arrows with diamonds (`*--`, `o--`) are composition / aggregation relationships (`WinMLEP` *owns* its `EPEntry` and `WinMLDevice` tuple; `WinMLEPDevice` *references* its `WinMLEP` and `WinMLDevice`).
- The `WinMLEPDevice o-- WinMLDevice` aggregation is governed by the invariant `pair.device in pair.ep.devices` (same object, not a copy) — see §3.6.
- Dashed arrows (`..>`) are dependency relationships (`EPDeviceTarget` resolves into a `WinMLEPDevice` via `registry.auto_device`; `EPSource.resolve()` yields `EPEntry`).
- Inheritance arrows (`<|--`) are subclass relationships (`PyPISource` extends `EPSource`, `WinMLQairtSession` extends `WinMLSession`). There is **no** `WinMLDevice` subclass hierarchy under v1.1; the prior `OpenVINODevice` / `QNNDevice` / `DmlDevice` / `CpuDevice` / `AzureDevice` / `UnknownDevice` subclasses are collapsed into the single concrete class with internal dispatch — see [`4_winml_device.md`](4_winml_device.md) §4-§5.
- `WinMLSession` deliberately does **not** call `WinMLEPRegistry.register_ep` from inside `__init__` post-refactor — the caller has already invoked `auto_device` (Path A) or done the inline loop (Path B) and passes in the resolved `WinMLEPDevice` pair. See [`2_coreloop.md`](2_coreloop.md) §3.3 for the `_build_session_options` body change that removes the in-`__init__` `register_ep` call.

## 10. Quick-Reference Cheat Sheet

| Class | One-line role | Where to read more |
|---|---|---|
| `EPDeviceTarget` | Pure intent — CLI/JSON-craftable. | [`§3.1`](#31-epdevicetarget), [`2_coreloop.md`](2_coreloop.md) §4.1 |
| `EPDeviceSpec` | Catalog row — static `(ep, device, defaults)`. | [`§3.2`](#32-winmlepdevicespec), [`3_design_ep.md`](3_design_ep.md) §6 |
| `EPEntry` | Filesystem-discovery record — one per `(ep_name, dll_path)`. | [`§3.3`](#33-epentry), [`../../ep-path-design.md`](../../ep-path-design.md) |
| `WinMLDevice` | Vendor-normalized adapter over `ort.OrtEpDevice` — single concrete class with per-EP dispatch. | [`§3.4`](#34-winmldevice), [`4_winml_device.md`](4_winml_device.md) |
| `WinMLEP` | Success-only per-source registration aggregate. | [`§3.5`](#35-winmlep), [`2_coreloop.md`](2_coreloop.md) §3.1 |
| `WinMLEPDevice` (flat pair) | `(ep, device)` pair — `ort.OrtEpDevice` mirror. | [`§3.6`](#36-winmlepdevice-the-flat-pair), [`2_coreloop.md`](2_coreloop.md) §2 |
| `EPSource` (ABC) + 5 subclasses | Discovery sources — one per origin (PyPI/NuGet/MSIX/Catalog/Directory). | [`§4`](#4-supporting-hierarchy--epsource-and-subclasses), [`../../ep-path-design.md`](../../ep-path-design.md) |
| `EPCatalog` | Static EP-metadata namespace (`name`, `dll_name`, vendor requirements). | [`§5`](#5-catalog--epcatalog), [`2_coreloop.md`](2_coreloop.md) §9.2 |
| `WinMLEPRegistry` | Process-singleton registrar, one public method. | [`§6`](#6-the-registry--winmlepregistry), [`2_coreloop.md`](2_coreloop.md) §3.1 |
| `WinMLSession` | Session wrapper; direct constructor (no `build()`). | [`§7`](#7-the-session--winmlsession), [`2_coreloop.md`](2_coreloop.md) §4.1 Step 5 |
| `WinMLQairtSession` | QAIRT SDK session subclass (CSV trace). | [`§7`](#7-the-session--winmlsession), [`session/qairt/qairt_session.py`](../../../src/winml/modelkit/session/qairt/qairt_session.py) |
| Scenario B exception trio | `UnknownListingPick` / `IncompatibleListingPick` / `AmbiguousListingPick`. | [`§8`](#8-exception-taxonomy), [`2_coreloop.md`](2_coreloop.md) §4.2 |
