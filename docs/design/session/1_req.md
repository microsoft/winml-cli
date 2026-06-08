# Session / EP Module — User-Facing Requirements

**Version**: 1.2
**Date**: 2026-06-07
**Status**: Draft — v1.2 adds the explicit JSON-reload behavior cross-reference to C1 and the `3_design_classes.md` pointer. The canonical seven source tags (`bundled`, `pypi`, `nuget`, `msix-microsoft`, `msix-workload`, `winml-catalog`, `directory`) carry over unchanged from v1.1.
**Module**: session
**Companion-To**:
- [`3_design_classes.md`](3_design_classes.md) — **canonical class reference** (read this first)
- [`2_coreloop.md`](2_coreloop.md) — type taxonomy, core APIs, the two paths
- [`3_design_ep.md`](3_design_ep.md) — Tier 1/2/3 model and registration internals
- [`4_winml_device.md`](4_winml_device.md) — `WinMLDevice` ABC and per-EP subclasses

> **For the canonical class reference, see [`3_design_classes.md`](3_design_classes.md).** New readers should land there first to fix the class vocabulary before reading the requirements below.

---

## Table of Contents

- [1. Purpose](#1-purpose)
- [2. User-Facing Requirements](#2-user-facing-requirements)
  - [R1 — Scenario A: by-name intent](#r1--scenario-a-by-name-intent)
  - [R2 — Scenario B: by-listing-pick intent](#r2--scenario-b-by-listing-pick-intent)
  - [R3 — `--list-ep` enumeration](#r3---list-ep-enumeration)
  - [R4 — Stable identifier for Scenario B](#r4--stable-identifier-for-scenario-b)
- [3. Constraints](#3-constraints)
  - [C1 — Stability caveat](#c1--stability-caveat)
  - [C2 — No hardcoded EP names](#c2--no-hardcoded-ep-names)
  - [C3 — No back-compat shims](#c3--no-back-compat-shims)
- [4. Out of Scope](#4-out-of-scope)

---

## 1. Purpose

This doc captures **what the session/EP module must do for the user**, independent of *how* the types and APIs that implement it are shaped. It is the answer to "what does a user type, and what should the tool do?" Implementation lives in [`2_coreloop.md`](2_coreloop.md) (the type taxonomy and the two paths through the module) and [`3_design_ep.md`](3_design_ep.md) (the Tier 1/2/3 registration model).

Two user workflows drive every requirement below:

- The user knows roughly what they want (an EP name, a device class, or "auto"), and the tool picks a concrete match.
- The user has run `winml sys --list-ep` once, seen a table of every EP+source on the machine, and wants to bind one specific row.

Both workflows must produce either a working `WinMLSession` or a loud, actionable error. Silent fallback that picks "something close enough" is rejected.

---

## 2. User-Facing Requirements

### R1 — Scenario A: by-name intent

A user invokes a command with one or both of `--ep` and `--device`. Either may carry the literal string `"auto"`. The CLI looks like:

```
winml perf --ep openvino --device gpu          # explicit (both axes)
winml perf --ep openvino                       # ep only — device deduced from catalog
winml perf --device npu                        # device only — ep deduced from installed plugins
winml perf                                     # both omitted — auto-detect strongest backed device
winml perf --ep auto --device npu              # explicit "auto" sentinel — equivalent to --device npu
```

The tool must:

- Accept short names (`openvino`, `qnn`, `dml`, `cpu`, `npu`, `gpu`) and full names (`OpenVINOExecutionProvider`).
- Treat `"auto"` and omitted as equivalent on each axis.
- When `--device` alone is given, pick the **installed** EP that targets that device class — not the catalog primary on a host where that EP isn't installed.
- When both are omitted, walk the hardware-priority list and pick the strongest device that has a registered EP behind it. If no plugin EPs are installed, fall back to bundled CPU.
- Raise loudly when the resolved EP isn't installed (`ValueError("EP X not registered on this host. Hint: install the plugin or set WINML_EP_PATH.")`), when the named EP doesn't exist in the catalog at all, or when the named device class is unknown.
- After resolution, register the EP DLL, bind the matching `OrtEpDevice` handle, and build a session. Any failure in this chain (DLL load, no matching device, ambiguous match) raises with enough context to debug.

The session-building chain must not retry against a different EP or a different device class behind the user's back. If `--ep openvino --device gpu` fails, the user sees the failure — not a silent fallback to CPU.

### R2 — Scenario B: by-listing-pick intent

The user runs `winml sys --list-ep`, sees an enumeration of every EP+source combination on the machine, and identifies one specific row (e.g. a `shadowed` MSIX entry that they want to test, or an `incompatible` row they want to confirm is broken). They invoke a subsequent command with the row's identifier:

```
winml perf --ep openvino@pypi                          # specific source, default device
winml perf --ep openvino@msix-microsoft                # different source, default device
winml perf --ep openvino@msix-microsoft --device npu   # specific source AND device
winml perf --ep openvino@winml-catalog                 # MSIX EP delivered via the WinML EP Catalog API
```

The tool must:

- Parse `--ep <ep-short-name>@<source-tag>` syntactically; the presence of `@` is the trigger that selects this scenario over R1.
- Look up the row matching `(ep-short-name, source-tag)` from the registry's enumeration.
- When the source contributes multiple device classes and `--device` is omitted, raise `AmbiguousListingPick` with the list of available classes; do not silently pick one.
- When the user names a row that the listing rendered as `incompatible`, raise `IncompatibleListingPick` carrying the original registration error as `__cause__`. **Do not fall back** to a different row — the user explicitly named a broken one and gets to see why it broke.
- When `(ep-short-name, source-tag)` doesn't match any row (typo, never-installed source), raise `UnknownListingPick` with a hint to re-run `winml sys --list-ep`.

Scenario B's contract is intentionally stricter than Scenario A's: A silently deduces from partial input; B refuses to fall back from a specific row.

### R3 — `--list-ep` enumeration

`winml sys --list-ep` must produce a one-page inventory of EPs grouped by EP name, with one numbered entry per discovered source under each group. The render is **DLL-oriented**: an EP appears only if (a) at least one discoverable DLL was found for it, or (b) it is bundled with ORT.

Each numbered entry has a status:

- `primary` — first source under this EP name whose DLL registered cleanly and contributed at least one device.
- `shadowed` — subsequent source under the same EP name that also registered cleanly. Available as a fallback target for Scenario B but is *not* what Scenario A's deduction picks.
- `incompatible` — a discovered source whose DLL failed to register, or whose vendor hardware isn't present on this host. The entry shows the failure reason inline.

The listing must not show phantom rows for EPs that the catalog declares but whose DLLs aren't installed (no "Intel NPU/GPU/CPU lie" — see [`3_design_ep.md`](3_design_ep.md) §6.5). A row appears only when grounded by either a discovered DLL or a bundled-EP runtime fact.

The display style is illustrated in [`console_mockup.py`](console_mockup.py). Each row carries an identifier suitable for Scenario B (see R4 below).

### R4 — Stable identifier for Scenario B

The identifier shown in each `--list-ep` row must be:

- **Readable** — `openvino@pypi` is a name a user can re-type; an opaque hex hash is not.
- **Decodable** — given the identifier string and the discovery state at lookup time, the registry produces exactly zero or one matching row. Two distinct rows in the same listing must never share the same identifier.
- **Deterministic per discovery run** — given a fixed set of EP sources on disk, the identifier rendered for each row depends only on that set; running `winml sys --list-ep` twice in succession (with no install/uninstall between) produces the same identifiers.

The identifier shape is `<ep-short-name>@<source-tag>`, where `<source-tag>` is the shortest unique label among peers for the same EP name. The disambiguation algorithm is specified in [`2_coreloop.md`](2_coreloop.md) §6. The closed set of base tags is the canonical seven: `bundled`, `pypi`, `nuget`, `msix-microsoft`, `msix-workload`, `winml-catalog`, `directory`. The two `msix-*` tags are named after the *publisher namespace* of the matched MSIX family (Microsoft's official publisher vs the Windows Workload publisher); the `winml-catalog` tag refers to the WinML EP Catalog API (`Microsoft.Windows.AI.MachineLearning.ExecutionProviderCatalog`).

Determinism beyond a single run is **not** guaranteed — see C1 below.

---

## 3. Constraints

### C1 — Stability caveat

Discovery is deterministic given stable inputs, but the inputs themselves are NOT stable across user actions. Users can install/uninstall MSIX packages, install/remove PyPI/NuGet packages, or edit `WINML_EP_PATH` between invocations. When that happens, source tags may shift (e.g. a directory disambiguator can change if a new `WINML_EP_PATH` entry is added).

Scenario B identifiers are **stable within a stable environment session**, not across environment changes. Scripts that pin `--ep openvino@msix-microsoft` will continue to work as long as the user's MSIX install state matches when the script was written. If MSIX install state changes, the script may either run successfully against a different version, or fail with "source not found" — depending on whether the tag still has a match.

This caveat applies to ALL Scenario B identifiers, but is most acute for `directory-*` tags (where parent-dir basenames disambiguate) and version-disambiguated tags (where coexisting versions may change).

A stronger "stable across environment changes" identifier was considered (option B in the design discussion) and deferred. Option A — the `<ep>@<source-tag>` shape above — ships in v1 because the caveat is acceptable for the workflows we know of (testing, debugging, one-shot pinning). The structural follow-up is logged in [`2_coreloop.md`](2_coreloop.md) §8 (Open Questions).

**Persisted-config (`compiler/configs.py:285`) reload behavior.** The stability caveat surfaces in one specific user-visible way: old JSON compile configs (before the `source` field was added) reload as `EPDeviceTarget(ep="...", device="...", source=None)`. Path A's `resolve(target)` re-runs at load time and fills `target.source` from the **current** host's precedence winner — there is no version pinning, no silent fallback, and no compatibility shim. New JSON configs (with `source`) are validated against the current discovery state; a `source` tag that no longer exists raises `UnknownListingPick` rather than silently re-binding. See [`2_coreloop.md`](2_coreloop.md) §4.5 for the round-trip walkthrough.

### C2 — No hardcoded EP names

Per [`CLAUDE.md`](../../../CLAUDE.md) cardinal rule 1, the implementation must never hardcode EP-name lists. EP names live in the `EP_DEVICE_SPECS` catalog (see [`ep_device.py`](../../../src/winml/modelkit/session/ep_device.py)) and the `EP_PATH` source list (see [`ep_path.py`](../../../src/winml/modelkit/ep_path.py)). Resolution, registration, and rendering code must consume these tables — never duplicate or branch on EP-name literals.

The closed set of `source_kind` values listed in R4 is the one exception: it is a small, taxonomy-defining enumeration owned by the discovery layer. New source kinds are added by registering a new `EPSource` subclass (currently still spelled `EpSource` in `src/`; rename queued — see [`2_coreloop.md`](2_coreloop.md) §8), not by branching on strings elsewhere.

### C3 — No back-compat shims

Per the user's [no-back-compat preference](../../../../.claude/projects/C--Users-zhengte-BYOM-ModelKits-winml/memory/feedback_no_back_compat.md), the refactor that ships R1–R4 is a hard break. Old call sites that constructed `WinMLEPDevice(ep, device)` (the old pure-intent meaning) get rewritten to `EPDeviceTarget(ep, device, source=None)`. The name `WinMLEPDevice` is **reassigned** to the new flat `(WinMLEP, WinMLDevice)` pair (see [`2_coreloop.md`](2_coreloop.md) §2). There is no transitional alias, no `__class_getitem__` polyfill, no "if you pass two strings we'll figure it out" coercion.

Tests, internal callers, and downstream tooling are updated in the same PR that lands the new types.

---

## 4. Out of Scope

- **Vendor-specific device adapter internals.** What `OpenVINODevice.memory_bytes` reads from `ep_metadata`, how `DmlDevice` parses `DxgiVideoMemory`, the schema for QNN — all live in [`4_winml_device.md`](4_winml_device.md). R3's rendered facts use these adapters but their internals are not a requirement on this doc.
- **Plugin discovery mechanics.** How `WINML_EP_PATH` is parsed, how MSIX package families are enumerated, how NuGet caches are walked — see [`../../ep-path-design.md`](../../ep-path-design.md). R3 consumes `discover_all_eps()`'s output but doesn't constrain how it is produced.
- **Compile-specific session options.** `SessionOptions.AddConfigEntry`, EP-context flags, capability gating — see [`../compiler/3_design_spec.md`](../compiler/3_design_spec.md). Path A's session-build step is shared, but compile-mode tail flags are owned by the compiler design.
- **EPDoctor smoke-test design.** `winml sys --doctor` is a Tier 3 surface separate from R3's `--list-ep`. The smoke-model construction, subprocess wrapper, and error classification all live in [`3_design_ep.md`](3_design_ep.md) §7.
- **Multi-instance hardware disambiguation.** "The second NVIDIA GPU specifically" is deferred — single-instance behavior is the v1 default. See [`3_design_ep.md`](3_design_ep.md) §11.
