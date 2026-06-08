# Session/EP Type Taxonomy — Superseded Stub

**Version**: 2.2
**Date**: 2026-06-07
**Status**: Superseded. The canonical class reference is [`3_design_classes.md`](3_design_classes.md); the path-level flows + class inventory appendix live in [`2_coreloop.md`](2_coreloop.md).
**Module**: session

> **For the canonical class reference, see [`3_design_classes.md`](3_design_classes.md).**

---

This doc's content has been **folded into [`3_design_classes.md`](3_design_classes.md) (the canonical class reference) and [`2_coreloop.md`](2_coreloop.md) (the path-level flows + appendix inventory)** for coherence — the type taxonomy, the core API surface, the resolver, the path-level flows, and the class inventory now all live next to each other rather than being scattered across multiple docs.

Pointers:

- **Canonical class reference (every class with role, lifecycle, usage example)** — [`3_design_classes.md`](3_design_classes.md).
- **Six-class table + naming principle** — [`2_coreloop.md`](2_coreloop.md) §2 (also recapped in [`3_design_classes.md`](3_design_classes.md) §3).
- **Core API surface** (`discover_all_eps`, `EPSource.resolve()`, `resolve`, `registry.register_ep`, `wrap_ort_device`, `WinMLSession(...)` direct constructor) — [`2_coreloop.md`](2_coreloop.md) §3.
- **`WinMLEPRegistry` lockdown** (state, singleton pattern, one public method) — [`2_coreloop.md`](2_coreloop.md) §3.1.
- **`_find_entry` tag-decode helper** — [`2_coreloop.md`](2_coreloop.md) §3.2.
- **Path A composition** (Scenarios A.1–A.4 by-name, A.5–A.6 by-listing-pick, P.1 programmatic, P.2 persisted-config; failure modes) — [`2_coreloop.md`](2_coreloop.md) §4.
- **Path B composition** (`--list-ep` and `--doctor` tails; inline 6-line loop) — [`2_coreloop.md`](2_coreloop.md) §5.
- **Stable identifier for Scenario B** — [`2_coreloop.md`](2_coreloop.md) §6.
- **Full class inventory** (every existing class in EP/session/discovery/sysinfo, with verdicts) — [`2_coreloop.md`](2_coreloop.md) §9.

**State changes since v2.1:**

- The previously-planned `WinMLSession.build` classmethod is **dropped**. The Path A user-facing tail is the direct `WinMLSession(onnx_path, ep_device, *, ep_config=None, ep_monitor=None, base_session_options=None)` constructor — the existing `__init__` at `session/session.py:200` keeps its eager-creation shape, with the new optional `ep_monitor` kwarg.
- Three new Scenario B exception classes: `UnknownListingPick`, `IncompatibleListingPick`, `AmbiguousListingPick`. See [`2_coreloop.md`](2_coreloop.md) §4.2 (raised-by spec) and §9.4 (inventory).

The taxonomy itself has changed shape since this doc's v1.1 — the locked-in set is now **six** classes (`EPDeviceTarget`, `EPDeviceSpec`, `EPEntry`, `WinMLDevice`, `WinMLEP`, `WinMLEPDevice`-the-pair), with `WinMLEP` redefined as the success-only per-source aggregate and `WinMLEPDevice` reassigned to the flat `(WinMLEP, WinMLDevice)` pair (the project's typed mirror of `ort.OrtEpDevice`). Failures live as `(EPEntry, Exception)` pairs alongside, not as a nullable field on `WinMLEP`. See [`2_coreloop.md`](2_coreloop.md) §2 for the table and naming principle.

**Naming note for historical references.** Earlier drafts (this doc's v1.1 appendix below, the `2026-05-13-*` audit-trail docs) used a different scratch name, `EPSource`, for the per-source filesystem-discovery record. That role is now filled by **`EPEntry`** — the renaming is locked in `2_coreloop.md` §2. `EPSource` survives as the name of the discovery ABC (the `EpSource → EPSource` rename of the existing `ep_path.py` class is queued — see `2_coreloop.md` §8 and §9.3). When reading old commit history or older PRs, treat `ResolvedEp` and the scratch-name `EPSource` (used as a record class) as both superseded by `EPEntry`.

---

## Appendix — Historical Open Design Issues (2026-06-06)

The v1.1 §12 entries below are preserved here as historical context. The redesigns that resolved them landed in [`2_coreloop.md`](2_coreloop.md); the entries themselves are kept because the comparisons in them (especially the resolve-vs-resolve_device side-by-side) are useful when reading old commit history or PRs.

### 12.1 — `resolve()` vs existing `resolve_device`

The earliest sketch defined `resolve(target, discovered) -> EPDeviceTarget` taking an injected `list[WinMLEP]`. The existing `resolve_device(ep, device)` in [`ep_device.py`](../../../src/winml/modelkit/session/ep_device.py) lines 369-446 was strictly stronger in every dimension that mattered: hardware-grounded auto-detect (vs hardcoded EP precedence), registration-aware device-only deduction, no caller-side boilerplate, lru-cached `available_eps()` consultation rather than per-call list-passing, and locked-in error-message hints. The locked-in `resolve` in [`2_coreloop.md`](2_coreloop.md) §3 preserves `resolve_device`'s shape verbatim and extends it to also fill `target.source` from the precedence winner so the resolved target is self-describing.

### 12.2 — Two user-input scenarios for picking an EP+device

The realization that drove the Scenario A vs Scenario B framing: the user can name a target by *intent* (short names + `"auto"`, with the system filling in deduced values) or by *listing pick* (an exact `(ep, source)` row they saw in `--list-ep`). The two have different validation strictness — A silently deduces from partial input; B refuses to silently substitute. Both terminate at the same four-step Path A composition; the difference is only in `resolve`'s strictness and in whether `target.source` was user-supplied or precedence-filled. See [`2_coreloop.md`](2_coreloop.md) §4.

The earlier draft modeled these as two separate sub-paths (Path A1, Path A2) with two separate resolvers (`resolve_target`, `resolve_listing_pick`). That split was dropped in v2.0 of [`2_coreloop.md`](2_coreloop.md): one `resolve` function dispatches on whether `target.source` is set, one `register_ep` is the registration sink for both, and the four-step composition is identical. The Scenario A / Scenario B framing remains as a documentation-level distinction.
