// Session / EP Module — Core Path & Core Design
// Compile: `typst compile coreloop_deck.typ coreloop_deck.pdf`
// Or watch: `typst watch coreloop_deck.typ`

#import "@preview/polylux:0.4.0": *

#set page(paper: "presentation-16-9", margin: 1.5em)
#set text(size: 22pt)
#show heading.where(level: 1): set text(size: 32pt, weight: "bold", fill: rgb(0, 102, 178))
#show heading.where(level: 2): set text(size: 24pt, weight: "bold", fill: rgb(60, 60, 60))
#show raw: set text(size: 16pt)

// ============================================================================
// Title
// ============================================================================
#slide[
  #set align(center + horizon)
  #text(size: 44pt, weight: "bold", fill: rgb(0, 102, 178))[
    Session / EP Module
  ]
  #v(0.5em)
  #text(size: 28pt)[Core Path \& Core Design]
  #v(1em)
  #text(size: 18pt, fill: gray)[
    Derived from `docs/design/session/2_coreloop.md` v2.4
  ]
]

// ============================================================================
// Purpose
// ============================================================================
#slide[
  = What this module solves

  Turn _"the user wants an EP and a device"_ into _"an `ort.InferenceSession`
  bound to the right `OrtEpDevice` handle."_

  #v(1em)

  Two end-to-end paths:
  - *Path A* — one user intent → one session
  - *Path B* — enumerate all → render report

  #v(1em)

  #align(center)[
    #box(stroke: 0.5pt, inset: 8pt, fill: rgb(245, 245, 250))[
      _CLI args_ ↓ #h(0.5em) _typed intent_ ↓ #h(0.5em) _resolved target_ #h(0.5em)
      ↓ #h(0.5em) _registered EP_ ↓ #h(0.5em) _matched device pair_ ↓ #h(0.5em)
      _InferenceSession_
    ]
  ]
]

// ============================================================================
// User Scenarios — overview
// ============================================================================
#slide[
  = User Scenarios — 9 actions, 2 paths

  *Path A* (single intent → single session) — 8 scenarios:
  - *A.1–A.4* by-name: `--ep openvino --device gpu` / `--ep openvino` / `--device npu` / bare
  - *A.5–A.6* by-listing-pick: `--ep openvino@pypi[ --device npu]`
  - *P.1* programmatic SDK / *P.2* persisted JSON config

  #v(0.8em)

  *Path B* (enumerate all → render) — 2 scenarios:
  - *B.1* `winml sys --list-ep`
  - *B.2* `winml sys --doctor` (#text(fill: orange)[PROPOSED])

  #v(0.8em)

  #text(size: 18pt, fill: gray)[
    All scenarios route through the same six-class taxonomy and seven-API
    surface. The CLI shape determines which path; the rest is shared.
  ]
]

// ============================================================================
// Class Taxonomy
// ============================================================================
#slide[
  = Six Classes, One Role Each

  #table(
    columns: (auto, 1fr, auto),
    inset: 6pt,
    align: (left, left, center),
    stroke: 0.4pt,
    [*Class*], [*Role*], [*Prefix*],
    [`EPDeviceTarget`], [User intent — `ep`, `device`, optional `source`], [—],
    [`EPDeviceSpec`], [Catalog row — what *could* exist for an EP], [`WinML*`],
    [`EPEntry`], [Filesystem-discovery record], [—],
    [`WinMLDevice`], [Adapter over `ort.OrtEpDevice` — single concrete class], [`WinML*`],
    [`WinMLEP`], [Successful per-source registration aggregate], [`WinML*`],
    [`WinMLEPDevice`], [Flat `(ep, device)` pair — mirror of `ort.OrtEpDevice`], [`WinML*`],
  )

  #v(0.8em)

  *Naming rule:* `WinML*` = predefined or system-generated (cannot be
  crafted from CLI strings; requires system API).
  No prefix = user-craftable (built from strings / JSON config / tests).
]

// ============================================================================
// Core APIs — overview
// ============================================================================
#slide[
  = Core APIs — 7 Primitives

  #table(
    columns: (auto, 1fr),
    inset: 6pt,
    align: (left, left),
    stroke: 0.4pt,
    [*API*], [*Returns*],
    [`discover_all_eps()`], [`list[EPEntry]` — flat filesystem scan],
    [`EPSource.resolve()`], [`Iterator[EPEntry]` — per-subclass],
    [`resolve_device(target)`], [`EPDeviceTarget` — pure deduction],
    [`WinMLDevice(handle)`], [`WinMLDevice` — direct constructor (no factory; v2.10)],
    [`WinMLEPRegistry.register_ep(entry)`], [`WinMLEP` — atomic, idempotent],
    [`WinMLEPRegistry.auto_device(target)`], [`WinMLEPDevice` — compound, retry-shadowed],
    [`WinMLSession(...)`], [`ort.InferenceSession` — Path A user entry],
  )

  #v(0.6em)

  #text(size: 18pt, fill: gray)[
    `register_ep` is atomic per-source. `auto_device` composes it with
    candidate-filter, source-tag dispatch, and shadowed-source retry.
  ]
]

// ============================================================================
// Two Decompositions — Paths × Tiers
// ============================================================================
#slide[
  = Two Decompositions — Orthogonal Axes

  *Path* = user-facing (which scenario the user is in).
  *Tier* = internal (which kind of work the layer is doing).

  #v(0.4em)

  #table(
    columns: (auto, auto, auto, auto),
    inset: 6pt,
    align: (left, center, center, center),
    stroke: 0.4pt,
    [], [*Tier 1 — Discovery*], [*Tier 2 — Registration*], [*Tier 3 — Validation*],
    [Cost / side-effect],
      [filesystem scan only \ (no DLL load)],
      [`register_execution_provider_library` \ (one DLL per call)],
      [smoke-test inference \ (small model run)],
    [Where it lives],
      [`discover_all_eps()` \ `EPSource.resolve()`],
      [`WinMLEPRegistry.register_ep` \ `WinMLEPRegistry.auto_device`],
      [`EPDoctor.diagnose()` \ #text(fill: orange)[(PROPOSED)]],
    [Path A uses?],
      [✓ via `_entries` cache],
      [✓ `auto_device` → one `WinMLEPDevice`],
      [✗],
    [Path B uses?],
      [✓ full broad walk],
      [✓ per-`EPEntry` inline loop],
      [✓ (`--doctor` only)],
  )

  #v(0.4em)

  #text(size: 18pt, fill: gray)[
    Every path consumes Tier 1 + Tier 2; only Path B's `--doctor` tail
    consumes Tier 3.
  ]
]

// ============================================================================
// Path A — Flow
// ============================================================================
#slide[
  = Path A — User Intent → Session

  ```
  CLI args / JSON config / Python SDK
            │
            ▼  EPDeviceTarget(ep, device, source?)
  ┌─────────────────────────┐
  │  resolve_device(target) │   pure deduction; fills "auto",
  └─────────────────────────┘   validates source against discovery
            │
            ▼  EPDeviceTarget — concrete, validated
  ┌─────────────────────────────────┐
  │  registry.auto_device(target)   │   filter EPEntries by ep+source,
  └─────────────────────────────────┘   retry-shadowed on failure
            │
            ▼  WinMLEPDevice(ep: WinMLEP, device: WinMLDevice)
  ┌──────────────────────────────────────┐
  │  WinMLSession(onnx_path, ep_device)  │   eager InferenceSession build
  └──────────────────────────────────────┘
            │
            ▼  ort.InferenceSession
  ```
]

// ============================================================================
// Path B — Flow
// ============================================================================
#slide[
  = Path B — Enumerate All → Report

  ```
  CLI: winml sys --list-ep
            │
            ▼
  ┌────────────────────────────────────────┐
  │  for entry in discover_all_eps():      │   broad scan;
  │      try:                              │   each entry triggers
  │          ep = registry.register_ep(    │   one DLL load via
  │              entry)                    │   register_ep
  │          results.append(ep)            │
  │      except WinMLEPRegistrationFailed: │   failures captured
  │          failures.append((entry, e))   │   as data
  └────────────────────────────────────────┘
            │
            ▼  (list[WinMLEP], list[(EPEntry, Exception)])
  ┌──────────────────────────┐
  │  render: primary /       │   per-row grouping with live
  │  shadowed / incompatible │   device facts from WinMLDevice
  └──────────────────────────┘
  ```
]

// ============================================================================
// Scenario B — Stable Identifier
// ============================================================================
#slide[
  = Scenario B — `--ep <name>@<source-tag>`

  *Seven canonical source tags* (closed set):
  #align(center)[
    `bundled` · `pypi` · `nuget` · `msix-microsoft` · `msix-workload` ·
    `winml-catalog` · `directory`
  ]

  #v(0.8em)

  *Round-trip:*

  ```bash
  $ winml sys --list-ep
  OpenVINOExecutionProvider
    [primary]    PyPI    onnxruntime-ep-openvino 1.4.1
    [shadowed]   Catalog (catalog default)
    [shadowed]   MSIX    MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.1.8

  $ winml perf --ep openvino@msix-microsoft --device npu  # ← user pastes back
  ```

  CLI parser splits on `@`; `EPDeviceTarget.__post_init__` validates the tag
  against `VALID_SOURCE_TAGS`. `auto_device` filters discovered candidates
  by `_entry_source_tag(e) == target.source`.
]

// ============================================================================
// Construction-time validation
// ============================================================================
#slide[
  = Construction-Time Validation

  `EPDeviceTarget.__post_init__` rejects bad input at construction —
  CLI parse, JSON load, test fixture — not 5 layers deeper.

  #v(0.6em)

  #table(
    columns: (auto, 1fr),
    inset: 6pt,
    align: (left, left),
    stroke: 0.4pt,
    [*Field*], [*Allowed*],
    [`device`], [`"auto"` or one of `VALID_DEVICES` = `{"npu","gpu","cpu"}`],
    [`ep`], [`"auto"` or `known_ep_short_names()` (derived from catalog, not hardcoded)],
    [`source`], [`None` or one of `VALID_SOURCE_TAGS` (the 7 canonical tags)],
  )

  #v(0.6em)

  *Structural only* — checks shape, not environment fit. Whether the EP
  is registered on this host, or whether the source tag has a match in
  `discover_all_eps()`, is `resolve_device()`'s job.

  *No hardcoded EP names* (CLAUDE.md cardinal rule #1) — `known_ep_short_names()`
  derives from `_SHORT_TO_FULL`.
]

// ============================================================================
// Key Design Decisions
// ============================================================================
#slide[
  = Key Design Decisions

  - *Two paths, one taxonomy* — Path A (single) and Path B (broad) share the
    same six classes and seven APIs

  - *Atomic + compound registry* — `register_ep(entry) → WinMLEP` is per-source
    atomic; `auto_device(target) → WinMLEPDevice` composes filter + retry-shadowed

  - *Failures-as-data on Path B* — `(EPEntry, Exception)` pairs alongside
    successful `WinMLEP`s; renderer renders both

  - *No back-compat shims* — hard-break naming, signature changes, JSON shape
    changes (forward-compat via `from_dict`'s `.get()` pattern)

  - *Single discovery cache* — `WinMLEPRegistry.__init__` calls
    `discover_all_eps()` once; `auto_device` and `resolve_device` read from
    the cached `self._entries` instead of re-scanning

  - *Single concrete `WinMLDevice` class* — not ABC + 5 subclasses (over-
    engineered for v1; dispatch via `self._ort.ep_name`)
]

// ============================================================================
// What's next
// ============================================================================
#slide[
  = What's next

  *Pending follow-ups* (backlogged tasks):

  - *Batch F* — CLI `@` parser fully wired into `commands/perf.py` /
    `commands/compile.py` (currently constructs `EPDeviceTarget` without
    splitting `@`)

  - *Batch G* — fix the `--list-ep` duplicate-MSIX-DLL render
    (v2.9 deleted `AmbiguousListingPick`; multi-class ambiguity now
    falls through to `DeviceNotFound`)

  - *Batch H* — `WinMLEPDevice.__post_init__` invariant assertion
    (`.device in .ep.devices`)

  - *Future PR* — `--doctor` smoke test (Path B.2 PROPOSED in design only)

  #v(0.8em)

  *Out of scope:* Tier 1/2/3 internals (`3_design_ep.md`),
  `WinMLDevice` dispatch tables (`4_winml_device.md`),
  monitor's per-op tracing (`monitor/2_coreloop.md`).
]

// ============================================================================
// Thank you
// ============================================================================
#slide[
  #set align(center + horizon)
  #text(size: 36pt, weight: "bold", fill: rgb(0, 102, 178))[
    Q\&A
  ]
  #v(1em)
  #text(size: 20pt)[
    Full spec: `docs/design/session/2_coreloop.md` v2.4 \
    Canonical class reference: `docs/design/session/3_design_classes.md` v1.1
  ]
]
