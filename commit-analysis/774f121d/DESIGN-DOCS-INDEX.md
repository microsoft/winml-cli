# Design / Plan / Review Docs Index — commit `774f121d`

**HEAD commit:** `774f121d feat(session): v2.9 unified-source EP refactor + WinMLSession redesign`
**Branch:** `feat/op-tracing-refactor_new-3`
**Mergebase with main:** `7a66c024`
**Cataloging date:** 2026-06-28

This index enumerates every design / plan / review / audit / verification document under `docs/design/session/`, `docs/design/`, and `temp/` (audit reports for the v2.9 squash itself) that pertains to the refactor squashed into commit `774f121d`. The squash carries 45 commits' worth of design iteration. Where a doc supersedes an earlier one in the same series, this is noted so a newcomer can read the surviving spec without redundantly walking the rejected drafts.

The catalog inherits 18 docs from the `a509a67` trail (with status updates where 774f121d advanced the implementation), adds 6 audit reports written during the 774f121d squash work (under `temp/`), and notes one new draft (`docs/design/session/2026-05-14-quant-spec-design.md`) that the squash partially executes.

---

## Landing page (canonical entry point — verified)

### `docs/design/session/2_coreloop.md`

- **Date:** 2026-06-09 (v2.7 banner) + v2.9 status banner inline at line 5
- **Doc type:** canonical design spec — "Session Core Loops — Scenarios, Classes, APIs, and Two Paths"
- **Status:** Active, banner-versioned. v2.9 status added at top to cover the BuiltinSource synthesis, idempotent `register_ep`, `_builtin_registered` cache, and the L1/L2 status taxonomy split. Doc drift remaining (see DEEP-DIVE D-05, D-06).
- **Summary:** The reference spec for the v2.9 EP-resolution flow. §2 fixes the six-class taxonomy (`EPDeviceTarget`, `EPDeviceSpec`, `EPEntry`, `WinMLDevice`, `WinMLEP`, `WinMLEPDevice`). §5 walks each API individually (`discover_all_eps`, `EPSource.resolve`, `resolve_device`, `WinMLDevice(handle)`, `register_ep`, `auto_device`, `WinMLSession.__init__`). §6 lays out Scenarios A.1-A.6 (Path A — user intent) and P.1-P.2 (programmatic + persisted JSON). §7 documents the `--list-ep` status taxonomy (L1 `[failed]` / L2 `[incompatible]` / `[primary]` / `[shadowed]`). §11 carries the appendices.
- **Key claims:**
  - The `EPSource` ABC + six concrete subclasses (`BuiltinSource`, `PyPISource`, `NuGetSource`, `DirectorySource`, `WinMLCatalogSource`, `MSIXPackageSource`) are the closed-set discovery taxonomy.
  - `WinMLEPRegistry.register_ep` is idempotent on `dll_path` (a cache hit returns the cached `WinMLEP`); the BuiltinSource branch is idempotent on `ep_name` via `_builtin_registered`.
  - `auto_device` walks candidates in precedence order, retries on `WinMLEPRegistrationFailed`, and raises `DeviceNotFound` if all candidates registered but none exposed `target.device`.
  - L1 vs L2 are independent layers — L1 = registration outcome, L2 = `EP_CATALOG.is_compatible(ep)`; both can be False simultaneously.
- **Open questions / drift surfaced by per-file review:**
  - §7.1.1 still says L2 is `entry.source.is_compatible()`; code is `EP_CATALOG.is_compatible(entry.ep_name)`. See `temp/sys_perf_flow_doc.md` D-01.
  - §5.5 pseudocode + §5.6 pseudocode + §5.7 still drift from current code (DEEP-DIVE D-05).
  - §11.7 still references `available_eps()` as "v2.17 memoizes" (implementation already shipped).
  - §11.1 `WinMLEPDevice` file:line refers to `ep_device.py:54` (wrong — it's at `ep_registry.py:142`).
  - §11.3 `ResolvedEp → EPEntry` rename is documented as "pending"; the rename is complete.
  - §6.3 still has a row for "Both given; resolved EP not in `available_eps()` → `ValueError`" — code does not implement that check (correctly removed in v2.6; the row is stale).

---

### `docs/design/session/1_req.md`

- **Date:** original from a509a67 trail, updated in 774f121d to drop `AmbiguousListingPick` raise reference
- **Doc type:** requirements
- **Status:** Active; v2.9-compatible.
- **Summary:** R1-R3 scenarios (R1 = Path A typed flow; R2 = source-pinning `--ep <name>@<source>`; R3 = `--list-ep` enumeration). v2.9 updates: R2 `AmbiguousListingPick` raise replaced with `DeviceNotFound` fall-through note.
- **Key claims:**
  - Scenario B's stricter contract: B refuses to fall back from a specifically-named row.
  - Multi-device-class ambiguity surfaces as `DeviceNotFound` after precedence loop exhaustion.

---

### `docs/design/session/3_design_classes.md`

- **Date:** updated during 774f121d squash
- **Doc type:** canonical class reference
- **Status:** Active; recent updates remove deleted classes (`AmbiguousListingPick`, `IncompatibleListingPick`, `AmbiguousMatch`).
- **Summary:** Per-class one-page reference. `EPDeviceTarget`, `EPDeviceSpec`, `EPEntry`, `WinMLDevice`, `WinMLEP`, `WinMLEPDevice`, `WinMLEPRegistry`, `WinMLSession`. The §8 exception taxonomy table is the authoritative list of what raises what.
- **Key claims:**
  - `WinMLDevice(handle)` is the direct constructor (no factory). The `wrap_ort_device` shim was deleted in v2.10.
  - Only `UnknownListingPick` survived the v2.9 Scenario-B exception trio.
  - `WinMLEPDevice.__post_init__` enforces the identity invariant: `device is one of ep.devices` (same object, not equality).

---

### `docs/design/session/3_design_ep.md`

- **Date:** original from a509a67 trail, updated in 774f121d for v2.9
- **Doc type:** EP-specific design supplement
- **Status:** Active.
- **Summary:** EP-registration mechanics (Stage 1 / Stage 2 partition), provider_options merge. v2.9 update: `raise AmbiguousMatch(...)` line in §6 pseudocode replaced with a "v2.9: deleted as dead code" comment.

---

### `docs/design/session/4_winml_device.md`

- **Date:** v1.5 (updated during 774f121d)
- **Doc type:** WinMLDevice single-class design supplement
- **Status:** Active; v1.5 changelog row records the `wrap_ort_device` deletion.
- **Summary:** Single concrete class with module-level dispatch tables for per-EP `ep_metadata` schemas. v1.5 records the factory's deletion: "Deletes the trivial `wrap_ort_device(d)` shim — it was a one-line forward to the constructor whose only justification (`_DEVICE_CLASSES` dispatch map) had already been collapsed in v1.4."

---

### `docs/design/session/5_type_taxonomy.md`

- **Date:** updated during 774f121d
- **Doc type:** taxonomy supplement
- **Status:** Active; updated to record the Scenario-B "two new classes → one new class" reduction.
- **Summary:** Locks the six-type taxonomy + naming principle. The "Two new Scenario B exception classes: `UnknownListingPick`, `AmbiguousListingPick`" claim was rewritten to "One new Scenario B exception class: `UnknownListingPick`" after v2.9 deleted `AmbiguousListingPick`.

---

### `docs/design/session/monitor/1_prd.md`

- **Date:** from a509a67 trail
- **Doc type:** product requirements doc for the op-tracing monitor
- **Status:** Active; v2.9 inherits FR-14 (op-type resolution chain) verbatim.
- **Summary:** Defines the op-tracing user-visible contract: `winml perf --monitor --op-tracing basic/detail`, `WinMLSession.perf(monitor=...)` context-manager API (FR-7), FR-14's L1→L2→L3→L4 op-type resolution chain (ONNX node.op_type → EP-authoritative QHAS → leaf-token heuristic → raw op_path), FR-9 HWMonitor context-manager composition.
- **Key claims:**
  - `session.perf(warmup=10, monitor=mon)` is a context manager that yields a `PerfContext(stats, monitor)`. Reaffirmed by 774f121d.
  - `EPMonitor.requires_session_teardown: ClassVar[bool]` ordering invariant (QNN flushes CSV only on `InferenceSession.__del__`, so the session is dropped *before* `monitor.__exit__`).

---

### `docs/design/session/monitor/2_coreloop.md`

- **Date:** v2.4.1
- **Doc type:** op-tracing flow / call-graph supplement
- **Status:** Active.
- **Summary:** Internal data-flow of QNN op-tracing: `WinMLSession.perf` sets monitor provider options → ORT runs → QNN CSV + QHAS files emitted → `_internal._extract_summary` parses → `OpTraceResult` returned.

---

## The pre-existing a509a67 trail (still active, status updated by 774f121d)

### `docs/design/session/2026-05-11-ep-device-refactor.md`

- **Date:** 2026-05-11 (v1.2 banner)
- **Doc type:** the seed spec for the entire EPDevice / WinMLSession refactor
- **Status:** **Superseded by 2_coreloop.md.** No v1.3 was ever written; instead the project pivoted to 2_coreloop.md as the canonical reference. Useful for archaeological context only.
- **Summary:** Originally specced `_QNN_BACKEND` map + `_ep_defaults` match-statement; both dissolved into the catalog `EP_DEVICE_SPECS` tuple by a509a67. 774f121d does not touch this doc.

### `docs/design/session/2026-05-12-cli-verification.md`

- **Date:** 2026-05-12
- **Doc type:** verification matrix (6/6 CLI commands × EP/device combinations)
- **Status:** Frozen at a509a67's HEAD `90b56e6d`. 774f121d's live verification (see SUMMARY.md "Verification") supersedes for this branch.

### `docs/design/session/2026-05-12-code-review/` (directory)

- **Doc type:** 23 per-file code reviews from a509a67 (one per `.py` under `src/winml/modelkit/session/` + adjacent dependents)
- **Status:** Superseded by `commit-analysis/774f121d/per-file/` (this trail) for the v2.9 squash. Useful for diffing the v2.9 review against the a509a67 baseline.

### `docs/design/session/2026-05-12-ep-taxonomy-sweep.md` + `2026-05-12-impl-status.md` + `2026-05-12-review-summary.md`

- **Date:** 2026-05-12
- **Doc type:** audit / status snapshots from a509a67's HEAD
- **Status:** Frozen pre-v2.9. The `2026-05-12-impl-status.md` table at line 198 flagged `_detect_best_device` as dead code; the 774f121d squash did not remove it (see DEEP-DIVE D-18).

### `docs/design/session/2026-05-13-*.md` (12 docs — taxonomy cleanup plan iterations v1/v2/v3 + verify + factcheck + audits + gap diagnostics)

- **Date:** 2026-05-13 sequence
- **Doc type:** mix of cleanup plans (v1/v2 rejected, v3 + v3-verify accepted), consolidation audit, post-refactor taxonomy audit, remaining-issues snapshot, two gap diagnostics (`gap1-diagnostic.md` — the dual-singleton DLL double-registration BLOCKER; `t6-analyze-crash-diagnostic.md` — the `analyze` command crash diagnostic).
- **Status:** Frozen pre-v2.9. The BLOCKERs they tracked were closed by a509a67 (commits `eb37f6c3` + `ec777caa`); 774f121d's BuiltinSource synthesis pushes the model further into "registry is the only authority" land. v3 + v3-verify are the surviving cleanup plans; v1/v2 are archaeology.

### `docs/design/session/2026-05-13-ep-device-spec-design.md`

- **Date:** 2026-05-13
- **Doc type:** the `EP_DEVICE_SPECS` catalog design
- **Status:** Active; load-bearing for 774f121d. Defines the catalog's ordering = preference convention and the per-row `default_provider_options` dict.

### `docs/design/session/2026-05-13-remaining-issues.md`

- **Date:** 2026-05-13
- **Doc type:** post-cleanup pre-merge snapshot for a509a67's HEAD `90b56e6d`
- **Status:** Superseded by 774f121d's `commit-analysis/774f121d/DEEP-DIVE.md` for the v2.9 branch. The two IMPORTANT items it deferred (consolidate the two EP-registration singletons; bump the design spec to v1.3) — both are addressed in 774f121d (BuiltinSource synthesis + 2_coreloop.md v2.9 status banner).

### `docs/design/session/2026-05-14-quant-spec-design.md`

- **Date:** 2026-05-14 (DRAFT)
- **Doc type:** design spec for a future `compiler/configs/quant_spec.py` module that would own `CalibrationConfig` and `QDQConfig` as typed value objects
- **Status:** DRAFT. **Partially executed by 774f121d.** The squash deleted `CalibrationConfig` and `QDQConfig` from `compiler/configs.py` in anticipation of the move — but did NOT land the replacement module, leaving `compiler/cli.py:15-17` with broken imports (DEEP-DIVE D-02).
- **Recommendation:** Either finish landing the quant_spec module or roll back the deletion in `configs.py`. **This is a ship blocker.**

---

## Top-level `docs/design/` subdirectories (cross-references)

The non-session subdirectories under `docs/design/` are referenced by 774f121d only at the import / facade level. They are not central but worth noting:

- **`docs/design/cli/`** — CLI command design. References the `EpAtSourceParamType` introduced by 774f121d.
- **`docs/design/compiler/`** — Compiler design. The DRAFT quant_spec doc above lives in `session/` rather than here.
- **`docs/design/config/`** — Build / runtime config schemas. Touched by 774f121d's `config/precision.py` taxonomy purge.
- **`docs/design/perf/`** — `winml perf` design. References `WinMLSession.perf(monitor=...)` context-manager from monitor/1_prd.md.
- **`docs/design/e2e_eval/`** — `winml eval` design. The `eval.py` `"auto"` passthrough (DEEP-DIVE D-19) needs verification against this spec.
- **`docs/design/build/`**, **`docs/design/importtime/`**, **`docs/design/inspect/`**, **`docs/design/logging/`**, **`docs/design/static_analyzer/`** — out of scope for the v2.9 refactor; mentioned for completeness.

---

## Audit reports written during the 774f121d squash

These live in `temp/` (gitignored) but document the audit chain that produced the v2.9 commit and surface findings that this DEEP-DIVE folds in.

### `temp/review_findings_2026-06-19.md`

- **Doc type:** initial 21-finding code review (F-01 through F-21) of the v2.9 unified-source refactor
- **Status:** Closed. F-01/F-02 were the dual `register_ep` double-call CRITICALs; F-03..F-10 were IMPORTANT bug fixes; F-11..F-12 PLAUSIBLE; F-13..F-18 CLEANUP; F-19..F-21 DOCS. All except F-15 (architectural — BuiltinSource abstraction altitude) and F-16 (lazy ORT init — perf tradeoff) were resolved during the squash.

### `temp/v2_9_review_doc.md`

- **Doc type:** independent reviewer agent's pass over the squash diff
- **Status:** Closed. 12 findings (R-01 through R-12). All confirmed by the next doc.

### `temp/v2_9_factcheck_doc.md`

- **Doc type:** fact-checker agent's verdicts on `v2_9_review_doc.md`
- **Status:** Closed. 11 CONFIRMED + 1 PARTIALLY-CONFIRMED (R-06 — both branches of `register_ep` need the `get_ep_devices` guard, not just one).

### `temp/v2_9_convergence_report.md`

- **Doc type:** post-fix convergence verification
- **Status:** Closed. 14/14 fixes converged; 2 NEW findings surfaced (`2_coreloop.md` §11.7 stale rows for `available_eps()` and `AmbiguousListingPick`).

### `temp/sys_perf_flow_doc.md`

- **Doc type:** narrative + design-cross-reference of the `winml sys` and `winml perf` flows
- **Status:** Active reference for newcomers. Surfaced 10 doc drifts D-01..D-10 in `2_coreloop.md` and two in DEEP-DIVE-this-doc (D-05, D-06).

### `temp/resolve_device_audit.md`

- **Doc type:** scenario-matrix audit of `resolve_device` + test coverage analysis
- **Status:** Active reference. 6/6 A-scenarios + 2/2 P-scenarios implemented correctly in code; 2 doc drifts (§6.3 stale failure row; §6.1 false "normalized to None" claim); 3 test coverage gaps (A.2 integration, P.2 stale-source `UnknownListingPick`, RED-comments stale).

### `temp/session_function_audit.md`

- **Doc type:** function-by-function necessity audit of `src/winml/modelkit/session/`
- **Status:** Active backlog. 3 high-confidence DELETE-verdict items (`_detect_best_device`, `_get_install_suggestion`, `WinMLEP.ep_devices`); 1 DELETE+replace (`_is_verbose`); 3 INLINE; 1 MERGE (`_format_bytes` duplicated in `report.py`); 3 NEW-RISK items (all kept after review).

### `temp/end_to_end_verification.md`

- **Doc type:** independent verification report run after the recent `_format_device_types` deletion and indent-constants refactor
- **Status:** Closed. GREEN verdict on all changes after a regression catch (auto.py:411 needed `.lower()`).

---

## Reading order recommended for a newcomer

The trail is large; the recommended reading order is **goal-then-evidence** rather than chronological.

1. **`commit-analysis/774f121d/SUMMARY.md`** — orient. The eight architectural moves give the big picture in ~5 min.
2. **`docs/design/session/2_coreloop.md`** §1–§6 — the canonical scenario matrix. Skim §5 if pressed for time.
3. **`commit-analysis/774f121d/per-file/ep_path.md`** + **`session__ep_device.md`** + **`session__ep_registry.md`** — understand the unified `EPSource` taxonomy + `EPCatalog` + the singleton registry.
4. **`commit-analysis/774f121d/per-file/commands__sys.md`** + **`commands__perf.md`** — the two end-to-end Path A / Path B flows that exercise everything above.
5. **`commit-analysis/774f121d/DEEP-DIVE.md`** — the impl-vs-design cross-reference. Read at minimum §D-01..D-04 (the ship blockers).
6. **`docs/design/session/monitor/1_prd.md`** + **`commit-analysis/774f121d/per-file/session__monitor__qnn_monitor.md`** + **`session__monitor__qnn___internal.md`** — only if you're doing op-tracing work.
7. **`temp/v2_9_factcheck_doc.md`** + **`temp/sys_perf_flow_doc.md`** + **`temp/resolve_device_audit.md`** + **`temp/session_function_audit.md`** — for the running backlog of issues / drifts / dead code.
8. **The pre-v2.9 `2026-05-*.md` docs** — read only for archaeology, or when reviewing a related a509a67 commit.

## Documents the squash references but DID NOT update

- **`docs/design/session/2026-05-14-quant-spec-design.md`** — the quant_spec module was deleted from `compiler/configs.py` but never landed. **Ship blocker** (DEEP-DIVE D-02).
- **`docs/design/session/2_coreloop.md`** §6.3 third failure row, §7.1.1 L2 mechanism prose, §7.1.2 pseudocode, §11.7 stale rows, §11.1/§11.3 file-line drifts — listed under DEEP-DIVE D-05, D-06 and `temp/sys_perf_flow_doc.md` D-01..D-10. Defer to a v2.9 doc-cleanup PR.

## Documents the squash retired implicitly

By landing `BuiltinSource` synthesis + `register_ep` idempotency + `EpAtSourceParamType` + the `--list-ep` rewrite, the squash supersedes the following docs without explicitly marking them retired:

- The 2026-05-13 v1 / v2 taxonomy-cleanup plans (already rejected pre-a509a67, but the project no longer needs them as historical archaeology — v3 + v3-verify suffice).
- The 2026-05-12 code-review per-file docs — superseded by this doc trail.

---

**Doc count:** 18 active design / spec docs in `docs/design/session/` + 2 monitor docs + 6 audit reports in `temp/` + this commit's three top-level synthesis docs = **29 documents** in the doc-trail for the v2.9 squash. The previous a509a67 trail catalogued 42; the 774f121d trail is leaner because the squash leverages the design work a509a67 already settled rather than introducing new spec material.
