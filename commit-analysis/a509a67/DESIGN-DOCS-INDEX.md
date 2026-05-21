# Design / Plan / Review Docs Index — commit `a509a67`

**HEAD commit:** `a509a67 feat(session): op-tracing perf monitor + EPDevice/WinMLSession refactor`
**Branch:** `feat/op-tracing-refactor_3`
**Cataloging date:** 2026-05-15

This index enumerates every design / plan / review / audit / verification document
under `docs/design/session/` and `docs/plans/` that pertains to the EPDevice / WinMLSession /
op-tracing refactor squashed into commit `a509a67`. The commit body declares this
trail to be 39 documents; this index catalogs **42** discoverable docs (4 plans + 7
top-level session design docs + a 6-doc taxonomy-cleanup sub-trail + 24 per-file code
reviews + 13 monitor-design docs in `monitor/` and `monitor/iterations/` + 1 DRAFT
QuantSpec design + 1 spec-design doc); the commit-body count of 39 is approximate.

---

## Landing page (canonical entry point — verified)

### `docs/design/session/2026-05-13-remaining-issues.md`

- **Date:** 2026-05-13
- **Doc type:** audit / status snapshot
- **Status:** final for the post-cleanup pre-merge snapshot at HEAD `90b56e6d` (NB: HEAD has since advanced through v2/v3 taxonomy cleanups and the QuantSpec draft)
- **Summary:** Post-cleanup snapshot enumerating the 11-commit chain since base `1bea4cf`, the verified 6/6 CLI verification matrix at HEAD `90b56e6d`, and the four open-issue buckets organized by severity (BLOCKER=0, IMPORTANT=2, MEDIUM=3, OUT-OF-SCOPE=2, PROCESS=4). It also cross-references every companion doc in the trail. The doc explicitly defers two IMPORTANT items to a follow-up PR (consolidate the two EP-registration singletons; bump the design spec to v1.3 to match shipped behaviour) and notes three MEDIUMs that could be cleaned in this PR (analyze slow probing; dead `_build_provider_options` method on `CompileStage`; `models/auto.py` positional `ep_device` audit).
- **Key claims:**
  - 0 BLOCKERs at HEAD `90b56e6d`; 11 commits since base.
  - All 6 CLI commands (`winml perf` × 3 ep/device combinations + `winml compile` + ctx perf + HF perf) return EXIT 0 with documented latencies (2.27–2.63 ms avg for perf paths).
  - Symmetric defensive guards on both `WinMLEPRegistry` and `winml.py:WinML` (commits `eb37f6c3` + `ec777caa`) close the dual-singleton DLL double-registration BLOCKER documented in `2026-05-13-gap1-diagnostic.md` and `2026-05-13-t6-analyze-crash-diagnostic.md`.
  - `WinMLSession.perf()` is implemented as a `@contextmanager` (intentional deviation from spec v1.2 which implied a regular method — needs spec v1.3 update).
  - Native QNN HTP AOT crash on QDQ-quantized graphs (O1) is upstream QNN SDK and not in scope.
- **Open questions / rejected options:** Five process items (force-push, update PR description, full pytest gate, rebase on `feat/update-pkg-deps`) are flagged as pending user discretion.

---

## Reading order recommended for a newcomer

The trail is largely chronological, but several documents supersede earlier ones in
the same series. The recommended reading order is **goal-then-evidence** rather than
strictly chronological:

1. **Start with the canonical EPDevice design.** Read `2026-05-11-ep-device-refactor.md` (v1.2 spec) to understand the problem statement, the proposed `EPDevice` descriptor + `resolve_device` API, and the hard-break migration. This is the seed doc; everything else extends, validates, or contradicts it.
2. **Read the implementation plan.** `docs/plans/2026-05-11-ep-device-refactor-plan.md` is the 14-task TDD plan that drove the bulk of the code change.
3. **Read the implementation audit + first CLI verification.** `2026-05-12-impl-status.md` and `2026-05-12-cli-verification.md` capture the gap between spec and implementation at the first CLI run (3/6 PASS).
4. **Read the code review summary.** `2026-05-12-review-summary.md` aggregates the per-file reviews under `2026-05-12-code-review/` and flags 9 issues across 25 files. Dip into specific per-file review docs as needed when investigating individual modules.
5. **Read the taxonomy sweep.** `2026-05-12-ep-taxonomy-sweep.md` catalogs the 47 places where EP/device knowledge was duplicated before consolidation.
6. **Read the taxonomy consolidation plan.** `docs/plans/2026-05-13-ep-taxonomy-consolidation-plan.md` (Phase 1 + Phase 2) executes the sweep.
7. **Read the post-consolidation audit.** `2026-05-13-consolidation-audit.md` verifies the plan's Phase 1 + Phase 2 commits.
8. **Read the BLOCKER diagnostics in order.** `2026-05-13-gap1-diagnostic.md` (original root-cause) then `2026-05-13-t6-analyze-crash-diagnostic.md` (the follow-up that found the fix protected the wrong caller). These two docs document the dual-singleton DLL double-registration that produced exit 127 / `STATUS_DLL_NOT_FOUND` in the `winml perf <hf-model>` path.
9. **Read the second CLI verification.** `2026-05-13-cli-claims-reverify.md` shows the 5/6 PASS state after `eb37f6c3` (still broken on T6 before the symmetric guard).
10. **Read the post-BLOCKER cleanup plan.** `docs/plans/2026-05-13-post-blocker-cleanup-plan.md` covers bundles A/B/C/D (monitor silent failures, `_build_session_options` mutation safety, taxonomy gaps, analyze slow probing).
11. **Read the landing page.** `2026-05-13-remaining-issues.md` is the deliberate exit point — read it last to ground the trail in the snapshot.
12. **Read the EPDeviceSpec catalog redesign and its cleanup sub-trail.** `2026-05-13-ep-device-spec-design.md` introduces the typed `EPDeviceSpec` catalog (single source of truth, 13 entries). The follow-up audit at `2026-05-13-post-refactor-taxonomy-audit.md` finds 6 items, which feed the v1 → v2 → v3 taxonomy-cleanup plan progression in §"Taxonomy cleanup".
13. **Read the QuantSpec DRAFT.** `2026-05-14-quant-spec-design.md` extends the EPDeviceSpec pattern with a `QuantSpec` field. **The commit body labels this DRAFT — do not implement.**
14. **(Optional)** Read the op-tracing monitor PRD + coreloop + 11 iteration docs in `monitor/` for the pre-history of the op-tracing-monitor surface that landed alongside (not orthogonal to) the EPDevice work.

Rationale for the goal-then-evidence ordering: the trail has so many diagnostics and
audits that a strict chronological read drowns the reader in 4–5 sweeps before they
have an anchor for what's being swept against. The seed doc + plan + first audit give
that anchor in three documents.

---

## Documents grouped by theme

### Theme 1 — EPDevice catalog design (the seed)

#### `docs/design/session/2026-05-11-ep-device-refactor.md`

- **Date:** 2026-05-11
- **Doc type:** design spec
- **Status:** v1.2 — implementation pending at write time; later marked spec-drift (needs v1.3 bump per landing page §I2)
- **Summary:** The canonical EPDevice/WinMLSession design. Replaces today's non-deterministic `_find_ep_device(ep_name)` first-match selection with a frozen `EPDevice` dataclass + a typed `resolve_device(ep, device)` resolver + a `WinMLEPRegistry.register_ep` additive method + a free-function `_build_session_options`. Hard break (Option A): no autoep, no policy paths, no compatibility shims; every call site is updated in this PR. Defines the 5-exception error taxonomy (`EPNotDiscovered`, `EPRegistrationFailed`, `DeviceNotFound`, `AmbiguousMatch`, `EPMonitorMismatch`) and the three-layer provider-options merge (EP defaults → user → monitor wins last). Dissolves `sysinfo/device.py`: the old `sysinfo.resolve_device(device="auto") -> tuple[str, list[str]]` is split — the available-devices list moves to `sysinfo.hardware.get_available_devices()`, the auto-pick logic moves to a free helper `auto_detect_device() -> str` in `session/ep_device.py`, and the bare name `resolve_device` is taken by the new typed resolver `(ep, device="auto"|None) -> EPDevice` which handles `"auto"` internally by calling `auto_detect_device()`.
- **Key claims / decisions made:**
  - `EPDevice` is a pure-data, frozen, JSON-serializable dataclass with `__post_init__` lowercase invariant.
  - `WinMLSession.__init__` becomes `(onnx_path, ep_device, *, ep_config, base_session_options)` — `ep_device` is **required positional**, no default; old `device=`/`ep=`/`session_options=` kwargs are deleted.
  - Monitor integrates via `WinMLSession.perf(monitor=...)`, **not** the constructor.
  - Strict 4-tuple match `(ep, device.type, vendor_id, device_id)` everywhere — including CPU.
  - `_build_session_options` / `_build_provider_options` are private **free functions** in `session.py`, not methods. The descriptor → `OrtEpDevice` bridge is inlined.
  - The QNN backend-type override (`_QNN_BACKEND["npu"] = "htp"`) is built into `_ep_defaults` in §3.4 of the spec — later contradicted by implementation when `add_provider_for_devices` made it unnecessary (and crashy) in ORT 1.23.5 (see `session_session.md` §3 `_ep_defaults`).
- **Open questions / rejected options:** The spec calls out a dependency on `feat/update-pkg-deps` for `canonicalize_ep_name`; this PR ships with a local stub carrying a `MIGRATION:` marker.

#### `docs/plans/2026-05-11-ep-device-refactor-plan.md`

- **Date:** 2026-05-11
- **Doc type:** implementation plan (TDD, 14 tasks)
- **Status:** plan — executed; Tasks 1–10 landed (with one deviation reconciled by `54cb6e81`), Task 11 fixture sweep landed in-flight, Tasks 12–14 carried forward
- **Summary:** Step-by-step TDD plan for the spec above. Lists 4 files to create (`session/ep_device.py`, three test files) and a long list of files to modify. Each task spells out the failing tests, minimal implementation, lint, and a commit message template. Carries a "wait for `feat/update-pkg-deps`" deferral note for the `canonicalize_ep_name` integration.
- **Key claims / decisions made:**
  - 14 tasks ordered from "Exceptions + EPDevice dataclass" (Task 1) through "WinMLSession ctor hard break" (Task 7) through "perf() refactor" (Task 8) through "dissolve `sysinfo/device.py` — split the old `sysinfo.resolve_device` into `sysinfo.hardware.get_available_devices()` plus a free `auto_detect_device()` helper, and reclaim the bare name `resolve_device` for the new typed `(ep, device) -> EPDevice` resolver in `session/ep_device.py`" (Task 9) through "CLI sweep" (Task 10) through fixture sweep / architecture regression / pytest gate / E2E (Tasks 11–14).
  - Task 11 expects the `qnn_npu_ep_device` fixture to land in top-level `tests/conftest.py`. It actually landed in `tests/unit/session/conftest.py` (impl-status §4 documents this lower-scope deviation).
  - Task 14 E2E gate (`uv run wmk perf <convnext> --ep qnn --device npu` end-to-end) is the merge gate.
- **Open questions / rejected options:** None — the plan is normative for the listed tasks.

### Theme 2 — Implementation status + verification of seed

#### `docs/design/session/2026-05-12-impl-status.md`

- **Date:** 2026-05-12
- **Doc type:** implementation audit vs. spec
- **Status:** final for HEAD `54cb6e81` — flagged 3 CRITICAL gaps (all in `models/auto.py` HF path) + 3 spec deviations
- **Summary:** Line-by-line audit of what landed vs. what the spec required, with a spec coverage matrix (every spec § cross-referenced to its commit + line in source) and a per-file implementation surface inventory. Captures the §1.1 "perf() contextmanager deviation" (spec implied regular method; implementation is `@contextmanager`) and the §1.2 (later in doc) lazy circular-import shim (`WinMLEPRegistry: Any = None` + `_get_ep_registry()`). Identifies that three `winml_class(..., device=device)` callsites in `models/auto.py:163–168, 196–203, 355–362` were NOT migrated — these block the `wmk perf <hf-model>` path (the §6 spec demo) and are the root cause of CLI cmd 6's exit 127 (or what was assumed to be a TypeError).
- **Key claims / decisions made:**
  - All spec §§3.1–3.6 + §4 + §5 + §7 are claimed DONE (with the noted deviations).
  - Architecture regression test at `tests/unit/architecture/test_winml_session_ctor.py` is NOT yet present; three equivalent inline assertions exist at `test_winml_session.py:649–662`.
  - `WinMLSession._build_session_options` (legacy instance method) survives at `session.py:462–485` and still uses `set_provider_selection_policy(PREFER_NPU)` — the very autoep path the spec said to delete. Used by `compile()`, `is_compatible()`, and `WinMLQairtSession._create_inference_session()`. Marked with 3 `TODO Task 8 [bridge]` markers.
- **Open questions / rejected options:** Spec v1.3 reconciliation list is open: (1) `perf()` is `@contextmanager`, (2) the `_get_ep_registry` shim, (3) the legacy `_build_session_options` bridge's tech-debt schedule, (4) `WinMLQairtSession` default-EP behaviour.

#### `docs/design/session/2026-05-12-cli-verification.md`

- **Date:** 2026-05-12
- **Doc type:** verification (first CLI run)
- **Status:** final for HEAD `db39b80d` — 3/6 PASS, 3/6 FAIL
- **Summary:** First end-to-end CLI verification of `winml export`, `winml analyze`, `winml optimize`, `winml quantize`, `winml compile`, and `winml perf` on `facebook/convnext-base-224` and `microsoft/resnet-50`. The pipeline works for export → optimize → quantize (cmds 1, 3, 4). `winml analyze` hangs at 30+ min on 667 nodes when the rule zip is missing (cmd 2). `winml compile` exits 1 with no output file because the legacy `WinMLSession._build_session_options` instance method does not wire QNN EPContext options (cmd 5). `winml perf <hf-model>` exits 127 from a subprocess crash 60s in (cmd 6, before benchmark). Also surfaces a `tee` pipe masking issue that initially reported all commands as EXIT 0.
- **Key claims / decisions made:**
  - Cmd 5 root cause: legacy `_build_session_options` instance method missing EPContext keys (matches audit Gap #1.1).
  - Cmd 6 root cause is unconfirmed but likely the `models/auto.py:355–362` `device=device` regression (matches audit Gap #1) — though the actual mode (exit 127, not TypeError) suggested a deeper failure later isolated by the gap1-diagnostic doc.
  - Cmd 2 root cause: `RuntimeCheckerQuery._is_ep_available_locally()` runs per-node ORT probes (667 × ~4s) when the rule zip is absent.
- **Open questions / rejected options:** Lists 5 recommendations including "cease piping through `tee`", ship the rule zip, and dedup probes by op_type.

#### `docs/design/session/2026-05-12-review-summary.md`

- **Date:** 2026-05-12
- **Doc type:** code review summary
- **Status:** final for HEAD `db39b80d` — aggregates per-file reviews under `2026-05-12-code-review/`
- **Summary:** Roll-up of the 24 per-file review docs (groups 1–4) plus a severity-ordered issue table. 3 BLOCKERs (`models/auto.py` HF-path `device=` slippage; cross-package private import `_EP_TO_DEVICE` in `compile.py`; `winml compile` no-output bug), 5 IMPORTANT (including `_build_session_options` mutation safety and the `cuda`/`tensorrt` taxonomy gap), 5 MEDIUM, 1 LOW. Also captures the two pending user decisions: Decision A (move taxonomy to top-level `winml.modelkit.ep_device`?) and Decision B (require `--device` on `winml compile`?).
- **Key claims / decisions made:**
  - 25 changed `*.py` files reviewed in 24 docs (`session_session.md` rolls two sub-modules into one).
  - The taxonomy duplication is real and large: see the companion sweep.
- **Open questions / rejected options:** Decision A (placement) and Decision B (`--device` required on compile) are explicitly user-pending and feed the consolidation plan.

#### `docs/design/session/2026-05-12-ep-taxonomy-sweep.md`

- **Date:** 2026-05-12
- **Doc type:** taxonomy sweep / inventory
- **Status:** final — 47 findings; feeds the consolidation plan
- **Summary:** Catalogs every place in `src/` and `tests/` where EP/device knowledge is hard-coded. 9 unique tables/constants across 5 files; 5 helper functions across 3 files; 9 inline literals; 3 CLI `click.Choice` lists with redundancy; 5 test fixtures; 6 duplicates. Recommends a 10-step consolidation plan (move sources before updating callers, ~90 LOC moved/deleted, net reduction ~50 LOC) and identifies 5 open questions including the `cuda`/`tensorrt` half-presence issue (in `precision._EP_TO_DEVICE` but not in `ep_device._SHORT_TO_CANONICAL`) and the uppercase `SUPPORTED_DEVICES` casing inconsistency.
- **Key claims / decisions made:**
  - The canonical home is `session/ep_device.py` (not yet moved at sweep time); `precision.py` and `utils/constants.py` are the duplicate-emitters.
  - Three inline copies of `{"cpu":"cpu","npu":"qnn","gpu":"dml"}` exist in `commands/perf.py:472`, `commands/perf.py:1552`, and `eval/evaluate.py:138`.
  - The `analyze/runtime_checker/check_ops.py` 5-EP list is acceptable as a subprocess-tool boundary curation.
- **Open questions / rejected options:** Five open questions, including whether `cuda`/`tensorrt` should be valid `--ep` values at all, and whether to add `ov`/`vitis` aliases.

### Theme 3 — Taxonomy consolidation (plan + audit + cleanup sub-trail)

#### `docs/plans/2026-05-13-ep-taxonomy-consolidation-plan.md`

- **Date:** 2026-05-13
- **Doc type:** plan (Phase 1 + Phase 2)
- **Status:** executed (commits `3b155784` Phase 1, `62807ac9` Phase 2) — followed by `e70c2a20` audit follow-up
- **Summary:** Two-phase plan. **Phase 1** moves all taxonomy tables (`_EP_TO_DEVICE`, `_DEVICE_TO_PROVIDER`, `VALID_EPS`, `_VALID_DEVICES`, `get_provider_for_device`) from `config/precision.py` to `session/ep_device.py`; deletes `utils/constants.py`'s `SUPPORTED_EPS` / `EP_ALIASES` / `ALL_EP_NAMES` / `SUPPORTED_DEVICES`; widens `resolve_device(ep, device)` to accept partials (4-case deduction). **Phase 2** adds `--device` to `winml compile`, threads `EPDevice` through `WinMLCompileConfig`, adds an architecture regression test that walks the AST to ban `from ...session.ep_device import ...` outside `session/`.
- **Key claims / decisions made:**
  - Decision A: implementation lives at `src/winml/modelkit/session/ep_device.py`; tests import via `from winml.modelkit.session import ...`; source imports via `from ..session import ...`.
  - Decision B: `resolve_device(ep, device)` accepts partials and deduces missing values once at the boundary.
  - Decision C: both `--ep` and `--device` are optional on `winml compile`; the handler calls `resolve_device(ep, device)` and stores the result in `WinMLCompileConfig.ep_device`.
- **Open questions / rejected options:** Out of scope: audit Gap #1 (`models/auto.py` device= slippage), audit Gap #3 (legacy `_build_session_options`), native QNN HTP AOT crashes, analyze slow probing.

#### `docs/design/session/2026-05-13-consolidation-audit.md`

- **Date:** 2026-05-13
- **Doc type:** post-execution audit of the consolidation plan
- **Status:** verdict PASS-WITH-CONCERNS — 29 requirements verified, 3 plan deviations (all in `commands/compile.py`), 3 surprises beyond plan, 0 regressions
- **Summary:** Per-§ audit of the consolidation plan against actual source. PASSes for taxonomy table relocation, session/__init__.py re-exports, `resolve_device` 4-case deduction, smoke tests, architecture regression test (15 passes). Three FAILs in `commands/compile.py`: (a) the CLI calls `_resolve_compile_provider` returning a string rather than `resolve_device` returning an `EPDevice` — violating Decision B's "deduce at boundary"; (b) `utils/constants.py` taxonomy entries (`SUPPORTED_EPS`, `EP_ALIASES`, `ALL_EP_NAMES`, `SUPPORTED_DEVICES`) NOT deleted despite plan Step 5. Confirmed: full unit test suite 480 passed / 5 skipped / 0 failed; `ruff check src/ tests/` clean.
- **Key claims / decisions made:**
  - `_EP_TO_DEVICE` is gone from `config/precision.py`; `_DEVICE_TO_PROVIDER` is gone.
  - `WinMLCompileConfig.ep_device` field added with `to_dict` / `from_dict` round-trip; `for_ep_device()` factory added.
  - Architecture regression test (`test_ep_device_import_rule.py`) is effective — adversarial verified.
  - Three plan deviations and the dead `_compile_provider` shadow dict survive in `ep_device.py` (later cleanup driver).
- **Open questions / rejected options:** P1 (medium): complete CLI boundary for `winml compile` to call `resolve_device`. P2-P4 (low): cleanup tasks.

#### `docs/design/session/2026-05-13-ep-device-spec-design.md`

- **Date:** 2026-05-13
- **Doc type:** design (catalog redesign)
- **Status:** design finalized; implementation landed in commit `680b232c`
- **Summary:** Introduces `EPDeviceSpec` — a frozen dataclass catalog entry — as the **single source of truth** for (EP, device) variants, replacing `_EP_TO_DEVICE` / `_DEVICE_TO_PROVIDER` / the `_ep_defaults` if/elif ladder. 13-entry tuple `EP_DEVICE_SPECS` with O(1) lookup via `lookup_device_spec`, and derived helpers `default_device_for_ep` / `default_ep_for_device`. Driven by the Phase 1 experiment showing +3× throughput from `htp_performance_mode='burst'` on QNN-NPU — those defaults now live as data in the QNN-NPU catalog entry.
- **Key claims / decisions made:**
  - **`EPDeviceSpec` (template) → `EPDevice` (instance)** — same shape as the Kubernetes `PodSpec`/`Pod` distinction.
  - Pattern A (dataclass tuple) chosen over enums, class hierarchies, or module constants; adding a new property is one dataclass line, adding a new variant is one tuple line.
  - 13 catalog entries enumerated, only QNN-NPU has non-empty `default_provider_options` (verified +3× burst-mode); all 12 others carry TODO comments for hardware measurement.
  - **Renames:** `_SHORT_TO_CANONICAL` → `_SHORT_TO_FULL` (commit `0a9d422a`); `_VALID_DEVICES` → `VALID_DEVICES` (later v1 plan).
- **Open questions / rejected options:** "Adding speculative defaults for unverified variants" is OUT OF SCOPE; per-variant session-config defaults are deferred.

#### `docs/design/session/2026-05-13-post-refactor-taxonomy-audit.md`

- **Date:** 2026-05-13
- **Doc type:** audit (post-EPDeviceSpec refactor)
- **Status:** verdict PASS-WITH-EXCEPTIONS — surfaces 6 items that feed the v1/v2/v3 cleanup
- **Summary:** Re-runs the 30+ taxonomy patterns from the 2026-05-12 sweep against HEAD `680b232c` (post-EPDeviceSpec). Confirms `_EP_TO_DEVICE` / `_DEVICE_TO_PROVIDER` are 0 production hits. Surfaces 6 findings: (R1 MEDIUM) `get_provider_for_device` embeds a standalone `_compile_provider` dict not derived from `EP_DEVICE_SPECS`; (R2 LOW) `sysinfo/device.py:61` duplicates `_VALID_DEVICES`; (R3 LOW) `commands/build.py:369–373` hardcoded NPU-capable list; (R4 LOW) `compiler/cli.py:53` stale `["qnn","cpu","cuda","dml"]`; (R5 LOW) 4 command-files' device choice lists not derived from `_VALID_DEVICES`; (R6 INFO) design doc §3 says "cpu-first: QNN" but catalog has OpenVINO/cpu before QNN/cpu.
- **Key claims / decisions made:**
  - `_EP_TO_DEVICE` and `_DEVICE_TO_PROVIDER` are FULLY removed from production code (only survive as docstring text in `ep_device.py`).
  - `EP_DEVICE_SPECS` has 13 entries; QNN-NPU burst defaults are present; `_BY_KEY`, `VALID_EPS`, `_VALID_DEVICES` are all derived from the catalog.
- **Open questions / rejected options:** Whether to add `is_compile_target: bool` or `is_npu_capable: bool` fields to `EPDeviceSpec` — deferred per design doc §11.

#### `docs/design/session/2026-05-13-final-taxonomy-cleanup-plan.md` *(v1 — SUPERSEDED by v2 → v3)*

- **Date:** 2026-05-13
- **Doc type:** plan (final taxonomy cleanup, v1)
- **Status:** **SUPERSEDED** — see "Successor" section at file bottom. Executed in commits `6ce5aa3d`, `720a4ed4`, `eee42e7f`, `8fc6e30b`. The follow-up `2026-05-13-final-taxonomy-cleanup-plan-v2.md` found 1 BLOCKER + 4 IMPORTANT + 7 NICE-TO-HAVE items this v1 didn't cover.
- **Summary:** Closes the 4 remaining audit items from the post-refactor taxonomy audit. 7 decisions: delete `get_provider_for_device`; reorder `EP_DEVICE_SPECS` so DML precedes OpenVINO for GPU and `CPUExecutionProvider` precedes others for CPU; add `eps_for_device(device)` helper; rename `_VALID_DEVICES` → `VALID_DEVICES` (public); replace `sysinfo/device.py:61` duplicate with session-facade import; fix `compiler/cli.py:53` stale `click.Choice` with `sorted(VALID_EPS)`; single atomic commit.
- **Key claims / decisions made:**
  - `default_ep_for_device("gpu")` changes from `OpenVINOExecutionProvider` → `DmlExecutionProvider` (semantic shift, but tests still pass since `compile_provider == "dml"` is preserved by the reorder).
  - `default_ep_for_device("cpu")` changes from `OpenVINOExecutionProvider` → `CPUExecutionProvider`.
- **Open questions / rejected options:** 3 user questions: catalog reorder confirmation, `_VALID_DEVICES` → `VALID_DEVICES` rename, `compiler/cli.py` EP choice expansion to all 9 short names.

#### `docs/design/session/2026-05-13-final-taxonomy-cleanup-plan-v2.md` *(v2 — SUPERSEDED by v3)*

- **Date:** 2026-05-14
- **Doc type:** plan (v2 — re-audit + fix list)
- **Status:** **SUPERSEDED** — see v3. Self-contradicts on item I4 (resolved by its own §2 Finding A-5); fact-checked separately by `-v2-factcheck.md`. Executed in commit `1ab32a76`.
- **Summary:** Comprehensive re-audit after v1 landed. Finds **1 BLOCKER** (B1: 5 occurrences of `NvTensorRTRTXExecutionProvider` wrong casing in `check_ops.py` lines 264/267/289/335/343 plus `winml.py:149` docstring — the catalog uses `NvTensorRtRtxExecutionProvider` and the wrong casing makes `lookup_device_spec` return `None`), **4 IMPORTANT** (I1: `analyze/analyzer.py` hardcoded 3-EP list; I2: `analyze/pattern/check_patterns.py` argparse boundary; I3: `utils/optimum_loader.py` CUDA carve-out; I4: false catalog-gap claim — DROPPED), **7 NICE-TO-HAVE** (including `QNN_VENDOR_ID` deduplication, deleted-name sentinels in architecture test, docstring fixes).
- **Key claims / decisions made:**
  - `lookup_device_spec("NvTensorRTRTXExecutionProvider", "gpu")` returns `None` at runtime — confirmed (latent bug).
  - I4 false-claim resolution: catalog has 13 entries including `CPUExecutionProvider/cpu` at index 2.
- **Open questions / rejected options:** v2 self-contradicts on I4 — flagged in the fact-check.

#### `docs/design/session/2026-05-13-final-taxonomy-cleanup-plan-v2-factcheck.md`

- **Date:** 2026-05-14
- **Doc type:** fact-check of v2 plan
- **Status:** final — corrects 4 errors in v2 (drop I4 entirely; widen B1 line list to include line 264; correct N4 scope from "5×" to "24+ occurrences"; fix status-snapshot count line)
- **Summary:** Verifies every claim in v2 §1 against actual source at HEAD `8fc6e30b`. 10 VERIFIED, 0 FALSE, 3 PARTIAL, 1 CONTRADICTING (I4 — v2 §1 says 12 entries, §2 self-corrects to 13). Ground-truth lookup table confirms 13 catalog entries; `lookup_device_spec("NvTensorRTRTXExecutionProvider", "gpu")` returns `None`; wrong-casing occurrences span 5 lines in `check_ops.py` (264, 267, 289, 335, 343) — v2's original "lines 267, 289, 335, 343" was missing line 264.
- **Key claims / decisions made:**
  - Forces v2 plan to drop I4, widen B1, correct N4 scope, and fix the status-snapshot count line before execution.
- **Open questions / rejected options:** None — output is corrections feeding v2/v3.

#### `docs/design/session/2026-05-13-final-taxonomy-cleanup-plan-v3.md` *(LATEST taxonomy state)*

- **Date:** 2026-05-14
- **Doc type:** plan (v3 — post-cleanup state)
- **Status:** final — captures the **post-cleanup state** at HEAD `39d95d73`. No v4 expected.
- **Summary:** Resolves every decision item from v1/v2 (D1–D7 done, B1 done after widening, I1–I3 done, I4 dropped as false, N1–N7 done) and adds V3-1 (wrap `resolve_device` in `compile.py` with explicit exception handling to produce clean `click.UsageError` instead of raw traceback — landed in `39d95d73`). Reports CLI verification matrix (QNN cpu/gpu/npu × perf/compile, 6 commands): 4 PASS, 1 HARDWARE N/A (QNN has no CPU backend on Snapdragon X-Elite), 1 BUG FIXED (compile DeviceNotFound → clean error after `39d95d73`).
- **Key claims / decisions made:**
  - `EP_DEVICE_SPECS` is 13 entries (verified by fact-check).
  - All removed names (`_EP_TO_DEVICE`, `_DEVICE_TO_PROVIDER`, `_EP_DEVICE_MAP`, `_DEVICE_EP_MAP`, `get_provider_for_device`, `_compile_provider`, `_VALID_DEVICES`, `SUPPORTED_EPS`, `EP_ALIASES`, `ALL_EP_NAMES`, `SUPPORTED_DEVICES`, `NvTensorRTRTXExecutionProvider` wrong casing) have **0 production hits** in `.py` files — only docstring/comment survivors.
  - `commands/compile.py` `--device cpu` (without QNN) produces EXIT 0 with `CPUExecutionProvider`; `--ep qnn --device cpu` produces clean DeviceNotFound at EXIT 1 (post-fix).
- **Open questions / rejected options:** Architecture test Gap 1 (literal-mapping detection beyond AST imports) and Gap 3 (`session.session` direct imports from outside `session/`) are deferred to follow-up PRs.

#### `docs/design/session/2026-05-13-final-taxonomy-cleanup-plan-v3-verify.md`

- **Date:** 2026-05-13 (file metadata — v3 verification ran in this window)
- **Doc type:** verification (QNN cpu/gpu/npu × perf/compile matrix)
- **Status:** final — 4/6 PASS, 1/6 HARDWARE N/A, 1 BUG FIXED
- **Summary:** 6-command CLI matrix exercising every (perf|compile) × (cpu|gpu|npu) combo against ResNet-50 on this Snapdragon X-Elite. Confirms `winml perf --ep qnn --device npu` at 1.99 ms avg / 501 samples/s (matches +3× burst-mode expectation); `winml compile --ep qnn --device cpu` initially gave raw traceback (EXIT 1) — fixed in commit `39d95d73` to give clean `click.UsageError` ("No OrtEpDevice for QNNExecutionProvider matches device='cpu'. Available: [(NPU, 0x4d4f4351, 0x41304430), (GPU, 0x4d4f4351, 0x36334330)]") at EXIT 1.
- **Key claims / decisions made:**
  - Snapdragon X-Elite QNN exposes only NPU and GPU backends; no CPU backend — the `--ep qnn --device cpu` failure is hardware-correct.
  - The fix in `39d95d73` brings `compile.py` exception-handling to parity with `perf.py`'s broader `except Exception` catch.
- **Open questions / rejected options:** None — verification confirms v3 state.

### Theme 4 — WinMLSession refactor (per-file code reviews)

The 24 per-file code review docs under `docs/design/session/2026-05-12-code-review/` are
structured as one doc per touched `*.py` file. Each follows the same shape: §1 Purpose,
§2 Changes summary, §3 Per-symbol review, §4 Cross-cutting concerns, §5 Confidence,
§6 Verbatim risk inventory. All are status="final" first-pass reviews of HEAD `db39b80d`.

**Group 1 — EPDevice core (6 docs):**

- **`session_ep_device.md`** — review of the new file. High confidence. Notes the unusual `WinMLEPRegistry: Any = None` module-level binding as the most unusual design element. Two MINOR risks: redundant `getattr(..., "") or ""` for vendor lookup; the module-level public-name shape is easy to confuse with a class import.
- **`session_ep_registry.md`** — adds `register_ep(name) -> list[OrtEpDevice]` (additive); `register_to_ort` unchanged. New `_registration_failures` ledger populated by `register_to_ort` only — asymmetric vs. `register_ep` which raises typed exceptions instead.
- **`session_session.md`** — the largest review (290+ / 170- lines changed). Medium-high confidence. Flags the **IMPORTANT** `base_session_options` mutation-in-place risk in `_build_session_options` (depends on ORT's `add_session_config_entry` idempotency, unverified) and the **IMPORTANT** surviving legacy `WinMLSession._build_session_options` instance method that still uses `set_provider_selection_policy(PREFER_NPU)`. Also captures the spec-drift items reconciled in v1.3.
- **`session_qairt_qairt_session.md`** — ctor signature changed `device: str` → `ep_device: EPDevice | None = None` (default `resolve_device("qnn","npu")`). `_create_inference_session` still calls the legacy `self._build_session_options(self._device)` — tech debt tracked by `TODO Task 8 [bridge]`.
- **`sysinfo_device.md`** — reviews the deletion of `sysinfo/device.py`. The old `resolve_device(device="auto")` tuple return is split across new homes: `sysinfo.hardware.get_available_devices()` for the list, `session.auto_detect_device()` for the auto-pick string, and the new typed `session.resolve_device(ep, device) -> EPDevice` takes the bare name. High confidence.
- **`sysinfo___init__.md`** — re-export rename. High confidence.

**Group 2 — Op-tracing monitor (8 docs):**

- **`session_monitor___init__.md`** — empty package marker (correct).
- **`session_monitor_ep_monitor.md`** — `EPMonitor` ABC extended with `requires_session_teardown`, `ep_name` ClassVars; `get_session_options`, `get_provider_options`, `set_onnx_op_types`, `result` concrete defaults; `to_dict()` removed (v2.4 contract change).
- **`session_monitor_op_metrics.md`** — relocated from `optracing/result.py`; adds `TraceStatus` literal, `OperatorMetrics.samples_us`, `OpTraceResult.status`/`error`.
- **`session_monitor_qnn_monitor.md`** — placeholder → full implementation. `requires_session_teardown=True`, `ep_name="qnn"`. Owner-enforced `profiling_level` / `profiling_file_path`.
- **`session_monitor_report.md`** — relocated from `optracing/report.py`. No functional changes.
- **`session_monitor_qnn___init__.md`** — public surface re-exports `parse_qhas` and `parse_qnn_profiling_csv` from `._internal`.
- **`session_monitor_qnn__internal.md`** — consolidates the deleted `csv_parser.py` + `qhas_parser.py` into one private module.
- **`session_monitor_qnn_viewer.md`** — relocated from `optracing/qnn/viewer.py`. No functional changes.

**Group 3 — Commands + eval (6 docs):**

- **`commands_perf.md`** — the largest commands review (342 lines). New `_resolve_ep_monitor` dispatch; `BenchmarkConfig.op_tracing` field; rewires the op-tracing report section. Boundary `ep_device` resolution at `commands/perf.py:1540–1548`.
- **`commands_live_chart.md`** — cosmetic constant bumps (`_CHART_WINDOW_SECONDS` 10→15, `chart_width` 80→120).
- **`commands_pre_bench.md`** — new file extracted from `perf.py`. Renders the pre-benchmark identity panel.
- **`commands_config.md`** — one-line import rename.
- **`commands_eval.md`** — two-line import rename.
- **`eval_evaluate.md`** — moves EPDevice construction from inside `WinMLAutoModel` to the `_load_model` boundary (the "CLI boundary resolve" pattern).

**Group 4 — Models + config + compiler (5 docs):**

- **`models_auto.md`** — `from_pretrained` and `from_onnx` signatures change: `device: str` + `ep: str | None` → `ep_device: EPDevice`.
- **`models_winml_base.md`** — `WinMLPreTrainedModel.__init__`: `device: str` → `ep_device: EPDevice` (required, no default).
- **`config_build.md`** — migrates 2 call sites off the deleted `sysinfo.resolve_device`. The new shape is `available_devices = get_available_devices()` plus `resolved_device = auto_detect_device() if device == "auto" else device`.
- **`config_precision.md`** — one-line docstring update. Notes `_EP_TO_DEVICE` and `VALID_EPS` are now duplicated with `ep_device.py` (resolved by the consolidation plan).
- **`compiler_compile.md`** — `CompileStage.process()` migrates session-creation from `device=` to `ep_device=`; uses `_EP_TO_DEVICE` from `precision.py` for the reverse map (later refactored to consume `EPDevice` from `WinMLCompileConfig`).

### Theme 5 — Op-tracing monitor pipeline (PRD + coreloop + 11 iteration docs)

#### `docs/design/session/monitor/1_prd.md`

- **Date:** 2026-05-08
- **Doc type:** product requirements document (PRD)
- **Status:** Draft v2.4.1
- **Summary:** Defines the requirements for replacing `QNNProfiler` / `OpTracer` with an extended `EPMonitor` hierarchy. Motivated by two defects: (D-1) `QNNProfiler` is silently broken with `onnxruntime-windowsml` because the explicit-providers API can't find the QNN DLL under `WindowsApps`; (D-2) `WinMLEPMonitor` and `OpTracer` duplicate ORT session-creation logic. v2.4 simplifies v2.3's separate `OpTraceParser` ABC down to a single ABC + concrete-default methods. v2.4 also drops `to_dict()` from the `EPMonitor` contract — replaced by typed accessors (`result -> OpTraceResult` for op-tracing, `proof -> ProofOfExecution` deferred to a follow-up PR for proof-of-execution monitors).
- **Key claims / decisions made:** 5 functional requirement areas (FR-1 through FR-20+); 5 non-functional requirements; the four-layer ONNX op-type fallback chain.
- **Open questions / rejected options:** Open Questions referenced in §9 flow back from `2026-05-03-op-trace-parser-interface-spec.md` v2.0 (not in `session/` — out of scope for this index).

#### `docs/design/session/monitor/2_coreloop.md`

- **Date:** 2026-05-08
- **Doc type:** core-loop design (the normative implementation guide for the PRD)
- **Status:** Draft v2.4.1
- **Summary:** Normative copy of class signatures and worked walkthroughs from `2026-05-03-op-trace-parser-interface-spec.md`. Defines the revised `EPMonitor` ABC, `NullEPMonitor`, `QNNMonitor`, `PerfContext`, and the revised `WinMLSession.perf()`. Specifies the factory helper in `commands/perf.py` (`_resolve_ep_monitor`).
- **Key claims / decisions made:** Detailed class signatures for the 7 classes/functions; integration points; testing strategy.
- **Open questions / rejected options:** Open questions in §10.1 flow back to the spec — same provenance as the PRD.

#### `docs/design/session/monitor/iterations/01.md` through `11.md`

- **Date:** ranges 2026-05-04 through 2026-05-08
- **Doc type:** 11 brainstorming-iteration docs (preserved working trail)
- **Status:** historical / informational — superseded by `1_prd.md` v2.4.1 and `2_coreloop.md` v2.4.1
- **Summary:** Eleven iterative working sessions that arrive at the v2.4 design. Each captures the entry-point question, the lead author's position, the critic's response, and the resolution. Topics (titles): 01 Problem statement; 02 EPMonitor hierarchy and unification question; 03 Is OpTracer a kind of op-level monitor; 04 Session integration — put monitor under the session; 05 Merge `session.monitor()` into `session.perf()`; 06 QNNMonitor key points (27 decisions); 07 Why on the base class — and what design pattern; 08 Who creates the ORT session; 09 Auto-reset and the six responsibilities; 10 Critic agent review of the six responsibilities; 11 **Delete QNNProfiler entirely** (no replacement — primitives compose).
- **Key claims / decisions made:** Iteration 11's decision to delete `QNNProfiler` entirely (rather than keep a thin `.run()` helper) is the final architectural commitment that shapes `1_prd.md` v2.4 and `2_coreloop.md` v2.4. Step 5 of the old profiler ("generate dummy inputs") is extracted to a generic utility; everything else moves to `WinMLSession + QNNMonitor`.
- **Open questions / rejected options:** Iteration 06's 27 decisions catalogue; not all are normative — many are explicitly carved out as out-of-scope. Read iterations sequentially if reconstructing the rationale, otherwise jump to `1_prd.md` / `2_coreloop.md` for the consolidated authoritative view.

### Theme 6 — CLI verification

#### `docs/design/session/2026-05-12-cli-verification.md`
*(cataloged above in Theme 2.)*

#### `docs/design/session/2026-05-13-cli-claims-reverify.md`

- **Date:** 2026-05-13
- **Doc type:** verification (re-run after Gap #1 partial fix)
- **Status:** final at HEAD `eb37f6c3` — 5/6 PASS, 1/6 FAIL (T6 still broken)
- **Summary:** Re-runs 6 CLI scenarios after the `eb37f6c3` defensive guard landed in `WinMLEPRegistry.register_ep`. T1 (perf qnn+npu explicit) through T5 (perf ctx onnx) all PASS at EXIT 0 with latencies 2.27–2.63 ms. **T6 (perf HF microsoft/resnet-50) still FAILS** with `0xC000026F` (`STATUS_DLL_NOT_FOUND`) — crashes at the analyze stage at 0/122 ops. The diagnostic that explains T6 is `2026-05-13-t6-analyze-crash-diagnostic.md` (next entry).
- **Key claims / decisions made:**
  - T4 compile produces `*_qnn_ctx.onnx` (931 B) + `*_qnn.bin` (49.4 MB) — both expected artifacts.
  - T5 perf-on-ctx confirms EPContext cache hit (no compile overhead; 2.27 ms identical to T2).
  - T6 exit code `0xC000026F` is a Windows native NTSTATUS, not a Python exception with traceback.
- **Open questions / rejected options:** T6 root cause is open at this verification; resolved in `t6-analyze-crash-diagnostic.md`.

### Theme 7 — BLOCKER diagnostics (dual-singleton DLL double-register)

#### `docs/design/session/2026-05-13-gap1-diagnostic.md`

- **Date:** 2026-05-13
- **Doc type:** root-cause diagnostic
- **Status:** ROOT CAUSE FOUND — exit 127 is a native ORT crash from double-registration of the QNN EP DLL
- **Summary:** Diagnoses the original exit-127 mode for `winml perf -m microsoft/resnet-50 --ep qnn --device npu`. Shows that `ort.register_execution_provider_library('QNNExecutionProvider', dll)` called twice in the same parent process causes a native `exit(127)` with no Python traceback. Two uncoordinated singleton systems both register: `WinMLEPRegistry` (in `session/ep_registry.py`, used by `_build_session_options` for compile) and `winml.py:WinML` (used by `_is_ep_available_locally` in the analyze loop). Each tracks registration internally but is invisible to the other.
- **Key claims / decisions made:**
  - Test 5 reproduces the crash with two `register_execution_provider_library` calls in sequence (one Python file).
  - Analyze-loop children survive their own crashes (122 nodes × ~0.4 s) because `ResilientRunner` catches `BrokenProcessPool`; the parent crashes only on the SECOND parent-process registration in the compile stage.
  - **Option A (minimal):** Add `ort.get_ep_devices()` pre-check to `WinMLEPRegistry.register_ep`. **Option B (preferred):** Remove `winml.register_execution_providers()` from `_is_ep_available_locally` entirely. The team chose Option A (`eb37f6c3`).
- **Open questions / rejected options:** Whether the long-term consolidation (single-singleton model) is in scope for this PR is left open (later answered "no, defer" per landing page §I1).

#### `docs/design/session/2026-05-13-t6-analyze-crash-diagnostic.md`

- **Date:** 2026-05-13
- **Doc type:** follow-up diagnostic
- **Status:** Root cause confirmed. Prior `eb37f6c3` fix protected the WRONG caller — crash persists at HEAD `eb37f6c3`.
- **Summary:** Diagnoses why the Gap #1 fix in `eb37f6c3` did NOT close the T6 crash. The fix correctly guarded `WinMLEPRegistry.register_ep` against `winml.py:WinML` running first. But in the `winml perf <hf-model>` path, the call order is reversed: `WinMLEPRegistry.register_ep` runs FIRST (during `_load_model` → `resolve_device`), then `winml.py:WinML.register_execution_providers()` runs SECOND (during the analyze loop's first node). The guard is on the wrong side. Confirms via Test 5 that ORT 1.23.2's `ort.register_execution_provider_library` is not idempotent: a second call for the same DLL calls C++ `exit(127)` with no traceback. Windows maps exit code 127 to `0xC000026F` (`STATUS_DLL_NOT_FOUND`) — misleading, the DLL is present and was successfully loaded by the first registration.
- **Key claims / decisions made:**
  - Fix: add the same `ort.get_ep_devices()` pre-check to `winml.py:WinML.register_execution_providers()` — symmetric defensive guards on both singletons. Landed in `ec777caa` (and supported by `eb37f6c3`).
  - Direct-ONNX perf path (T1/T2/T3) doesn't exercise this code path: `_run_onnx_benchmark` doesn't call `resolve_device` early; only `winml.py:WinML` registers QNN; no double registration.
- **Open questions / rejected options:** Preferred long-term fix (remove the dual-singleton entirely) is again deferred.

### Theme 8 — Post-BLOCKER cleanup plan

#### `docs/plans/2026-05-13-post-blocker-cleanup-plan.md`

- **Date:** 2026-05-13
- **Doc type:** cleanup plan (bundles A/B/C/D, post-BLOCKER fixes)
- **Status:** plan — Bundles A + B + C1 executed; C2 + D outstanding per landing page
- **Summary:** Bundles the IMPORTANT/MEDIUM punch list into three parallel agent assignments. **Bundle A** (monitor pipeline silent failures): `int()` truncation in `qnn_monitor.py:439–441`; hard `dict[key]` in `qnn/_internal.py`; JSON-written-before-status-check in `commands/perf.py`. **Bundle B** (`_build_session_options` mutation safety): verify ORT `add_session_config_entry` idempotency; copy-on-use or track-and-clear. **Bundle C** (taxonomy gaps): C1 add `cuda`/`tensorrt` to `_SHORT_TO_FULL` (now in catalog); C2 consolidate the two EP-registration singletons. **Bundle D** (analyze command): rule-zip integration or per-op-type dedup. Agents X/Y parallel, agent Z open-ended.
- **Key claims / decisions made:**
  - Bundles' rollback is per-bundle (`git revert <sha>`).
  - Out of scope: native QNN HTP AOT crashes; `feat/update-pkg-deps` territory; removing the legacy `WinML.register_execution_providers()` singleton entirely.
- **Open questions / rejected options:** Bundle D investigation-only; commit anything that materializes.

### Theme 9 — QuantSpec (DRAFT — do NOT implement)

#### `docs/design/session/2026-05-14-quant-spec-design.md`

- **Date:** 2026-05-14
- **Doc type:** design (extension to EPDeviceSpec)
- **Status:** **DRAFT — design captured, decision pending.** Predecessor: `2026-05-13-ep-device-spec-design.md`. The doc itself opens with "DRAFT" header and §11 lists three decision checkboxes (Direction approved? Landing option? Open questions answered?) that are unchecked.
- **Summary:** Proposes a `QuantSpec` frozen dataclass attached to `EPDeviceSpec.default_quant` (and `EPDevice.quant`) capturing per-variant quantization specification (precision + weight type + activation type + symmetric flag + per_channel_weights). Driven by two gaps: (1) verification gap — the EPDeviceSpec PR's CLI matrix exercised FP32 ResNet-50 on QNN-NPU, not the canonical QDQ-direct path (which is 2.75× faster, 1368 samp/s vs. 498 samp/s, verified in Appendix A on 2026-05-15); (2) design gap — `config/precision.py:62–69`'s `_BITS_TO_WEIGHT_TYPE` / `_BITS_TO_ACTIVATION_TYPE` global tables have an explicit TODO for "EP-specific override layer" that this proposal would close. Three landing options: (a) DEFER (capture this doc + ship — author lean), (b) MINIMAL (types only, no consumer migration, ~50 LOC), (c) FULL (types + migration + tests, ~300–500 LOC).
- **Key claims / decisions made:**
  - QuantSpec fields: `precision`, `weight_type`, `activation_type`, `symmetric`, `per_channel_weights`. `__post_init__` derives weight/activation types from precision via fallback tables.
  - QNN-HTP wants `uint8/uint16/asymmetric/per_tensor`; VitisAI-NPU wants `int8/int8/symmetric/per_channel`; DML-GPU wants `fp16` (informational); CPU has no opinion (`default_quant=None`).
  - **Appendix A measurement: QDQ → NPU at 0.73 ms avg / 1368 samples/s vs. FP32 → NPU at 2.01 ms / 498 samples/s — 2.75× speedup confirms the canonical QDQ path works end-to-end.**
- **Open questions / rejected options:** §10 lists 5 open questions that must be answered before promoting from DRAFT: field set finalization, `None` semantics, override-vs-merge for user-supplied `quant=`, validation policy (warn vs. error), naming (`QuantSpec` vs. `QuantizationSpec` vs. `QuantScheme`). **The earlier proposal "add `default_precision: Literal[...]` to `EPDeviceSpec`" was REJECTED** (insufficient structural variance — every NPU variant wants int8).

---

## Superseded-document map (what replaces what)

| Original doc | Successor | Reason |
|---|---|---|
| `2026-05-11-ep-device-refactor.md` v1.2 | (in-flight) v1.3 — not yet written | Spec drift: `perf()` is `@contextmanager`, the `_get_ep_registry` shim, the legacy `_build_session_options` instance bridge tech debt, `WinMLQairtSession` default-EP behaviour. Tracked by landing page §I2. |
| `2026-05-13-final-taxonomy-cleanup-plan.md` v1 | `…-v2.md` (then `…-v3.md`) | v1 missed 1 BLOCKER + 4 IMPORTANT + 7 NICE-TO-HAVE items |
| `2026-05-13-final-taxonomy-cleanup-plan-v2.md` | `…-v3.md` | v2 self-contradicts on I4 + fact-check flagged 4 errors; v3 is the post-cleanup state |
| Monitor iterations 01–11 in `monitor/iterations/` | `monitor/1_prd.md` v2.4.1 + `monitor/2_coreloop.md` v2.4.1 | Iteration trail is preserved historically; PRD + coreloop are the consolidated normative docs |
| Decisions in earlier iterations (e.g., iteration 10's "keep a thin `QNNProfiler.run()` helper") | Iteration 11's "delete `QNNProfiler` entirely" | Final architectural commitment per user pushback |
| `_BITS_TO_WEIGHT_TYPE` / `_BITS_TO_ACTIVATION_TYPE` global tables in `config/precision.py:62–69` (with TODO) | `2026-05-14-quant-spec-design.md` (DRAFT — pending decision) | Design closure for the explicit TODO; not yet executed |

---

## DRAFT / do-not-implement markers (per commit body)

- **`2026-05-14-quant-spec-design.md`** is explicitly marked **DRAFT — design captured, decision pending**. The commit body's mention that this PR includes a DRAFT QuantSpec doc aligns with this file. **§11 of the doc lists three unchecked decision boxes** (direction approved? landing option? open questions answered?). Do NOT begin implementation without the user clearing those three boxes; the author's own lean (§9) is option (a) DEFER.
- All 11 documents under `monitor/iterations/` are labeled "iteration" working notes — superseded by `1_prd.md` and `2_coreloop.md` v2.4.1. They are not "do-not-implement" so much as "do-not-treat-as-normative" (the v2.4 PRD + coreloop are normative).
- `2026-05-11-ep-device-refactor.md` carries an embedded **`MIGRATION:`** marker for the `canonicalize_ep_name` stub awaiting `feat/update-pkg-deps`. The stub is a one-line replacement target, not a DRAFT — but the doc is at v1.2 and needs v1.3 to reconcile with shipped behaviour.

---

## Quick-reference summary table

| Doc | Theme | Status |
|---|---|---|
| `2026-05-11-ep-device-refactor.md` | EPDevice design | v1.2 — needs v1.3 |
| `2026-05-11-ep-device-refactor-plan.md` (`docs/plans/`) | Plan | Executed (Tasks 1–11) |
| `2026-05-12-impl-status.md` | Audit | Final — 3 CRITICAL gaps + 3 deviations |
| `2026-05-12-cli-verification.md` | Verification | Final — 3/6 PASS |
| `2026-05-12-review-summary.md` | Code review summary | Final — 3 BLOCKER + 5 IMPORTANT |
| `2026-05-12-ep-taxonomy-sweep.md` | Taxonomy inventory | Final — 47 findings |
| `2026-05-12-code-review/` (24 docs) | Per-file code reviews | Final first-pass |
| `2026-05-13-ep-taxonomy-consolidation-plan.md` (`docs/plans/`) | Plan | Executed |
| `2026-05-13-consolidation-audit.md` | Audit | PASS-WITH-CONCERNS (3 deviations) |
| `2026-05-13-ep-device-spec-design.md` | Catalog redesign | Design final; landed `680b232c` |
| `2026-05-13-post-refactor-taxonomy-audit.md` | Audit | PASS-WITH-EXCEPTIONS (6 items) |
| `2026-05-13-final-taxonomy-cleanup-plan.md` | Plan (v1) | **SUPERSEDED** by v2/v3 |
| `2026-05-13-final-taxonomy-cleanup-plan-v2.md` | Plan (v2) | **SUPERSEDED** by v3 |
| `2026-05-13-final-taxonomy-cleanup-plan-v2-factcheck.md` | Fact-check of v2 | Final — 4 corrections |
| `2026-05-13-final-taxonomy-cleanup-plan-v3.md` | Plan (v3) | Final — latest taxonomy state |
| `2026-05-13-final-taxonomy-cleanup-plan-v3-verify.md` | Verification | Final — 4/6 PASS, 1/6 H/W N/A, 1 fixed |
| `2026-05-13-cli-claims-reverify.md` | Verification | Final — 5/6 PASS |
| `2026-05-13-gap1-diagnostic.md` | Diagnostic | Root cause found |
| `2026-05-13-t6-analyze-crash-diagnostic.md` | Diagnostic | Root cause confirmed (fix protected wrong caller) |
| `2026-05-13-post-blocker-cleanup-plan.md` (`docs/plans/`) | Plan | Bundles A + B + C1 executed; C2 + D outstanding |
| `2026-05-13-remaining-issues.md` | **LANDING PAGE** | Final post-cleanup snapshot |
| `2026-05-14-quant-spec-design.md` | QuantSpec follow-on | **DRAFT — do not implement** |
| `monitor/1_prd.md` | Op-tracing monitor PRD | Draft v2.4.1 (normative) |
| `monitor/2_coreloop.md` | Op-tracing monitor coreloop | Draft v2.4.1 (normative) |
| `monitor/iterations/01.md`–`11.md` | Iteration working notes | Historical — superseded by PRD + coreloop |
