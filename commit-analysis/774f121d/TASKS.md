# Action Plan — 774f121d Squash Cleanup

*Based on the verified findings in `FINAL-VERDICTS.md`, `DEEP-DIVE.md`, and the 5 verification batch reports under `verification/`. Every task references the proof location so a contributor can re-verify before acting.*

**Branch:** `feat/op-tracing-refactor_new-3` (HEAD: `774f121d`)
**Mergebase with main:** `7a66c024`

---

## Phase 1 — Ship-blocker fixes (must land before merge)

> **Goal:** Clear the 4 confirmed 🔴 regressions. Each is mechanical; bundle into one `fix(session): v2.9 squash regressions` commit.

### T-01 — Add `import warnings` to `compiler/configs.py` `[x] 2026-06-29 — Added at configs.py:19; 24/24 TestDeprecationWarnings tests pass (was 16 failed / 8 passed)`
- **Proof:** `verification/batch-01.md` confirms 8 live `warnings.warn(...)` calls at lines 161, 173, 187, 201, 215, 229, 243, 257 with no `import warnings` at file top.
- **Symptom:** `NameError: name 'warnings' is not defined` whenever any caller passes the deprecated `quantize=` kwarg. Currently dormant (no test exercises that path → CI passes blind).
- **Action:**
  - Add `import warnings` at the top of `src/winml/modelkit/compiler/configs.py`.
- **Verification:**
  - `uv run pytest tests/unit/compiler/test_compiler_configs.py::TestDeprecationWarnings -v` should produce 4 passing tests (currently failing with `NameError`).
- **Effort:** 5 min. **Risk:** none.

### T-02 — Resolve `compiler/cli.py` broken imports `[x] 2026-06-29 — Option B (delete) — orphan 398-LOC sub-CLI with zero importers in src/ or tests/; only egg-info SOURCES.txt referenced it. RED: ImportError reproduced. GREEN: deleted file; uv run python -c "from winml.modelkit import compiler" succeeds; 1072/1072 unit tests pass (was 1072/1072 baseline-after-T-01)`
- **Proof:** `verification/batch-01.md` empirically reproduces `ImportError: cannot import name 'CalibrationConfig'`. The `python -m winml.modelkit.compiler` sub-CLI is bricked at module-import time.
- **Two options — pick one (ask the original author if unclear):**
  - **Option A — Restore the deleted classes.** Bring `class CalibrationConfig` and `class QDQConfig` back into `compiler/configs.py` (or expose them from the `quant.config` module the docstring points to and update `compiler/cli.py` imports accordingly).
  - **Option B — Delete `compiler/cli.py` entirely.** The top-level `winml compile` (via `commands/compile.py`) is the supported entry; the sub-CLI duplicated logic and now diverges (no `--device`, no `resolve_device`).
- **Verification:**
  - `uv run python -c "from winml.modelkit.compiler import cli"` must succeed.
  - If keeping cli: `python -m winml.modelkit.compiler compile --help` must render. If deleting: confirm no other consumer references `winml.modelkit.compiler.cli`.
- **Effort:** Option A: 30 min; Option B: 10 min. **Risk:** Option A has higher API-surface implication.

### T-03 — Restore `_COMMON_SDK_PATHS` fallback in `find_qnn_sdk()` `[x] 2026-06-30 — RESOLVED via Option B. User explicitly confirmed (verbatim: "whatis this? ... D:\\QC, C:\\Qualcomm\\AIStack\\qairt?! delete it!!") that the developer-machine hardcoded paths must stay deleted. No code change needed — _COMMON_SDK_PATHS was already removed in v2.9; this resolution only closes the audit finding. DEEP-DIVE D-03 should be reclassified as "deliberate design choice, not a regression — hardcoded dev-machine paths are forbidden per project policy". The 3 enforcement tests in test_viewer.py stand. Follow-on hardening: codebase-wide audit for other non-common/not-well-known hardcoded paths kicked off the same firing.`
- **Proof:** `verification/batch-03.md` confirms `_COMMON_SDK_PATHS = [r"D:\QC", r"C:\Qualcomm\AIStack\qairt"]` existed in parent `7a66c024:src/winml/modelkit/optracing/qnn/viewer.py`, but is gone from the post-refactor `src/winml/modelkit/session/monitor/qnn/_internal.py:find_qnn_sdk()`.
- **Symptom:** Dev boxes without `QNN_SDK_ROOT` set silently degrade to `basic_fallback` (no QHAS detail).
- **Action:**
  - In `src/winml/modelkit/session/monitor/qnn/_internal.py`, restore the constant near the top of the file and consult it inside `find_qnn_sdk()` when `os.environ.get("QNN_SDK_ROOT")` is unset.
  - Suggested shape:
    ```python
    _COMMON_SDK_PATHS = (Path(r"D:\QC"), Path(r"C:\Qualcomm\AIStack\qairt"))

    def find_qnn_sdk() -> Path | None:
        env = os.environ.get("QNN_SDK_ROOT")
        if env:
            p = Path(env)
            if p.is_dir():
                return p
        for p in _COMMON_SDK_PATHS:
            if p.is_dir():
                return p
        return None
    ```
- **Verification:**
  - Live: unset `QNN_SDK_ROOT`, run `winml perf -m microsoft/resnet-50 --monitor --op-tracing detail --iterations 30` on a box with one of the common-path SDKs installed; confirm the run uses **detail** (QHAS) mode, not `basic_fallback`.
  - Unit: add `tests/unit/session/monitor/test_find_qnn_sdk.py` with `monkeypatch.delenv("QNN_SDK_ROOT")` and a tmp dir mirroring one of the common paths.
- **Effort:** 15 min code + 30 min test. **Risk:** none.

### T-04 — `auto_device` `last_error` reset on successful registration `[x] 2026-06-29 — RED test added (test_fail_then_succeed_but_wrong_device_raises_device_not_found) reproducing the bug (raised WinMLEPRegistrationFailed instead of DeviceNotFound); GREEN fix at ep_registry.py:409-413 — last_error = None after the inner device-class loop falls through. Full sweep: 983/983 session+commands tests pass.`
- **Proof:** `verification/batch-03.md` quotes the loop body at `src/winml/modelkit/session/ep_registry.py:398-414`. `last_error = e` fires on exception path but never resets to `None` after a successful `register_ep`. Sequence (fail, then succeed-with-wrong-device) yields `WinMLEPRegistrationFailed` with **candidate #1's** stale traceback instead of `DeviceNotFound`.
- **Action:**
  - In `src/winml/modelkit/session/ep_registry.py:auto_device`, inside the `for entry in candidates:` loop, **after** the `winml_ep = self.register_ep(entry)` call succeeds and the inner device-class loop falls through without returning, **explicitly reset** `last_error = None` so the next iteration starts clean.
  - Suggested shape:
    ```python
    for entry in candidates:
        try:
            winml_ep = self.register_ep(entry)
        except WinMLEPRegistrationFailed as e:
            last_error = e
            continue
        for device in winml_ep.devices:
            if device.device_type == target_device_upper:
                return WinMLEPDevice(ep=winml_ep, device=device)
        last_error = None   # ← THE FIX
    if last_error is not None:
        raise WinMLEPRegistrationFailed(...) from last_error
    raise DeviceNotFound(...)
    ```
- **Verification (TDD red→green):**
  - Add `tests/unit/session/test_auto_device.py::test_fail_then_succeed_but_wrong_device_raises_device_not_found`:
    - Stub `_discovered` with two `EPEntry` candidates for the same `ep_name`.
    - Patch `register_ep` so the first call raises `WinMLEPRegistrationFailed` and the second returns a `WinMLEP` whose `devices` does NOT include `target.device`.
    - Assert that `auto_device(target)` raises `DeviceNotFound`, NOT `WinMLEPRegistrationFailed`.
  - RED before fix: test fails with `WinMLEPRegistrationFailed`. GREEN after.
- **Effort:** 15 min code + 30 min test. **Risk:** none.

### Phase-1 wrap-up
- **Commit message template:**
  ```
  fix(session): v2.9 squash regressions

  - compiler/configs.py: restore missing `import warnings` (T-01)
  - compiler/cli.py: <resolved per T-02 option chosen>
  - session/monitor/qnn/_internal.py: restore `_COMMON_SDK_PATHS` (T-03)
  - session/ep_registry.py: reset `last_error` after successful register_ep (T-04)
  - Tests added for T-03 and T-04 regression guards.

  Refs: commit-analysis/774f121d/TASKS.md Phase 1
  ```
- **Total Phase-1 effort:** ~1.5 hours code + 1.5 hours TDD tests + 30 min review = **half a day**.

---

## Phase 2 — Dead code removal (single small PR)

> **Goal:** Delete the dead methods/classes/constants confirmed by Batch 3 and Batch 4 verification. Each item below has zero callers in `src/` or `tests/`.

### T-05 — Delete `WinMLSession._detect_best_device()` + `_get_install_suggestion()` `[x] 2026-06-29 — Both methods deleted from session.py (was at lines 538-549 and 565-571 pre-fix). Zero callers confirmed via grep. _get_compile_suggestion at line 551 NOT touched (still load-bearing — separate method). 983/983 session+commands tests pass.`
- **Proof:** Verified zero callers in `src/winml/` or `tests/` via grep. Methods reference a removed "PREFER_NPU policy". (Per `verification/batch-04.md` and `temp/session_function_audit.md` — `_detect_best_device` flagged dead since `2026-05-12-impl-status.md:198` and never removed.)
- **Action:** Remove both methods from `src/winml/modelkit/session/session.py`.
- **Verification:** `uv run pytest tests/unit/session/ tests/unit/commands/` should still pass.
- **Effort:** 10 min.

### T-06 — Delete `compiler/configs.py:_EP_CONTEXT_DEFAULTS` `[x] 2026-06-29 — Constant deleted (was at line 33, frozenset({"qnn","openvino"}), zero callers). Also removed Final from typing import since it was the only consumer (F401 fix). 89/89 compiler tests pass. (Pre-existing F821 'Any undefined' errors at lines 127, 261 untouched — out of scope.)`
- **Proof:** `verification/batch-01.md`. Constant defined, zero readers.
- **Action:** Remove the constant. (If T-02 Option A is chosen and the consolidation is also done here per T-09, this constant becomes the driver instead — in that case keep it.)
- **Verification:** `uv run pytest tests/unit/compiler/` should still pass.
- **Effort:** 5 min.

### T-07 — Delete `session/monitor/openvino_monitor.py` `[x] 2026-06-29 — File deleted + removed from session/__init__.py (import line 32 + __all__ entry). Deleted 4 test methods in tests/unit/session/test_ep_monitor.py: TestOpenVINOMonitor class (3 methods) + test_import_openvino_monitor_from_submodule + test_import_openvino_monitor_from_session + test_openvino_monitor_to_dict_json. Updated docstring mentions in monitor/ep_monitor.py (removed parenthetical refs). 977/977 session+commands tests pass (was 983; net -6 from removed tests).`
- **Proof:** `verification/batch-03.md`. `OpenVINOMonitor.is_available()` returns `False` literally. `commands/perf.py:_resolve_ep_monitor` only dispatches to `qnn` → `QNNMonitor` and `vitisai` → `VitisAIMonitor`. Never selects OpenVINO.
- **Action:**
  - Delete `src/winml/modelkit/session/monitor/openvino_monitor.py`.
  - Drop `OpenVINOMonitor` from `session/monitor/__init__.py` `__all__` if present.
  - Document the deletion in the v2.9 changelog so future op-tracing work for OpenVINO knows there's no existing scaffolding.
- **Verification:** `uv run pytest tests/unit/session/monitor/` should still pass; `winml perf -m microsoft/resnet-50 --device cpu --op-tracing basic` should give a clear "no monitor for openvino" error, not a silent fallback.
- **Effort:** 15 min.

### Phase-2 wrap-up
- **Commit message:** `chore(session): delete v2.9 dead code (_detect_best_device, _EP_CONTEXT_DEFAULTS, OpenVINOMonitor)`
- **Total effort:** ~30 min.

---

## Phase 3 — De-duplication extracts (single PR)

> **Goal:** Eliminate the verbatim-copy-paste blocks surfaced by verification.

### T-08 — Extract `_reject_ep_source(ep, command_name)` `[x] 2026-06-29 — Added _reject_ep_source helper in commands/_ep_arg.py (returns bare ep str, None passthrough, raises click.UsageError on source-tag). Refactored both call sites: build.py:374-387 (15 LOC) and config.py:258-270 (15 LOC) collapsed to 2-line helper calls each. RED: 3 new tests in test_ep_arg.py confirmed ImportError. GREEN: all 3 helper tests pass; full sweep 980 passed, 8 skipped (legitimate: openvino not installed, NPU hardware-gated, pre-existing TODO).`
- **Proof:** Verified by Batch 1. `commands/build.py` and `commands/config.py` each carry an identical ~12-LOC try/except block rejecting `--ep <name>@<source>`.
- **Action:**
  - Add `def _reject_ep_source(ep, command_name: str) -> str | None` to `src/winml/modelkit/commands/_ep_arg.py`. It should accept whichever shape the existing blocks accept (the per-file docs cite a tuple/None pair from `EpAtSourceParamType`).
  - Call from `build.py` and `config.py`; delete the duplicated try/except blocks.
- **Verification:** Existing CLI tests for `winml build` and `winml config` should still pass.
- **Effort:** 20 min.

### T-09 — Collapse 8 `for_*` factories in `compiler/configs.py` `[x] 2026-06-29 — Added _PROVIDER_DEFAULTS table (12 LOC) and rewrote for_provider(p, quantize=None) to consume it. Deleted 8 named factories (~117 LOC net). Fixed F821 Any-undefined by re-importing Any. Simplified for_ep_device — dropped dead generic fallback (for_provider only returns None when provider is None, which the EPDeviceTarget already rules out). Test surface: removed redundant TestDeprecationWarnings class + 8 per-EP test methods; added consolidated TestForProvider::test_for_provider_quantize_emits_deprecation (parametrized 8×2) + test_for_provider_no_quantize_no_warning. Updated 7 sites that called named factories (6 in test_compiler_configs.py + 1 in test_config.py). Docstring examples in compiler/__init__.py, compiler.py, configs.py updated to for_provider("qnn"). RED confirmed: 8 TypeErrors on quantize= kwarg before refactor. GREEN: 40/40 compiler_configs + full sweep 1516 passed, 7 legit skips, 3 pre-existing TestFromOnnxDictDispatch failures (ep_device kwarg-only, unrelated; verified pre-existing via git stash baseline).`
- **Proof:** Verified by Batch 1. Eight factories (`for_qnn`, `for_cpu`, `for_dml`, `for_vitisai`, `for_openvino`, `for_nv_tensorrt_rtx`, `for_tensorrt`, `for_migraphx`) each ~14 LOC, all with the same `quantize=` deprecation block. DEEP-DIVE D11 estimates ~120 LOC drop.
- **Action:**
  - Build `_PROVIDER_DEFAULTS: dict[str, dict[str, Any]]` keyed by full EP name.
  - Add `for_provider(ep: str, ...)` driven by that dict.
  - Have the eight named factories delegate (one-liner each) for back-compat, OR delete them and update callers to `for_provider("QNNExecutionProvider", ...)`.
- **Verification:** `uv run pytest tests/unit/compiler/test_compiler_configs.py` should still pass (with whatever deprecation surface you choose).
- **Effort:** 1 hour code + 30 min test update.

### T-10 — Extract `_top_n(metrics, n, key)` in `session/monitor/report.py` `[x] 2026-06-29 — Added _top_n(operators, n, key) helper at report.py with the defensive-sort comment moved into its docstring; tie-break on op_path centralized inside the helper (callers pass key=lambda o: -o.percent_of_total only). Replaced the byte-identical 7-line block in both _display_basic_report (line 134) and _display_detail_report (line 206). Added TestTopN class (4 tests: sort order, slice cap, empty input, n>len(ops)) to test_report.py. RED: ImportError on _top_n confirmed. GREEN: 53/53 report tests + 1058/1058 session+commands+compiler sweep.`
- **Proof:** Verified by Batch 4. Sort+slice+empty-guard block is **byte-identical** across basic and detail rendering paths.
- **Action:** Extract `_top_n(metrics: list[OperatorMetrics], n: int, key) -> list[OperatorMetrics]` near the top of the file; call from both paths.
- **Verification:** `uv run pytest tests/unit/session/monitor/test_report.py` should still pass.
- **Effort:** 15 min.

### T-11 — Make `_describe_source` delegate to `_entry_source_tag` `[x] 2026-06-29 — _describe_source now imports _entry_source_tag from session.ep_registry and emits desc["source_tag"] alongside the legacy desc["source_kind"] (class name). The tag-decision (which canonical short tag a source maps to) lives in exactly one place. Existing isinstance ladder for per-kind extras (distribution, family_name_prefix, etc.) preserved — they emit fields the tag table doesn't know about. Added tests/unit/commands/test_describe_source.py — 7 parametrized cases × source_tag assertion + 1 legacy source_kind preservation test. RED: 8 KeyError('source_tag') before fix. GREEN: 42/42 (test_describe_source + test_cli + test_entry_source_tag); full sweep 1066 passed, 8 legit skips.`
- **Proof:** Two implementations of the same `isinstance`-on-`EPSource` dispatch — once in `commands/sys.py:_describe_source` (returns a dict descriptor including the tag string) and once in `session/ep_registry.py:_entry_source_tag` (returns just the tag string). Adding a new `EPSource` subclass requires updating both.
- **Action:** Inside `_describe_source`, call `_entry_source_tag(entry)` to derive the tag field; keep the dict-descriptor body for the other fields it constructs.
- **Verification:** `uv run pytest tests/unit/commands/test_cli.py` and `tests/unit/session/test_entry_source_tag.py` should still pass.
- **Effort:** 15 min.

### T-12 — Collapse 14 `_require` calls in `_internal._extract_summary` `[x] 2026-06-29 — Option B chosen (no extra dataclass surface). Added _REQUIRED_SUMMARY_KEYS: tuple[tuple[str, str], ...] (14 (render_key, raw_key) pairs) at _internal.py module level. Collapsed 14 hand-written _require calls in _extract_summary into a single dict-comprehension over the table. The renderer (report.py) remains the source of truth for user-facing names; raw QHAS keys come from the SDK schema. Added test_required_summary_keys_drives_full_renderer_surface — pins the table content. RED: ImportError on _REQUIRED_SUMMARY_KEYS. GREEN: 33/33 qnn monitor tests + 1067/1067 session+commands+compiler sweep, 8 legit skips.`
- **Proof:** Verified by Batch 3. `_internal.py:357-370` — 14 boilerplate `_require(key, dict, schema_name)` calls.
- **Action — pick one:**
  - **Option A** — Define a `dataclass QHASSummary` with typed fields; use `dataclasses.fields()` to drive the validation loop in one pass.
  - **Option B** — Replace with a single dict-comprehension over a `_REQUIRED_KEYS: tuple[str, ...]` constant that validates and projects in one pass.
- **Verification:** `uv run pytest tests/unit/session/monitor/qnn/` should still pass (the QHAS sample tests).
- **Effort:** Option A: 1 hour; Option B: 30 min.

### T-13 — Extract `_ep_short_or_none(ep_full)` helper `[x] 2026-06-29 — Added _ep_short_or_none(ep_full: str) -> str | None to session/ep_device.py per plan signature. Refactored both call sites: config/build.py:613-614 and config/precision.py:276-277 now collapse to `_ep_short_or_none(_canonical) if _canonical is not None else None`. precision.py drops short_ep_name import (only consumer). Added TestEpShortOrNone class (2 tests: non-CPU short-name passthrough + CPU collapses to None). RED: ImportError on _ep_short_or_none. GREEN: 324/324 config+ep_device + 1337/1337 wide sweep (session+commands+compiler+config), 8 legit skips.`
- **Proof:** `_short if _short != "cpu" else None` duplicated verbatim in `config/build.py` and `config/precision.py`.
- **Action:** Add `def _ep_short_or_none(ep_full: str) -> str | None: short = short_ep_name(ep_full); return None if short == "cpu" else short` to `session/ep_device.py`; import in both consumers.
- **Verification:** `uv run pytest tests/unit/config/` should still pass.
- **Effort:** 10 min.

### T-14 — Dedup `_format_bytes` `[x] 2026-06-29 — Deleted ep_device.py's local _format_bytes (9 LOC) and replaced with a module-level re-export: `from .monitor.report import _format_bytes` per the plan's "report.py is the strict superset" guidance. session.monitor.report._format_bytes is now the single source of truth. USER-VISIBLE FORMAT CHANGE: WinMLDevice.ep_facts() now emits "Memory: 8.0 GB" instead of "Memory: 8.0 GiB" (1024-based math preserved; IEC binary "i" label dropped in favor of vendor-spec-style decimal label). Updated 7 tests in test_winml_device.py (TestFormatBytesHelper renamed test_format_gib→gb/mib→mb/kib→kb; test_format_via_ep_facts asserts "8.0 GB"; 2 TestEpFacts memory assertions updated from "GiB"→"GB"). Added test_format_bytes_is_single_source_of_truth pinning `ep_device._format_bytes is monitor.report._format_bytes`. RED: 5 failures on label mismatch + 1 identity-failure before fix. GREEN: 15/15 _format_bytes consumer tests + 1338/1338 wide sweep (session+commands+compiler+config), 8 legit skips.`
- **Proof:** `_format_bytes` defined in both `session/ep_device.py:740-748` and `session/monitor/report.py:67-79` with slightly different signatures (`int` vs `int|float|None`). The `report.py` version is a strict superset.
- **Action:** Delete the `ep_device.py` copy; have `WinMLDevice.ep_facts()` import `_format_bytes` from `session.monitor.report` (or move `_format_bytes` to `utils/_format.py` and have both consumers import from there).
- **Verification:** `uv run pytest tests/unit/session/` should still pass.
- **Effort:** 15 min.

### Phase-3 wrap-up
- **Commit message:** `refactor(session): dedup v2.9 verbatim copies (T-08..T-14)`
- **Total effort:** ~3.5 hours.

---

## Phase 4 — Taxonomy + casing fix (separate PR)

### T-15 — Resolve `VALID_EPS` vs `known_ep_short_names()` disagreement `[x] 2026-06-29 — Option A chosen: dropped "cuda": "CUDAExecutionProvider" from _SHORT_TO_FULL. Decision driver: the existing test_ep_device_specs_count's comment at test_ep_device.py:254-256 explicitly documents "CUDAExecutionProvider was dropped in the v1 catalog — not currently measured by this project". CUDA was documentation drift, the catalog is authoritative. Added the plan's verification-gate test test_valid_eps_matches_known_short_names asserting VALID_EPS == known_ep_short_names(). Updated test_expand_ep_name_cuda_tensorrt → test_expand_ep_name_tensorrt and test_short_ep_name_cuda_tensorrt_round_trip → test_short_ep_name_tensorrt_round_trip (cuda no longer a recognized alias). Removed "cuda" from architecture's _EP_SHORT_NAMES inline-literal detector set. Left ORT marker map (conftest.py), EPConfig dataclass test (test_config.py), and e2e marker test (test_session.py) unchanged — those reference cuda in non-catalog contexts (ORT EP availability, dataclass field passthrough, hardware-gated marker). ALSO: cleaned up my T-13 architecture-test violations by exporting known_ep_short_names and _ep_short_or_none through session/__init__.py and fixing config/build.py + config/precision.py + test_ep_device.py to import through the facade. RED: VALID_EPS != known_ep_short_names() (extra "cuda" in latter). GREEN: 1569/1569 passed + 8 legit skips + 5 pre-existing baseline failures (3 TestFromOnnxDictDispatch + 2 architecture-test baseline violations in commands/compile.py + commands/_ep_arg.py + tests with pre-existing direct ep_device imports — all confirmed pre-existing via git stash).`
- **Proof:** `verification/batch-A` (logged in SUMMARY.md "Internal inconsistencies"). `VALID_EPS` is 8 short names from `EP_DEVICE_SPECS`; `known_ep_short_names()` is 9 from `_SHORT_TO_FULL` including `cuda`. `EPDeviceTarget(ep="cuda", ...)` passes validation but has no catalog row; silent crash when `default_device_for_ep("CUDAExecutionProvider")` returns `None`.
- **Action — pick one:**
  - **Option A** — Drop `cuda` from `_SHORT_TO_FULL`.
  - **Option B** — Add a catalog row for `CUDAExecutionProvider` to `EP_DEVICE_SPECS`.
- **Decision driver:** is CUDA in scope for the project's target hardware envelope? If yes (per `docs/design/session/2026-05-13-ep-device-spec-design.md` discussion), Option B. If CUDA is documentation drift, Option A.
- **Verification:** Add `tests/unit/session/test_ep_device.py::test_valid_eps_matches_known_short_names` asserting `VALID_EPS == known_ep_short_names()`.
- **Effort:** 20 min (Option A) or 1 hour (Option B with provider_options defaults).

### T-16 — Lowercase the `DEVICE_TO_DEVICE_TYPE` / `DEVICE_TYPE_TO_DEVICE` maps + migrate out of `utils/constants.py` `[x] 2026-06-29 — Moved DEVICE_TO_DEVICE_TYPE + DEVICE_TYPE_TO_DEVICE to session/ep_device.py with lowercase {"cpu","gpu","npu"} keys (and lowercase string values), exported through session/__init__.py for facade access. Moved normalize_ep_name + extract_ep_options + _EP_CLI_PREFIXES to utils/cli.py. Updated 4 caller files: analyze/runtime_checker/check_ops.py (.lower() at DEVICE_TO_DEVICE_TYPE lookup, .upper() at filename construction to preserve _NPU_ convention), analyze/pattern/check_patterns.py (same pattern), analyze/core/runtime_checker_query.py (.lower() at lookup), analyze/analyzer.py + commands/analyze.py + commands/config.py (normalize_ep_name import path). Deleted src/winml/modelkit/utils/constants.py entirely. Fixed pre-existing latent bug exposed by the migration: added "migraphx" to _EP_CLI_PREFIXES to match test_new_aliases_work's pinned contract (test file was collection-broken at baseline due to 3 non-existent symbol imports — fixed those too: dropped TestSupportedEPs/TestEPAliases/TestAllEPNames classes targeting symbols already removed pre-T-16). Added tests/unit/session/test_device_type_maps.py — 4 tests pinning new contract (lowercase keys + values + inverse round-trip + utils.constants module deletion). RED: ImportError on DEVICE_TO_DEVICE_TYPE from session.ep_device + DID NOT RAISE on utils.constants import. GREEN: 4/4 new contract tests + 27/27 ep_constants tests + full sweep across session+commands+compiler+config+utils+analyze+architecture+models = 2989 passed, 53 legit skips, 27 pre-existing baseline failures (22 zombie analyze tests against gone sysinfo.device module + 3 TestFromOnnxDictDispatch + 2 architecture baseline violations — all confirmed pre-existing via git stash).`
- **Proof:** `verification/batch-04.md`. `utils/constants.py` is 92 lines with `normalize_ep_name` + `extract_ep_options` still real consumers, but the two enum-bridge maps use **uppercase** keys while the project convention is lowercase — silent casing-mismatch footgun documented in DEEP-DIVE D12.
- **Action:**
  - Move `DEVICE_TO_DEVICE_TYPE` + `DEVICE_TYPE_TO_DEVICE` into `session/ep_device.py` with **lowercase** keys.
  - Update the four callers (`analyze/runtime_checker/check_ops.py`, `analyze/pattern/check_patterns.py`, `analyze/core/runtime_checker_query.py`, `utils/cli.py`) to lowercase at the call site OR migrate them to lowercase too.
  - Move `normalize_ep_name` + `extract_ep_options` to `commands/_cli_helpers.py` (or `utils/cli.py` if that's the more appropriate landing pad).
  - Delete `utils/constants.py` entirely.
- **Verification:** `uv run pytest tests/unit/` should pass; live: `winml sys --list-ep` should render unchanged.
- **Effort:** 1.5 hours.

### Phase-4 wrap-up
- **Commit message:** `refactor(session): taxonomy + casing cleanup (T-15, T-16)`
- **Total effort:** ~2 hours.

---

## Phase 5 — Design-doc cleanup (separate PR — no code change)

> **Goal:** Land the v2.9 doc-cleanup items so `docs/design/session/2_coreloop.md` reflects the shipped code.

### T-17 — Apply D-05..D-08 from prior `temp/sys_perf_flow_doc.md` audit `[x] 2026-06-29 — Doc cleanup applied to docs/design/session/2_coreloop.md (1199 → 1213 LOC). Item-by-item: (1) §7.1.1 L2 mechanism: entry.source.is_compatible() → EP_CATALOG.is_compatible(entry.ep_name) at 3 sites (table mechanism column + clarification prose + status-derivation pseudocode); (2) §7 fan-out pseudocode: (results, failures) two-list pattern replaced with ep_records: dict[str, list[(EPEntry, WinMLEP|None, Exception|None, dict|None)]] 4-tuple aggregation; (3) §5.5 BuiltinSource pseudocode: ort.get_ep_devices() call switched to _ort_get_ep_devices_or_fail(entry); Raises section gained explicit _ort_get_ep_devices_or_fail clause; (4) §5.6 auto_device pseudocode: dropped ep.ep_devices() iteration; rewrote as for device in winml_ep.devices with explicit WinMLEPDevice(ep=winml_ep, device=device) construction + last_exc=None reset after successful registration but no device-class match (mirrors the T-04 ep_registry.py:413 fix); (5) §5.7 __init__: added enable_ep_context conditional that defers InferenceSession construction and sets _state=INITIALIZED instead of COMPILED; (6) §5.9 struck "current body at session/session.py:191 calls register_ep" stale sentence; (7) §11.7 "v2.17 memoizes" → "Currently implemented as" memoized; (8) §11.1 WinMLEPDevice file:line corrected from ep_device.py:54 to ep_registry.py:143 (current location confirmed via grep); (9) §10 Open Questions: struck "Ep → EP casing sweep ... queued for a one-shot rename PR" + "EPSource.resolve() → Iterator[EPEntry] refactor queued" — both shipped, replaced with (shipped) markers; (10) §6.3 deleted orphaned "Both given; resolved EP not in available_eps()" failure row (removed in v2.6); (11) §6.1 "auto sentinel normalized to None" → "compared against the literal 'auto' at each branch — no up-front normalization" matching the implementation; (12) §11.7 verified — no standalone AmbiguousMatch / IncompatibleListingPick / AmbiguousListingPick table rows remain (already absent; existing §11.4 UnknownListingPick row explicitly notes v2.9 deletion); (13) docs/design/session/4_winml_device.md v1.5 verified — explicitly records "Deletes the trivial wrap_ort_device(d) shim" at the v1.5 changelog entry; (14) two inline # wrap_ort_device comments in §5.5 pseudocode: verified absent (already cleaned). Pure docs change — no code or tests touched.`
- §7.1.1 L2 mechanism prose (`entry.source.is_compatible()` → `EP_CATALOG.is_compatible(entry.ep_name)`)
- §7 fan-out pseudocode (`(results, failures)` two-list → `ep_records: dict[str, list[4-tuple]]`)
- §5.5 BuiltinSource pseudocode (use `_ort_get_ep_devices_or_fail`; add `WinMLEPRegistrationFailed` to Raises)
- §5.6 `auto_device` pseudocode (drop `ep.ep_devices()`; use `for device in winml_ep.devices`)
- §5.7 `__init__` conditional `InferenceSession` deferral when `enable_ep_context=True`
- §5.9 strike "current body at session.py:191 calls register_ep" stale sentence
- §11.7 replace "v2.17 memoizes" with "currently implemented as"
- §11.1 fix `WinMLEPDevice` file:line (`ep_device.py:54` → `ep_registry.py:142`)
- §11.3 strike `ResolvedEp → EPEntry` "rename pending" (already done)
- §6.3 delete the orphaned "Both given; resolved EP not in `available_eps()`" failure row (removed in v2.6)
- §6.1 fix "auto sentinel normalized to None" prose (code compares against `"auto"` literal, never normalizes)
- §1170/§1199 `2_coreloop.md` table rows: remove deleted `AmbiguousMatch` / `IncompatibleListingPick` / `AmbiguousListingPick`
- `4_winml_device.md` v1.5 — already records `wrap_ort_device` deletion; verify
- Two inline `# wrap_ort_device` comments in §5.5 pseudocode block: rename

- **Effort:** 1.5 hours.

### Phase-5 wrap-up
- **Commit message:** `docs(session): v2.9 doc-cleanup sweep`
- **Total effort:** ~1.5 hours.

---

## Phase 6 — Optional: monitor extension surface (deferred discussion)

### T-18 — Decide on monitor extensibility (`register_tracer` registry)
- **Status:** Open design question. The pre-refactor `optracing` package had a `register_tracer(ep_pattern, level)` registry; the v2.9 refactor dropped it. `commands/perf.py:_resolve_ep_monitor` now hardcodes the dispatch.
- **Trade-off:**
  - **Keep hardcoded:** testable, grep-able, no plugin surface to maintain.
  - **Restore registry:** allows out-of-tree EP monitors.
- **Recommendation:** Leave deferred until an external plugin actually needs it. Document in `docs/design/session/monitor/1_prd.md` as an open question.
- **Effort if pursuing:** ~1 day (registry + decorator + docs).

---

## Summary scorecard

| Phase | Tasks | Total effort | Risk | Priority |
|---|---|---|---|---|
| 1 — Ship blockers | T-01..T-04 | ~half a day | Low (mechanical) | **MUST** |
| 2 — Dead code removal | T-05..T-07 | ~30 min | Low | **HIGH** |
| 3 — De-duplication | T-08..T-14 | ~3.5 hours | Low | **MEDIUM** |
| 4 — Taxonomy + casing | T-15, T-16 | ~2 hours | Medium (touches 4 consumers) | **MEDIUM** |
| 5 — Doc cleanup | T-17 | ~1.5 hours | None (no code) | **HIGH** |
| 6 — Optional registry | T-18 | ~1 day | n/a | **LOW (deferred)** |
| **Total to ship-ready** | 1 + 2 + 5 = T-01..T-07 + T-17 | **~3 hours** | Low | |
| **Total to fully polished** | All phases except 6 | **~9 hours** | Low–medium | |

## Ordering

Phases are sequenceable. Recommended commit order:
1. **Phase 1** first (unblocks ship).
2. **Phase 2** + **Phase 5** can land in parallel as separate PRs.
3. **Phase 3** + **Phase 4** can land after Phase 1 in any order.
4. **Phase 6** is open-ended; track in a follow-up issue.

## Verification gate at each phase end

After each phase commit, run:
```bash
find . -name "*.pyc" -path "*winml*" -delete
uv run pytest tests/unit/ -x --timeout=60 -q
uv run winml sys --list-ep        # spot-check renders unchanged
uv run winml perf -m microsoft/resnet-50 --iterations 10 --device cpu  # 1-cycle smoke
```
A green sweep is the gate to the next phase.

## What this plan does NOT cover

- **D-19 `eval.py` `"auto"` passthrough latent risk** — needs live test before deciding (not addressed in TASKS; live-verify first, then plan).
- **`_transformers_compat.py` lifetime** — orthogonal optimum-onnx 0.1.0 ↔ transformers 5.x shim; delete when optimum-onnx 0.2 ships. Track as a calendar reminder, not a code task.
- **Two retracted findings (R6, R8)** — these were FALSE per `FINAL-VERDICTS.md` and require no action.
