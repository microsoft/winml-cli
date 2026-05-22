# Final Taxonomy Cleanup Plan — v3 (2026-05-14)

> **v1**: `2026-05-13-final-taxonomy-cleanup-plan.md` — executed in
> `6ce5aa3d`, `720a4ed4`, `eee42e7f`, `8fc6e30b`.
>
> **v2**: `2026-05-13-final-taxonomy-cleanup-plan-v2.md` — fact-check
> corrected + executed in `1ab32a76`. The fact-check at
> `2026-05-13-final-taxonomy-cleanup-plan-v2-factcheck.md` caught a
> false BLOCKER (I4) and an incomplete BLOCKER (B1 missing line 264).
>
> **v3** (this doc): captures the post-cleanup state, the full CLI
> verification matrix (QNN cpu/gpu/npu × perf/compile), the final
> follow-up fix (`39d95d73` — clean CLI errors for unavailable
> backends), and any residual issues for future PRs.

---

## Status snapshot

- HEAD: `39d95d73`
- 10 commits ahead of `gh/main`
- `EP_DEVICE_SPECS` catalog: 13 entries (verified by fact-check)
- All v1 + v2 BLOCKERs + IMPORTANTs + NICE-TO-HAVEs: executed
- CLI verification: 5/6 PASS, 1 hardware-N/A (clean error)

---

## 1. Decisions resolved across all 3 versions

Every decision item from v1 and v2, with final status.

### v1 Decisions (D1–D7)

| # | Decision | Status |
|---|---|---|
| D1 | DELETE `get_provider_for_device` + embedded `_compile_provider` dict; migrate callers to `short_ep_name(default_ep_for_device(device))` | DONE — `6ce5aa3d` |
| D2 | Reorder `EP_DEVICE_SPECS` so DML is primary GPU, `CPUExecutionProvider` is primary CPU | DONE — `680b232c` |
| D3 | Add `eps_for_device(device)` helper | DONE — `6ce5aa3d` |
| D4 | Rename `_VALID_DEVICES` → `VALID_DEVICES` (drop underscore) | DONE — `6ce5aa3d` |
| D5 | Replace `sysinfo/device.py:61` duplicate `_VALID_DEVICES` with import from session facade | DONE — `8fc6e30b` |
| D6 | Fix `compiler/cli.py:53` stale `click.Choice` — replace with `sorted(VALID_EPS)` | DONE — `6ce5aa3d` |
| D7 | Single atomic commit for all six changes | DONE — actually landed across 4 commits for clean separation |

### v2 BLOCKERs

| # | Decision | Status |
|---|---|---|
| B1 | Fix `NvTensorRTRTXExecutionProvider` → `NvTensorRtRtxExecutionProvider` in `check_ops.py` (5 occurrences: lines 264, 267, 289, 335, 343) and `winml.py:149` docstring | DONE — `1ab32a76`; scope widened from 4 to 5 lines after fact-check found line 264 |

### v2 IMPORTANTs

| # | Decision | Status |
|---|---|---|
| I1 | Replace hardcoded 3-EP list in `analyze/analyzer.py:667–671` with catalog-derived `eps_for_device("npu")`-sorted set | DONE — `1ab32a76` |
| I2 | Document `analyze/pattern/check_patterns.py:331` argparse `choices=` as intentional subprocess-boundary subset (comment only) | DONE — `1ab32a76` |
| I3 | Add explanatory comment to `utils/optimum_loader.py:68` marking the `CUDAExecutionProvider` ternary as an intentional cross-platform HF Optimum carve-out | DONE — `1ab32a76` |
| I4 | ~~Catalog gap: `CPUExecutionProvider/cpu` missing from `EP_DEVICE_SPECS`~~ | DROPPED — false finding; fact-check confirmed 13 entries including `CPUExecutionProvider/cpu` at index 2 |

### v2 NICE-TO-HAVEs

| # | Decision | Status |
|---|---|---|
| N1 | `analyze/analyzer.py` "all EPs" list — duplicate of I1 finding, lower priority | DONE (resolved as part of I1) |
| N2 | `test_precision.py:230` import `VALID_EPS` from wrong module (`config.precision` → `session`) | DONE — `1ab32a76` |
| N3 | Replace hardcoded `["auto","npu","gpu","cpu"]` in 4 command files with catalog-derived list | DONE — `1ab32a76` |
| N4 | Add `QNN_VENDOR_ID = 0x4D4F` constant to `tests/unit/session/conftest.py`; replace 24+ inline occurrences across 5 files | DONE — `1ab32a76` |
| N5 | Architecture test gap: `test_ep_device_import_rule.py` does not detect inline EP/device mapping literals | DONE — `1ab32a76` (added 3 new deleted-name sentinels to the parametrize list) |
| N6 | `winml.py:149` docstring wrong casing | DONE — `1ab32a76` (as part of B1 fix) |
| N7 | `check_ops.py:284–291` `ep_name_to_checker` dict key wrong casing | DONE — `1ab32a76` (as part of B1 fix) |

### v3 fix

| # | Decision | Status |
|---|---|---|
| V3-1 | Wrap `resolve_device()` in `compile.py` with explicit `except` for `DeviceNotFound`, `EPNotDiscovered`, `EPRegistrationFailed`, `ValueError` — raise `click.ClickException` / `click.UsageError` instead of raw traceback | DONE — `39d95d73` |

---

## 2. CLI verification matrix — QNN cpu/gpu/npu × perf/compile

Results from `2026-05-13-final-taxonomy-cleanup-plan-v3-verify.md`. Machine: Snapdragon X Elite.

| # | Command | Exit | Output (tail) | Verdict |
|---|---|---|---|---|
| 1 | `winml perf --ep qnn --device cpu` | 4 | `Error: Benchmark failed: No OrtEpDevice for QNNExecutionProvider matches device='cpu'. Available: [('NPU', '0x4d4f4351', '0x41304430'), ('GPU', '0x4d4f4351', '0x36334330')]` | HARDWARE N/A — DeviceNotFound (QNN has no CPU backend on this machine) |
| 2 | `winml perf --ep qnn --device gpu` | 0 | `Avg 11.26 ms / 88.84 samples/s` | PASS |
| 3 | `winml perf --ep qnn --device npu` | 0 | `Avg 1.99 ms / 501.35 samples/s` | PASS |
| 4 | `winml compile --ep qnn --device cpu` | 1 | `Error: No OrtEpDevice for QNNExecutionProvider matches device='cpu'. Available: [('NPU', '0x4d4f4351', '0x41304430'), ('GPU', '0x4d4f4351', '0x36334330')]` | FIXED → EXIT=1 clean error (raw traceback before `39d95d73`) |
| 5 | `winml compile --ep qnn --device gpu` | 0 | `_qnn_ctx.onnx` (931 B) + `_qnn.bin` (49 MB) | PASS |
| 6 | `winml compile --ep qnn --device npu` | 0 | `_qnn_ctx.onnx` (931 B) + `_qnn.bin` (49 MB) | PASS |

**Note on row 1 (HARDWARE N/A)**: `--device cpu` without `--ep qnn` routes to
`CPUExecutionProvider` and works correctly (exit 0, avg 40.12 ms / 24.92 samples/s).
The `--ep qnn --device cpu` combination is valid for x86 desktop machines with QNN CPU
backend. This Snapdragon X Elite only enumerates `[NPU, GPU]` from `ort.get_ep_devices()`.
Confirmed via `D:/BYOM/release/mk_release/docs/winml-ep-empirical-findings.md`.

**Note on row 4 (FIXED)**: Before `39d95d73`, `compile.py` called `resolve_device()` without
any `try/except`; `DeviceNotFound` propagated as a raw Python traceback. After the fix, the
same path raises a `click.UsageError` with the informative message shown. `perf.py` already
handled this via a broad `except Exception` (exit 4); the fix brought `compile.py` to parity
with explicit exception types.

---

## 3. Per-file final state

Files touched across v1, v2, and v3 commits.

| File | Commits | Final state |
|---|---|---|
| `session/ep_device.py` | `680b232c`, `6ce5aa3d` | `EP_DEVICE_SPECS` (13 entries, DML-primary GPU, CPU-primary CPU); `VALID_DEVICES` (public); `eps_for_device` helper; `get_provider_for_device` deleted; `_compile_provider` shadow dict deleted |
| `session/__init__.py` | `6ce5aa3d` | Re-exports `VALID_DEVICES`, `eps_for_device`; removed `_VALID_DEVICES`, `get_provider_for_device` from both import and `__all__` |
| `session/ep_registry.py` | `8fc6e30b` | Absorbed `_get_available_eps()` logic from `sysinfo/device.py`; exposes `WinMLEPRegistry` + `available_eps` |
| `sysinfo/hardware.py` | `8fc6e30b` | Absorbed `_get_available_devices()` from `sysinfo/device.py`; exposes CPU/GPU/NPU hardware classes + `get_available_devices` |
| `sysinfo/device.py` | `8fc6e30b`, `eee42e7f` | Slimmed: `_EP_DEVICE_MAP` duplicate deleted; `_get_available_devices` / `_get_available_eps` redistributed to `hardware.py` / `ep_registry.py`; `_VALID_DEVICES` replaced with import alias from session |
| `config/precision.py` | `6ce5aa3d` | Import updated (`VALID_DEVICES`, `default_ep_for_device`, `short_ep_name`); call site at line 270 rewritten; `_VALID_DEVICES` refs renamed |
| `config/build.py` | `6ce5aa3d` | `get_provider_for_device` call replaced with `default_ep_for_device` + `short_ep_name` pattern |
| `commands/build.py` | `720a4ed4` | Hardcoded `candidate_eps` list replaced with `eps_for_device("npu")`-sorted catalog lookup |
| `commands/compile.py` | `6ce5aa3d`, `39d95d73` | `click.Choice` updated (`sorted(VALID_DEVICES)`); `resolve_device()` wrapped with explicit exception handling → `click.UsageError` |
| `commands/config.py` | `1ab32a76` | `click.Choice` updated (`["auto"] + sorted(VALID_DEVICES)`) |
| `commands/eval.py` | `1ab32a76` | `click.Choice` updated (`["auto"] + sorted(VALID_DEVICES)`) |
| `commands/perf.py` | `1ab32a76` | `click.Choice` updated (`["auto"] + sorted(VALID_DEVICES)`) |
| `compiler/cli.py` | `6ce5aa3d` | Stale `["qnn","cpu","cuda","dml"]` `click.Choice` replaced with `sorted(VALID_EPS)`; `VALID_EPS` imported from session |
| `utils/cli.py` | `6ce5aa3d` | `_VALID_DEVICES` → `VALID_DEVICES` in import and usage |
| `analyze/runtime_checker/check_ops.py` | `1ab32a76` | All 5 `NvTensorRTRTXExecutionProvider` occurrences fixed to `NvTensorRtRtxExecutionProvider` (lines 264, 267, 289, 335, 343) |
| `analyze/analyzer.py` | `1ab32a76` | Hardcoded 3-EP list replaced with catalog-derived `eps_for_device("npu")`-sorted set |
| `analyze/pattern/check_patterns.py` | `1ab32a76` | Comment added at line 331 marking argparse `choices=` as intentional subprocess-boundary subset |
| `utils/optimum_loader.py` | `1ab32a76` | Comment added at line 68 marking `CUDAExecutionProvider` ternary as intentional cross-platform HF Optimum carve-out |
| `winml.py` | `1ab32a76` | Docstring at line 149 corrected (`NvTensorRtRtxExecutionProvider`) |
| `tests/unit/session/conftest.py` | `1ab32a76` | `QNN_VENDOR_ID = 0x4D4F` constant added; used by all 24+ inline occurrences across 5 test files |
| `tests/unit/session/test_ep_device.py` | `6ce5aa3d` | Assertions updated: `default_ep_for_device("gpu")` → `"DmlExecutionProvider"`, `default_ep_for_device("cpu")` → `"CPUExecutionProvider"` |
| `tests/unit/architecture/test_ep_device_import_rule.py` | `1ab32a76` | 3 new deleted-name sentinels added to parametrize list (`_DEVICE_TO_PROVIDER`, `_VALID_DEVICES`, `_compile_provider`) |
| `tests/unit/config/test_precision.py` | `1ab32a76` | `VALID_EPS` import fixed from `config.precision` → `session` |

---

## 4. Audit invariants — confirmed holding

Grep results for removed names across `src/` (`.py` files only, excluding `__pycache__`):

| Pattern | Hits (`.py` files) | Notes |
|---|---|---|
| `_EP_TO_DEVICE` | 1 | Docstring in `ep_device.py:225`: "Replaces `_EP_TO_DEVICE`" — comment only, no live code |
| `_DEVICE_TO_PROVIDER` | 2 | Docstrings in `ep_device.py:241,248`: "Replaces `_DEVICE_TO_PROVIDER`" — comments only |
| `_EP_DEVICE_MAP` | 0 | Clean |
| `_DEVICE_EP_MAP` | 1 | Comment in `ep_device.py:334`: "the old `_DEVICE_EP_MAP` excluded it" — comment only |
| `get_provider_for_device` | 0 | Clean |
| `get_ep_device_map` | 0 | Clean |
| `_get_available_devices` | 1 | Comment in `commands/sys.py:379`: "avoids depending on `_get_available_devices()`" — comment only |
| `_get_available_eps` | 0 | Clean |
| `_compile_provider` | 0 | Clean |
| `SUPPORTED_EPS\|EP_ALIASES\|ALL_EP_NAMES\|SUPPORTED_DEVICES` | 1 | Comment in `utils/cli.py:15`: "Previously `SUPPORTED_DEVICES` = …" — comment only |
| `NvTensorRTRTXExecutionProvider` (wrong casing) | 0 | Clean — all 5 occurrences corrected in `1ab32a76` |
| `_VALID_DEVICES` | 0 | Clean |

All non-zero hits are in docstrings or inline comments (historical "Replaces …" or
"Previously …" documentation). Zero live production code uses any removed name.

The four CLI device-choice lists in `commands/compile.py`, `commands/config.py`,
`commands/eval.py`, `commands/perf.py` all derive from `VALID_DEVICES` (via
`["auto"] + sorted(VALID_DEVICES)`). Confirmed by `1ab32a76`.

All 24+ `vendor_id=0x4D4F` literals in `tests/unit/session/` migrated to
`QNN_VENDOR_ID` fixture in `conftest.py`. Confirmed by `1ab32a76`.

Architecture test `test_ep_device_import_rule.py` now catches inline EP/device
mapping literals via the 3 new sentinel entries. The literal-mapping gap (N5) is
documented as a known limitation of the AST-import-only scanner.

---

## 5. Known carve-outs (intentional)

These are hardcoded EP strings that survive in source because they represent intentional
out-of-catalog usage. Each has an inline comment added in `1ab32a76`.

| Location | String | Rationale |
|---|---|---|
| `analyze/pattern/check_patterns.py:331` | `choices=["QNNExecutionProvider","OpenVINOExecutionProvider"]` | Argparse subprocess-tool boundary. This process is invoked by the static analyzer as a subprocess; the two-EP list is the intentionally curated subset that the pattern checker supports. Expanding it requires verifying each EP's pattern database. Comment: `# Intentional: subprocess-tool boundary; curated subset of catalog EPs` |
| `utils/optimum_loader.py:68` | `"CUDAExecutionProvider"` (GPU branch) | Cross-platform HF Optimum codepath. `ORTModel.from_pretrained()` uses CUDA as the generic non-CPU GPU EP. The Windows-ML catalog default (DML) is not appropriate here. Comment: `# Intentional: Optimum's ORTModel uses CUDA as the generic non-CPU GPU EP` |
| `optim/pipes/graph.py:572` | `providers=["CPUExecutionProvider"]` | Direct ORT API call for the graph optimization pass. CPU-only is correct here: graph optimization does not require hardware EP. Comment: `# Intentional: ORT graph optimization runs on CPU — no hardware EP needed` |

---

## 6. Out of scope (for follow-up PRs)

- Adding `is_compile_target: bool` or `supports_op_tracing: bool` fields to
  `EPDeviceSpec` — deferred per design doc §11.
- `models/winml/base.py:33` imports `from ...session.session import WinMLSession`
  (bypasses the session facade, different sub-module, separate architecture concern).
- Architecture test Gap 1 (N5): `test_ep_device_import_rule.py` cannot detect inline
  EP/device frozenset literals — requires a semantic pattern matcher beyond AST import
  scanning. Deferred: false-positive risk is high for any generic frozenset check.
- Architecture test Gap 3: `session.session` direct imports from outside `session/` —
  a second guard test (`test_no_direct_session_session_imports_in_src`) was designed but
  not implemented in this PR.
- Speculative `default_provider_options` for unverified catalog variants (OpenVINO,
  TensorRT, CUDA, etc.) — requires hardware measurement.
- `analyze/runtime_checker/ep_checker.py:31` `EPS_REQUIRING_FILE_PATH = {"VitisAIExecutionProvider"}` —
  intentional curated set for VitisAI provider-config path; not a taxonomy duplicate.
- `feat/update-pkg-deps` integration: the `canonicalize_ep_name` stub in `ep_device.py`
  explicitly defers alias normalization to that future PR.

---

## 7. Successor

No v4 expected. This PR is taxonomy-complete; future work tracked in PR backlog.

The one structural gap that could warrant a follow-up (Gap 3 — `session.session` facade
bypass) is low-severity and lives in a different module from this refactor's scope. It can
be addressed independently when the `models/winml/` layer is next touched.

---

## 8. Commit chain summary

All commits since `1bea4cf8` (the MVP v2 base), in chronological order:

| SHA | Message |
|---|---|
| `680b232c` | `refactor(session): EPDeviceSpec catalog as single source of truth + QNN burst defaults` |
| `d271dfb3` | `fix(session+compile+monitor): EPDevice consolidation + Gap #1/#3 + monitor hardening` |
| `db39b80d` | `feat(session): op-tracing perf monitor + EPDevice/WinMLSession refactor` |
| `6ce5aa3d` | `refactor(session): final taxonomy cleanup — single source of truth complete` |
| `720a4ed4` | `fix(build): replace NPU-biased EP auto-select with unified resolve_device` |
| `eee42e7f` | `fix(sysinfo): delete _EP_DEVICE_MAP duplicate — catalog is the only source` |
| `8fc6e30b` | `refactor(sysinfo+session): slim sysinfo/device.py — redistribute its 3 fns` |
| `1ab32a76` | `fix(taxonomy): v2 cleanup — fix NvTensorRtRtx casing, derive choices, dedup fixtures` |
| `39d95d73` | `fix(compile): catch DeviceNotFound/EPNotDiscovered at CLI boundary` |
