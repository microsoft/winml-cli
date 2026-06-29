# Final Verdicts — Fact-Check of `commit-analysis/774f121d/` per-file reviews

*Verification pass over all 81 per-file review docs. Each claim was checked against actual source content. Reports landed at `verification/batch-{00..04}.md`.*

## Confirmed regressions — all four 🔴 hold up under scrutiny

| ID | Claim | Verification |
|---|---|---|
| **D1** | `compiler/configs.py` missing `import warnings` with 8 live `warnings.warn(...)` call sites | ✅ **REAL.** Lines 161, 173, 187, 201, 215, 229, 243, 257 each call `warnings.warn(...)`. Top-of-file has no `import warnings`. Bug is dormant (module imports fine) but fires `NameError` whenever any caller passes deprecated `quantize=` kwarg. No `tests/unit/compiler/` test exercises that path → CI passes blind. |
| **D2** | `compiler/cli.py` imports `CalibrationConfig` + `QDQConfig` from `.configs` — neither exists | ✅ **REAL — empirically reproduced.** `compiler/cli.py:14-19` does `from .configs import (CalibrationConfig, ..., QDQConfig, ...)`. Both classes were deleted (moved to `WinMLQuantizationConfig` in `quant.config` per `configs.py` docstring). `uv run python -c "from winml.modelkit.compiler import cli"` fails with `ImportError: cannot import name 'CalibrationConfig'`. The `python -m winml.modelkit.compiler` sub-CLI is bricked at module import. Top-level `winml compile` via `commands/compile.py` unaffected. |
| **D3** | `find_qnn_sdk()` lost `_COMMON_SDK_PATHS` fallback (`D:\QC`, `C:\Qualcomm\AIStack\qairt`) | ✅ **REAL.** Verified by reading parent commit `7a66c024`'s `optracing/qnn/viewer.py` (had `_COMMON_SDK_PATHS = [r"D:\QC", r"C:\Qualcomm\AIStack\qairt"]`). Now gone in the v2.9 `session/monitor/qnn/_internal.py:find_qnn_sdk()`. Dev boxes without `QNN_SDK_ROOT` silently degrade to `basic_fallback`. |
| **D4** | `WinMLEPRegistry.auto_device` `last_error` never reset after successful registration → wrong exception type | ✅ **REAL.** `ep_registry.py:398-414`: the loop sets `last_error = e` on exception but never resets to `None` after a successful registration that misses the device-class filter. Result: when candidate #1 fails and candidate #2 registers cleanly but exposes no `target.device` match, the precedence loop exhausts and raises `WinMLEPRegistrationFailed` with candidate #1's stale traceback — when the correct exception type is `DeviceNotFound`. Single-line fix: `last_error = None` after the successful `register_ep` return. |

## Withdrawn claims — these were FALSE

| Previous ID | Claim | Verification |
|---|---|---|
| **R6** | `qairt_session.py:from ..session import _build_session_options` is fragile attribute-fallthrough | ❌ **FALSE — empirically verified.** The relative form `..session` from inside `qairt/` resolves to the **sibling submodule** `session.session` via standard PEP-328 relative-import semantics. `from winml.modelkit.session import _build_session_options` (absolute, package-level) would fail because the symbol isn't on `__init__.py`'s surface, but the relative form succeeds correctly. No fragility. **WITHDRAWN.** |
| **R8** | `WinMLDevice.ort_handle` public accessor unused; delete or wire `session.py` to use it | ❌ **FALSE.** `analyze/runtime_checker/ep_checker.py:67` consumes `ort_handle` as a real production caller. The property's own docstring codifies the public-accessor-for-external / `_ort`-for-internal-session-build split as a **deliberate API boundary** — consistent with the "no private symbol imports outside session/" rule. **WITHDRAWN.** |

## Overstated claims — kernel of truth but imprecise

| Doc | Claim | Correction |
|---|---|---|
| `commands__build.md` | `build.py` migrated `--device` Choice to `["auto", *sorted(VALID_DEVICES)]` | ⚠ False. `build.py:--device` is free-form `str \| None` with no Choice. Compile/config/eval/perf got the Choice migration; **build did not**. |
| `ep_path.md` | `__all__` exports 10 names | ⚠ Actually 11. |
| `eval__evaluate.md` | Diff is +8/-2 | ⚠ Actually +9/-2. |
| `ep_path.md` | `EPCatalog` lines 68-186 | ⚠ Conflates class body (68-151) with module-level `EP_CATALOG` instance (154-186). |
| `analyze__analyzer.md` | VitisAI carve-out at 716-717 | ⚠ Actually line 742. |
| `analyze__analyzer.md` | Default-device fallback at 687 | ⚠ Actually 688. |
| `D12` (DEEP-DIVE — `utils/constants.py` "near-empty") | 60-line stub, two ORT-enum maps + a CLI-prefix tuple | ⚠ Actually 92 lines; missed mentioning `normalize_ep_name` + `extract_ep_options` (two real functions). The "could be deleted entirely" recommendation still stands provided the two functions migrate (per D12's own recommendation block, which does in fact list both functions in its proposed migration). |
| 3× StrEnum docs (`ihv_type.md`, `information.md`, `onnx_model.md`) | "Other files still use `(str, Enum)`" cross-file context | ⚠ FALSE cross-file claims. Those other files are already on StrEnum or were migrated in the same commit. Within-file claims about the three docs themselves are correct. |
| `analyze__core__runtime_checker_query.md` | `check_ops.py:41` still calls `winml.register_execution_providers(ort=True)` | ⚠ FALSE — that call was removed in the same commit. (The doc itself separately notes the removal — inconsistent with its own line citation.) |

## Verified solidly

- ✅ **D-07** OpenVINOMonitor dead stub (`is_available()` returns `False` literally, `_resolve_ep_monitor` only dispatches QNN+VitisAI)
- ✅ **D-15** `report.py` duplicate sort+slice+empty-guard (byte-identical duplication at cited lines)
- ✅ **D-16** 14 `_require` calls in `_internal._extract_summary` (exact count verified at `_internal.py:357-370`)
- ✅ **D-18** `_detect_best_device` + `_get_install_suggestion` dead methods (zero callers in `src/` or `tests/`)
- ✅ **auto.py:411 `.lower()` fix landed** (all 4 call sites consistently use `.lower()`; verified at lines 173, 218, 356, 411)
- ✅ All 7 `optracing/*.py` deletion docs correct (`-34`, `-35`, `-227`, `-351`, `-113`, `-64`, `-99` lines; `src/winml/modelkit/optracing/` no longer exists)
- ✅ `ep_path.md` review of the 1518-LOC new file is meticulously grounded — EPSource ABC structure, 6 concrete subclasses (`BuiltinSource`, `PyPISource`, `NuGetSource`, `DirectorySource`, `WinMLCatalogSource`, `MSIXPackageSource`), immutable `EPCatalog` with `MappingProxyType` + locked `__setattr__`, `EPEntry.is_filesystem_backed()`, `discover_all_eps` dedup — all match source verbatim.

## Aggregated counts across all 5 verification batches

| Batch | Docs reviewed | Claims | Verified | Overstated | False |
|---|---|---|---|---|---|
| 0 | 13 (top-level + analyze) | ~142 | ~115 | ~17 | ~9 |
| 1 | 18 (analyze/commands/compiler/config) | not enumerated | 17 docs fully verified | 1 doc overstated | 0 docs false (D-01/D-02 confirmed) |
| 2 | 19 (ep_path/models/optracing/serve) | many | 19 docs fully verified | minor metric off-by-1 | 0 false |
| 3 | 15 (session/serve/monitor partial) | many | 14 docs verified | minor count overstatements | 1 false (R8 `ort_handle`) |
| 4 | 16 (session/qnn_monitor/sysinfo/utils/telemetry/winml) | many | 14 docs verified | 1 overstated (D-09 / D12 utils constants) | 1 false-by-framing (R8 again; R6 qairt) |

## Net judgment

The per-file review corpus is **substantively accurate** — every architectural claim about the v2.9 refactor (BuiltinSource synthesis, idempotent `register_ep`, `EpAtSourceParamType`, L1/L2 status taxonomy unification, monitor control-inversion) checks out against actual code. **All four 🔴 regressions hold under scrutiny** and remain ship blockers.

The error pattern across the corpus is **cross-file context drift**: the agents were good at within-file verification (open file, read line, quote) but occasionally over-reached on claims about other files in the codebase (StrEnum cross-references, `check_ops.py` register call already removed, `ort_handle` "unused" while another package consumes it, `_build_session_options` "fragile" when it's standard PEP-328). The fix for the DEEP-DIVE corpus is to **demote** R6 and R8 to "withdrawn" status (done) and to add a "fact-check pass" note pointing at this verdicts doc.

## What was updated

1. **`commit-analysis/774f121d/DEEP-DIVE.md`** — R6 and R8 marked **WITHDRAWN** with full justification inline.
2. **`commit-analysis/774f121d/SUMMARY.md`** — already correct (qairt fragile-import was never propagated to SUMMARY; only mentioned during the Batch A discussion, not in the final SUMMARY text).
3. **`commit-analysis/774f121d/DESIGN-DOCS-INDEX.md`** — no changes needed (the index doesn't make per-finding claims).

## What remains in the per-file corpus

The 81 per-file docs retain their original content. The line-count overstatements (off-by-1, off-by-25) and the cross-file context errors are noted in the per-batch `verification/batch-NN.md` reports for the curious. Fixing them in-place would obscure the audit trail; readers consulting a per-file doc should treat its narrative as authoritative but spot-check line numbers via `git show 774f121d:<path>`.
