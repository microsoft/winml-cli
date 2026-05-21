# Remaining Issues — Post-Cleanup Snapshot

**Date:** 2026-05-13
**Branch:** `feat/op-tracing-refactor`
**HEAD:** `90b56e6d`
**Base:** `1bea4cf` (MVP v2)

## Status

**6 of 6 CLI claims verified passing.** Zero BLOCKERs. 11 commits since the base.

## Commit chain

```
90b56e6d  fix(monitor): close 3 silent-failure paths in QNN op-tracing data flow
7fae1b30  fix(session): SessionOptions mutation safety + cuda/tensorrt taxonomy gap
ec777caa  fix(winml): symmetric defensive register guard — completes Gap #1 fix
eb37f6c3  fix(session): WinMLEPRegistry defensive against dual-singleton DLL register
7c64ca23  fix(session): WinMLSession.compile() actually runs ORT ModelCompiler — Gap #3
e70c2a20  fix(consolidation): complete plan §C, §5, §6 — audit follow-up
62807ac9  feat(compile): winml compile --device flag + ep_device threading
0a9d422a  refactor(session): _SHORT_TO_CANONICAL -> _SHORT_TO_FULL naming
3b155784  refactor(session): consolidate EP/device taxonomy under session/ facade
db39b80d  feat(session): op-tracing perf monitor + EPDevice/WinMLSession refactor
1bea4cf8  feat: MVP v2 (base)
```

## CLI verification matrix (re-confirmed at this HEAD)

| # | Command | Result |
|---|---|---|
| 1 | `winml perf -m <fp32>.onnx --ep qnn --device npu` | EXIT 0, 2.63ms avg |
| 2 | `winml perf -m <fp32>.onnx --ep qnn` (device deduced) | EXIT 0, 2.27ms avg |
| 3 | `winml perf -m <fp32>.onnx --device npu` (ep deduced) | EXIT 0, 2.35ms avg |
| 4 | `winml compile -m <fp32>.onnx --ep qnn --device npu` | EXIT 0, `*_qnn_ctx.onnx` + `*_qnn.bin` produced |
| 5 | `winml perf -m <ctx>.onnx --ep qnn --device npu` | EXIT 0, 2.27ms avg |
| 6 | `winml perf -m microsoft/resnet-50 --ep qnn --device npu` | EXIT 0, full HF auto-build pipeline runs end-to-end |

Source: `docs/design/session/2026-05-13-cli-claims-reverify.md`.

---

## Remaining issues, organized by severity

### 🟡 IMPORTANT — defer to follow-up PR

#### I1. Consolidate the two EP-registration singletons

**Current state:** Two singletons (`WinMLEPRegistry` and `winml.py:WinML`) both call `ort.register_execution_provider_library`. Symmetric defensive guards in both (commits `eb37f6c3` + `ec777caa`) prevent the double-register crash that surfaced as exit 127 / `STATUS_DLL_NOT_FOUND`.

**Why deferred:** The patch covers user-visible behavior. The proper fix is to have ONE singleton be the canonical registrar (probably `WinMLEPRegistry`) and have everything else consume via its public surface. That refactor would change `winml.py:WinML.register_execution_providers()` signature and risk touching `analyze/runtime_check_rules` callsites that we haven't audited.

**Trigger to revisit:** When `analyze` is refactored, or when `winml.py:WinML` accumulates new responsibilities, or when we see a third singleton emerging.

**Affected files:**
- `src/winml/modelkit/winml.py:WinML.register_execution_providers()`
- `src/winml/modelkit/session/ep_registry.py:WinMLEPRegistry.register_ep()`
- Any analyze code that calls `winml.register_execution_providers()`

#### I2. Spec drift — bump design doc to v1.3

`docs/design/session/2026-05-11-ep-device-refactor.md` (v1.2) doesn't reflect what actually shipped:

- `WinMLSession.perf()` is a `@contextmanager` not a regular method.
- `session/ep_device.py` has a `_get_ep_registry()` lazy-import shim to dodge a circular import (`ep_registry` imports `ep_device`).
- `ep_to_device(ep_name)` helper added (not in original spec but consumed by `compile.py` display logic).
- `WinMLEPRegistry.register_ep()` has a bundled-EP fallback for `CPUExecutionProvider` (commit `9cce0163` rolled into the squashed `db39b80d`).
- `short_ep_name`/`expand_ep_name` vocabulary uses "full" not "canonical" (commit `0a9d422a`).
- Symmetric defensive guards across two singletons (commits `eb37f6c3` + `ec777caa`).
- `cuda` / `tensorrt` short→full mappings added (commit `7fae1b30`).

**Why deferred:** Documentation, not code. Doesn't block shipping. Best done as a single pass after the PR merges so the spec matches main.

---

### 🟢 MEDIUM — optional in this PR

#### M1. `winml analyze` slow probing (cmd 2 timeout)

**Symptom:** `winml analyze -m <fp32>.onnx --optim-config <path>` runs the runtime-checker on every ONNX node when the rule zip is missing. ConvNeXt has 667 nodes → 30+ min timeout observed in CLI verification.

**Root cause:** Rule zip `QNNExecutionProvider_npu_ai.onnx_opset17.zip` is missing from the package; analyze falls back to per-node probe sessions. Each probe creates an `ort.InferenceSession` per node (× 667).

**Mitigations:**
- **A. Ship the rule zip with the package** — proper fix. Avoids the fallback entirely.
- **B. Dedup probes by unique op-type** — ConvNeXt has 667 nodes but only ~11 distinct op types. Probing each unique type once → 60× speedup on the fallback path.

**Effort:** B is in our code (~10 lines + test). A requires package-build coordination.

**Affected files:** `src/winml/modelkit/analyze/runtime_check_rules/` (or wherever the runtime checker lives).

#### M2. `CompileStage._build_provider_options` dead method

**Symptom:** Method on `CompileStage` class that's never called from `process()`. Gives false impression of where provider options are applied.

**Source:** Audit (group 4 per-file review of `compile.py`).

**Fix:** Trivial deletion.

**Affected files:** `src/winml/modelkit/compiler/stages/compile.py`.

#### M3. `models/auto.py` positional `ep_device` audit

**Symptom:** The audit (group 4 per-file review) noted that adding `ep_device` as 2nd positional parameter on `WinMLAutoModel.from_pretrained` / `from_onnx` means any caller passing `task=`/`config=`/`WinMLBuildConfig` in position 2 silently rebinds it. No static type guard.

**Action:** Audit all callsites of these two methods. Confirm every caller uses keyword args for everything past position 1, OR converts to keyword args.

**Affected files:** Greps across `src/winml/modelkit/` for `WinMLAutoModel.from_pretrained(` and `WinMLAutoModel.from_onnx(`.

---

### ⚪ Out of scope (upstream or separate branch)

#### O1. Native QNN HTP AOT crashes on QDQ-quantized graphs

**Symptom:** `ort.ModelCompiler.compile_to_file()` crashes natively (no Python traceback) when fed an INT8 QDQ-wrapped graph that contains certain op patterns (specifically: QDQ-wrapped `Gemm` in `microsoft/resnet-50`).

**Confirmed not our code:** Direct FP32 → CTX compile works (T4 ✓). The crash is inside QNN HTP's offline compiler DLL.

**Workaround:** Use FP32 → CTX compile path; skip the quantize step for models with unsupported ops. Document the limitation in user-facing docs.

#### O2. `feat/update-pkg-deps` integration

**Status:** Separate branch on `gh/feat/update-pkg-deps`. Provides:
- `canonicalize_ep_name()` proper implementation (we have a casing-stub).
- `MODELKIT_EP_PATH` env-var override (we don't implement; we inherit if upstream lands first).
- `EP_DLL_NAMES`, `EpSource` discovery shim.
- 5-EP support (QNN, OpenVINO, VitisAI, MIGraphX, NvTensorRtRtx).

**When they merge:** Rebase our branch on top. Replace our stub `canonicalize_ep_name` with `from .ep_path import canonicalize_ep_name`. Delete our `_EP_NAME_ALIASES` stub.

---

## Process work before ship

| Step | Status |
|---|---|
| Force-push to `gh/feat/op-tracing-refactor` (remote diverges due to soft-reset rebase) | Pending — user discretion |
| Update GitHub PR description (PR body reflects pre-squash 112-commit chain) | Pending |
| Final full pytest gate: `uv run pytest tests/unit/ --tb=short -q` | Pending |
| Rebase onto `feat/update-pkg-deps` when it merges to main | Pending — blocked on their merge |

---

## Companion docs (for cross-reference)

| Doc | What it covers |
|---|---|
| `docs/plans/2026-05-13-ep-taxonomy-consolidation-plan.md` | The consolidation plan (Phase 1 + Phase 2) — executed in commits `3b155784`, `62807ac9` |
| `docs/plans/2026-05-13-post-blocker-cleanup-plan.md` | The post-BLOCKER cleanup plan — Bundles A + B + C1 executed; C2 + D outstanding |
| `docs/design/session/2026-05-12-review-summary.md` | Consolidated per-file review findings (9 issues across 25 reviewed files) |
| `docs/design/session/2026-05-12-impl-status.md` | Implementation audit vs. spec |
| `docs/design/session/2026-05-12-ep-taxonomy-sweep.md` | 47 EP/device taxonomy findings (addressed in Phase 1 + audit follow-up) |
| `docs/design/session/2026-05-13-consolidation-audit.md` | Audit of the consolidation commits — 3 issues, all fixed in `e70c2a20` |
| `docs/design/session/2026-05-13-gap1-diagnostic.md` | Original Gap #1 root-cause investigation |
| `docs/design/session/2026-05-13-t6-analyze-crash-diagnostic.md` | The follow-up diagnostic that found the asymmetric guard issue |
| `docs/design/session/2026-05-13-cli-claims-reverify.md` | Fresh CLI re-verification (5/6 pass + T6 found broken before symmetric guard) |
| `docs/design/session/2026-05-12-code-review/` | 24 per-file review docs from the initial code-review pass |
| `docs/design/session/2026-05-12-cli-verification.md` | First CLI verification run (cmd 1 ✓ / 3 / 4 ✓; 2 timeout, 5 / 6 failed before BLOCKER fixes) |

## Summary table

| Severity | Count | What |
|---|---|---|
| 🔴 BLOCKER | 0 | — |
| 🟡 IMPORTANT (deferred follow-up) | 2 | Consolidate singletons (I1); Spec drift v1.3 (I2) |
| 🟢 MEDIUM (optional this PR) | 3 | Analyze slow probing (M1); `_build_provider_options` dead method (M2); `models/auto.py` positional audit (M3) |
| ⚪ OUT OF SCOPE | 2 | QNN HTP AOT QDQ crash (O1); `feat/update-pkg-deps` integration (O2) |
| 📋 PROCESS | 4 | Force-push, PR description, final pytest, rebase-on-deps |

## Next moves (user choice)

1. **Quick polish:** knock out the 3 MEDIUMs in 1-2 small commits (~30 min). Then ship.
2. **Ship now:** skip MEDIUMs, force-push, update PR description, request review.
3. **Pause for review:** read the 11-commit chain + this doc before deciding.
