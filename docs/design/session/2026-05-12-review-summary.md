# Code Review Summary — 2026-05-12

**Branch:** `feat/op-tracing-refactor` at `db39b80d` (squashed single commit on `1bea4cf`).
**Scope of review:** 25 changed `*.py` files in `src/winml/modelkit/` since the rebase base.
**Companion docs:**
- Per-file reviews: `docs/design/session/2026-05-12-code-review/` (24 files; one per Python file reviewed)
- CLI verification: `docs/design/session/2026-05-12-cli-verification.md`
- Taxonomy sweep (in flight): `docs/design/session/2026-05-12-ep-taxonomy-sweep.md`
- Impl audit (earlier): `docs/design/session/2026-05-12-impl-status.md`

---

## 1. Issues by severity

### BLOCKER

| File:Line | Issue |
|---|---|
| `models/auto.py:163, 196, 355` | HF auto-build path: legacy `device=` string still flows through `WinMLAutoModel.from_pretrained` → `winml_class(..., device=device)`. Should `TypeError` at runtime against the new `WinMLPreTrainedModel(ep_device=...)` signature. **However**, CLI verification's cmd 6 (`winml perf -m microsoft/resnet-50 --ep qnn --device npu`) exit 127 differs from the predicted TypeError — suggests the migration is more complete than the audit assumed. Still worth verifying. |
| `compiler/stages/compile.py:17` | Cross-package import of private `_EP_TO_DEVICE` from `config/precision.py` violates CLAUDE.md import rules; the device-from-EP guess (`_EP_TO_DEVICE[ep_str]`) regresses to autoep behavior the spec promised to delete. Fix: thread `device` through `CompileContext`; require `--device` on `winml compile` CLI. |
| `compiler/stages/compile.py` (via `WinMLSession.compile()`) | `winml compile` exits 1 with no output file — the legacy `WinMLSession._build_session_options` instance method (still used by `compile()`) does not wire QNN EPContext options. Audit Gap #3 / Task 8 bridge. Blocks every downstream consumer of compiled artifacts. |

### IMPORTANT

| File:Line | Issue |
|---|---|
| `session/session.py:172` (`_build_session_options` free function) | Mutates caller-supplied `ort.SessionOptions` in-place via `add_session_config_entry`. If ORT's `add_session_config_entry` is not idempotent for repeated same-key calls, monitor session-config entries will accumulate across `perf()` windows rather than being applied once. Requires ORT behavior confirmation or copy-on-use semantics. |
| `session/session.py:462-485` (`WinMLSession._build_session_options` method) | Legacy instance method survives with `set_provider_selection_policy(PREFER_NPU)`. Called by `compile()` (lines 315, 336), `is_compatible()`, `WinMLQairtSession._create_inference_session()`. Defeats the deterministic-binding goal. Has 3 `TODO Task 8/11` markers. |
| `models/auto.py:from_pretrained` | `ep_device` added as second positional parameter. Any caller passing `task=`/`config=`/`WinMLBuildConfig` in position 2 silently rebinds to `ep_device` — TypeError at first session construction, not at parameter parse. No static type guard. |
| `config/precision.py:97 / compile.py:69` | `cuda` and `tensorrt` are in `VALID_EPS` + `_EP_TO_DEVICE` (precision passes them as valid), but NOT in `ep_device._SHORT_TO_CANONICAL`. `expand_ep_name("cuda")` returns `"cuda"` (passthrough), then `register_ep("cuda")` raises `EPNotDiscovered`. Confusing UX: "valid at precision-resolution, unresolvable at session-creation." |
| Brief: cmd 2 hang at `winml analyze` | Analyze runtime-checker runs unconditionally when the rule zip is missing. 667 ConvNeXt nodes × probing → 30+ min runtime. `--optim-config` does NOT gate this (only gates final save). Per CLI verification doc recommendation #3. |

### MEDIUM

| File:Line | Issue |
|---|---|
| `session/monitor/qnn_monitor.py:439-441` | `int(meta.get("accel_execute_cycles", 0) or 0)` truncates float strings. If QNN ever emits `"12345.6"`, `cycle_to_us` ratio is wrong → all `duration_us` systematically wrong, no warning. Use `round(float(...))`. |
| `session/monitor/qnn/_internal.py:311, 342-355, 402-403` | Hard `dict[key]` access × 14 in QHAS parser. A single QNN SDK key rename → all detail-mode profiles silently degrade to `status="basic_fallback"` via the outer `except Exception` in `_try_qhas`. Diagnostic signal swallowed. |
| `commands/perf.py:1577-1605` | Benchmark JSON written before op-trace status check. If `trace_result.status == "no_data"`, exit 4 but JSON artifact is on disk. CI sees failed exit + a usable JSON — dual outcome that needs explicit handling. |
| `commands/perf.py:472, 1552 + eval/evaluate.py:138` | `_default_ep_for_device = {"cpu": "cpu", "npu": "qnn", "gpu": "dml"}` duplicated inline at 3 sites. Adding a target requires 3 edits with no compile-time enforcement. Should be a named constant. |
| `compiler/stages/compile.py` | `CompileStage._build_provider_options()` method is dead code — never called from `process()`. Gives false impression of where provider options are applied. |

### LOW

| File:Line | Issue |
|---|---|
| `session/monitor/qnn/_internal.py:167-185` | Silent pre-boundary row loss in QNN CSV parser. If file starts with NODE SUB-EVENT rows before the first ROOT `Accelerator (execute) time (cycles)` boundary (malformed/truncated), rows are silently dropped. Result: `status="ok"` with empty operator list rather than `status="no_data"`. False success signal. |

---

## 2. CLI verification results (from `2026-05-12-cli-verification.md`)

| # | Command | Result | Root cause |
|---|---|---|---|
| 1 | `winml export -m facebook/convnext-base-224 -o convnext/model.onnx` | ✅ PASS | — |
| 2 | `winml analyze -m convnext/model.onnx --optim-config optim.json` | ❌ TIMEOUT (~30 min) | Analyze runtime-checker probes all 667 nodes; missing rule zip + no per-op-type dedup |
| 3 | `winml optimize -m convnext/model.onnx -c optim.json` | ✅ PASS | — |
| 4 | `winml quantize -m convnext/model_opt.onnx` | ✅ PASS | — |
| 5 | `winml compile -m convnext/model_opt_qdq.onnx` | ❌ exit 1, no output | Legacy `WinMLSession._build_session_options` method doesn't wire QNN EPContext options |
| 6 | `winml perf -m microsoft/resnet-50 --ep qnn --device npu` | ❌ exit 127 (~60s) | Subprocess crash; child-process exception swallowed by `_OutputCapturingWrapper` |

**Notable**: cmd 6 exit 127 mode is *not* the audit's predicted `TypeError` from Gap #1. The HF-path migration appears more complete than the audit assumed. Diagnosis still needed for the actual exit-127 cause.

---

## 3. Per-file review docs (inventory)

24 docs under `docs/design/session/2026-05-12-code-review/`:

### Group 1 — EPDevice core (6 files, 138-184 lines each)
- `session_ep_device.md`
- `session_ep_registry.md`
- `session_session.md`
- `session_qairt_qairt_session.md`
- `sysinfo_device.md`
- `sysinfo___init__.md`

### Group 2 — Op-tracing monitor (8 files, 38-182 lines)
- `session_monitor___init__.md`
- `session_monitor_ep_monitor.md`
- `session_monitor_op_metrics.md`
- `session_monitor_qnn_monitor.md`
- `session_monitor_report.md`
- `session_monitor_qnn___init__.md`
- `session_monitor_qnn__internal.md`
- `session_monitor_qnn_viewer.md`

### Group 3 — Commands + eval (6 files, 55-342 lines)
- `commands_live_chart.md`
- `commands_pre_bench.md`
- `commands_config.md`
- `commands_eval.md`
- `commands_perf.md` (342 lines — longest)
- `eval_evaluate.md`

### Group 4 — Models + config + compiler (5 files, 93-147 lines)
- `models_auto.md`
- `models_winml_base.md`
- `config_build.md`
- `config_precision.md`
- `compiler_compile.md`

---

## 4. Pending decisions awaiting user

### Decision A — `EPDevice` + EP/device taxonomy placement

**Proposed (my recommendation):** Move to top-level `winml.modelkit.ep_device`. Consolidate the following currently-split symbols into one module:

| Currently at | Symbol | Move to |
|---|---|---|
| `session/ep_device.py` | `EPDevice`, `resolve_device`, `expand_ep_name`, `short_ep_name`, `canonicalize_ep_name`, `_SHORT_TO_CANONICAL`, `_CANONICAL_TO_SHORT`, `_EP_NAME_ALIASES`, 5 exceptions | `ep_device.py` (top-level) |
| `config/precision.py` | `_EP_TO_DEVICE`, `_DEVICE_TO_PROVIDER`, `VALID_EPS`, `_VALID_DEVICES`, `get_provider_for_device` | `ep_device.py` (top-level) |

**What stays put:**

| Symbol | Location | Why |
|---|---|---|
| `WinMLEPRegistry` | `session/ep_registry.py` | Session-lifecycle bound; registers/unregisters with ORT during compile |
| `resolve_device_category` | `sysinfo/device.py` | Different responsibility — host-perspective system inventory (WMI-based) |
| `_NAMED_PRECISIONS`, `PrecisionPolicy`, `resolve_precision` | `config/precision.py` | Genuinely precision-domain; consumes the ep_device taxonomy but adds precision logic on top |

### Decision B — `winml compile` CLI `--device` requirement

Should `winml compile` require explicit `--device {cpu,gpu,npu}` (matches Option A hard break), or accept a per-EP default (e.g., qnn→npu, dml→gpu, cpu→cpu)?

Currently the CLI accepts `--ep` only. Adding `--device` is necessary to fix the cross-package private import in `compile.py` properly (rather than guessing via `_EP_TO_DEVICE`).

---

## 5. Recommended next steps

1. **User decision** on §4 above.
2. **EPDevice taxonomy consolidation** — dispatch refactor agent after the in-flight sweep doc returns.
3. **Fix compile pipeline** — add `--device` CLI flag, thread through `CompileContext`, drop `_EP_TO_DEVICE` import from `compile.py`.
4. **Fix BLOCKERS** — legacy `_build_session_options` instance method (audit Gap #3); ResNet-50 exit 127 root-cause (likely auto.py migration verification needed).
5. **Bookkeeping IMPORTANT items** — `add_session_config_entry` idempotency, `cuda`/`tensorrt` taxonomy gap, analyze hang on missing rule zip.
6. **MEDIUM cleanup** — extract `_default_ep_for_device` to consolidated module, `int()`→`round(float())` in `qnn_monitor.py`, robustness in `qnn/_internal.py`, JSON-before-status-check ordering in `perf.py`.
7. **LOW** — defer.
