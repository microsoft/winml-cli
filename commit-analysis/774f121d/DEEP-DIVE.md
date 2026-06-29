# Commit 774f121d — Deep-Dive: Implementation vs Design

> **Fact-check pass applied (2026-06-28).** This DEEP-DIVE was independently verified against actual source by 5 verification agents (reports at `verification/batch-{00..04}.md`); the consolidated verdict is at [`FINAL-VERDICTS.md`](FINAL-VERDICTS.md). All four 🔴 regression claims (D1, D2, D3, D4) survived scrutiny. **R6 and R8 (qairt "fragile import"; `WinMLDevice.ort_handle` "unused") were retracted as FALSE** — see their rows in the table at §"Q-numbered open questions" for the inline retraction.

## How to read this doc

This is the opinionated cross-reference of commit `774f121d`
("feat(session): v2.9 unified-source EP refactor + WinMLSession redesign"). Three upstream artifacts are assumed in front of the reader: the v2.9 design corpus under `docs/design/session/` (1_req.md, 2_coreloop.md, 3_design_classes.md, 3_design_ep.md, 4_winml_device.md, 5_type_taxonomy.md — all touched in this commit), the SUMMARY.md narrative of what shipped, and the 81 per-file diff analyses under `commit-analysis/774f121d/per-file/`. This doc does **not** recap any of that material; it judges the gap between what was designed and what was shipped, names the load-bearing regressions, and identifies the simplification debt the squash deferred.

The headline finding is not architectural. The architecture is sound: the v2.9 design (unified `BuiltinSource` synthesis, six-class taxonomy with `EPDeviceTarget`/`WinMLEP`/`WinMLEPDevice`, atomic-vs-compound registry split, single-class `WinMLDevice` adapter with internal dispatch) lands faithfully. The headline finding is that the squash shipped **four runtime regressions** — two import-time `NameError`/`ImportError` (the `quantize=True` `warnings` path; the dead `compiler/cli.py` sub-CLI), one silent UX regression for QNN SDK developer boxes (lost `_COMMON_SDK_PATHS` auto-discovery), and one silent perf regression for the bundled-ORT QNN HTP path (lost `htp_performance_mode=high_performance` + `enable_htp_fp16_precision=1` defaults). The architectural wins are large; the regressions are mechanical and entirely fixable in one follow-up.

## Methodology

The cross-reference reads from three corners simultaneously: (1) the v2.9 design corpus (`1_req.md` v1.2, `2_coreloop.md` v2.9 1200-line spec including the unified `BuiltinSource` rationale, `3_design_classes.md` v1.2 canonical class table, `3_design_ep.md` v1.0 Stage 1/Stage 2 partition, `4_winml_device.md` v1.5 device-vs-EP attribute split, and `5_type_taxonomy.md` v2.2 superseded-stub) — for what was *promised*; (2) the 81 per-file diff analyses and the live source at `src/winml/modelkit/` — for what *shipped*; and (3) targeted grep verification for the four shipped regressions (verified: `import warnings` is absent from `compiler/configs.py` but eight `warnings.warn(...)` call sites are present at lines 161/173/187/201/215/229/243/257; `compiler/cli.py` lines 15+17 still import `CalibrationConfig` and `QDQConfig` from `.configs`; `_COMMON_SDK_PATHS`/`backend_path`/`htp_performance_mode` are zero-hit greps in `session/monitor/qnn/`). Confidence is highest on the four confirmed regressions (mechanical, grep-verified) and on the design-vs-impl architectural alignment (the v2.9 corpus explicitly anticipates the major shipped shapes). Confidence is lower on the test-coverage critique (no test runs in the analysis; the regression-detection is by code-review only) and on the cross-EP runtime claims (no QNN, no VitisAI, no OpenVINO hardware on the analysis host).

## Headline: where impl diverges from design

The divergences below are ranked by **practical blast radius** — how likely a future contributor or end-user stumbles into the gap, weighted by how badly they get hurt. The four 🔴 regressions head the list; the architectural drifts and improvements follow. All four 🔴 items are mechanical fixes, not design rethinks.

---

### D1. 🔴 `compiler/configs.py` calls `warnings.warn` from eight factories without importing `warnings` — `NameError` at runtime

- **Divergence title:** Eight `for_*` factories (`for_qnn`, `for_cpu`, `for_cuda`, `for_dml`, `for_nv_tensorrt_rtx`, `for_openvino`, `for_vitisai`, `for_migraphx`) emit a `DeprecationWarning` via `warnings.warn(...)` when called with the legacy `quantize=` kwarg. The squash deleted `import warnings` from the top of `compiler/configs.py` but left the eight call sites intact. Calling any factory with `quantize=True` raises `NameError: name 'warnings' is not defined`.
- **What design said:** Nothing direct. The design corpus does not specify the deprecation-warning contract of the `for_*` factories. The v2.9 corpus does, however, position `WinMLCompileConfig.for_ep_device(...)` as the canonical factory and treats `for_qnn` / `for_cpu` / etc. as legacy surface (per `compiler/configs.py`'s own TL;DR comment about consolidation).
- **What impl shipped:** `src/winml/modelkit/compiler/configs.py` has eight `warnings.warn(...)` call sites at lines 161, 173, 187, 201, 215, 229, 243, 257. Grep for `^import warnings|^from warnings` in the same file returns zero matches. The deprecation path is dormant in CI (the test suite's `test_quantize_false_emits_deprecation[for_*]` is failing per the brief) because current callers pass `quantize=None`, but the bug is lethal for any user passing the legacy kwarg — and is exactly the surface the deprecation warning was meant to gently steer.
- **Judgment:** **Regression.** A clean one. The deprecation path is the *only* path the warning fires on; the only signal the codebase emits to legacy callers is a `NameError` instead of a deprecation hint.
- **Why:** Squash-merge collateral. The `compiler__configs.md` per-file analysis names this directly: "removed `import warnings`" was paired with `from typing import Any` removal (the latter is saved by `from __future__ import annotations` stringifying type hints, but `warnings` is referenced at runtime). The dead `_EP_CONTEXT_DEFAULTS: Final[frozenset[str]] = frozenset({"qnn", "openvino"})` constant at module-level (also added by this squash, never read) is a sibling indicator of an unfinished consolidation pass — the author was clearly mid-refactor when they removed `warnings`, intending to drop the eight `for_*` factories in favour of a single `for_provider`-driven dispatcher consuming `_EP_CONTEXT_DEFAULTS`, but shipped the half-state.
- **Recommendation:** **Pre-merge fix.** One-line `import warnings` add, OR finish the consolidation: collapse the eight `for_*` factories into one `for_provider` factory that consumes `_EP_CONTEXT_DEFAULTS` for the `enable_ep_context` decision; delete the deprecated `quantize=` kwarg entirely (per the `MEMORY.md` no-back-compat preference, deprecation shims are not the project's style). Either fix unblocks the failing tests.

---

### D2. 🔴 `compiler/cli.py` imports `CalibrationConfig` and `QDQConfig` from `.configs` — neither symbol exists; sub-CLI is dead-on-arrival

- **Divergence title:** `compiler/cli.py` at line 15 and line 17 imports `CalibrationConfig` and `QDQConfig` from `.configs`. The same commit moved both classes to `winml.modelkit.quant.config` (per `configs.py`'s docstring on the quant-config split). `python -m winml.modelkit.compiler compile ...` fails with `ImportError: cannot import name 'CalibrationConfig' from 'winml.modelkit.compiler.configs'` at module-load time, before any command runs.
- **What design said:** The design corpus does not address the sub-CLI's existence at all. The top-level `winml compile` CLI in `commands/compile.py` is the supported path per `2_coreloop.md` §9 CLI Surface Mapping; `compiler/cli.py` is undocumented legacy.
- **What impl shipped:** `src/winml/modelkit/compiler/cli.py` lines 14-17 (per the per-file analysis): `from .configs import (CalibrationConfig, EPConfig, QDQConfig, WinMLCompileConfig)`. Lines 179-196 use the removed symbols (`QDQConfig(...)`, `CalibrationConfig(...)`, `WinMLCompileConfig(qdq_config=..., calibration_config=...)`). The file is non-functional after this squash; a direct `python -m` invocation aborts before parsing any flag.
- **Judgment:** **Regression** (but in dead code). The sub-CLI was probably already unreachable from the top-level `winml` entry point, but it shipped broken — anyone running the sub-CLI for a smoke test or anyone who imports the module sees an immediate `ImportError`. Tests that exercise `compiler.cli` would fail at collection time.
- **Why:** Same root cause as D1: the quant-config split (moving `CalibrationConfig` / `QDQConfig` to `winml.modelkit.quant.config`) was done in `configs.py` but the sub-CLI wasn't updated. The widening of `--ep` to `sorted(VALID_EPS)` (the one productive change in this file) is correct but landed alongside an `ImportError` that masks the win.
- **Recommendation:** **Pre-merge fix.** Two options: (a) delete `compiler/cli.py` entirely (per the per-file analysis's TL;DR: "the cleanest move is to delete `OpenVINOMonitor` and `openvino_monitor.py`"-style logic applies — the sub-CLI duplicates `commands/compile.py` semantically and has no supported path), or (b) update the imports to `from ..quant.config import CalibrationConfig, QDQConfig` and strip the quant-related kwargs from the `WinMLCompileConfig(...)` call. Option (a) is the project's stated style (`MEMORY.md`: hard-break refactors, no compat shims).

---

### D3. 🔴 `find_qnn_sdk()` lost the `_COMMON_SDK_PATHS` fallback — QNN SDK auto-discovery silently dropped

- **Divergence title:** The pre-state `find_qnn_sdk()` in `optracing/qnn/viewer.py` checked `QNN_SDK_ROOT`, then walked a hardcoded `_COMMON_SDK_PATHS = [r"D:\QC", r"C:\Qualcomm\AIStack\qairt"]` list and returned the highest-versioned child containing `bin/`. The post-state `find_qnn_sdk()` in `session/monitor/qnn/viewer.py` is env-var-only: no `_COMMON_SDK_PATHS` constant, no version-sorted directory walk. Dev boxes with QNN SDK installed at `D:\QC\<version>` or `C:\Qualcomm\AIStack\qairt\<version>` but no `QNN_SDK_ROOT` set silently drop from detail mode to basic CSV after this squash. Grep for `_COMMON_SDK_PATHS|backend_path|htp_performance_mode` in `session/monitor/qnn/` returns zero hits — confirmed absent.
- **What design said:** The design corpus is silent on QNN SDK auto-discovery — the topic isn't covered. The closest design touchpoint is `2_coreloop.md` §5.5 (built-in vs plugin dispatch), which mentions QNN as a plugin EP but does not specify the SDK runtime discovery contract.
- **What impl shipped:** `src/winml/modelkit/session/monitor/qnn/viewer.py` — a pure `QNN_SDK_ROOT` reader. Failure returns `None`; the caller (`QNNMonitor._try_qhas` at `qnn_monitor.py:558`) treats `None` from `find_qnn_sdk` as "skip QHAS" → status downgrades to `basic_fallback`. The warning message ("set QNN_SDK_ROOT to enable detail mode (falling back to basic CSV)") is unchanged, which masks the regression: a user previously running detail mode via auto-discovery now sees the same warning they always saw but their detail mode actually stopped working.
- **Judgment:** **Regression** (silent UX). Quiet behavior change for the QNN developer machine fleet — the kind of thing that surfaces as "why did my CI suddenly start emitting `basic_fallback`?" three weeks after merge.
- **Why:** Per the per-file analysis: "Was this intentional (env-var-only is the supported contract) or accidental?" The relocate commit collapsed `optracing/qnn/viewer.py` into `session/monitor/qnn/viewer.py` with a -24/+21 diff concentrated in `find_qnn_sdk()`'s resolution policy. Either the author wanted env-var-only as a hardening move (no hardcoded Windows paths in source) or the auto-discovery accidentally fell off during the relocate. The intent is undocumented.
- **Recommendation:** **Pre-merge decision.** If intentional: document the breaking change explicitly in the commit body and the v2.9 design doc; bump the warning message to "QNN_SDK_ROOT not set — detail mode falling back to basic CSV (auto-discovery was removed in v2.9; set the env var to re-enable)". If accidental: restore `_COMMON_SDK_PATHS = (r"D:\QC", r"C:\Qualcomm\AIStack\qairt")` and the version-sorted walk. Either way the silent regression must close.

---

### D4. 🔴 QNN HTP provider-option defaults dropped — bundled-ORT users lose `backend_path` / `htp_performance_mode` / `enable_htp_fp16_precision` with no migration note

- **Divergence title:** The pre-state `QNNProfiler._build_provider_options` unconditionally set `backend_path=QnnHtp.dll`, `htp_performance_mode=high_performance`, `htp_graph_finalization_optimization_mode=3`, `enable_htp_fp16_precision=1`. The post-state `QNNMonitor.get_provider_options()` sets **none** of these. Bundled-ORT users must now supply them via `extra_provider_options=`; WinML-ORT users get WinML's tuned defaults preserved. Grep for `backend_path|htp_performance_mode` in `session/monitor/qnn/` returns zero hits — confirmed absent.
- **What design said:** The `EP_DEVICE_SPECS` catalog in `session/ep_device.py` (per `2_coreloop.md` §4 and `3_design_ep.md` §6.1) ships a QNN/NPU row with `default_provider_options = {"htp_performance_mode": "burst", "htp_graph_finalization_optimization_mode": "3"}` — but this is the *session-level* catalog default, applied to **all** QNN sessions via `_ep_defaults(ep_device)` in `session/session.py:85-101`. The monitor-level defaults that the old `QNNProfiler` carried (HTP `high_performance` vs catalog's `burst`; `backend_path=QnnHtp.dll`; `enable_htp_fp16_precision=1`) are not in any design doc. The four missing defaults are an undocumented migration.
- **What impl shipped:** `QNNMonitor.get_provider_options()` in `session/monitor/qnn_monitor.py` returns only `{profiling_level: <level>, profiling_file_path: <path>}` merged on top of `self._extra` (the caller-supplied `extra_provider_options`). The docstring justifies the omission: WinML's pre-tuned absolute `backend_path` would be overwritten if the monitor stamped `QnnHtp.dll`. But the docstring does not call out the bundled-ORT path's loss.
- **Judgment:** **Regression** (silent perf). For users on the `onnxruntime-qnn` bundled wheel running op-tracing, every previously-tuned HTP run now ships with default values (whatever bundled ORT defaults to, which the per-file analysis says is something *other* than `high_performance` / `enable_htp_fp16_precision=1`). The `burst` mode in the session catalog *does* fire for non-op-tracing sessions (correct), but the op-tracing path inherits the bundled-ORT default rather than `high_performance` because the catalog only applies when no monitor merges over it... actually re-reading `_build_provider_options` in `session/session.py:104-125`: catalog applies *first*, then user config, then monitor. So `burst` from the catalog does fire for op-tracing too. The genuine loss is `backend_path=QnnHtp.dll` (which the monitor used to set on bundled-ORT; WinML-ORT handles backend_path differently) and `enable_htp_fp16_precision=1` (not in catalog).
- **Why:** Per the per-file analysis: "those four HTP defaults must now be supplied by the caller via `extra_provider_options` if running the bundled-ORT path." The architectural choice to make `QNNMonitor` neutral on backend-routing keys is correct (`3_design_ep.md` §4 documents that routing keys are ignored under the plugin-EP API; setting `backend_path` is the kind of legacy footgun that crashed ORT 1.23.5 with `exit(127)` per commit `a509a67`'s SD2 finding). But the *tuning* keys (`htp_performance_mode`, `enable_htp_fp16_precision`) are also gone, which is over-aggressive — those are not routing keys.
- **Recommendation:** **Pre-merge fix or doc.** Option (a) add a `QNNMonitor.for_bundled_ort(level)` factory that pre-populates the lost HTP defaults — clean migration story per the per-file analysis's "no factory like this exists" complaint. Option (b) move the four lost defaults into `EP_DEVICE_SPECS[QNN/NPU].default_provider_options` so the catalog covers both monitor and non-monitor paths — `backend_path=QnnHtp.dll` is risky here because catalog defaults apply universally and would re-introduce the ORT 1.23.5 crash on WinML-ORT. Option (c) leave shipped behavior and add a deprecation block in the commit body. Option (a) is the cleanest.

---

### D5. Pluggable-tracer registry dropped — `register_tracer` hook gone; new monitors require editing `commands/perf.py`

- **Divergence title:** The pre-state `optracing/registry.py` exposed `register_tracer(pattern, level, tracer_cls)` so third-party code could register a new tracer by EP-name substring pattern + level. The post-state has no extension hook: `commands/perf.py:_resolve_ep_monitor` is an explicit `if/elif` ladder with hardcoded `qnn` / `vitisai` arms (OpenVINO falls through to `RuntimeError`).
- **What design said:** The design corpus does not specify an extension mechanism for new monitors. `2_coreloop.md` §9 lists the CLI surface; `3_design_classes.md` §3.5 lists the four in-tree monitor subclasses but does not address third-party extensibility.
- **What impl shipped:** `src/winml/modelkit/commands/perf.py:_resolve_ep_monitor` (lines 117-187 per the per-file analysis). Per `optracing____init__.md` "Where the functionality moved": *"`get_tracer`, `register_tracer` — **Dropped entirely.** Substring-pattern registry replaced by explicit `_resolve_ep_monitor()` dispatch in `commands/perf.py`."*
- **Judgment:** **Drift** (acceptable for v2.9). The registry pattern was over-engineered for a 2-real-monitor world. The explicit dispatch is simpler and easier to follow. But the door for third-party EP monitors (a CUDA monitor, a TensorRT monitor) is now closed — those would require editing a CLI module, which is the wrong layering.
- **Why:** The author chose ergonomics over extensibility. The substring-pattern registry was indeed a complexity tax for what is currently a 2-arm dispatch. But the deletion is a one-way door: re-adding extensibility later means re-introducing a registry layer with the same complexity.
- **Recommendation:** **Defer.** Note the trade-off in the v2.9 design corpus (`2_coreloop.md` §10 Open Questions is the right place). If a third real monitor ever lands (CUDA/TRT) and is not in-tree, revisit. For now the explicit dispatch is fine.

---

### D6. `OpenVINOMonitor` is dead stub — `is_available()` returns `False`, never selected by `_resolve_ep_monitor`

- **Divergence title:** Per `session__monitor__openvino_monitor.md` TL;DR: the class is a pure no-op placeholder — `__enter__` and `__exit__` do nothing, `is_available()` always returns `False`, and `to_dict()` returns a hardcoded stub `{"ep": "OpenVINO", "device": "NPU", "status": "not_implemented"}`. The class is exported from `session/__init__.py` but **never instantiated anywhere** in the codebase: `commands/perf.py:_resolve_ep_monitor` has explicit arms for `qnn` and `vitisai` only; OpenVINO falls through to `RuntimeError("Op-tracing not available for EP 'openvino' on device {device!r}...")`.
- **What design said:** The design corpus does not specify OpenVINO op-tracing support. The PRD-style approach is "placeholder for parity" (per `commands/perf.py` commit body).
- **What impl shipped:** `src/winml/modelkit/session/monitor/openvino_monitor.py` (49 lines, identical to pre-state except `EPMonitor → WinMLEPMonitor` rename and `typing_extensions → typing` migration). The cosmetic-only diff perpetuates the dead class.
- **Judgment:** **Drift** (clean up debt). The class is dead weight. It exports one name, anchors one docstring-citation chain through `ep_monitor.py`, and provides no functionality.
- **Why:** Placeholder pattern carried forward across the v2.9 rename. The cleanest move is deletion until a real implementation lands; the second cleanest is to wire it through `_resolve_ep_monitor` as a parallel arm to `VitisAIMonitor` so OpenVINO + `--op-tracing basic` produces a typed no-op result rather than a `RuntimeError`. As-shipped is the worst of both worlds.
- **Recommendation:** **Follow-up issue.** Either delete `openvino_monitor.py` + the export (3 lines) and let `_resolve_ep_monitor` fall through to a consistent unsupported-EP error, or wire it as a `NullEPMonitor`-equivalent arm in the dispatcher with a one-line WARN. Pick one.

---

### D7. Duplicate `--ep <name>@<source>` rejection blocks in `build.py` + `config.py` — extract `_reject_ep_source(ep, command_name)`

- **Divergence title:** `commands/build.py` and `commands/config.py` each carry a verbatim-identical seven-line try/except block that rejects the `@<source-tag>` syntax. The only difference is the command name in the message. This is the strongest case in the squash for a shared helper.
- **What design said:** The design corpus (specifically `2_coreloop.md` §6.2 Scenarios A.5/A.6) specifies which commands accept source pinning (`perf`, `compile`) and which don't (`build`, `config`, `analyze`). The design does not specify how the rejection should be implemented (decorator factory, shared helper, per-command try/except).
- **What impl shipped:** Per `commands__build.md` and `commands__config.md`: both files carry the same block. The per-file analysis for `_ep_arg.py` flags this as "Simplification #2: A `EpAtSourceParamType.reject_source(ep, command_name)` companion classmethod would collapse the `build.py` / `config.py` duplicated 'source pin not allowed for this command' rejection block (currently ~7 lines per command)."
- **Judgment:** **Drift / simplification target.** Not a bug, just code that exists in two places when it should exist in one.
- **Why:** The squash landed both rejections under deadline pressure without abstracting. The decorator-factory approach (parameterize `EpAtSourceParamType(support_source=False)`) is the cleanest fix; a classmethod (`EpAtSourceParamType.reject_source(ep, command_name)`) works too.
- **Recommendation:** **Follow-up PR (small).** Add `_reject_ep_source(ep, *, command_name) -> str | None` in `commands/_ep_arg.py`; collapse both call sites to `ep = _reject_ep_source(ep, command_name="build")`. Saves ~12 lines and centralizes the message format. Trivial to add later but trivial to forget too.

---

### D8. Duplicate CPU guards: `_short if _short != "cpu" else None` in `config/precision.py` + `config/build.py`

- **Divergence title:** Both `config/precision.py` and `config/build.py` independently special-case `"cpu"` to `None` after running `default_ep_for_device → short_ep_name`. The CPU short-name is a sentinel because `compile_provider=None` means "no compile stage."
- **What design said:** The design corpus does not address this guard. The `EPDeviceSpec` catalog has a CPU/CPU row but doesn't carry a `no_compile: bool` flag to encode the semantic.
- **What impl shipped:** Per `config__precision.md`: *"the duplicated 'cpu → None' translation between this file and `config/build.py` is a smell. If a future EP gets a short name like `"cpu_arm"` or similar, this string compare won't catch it. Probably a `default_ep_for_device` should grow an `or_none_for_cpu: bool = False` parameter — or the spec for CPU could be marked 'no-compile' structurally (a `EPDeviceSpec.no_compile: bool` field)."*
- **Judgment:** **Drift.** The string compare encodes a structural fact (CPU has no compile stage) at two call sites — exactly the kind of duplication the `EP_DEVICE_SPECS` catalog was designed to eliminate.
- **Recommendation:** **Follow-up PR (small).** Add `EPDeviceSpec.no_compile: bool = False`, set `True` for the CPU/CPU row, and rewrite both call sites to consult the catalog. Or simpler: add `default_compile_ep_for_device(device) -> str | None` to `session/ep_device.py` with the CPU-to-None mapping baked in; both files import the helper.

---

### D9. Duplicate sort+slice+empty-guard in `report.py` basic and detail paths

- **Divergence title:** `_display_basic_report` and `_display_detail_report` in `session/monitor/report.py` share **three identical blocks**: head/blank lines, the 12-line sort+slice+empty-guard (defensive `sorted(operators, key=lambda op: (-op.percent_of_total, op.op_path))[:top_n]`), and the trailing `console.print(table)`. The defensive-sort comment is duplicated verbatim (8 lines) in both functions.
- **What design said:** The design corpus does not specify the report renderer's internal decomposition.
- **What impl shipped:** Per `session__monitor__report.md`: *"The sort+empty-guard is byte-for-byte identical and could collapse to a private `_topk(result, top_n) -> list[OperatorMetrics] | None` that returns `None` (caller prints the dim message + returns) or a list. That removes ~14 duplicated lines and centralizes the 'comment block about CSV vs QHAS ordering' that currently appears twice."*
- **Judgment:** **Drift / simplification target.** Not a bug, just maintenance debt. The duplicated comment block is the canary — if one comment drifts in a future edit, the two paths silently disagree on ordering semantics.
- **Recommendation:** **Follow-up PR (small).** Extract `_topk(operators: list[OperatorMetrics], top_n: int) -> list[OperatorMetrics] | None`; both `_display_basic_report` and `_display_detail_report` call it. Saves ~14 LOC + the duplicated comment.

---

### D10. 14 near-identical `_require(d, key, context)` calls in `_internal._extract_summary`

- **Divergence title:** `session/monitor/qnn/_internal.py::_extract_summary` is 14 nearly-identical `_require(raw, "<source-key>", ctx)` calls that build the renderer-facing summary dict. A small `_RENAME_MAP: dict[str, str]` (renderer-key → source-key) plus a single comprehension would halve the line count and surface the rename contract as data.
- **What design said:** The design corpus does not specify the parser's internal shape.
- **What impl shipped:** Per `session__monitor__qnn___internal.md`: *"`_extract_summary` is 14 nearly-identical `_require(raw, '<source>', ctx)` calls. A small `_RENAME_MAP: dict[str, str]` (renderer-key → source-key) plus a single comprehension `{rk: _require(raw, sk, ctx) for rk, sk in _RENAME_MAP.items()}` would halve the line count and surface the rename contract as data, not control flow. The 14 individually-named entries also make it easy to silently drop one in a future edit."*
- **Judgment:** **Drift / simplification target.** Not a bug, just a maintainability landmine — a future edit that fixes one rename and forgets another silently breaks the renderer.
- **Recommendation:** **Follow-up PR (small).** Replace the 14 hand-rolled assignments with `_RENAME_MAP` + comprehension. Single source of truth for the rename contract.

---

### D11. `_EP_CONTEXT_DEFAULTS` is dead in `compiler/configs.py` — half-finished consolidation

- **Divergence title:** `compiler/configs.py` adds `_EP_CONTEXT_DEFAULTS: Final[frozenset[str]] = frozenset({"qnn", "openvino"})` at module level, annotated as "Single source of truth — replaces the per-EP factory boilerplate that used to encode this same bit across 8 methods" but **never referenced**. The eight `for_*` factories still hand-encode `enable_ep_context=False` (or `True`) inline.
- **What design said:** The design corpus does not address `enable_ep_context` defaults at the factory level.
- **What impl shipped:** Per `compiler__configs.md`: *"A half-finished consolidation. Either delete the constant or actually use it (e.g. `cls(ep_config=EPConfig(provider=ep, enable_ep_context=ep in _EP_CONTEXT_DEFAULTS))`)."*
- **Judgment:** **Drift / simplification target.** The constant exists, the docstring promises consolidation, but the consolidation never landed. Sibling indicator of the `import warnings` regression (D1) — the author was clearly mid-refactor.
- **Recommendation:** **Bundle with D1 fix.** When fixing the `warnings` import, also finish the consolidation: collapse the eight `for_*` factories into one `for_provider` driven by `_EP_CONTEXT_DEFAULTS`. Drops ~120 LOC.

---

### D12. `utils/constants.py` is near-empty after taxonomy retirement — could be deleted entirely

- **Divergence title:** `utils/constants.py` is now a 60-line stub. The legacy second source of truth (`SUPPORTED_EPS`, `EP_ALIASES`, `ALL_EP_NAMES`, `SUPPORTED_DEVICES`, `SUPPORTED_DEVICES_WITH_AUTO`, `_get_supported_eps`) is gone. The remaining content: a 5-entry `_EP_CLI_PREFIXES` tuple (for `extract_ep_options`), a small `_short_aliases` dict (for `normalize_ep_name`), and the ORT-enum bridge maps (`DEVICE_TO_DEVICE_TYPE`, `DEVICE_TYPE_TO_DEVICE`) which still use uppercase keys — a known casing-mismatch footgun deferred to a follow-up.
- **What design said:** The design corpus (`2_coreloop.md` §10 Open Questions) flags the casing sweep + nesting cleanup as queued but unwritten. `utils/constants.py` is not addressed directly; it's a downstream artifact of the v2.9 catalog-as-truth move.
- **What impl shipped:** Per `utils__constants.md`: *"Move `DEVICE_TO_DEVICE_TYPE` / `DEVICE_TYPE_TO_DEVICE` into `session/` and lowercase the keys. Today they're the only remaining unique content in this file other than the prefix tuple — both arguably belong elsewhere. Doing so would let `utils/constants.py` be deleted entirely."*
- **Judgment:** **Simplification target / drift.** The file is nearly dead but the two enum-bridge maps' uppercase-vs-lowercase footgun is unaddressed. Either the maps lowercase and move into `session/` and the file deletes, or the file persists as a CLI-helper landing pad.
- **Recommendation:** **Follow-up PR (medium).** Migrate `DEVICE_TO_DEVICE_TYPE` / `DEVICE_TYPE_TO_DEVICE` into `session/ep_device.py` with lowercase keys; update the four callers (`analyze/runtime_checker/check_ops.py`, `analyze/pattern/check_patterns.py`, `analyze/core/runtime_checker_query.py`, `utils/cli.py`) to `.upper()` at the call site OR migrate them to lowercase too. Delete `utils/constants.py`; move `normalize_ep_name` + `extract_ep_options` into `commands/_cli_helpers.py` or directly into the consumers.

---

### D13. `_transformers_compat.py` (+304 NEW): orthogonal optimum-onnx 0.1.0 ↔ transformers 5.x compatibility shim — squash collateral

- **Divergence title:** The squash includes a 304-line side-effecting compatibility module that re-injects transformers 4.x symbols into transformers 5.x's `_LazyModule._objects` registry so optimum-onnx 0.1.0 can still be imported. Loaded once at package-init from `winml/modelkit/__init__.py`. This is *entirely orthogonal* to the v2.9 session/EP refactor.
- **What design said:** Nothing. The design corpus does not address third-party dependency compatibility shims.
- **What impl shipped:** `src/winml/modelkit/_transformers_compat.py` — 2 module-level classes (`CLIPFeatureExtractor`, `MT5Tokenizer`), 3 conditionally-defined inner classes/funcs, 1 module-level attribute (`_top_objects`), and a runtime monkey-patch to `optimum.exporters.onnx.model_patcher.sdpa_mask_without_vmap`. Per the per-file analysis, the author explicitly labels it temporary: "Drop this file (and the corresponding override in pyproject.toml) once optimum-onnx 0.2+ ships with transformers 5.x compatibility."
- **Judgment:** **Squash collateral.** Belongs in its own commit. The v2.9 session refactor and the optimum-onnx compat shim are unrelated changes; bundling them inflates the diff and obscures both reviews.
- **Why:** Either the v2.9 work happened to land on a branch that also needed the transformers 5.x bump, or the author was driving both in parallel. The result is a single commit doing two unrelated jobs.
- **Recommendation:** **Document the bundling in the commit body.** No code fix required, but the commit message should call out the orthogonal addition so future archaeology doesn't conflate the two changes. Going forward, follow the project's commit-hygiene preference (per `CLAUDE.md`'s revision conventions) to split unrelated work into separate commits.

---

### D14. Python 3.10 → 3.11 modernization — squash collateral

- **Divergence title:** The squash migrates `Self` imports from `typing_extensions` to `typing` across at least four files (`openvino_monitor.py`, `vitisai_monitor.py`, `ep_monitor.py`, others). This implies a minimum Python 3.11 (per `typing.Self` availability). This is also orthogonal to the v2.9 session refactor.
- **What design said:** Nothing. The design corpus does not specify a Python version floor.
- **What impl shipped:** Across multiple files, `from typing_extensions import Self` → `from typing import Self`. Per `session__monitor__openvino_monitor.md`: *"`Self` import migrated from `typing_extensions` to `typing`."*
- **Judgment:** **Squash collateral.** Same critique as D13. The migration is fine on its own but bundling it with the v2.9 refactor inflates the diff.
- **Recommendation:** **Document the bundling in the commit body.** Same as D13 — split unrelated work going forward.

---

## Verified design-impl alignment

Where the impl genuinely matches the v2.9 design. Parsimonious:

- **Six-class taxonomy.** `EPDeviceTarget`, `EPDeviceSpec`, `EPEntry`, `WinMLDevice`, `WinMLEP`, `WinMLEPDevice` (the flat pair) — all six classes ship at exactly the contract specified in `3_design_classes.md` §3. The `WinMLEPDevice` reassignment (was "pure intent string pair"; now "flat (WinMLEP, WinMLDevice) pair") lands cleanly with no compat shim — per `1_req.md` §3 C3 hard-break rule. (Cite: `session/ep_device.py:167-235` for `EPDeviceTarget`; `session/ep_device.py:244-303` for `EPDeviceSpec`; `ep_path.py:268-301` for `EPEntry`; `session/ep_device.py:555-737` for `WinMLDevice`; `session/ep_registry.py:112-162` for `WinMLEP` and `WinMLEPDevice`.)
- **`resolve_device(target)` is pure deduction.** Matches `2_coreloop.md` §5.3 v2.6: "Pure deduction; no DLL load, no filesystem scan, no registry I/O. `source` passes through unchanged." Verified at `session/ep_device.py:470-549`.
- **Atomic (`register_ep`) + compound (`auto_device`) registry surface.** Matches `2_coreloop.md` §5.8 lockdown. The four-method surface (`register_ep`, `auto_device`, `all_discovered`, `available_eps`) ships exactly as specified. Verified at `session/ep_registry.py:259-489`.
- **`WinMLEPRegistry.instance()` classmethod singleton.** Matches `2_coreloop.md` §5.8 "Singleton pattern" rewrite away from `__new__ + _initialized`. The new shape uses `_instance: ClassVar[WinMLEPRegistry | None] = None` and a classmethod that constructs on first call. Verified at `session/ep_registry.py:479-489`.
- **BuiltinSource synthesis at `__init__`.** v2.9's central architectural choice: built-in EPs (CPU/Dml/Azure) flow through the same `_discovered` pipeline as plugin EPs via synthesized `BuiltinSource` entries appended at registry construction. Matches `2_coreloop.md` v2.9 narrative exactly. Verified at `session/ep_registry.py:235-239` (the synthesis loop) and `ep_path.py:347-372` (the `BuiltinSource` marker class).
- **`_entry_source_tag` dispatching to canonical seven tags.** Matches `1_req.md` §R4 and `2_coreloop.md` §8.2: `bundled`, `pypi`, `nuget`, `msix-microsoft`, `msix-workload`, `winml-catalog`, `directory`. Verified at `session/ep_registry.py:75-109` for the dispatch helper and `ep_device.py:139-145` for the closed `VALID_SOURCE_TAGS` constant.
- **L1 (`failed`) / L2 (`incompatible`) status taxonomy in `sys.py:_gather_ep_info`.** Matches `2_coreloop.md` §7.1.1 two-independent-layer model. L1 fires when `register_ep` raises; L2 fires when `EP_CATALOG.is_compatible()` returns False even after successful registration. Verified per `commands__sys.md`: "EP-level 'compatible' is derived at render time from `entry[status]`" — single source of truth. The render path correctly handles the four-status enumeration (`primary` / `shadowed` / `failed` / `incompatible`).
- **WinMLDevice as a single concrete class with internal dispatch.** Matches `4_winml_device.md` v1.4 collapse of the six-subclass ABC into one concrete class with module-level dispatch tables. Verified at `session/ep_device.py:555-737`. The `device_facts()` / `ep_facts()` split (§4.1 attribute attribution) matches v1.5.
- **Hard-break, no compat shims (`1_req.md` §C3).** Verified across the entire surface: `WinMLSession.__init__` requires `ep_device=` positional/kwarg with no `device=` / `ep=` fallback; `WinMLAutoModel.from_pretrained` takes `ep_device` positional with no string-tuple fallback; `utils/constants.py` deletes legacy `SUPPORTED_EPS` / `EP_ALIASES` outright. No `if isinstance(arg, str): warn(...) else ...` coercion paths.

## Spec drift that nobody documented

The design corpus is mostly catch-up; v2.9 is the locked-in spec for what ships. But several shipped behaviors have no design-doc paper trail beyond per-file analyses. Each is classified as **spec rot** (impl improved, spec not updated) or **scope creep** (impl exceeded scope without acknowledgment).

| # | Spec says | Code does | Type |
|---|---|---|---|
| SD1 | `compiler/configs.py` factories emit deprecation warnings on `quantize=` | Eight `warnings.warn` call sites with no `import warnings` — `NameError` at runtime | regression (D1) |
| SD2 | `compiler/cli.py` sub-CLI is a supported entry point | `ImportError` on module-load due to missing `CalibrationConfig`/`QDQConfig` | regression (D2) |
| SD3 | `find_qnn_sdk()` walks `_COMMON_SDK_PATHS` fallback | Env-var-only; auto-discovery silently dropped | regression (D3) |
| SD4 | `QNNMonitor.get_provider_options()` includes the four bundled-ORT HTP defaults | Returns `{profiling_level, profiling_file_path}` + caller's `_extra` only | regression (D4) |
| SD5 | The pluggable `register_tracer` hook is the extension mechanism | Hook gone; new monitors require editing `commands/perf.py` | spec rot (D5) |
| SD6 | `OpenVINOMonitor` is a future-implementation placeholder | Same as before, still dead; never selected | drift (D6) |
| SD7 | `@<source-tag>` rejection is a per-command concern | Verbatim-duplicated block in `build.py` + `config.py` | simplification target (D7) |
| SD8 | CPU short-name → None is implicit in the catalog | Duplicate explicit guard in `config/precision.py` + `config/build.py` | simplification target (D8) |
| SD9 | `report.py` renderers are independent paths | Duplicate sort+slice+empty-guard + identical comment block in basic + detail | simplification target (D9) |
| SD10 | `_extract_summary` rename contract is data | 14 hand-rolled `_require` calls; one drop silently breaks the renderer | simplification target (D10) |
| SD11 | `_EP_CONTEXT_DEFAULTS` replaces per-EP factory boilerplate | Dead — referenced nowhere; eight factories still hand-encode `enable_ep_context` | simplification target (D11) |
| SD12 | `utils/constants.py` is the EP/device taxonomy second-source-of-truth | Near-empty after retirement; two enum bridges remain with uppercase keys | simplification target (D12) |
| SD13 | The squash is exclusively a session/EP refactor | Includes orthogonal `_transformers_compat.py` (+304) | squash collateral (D13) |
| SD14 | The squash is exclusively a session/EP refactor | Includes Python 3.10 → 3.11 modernization across multiple files | squash collateral (D14) |
| SD15 | `auto_device`'s `last_error` tracks only true registration failures | Bug: `last_error` survives a successful registration that didn't expose the target device, so `WinMLEPRegistrationFailed` may fire when `DeviceNotFound` is the correct error | spec rot (latent bug — see Q4 below) |
| SD16 | `_build_session_options` is package-public via `session/__init__.py` | Not in `__all__`; `qairt_session.py` imports via fragile Python attribute fallthrough | spec rot (Q5 below) |
| SD17 | `_detect_best_device()` and `_get_install_suggestion()` in `session.py` serve real purposes | Dead code; the former references a "PREFER_NPU policy" that no longer exists, the latter has no caller | spec rot |
| SD18 | `_active_session_option_entries` field on `WinMLSession` tracks session-options layer | Dead state; initialized to `{}`, snapshotted in `perf()`, never populated | spec rot |
| SD19 | `WinMLAutoModel.from_pretrained` and `from_onnx` have consistent `ep_device` ergonomics | `from_pretrained` takes it positional; `from_onnx` keyword-only — worst of both worlds | drift (M3-equivalent from a509a67) |

**Summary:** 4 regressions + 8 simplification targets + 2 squash-collateral + 5 spec-rot items = 19 documented drift instances. The four regressions block the merge; the eight simplification targets and five spec-rot items are follow-up debt.

## Architectural critique

Independent of the design docs. Five questions that determine whether the v2.9 architecture is the right shape:

### Q1: Is the unified-source synthesis (`BuiltinSource` appended into `_discovered` at registry `__init__`) the right contract?

**Answer: Yes, this is a big architectural win.** The pre-v2.9 codebase had two parallel pipelines: built-in EPs (CPU/Dml/Azure) were a special case in `WinMLEPRegistry.__init__` that bypassed `discover_all_eps`, while plugin EPs went through the standard discovery + registration flow. `commands/sys.py:_gather_ep_info` had a two-pass merge to combine them. The renderer had to know which was which to format rows correctly. The v2.9 unification — synthesize a `BuiltinSource` entry per ORT-available-provider name not covered by filesystem discovery, append to `_discovered` — collapses the two pipelines into one. The `register_ep` dispatcher branches on `isinstance(entry.source, BuiltinSource)`, but every other consumer treats the entries uniformly. `--ep cpu@bundled` round-trips correctly. The `_entry_source_tag` dispatcher recognizes `BuiltinSource → "bundled"` as one of the canonical seven. `commands/sys.py` deleted ~100 lines of the two-loop merge. The cost is a `Path()` sentinel for the `dll_path` field on `BuiltinSource` entries — handled cleanly by `EPEntry.is_filesystem_backed()`. **Verdict: correct call; the architectural integration is one of the squash's headline wins.**

### Q2: Is `WinMLDevice(handle)` direct construction (no `wrap_ort_device(handle)` factory) the right shape?

**Answer: Yes, deletion of the shim was the right call.** Per `4_winml_device.md` §6 (and §3.4): *"The earlier `wrap_ort_device(handle)` factory shim was deleted in v2.10 — it forwarded one-line to the constructor and existed only to host a per-EP dispatch table that v1.4 of `4_winml_device.md` had already collapsed into property-access dispatch."* The shim was vestigial. The single concrete `WinMLDevice` class with internal dispatch tables (per-EP `memory_bytes` / `architecture` / `capabilities` / `driver_version` schemas, dispatched on `self._ort.ep_name`) is correct: it has no consumer of per-EP method polymorphism in the codebase, the `--list-ep` renderer reads the unified API, and EPDoctor is speculative. The factory deletion is consistent with the design's "no over-abstraction" stance. **Verdict: correct call; the shim deletion is consistent with the architecture.**

### Q3: Is the `EpAtSourceParamType` click `ParamType` the right level for `@<source-tag>` parsing?

**Answer: Yes, click parse time is the right boundary.** Pre-v2.9 every command parsed `<name>@<source-tag>` in its own try/except, producing inconsistent error messages and validation timing. The new `EpAtSourceParamType` in `commands/_ep_arg.py` (98 lines, NEW) does five validation steps at click parse time: whitespace check, multi-`@` check, no-`@` shortcut, empty-half rejection, source-tag closed-set validation against `VALID_SOURCE_TAGS`. Errors surface as `click.UsageError` with proper formatting before the callback runs. The architectural fit is excellent — the design doesn't address how the rejection should be implemented, and `EpAtSourceParamType` picks the right layer. The two complaints: (a) it imports `VALID_SOURCE_TAGS` directly from `..session.ep_device` (a submodule reach, not the facade — should be re-exported from `session/__init__.py`); (b) the `--ep <name>@<source>` rejection block is duplicated verbatim across `build.py` + `config.py` (D7 above). Both are easy follow-ups. **Verdict: correct level; minor refactor to consolidate the rejection block.**

### Q4: Is `auto_device`'s `last_error` logic correct?

**Answer: No — there's a latent bug.** Per `session__ep_registry.md` Risks #1: *"If candidate A fails with `WinMLEPRegistrationFailed` and candidate B succeeds but doesn't expose the target device, the user sees `WinMLEPRegistrationFailed` (with A's traceback) when they should see `DeviceNotFound`."* The fix is three lines:

```python
for entry in candidates:
    try:
        winml_ep = self.register_ep(entry)
    except WinMLEPRegistrationFailed as e:
        last_error = e
        continue
    last_error = None  # successful registration; reset
    for device in winml_ep.devices:
        if device.device_type == target_device_upper:
            return WinMLEPDevice(ep=winml_ep, device=device)
```

The bug surface depends on EP availability — mostly latent today but real for any host with one failing plugin and one working plugin that doesn't expose the requested device class. **Verdict: latent bug; three-line fix; should land before merge.**

### Q5: Is the `_build_session_options` import path from `WinMLQairtSession` acceptable?

**Answer: No — the import is fragile.** Per `session__session.md` Risks #4: *"the qairt subclass imports `_build_session_options` by underscore-prefixed name from `..session` (not from `..session.session`). This requires either `_build_session_options` to be in `session/__init__.py` `__all__` or for `from ..session import _build_session_options` to work via implicit module attribute access. Looking at `session/__init__.py`: it's not in `__all__`. The import works only because Python's `from package import name` looks up `name` as an attribute of the package, which falls through to `package.session._build_session_options` ONLY when no `name` shadow exists. This is fragile."* The cleanest fix is to move the three module-level free functions (`_ep_defaults`, `_build_provider_options`, `_build_session_options`) into a `session/_session_options.py` submodule and import from there explicitly. **Verdict: works today by Python attribute fallthrough; refactor recommended; one-line fix is also acceptable (add to `__init__`'s `__all__`).**

## Test coverage critique

The four regressions head the test-coverage failure list, but several other gaps are worth flagging. Ranked by load-bearing-ness:

### TC1: `compiler/configs.py` deprecation warning path has tests that *should* be passing but are failing because the import is missing

The brief names `test_quantize_false_emits_deprecation[for_*]` as failing. The test does the right thing — exercises the deprecation path that `warnings.warn` emits — and the failure is exactly the `NameError` D1 describes. The CI signal is correct; the bug is mechanical. **Severity: high. Effort: trivial (1-line import).**

### TC2: `compiler/cli.py` sub-CLI has no smoke test that exercises module-load

The `ImportError` at module-load (D2) would be caught by a single test that does `import winml.modelkit.compiler.cli`. Currently the test suite either doesn't import it, or imports it under conditions that suppress the failure. **Severity: high. Effort: S (one test; or delete the file).**

### TC3: `find_qnn_sdk()` auto-discovery removal has no test

The pre-state `find_qnn_sdk()` had a fallback over `_COMMON_SDK_PATHS`. The post-state removed it. No test asserts the env-var-only contract. A unit test with `QNN_SDK_ROOT` unset + a mocked filesystem at `D:\QC\<version>\bin\` would have caught the silent behavior change. **Severity: medium (only affects QNN dev boxes). Effort: S.**

### TC4: `QNNMonitor.get_provider_options()` HTP-defaults regression has no test

The pre-state hardcoded four HTP defaults. The post-state ships none of them. A unit test that asserts which keys appear in `get_provider_options()` for the bundled-ORT path vs the WinML-ORT path would have caught the silent change. **Severity: medium (silent perf regression on bundled wheel). Effort: S.**

### TC5: `auto_device`'s `last_error` reset-on-success has no test

The bug Q4 describes is not asserted anywhere. A two-step test (candidate A raises `WinMLEPRegistrationFailed`; candidate B succeeds but doesn't expose target device class) would pin the correct error class as `DeviceNotFound`. **Severity: medium (latent bug; surfaces when EP availability is imbalanced). Effort: S.**

### TC6: `auto_device` ↔ `register_ep` idempotency under realistic load

Per `session__ep_registry.md`: the `_registered` cache is keyed by `entry.dll_path`. Two `EPEntry` rows resolving to the same absolute DLL path should hit the same cache slot. There's no test that exercises this with a precedence retry that re-encounters a previously-cached candidate. **Severity: low (the implementation is correct). Effort: S.**

### TC7: `WinMLEPMonitor.__init_subclass__` guard coverage

The class-definition-time `__init_subclass__` rejects non-bool shadow of `requires_session_teardown`. The narrow miss case (instance-level shadow in `__init__`) is not catchable by `__init_subclass__` but also not asserted negatively. A `pytest.raises(TypeError)` test that defines a non-bool subclass at module load would lock the contract. **Severity: low. Effort: S.**

### TC8: `_resolve_op_type` four-layer fallback chain coverage

The PRD FR-14 fallback chain has four layers (L1 ONNX → L2 EP-authoritative → L3 heuristic → L4 raw). Per `session__monitor__qnn_monitor.md`, `_resolve_op_type` is implemented but per-layer coverage is not enumerated. A coverage matrix asserting L1 hit, L2 hit (L1 miss), L3 hit (L1+L2 miss), L4 hit (L1+L2+L3 miss) is the load-bearing test. **Severity: medium. Effort: S.**

### TC9: `_to_int` hardening has gaps

Per `session__monitor__qnn_monitor.md`: `_to_int(val, field)` is applied to `accel_execute_cycles` and `accel_execute_us` but **not** to `num_samples` (line 511: `int(meta.get("num_samples", 0) or 0)`). If QNN ever emits `"5.0"` for this field, it raises `ValueError`. No test exercises this. **Severity: low (unlikely SDK schema). Effort: S.**

### TC10: `WinMLDevice` dispatch coverage matrix

`WinMLDevice` has 5 vendor-specific properties (`memory_bytes`, `architecture`, `capabilities`, `driver_version`, `compiler_version`) dispatching on `ep_name`. Each property covers OpenVINO + DML (memory) or OpenVINO only (architecture, driver_version, compiler_version). Test coverage per property × per EP is patchy. A table-driven test that mocks `OrtEpDevice` with realistic vendor metadata for each EP × property combination would lock the contract. **Severity: low (display only). Effort: M.**

## Risks ranked by likelihood × blast radius

| # | Risk | Likelihood | Blast radius | Trigger | Mitigation |
|---|---|---|---|---|---|
| R1 | D1 — `warnings.warn` `NameError` lethal for legacy `quantize=` callers | High (any caller using legacy kwarg) | UX (one factory call) | `WinMLCompileConfig.for_qnn(quantize=True)` | Add `import warnings` or finish factory consolidation |
| R2 | D2 — `compiler/cli.py` `ImportError` at module load | High (any test or user that touches the module) | Module (whole sub-CLI dead) | `python -m winml.modelkit.compiler compile ...` | Delete the file or fix imports |
| R3 | D3 — Silent QNN SDK auto-discovery loss | Medium (QNN dev boxes without env var set) | Local (one user's op-trace falls to basic) | Pre-existing `D:\QC\<version>` install without `QNN_SDK_ROOT` | Restore `_COMMON_SDK_PATHS` or document breaking change in warning |
| R4 | D4 — Silent QNN HTP defaults loss on bundled-ORT | Medium (any bundled-wheel user running op-trace) | Local (perf characteristics change) | `winml perf --ep qnn --op-tracing basic` on `onnxruntime-qnn` wheel | Add `QNNMonitor.for_bundled_ort` factory or migrate defaults to catalog |
| R5 | Q4 — `auto_device` last-error survives successful registration | Medium (latent until EP availability is imbalanced) | Local (wrong exception class) | One failing plugin + one working plugin that doesn't expose target device | Three-line fix in `auto_device` loop |
| R6 | ~~`_build_session_options` import path is fragile~~ — **WITHDRAWN per fact-check Batch 4.** `qairt_session.py:from ..session import _build_session_options` correctly resolves to the **sibling submodule** `session.session` via standard PEP-328 relative-import semantics, NOT to a parent-package attribute fallthrough. Empirically: `from winml.modelkit.session import _build_session_options` (absolute, from package) fails because the symbol is not on the `__init__.py` surface, but the relative form inside `qairt/` succeeds because `..session` from inside `qairt/` names the sibling `.session` submodule. No fragility. | — | — | — | — |
| R7 | `_FULL_TO_SHORT` claimed lazy but is eager (`session/ep_device.py:113`) | Low (no in-tree mutator) | Module (stale inverse if `_SHORT_TO_FULL` mutated post-import) | Test that mutates `_SHORT_TO_FULL` | Fix comment or actually lazy-build via `@functools.cache` |
| R8 | ~~`WinMLDevice.ort_handle` public accessor unused~~ — **WITHDRAWN per fact-check Batch 4.** `ort_handle` IS used by `analyze/runtime_checker/ep_checker.py:67` (a real production caller). The property's own docstring codifies the public-accessor-for-external-modules / `_ort`-for-internal-session-build split as a **deliberate API boundary**, not a conflict. The split is consistent with the "no private symbol imports outside session/" rule cited elsewhere in this DEEP-DIVE. Reclassify: the apparent inconsistency was a misread of the docstring contract. | — | — | — | — |
| R9 | `WinMLAutoModel.from_pretrained` positional `ep_device` is a footgun | Medium (any caller passing config positionally) | Local (silent rebind) | `from_pretrained(model_id, my_config)` | Make `ep_device` keyword-only |
| R10 | `_transformers_compat.py` patches transformers' private `_objects` | Low (transformers 5.x stable) | Process-wide (every consumer) | Transformers minor version that renames `_objects` | Version-pin transformers or detect-and-warn at compat load |
| R11 | `OpenVINOMonitor` placeholder anchors a dead export | Low (no consumer) | Module (one dead class) | None — purely maintenance debt | Delete the file |
| R12 | `compiler/cli.py` even after the import fix has diverged behavior from `commands/compile.py` (no `--device`, no `resolve_device`) | Medium (anyone using sub-CLI for smoke test) | Local (different code path than top-level CLI) | Sub-CLI invocation | Delete the file (preferred) or unify with top-level CLI |

## Open questions the design corpus explicitly left dangling

### OQ1 — Casing sweep + `EpEntry` nesting in `src/`

Per `2_coreloop.md` §10: *"The locked-in classes here use `EP` uppercase per the canonical acronym table. Current `src/winml/modelkit/ep_path.py` still has `EpCatalog`, `EpSource`, `PyPiSource`, `MsixPackageSource`, `ResolvedEp` — all queued for a one-shot rename PR (see §11 inventory). The same PR nests the old `EpEntry` (EP-metadata catalog row) into `EPCatalog` as `EPCatalog.Row` so the top-level `EPEntry` name belongs to the new discovery record (renamed from `ResolvedEp`). Test imports and downstream consumers update in the same PR."* Verifying against the shipped `ep_path.py`: the file uses `EPCatalog`, `EPCatalog.Row`, `EPEntry`, `EPSource`, `BuiltinSource`, `PyPISource`, `NuGetSource`, `DirectorySource`, `WinMLCatalogSource`, `MSIXPackageSource` — the rename happened **in this commit**. So OQ1 is closed.

### OQ2 — `_finalize_output` filename naming protocol

The compile pipeline's `_finalize_output` uses a three-way input search preferring `{stem}_{device_category}_ctx.onnx`. This was an open question in commit `a509a67` (D12). The v2.9 design corpus does not revisit it. Per the per-file analysis for `compiler__stages__compile.md` (not read in this analysis but referenced), the input-vs-output naming asymmetry persists. **Status: open, deferred from a509a67.**

### OQ3 — `models/auto.py` `from_pretrained` positional `ep_device`

Per `models__auto.md` Risks #4: *"Positional-vs-keyword asymmetry across methods. `from_pretrained` has `ep_device` positional; `from_onnx` has it keyword-only. The asymmetry is a footgun."* This was M3 in commit `a509a67` and is still not closed. **Status: open.**

### OQ4 — Per-EP install hints in `commands/compile.py`

a509a67 §OQ7 noted that `commands/compile.py` hardcodes `EPNotDiscovered`'s install hint to mention `onnxruntime-qnn`. The v2.9 corpus does not address this; per the per-file analyses, the hardcode persists. **Status: open.**

### OQ5 — `WinMLCatalogSource` provider version probing

Per `ep_path.md`: *"TODO `ep_path` (lines 282-283): 'OQ-2 deferred — provider.version probing.' `WinMLCatalogSource` always emits `version=None`."* The design corpus does not address this. **Status: open, low priority.**

### OQ6 — `EPCatalog`'s MIGraphX `dll_name` is unverified

Per `ep_path.md`: *"The catalog row is built with the guessed name `onnxruntime_providers_migraphx.dll`. If wrong, `_list_msix_eps` will skip valid MIGraphX MSIX packages because `EP_CATALOG.ep_for_dll(dll.name)` returns `None`."* The design corpus does not specify the leaf-DLL inventory. **Status: open; blocks MIGraphX MSIX discovery.**

### OQ7 — ProofOfExecution typed accessor

a509a67's OQ5 — `VitisAIMonitor` / `OpenVINOMonitor` still expose data via `to_dict()` rather than a typed `proof: ProofOfExecution | None` accessor. The v2.9 corpus does not revisit it. Status: open, deferred.

### OQ8 — `_active_session_option_entries` dead state

Per `session__session.md` Simplification #9: *"the `_active_session_option_entries` field is initialized to `{}` and snapshotted in `perf()` but never populated outside of perf snapshots... never written. The field is dead state."* The design corpus does not mention it. **Status: open, low priority.**

## What I would do next (prioritized)

Twelve concrete items. Effort: S < 1 day, M < 3 days, L < 1 week.

1. **Pre-merge fix D1: `compiler/configs.py` `import warnings`.** Either add the one-line import or finish the consolidation (collapse the eight `for_*` factories into one `for_provider` driven by `_EP_CONTEXT_DEFAULTS`; delete the deprecated `quantize=` kwarg). **Effort: S** for the import; **M** for the consolidation. Sites: `src/winml/modelkit/compiler/configs.py`.

2. **Pre-merge fix D2: `compiler/cli.py` `ImportError`.** Delete the file (preferred per `MEMORY.md` hard-break stance) OR fix the imports to `from ..quant.config import CalibrationConfig, QDQConfig` and strip the quant-related kwargs from the `WinMLCompileConfig(...)` call. **Effort: S.** Site: `src/winml/modelkit/compiler/cli.py`.

3. **Pre-merge fix D3: restore QNN SDK auto-discovery OR document the breaking change.** Either re-introduce `_COMMON_SDK_PATHS = (r"D:\QC", r"C:\Qualcomm\AIStack\qairt")` and the version-sorted walk in `find_qnn_sdk()`, OR update the warning message to explicitly call out the removal and document in the v2.9 design corpus. **Effort: S.** Site: `src/winml/modelkit/session/monitor/qnn/viewer.py`.

4. **Pre-merge fix D4: add `QNNMonitor.for_bundled_ort(level)` factory OR migrate HTP defaults to `EP_DEVICE_SPECS`.** The factory option re-introduces the four HTP defaults for bundled-wheel users without affecting the WinML-ORT path; the catalog option applies universally but re-introduces the ORT 1.23.5 `backend_path` crash risk on the WinML-ORT path. **Effort: S** for the factory; **M** for the catalog migration. Site: `src/winml/modelkit/session/monitor/qnn_monitor.py` or `src/winml/modelkit/session/ep_device.py`.

5. **Pre-merge fix Q4: `auto_device` `last_error` reset on successful registration.** Three-line fix. **Effort: S.** Site: `src/winml/modelkit/session/ep_registry.py:357-418`.

6. **Pre-merge tests for the regressions (TC1-TC4).** Four new tests, one per regression. **Effort: S.** Sites: `tests/unit/compiler/test_configs_deprecation.py` (new), `tests/unit/compiler/test_cli_import.py` (new), `tests/unit/session/monitor/qnn/test_viewer.py` (new), `tests/unit/session/monitor/test_qnn_monitor_provider_options.py` (new).

7. **Follow-up D7: extract `_reject_ep_source(ep, *, command_name)` helper.** Collapse the duplicated rejection block in `build.py` + `config.py` into one helper in `commands/_ep_arg.py`. **Effort: S.** Sites: `src/winml/modelkit/commands/_ep_arg.py`, `commands/build.py`, `commands/config.py`.

8. **Follow-up D8: add `EPDeviceSpec.no_compile: bool = False`.** Mark the CPU/CPU row as `no_compile=True`; rewrite the two CPU-guard call sites to consult the catalog. **Effort: S.** Sites: `src/winml/modelkit/session/ep_device.py`, `config/precision.py`, `config/build.py`.

9. **Follow-up D9 + D10: extract `_topk` helper + `_RENAME_MAP`.** Two parallel renderer-side simplifications. **Effort: S each.** Sites: `src/winml/modelkit/session/monitor/report.py`, `src/winml/modelkit/session/monitor/qnn/_internal.py`.

10. **Follow-up D12: migrate `DEVICE_TO_DEVICE_TYPE` to `session/ep_device.py` and delete `utils/constants.py`.** Lowercase the keys; update the four consumers in `analyze/`. **Effort: M.** Sites: `src/winml/modelkit/session/ep_device.py`, `utils/constants.py` (deleted), `analyze/runtime_checker/check_ops.py`, `analyze/pattern/check_patterns.py`, `analyze/core/runtime_checker_query.py`, `utils/cli.py`.

11. **Follow-up Q5: explicit `_build_session_options` re-export.** Add to `session/__init__.py`'s `__all__` or move the three free functions to `session/_session_options.py` and import explicitly from both `session.py` and `qairt_session.py`. **Effort: S.** Sites: `src/winml/modelkit/session/__init__.py` or new `session/_session_options.py`.

12. **Follow-up D6: delete `OpenVINOMonitor` or wire it.** Per `MEMORY.md` hard-break stance: delete the placeholder file and the `session/__init__.py` export. When real OpenVINO telemetry lands, recreate. **Effort: S.** Sites: `src/winml/modelkit/session/monitor/openvino_monitor.py` (deleted), `session/__init__.py`.

## Confidence statement

**High confidence:**
- The four 🔴 regressions (D1-D4). All grep-verified: `import warnings` is absent from `compiler/configs.py` despite 8 call sites; `CalibrationConfig` and `QDQConfig` are imported from `.configs` in `compiler/cli.py` but absent from that module; `_COMMON_SDK_PATHS` / `backend_path` / `htp_performance_mode` are zero-hit greps in `session/monitor/qnn/`.
- The architectural alignment claims. `EPDeviceTarget`, `EPDeviceSpec`, `EPEntry`, `WinMLDevice`, `WinMLEP`, `WinMLEPDevice` all ship at the contracts specified in `3_design_classes.md` v1.2. The unified `BuiltinSource` synthesis matches `2_coreloop.md` v2.9 exactly. The atomic + compound registry surface matches §5.8 lockdown.
- The simplification-target enumeration (D7-D12). The per-file analyses already named most items; this doc consolidates and prioritizes.
- The `auto_device` `last_error` bug (Q4). The bug surface is visible in the code; the fix is three lines.

**Medium confidence:**
- The D3 (QNN SDK auto-discovery) impact assessment. The per-file analysis explicitly flags it as "Was this intentional (env-var-only is the supported contract) or accidental?" Without the commit-body justification, the severity could be lower (intentional hardening) or higher (silent break). The recommendation handles both cases.
- The D4 (QNN HTP defaults) impact. The catalog's `htp_performance_mode=burst` does still apply via `_ep_defaults` for the bundled-ORT path, mitigating some of the loss. The genuine losses are `backend_path=QnnHtp.dll` and `enable_htp_fp16_precision=1`. Without a QNN testbed, the perf impact of dropping `enable_htp_fp16_precision=1` is uncharacterized.
- The "squash collateral" framing (D13, D14). The `_transformers_compat.py` and Python 3.10 → 3.11 migrations are clearly orthogonal to the v2.9 refactor. Whether they should have been separate commits is a stylistic judgment; some projects bundle, some don't.

**Lower confidence (where a domain expert might overrule):**
- The Q5 (`_build_session_options` import path) verdict. The Python attribute-fallthrough behavior is real but subtle; I'd want to verify against the actual `qairt_session.py` import behavior. The per-file analysis describes it confidently but I haven't reproduced.
- The `_FULL_TO_SHORT` claim that the comment lies about laziness (R7). The dict literal is built eagerly; the docstring claim is wrong. But the practical impact is zero unless someone mutates `_SHORT_TO_FULL` post-import, which never happens in production.
- The "PREFER_NPU policy" dead code claim in `_detect_best_device()`. The per-file analysis claims it's dead; without searching every caller, I can't verify it's never called.

---

**Ship readiness verdict: BLOCKED on the four 🔴 regressions.** D1 and D2 are mechanical import-time bugs that break user-visible test paths. D3 and D4 are silent UX/perf regressions that affect specific user cohorts (QNN dev boxes, bundled-ORT users) and should at minimum be documented. The architectural work is excellent — the v2.9 corpus and the shipped code are aligned and the unified-source synthesis is a major win — but the squash collateral landed without the consolidation passes that would have caught the four regressions. Fix the four; document the squash collateral; ship.
