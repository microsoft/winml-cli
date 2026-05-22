# Commit a509a67 — Summary

## What this commit is

Commit a509a67, titled **"feat(session): op-tracing perf monitor + EPDevice/WinMLSession refactor"**, is the squash of 13 development commits on branch `feat/op-tracing-refactor_3` (db39b80..ee0ba33), touching 234 files at +41,822 / -3,740 lines. The commit body breaks the work into six themes: an **op-tracing perf monitor** (new `session/monitor/` hierarchy under an `EPMonitor` ABC, replacing the deleted `optracing/` tree), an **EPDevice + WinMLSession hard-break refactor** (Option A — no compat shims), an **EPDeviceSpec catalog as single source of truth** (one ordered tuple replaces four parallel taxonomy dicts), a **Compile CLI + Gap #1/#3 fix block** (real `ort.ModelCompiler` invocation, naming-protocol fix, symmetric defensive guards against ORT's non-idempotent native registration), a **monitor pipeline hardening** pass (three silent-failure paths closed), and a **Sysinfo + taxonomy cleanup** that dissolves `sysinfo/device.py` into three natural homes. This summary integrates the per-file analyses of all 48 changed Python files under `src/winml/modelkit/` to give a coherent picture of the architectural change without recapitulating every line.

## File-count by theme

| Theme | Files touched | New | Deleted | Modified |
|---|---|---|---|---|
| Session core (EPDevice + registry + facade) | 4 | 1 | 0 | 3 |
| Session lifecycle (session.py, qairt) | 2 | 0 | 0 | 2 |
| Op-tracing monitor pipeline (new tree) | 9 | 5 | 1 | 3 |
| Legacy optracing/ tree (deleted wholesale) | 7 | 0 | 7 | 0 |
| CLI commands | 10 | 1 | 0 | 9 |
| Compiler + config | 4 | 0 | 0 | 4 |
| Sysinfo redistribution | 3 | 0 | 1 | 2 |
| Analyze subsystem (downstream) | 3 | 0 | 0 | 3 |
| Eval + models (downstream) | 3 | 0 | 0 | 3 |
| Utils + top-level | 3 | 0 | 0 | 3 |
| **Total (Python under `src/winml/modelkit/`)** | **48** | **7** | **9** | **32** |

Note: the 48 in the per-file analysis set includes three rename events (`session/monitor/__init__.py` renamed from `tests/unit/optracing/test_detection.py`, `session/monitor/qnn/viewer.py` and `session/monitor/report.py` renamed from `optracing/*`), which `git --name-status` reports as `R` rather than `A`. They are net new at the destination path.

## The six architectural moves

### 1. `EPDevice` as a typed value object + `WinMLSession.__init__` hard-break — opaque `(ep:str, device:str)` pair-passing replaced; legacy kwargs removed

`session/ep_device.py` (446 LOC, new) introduces a frozen `EPDevice` dataclass with the fields `ep: str` (canonical full name, e.g. `"QNNExecutionProvider"`), `device: str` (lowercased category, e.g. `"npu"`), `vendor_id: int`, `device_id: int`, and `vendor: str = ""`. It is the wire format between the CLI boundary and `WinMLSession`. Construction normalizes `device` via `object.__setattr__` in `__post_init__` (the frozen-dataclass workaround). The `OrtEpDevice` handle is *not* stored on the object; it is re-derived at session-build time inside `session.py` via `WinMLEPRegistry.get_instance().register_ep(...)` plus a `(device.type.name, vendor_id, device_id)` filter. `EPDevice` carries `to_dict` / `from_dict` for JSON round-trip through `WinMLCompileConfig.to_dict / from_dict`, which is what lets `compiler/stages/compile.py` consume a typed `EPDevice` without cross-package private imports.

Co-equal with the new type is a **hard-break of `WinMLSession.__init__`** (Option A — no compat shims): the `device=`, `ep=`, and `session_options=` kwargs are **gone**, replaced by a single positional-required `ep_device: EPDevice` argument. The new signature is `WinMLSession(onnx_path, ep_device, *, ep_config=None)` — every caller that previously passed `device="npu"` or `ep="qnn"` is broken by design. The hard-break propagates through `WinMLPreTrainedModel.__init__(onnx_path, ep_device, config=None)` (where `ep_device` is positional before `config`), `WinMLAutoModel.from_onnx(*, ep_device, ...)` (keyword-only), and `WinMLAutoModel.from_pretrained(model_id_or_path, ep_device, *, ...)` (positional — the asymmetry across the two factory methods is a footgun). The related deletion of `WinMLSession._build_session_options` (instance method) in favor of a module-level free `_build_session_options(ep_device, ep_config, monitor, base_session_options)` is the pattern shift that lets the compile pipeline (move 4) consume it without instantiating a session first.

Key files: `session/ep_device.py`, `session/__init__.py` (re-exports 23 new public names), `session/session.py` (`__init__` signature break + free `_build_session_options`), `models/auto.py` (`from_onnx` keyword-only `ep_device`, `from_pretrained` positional `ep_device`), `models/winml/base.py` (`WinMLPreTrainedModel.__init__(onnx_path, ep_device, config=None)`), `eval/evaluate.py`, `commands/perf.py`, `commands/compile.py`, `compiler/configs.py` (new `for_ep_device` factory + optional `ep_device: EPDevice | None` field), `compiler/stages/compile.py` (3-way resolver: dict → field → fallback `resolve_device(ep=...)`).

### 2. `EPDeviceSpec` catalog as single source of truth — order encodes preference

`EP_DEVICE_SPECS` is a 13-entry ordered tuple of `EPDeviceSpec(ep, device, default_provider_options)` dataclasses. **Order encodes preference**: QNN-NPU is first, then DML-GPU, then CPU-CPU (the three primary entries), followed by QNN secondary (GPU, CPU), OpenVINO (NPU/GPU/CPU as three separate rows), VitisAI-NPU, MIGraphX-GPU, Tensorrt-GPU, CUDA-GPU, NvTensorRtRtx-GPU. Five legacy data structures were dissolved into derivations from this tuple: `_EP_TO_DEVICE` (`config/precision.py`), `_DEVICE_TO_PROVIDER` (`config/precision.py`), `get_provider_for_device` (`config/precision.py`), the `_ep_defaults` if/else ladder (formerly in `session/session.py`), and `_EP_DEVICE_MAP` / `_DEVICE_EP_MAP` (`sysinfo/device.py`, both deleted in the same commit). `VALID_EPS` and `VALID_DEVICES` are now derived structurally — `VALID_EPS` is a frozenset of **short** names (built via `short_ep_name(s.ep)`), `VALID_DEVICES` is a frozenset of device categories. The catalog's QNN-NPU entry embeds `htp_performance_mode="burst"` + `htp_graph_finalization_optimization_mode="3"` defaults — the verification block in the commit body claims roughly +3× ResNet-50 throughput on QNN-NPU (5.73 ms → 1.90 ms / 175 → 526 samples/s) vs an empty-defaults baseline.

Key files: `session/ep_device.py` (catalog + 10 helpers: `lookup_device_spec`, `default_device_for_ep`, `default_ep_for_device`, `eps_for_device`, `ep_to_device`, `canonicalize_ep_name`, `expand_ep_name`, `short_ep_name`, `resolve_device`, `auto_detect_device`). Downstream consumers: `commands/sys.py` (one row per `(ep, device)` spec instead of one row per EP), `commands/cli.py`/`utils/cli.py` (Click choices derive from `VALID_DEVICES` / `VALID_EPS`), `compiler/cli.py`, `analyze/analyzer.py` (default EP fan-out now `sorted(eps_for_device("npu"))`), `config/precision.py` (purged of own taxonomy tables; now imports from `..session`).

### 3. Op-tracing pipeline rebuilt under `session/monitor/`; old `optracing/` deleted wholesale

The pre-existing `src/winml/modelkit/optracing/` package is **deleted wholesale**. Three concerns were preserved by relocation: `result.py` → `session/monitor/op_metrics.py` (extended with `samples_us` list + `avg_us` / `total_us` / `p90_us` / `sample_count` properties; closed `TraceStatus` literal alias; `status` + `error` lifecycle fields on `OpTraceResult`); `report.py` → `session/monitor/report.py` (4-column basic / 10-column detail Rich tables, defensive sort by `percent_of_total`, left-truncating node-path formatter); `qnn/viewer.py` → `session/monitor/qnn/viewer.py` (shell-out to `qnn-profile-viewer.exe`, never raises, returns `None` on every failure mode). The CSV and QHAS parsers were merged into one private module: `session/monitor/qnn/_internal.py` — the **CLAUDE.md `_`-prefixed-private exception** is in use here, and a regression test `tests/unit/architecture/test_qnn_imports.py` (PRD NFR-8) enforces that only `qnn_monitor.py` may import non-`_`-prefixed names from it.

The `OpTracer` ABC (control flow: caller invokes `tracer.run(iterations, warmup)` which owns the inference loop) is **replaced** by `EPMonitor` (control flow: `WinMLSession.perf(monitor=mon)` owns the loop; the monitor only contributes options via `get_session_options()` / `get_provider_options()`, sets `set_onnx_op_types(...)` on entry, and parses artifacts on `__exit__`). The monitor exposes a typed `result` property (`OpTraceResult | None`) and three class-level invariants: `requires_session_teardown: ClassVar[bool]` (C-2 ordering invariant — QNN flushes CSV only on `InferenceSession.__del__`, so `WinMLSession.perf().__exit__` drops the session **before** calling `monitor.__exit__`), `ep_name: ClassVar[str | None]` (pins the perf session to a specific EP so `add_provider_for_devices` receives the monitor's provider options), and an `__init_subclass__` guard rejects non-bool `requires_session_teardown` shadowing at class-definition time. The runtime registry layer (`get_tracer(ep_pattern, level)` substring-matched against `_TRACERS`) is **dropped with no replacement**: callers explicitly instantiate `QNNMonitor()` and pass it to `session.perf(monitor=...)`. Selection at the CLI layer happens inside `commands/perf.py::_resolve_ep_monitor`, an explicit dispatch (`qnn` → `QNNMonitor`, `vitisai` → `VitisAIMonitor`, else `NullEPMonitor` or `RuntimeError`).

`QNNMonitor` (`session/monitor/qnn_monitor.py`, +633 LOC) is the concrete QNN op-tracer. It owns the two profiling provider options (`profiling_level`, `profiling_file_path`), drives the QHAS-viewer shell-out (detail) or CSV-only path (basic), and emits an `OpTraceResult` of `OperatorMetrics`. Its `_resolve_op_type` implements the v2.4 FR-14 fallback chain L1 (ONNX `node.op_type` map) → L2 (EP-authoritative `qnn_op_type`) → L3 (heuristic token-strip leaf-split) → L4 (raw `op_path`). The architecture regression test enforces information-hiding at the qnn._internal boundary.

Key files (11 present in `session/monitor/` at HEAD): `session/monitor/__init__.py` (empty, no `__all__` — see "Internal inconsistencies"), `session/monitor/ep_monitor.py` (ABC, +95/-9), `session/monitor/qnn_monitor.py` (+633), `session/monitor/op_metrics.py` (+168), `session/monitor/report.py` (+253), `session/monitor/hw_monitor.py` (live hardware-counter sampler), `session/monitor/openvino_monitor.py` (placeholder-for-parity stub), `session/monitor/vitisai_monitor.py` (placeholder-for-parity stub), `session/monitor/qnn/__init__.py`, `session/monitor/qnn/_internal.py` (+443, new — the CLAUDE.md `_`-prefixed-private boundary), `session/monitor/qnn/viewer.py` (+206). The deleted `session/monitor/live_display.py` (-207, duplicate of `commands/_live_chart.py::LiveMonitorDisplay`) is **not** in the present set.

### 4. `WinMLSession.compile()` resurrected + Compile CLI rewired through `EPDevice`

Pre-this-PR, `WinMLSession.compile()` did not actually drive the compile pipeline — it was a stub-method that never invoked `ort.ModelCompiler`. Two coupled bugs blocked it: **Bug A** — the eager `InferenceSession` construction inside `WinMLSession.__init__` ran with `enable_ep_context=True` whenever the caller passed compile-style session options, which forced ORT to look for an EPContext binary that didn't exist yet, raising a runtime error before `compile()` ever got control. **Bug B** — `compile()` had no call into `ort.ModelCompiler` and no path through the new free `_build_session_options(...)`; it relied on the deleted instance method. The fix defers `InferenceSession` creation when `enable_ep_context=True` (so the compile pipeline runs first) and then calls `_build_session_options(ep_device, ep_config, monitor=None, base_session_options=...)` followed by `ort.ModelCompiler.compile_to_file(...)`. `ort.ModelCompiler` is wrapped in `_suppress_native_output(compile_log)` to redirect QNN SDK native stdout to `<onnx_path>.parent/compile.log`. Failed compile falls back to the original model with a `logger.warning` rather than failing hard.

The CLI side threads `EPDevice` end-to-end: `commands/compile.py` resolves `EPDevice` at the CLI boundary via `session.resolve_device(ep, device)`, constructs `WinMLCompileConfig.for_ep_device(ep_device, ...)` (new factory), and stages run inside a `CompileContext` that carries the typed `EPDevice` alongside the dict-form config (for JSON round-trip via `WinMLCompileConfig.to_dict / from_dict`). `CompileStage._finalize_output`'s three-way filename search — preferring `{stem}_{device_category}_ctx.onnx` (e.g. `*_npu_ctx.onnx`) over the legacy `_qnn_ctx.onnx` and `_ctx.onnx` patterns — closes the input-search gap exposed by the device-category naming the new `WinMLSession.compile()` emits. A new `--device` flag is added to the compile CLI; `DeviceNotFound`/`EPNotDiscovered`/`EPRegistrationFailed` are caught at the CLI boundary with remediation hints (`onnxruntime-qnn` install pointer for `EPNotDiscovered`, available-device list for `DeviceNotFound`).

Key files: `session/session.py` (`compile()` body), `compiler/configs.py` (`WinMLCompileConfig.for_ep_device(...)` factory + optional `ep_device: EPDevice | None` field), `compiler/stages/compile.py` (`CompileContext` carrying `EPDevice`, `_finalize_output` three-way search), `commands/compile.py` (`--device` flag + EPDevice-resolution + structured error UX).

### 5. Two-singleton ORT DLL registration race patched via symmetric defensive guards (tactical fix; consolidation deferred to I1)

`ort.register_execution_provider_library(name, path)` is **not idempotent at the C++ layer**: a second call for the same DLL invokes `exit(127)` natively with no Python traceback, surfacing as `STATUS_DLL_NOT_FOUND` / `0xC000026F`. Two registration entry points exist in this repo — `winml.py:WinML.register_execution_providers` (the Windows AppSDK / WinML singleton) and `session/ep_registry.py:WinMLEPRegistry.register_ep` (the new selective-registration path). The fix is **symmetric defensive guards** on both: before calling `register_execution_provider_library`, query `module.get_ep_devices()` and skip the load if any device already advertises that `ep_name`. Both guards wrap the probe in `try/except Exception → already_loaded = False` so an older ORT lacking `get_ep_devices()` falls through to the existing registration attempt without regression.

The commit body labels the proper fix — collapsing the two singletons — as **issue I1, "Singleton consolidation deferred"**, and the per-file analysis of `winml.md` notes the diagnostic doc at `docs/design/session/2026-05-13-t6-analyze-crash-diagnostic.md` traced the original perf-HF-pipeline crash to this exact double-registration. The deferred t6-analyze-crash dual-singleton fix lands here as the patch — not the design.

Key files: `winml.py` (12 lines added inside the registration loop), `session/ep_registry.py` (the `register_ep` body's `already_loaded` probe), and the deferred consolidation is the documented follow-up.

### 6. Sysinfo dissolved — `device.py` functions redistributed; device-resolution surface re-homed around `EPDevice`

`src/winml/modelkit/sysinfo/device.py` (191 LOC) is **deleted in its entirety**. The module conflated three concerns: (a) static EP↔device knowledge — moved to `EPDeviceSpec` catalog in `session/ep_device.py`; (b) live hardware introspection — moved to `sysinfo/hardware.py::get_available_devices` (formerly private `_get_available_devices`, now public, +29 LOC); (c) live EP-plugin introspection — moved to `session/ep_registry.py::available_eps()` (formerly private `_get_available_eps`, now public, `lru_cache(maxsize=1)` semantics preserved). The old entry-point function `sysinfo.resolve_device(device="auto") -> tuple[str, list[str]]` is gone: its available-devices list is now obtained via `sysinfo.hardware.get_available_devices()`, and its "auto-pick the strongest device" behavior is provided by the new free helper `session.auto_detect_device() -> str`. The bare name `resolve_device` is now taken by an entirely new function — `session.resolve_device(ep, device="auto"|None) -> EPDevice` — a typed resolver that handles `device="auto"` internally by delegating to `auto_detect_device()` and registers the EP as a side effect. `_EP_DEVICE_MAP` was a duplicate of the catalog and is deleted.

Key files: `sysinfo/__init__.py` (re-exports `get_available_devices` instead of the two device-routing helpers), `sysinfo/device.py` (deleted), `sysinfo/hardware.py` (gains the public function + a module logger), `session/ep_device.py` (hosts `resolve_device` and `auto_detect_device`), `config/build.py` and `config/precision.py` (migrate from `sysinfo.resolve_device` → `get_available_devices()` + `auto_detect_device()` for the auto-pick path, and from private `get_provider_for_device` → `default_ep_for_device` + `short_ep_name`).

## Three silent-failure paths closed

The commit body lists three "silent-failure paths" closed by the monitor-pipeline hardening pass. The per-file analyses verify each one, and one count is off:

1. **`int("0" or 0)` → `round(float(...))` in `qnn_monitor.py`** (the `_to_int` helper at `session/monitor/qnn_monitor.py`). Previously, the metadata extraction did `int(meta.get("accel_execute_cycles", 0) or 0)`. A QNN SDK that emits `"12345.6"` instead of `"12345"` for `accel_execute_us` would raise `ValueError: invalid literal for int()`, caught by surrounding `except Exception` and converted into `parse_failed` with all op rows lost. Worse, if metadata extraction silently zeroed out, `total_cycles=0` and `accel_us=0` yield `cycle_to_us=0.0` and every `OperatorMetrics.duration_us=0.0` — the trace would render "the entire model ran in 0 µs." `_to_int` now does `round(float(val))` and logs WARNING on parse failure, returning 0.

2. **`dict[key]` → `_require(d, key, context)` × 19 in `qnn/_internal.py`** (commit body says "18"; the per-file count is **19 actual `_require` call sites**: 1 in `parse_qhas` for `"data"`, 14 in `_extract_summary` for the htp_overall_summary row, 4 in `_transform_op` for the qnn_op_instances_nodes entry). The commit body's "18" likely excludes the outermost `"data"` access in `parse_qhas`. The `_require` helper raises `KeyError(f"Required QHAS field {key!r} is missing in {context}")` so SDK schema drift surfaces the exact missing key plus its structural context in the WARNING log line that follows the outer `_try_qhas`'s `except Exception`.

3. **Benchmark JSON write moved AFTER op-trace status check in `commands/perf.py`**. Pre-state: `write_json_report(result, output)` ran unconditionally right after `display_console_report`, **before** the `if op_tracing:` block — so a failed op-trace (exit 4) left a misleading JSON artifact on disk. Post-state: both `if op_tracing` and `else` branches write JSON, but the op-tracing branch writes only **after** the status-check fail-fast guards (`no_data`, `parse_failed`, missing trace_result all `sys.exit(4)` before the JSON write).

## Bug fixes worth calling out

(Note: the `WinMLSession.compile()` resurrection — Bug A + Bug B — is promoted to architectural move #4. The `CompileStage._finalize_output` three-way naming-protocol fix is folded into move #4 as well.)

- **`NvTensorRtRtx` casing fix across 5 spots** (`analyze/runtime_checker/check_ops.py`: error message, `super().__init__` ep_name, `get_ep_checker` dict key, two argparse `choices`/`help` strings) plus the `winml.py` docstring fix. Pre-state casing was `NvTensorRTRTXExecutionProvider`; verified against `ort.get_all_providers()`, the correct casing is `NvTensorRtRtxExecutionProvider`. Hard break for any external caller that scripted the old casing — the argparse `choices` list rejects the old name at parse time.
- **Two missing `_SHORT_TO_FULL` entries**: `cuda → CUDAExecutionProvider` and `tensorrt → TensorrtExecutionProvider`. Previously listed in `VALID_EPS` but `expand_ep_name` passed them through unchanged — causing `EPNotDiscovered` at register time when the user invoked `--ep cuda` or `--ep tensorrt`. Now both are first-class entries.
- **Deferred t6-analyze-crash dual-singleton `STATUS_DLL_NOT_FOUND` fix landed via symmetric guards** in `winml.py` and `session/ep_registry.py` (see Architectural Move #5 above).

## Breaking changes for callers (no compat shims — "Option A hard-break")

- `WinMLSession.__init__` requires `ep_device` positional. The `device=`, `ep=`, and `session_options=` kwargs are **gone**.
- `WinMLPreTrainedModel.__init__` signature is `(self, onnx_path, ep_device, config=None)` — `ep_device` is **positional** before `config`; positional `(WinMLModelForX("path", hf_cfg))` will misinterpret `hf_cfg` as `ep_device`.
- `WinMLAutoModel.from_onnx(*, ep_device, ...)` — `ep_device` is **keyword-only**.
- `WinMLAutoModel.from_pretrained(model_id_or_path, ep_device, *, ...)` — `ep_device` is **positional**. The positional-vs-keyword asymmetry across the two factory methods is a footgun.
- `WinMLEvaluationConfig` has **no `ep` field** — the eval CLI lost the ability to specify EP independently of device.
- `WinMLSession.perf()` yields a `PerfContext(stats, monitor)`, not a bare `PerfStats`. Callers must use `ctx.stats` / `ctx.monitor`. The per-file analysis on `models/winml/base.py` notes that the only signal of this change in `WinMLPreTrainedModel.perf()` is the docstring update.
- `WinMLSession.perf()` signature now accepts `monitor: EPMonitor | None = None` and raises `RuntimeError` on nested entry / `EPMonitorMismatch` on monitor-vs-session EP disagreement.
- `WinMLSession._build_session_options` instance method is **gone**; replaced by module-level free `_build_session_options(ep_device, ep_config, monitor, base_session_options)`. `qairt/qairt_session.py` lazily imports the free function via `from ..session import _build_session_options` — reaching a `_`-prefixed name across module boundaries.
- `WinMLSession.is_compatible()` no longer accepts a `device=` override.
- `WinMLSession.compile()` no longer auto-detects "auto" → best device; device is immutable from `__init__`.
- `WinMLQairtSession.__init__(ep_device: EPDevice | None = None, ...)` — the `device="qnn"` kwarg is **gone**; default-`None` auto-resolves to `resolve_device("qnn", "npu")` at construction. Inconsistent with `WinMLSession` (where `ep_device` is required positional with no default) but intentional convenience for the QAIRT-specific subclass.
- `winml.modelkit.optracing` package is **gone** — every symbol it exported is unreachable. `OpTracer`, `get_tracer`, `register_tracer`, `is_qnn_profiling_available` are dropped with **no replacement**.
- `winml.modelkit.sysinfo.resolve_device` is **gone** — `ImportError` now. The old tuple return is split across two new APIs: `sysinfo.hardware.get_available_devices() -> list[str]` for the available-devices list, and `session.auto_detect_device() -> str` for the auto-picked device string. Callers that want a fully-typed descriptor instead should use `session.resolve_device(ep, device="auto"|None) -> EPDevice`, which handles `"auto"` internally and registers the EP.
- `winml.modelkit.sysinfo.get_ep_device_map` is **gone with no public replacement**.
- `config.precision.get_provider_for_device` is **gone**. Use `default_ep_for_device(device)` + `short_ep_name(...)`.
- `utils.constants.SUPPORTED_EPS` / `EP_ALIASES` / `ALL_EP_NAMES` / `SUPPORTED_DEVICES` are **gone**. Use `session.VALID_EPS` / `VALID_DEVICES`.
- `compiler.cli.compile --ep <full name>` (e.g. `--ep QNNExecutionProvider`) is now rejected at click-parse — only short names in `VALID_EPS` are accepted. Two-letter aliases `ov` and `vitis` are also no longer in the click `Choice` (they survive only inside `normalize_ep_name`'s `_legacy` dict, which is unreachable from the CLI).
- `winml compile --device <uppercase>` is now accepted via `case_sensitive=False`; default device value flowing into command bodies is `"npu"` not `"NPU"`. Any downstream `if device == "NPU"` comparison silently misses.
- `argparse choices` in `analyze/runtime_checker/check_ops.py` reject `NvTensorRTRTXExecutionProvider` (old casing).

## Behavior changes (caller-visible but not API-breaking)

- `WinMLAutoModel.from_pretrained` — `ep_device` is **positional**; `from_onnx` — `ep_device` is **keyword-only**. Inconsistent across the two factory methods.
- `WinMLEvaluationConfig` has no `ep` field → eval CLI lost the ability to specify EP independently of device. The commit body lists `eval/evaluate.py` as migrated, but the CLI surface is not migrated to expose `--ep`.
- `analyzer.py` default EP fan-out **changed from QNN-first to alphabetic** (now `sorted(eps_for_device("npu"))` yields `OpenVINOExecutionProvider, QNNExecutionProvider, VitisAIExecutionProvider`). Tests that depend on iteration order or the first key of `check_op_results` will see different ordering — `AnalysisOutput.aggregate` receives the dict insertion-ordered (Python 3.7+), so OpenVINO populates first now.
- `model.device` property on `WinMLPreTrainedModel` **still reads from `self._session.device`, NOT `self._ep_device.device`** — requested vs effective device can diverge silently.
- `model._ep_device` is **write-only state** — stored but never read in `models/winml/base.py` or any of its subclasses. Possible bit-rot candidate.
- OpenVINO is now a candidate EP for all three device categories (CPU/GPU/NPU). The old `sysinfo/device.py::_EP_DEVICE_MAP` stored `"npu/gpu/cpu"` as a *single string* value for OpenVINO and excluded it from the inverse map via `if "/" not in _device`. The new catalog encodes OpenVINO as three separate rows. On an OpenVINO-only host, `auto_detect_device()` may now resolve to a different device than pre-commit.
- `--device` choices in CLI commands are now **alphabetically sorted** (`auto, cpu, gpu, npu`) rather than the prior hand-ordered list.
- `winml sys` now emits **one row per `(ep, device)` spec** rather than one row per EP — OpenVINO produces three rows (NPU, GPU, CPU), each gated on the EP being installed. JSON consumers counting EP rows to count unique EPs must switch to `set(row["name"])`.
- `available_eps()` in `session/ep_registry.py` is `lru_cache(maxsize=1)` and has **no invalidation API**. After first call, dynamic plugin install is invisible until process restart.
- Log levels uplifted: `WinMLEPRegistry._discover_eps`, `_fix_winrt_runtime`, `get_ort_available_providers`, and `register_to_ort` failure paths uplifted from DEBUG → WARNING per NFR-2. CI logs will now show these as warnings.
- HW chart widened from 10s/80c to 15s/120c (`commands/_live_chart.py`). Terminals narrower than 120 cols may wrap.
- Default `top_k` for op-tracing report is 5 (pre-state in `optracing/report.py` was 15). `--top-k N` CLI flag added to `winml perf`, requires `--op-tracing`, validated `>= 1`.
- Smart default: `--op-tracing` without explicit `--iterations` collapses to 1 iteration.
- `--op-tracing` on a direct `.onnx` input is rejected at click time (`click.UsageError`). HF model IDs and built model dirs only.
- New pre-bench identity block replaces `_print_model_info`. Dynamic dims render as `"?"` instead of `0`. Optional opset row exists in the rendering scaffold but is always passed `None` (no data source).
- QHAS summary keys renamed to renderer vocabulary: `time_us → inference_us`, `graph_execute_us → execute_us`, `percent_utilization → utilization_pct`, `total_dram_read → dram_read_bytes`, `total_dram_write → dram_write_bytes`, `total_vtcm_read → vtcm_read_bytes`, `total_vtcm_write → vtcm_write_bytes`, `peak_vtcm_alloc → vtcm_peak_bytes`. External consumers reading the old keys silently see `KeyError` or `None`.
- `_token_*` strip in QHAS now applied to `op_path` (pre-state, raw `qnn_op` was used). Without this strip, the FR-14 L1 ONNX-primary lookup silently misses in detail mode because production map keys are clean but raw QHAS `qnn_op` carries `_token_1_2`.
- `OpTraceResult.model` widened from `str` to `str | None`.
- `OpTraceResult.to_dict()` is **additive**: existing nested `metadata`/`summary`/`operators`/`statistics`/`artifacts` keys unchanged; new top-level `status` and `error` keys appended.
- `_finalize_output` input-search list prefers `{stem}_{device_category}_ctx.onnx` (e.g. `*_npu_ctx.onnx`), falling back to `{stem}_{provider_short}_ctx.onnx` and then `{stem}_ctx.onnx`. Output filename is still `{stem}_{provider_short}_ctx.onnx`.
- `QNNMonitor` no longer hardcodes the HTP/backend provider options (`backend_path=QnnHtp.dll`, `htp_performance_mode=high_performance`, etc.). Callers pass via `extra_provider_options`; the catalog supplies burst-mode defaults via three-layer merge.
- `QNNMonitor` no longer sets `session.disable_cpu_ep_fallback=1` (intentional change to support `onnxruntime-windowsml` Q/DQ-on-CPU + EPContext-on-QNN partitions; users who relied on the loud failure for QNN-only graphs will now see partial-CPU partitions silently).
- `os.chdir` removed from QNN profiling pipeline; CWD-based schematic fallback is now mtime-gated (CSV mtime − 5 s tolerance) so stale schematics from prior CI runs cannot poison QHAS results with `status="ok"`.

## Internal inconsistencies / smells found in the per-file analyses

- **`session/monitor/__init__.py` is empty** — no `__all__`, no re-exports. Deviates from CLAUDE.md Import Rules. All external callers must reach into `session/monitor/ep_monitor.py`, `session/monitor/qnn_monitor.py`, etc. directly.
- **`model.device` divergence** (above): reads `self._session.device`, not `self._ep_device.device`. Requested vs effective can disagree silently.
- **`benchmark._perf_ctx` is set as an attribute on `PerfBenchmark` but never declared in `__init__`** (`commands/perf.py`). The post-benchmark op-tracing reader uses `getattr(benchmark, "_perf_ctx", None)` defensively, but both simple and monitored paths now stash it. Future refactor risk of silent "no profiling data" with exit 4 if either branch forgets the stash.
- **`_resolve_ep_monitor` only knows QNN and VitisAI** (`commands/perf.py`). OpenVINO is described in the commit body as a "placeholder for parity" but is not wired in `_resolve_ep_monitor` — `--ep openvino --op-tracing basic` raises `RuntimeError("Op-tracing not available for EP 'openvino' …")`.
- **`analyze` CLI not migrated to resolver / typed `EPDevice`** — `commands/analyze.py` is cosmetic-only (lowercase device name, `ov → openvino` in docstring). It still passes raw `ep` and `device` strings straight through to `analyzer.analyze`. Gap relative to sibling CLIs.
- **`build.py` partial migration** — `commands/build.py` calls `resolve_device(ep=None, device=None)` for auto-EP selection but **still passes strings downstream**. Unlike `compile.py`, it does NOT thread `EPDevice` deeper through the build pipeline. Also: it **swallows resolver exceptions broadly** (`except Exception: ... logger.warning ...`), losing the remediation-hint UX. Also: `device` CLI value is **not auto-set** when only `--ep` is omitted — only `ep` gets auto-filled.
- **`WinMLAutoModel` docstring examples stale** (`models/auto.py`): the class docstring still shows `WinMLAutoModel.from_onnx("model.onnx", device="npu")` and `model.to("npu")`. These no longer work — `device=` is gone, `ep_device=` is required, and `.to("npu")` is a no-op.
- **`commands/sys.py:379` stale comment** referencing the removed `_get_available_devices` private function — comment-only, no functional impact.
- **`_EP_NAME_ALIASES` migration-stub in `ep_device.py`** marked for removal post-`feat/update-pkg-deps` merge. Until then, every new casing-mismatch must be added by hand.
- **`get_provider_for_device` deletion duplicates `"cpu" → None` post-mapping** in two places (`config/build.py`, `config/precision.py`), because `default_ep_for_device("cpu") == "CPUExecutionProvider"`, not None. Code smell — either `default_ep_for_device("cpu")` should return None or the post-mapping should be centralized.
- **`session/session.py::_active_session_option_entries` snapshot/restore in `perf()`** appears unused — initialized to `{}` and never written elsewhere. Save/restore dance may be dead weight.
- **`session/session.py::_detect_best_device`, `_get_compile_suggestion`, `_get_install_suggestion`** still exist and reference the old string-device taxonomy. `_detect_best_device` is no longer called by `compile()`. Dead-code candidates.
- **`session/session.py::_ep` legacy alias** explicitly flagged "TODO Task 10: replace consumers and remove" — outstanding cleanup.
- **`session/session.py::_build_op_type_map` is a `@staticmethod` despite not using `cls`/`self`** — should be a module-level free function (parity with `_build_session_options`).
- **`compiler/cli.py` is a secondary entry point** for `winml compiler compile`. It accepts `--ep` from `VALID_EPS` but has no `--device` flag and does not call `resolve_device` at the boundary — diverged from `commands/compile.py` (the top-level CLI) which fully migrated. The two entry points now have different behavior re. EPDevice threading.
- **`run_basic_viewer` in `session/monitor/qnn/viewer.py`** has no caller — dead code at HEAD. Kept for future parity or external consumers.
- **`session/monitor/qnn/viewer.py` is public-by-omission** — not `_`-prefixed (unlike `_internal.py`) but also not re-exported from `qnn/__init__.py`. Inconsistent middle ground.
- **`extract_ep_options` legacy two-letter aliases (`ov`, `vitis`) partially orphaned** (`utils/constants.py`). Appear in `_EP_CLI_PREFIXES` and in `normalize_ep_name`'s `_legacy` dict, but **not** in `utils/cli.py`'s `_EP_CHOICES` — so `--ep ov` is rejected by Click. Inconsistent surface.
- **`DEVICE_TO_DEVICE_TYPE` map (`utils/constants.py`) still uses uppercase keys** (`"CPU"`/`"GPU"`/`"NPU"`). Rest of the codebase migrated to lowercase; any caller grabbing `device` from the lowercase pipeline must `.upper()` before indexing.
- **`available_eps` returns short OR full names depending on source** (`session/ep_registry.py`). Docstring doesn't acknowledge that registration *does* change the set despite the cache claim "hardware/EPs don't change during a process lifetime".
- **`register_ep` does not record its own failures into `self._registration_failures`** — `register_to_ort` does. Callers reading `registration_failures` may see stale entries; asymmetric behavior.
- **`OpTraceResult.status` default is `"ok"`** but the `TraceStatus` docstring describes `"not_run"` as the pre-`__exit__` state. A monitor that fails to set status explicitly will misreport as successful.
- **`commands/compile.py`'s `EPNotDiscovered` install hint is hardcoded `onnxruntime-qnn`** — misleading if the missing EP is OpenVINO/VitisAI/CUDA.
- **`commands/compile.py --list` now requires `resolve_device` to succeed** even for a pure listing. If no EP is registered on the host, `winml compile --list` fails with `EPNotDiscovered`. Regression in pure-CLI listing UX.
- **`compiler/stages/compile.py` three-way precedence priority inversion**: `context.config.get("ep_device")` (a dict) wins over `compile_cfg.ep_device` (the rehydrated object), even though the latter is derived from the former through `from_dict`.
- **`utils/optimum_loader.py` carve-out comment is a contract, not enforcement** — nothing prevents a future mechanical refactor from replacing `provider="CUDAExecutionProvider"` with `default_ep_for_device("gpu")`. A regression test for this site is absent.
- **`analyze/runtime_checker/check_ops.py` has a hardcoded 5-EP `choices` list** with no `CARVE-OUT` comment (unlike the sibling `check_patterns.py` which got the comment in this commit). Future contributors adding a 6th NPU EP will not see this site update.

## Verification evidence captured in commit body

6-command CLI matrix on **Snapdragon X-Elite, ResNet-50**:

| Command | Result |
|---|---|
| `winml perf --ep qnn --device npu` (fp32) | 2.63 ms / 380 s/s |
| `winml perf --ep qnn` (device deduced) | 2.27 ms / 441 s/s |
| `winml perf --device npu` (ep deduced) | 2.35 ms / 425 s/s |
| `winml compile --ep qnn --device npu` (fp32) | `*_qnn_ctx.onnx` + `.bin` produced |
| `winml perf` (compiled ctx) | 2.27 ms / 441 s/s |
| `winml perf` (QDQ on NPU) | 2.75× speedup over fp32 path |

Plus QNN-NPU burst-mode defaults verification: catalog defaults (`htp_performance_mode='burst'` + `htp_graph_finalization_optimization_mode='3'`) deliver roughly +3× ResNet-50 throughput vs empty defaults (5.73 → 1.90 ms avg, 175 → 526 samples/s).

Tests: approximately 720 passing after the squash, with a new architecture regression test at `tests/unit/architecture/test_qnn_imports.py` (PRD NFR-8) enforcing the `qnn._internal` information-hiding boundary, plus `tests/unit/architecture/test_ep_device_import_rule.py` enforcing the "no private `_EP_TO_DEVICE` / `_DEVICE_TO_PROVIDER` / `_SHORT_TO_FULL` outside `session/ep_device.py`" directive.

## Open follow-ups deferred

- **I1: singleton consolidation.** The two `register_execution_provider_library` entry points coexist with symmetric defensive guards; the design fix is explicitly out of scope.
- **QuantSpec doc DRAFT** — "per-variant quantization attached to EPDevice" is DRAFT; do not implement. Hooks exist in `commands/_pre_bench.py` (the "Surface" sub-block placeholder reserves the space) and in `_EP_NAME_ALIASES` (the migration stub).
- **`feat/update-pkg-deps` rebase** — `_EP_NAME_ALIASES` migration stub marked for replacement by `from .ep_path import canonicalize_ep_name` once that branch merges.
- **CUDA / TensorRT runtime not tested** — the commit fixed the `_SHORT_TO_FULL` deduction-path bug for these EPs, but the 6-command verification matrix only exercised QNN on Snapdragon X-Elite.
- **Eval CLI does not expose `--ep`** — `WinMLEvaluationConfig` has no `ep` field.
- **Analyze CLI not migrated to `EPDevice`** — `commands/analyze.py` was cosmetic-only; downstream `analyzer.analyze` still takes loose strings.
- **`build` pipeline partial migration** — `commands/build.py` auto-selects via `resolve_device` but passes strings downstream; `build_hf_model` and the analyze loop still speak `ep: str | None, device: str | None`.
- **`compiler/cli.py` (the secondary CLI) divergence from `commands/compile.py`** — the secondary CLI has no `--device` flag and does not resolve at the boundary.
- **`WinMLAutoModel` docstring examples** still reference `device="npu"` and `model.to("npu")` — both stale.
- **`commands/sys.py:379` stale comment** referencing removed `_get_available_devices` private function.
- **Op-tracing parity for OpenVINO / VitisAI** — `_resolve_ep_monitor` does not wire OpenVINO. Per the commit body, OpenVINO and VitisAI monitors are "placeholders for parity" but only QNN is functional.
- **QHAS schema-drift type rather than bare `KeyError`** — `_require` raises `KeyError`; a typed `QhasSchemaError` would let callers disambiguate.
- **Per-EP install hints** — `commands/compile.py` hardcodes the `EPNotDiscovered` hint to mention `onnxruntime-qnn`. Should be parameterized per the resolved EP.

## How to read this commit (suggested order)

1. **`session/ep_device.py`** — the new contract. Read the module docstring, the `EPDevice` dataclass, then the `EP_DEVICE_SPECS` catalog with its ordering note, then `resolve_device(ep, device="auto"|None)` (the typed entry-point function that returns `EPDevice` and handles `"auto"` internally), and finally `auto_detect_device()` (the free helper that owns the "pick the strongest available device" walk and that `resolve_device` delegates to when `device="auto"`). All other files in the diff either consume or are consumed by this module.
2. **`session/ep_registry.py`** — the registration mechanic. The new `register_ep(ep_name)` method's three-branch behavior (catalog-hit-not-registered with defensive `already_loaded` probe, catalog-hit-registered, catalog-miss fallback to `ort.get_ep_devices()`) is the load-bearing part; everything else is plumbing. Pair-read with `winml.py` for the **symmetric** defensive guard.
3. **`session/session.py`** — lifecycle integration. Read the new `__init__` signature, the free `_build_session_options` function, and especially the `perf()` save-restore lifecycle around the optional `EPMonitor` (the C-2 ordering invariant and the auto-reset-on-different-provider-options dance). The `compile()` method is the Bug A + Bug B fix.
4. **`session/monitor/*`** — the op-tracing rebuild. Start at `ep_monitor.py` (the ABC), then `op_metrics.py` (the data schema), then `qnn_monitor.py` (the concrete implementation), then `qnn/_internal.py` (the parsers and `_require` rollout), then `qnn/viewer.py` (the shell-out), and finally `report.py` (the renderer). Skip `live_display.py` (deleted) and the empty `monitor/__init__.py`.
5. **CLI boundary files** — `commands/compile.py` and `commands/perf.py` are the fully-refactored exemplars. They demonstrate the one-step CLI-boundary pattern (`session.resolve_device(ep, device)` handles `device="auto"` internally and returns an `EPDevice`) and the structured error-UX with `DeviceNotFound`/`EPNotDiscovered` remediation hints.
6. **Everything else** — mostly mechanical migration. The remaining files (commands/eval.py, commands/config.py, commands/sys.py, compiler/cli.py, commands/analyze.py, commands/build.py, utils/cli.py, utils/constants.py, eval/evaluate.py, models/auto.py, models/winml/base.py, compiler/configs.py, compiler/stages/compile.py, config/build.py, config/precision.py, sysinfo/__init__.py, sysinfo/hardware.py, analyze/analyzer.py, analyze/runtime_checker/check_ops.py, analyze/pattern/check_patterns.py, utils/optimum_loader.py, and the deleted optracing/* and sysinfo/device.py) are 5–50 line changes that follow patterns established by the files in steps 1–5.
