# EP Registration & Provider Options — Design Doc

**Version**: 1.0
**Date**: 2026-05-15
**Status**: Draft
**Module**: session
**Companion-To**: [`2_coreloop.md`](monitor/2_coreloop.md) — the monitor pipeline that consumes the bound EP
**Depends-On**: [`../../../ep-path-design.md`](../../ep-path-design.md), [`2026-05-13-ep-device-spec-design.md`](2026-05-13-ep-device-spec-design.md), [`2026-05-11-ep-device-refactor.md`](2026-05-11-ep-device-refactor.md)
**Builds-On**: prior session refactor docs (the [audit trail at `2026-05-13-remaining-issues.md`](2026-05-13-remaining-issues.md))

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Scope](#2-scope)
- [3. Two-Stage Model](#3-two-stage-model)
- [4. Legacy vs Plugin-EP API](#4-legacy-vs-plugin-ep-api)
- [5. Stage 1: EP Registration + Plugin Discovery](#5-stage-1-ep-registration--plugin-discovery)
- [6. Stage 2: Device-Handle Selection](#6-stage-2-device-handle-selection)
  - [6.4 Registration-aware deduction (REQUIRED)](#64-registration-aware-deduction-required)
- [7. Provider Options Reference](#7-provider-options-reference)
- [8. Catalog Status](#8-catalog-status)
- [9. Empirical Evidence](#9-empirical-evidence)
- [10. Open Questions](#10-open-questions)
- [11. Appendix](#11-appendix)

---

## 1. Executive Summary

`WinMLSession` selects an Execution Provider (EP) and a device in two stages: **(1) register the EP plugin DLL** so ORT can enumerate every `OrtEpDevice` it claims to support, then **(2) pick one specific `OrtEpDevice` handle and bind it** to the session. The "device" is the handle, not a string in `provider_options`.

This doc unifies what the existing design docs cover piecewise:
- [`ep-path-design.md`](../../ep-path-design.md) explains **Stage 1**: how the codebase walks `EP_PATH` to find plugin DLLs from PyPI / NuGet / MSIX / filesystem and calls `register_execution_provider_library`.
- [`2026-05-13-ep-device-spec-design.md`](2026-05-13-ep-device-spec-design.md) and [`2026-05-11-ep-device-refactor.md`](2026-05-11-ep-device-refactor.md) explain **Stage 2**: the `WinMLEPDeviceSpec` catalog, the `WinMLEPDevice` typed descriptor, and the deterministic handle-pick replacing the old non-deterministic `_find_ep_device`.
- This doc connects the two stages, contrasts the legacy ORT API mental model with the plugin-EP API the codebase actually uses, and specifies how `provider_options` are layered (catalog → user → monitor).

**Key result, validated empirically on ORT 1.24.5**: under `add_provider_for_devices([handle], options)`, the OrtEpDevice handle drives device selection. Routing keys like OpenVINO's `device_type` and QNN's `backend_type` / `backend_path` are **ignored** (OpenVINO logs a warning; QNN crashes natively in some ORT versions). `provider_options` should carry only **tuning** keys, never **routing** keys.

## 2. Scope

**In scope:**
- How `register_execution_provider_library` interacts with `add_provider_for_devices` to produce a working `InferenceSession`.
- Layered semantics of `provider_options` (catalog defaults → user config → monitor overrides).
- The set of `provider_options` keys that are meaningful per (EP, device) under the plugin-EP API.
- The current `EP_DEVICE_SPECS` catalog and what it omits.

**Out of scope:**
- Compile-time options (`ep.context_enable` etc.) — those go through `SessionOptions.AddConfigEntry`, not `provider_options`. Documented in [`2026-05-11-ep-device-refactor.md`](2026-05-11-ep-device-refactor.md) §3.
- The `WinMLEPMonitor` ABC and per-EP monitor concrete classes — documented in [`2_coreloop.md`](monitor/2_coreloop.md).
- Silicon-specific QNN tuning (`soc_model`, `htp_arch`) — documented status: requires runtime detection; see §10.

## 3. Two-Stage Model

```
            ┌─────────────────────────────────────────────────┐
            │  Stage 1: Register the EP plugin DLL with ORT   │
            │                                                  │
            │  ort.register_execution_provider_library(        │
            │      "OpenVINOExecutionProvider",                │
            │      "/path/to/onnxruntime_providers_openvino_   │
            │       plugin.dll")                               │
            │                                                  │
            │  Effect: ORT can now enumerate every             │
            │  OrtEpDevice this EP claims to support.          │
            └─────────────────────────────────────────────────┘
                            │
                            │  ort.get_ep_devices()
                            ▼
            ┌─────────────────────────────────────────────────┐
            │  All OrtEpDevices for this EP                    │
            │                                                  │
            │  [ep='OpenVINOExecutionProvider' type=NPU ...]   │
            │  [ep='OpenVINOExecutionProvider' type=GPU ...]   │
            │  [ep='OpenVINOExecutionProvider' type=GPU ...]   │
            │  [ep='OpenVINOExecutionProvider' type=CPU ...]   │
            └─────────────────────────────────────────────────┘
                            │
                            │  resolve_device(ep, device)
                            │   • filter by device.type
                            │   • dedup by (vendor_id, device_id)
                            │   • pick the first remaining
                            ▼
            ┌─────────────────────────────────────────────────┐
            │  ONE OrtEpDevice handle + WinMLEPDevice descriptor    │
            │                                                  │
            │  WinMLEPDevice(ep='OpenVINOExecutionProvider',        │
            │           device='gpu',                          │
            │           vendor_id=0x8086,                      │
            │           device_id=0x64a0,                      │
            │           vendor='Intel Corporation')            │
            └─────────────────────────────────────────────────┘
                            │
                            │  _build_session_options:
                            │     re-filter, _build_provider_options
                            ▼
            ┌─────────────────────────────────────────────────┐
            │  Stage 2: Bind the handle to the session         │
            │                                                  │
            │  so.add_provider_for_devices([handle], options)  │
            │                                                  │
            │  options: tuning only (no device_type,           │
            │           no backend_type — routing is in the    │
            │           handle).                               │
            └─────────────────────────────────────────────────┘
                            │
                            ▼
            ┌─────────────────────────────────────────────────┐
            │  ort.InferenceSession(model, so)                 │
            └─────────────────────────────────────────────────┘
```

**Stage 1 is per-process** (the DLL loads once into ORT's runtime). **Stage 2 is per-session** (each `WinMLSession` picks one handle and binds it). The two stages are decoupled: registration alone doesn't pick a device, and you can register many EPs and pick any one (or many) per session.

## 4. Legacy vs Plugin-EP API

ONNX Runtime supports two coexisting APIs for attaching EPs to a session. The codebase exclusively uses the plugin-EP API.

| Concern | Legacy API | Plugin-EP API (`add_provider_for_devices`) |
|---|---|---|
| Python call | `sess.set_providers([("OpenVINOExecutionProvider", options)])` or `sess.set_providers(["OpenVINOExecutionProvider"])` | `so.add_provider_for_devices([ep_device], options)` |
| C API name | `SessionOptionsAppendExecutionProvider_*` | `SessionOptionsAppendExecutionProvider_V2` |
| Device selection mechanism | String key in `provider_options` (`device_type=NPU`, `backend_type=htp`) | OrtEpDevice handle passed to the call |
| Source of truth for "which device" | The string the user typed | The handle's `device.type` + `vendor_id` + `device_id` (read from system at registration) |
| Bundled vs plugin EP | Both. Bundled EPs (CPU, DML) used built-in; plugin EPs (QNN, OpenVINO, VitisAI) needed `register_execution_provider_library` first. | Both via the same handle path. Bundled EPs appear in `ort.get_ep_devices()` without `register_execution_provider_library`. |
| Failure mode if you pass the routing string anyway | The string drives selection (intended behavior) | Routing string is **ignored**. OpenVINO logs a WARN; QNN crashes natively in some ORT versions (see §9). |

**Why the codebase uses the plugin-EP API exclusively**: it lets a single OrtEpDevice represent "EP × specific hardware device" deterministically, so `WinMLSession` can serialize an `WinMLEPDevice` descriptor and round-trip it across CLI / compile pipeline / perf monitor without any string-based device-routing logic. The non-determinism of `_find_ep_device` documented in [`2026-05-11-ep-device-refactor.md`](2026-05-11-ep-device-refactor.md) §1 was caused by string-based routing returning the first matching device arbitrarily (e.g., CPU when the user meant NPU). Handle-based routing removes the ambiguity.

## 5. Stage 1: EP Registration + Plugin Discovery

Authoritative spec: [`docs/ep-path-design.md`](../../ep-path-design.md) and [`docs/ep-path-msix-source.md`](../../ep-path-msix-source.md). Summary here only.

### 5.1 The `EP_PATH` walker

The codebase exposes a single `discover_eps()` function that walks an ordered `EP_PATH: list[EpSource]` (analogous to the shell's `PATH`). Each `EpSource` is a typed entry covering one of four origins:

| `EpSource` subclass | Origin | Example |
|---|---|---|
| `PyPiSource` | pip-installed plugin wheel | `onnxruntime-ep-openvino 1.4.1` ships `onnxruntime_providers_openvino_plugin.dll` |
| `NuGetSource` | NuGet-cached plugin package (`~/.nuget/packages/`) | `Intel.ML.OnnxRuntime.EP.OpenVINO` |
| `MsixPackageSource` | Microsoft Store / Windows-Workloads MSIX | `WindowsWorkload.EP.Intel.OpenVINO.1.8` |
| `FilesystemSource` | Vendor-installer drop dir (`%RYZEN_AI_INSTALLATION_PATH%`, `%NVIDIA_TRT_RTX_EP%`) or arbitrary path via `MODELKIT_EP_PATH` | `D:\src\onnxruntime\build\Release\` |
| `WinMlCatalogSource` | Windows App SDK `ExecutionProviderCatalog` runtime API | MSIX EPs whose path is decided by OS package manager |

`discover_eps()` returns `dict[ep_name -> (Path, EpSource)]` (first-match-wins) and the inventory variant `discover_eps(return_shadowed=True)` returns all hits per EP for the `winml sys --list-ep` inventory CLI.

### 5.2 `WinMLEPRegistry.register_ep()`

The post-discovery sink. Given an EP name (e.g., `"OpenVINOExecutionProvider"`):

1. If `ep_name` is already loaded (per `ort.get_ep_devices()`): no-op short-circuit. **Mandatory** — ORT's `register_execution_provider_library` is non-idempotent at the C++ layer and a second call for the same DLL calls `exit(127)` natively (this is the dual-singleton crash defended against in [`2026-05-13-t6-analyze-crash-diagnostic.md`](2026-05-13-t6-analyze-crash-diagnostic.md)).
2. Otherwise look up the DLL path via `self._ep_paths` (populated by `_discover_eps` from the ep_path walker).
3. Call `ort.register_execution_provider_library(ep_name, dll_path)`.
4. Return all `OrtEpDevice` handles that ORT now reports for this EP — this is what Stage 2 filters down to one.

Bundled EPs (CPU, DML) bypass registration entirely: they appear in `ort.get_ep_devices()` without any plugin DLL load.

## 6. Stage 2: Device-Handle Selection

Authoritative spec: [`2026-05-13-ep-device-spec-design.md`](2026-05-13-ep-device-spec-design.md). Summary + interaction with Stage 1.

### 6.1 Catalog: `EP_DEVICE_SPECS`

A `tuple[WinMLEPDeviceSpec, ...]` defined in [`session/ep_device.py`](../../../src/winml/modelkit/session/ep_device.py) at module scope. **Order encodes deduction preference**: walking the catalog finds the first matching entry per device (`npu` → `QNNExecutionProvider`, `gpu` → `DmlExecutionProvider`, `cpu` → `CPUExecutionProvider`).

Each entry is:

```python
@dataclass(frozen=True)
class WinMLEPDeviceSpec:
    ep: str                                              # canonical full EP name
    device: str                                          # one of {npu, gpu, cpu}
    default_provider_options: Mapping[str, str] = field(default_factory=dict)
```

The catalog contains 13 entries today (one row per `(ep, device)` target). OpenVINO has 3 (npu/gpu/cpu), QNN has 3 (npu/gpu/cpu), and the rest are single-target EPs.

### 6.2 `resolve_device(ep, device)` → `WinMLEPDevice`

The pure-function resolver. Given a user-supplied `(ep, device)` pair (either may be `None` or `device="auto"`):

```python
def resolve_device(ep: str | None = None, device: str | None = None) -> WinMLEPDevice:
    # 1. Deduction phase
    if device is None or device.lower() == "auto":
        device = auto_detect_device()              # walks sysinfo + catalog for best match
    if ep is None:
        ep = short_ep_name(default_ep_for_device(device))   # see §6.4 — MUST be registration-aware

    # 2. Resolution phase
    ep_full = expand_ep_name(ep)                   # qnn -> QNNExecutionProvider
    devices = registry.register_ep(ep_full)         # ← Stage 1 invoked here
    matching = [d for d in devices if d.device.type.name.lower() == device]
    deduped = dedup_by(matching, key=(vendor_id, device_id))
    if not deduped: raise DeviceNotFound(...)
    if len(deduped) > 1: raise AmbiguousMatch(...)
    chosen = deduped[0]

    return WinMLEPDevice(ep=ep_full, device=device,
                    vendor_id=chosen.device.vendor_id,
                    device_id=chosen.device.device_id,
                    vendor=chosen.device.vendor)
```

The returned `WinMLEPDevice` is a plain-data descriptor — JSON-serializable, no ORT runtime dependency. It flows through CLI args, compile config, perf monitor without re-resolving.

### 6.3 `_build_session_options()` → `add_provider_for_devices`

At session-build time, the descriptor is rehydrated to a concrete handle:

```python
def _build_session_options(ep_device, ep_config, ep_monitor):
    so = ort.SessionOptions()
    devices = registry.register_ep(ep_device.ep)             # idempotent (short-circuits)
    matching = [d for d in devices
                if d.device.type.name.lower() == ep_device.device
                and d.device.vendor_id == ep_device.vendor_id
                and d.device.device_id == ep_device.device_id]
    # dedup for hosts that double-list the same device handle
    if len({(d.ep_name, d.device.type.name,
             d.device.vendor_id, d.device.device_id) for d in matching}) == 1:
        matching = matching[:1]
    options = _build_provider_options(ep_device, ep_config, ep_monitor)
    so.add_provider_for_devices([matching[0]], options)      # ← bind
    return so
```

The dedup step (added 2026-05-15) is required for hosts that report the same OrtEpDevice twice with bit-identical `(vendor_id, device_id)`, observed on Intel iGPU configurations.

### 6.4 Registration-aware deduction (REQUIRED)

**Requirement.** Whenever the codebase deduces an EP from a device alone (`default_ep_for_device(device)` and every caller that does the same walk of `EP_DEVICE_SPECS`), the deduction MUST filter by EPs that are actually registered on the host (i.e., present in `available_eps()` from `session/ep_registry.py`). The static catalog order encodes *preference among installed EPs*, not *unconditional defaults*.

**Why.** `EP_DEVICE_SPECS` is ordered with QNN first for `npu`, DML first for `gpu` (see §6.1). A pure first-match walk returns the catalog default regardless of what is installed. On an OpenVINO-only dev box (no QNN wheel, no Snapdragon), this produces:

```
default_ep_for_device("npu")  →  "QNNExecutionProvider"    # WRONG: QNN is not registered
```

Downstream this propagates into `compile_provider`, `WinMLCompileConfig`, and `resolve_device`'s deduction branch — the build pipeline ends up pointed at an EP that isn't on disk.

**Contract.** A registration-aware deduction walks `EP_DEVICE_SPECS` in order and returns the first `spec.ep` that satisfies `spec.ep in available_eps()`. If no catalog entry for the requested device has a registered EP, return `None` and let the caller decide (raise, fall back to CPU, etc.). The pattern is already correct in `auto_detect_device()` (ep_device.py:286) — the same filter must apply to per-device EP deduction.

```python
# Sketch — final shape TBD; what matters is the FILTER, not the function name.
def default_ep_for_device(device: str) -> str | None:
    eps = available_eps()                            # from ep_registry
    return next(
        (s.ep for s in EP_DEVICE_SPECS
         if s.device == device and s.ep in eps),
        None,
    )
```

**Call sites that must observe this contract** (today they don't):

| Site | Symptom |
|---|---|
| `session/ep_device.py:242` `default_ep_for_device` | Returns static-catalog default ignoring registration |
| `session/ep_device.py:379` `resolve_device` device-only branch | Deduces unregistered EP → registration fails later with a confusing message |
| `config/precision.py:275` (in `resolve_precision`) | `compile_provider` carries the unregistered short name into `PrecisionPolicy` |
| `config/build.py:612` (in `generate_hf_build_config`, auto/auto path) | `WinMLCompileConfig.for_provider(...)` constructed against an unregistered EP |

**Out of scope.** This requirement only constrains the **deduction default**. Callers may still ask for an explicit EP that is not registered — `resolve_device("qnn", "npu")` on a non-Snapdragon box must continue to raise loudly (today: via `register_ep` registration failure or `DeviceNotFound`). Registration-awareness changes the *default*, not the *explicit* path.

## 7. Provider Options Reference

### 7.1 Three-layer merge

```python
def _build_provider_options(ep_device, ep_config, ep_monitor) -> dict[str, str]:
    options = _ep_defaults(ep_device)                    # Layer 1: catalog preset
    if ep_config is not None:
        options.update(ep_config.provider_options)       # Layer 2: user
    if ep_monitor is not None:
        options.update(ep_monitor.get_provider_options())# Layer 3: monitor (wins)
    return options
```

- **Layer 1 (catalog)**: `WinMLEPDeviceSpec.default_provider_options`. Per-target tuning measured by us.
- **Layer 2 (user)**: `EPConfig.provider_options` from a JSON config (when CLI is invoked with `-c`).
- **Layer 3 (monitor)**: `WinMLEPMonitor.get_provider_options()`. For `QNNMonitor` this carries `profiling_level=optrace` etc.; monitors win because tracing correctness depends on their options reaching the EP.

### 7.2 Routing keys — IGNORED under the plugin-EP API

These keys exist in each EP's legacy-API documentation but **must not be set** when calling `add_provider_for_devices`:

| EP | Routing key | Plugin-EP API behavior |
|---|---|---|
| OpenVINOExecutionProvider | `device_type` | Logged as ignored: `ov_plugin_options.cc:130 ... Provider option device_type is ignored. Please pass one of the enumerated EP device to AppendExecutionProvider_V2 for device selection.` |
| QNNExecutionProvider | `backend_type` | Native `exit(127)` in ORT 1.23.5 (project finding, commit `a509a67`); behavior on 1.24.5 not retested |
| QNNExecutionProvider | `backend_path` | Same as `backend_type`. Mutually exclusive with it in the legacy API. |

The handle's `device.type.name` (NPU/GPU/CPU) drives EP routing automatically. The handle's `vendor_id` and `device_id` further pin to a specific piece of silicon (Intel iGPU 0x64a0 vs Intel NPU 0x643e on the same host).

### 7.3 Tuning keys — meaningful under the plugin-EP API

For each EP, the tuning keys that the catalog or user should set. Authoritative source for each EP is its [ORT EP doc page](https://onnxruntime.ai/docs/execution-providers/).

#### OpenVINO (per device)

OV 2025.3 / ORT 1.23 onward: `load_config` (a JSON-string of nested OV runtime properties) is the canonical channel. Legacy individual keys (`precision`, `num_of_threads`, `num_streams`, `cache_dir`, `model_priority`, `enable_qdq_optimizer`) are DEPRECATED. Doc reference: [OpenVINO EP `#load_config`](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#load_config).

```python
# Recommended baseline per device (load_config wraps OV runtime properties)
{
    "OpenVINO/cpu": {
        "load_config": '{"CPU":{"PERFORMANCE_HINT":"LATENCY",'
                       '"NUM_STREAMS":"1"}}',
    },
    "OpenVINO/gpu": {
        "load_config": '{"GPU":{"PERFORMANCE_HINT":"LATENCY",'
                       '"INFERENCE_PRECISION_HINT":"f16",'
                       '"CACHE_DIR":"./.ov_cache"}}',
    },
    "OpenVINO/npu": {
        "load_config": '{"NPU":{"NPU_QDQ_OPTIMIZATION":"YES"}}',
        # plus, for dynamic-shape models on NPU:
        # "reshape_input": "input_name[1..16,3,224,224]",
    },
}
```

Notes:
- `INFERENCE_PRECISION_HINT:f16` matters on GPU for throughput on Intel iGPU; not applicable on NPU (NPU is FP16-only) or CPU (FP32-only).
- `CACHE_DIR` caches compiled blob across runs. Path is user-configurable.
- `NPU_QDQ_OPTIMIZATION` is OV's NPU-specific QDQ-pass switch; required for quantized models targeting NPU.
- `reshape_input` is REQUIRED on NPU for dynamic-shape models (NPU has no dynamic-shape support).

#### QNN (per device)

Doc reference: [QNN EP `#configuration-options`](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html#configuration-options).

```python
# Recommended baseline for QNN-NPU (HTP) production
{
    "QNN/npu": {
        "htp_performance_mode": "burst",                       # vs default — measured +3× ResNet-50
        "htp_graph_finalization_optimization_mode": "3",       # 0-3, 3 = best graph / slowest compile
        "enable_htp_fp16_precision": "1",                      # default "1" (explicit for clarity)
        "offload_graph_io_quantization": "1",                  # default "1"; Q/DQ on CPU EP, body on HTP
        "vtcm_mb": "8",                                        # HTP scratch RAM; default 0 (SDK picks)
        "qnn_context_priority": "normal",
        # Silicon-specific — runtime detection required, NOT in catalog:
        # "soc_model": "60",    # 60 = Snapdragon X (per QNN SDK)
        # "htp_arch": "73",     # 73 = X-class HTP arch
    },
    "QNN/gpu": {},  # backend_type=gpu is routing — skipped under plugin-EP API
    "QNN/cpu": {},  # backend_type=cpu — same
}
```

Notes:
- `htp_performance_mode=burst` is the largest single perf win on QNN-NPU; measured +3× throughput on ResNet-50.
- `htp_graph_finalization_optimization_mode=3` trades compile time for runtime; default `0` is slowest at runtime.
- `vtcm_mb=8` is a safe default for ResNet-50-class models on Snapdragon X HTP; tune by silicon.
- `soc_model` and `htp_arch` are silicon enums from the QNN SDK headers. The QNN doc only lists Snapdragon X values (`60`, `73`); other silicon needs SDK header lookup. **These MUST come from runtime detection, not a static catalog**, because the same wheel may be used across multiple Snapdragon revisions.

## 8. Catalog Status

Snapshot of `EP_DEVICE_SPECS` as of 2026-05-15:

| Catalog entry | Currently ships | Per-doc recommendation | Gap |
|---|---|---|---|
| `QNN/npu` | `{"htp_performance_mode":"burst", "htp_graph_finalization_optimization_mode":"3"}` | + `vtcm_mb=8`, `qnn_context_priority=normal`; runtime-detect `soc_model`, `htp_arch` | partial |
| `QNN/gpu` | `{}` | `{}` (backend_type routing is skipped under plugin-EP API) | OK |
| `QNN/cpu` | `{}` | `{}` (same as above) | OK |
| `OpenVINO/npu` | `{}` | `{"load_config": '{"NPU":{"NPU_QDQ_OPTIMIZATION":"YES"}}'}` | missing NPU-specific QDQ pass |
| `OpenVINO/gpu` | `{}` | `{"load_config": '{"GPU":{"PERFORMANCE_HINT":"LATENCY","INFERENCE_PRECISION_HINT":"f16","CACHE_DIR":"./.ov_cache"}}'}` | missing latency-mode hint + FP16 + cache |
| `OpenVINO/cpu` | `{}` | `{"load_config": '{"CPU":{"PERFORMANCE_HINT":"LATENCY","NUM_STREAMS":"1"}}'}` | missing latency-mode hint |
| `DML/gpu` | `{}` | (no public ORT doc for DML provider_options) | unknown |
| `CPU/cpu` | `{}` | (bundled EP — no tuning needed) | OK |
| `VitisAI/npu`, `MIGraphX/gpu`, `Tensorrt/gpu`, `CUDA/gpu`, `NvTensorRtRtx/gpu` | `{}` | not yet measured by us | future work |

The catalog is **architecturally correct** (carries no routing keys, three-layer merge wired) but **incompletely tuned** (most entries are `{}` and should carry doc-recommended baselines).

## 9. Empirical Evidence

Tests run 2026-05-15 on this branch (post-rebase, `c48c04b` + `57d7a0a` + uncommitted refactor).

### 9.1 OpenVINO: handle wins over `device_type`

Test fixture: `convnext/model_opt.onnx`. Each case binds the GPU handle but tries a different `device_type` string in `provider_options`. If the string drove routing, case C would run at CPU latency (~100 ms); if the handle drove routing, all three cases would run at GPU latency (~8 ms).

```
Handle = GPU  (vendor_id=0x8086 device_id=0x64a0)
  A) options={}                       : avg 8.08 ms / 10 iters
  B) options={'device_type':'GPU'}    : avg 7.81 ms / 10 iters
  C) options={'device_type':'CPU'}    : avg 7.79 ms / 10 iters
```

Cases B and C both produced an ORT warning at session-create:

```
W:onnxruntime:, ov_plugin_options.cc:130
  onnxruntime::openvino_ep_plugin::OpenVINOEpPluginOptions::ParseProviderOptions:
  Provider option device_type is ignored. Please pass one of the enumerated EP device
  to AppendExecutionProvider_V2 for device selection.
```

This is authoritative — ORT's own source logged the warning. The handle wins. Routing keys are silently swallowed (with a one-time WARN).

### 9.2 QNN: documented in commit body (not retested)

The squash commit `a509a67` documents: "second call previously crashed the process natively (exit(127), STATUS_DLL_NOT_FOUND)" when `backend_type` was set under `add_provider_for_devices` on ORT 1.23.5. The mitigation was the symmetric defensive guard in `winml.py` + `session/ep_registry.py` (probe `get_ep_devices()` before any `register_execution_provider_library` call).

QNN is not installed on the current Intel host; the crash behavior on ORT 1.24.5 has not been retested. Catalog avoids the issue by shipping `{}` for QNN-GPU/QNN-CPU and only tuning keys (no `backend_*`) for QNN-NPU.

### 9.3 Handle-driven routing — full table

```
--device npu
  resolved WinMLEPDevice    : WinMLEPDevice(ep='OpenVINOExecutionProvider', device='npu',
                                  vendor_id=32902, device_id=25662,
                                  vendor='Intel Corporation')
  bound OrtEpDevice    : ep_name='OpenVINOExecutionProvider' type=NPU
                         vendor_id=0x8086 device_id=0x643e
  provider_options sent: {}

--device gpu
  resolved WinMLEPDevice    : WinMLEPDevice(ep='OpenVINOExecutionProvider', device='gpu',
                                  vendor_id=32902, device_id=25760,
                                  vendor='Intel Corporation')
  bound OrtEpDevice    : ep_name='OpenVINOExecutionProvider' type=GPU
                         vendor_id=0x8086 device_id=0x64a0
  provider_options sent: {}

--device cpu
  resolved WinMLEPDevice    : WinMLEPDevice(ep='OpenVINOExecutionProvider', device='cpu',
                                  vendor_id=32902, device_id=7, vendor='Intel')
  bound OrtEpDevice    : ep_name='OpenVINOExecutionProvider' type=CPU
                         vendor_id=0x8086 device_id=0x0007
  provider_options sent: {}
```

Identical structure across all three devices: the catalog `default_provider_options` is `{}` for every OpenVINO entry, no `EPConfig` was passed, no monitor was attached → final `provider_options = {}`. ORT routes by handle only.

## 10. Open Questions

1. **Strip-known-routing-keys helper** — should `_build_provider_options` actively strip `device_type` (OV) and `backend_type` / `backend_path` (QNN) before calling `add_provider_for_devices`, to silence the warning and prevent the QNN crash for users who migrate from the legacy API with stale options? Or rely on user discipline + the warning?
2. **QNN silicon detection** — `soc_model` / `htp_arch` are required for production QNN-NPU but are silicon-specific. Should the catalog grow a `provider_options_fn(ep_device) -> dict` callable for runtime-detected values, or should detection live in `_build_provider_options`?
3. **Catalog completeness** — should the catalog ship the doc-recommended baselines for OpenVINO (Layer 1 `load_config`) and the missing QNN tuning keys (`vtcm_mb`, `qnn_context_priority`)? This is non-breaking but changes the perf characteristics for users who currently get `{}`.
4. **DML provider_options** — ORT docs publish nothing on DmlExecutionProvider's `provider_options` keys. Source inspection needed. Current catalog ships `{}`, which may be all that's available.
5. **OpenVINO under plugin-EP API: does `load_config` work?** — the OV doc page documents only the legacy API; whether the plugin-EP factory honors `load_config` requires verification. Empirical test: bind GPU handle with `load_config={"GPU":{"PERFORMANCE_HINT":"LATENCY"}}` and measure latency delta vs `{}`. If no delta, `load_config` is being ignored too.
6. **Registration-aware deduction rollout (see §6.4)** — `default_ep_for_device` and the device-only branch of `resolve_device` currently return static-catalog defaults that may not be registered (e.g. QNN on an OpenVINO-only box). Decide: (a) bake the `available_eps()` filter into `default_ep_for_device` itself, or (b) introduce a sibling `default_available_ep_for_device(device)` and switch the four call sites listed in §6.4. The pure-catalog helper is still useful for sysinfo/inventory views.

## 11. Appendix

### 11.1 Glossary

| Term | Meaning |
|---|---|
| **EP** | Execution Provider — an ORT plugin that knows how to partition + run an ONNX graph on a specific hardware family. |
| **OrtEpDevice** | A handle returned by `ort.get_ep_devices()`, encoding `(ep_name, device.type, vendor_id, device_id, vendor)`. Uniquely identifies "this EP on this specific hardware device." |
| **WinMLEPDevice** | Project-defined plain-data descriptor (frozen dataclass) at `session/ep_device.py`. Captures the OrtEpDevice's fields without holding the handle — JSON-serializable. |
| **WinMLEPDeviceSpec** | Project-defined catalog entry (frozen dataclass) — a static template `(ep, device, default_provider_options)`. The catalog `EP_DEVICE_SPECS` is a tuple of these. |
| **Plugin-EP API** | The newer ORT API (`add_provider_for_devices` / `SessionOptionsAppendExecutionProvider_V2`) that takes OrtEpDevice handles instead of EP-name strings. |
| **Legacy API** | The older ORT API (`sess.set_providers([("EP", options)])` / `SessionOptionsAppendExecutionProvider`) that takes EP-name strings + provider_options for both routing and tuning. |
| **Bundled EP** | An EP statically linked into the ORT wheel (CPU, DML). Appears in `ort.get_ep_devices()` without `register_execution_provider_library`. |
| **Plugin EP** | An EP shipped as a separate DLL (QNN, OpenVINO, VitisAI, etc.). Requires `register_execution_provider_library` to make ORT aware of it. |
| **Routing key** | A `provider_options` key whose value selects which device (legacy API). Ignored by the plugin-EP API. |
| **Tuning key** | A `provider_options` key whose value tunes how the EP runs on a given device (perf modes, cache dirs, FP16 hints). Honored by both APIs. |

### 11.2 References

- [`docs/ep-path-design.md`](../../ep-path-design.md) — Stage 1 design (plugin discovery)
- [`docs/ep-path-msix-source.md`](../../ep-path-msix-source.md) — MSIX source + `winml sys --list-ep`
- [`docs/design/session/2026-05-11-ep-device-refactor.md`](2026-05-11-ep-device-refactor.md) v1.2 — Stage 2 design, WinMLEPDevice descriptor
- [`docs/design/session/2026-05-13-ep-device-spec-design.md`](2026-05-13-ep-device-spec-design.md) — WinMLEPDeviceSpec catalog design
- [`docs/design/session/2026-05-13-remaining-issues.md`](2026-05-13-remaining-issues.md) — landing page for the session refactor audit trail
- [`docs/design/session/monitor/2_coreloop.md`](monitor/2_coreloop.md) — WinMLEPMonitor pipeline (Layer 3 of the provider_options merge)
- [OpenVINO EP — ONNX Runtime docs](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)
- [QNN EP — ONNX Runtime docs](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- ORT source — `onnxruntime/core/providers/openvino/ov_plugin_options.cc` (the warning at line 130 quoted in §9.1)

### 11.3 Document History

| Version | Date | Author | Change |
|---|---|---|---|
| 1.0 | 2026-05-15 | session refactor team | Initial draft consolidating Stage 1 + Stage 2 design and provider_options reference. Captures empirical evidence from this session: OpenVINO `device_type` is ignored under plugin-EP API; handle-driven routing verified across NPU/GPU/CPU on Intel host. |

### 11.4 Code Reference Map

| Concern | File | Symbol |
|---|---|---|
| Catalog | `src/winml/modelkit/session/ep_device.py` | `EP_DEVICE_SPECS`, `WinMLEPDeviceSpec` |
| Resolver | `src/winml/modelkit/session/ep_device.py` | `resolve_device`, `auto_detect_device` |
| EP-path walker | `src/winml/modelkit/ep_path.py` | `discover_eps`, `EP_PATH`, `EpSource` subclasses |
| Plugin registration | `src/winml/modelkit/session/ep_registry.py` | `WinMLEPRegistry`, `register_ep` |
| Singleton-safe registration | `src/winml/modelkit/winml.py` | `WinML.register_execution_providers` (defensive guard) |
| Session-options builder | `src/winml/modelkit/session/session.py` | `_build_session_options`, `_build_provider_options`, `_ep_defaults` |
| MSIX inventory CLI | `src/winml/modelkit/commands/sys.py` | `_gather_ep_info`, `winml sys --list-ep` |
