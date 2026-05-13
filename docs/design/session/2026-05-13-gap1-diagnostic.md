# Gap #1 Diagnostic: `winml perf -m microsoft/resnet-50 --ep qnn --device npu` exits 127

**Date**: 2026-05-13
**Branch**: `feat/op-tracing-refactor` (HEAD: `db39b80d`)
**Working dir**: `D:\BYOM\ModelKit_PRs\op_tracing`
**Status**: ROOT CAUSE FOUND — exit 127 is a native ORT crash from double-registration of the QNN EP DLL

---

## Test Matrix

| # | Test | Command / Script | Exit | Elapsed | Finding |
|---|------|-----------------|------|---------|---------|
| 1 | `winml perf` full run | `uv run winml perf -m microsoft/resnet-50 --ep qnn --device npu -v` | **127** | ~54s | Crashes after analyze loop |
| 2 | `winml compile` on QDQ model | `uv run winml compile -m imgcls_..._quantized.onnx --ep qnn --device npu` | 0 | 1.98s | Compile works in isolation |
| 3 | `ort.ModelCompiler` no-EP parent | Direct call with empty SessionOptions in parent process | **127** | <1s | Confirmed native crash vector |
| 4 | v200 venv `ort.get_ep_devices()` | `D:/QC/ORT_QNN/v200/venv_200/Scripts/python.exe` ORT 1.24.4 | 0 | — | v200 sees only CPU EP (no WinML catalog) |
| 5 | `InferenceSession` on QDQ ctx | Load `imgcls_..._quantized_npu_ctx.onnx` with QNN EP | 0 | 2.7s | Session loads with QNN+CPU providers |
| 6 | Inference on compiled QDQ | `session.run({'pixel_values': np.random.randn(1,3,224,224)})` | 0 | — | Outputs `{logits: (1,1000) float32}` |
| 7 | Double QNN DLL registration | `ort.register_execution_provider_library(qnn, dll)` twice in same process | **127** | <1s | **Root cause isolated** |

---

## Root Cause

**`ort.register_execution_provider_library('QNNExecutionProvider', dll_path)` called twice in the same parent process causes a native exit 127.**

### Crash Sequence

1. **Analyze loop starts** (inside `build_hf_model` → `run_optimize_analyze_loop` → `_run_analyze_loop` → `analyze_onnx`)
2. **First node**: `RuntimeCheckerQuery.run_for_node()` → Phase 3: QNN rules ZIP missing → `_try_local_ep_check()` → `_is_ep_available_locally()` → **`winml.register_execution_providers(ort=True)`** → calls `WinML()._register_execution_providers()` → `ort.register_execution_provider_library('QNNExecutionProvider', dll_path)` — **FIRST REGISTRATION** in `WinML` singleton
3. **All 122 nodes** processed via `ResilientRunner`: each child spawns, calls `winml.add_ep_for_device()` which does NOT register the DLL (child's `ort.get_ep_devices()` returns only CPU because `WinML` singleton is not initialized in spawned children) → `ort.ModelCompiler(no-EP-so, bytes, ERROR_IF_NO_NODES_COMPILED)` → child exits 127 → parent catches `BrokenProcessPool` → returns `{"success": False}`. Each child crash takes ~0.4–0.6s. Total ~50s for 122 nodes.
4. **Analyze completes**: all nodes classified as `UNKNOWN` (`no_data=True`). `has_errors=False` (UNKNOWN != UNSUPPORTED), so no `RuntimeError` raised.
5. **Compile stage** (`CompileStage.process()`) calls `_build_session_options()` → `WinMLEPRegistry.get_instance().register_ep('QNNExecutionProvider')` → `'QNNExecutionProvider' not in self._registered_eps` (WinMLEPRegistry has no knowledge of the WinML-singleton registration in step 2) → **`ort.register_execution_provider_library('QNNExecutionProvider', dll_path)` — SECOND REGISTRATION** → **parent process crashes with native exit 127**. No Python traceback is emitted.

### Proof

```python
# Reproducer (exits 127):
from winml.modelkit import winml
winml.register_execution_providers(ort=True)  # registers via WinML singleton

from winml.modelkit.session.ep_registry import WinMLEPRegistry
registry = WinMLEPRegistry.get_instance()
# registry._registered_eps is [] — WinML singleton is blind to it
devices = registry.register_ep('QNNExecutionProvider')  # calls register_execution_provider_library AGAIN → EXIT 127
```

After the first registration, `ort.get_ep_devices()` already shows QNN listed **twice** — a sign of state corruption.

---

## Why "No Python Traceback"

`ort.register_execution_provider_library()` calls native C++ code that calls `exit(127)` directly, bypassing Python's exception machinery. The Python interpreter has no chance to write a traceback.

---

## The Two Separate WinML Singletons

There are TWO uncoordinated singleton systems that both call `ort.register_execution_provider_library`:

| Singleton | Module | Used by | Tracks registration? |
|-----------|--------|---------|---------------------|
| `WinML` | `winml.py` | `_is_ep_available_locally()` in the analyze loop | Tracks in `_registered_eps` within `WinML`, invisible to `WinMLEPRegistry` |
| `WinMLEPRegistry` | `ep_registry.py` | `_build_session_options()` in session/compile | Tracks in `WinMLEPRegistry._registered_eps`, invisible to `WinML` |

When both systems run in the same process (analyze loop + compile stage), the same QNN DLL is registered twice.

---

## Why the Analyze Loop Children Don't Crash the Parent

- `ResilientRunner` uses `ProcessPoolExecutor(max_workers=1, mp_context="spawn")`
- Spawned children do NOT inherit the parent's `WinML` singleton state
- In children, `ort.get_ep_devices()` returns only `CPUExecutionProvider`
- `winml.add_ep_for_device(so, 'QNNExecutionProvider', ...)` silently does nothing (no matching EP)
- `ort.ModelCompiler(no-EP-so, bytes, ERROR_IF_NO_NODES_COMPILED)` → child exits 127
- Parent catches `BrokenProcessPool` at `future.result(timeout=60)`, recreates executor
- **Parent survives** all 122 child crashes

---

## Test 2: `winml compile` on QDQ model

`winml compile` on `imgcls_..._quantized.onnx` works (EXIT=0, 1.98s) because:
- It does NOT go through the analyze loop
- `WinML` singleton is never called
- `WinMLEPRegistry.register_ep()` is the FIRST and only QNN registration
- `ort.ModelCompiler` compiles correctly → `quantized_qnn_ctx.onnx` (1587B) + `_qnn.bin` (25MB)

---

## Test 4: v200 Venv Comparison

The v200 venv (ORT 1.24.4 + `onnxruntime_qnn` package) does NOT use the WinML ExecutionProviderCatalog mechanism:
- `ort.get_ep_devices()` returns only `CPUExecutionProvider` — QNN is bundled differently
- The `WinML` + `WinMLEPRegistry` double-registration conflict does NOT occur in v200
- **The same `winml perf` pipeline would not exhibit exit 127 in v200** (though it also would not find QNN EP through `_is_ep_available_locally()`)

---

## Intermediate Artifacts State

Cache directory: `C:\Users\zhengte\.cache\winml\artifacts\microsoft_resnet-50\`

| File | Size | Timestamp | State |
|------|------|-----------|-------|
| `imgcls_..._export.onnx` | 102 MB | May 13 17:55 | FP32 export, valid |
| `imgcls_..._optimized.onnx` | 102 MB | May 13 17:56 | FP32 optimized, valid |
| `imgcls_..._quantized.onnx` | 26 MB | May 12 11:36 | QDQ INT8, valid |
| `imgcls_..._quantized_npu_ctx.onnx` | 1587 B | May 13 17:38 | Compiled QDQ, valid |
| `imgcls_..._quantized_npu_ctx_qnn.bin` | 25 MB | May 13 17:38 | QNN binary, valid |
| `imgcls_..._model.onnx` | — | **MISSING** | Build never completes |

`model.onnx` is missing because the parent process crashes (exit 127) during the compile stage, which runs AFTER the analyze loop and quantize step. On re-run, the full pipeline re-executes.

---

## Fix Direction

**Option A (Minimal)**: In `WinMLEPRegistry.register_ep()`, check `ort.get_ep_devices()` FIRST before calling `ort.register_execution_provider_library()`. If the EP is already visible in `ort.get_ep_devices()`, skip the DLL registration.

```python
# In WinMLEPRegistry.register_ep():
if ep_name in self._ep_paths and ep_name not in self._registered_eps:
    # Check if already registered by another path (e.g. WinML singleton)
    already_registered = any(d.ep_name == ep_name for d in ort.get_ep_devices())
    if not already_registered:
        ort.register_execution_provider_library(ep_name, dll_path)
    self._registered_eps.append(ep_name)  # mark as known regardless
```

**Option B (Preferred)**: Consolidate into a single `WinMLEPRegistry` call site. Remove `winml.register_execution_providers()` from `_is_ep_available_locally()` and replace with `WinMLEPRegistry.get_instance().register_to_ort()`, which has the `_registered_eps` guard.

---

## Answers to Diagnostic Questions

1. **Does `winml compile` work on QDQ-quantized resnet-50?** YES — EXIT=0, 1.98s, produces valid ctx + bin.
2. **Does the full HF auto-build path work end-to-end?** NO — crashes at compile stage with exit 127 (double QNN DLL registration).
3. **At exactly which stage does it fail?** In `CompileStage.process()` → `WinMLEPRegistry.register_ep()` → `ort.register_execution_provider_library()` (second call) → native exit 127.
4. **Does QDQ model work in project `.venv` vs v200?** Yes in project venv (Test 5/6 pass). v200 doesn't use WinML catalog so the double-registration conflict doesn't arise there.
