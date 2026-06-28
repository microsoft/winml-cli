# T6 Analyze-Stage DLL-Not-Found Crash — Diagnostic 2026-05-13

**Branch**: `feat/op-tracing-refactor` (HEAD: `eb37f6c3`)  
**Status**: Root cause confirmed. Prior fix (`eb37f6c3`) guarded the wrong caller — crash persists.

---

## TL;DR

`winml perf -m microsoft/resnet-50 --ep qnn --device npu` crashes with `0xC000026F`
(`STATUS_DLL_NOT_FOUND`) at the **first node of the analyze loop** because two uncoordinated
singletons each call `ort.register_execution_provider_library('QNNExecutionProvider', dll_path)`:

1. `WinMLEPRegistry.register_ep()` — called first, during `_load_model` → `resolve_device`
2. `winml.py:WinML.register_execution_providers()` — called second, during the analyze loop's
   `_is_ep_available_locally()` on the first node

`ort.register_execution_provider_library` is **not idempotent** in ORT 1.23.2: a second call
for the same DLL calls C++ `exit(127)` with no Python traceback. The commit `eb37f6c3` (Gap #1
fix) patched `WinMLEPRegistry.register_ep()` to check `ort.get_ep_devices()` first, but that
only guards the case where `winml.py:WinML` ran **before** `WinMLEPRegistry`. In the `perf` HF
path, the ordering is reversed: `WinMLEPRegistry` runs first, so the guard in `ep_registry.py`
is irrelevant — the crash comes from `winml.py:WinML` running second.

**Fix**: Add the same `ort.get_ep_devices()` pre-check to
`winml.py:WinML.register_execution_providers()`, or — better — remove the `winml.py:WinML`
call from `_is_ep_available_locally()` and replace it with
`WinMLEPRegistry.get_instance().register_to_ort()` which already has the correct guard.

---

## Test Results

### Test 1: `winml perf -m microsoft/resnet-50 --ep qnn --device npu` (current HEAD `eb37f6c3`)

Exit: **3221226095** = `0xC000026F` = `STATUS_DLL_NOT_FOUND`

Last lines of stderr before crash (whitespace-collapsed):
```
[2026-05-13T19:01:48] WARNING: No pattern matches found by PatternMatcher
  0%|          | 0/122 ...
[2026-05-13T19:01:49] WARNING: Rule zip not found: ...QNNExecutionProvider_npu_ai.onnx_opset17.zip.
[2026-05-13T19:01:49] WARNING: Rule zip file not found: ...same file...
<process exits 0xC000026F — no further output>
```

The process crashes immediately after the two "Rule zip not found" warnings — at op #1 of 122,
before any node result is emitted.

### Test 2: `winml perf -m microsoft/resnet-50 --ep qnn --device npu --verbose`

Exit: **3221226095** = `0xC000026F`. Same crash point. `--verbose` produces richer pre-crash
logging but does not change the crash or its timing.

### Test 3: `winml perf -m <fp32-export.onnx> --ep qnn --device npu` (direct ONNX path)

Exit: **0**. The direct ONNX path (`_run_onnx_benchmark`) does NOT call `resolve_device` early,
so `WinMLEPRegistry.register_ep` is not called before the analyze stage. The `winml.py:WinML`
singleton registers QNN first (and only once). No crash.

### Test 4: Spawn-child crash reproduction

```
ResilientRunner → spawn child → EPChecker.check_compile → ort.ModelCompiler(no-EP-so, bytes, ERROR_IF_NO_NODES_COMPILED)
```

The spawned child process crashes (exit 127) because in each fresh spawn:
- `winml.py:WinML` singleton is not inherited (spawn, not fork)
- `winml.add_ep_for_device()` in `EPChecker._get_sess_options()` silently does nothing (no QNN device found via `ort.get_ep_devices()` in the fresh child)
- `ort.ModelCompiler` with an empty `SessionOptions` and `ERROR_IF_NO_NODES_COMPILED` exits 127

Parent catches `BrokenProcessPool` and recovers. **This is expected behavior** (122 child
crashes, all recovered by `ResilientRunner`).

### Test 5: Double-registration reproducer

```python
import onnxruntime as ort
# Simulate WinMLEPRegistry.register_ep (first):
ort.register_execution_provider_library('QNNExecutionProvider', dll_path)

# Simulate winml.py:WinML.register_execution_providers (second):
ort.register_execution_provider_library('QNNExecutionProvider', dll_path)
# ^ calls native exit(127) — NO Python exception
```

Confirmed: second call crashes the process.

---

## Crash Localization

**File**: `src/winml/modelkit/winml.py`  
**Line**: `module.register_execution_provider_library(name, path)` inside
`WinML.register_execution_providers()`  
**Triggered by**: `runtime_checker_query.py:1148` — `winml.register_execution_providers(ort=True)`
inside `_is_ep_available_locally()`, called for the **first** of the 122 ResNet-50 nodes when
the rule zip is missing and `run_unknown_op=True`.

Call chain:
```
perf.py → PerfBenchmark._load_model()
  → resolve_device(ep='qnn', device='npu')
    → WinMLEPRegistry.get_instance().register_ep('QNNExecutionProvider')
      → ort.register_execution_provider_library('QNNExecutionProvider', dll_path)  ← FIRST

perf.py → PerfBenchmark.run() → _load_model() → WinMLAutoModel.from_pretrained()
  → build_hf_model() → run_optimize_analyze_loop() → _run_analyze_loop() → analyze_onnx()
    → ONNXStaticAnalyzer.analyze()
      → RuntimeChecker.op_support()
        → RuntimeCheckerQuery.run_for_node(node[0])  # first of 122 nodes
          → Phase 3: zip missing → _try_local_ep_check()
            → _is_ep_available_locally()
              → winml.register_execution_providers(ort=True)  # WinML singleton first-time init
                → WinML._registered_eps['onnxruntime'] = []  # empty — WinMLEPRegistry invisible
                → ort.register_execution_provider_library('QNNExecutionProvider', dll_path)  ← SECOND → EXIT 127
```

No Python stack trace is emitted because `ort.register_execution_provider_library` calls
C++ `exit(127)` directly.

---

## Missing DLL Identification

`0xC000026F` (`STATUS_DLL_NOT_FOUND`) is misleading. The crash is **not** caused by a genuinely
missing DLL. The QNN EP DLL is present and was successfully loaded by the first
registration. The NTSTATUS is generated by ORT's internal EP-registration abort logic
(`exit(127)`) when the same DLL is registered twice — it manifests as `STATUS_DLL_NOT_FOUND`
because Windows maps exit code 127 to that NTSTATUS when interpreting a native process exit.

The QNN EP DLL (`QNNExecutionProvider.dll` or equivalent) is **present and working**:
- T1–T5 in the reverify doc all pass with QNN on NPU.
- `winml compile` with QNN works (exit 0).
- Direct ONNX perf with QNN works (exit 0).

---

## Comparison: Why Does Direct-ONNX Perf (T1) Work But HF Perf (T6) Crash?

| Aspect | Direct-ONNX path (T1, T2, T3) | HF path (T6) |
|--------|------------------------------|--------------|
| `resolve_device` called early | NO — `_run_onnx_benchmark` takes `ep_device` already resolved | YES — `_load_model` calls `resolve_device` |
| First QNN registration | `winml.py:WinML` in analyze `_is_ep_available_locally` | `WinMLEPRegistry.register_ep` in `_load_model` |
| Second QNN registration | None — `WinML._registered_eps` guards subsequent calls | `winml.py:WinML` in analyze `_is_ep_available_locally` (WinML singleton blind to WinMLEPRegistry) |
| Crash | NO | YES — `STATUS_DLL_NOT_FOUND` at node 1 of 122 |

---

## Why the `eb37f6c3` Fix Did Not Work

The fix added a `ort.get_ep_devices()` pre-check inside `WinMLEPRegistry.register_ep()`:

```python
already_loaded = any(d.ep_name == ep_name for d in ort.get_ep_devices())
if already_loaded:
    self._registered_eps.append(ep_name)
else:
    ort.register_execution_provider_library(ep_name, dll_path)
```

This correctly prevents `WinMLEPRegistry` from double-registering if `winml.py:WinML` ran
**first** (the case documented in the commit). But in the `perf` HF path, the actual order is:

1. `WinMLEPRegistry.register_ep` runs first — `ort.get_ep_devices()` shows no QNN yet → registers → now QNN is in ORT
2. `winml.py:WinML.register_execution_providers` runs second — `WinML._registered_eps['onnxruntime']` is still `[]` → doesn't check `ort.get_ep_devices()` → registers again → **crash**

The fix protected the wrong caller.

---

## Root Cause Classification

**(A) Our code.** Two uncoordinated singleton systems (`winml.py:WinML` and
`session/ep_registry.py:WinMLEPRegistry`) both call
`ort.register_execution_provider_library` with no cross-awareness. ORT 1.23.2 does not
tolerate duplicate registration.

---

## Recommendation

**Fix `winml.py:WinML.register_execution_providers()`** to check `ort.get_ep_devices()` before
registering each EP — identical to the guard already added to `WinMLEPRegistry.register_ep()`:

```python
# In winml.py:WinML.register_execution_providers()
for name, path in self._ep_paths.items():
    for module in modules:
        if name not in self._registered_eps[module.__name__]:
            # Guard: ORT is NOT idempotent. Check live device list first.
            already_loaded = any(d.ep_name == name for d in module.get_ep_devices())
            if not already_loaded:
                try:
                    module.register_execution_provider_library(name, path)
                except Exception as e:
                    ...
            self._registered_eps[module.__name__].append(name)
```

**Preferred long-term fix**: remove the `winml.register_execution_providers(ort=True)` call from
`RuntimeCheckerQuery._is_ep_available_locally()` and replace with
`WinMLEPRegistry.get_instance().ensure_initialized()` (or `register_to_ort()`). This consolidates
all EP registration through the single guarded path and eliminates the dual-singleton problem.
