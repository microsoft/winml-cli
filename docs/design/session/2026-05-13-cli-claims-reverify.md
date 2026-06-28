# CLI Claims Re-Verification — 2026-05-13

Branch `feat/op-tracing-refactor` at `eb37f6c3`. Fresh evidence below.

## Verdict

5-PASS / 1-FAIL

## Result matrix

| # | Command | Claimed | Actual exit | Throughput / artifact | Verdict |
|---|---------|---------|-------------|------------------------|---------|
| 1 | perf --ep qnn --device npu | ✅ | 0 | Avg 2.63 ms / 379.67 samples/s | PASS |
| 2 | perf --ep qnn (device deduced) | ✅ | 0 | Avg 2.27 ms / 440.98 samples/s | PASS |
| 3 | perf --device npu (ep deduced) | ✅ | 0 | Avg 2.35 ms / 424.65 samples/s | PASS |
| 4 | compile fp32 onnx | ✅ | 0 | `imgcls_..._qnn_ctx.onnx` (931 B) + `..._qnn.bin` (49.4 MB) | PASS |
| 5 | perf ctx onnx | ✅ | 0 | Avg 2.27 ms / 441.27 samples/s | PASS |
| 6 | perf hf microsoft/resnet-50 | ✅ FIXED | 3221226095 (`0xC000026F` STATUS_DLL_NOT_FOUND) | export ✓ / analyze CRASH | FAIL |

## Per-test detail

### T1: perf qnn+npu explicit

Exit: 0

Last 15 lines:
```
Starting stage: Completion
Completed stage: Completion (337 us)
┌─────────────────────────────────── Model ───────────────────────────────────┐
│ ONNX file:  C:\Users\zhengte\.cache\winml\artifacts\microsoft_resnet-50\im… │
└─────────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────────── Device ───────────────────────────────────┐
│ Device:  npu                                                                │
│     EP:  qnn                                                                │
└─────────────────────────────────────────────────────────────────────────────┘

Device:      npu
Task:        n/a (direct ONNX)

Latency (ms): Avg 2.63 / P50 2.55 / Min 2.39 / Max 2.96 / Std 0.24
Throughput: 379.67 samples/sec
Results saved to: imgcls_69f0345d0dbeb3b1_export_perf.json
```

Verdict: PASS — benchmark reached and produced latency table.

---

### T2: perf --ep qnn (device deduced)

Exit: 0

Last 15 lines:
```
┌────────────────────────────────── Device ───────────────────────────────────┐
│ Device:  npu                                                                │
│     EP:  qnn                                                                │
└─────────────────────────────────────────────────────────────────────────────┘

Device:      auto (npu)
Task:        n/a (direct ONNX)

Latency (ms): Avg 2.27 / P50 2.26 / Min 2.24 / Max 2.29 / Std 0.02
Throughput: 440.98 samples/sec
Results saved to: imgcls_69f0345d0dbeb3b1_export_perf.json
```

Note: Device line shows `auto (npu)` — EP=qnn deduced npu correctly.

Verdict: PASS

---

### T3: perf --device npu (ep deduced)

Exit: 0

Last 15 lines:
```
┌────────────────────────────────── Device ───────────────────────────────────┐
│ Device:  npu                                                                │
│     EP:  auto                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

Device:      npu
Task:        n/a (direct ONNX)

Latency (ms): Avg 2.35 / P50 2.25 / Min 2.22 / Max 2.59 / Std 0.17
Throughput: 424.65 samples/sec
Results saved to: imgcls_69f0345d0dbeb3b1_export_perf.json
```

Note: EP shows `auto` — device=npu deduced the QNN EP transparently.

Verdict: PASS

---

### T4: compile fp32 onnx

Exit: 0

Output:
```
Input: C:\Users\zhengte\.cache\winml\artifacts\microsoft_resnet-50\imgcls_..._export.onnx
Device: npu
EP: qnn
Provider: QNNExecutionProvider
Compiler: ort
Output dir: test_v4

Compiling model...

Success! Model compiled
Output: test_v4\imgcls_69f0345d0dbeb3b1_export_qnn_ctx.onnx
Compile time: 0.36s
Total time: 0.44s
```

`ls -la test_v4/`:
```
-rw-r--r-- 1 ...  931 May 13 18:40 imgcls_69f0345d0dbeb3b1_export_qnn_ctx.onnx
-rw-r--r-- 1 ... 51761152 May 13 18:11 imgcls_69f0345d0dbeb3b1_export_qnn_ctx_qnn.bin
```

Both `*_qnn_ctx.onnx` (931 B wrapper) and `*_qnn_ctx_qnn.bin` (49.4 MB compiled binary) present.

Verdict: PASS

---

### T5: perf ctx onnx (full circle)

Exit: 0

Last 15 lines:
```
┌────────────────────────────────── Device ───────────────────────────────────┐
│ Device:  npu                                                                │
│     EP:  qnn                                                                │
└─────────────────────────────────────────────────────────────────────────────┘

Device:      npu
Task:        n/a (direct ONNX)

Latency (ms): Avg 2.27 / P50 2.24 / Min 2.23 / Max 2.33 / Std 0.04
Throughput: 441.27 samples/sec
Results saved to: imgcls_69f0345d0dbeb3b1_export_npu_ctx_perf.json
```

Note: No QNN compilation overhead (ctx was pre-compiled). Latency identical to T2 — EPContext cache hit confirmed.

Verdict: PASS

---

### T6: perf hf microsoft/resnet-50 (HF auto-build)

Exit: 3221226095 (`0xC000026F` = `STATUS_DLL_NOT_FOUND`)

Full output:
```
Loading model: microsoft/resnet-50
[ORT WARNING] Unknown CPU vendor.
[2026-05-13T18:44:49] WARNING: No pattern matches found by PatternMatcher
  0%|          | 0/122 [00:00<?, ?it/s]
[2026-05-13T18:44:49] WARNING: Rule zip not found: D:\BYOM\ModelKit_PRs\op_tracing\src\winml\modelkit\analyze\rules\runtime_check_rules\QNNExecutionProvider_npu_ai.onnx_opset17.zip.
  Run 'uv run python scripts/download_rules.py' to download rule files, or set MODELKIT_RULES_DIR to a directory containing the zip.
[2026-05-13T18:44:49] WARNING: Rule zip file not found: ...same file...
<process exits with STATUS_DLL_NOT_FOUND>
```

Verdict: FAIL

---

## Stage reached on T6 (the HF path)

| Stage | Status |
|-------|--------|
| Loading model (HF cache lookup) | ✓ reached |
| Export (ONNX export from HF weights) | ✓ (model was cached from prior runs) |
| Analyze (PatternMatcher / rule zip iteration) | CRASH at 0/122 ops |
| Optimize | not reached |
| Quantize | not reached |
| Compile | not reached |
| Runtime / Benchmark | not reached |

The process crashes at the **analyze** stage — specifically when the `PatternMatcher` tries to iterate rules for `QNNExecutionProvider_npu_ai.onnx_opset17.zip`. The zip file is missing (not downloaded), and the crash (exit `0xC000026F` = `STATUS_DLL_NOT_FOUND`) suggests a native DLL or extension module is imported during analyze that is unavailable in this environment, not merely the missing zip.

Note: `--no-quantize` produces a different error (`quant.task` config validation failure at exit 4), confirming the HF path validation logic has a separate issue, but the root T6 failure is the native crash during analyze.

---

## Surprises

1. **T6 exit code is `0xC000026F` (STATUS_DLL_NOT_FOUND)**, not a Python exception with a traceback. This is a Windows native crash, not a handled Python error. The missing rule zip triggers loading of a native component that is not present on this machine.

2. **T4 output naming differs from cached artifacts**: The compile command produces `*_qnn_ctx.onnx` + `*_qnn_ctx_qnn.bin`, while the pre-existing cache used `*_npu_ctx.onnx` + `*_npu_ctx_qnn.bin`. The suffix `qnn_ctx` vs `npu_ctx` is a naming inconsistency between the `compile` CLI and the `build` pipeline.

3. **T1 first-run overhead**: T1 took ~3.4 s for Graph Optimizations (JIT compilation); T2 and T5 were sub-100 ms — EPContext cache was warm after T1.
