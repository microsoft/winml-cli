# Phase 2 Verification — QNN cpu/gpu/npu × perf/compile

After commit 1ab32a76 (v2 cleanup), verify the 6-command CLI matrix.

## Reference

`D:/BYOM/release/mk_release` inspection:
- The reference release does not contain explicit CLI invocation scripts for
  `winml perf` / `winml compile`. The `_qnn.log` file is a binary QNN profiling
  artifact (not an invocation script).
- Key finding from `docs/winml-ep-empirical-findings.md`: this Snapdragon X Elite
  machine has QNN with NPU and GPU backends only — no CPU backend.
  `QNNExecutionProvider` enumerates `[NPU, GPU]` from `ort.get_ep_devices()`.
- Key finding: `onnxruntime-qnn` 2.1.0 (PyPI) is installed; the DLL lives at
  `.venv/Lib/site-packages/onnxruntime_qnn/libs/amd64/onnxruntime_providers_qnn.dll`.
- Key finding: `--ep qnn --device cpu` fails on this machine (DeviceNotFound) —
  QNN cpu backend is Qualcomm-specific and not wired here. The corrected cpu
  invocation uses `--device cpu` (routes to CPUExecutionProvider).
- No env vars (QNN_SDK_ROOT, etc.) needed for the perf/compile matrix.

## Matrix results

| # | Command | Exit | Output (tail) | Verdict |
|---|---|---|---|---|
| 1 | perf qnn cpu (`--ep qnn --device cpu`) | 4 | `Error: Benchmark failed: No OrtEpDevice…` | HARDWARE N/A — DeviceNotFound (QNN has no CPU backend on this machine) |
| 2 | perf qnn gpu (`--ep qnn --device gpu`) | 0 | `Avg 11.26 ms / 88.84 samples/s` | PASS |
| 3 | perf qnn npu (`--ep qnn --device npu`) | 0 | `Avg 1.99 ms / 501.35 samples/s` | PASS |
| 4 | compile qnn cpu (`--ep qnn --device cpu`) | 1 | `Error: No OrtEpDevice…` (raw traceback before fix; clean after) | FIXED → EXIT=1 clean error |
| 5 | compile qnn gpu (`--ep qnn --device gpu`) | 0 | `_qnn_ctx.onnx` (931 B) + `_qnn.bin` (49 MB) | PASS |
| 6 | compile qnn npu (`--ep qnn --device npu`) | 0 | `_qnn_ctx.onnx` (931 B) + `_qnn.bin` (49 MB) | PASS |

## Per-command details

### 1. winml perf --ep qnn --device cpu

```
Error: Benchmark failed: No OrtEpDevice for QNNExecutionProvider matches
device='cpu'. Available: [('NPU', '0x4d4f4351', '0x41304430'), ('GPU', '0x4d4f4351', '0x36334330')]
EXIT=4
```

**Verdict**: HARDWARE N/A. This machine's QNN EP only exposes NPU and GPU
backends. `--device cpu` without `--ep qnn` routes to CPUExecutionProvider and
works correctly (exit 0, latency ~40 ms). The `--ep qnn --device cpu` spec is
valid for machines with QNN CPU backend (e.g. x86 desktop QNN SDK), not this
Snapdragon X Elite hardware.

Corrected cpu invocation: `winml perf -m <onnx> --device cpu` → EXIT=0,
Avg 40.12 ms / 24.92 samples/s (CPUExecutionProvider).

### 2. winml perf --ep qnn --device gpu

```
Device:  gpu
EP:  qnn
Avg │  P50 │  P90 │  P95 │  P99 │  Min │  Max │  Std
11.26│ 11.18│ 11.60│ 11.60│ 11.60│ 10.98│ 11.60│ 0.25
Throughput: 88.84 samples/sec
EXIT=0
```

**Verdict**: PASS. QNN GPU backend runs ResNet-50 at 11.26 ms avg.

### 3. winml perf --ep qnn --device npu

```
Device:  npu
EP:  qnn
Avg │ P50 │ P90 │ P95 │ P99 │ Min │ Max │ Std
1.99│ 1.99│ 2.01│ 2.01│ 2.01│ 1.98│ 2.01│ 0.01
Throughput: 501.35 samples/sec
EXIT=0
```

**Verdict**: PASS. QNN NPU (HTP backend) at 1.99 ms / 501 samples/s —
matches the prior >400 samples/s expectation with burst-mode defaults.

### 4. winml compile --ep qnn --device cpu

**Before fix (commit 1ab32a76)**: raw Python traceback, EXIT=1.
**After fix (commit 39d95d73)**: clean error message, EXIT=1.

```
Error: No OrtEpDevice for QNNExecutionProvider matches device='cpu'.
Available: [('NPU', '0x4d4f4351', '0x41304430'), ('GPU', '0x4d4f4351', '0x36334330')]
EXIT=1
```

**Root cause**: `compile.py` called `resolve_device()` outside any try/except;
`DeviceNotFound` propagated as an unhandled exception. `perf.py` wrapped the
entire benchmark in `except Exception`, hiding the same class of error there
(exit 4 with `[red]Error:[/red]`).

**Fix**: wrap `resolve_device()` in `compile.py` with explicit `except` for
`DeviceNotFound`, `EPNotDiscovered`, `EPRegistrationFailed`, `ValueError` —
each raises a `click.ClickException` or `click.UsageError`.

Corrected cpu compile: `winml compile -m <onnx> --device cpu` → EXIT=0,
CPUExecutionProvider, no EPContext artifact (by design: `enable_ep_context=False`).

### 5. winml compile --ep qnn --device gpu

```
Success! Model compiled
Output: temp\verify_qnn_matrix\gpu\imgcls_69f0345d0dbeb3b1_export_qnn_ctx.onnx
Compile time: 6.01s
Total time: 6.10s
EXIT=0
```

Artifacts:
- `imgcls_69f0345d0dbeb3b1_export_qnn_ctx.onnx` — 931 bytes (EPContext stub)
- `imgcls_69f0345d0dbeb3b1_export_qnn_ctx_qnn.bin` — 49 MB (QNN compiled graph)

**Verdict**: PASS.

Note: artifact naming uses `_qnn_ctx` suffix (not `_gpu_ctx`) because
`CompileContext.execution_provider` is the short provider name (`"qnn"`),
not the device. Both GPU and NPU compile produce identically-named files;
the binary payload is device-specific.

### 6. winml compile --ep qnn --device npu

```
Success! Model compiled
Output: temp\verify_qnn_matrix\npu\imgcls_69f0345d0dbeb3b1_export_qnn_ctx.onnx
Compile time: 0.41s
Total time: 0.49s
EXIT=0
```

Artifacts:
- `imgcls_69f0345d0dbeb3b1_export_qnn_ctx.onnx` — 931 bytes
- `imgcls_69f0345d0dbeb3b1_export_qnn_ctx_qnn.bin` — 49 MB

**Verdict**: PASS. Fast (0.41 s) because the NPU binary was cached from the
earlier `perf --ep qnn --device npu` run (WinMLSession.compile() checks mtime).

## Iteration log

- **Issue 1**: `winml compile --ep qnn --device cpu` (and any other
  DeviceNotFound path) produced a raw Python traceback instead of a
  `click.ClickException`. The `perf` command already handled this gracefully;
  `compile` did not. → Fixed in commit **39d95d73**:
  `fix(compile): catch DeviceNotFound/EPNotDiscovered at CLI boundary`

## Observations from D:/BYOM/release/mk_release

- `docs/winml-ep-empirical-findings.md` confirms this machine's QNN EP
  only enumerates NPU and GPU backends (sections 4.3, 7.1, Appendix B).
- The first-hit-wins PyPI→catalog dedup means PyPI QNN wins over MSIX QNN;
  the perf/compile matrix tests the PyPI path.
- No `QNN_SDK_ROOT` env var needed for ORT-compiler path (only QAIRT).
- `onnxruntime-qnn` 2.1.0 (PyPI) is the active DLL; `wasdk` catalog QNN is
  also installed (MSIX 2.x) but shadowed by first-hit-wins ordering.

## Verdict

**4/6 PASS** (gpu+npu for both perf and compile), **1/6 HARDWARE N/A** (cpu
row: QNN has no CPU backend on this Snapdragon), **1 BUG FIXED** (compile
DeviceNotFound → clean error after commit 39d95d73).

The "QNN cpu" row is not a code bug on this machine — CPUExecutionProvider
serves the cpu device correctly. The `--ep qnn --device cpu` invocation is
for hardware that ships QNN with CPU backend (e.g. x86 desktop QNN SDK).
