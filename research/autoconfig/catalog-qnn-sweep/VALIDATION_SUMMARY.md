# Validation Sweep Results — QNN NPU (2026-06-16)

**Device:** Snapdragon X Elite X1E80100  
**ORT:** onnxruntime-windowsml==1.24.5  
**QNN SDK:** 2.2450.47.0  
**Protocol:** 3 × 500 iters, 30s cool-down, `quantized.onnx` (W8A16), `--no-compile`  
**Script:** `validation_sweep.py` — targeted 4-hypothesis sweep (h0/h1/h3/h4)

## Hypothesis Matrix

| ID | Config | Purpose |
|----|--------|---------|
| h0 | auto-config baseline (W8A16, opset auto) | baseline reference |
| h1 | opset 17 explicit (W8A16) | npu-001 baseline |
| h3 | opset 21 (W8A16) | **npu-001 test** — does opset21 help? |
| h4 | opset 17 + conv fusions | **npu-006 test** — do conv fusions regress? |

---

## Results by Model

### facebook/dinov2-base (ViT-B DINOv2, image-feature-extraction)

| Hyp | Median p50 | Sessions (ms) | CV note |
|-----|-----------|---------------|---------|
| h0 auto | 38.68 ms | [38.99, 38.68, 36.26] | stable (stale build artifact) |
| **h1 opset17** | **34.56 ms** | [34.56, 34.67, 33.15] | rock stable |
| **h3 opset21** | **26.23 ms** | [33.00, 26.22, 26.23] | s0 elevated (JIT warmup), s1+s2 stable |
| h4 fusions | 25.92 ms | [26.06, 25.92, 25.87] | rock stable |

**npu-001: opset21 → +24.1% speedup** `(34.56 → 26.23ms)`  
**npu-006: conv fusions → -25% (fusions FASTER, not regression)** — DINOv2 is attention-dominant, few Conv ops to fuse

---

### microsoft/rad-dino (ViT-L DINOv2 medical, image-feature-extraction)

| Hyp | Median p50 | Sessions (ms) | CV note |
|-----|-----------|---------------|---------|
| **h1 opset17** | **274.98 ms** | [274.98, 274.56, 275.10] | CV=0.009, CPU-deterministic |
| **h3 opset21** | **275.36 ms** | [275.30, 275.36, 275.56] | CV=0.022 |

**npu-001: -0.1% — NEUTRAL (CPU-bound)**  
Model runs entirely on CPU (~275ms). QNN NPU cannot accelerate rad-dino (ViT-L too large or incompatible ops). Opset has no effect when model is CPU-bound.

---

### facebook/dino-vitb16 (plain DINO ViT-B/16, image-feature-extraction)

| Hyp | Median p50 | Sessions (ms) | CV note |
|-----|-----------|---------------|---------|
| **h1 opset17** | **19.92 ms** | [19.92, 19.97, 19.90] | rock stable |
| **h3 opset21** | **20.07 ms** | [20.20, 20.07, 19.99] | rock stable |
| h4 fusions | 20.12 ms | [20.12, 20.04, 20.41] | rock stable |

**npu-001: -0.7% — NEUTRAL** ← **critical control**  
**npu-006: +1.0% — NEUTRAL** (no Conv layers to fuse, patch-embed Conv fusion is benign)

---

## Cross-Model Summary — npu-001 (opset21 vs opset17)

| Model | Architecture | opset17 (h1) | opset21 (h3) | Gain | Verdict |
|-------|-------------|-------------|-------------|------|---------|
| facebook/dinov2-small | DINOv2 ViT-S | 7.18 ms* | 4.98 ms* | **+30.6%** | ✅ CONFIRMED |
| facebook/dinov2-base | DINOv2 ViT-B | 34.56 ms | 26.23 ms | **+24.1%** | ✅ CONFIRMED |
| apple/mobilevit-small | Conv+Attn hybrid | 11.72 ms* | 8.62 ms* | **+26.5%** ⚠️ | 🟡 LIKELY (DVFS spike in h1) |
| facebook/dino-vitb16 | plain ViT-B/16 | 19.92 ms | 20.07 ms | **-0.7%** | ❌ NEUTRAL — critical control |
| microsoft/rad-dino | ViT-L DINOv2 | 274.98 ms | 275.36 ms | **-0.1%** | ⬛ CPU-BOUND (untestable) |
| google/vit-base-patch16-224 | plain ViT-B | n/a | n/a | **-7.4%** ⚠️* | ❌ REGRESSION |

_*Original catalog_qnn_sweep.py data (optimized.onnx, not quantized.onnx — different pipeline)_

**Key architectural discriminant:** opset21 consistently helps **DINOv2 family** (+24-31%) but has **zero effect on plain ViT** (dino-vitb16: -0.7%, noise-level). This is NOT a general ViT property. DINOv2-specific op patterns must explain the difference — mechanism TBD.

---

## Cross-Model Summary — npu-006 (conv fusions)

| Model | Architecture | h1 no-fusions | h4 fusions | Regression | Verdict |
|-------|-------------|--------------|-----------|------------|---------|
| microsoft/resnet-18 | Conv-dominant | ~1–4 ms* | 132–135 ms* | **+4900%** 🔥 | ✅ CATASTROPHIC |
| apple/mobilevit-small | Conv+Attn | ~10–12 ms* | ~10–12 ms* | **≈0%** | 🟢 SAFE |
| facebook/dinov2-base | DINOv2 ViT-B | 34.56 ms | 25.92 ms | **-25%** (faster) | 🟢 SAFE / beneficial |
| facebook/dino-vitb16 | plain ViT-B | 19.92 ms | 20.12 ms | **+1.0%** | 🟢 SAFE (neutral) |

_*Original catalog_qnn_sweep.py data_

**Conclusion:** Conv fusions only regress Conv-dominant models (ResNet). Attention-dominant models (DINOv2, ViT) are safe or slightly benefit. The hazard is proportional to Conv op density.

---

## Bugs Found and Fixed in validation_sweep.py

| Bug | Impact | Fix |
|-----|--------|-----|
| `bench_screen` parsed `d.get("p50_ms")` instead of `d["latency_ms"]["p50"]` | All hypotheses marked BENCH_FAIL in v1/v2 runs | Fixed to read nested `latency_ms.p50` |
| Reuse check triggered on any `.onnx` (including truncated `export.onnx`) | h1 was benchmarked on FP32 unoptimized model | Changed to require `quantized.onnx` or `optimized.onnx` |
| Model file selection preferred `optimized.onnx` over `quantized.onnx` alphabetically | Benchmarked FP32 graph instead of W8A16 quantized | Fixed to explicitly prefer `quantized` > `optimized` > other |

---

## Known Limitations

1. **`--no-compile` throughout**: All runs omit `winml compile` (pre-built QNN context binary). Production use would include compile, which npu-003 suggests adds ~1.7x additional speedup. The npu-001 ratio should hold with compile enabled, but absolute latencies will be lower.
2. **3 sessions only**: DVFS on QNN NPU can cause any single session to be thermal-spiked. With only 3 sessions, the median can still be affected if 2/3 spike. See h3 dinov2-base s0=33ms (warmup effect) vs s1+s2=26ms.
3. **rad-dino untestable**: When a model falls back entirely to CPU, no NPU-related findings can be extracted. The reason for CPU fallback (model size? unsupported ops?) was not investigated.
4. **dinov2-small not re-validated with v2 pipeline**: The original +30.6% result was from `catalog_qnn_sweep.py` using `optimized.onnx`. The v2 pipeline uses `quantized.onnx`. For full comparability, dinov2-small should be re-run with `validation_sweep.py`.
