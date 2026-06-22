# QNN NPU Optimization Sweep — Catalog Models

Generated: 2026-06-22T09:40:21  
EP: `qnn` / device: `npu`  
Bench protocol: Phase-A 200 iters (high CV expected on QNN NPU — DVFS), Phase-B 500x3 sessions, 30s cool-down  
npu-001 criterion: median >=5% gain AND ranges non-overlapping  
npu-006 criterion: Conv% of ops; h4/h5 marked catastrophic if >=5x baseline  
Effect-size gate: gain reliable only if gain% >= 2×(session-CV) AND ranges separated  

---

## Per-Model Results

| Model | Conv% | Baseline p50 | Best p50 | Best config | Gain% | Reliable? | npu-001? | npu-006 regression? | Notes |
|-------|-------|-------------|----------|-------------|-------|-----------|----------|---------------------|-------|
| `apple/mobilevit-small` | 2% | 5.5 ms | 5.4 ms | h3 (opset 21 (tests npu-001 bypass)) | 2.8% | ⚠️ within noise | neutral | no | none |

## Hypothesis Breakdown per Model

### apple/mobilevit-small

| Hypothesis | Opset | Screen p50 | Full p50 (median) | CV | Status | Accuracy |
|------------|-------|-----------|-------------------|-----|--------|---------|
| h0 (baseline (auto-config, W8A16)) | 17 | 5.0 | 5.5 | 0.093 | OK | — |
| h1 (opset 17 explicit) | 17 | 5.8 | 5.6 | 0.304 | OK_HIGH_CV ⚡DVFS | — |
| h2 (opset 19) | 19 | 5.8 | 6.6 | 0.120 | OK | — |
| h3 (opset 21 (tests npu-001 bypass)) | 21 | 5.2 | 5.4 | 0.163 | OK_HIGH_CV ⚡DVFS | — |
| h4 (opset 17 + conv fusions) | 17 | 6.7 | 6.5 | 0.181 | OK_HIGH_CV ⚡DVFS | — |
| h5 (opset 21 + conv fusions) | 21 | 6.2 | 6.7 | 0.153 | OK_HIGH_CV ⚡DVFS | — |
| h6 (opset 21 + matmul_transpose_fusion) | 21 | 5.9 | 6.2 | 0.229 | OK_HIGH_CV ⚡DVFS | — |
| h7 (opset 21 + bias_softmax_fusion) | 21 | 4.6 | 6.4 | 0.043 | OK | — |
| h8 (opset 21 + attention_fusion) | 21 | 6.5 | 5.8 | 0.455 | OK_HIGH_CV ⚡DVFS | — |
| h9 (opset 21 + highdimRTR_lowdimRTR) | 21 | 5.7 | 6.5 | 0.190 | OK_HIGH_CV ⚡DVFS | — |
| h10 (opset 17 + conv_add_fusion only) | 17 | 6.7 | 5.9 | 0.188 | OK_HIGH_CV ⚡DVFS | — |

---

## Cross-Model Patterns

### npu-001: Does opset 21 bypass help broadly?

- **Helps (0 models):** none
- **Hurts (0 models):** none
- **Neutral (1 models):** `apple/mobilevit-small`
- **N/A (0 models):** none

> **Finding**: Mixed results (0 help, 0 hurt, 1 neutral). Architecture-dependent. Confirm ORT `kMaxSupportedOpset` version before drawing conclusions.

### Feature Gaps

- No feature gaps observed

### Build / Compatibility Issues


---

## Updated Recommendations for `ep_knowledge/qnn_npu.json`

Based on this cross-architecture sweep:


### Conv Fusion Findings (h4 vs h1, h5 vs h3)

- **`apple/mobilevit-small`**: conv-fusions on opset17: -16.0% (5.6→6.5ms); conv-fusions on opset21: -25.3% (5.4→6.7ms)
