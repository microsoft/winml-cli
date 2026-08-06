# QNN NPU Optimization Sweep — Catalog Models

Generated: 2026-06-22T12:29:01  
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
| `deepset/roberta-base-squad2` | N/A | 14.9 ms | 14.7 ms | h1 (opset 17 explicit) | 1.5% | N/A | neutral | no | Model timed out at 1466s (before h4); Model timed out at 1466s (before h5) |
| `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | N/A | 19.5 ms | 19.5 ms | h2 (opset 19) | 0.0% | N/A | neutral | no | Model timed out at 1385s (before h5) |
| `facebook/dinov2-small` | N/A | 6.6 ms | 5.0 ms | h3 (opset 21 (tests npu-001 bypass)) | 24.1% | N/A | YES (median) | no | Model timed out at 1333s (before h4); Model timed out at 1333s (before h5) |
| `google/vit-base-patch16-224` | N/A | 9.0 ms | 9.0 ms | h0 (baseline (auto-config, W8A16)) | 0.0% | N/A | NO | no | h2: BUILD_FAIL; Model timed out at 1204s (before h4); Model timed out at 1204s ( |
| `hustvl/yolos-small` | 0% | 49.6 ms | 48.6 ms | h3 (opset 21 (tests npu-001 bypass)) | 2.0% | ⚠️ within noise | N/A | no | h2 (opset 19), h4/h5 (conv fusions): not measured — agent deprioritized (yolos i |
| `microsoft/resnet-18` | N/A | 1.0 ms | 1.0 ms | h0 (baseline (auto-config, W8A16)) | 0.0% | N/A | YES (median) | no | Model timed out at 1560s (before h5) |
| `sentence-transformers/all-MiniLM-L6-v2` | N/A | 5.8 ms | 5.8 ms | h0 (baseline (auto-config, W8A16)) | 0.0% | N/A | neutral | no | Model timed out at 1346s (before h5) |

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

### deepset/roberta-base-squad2

| Hypothesis | Opset | Screen p50 | Full p50 (median) | CV | Status | Accuracy |
|------------|-------|-----------|-------------------|-----|--------|---------|
| h0 (baseline (auto-config, W8A16)) | 17 | 14.9 | 14.9 | 0.119 | OK | — |
| h1 (opset 17 explicit) | 17 | 14.7 | 14.7 | 0.129 | OK | — |
| h2 (opset 19) | 19 | 15.3 | 14.9 | 0.234 | OK_HIGH_CV ⚡DVFS | — |
| h3 (opset 21 (tests npu-001 bypass)) | 21 | 14.8 | 14.9 | 0.116 | OK | — |
| h4 (opset 17 + conv fusions) | ? | — | — | ? | TIMEOUT | — |
| h5 (opset 21 + conv fusions) | ? | — | — | ? | TIMEOUT | — |

### distilbert/distilbert-base-uncased-finetuned-sst-2-english

| Hypothesis | Opset | Screen p50 | Full p50 (median) | CV | Status | Accuracy |
|------------|-------|-----------|-------------------|-----|--------|---------|
| h0 (baseline (auto-config, W8A16)) | 17 | 19.5 | 19.5 | 0.156 | OK_HIGH_CV ⚡DVFS | — |
| h1 (opset 17 explicit) | 17 | 19.7 | 19.5 | 0.272 | OK_HIGH_CV ⚡DVFS | — |
| h2 (opset 19) | 19 | 19.4 | 19.5 | 0.195 | OK_HIGH_CV ⚡DVFS | — |
| h3 (opset 21 (tests npu-001 bypass)) | 21 | 19.4 | 19.5 | 0.290 | OK_HIGH_CV ⚡DVFS | — |
| h4 (opset 17 + conv fusions) | 17 | 19.4 | 19.6 | 0.237 | OK_HIGH_CV ⚡DVFS | — |
| h5 (opset 21 + conv fusions) | ? | — | — | ? | TIMEOUT | — |

### facebook/dinov2-small

| Hypothesis | Opset | Screen p50 | Full p50 (median) | CV | Status | Accuracy |
|------------|-------|-----------|-------------------|-----|--------|---------|
| h0 (baseline (auto-config, W8A16)) | 17 | 7.2 | 6.6 | 0.344 | OK_HIGH_CV ⚡DVFS | — |
| h1 (opset 17 explicit) | 17 | 4.9 | 7.2 | 0.457 | OK_HIGH_CV ⚡DVFS | — |
| h2 (opset 19) | 19 | 7.0 | 7.2 | 1.805 | OK_HIGH_CV ⚡DVFS | — |
| h3 (opset 21 (tests npu-001 bypass)) | 21 | 9.4 | 5.0 | 0.936 | OK_HIGH_CV ⚡DVFS | — |
| h4 (opset 17 + conv fusions) | ? | — | — | ? | TIMEOUT | — |
| h5 (opset 21 + conv fusions) | ? | — | — | ? | TIMEOUT | — |

### google/vit-base-patch16-224

| Hypothesis | Opset | Screen p50 | Full p50 (median) | CV | Status | Accuracy |
|------------|-------|-----------|-------------------|-----|--------|---------|
| h0 (baseline (auto-config, W8A16)) | 17 | 9.2 | 9.0 | 1.289 | OK_HIGH_CV ⚡DVFS | 0.740 |
| h1 (opset 17 explicit) | 17 | 9.7 | 9.3 | 0.743 | OK_HIGH_CV ⚡DVFS | — |
| h2 (opset 19) | 19 | — | — | ? | BUILD_FAIL | — |
| h3 (opset 21 (tests npu-001 bypass)) | 21 | 11.6 | 10.0 | 2.159 | OK_HIGH_CV ⚡DVFS | — |
| h4 (opset 17 + conv fusions) | ? | — | — | ? | TIMEOUT | — |
| h5 (opset 21 + conv fusions) | ? | — | — | ? | TIMEOUT | — |

### hustvl/yolos-small

| Hypothesis | Opset | Screen p50 | Full p50 (median) | CV | Status | Accuracy |
|------------|-------|-----------|-------------------|-----|--------|---------|
| h0 (baseline (auto-config, W8A16)) | 17 | 48.7 | 49.6 | 0.067 | OK | — |
| h1 (opset 17 explicit) | 17 | 66.4 | 65.9 | 0.226 | OK_HIGH_CV ⚡DVFS | — |
| h2 (opset 19) | ? | — | — | ? | TIMEOUT | — |
| h3 (opset 21 (tests npu-001 bypass)) | 21 | 48.8 | 48.6 | 0.050 | OK | — |
| h4 (opset 17 + conv fusions) | ? | — | — | ? | TIMEOUT | — |
| h5 (opset 21 + conv fusions) | ? | — | — | ? | TIMEOUT | — |
| h6 (opset 21 + matmul_transpose_fusion) | 21 | 49.0 | 50.0 | 0.048 | OK | — |
| h7 (opset 21 + bias_softmax_fusion) | 21 | 49.0 | 51.6 | 0.062 | OK | — |
| h8 (opset 21 + attention_fusion) | 21 | 51.3 | 49.5 | 0.078 | OK | — |

### microsoft/resnet-18

| Hypothesis | Opset | Screen p50 | Full p50 (median) | CV | Status | Accuracy |
|------------|-------|-----------|-------------------|-----|--------|---------|
| h0 (baseline (auto-config, W8A16)) | 17 | 4.0 | 1.0 | 1.690 | OK_HIGH_CV ⚡DVFS | 0.660 |
| h1 (opset 17 explicit) | 17 | 3.1 | 2.7 | 2.036 | OK_HIGH_CV ⚡DVFS | — |
| h2 (opset 19) | 19 | 4.0 | 1.1 | 1.517 | OK_HIGH_CV ⚡DVFS | — |
| h3 (opset 21 (tests npu-001 bypass)) | 21 | 3.0 | 2.2 | 1.176 | OK_HIGH_CV ⚡DVFS | — |
| h4 (opset 17 + conv fusions) | 17 | 128.1 | 132.3 | 1.405 | OK_HIGH_CV ⚡DVFS | — |
| h5 (opset 21 + conv fusions) | ? | — | — | ? | TIMEOUT | — |

### sentence-transformers/all-MiniLM-L6-v2

| Hypothesis | Opset | Screen p50 | Full p50 (median) | CV | Status | Accuracy |
|------------|-------|-----------|-------------------|-----|--------|---------|
| h0 (baseline (auto-config, W8A16)) | 17 | 5.9 | 5.8 | 0.222 | OK_HIGH_CV ⚡DVFS | — |
| h1 (opset 17 explicit) | 17 | 5.9 | 5.9 | 0.999 | OK_HIGH_CV ⚡DVFS | — |
| h2 (opset 19) | 19 | 5.3 | 6.0 | 0.205 | OK_HIGH_CV ⚡DVFS | — |
| h3 (opset 21 (tests npu-001 bypass)) | 21 | 6.0 | 5.9 | 1.127 | OK_HIGH_CV ⚡DVFS | — |
| h4 (opset 17 + conv fusions) | 17 | 5.5 | 6.0 | 0.134 | OK | — |
| h5 (opset 21 + conv fusions) | ? | — | — | ? | TIMEOUT | — |

---

## Cross-Model Patterns

### npu-001: Does opset 21 bypass help broadly?

- **Helps (2 models):** `facebook/dinov2-small`, `microsoft/resnet-18`
- **Hurts (1 models):** `google/vit-base-patch16-224`
- **Neutral (4 models):** `apple/mobilevit-small`, `deepset/roberta-base-squad2`, `distilbert/distilbert-base-uncased-finetuned-sst-2-english`, `sentence-transformers/all-MiniLM-L6-v2`
- **N/A (1 models):** `hustvl/yolos-small`

> **Finding**: Mixed results (2 help, 1 hurt, 4 neutral). Architecture-dependent. Confirm ORT `kMaxSupportedOpset` version before drawing conclusions.

### Feature Gaps

- No feature gaps observed

### Build / Compatibility Issues

**`deepset/roberta-base-squad2`**
  - Model timed out at 1466s (before h4)
  - Model timed out at 1466s (before h5)
**`distilbert/distilbert-base-uncased-finetuned-sst-2-english`**
  - Model timed out at 1385s (before h5)
**`facebook/dinov2-small`**
  - Model timed out at 1333s (before h4)
  - Model timed out at 1333s (before h5)
**`google/vit-base-patch16-224`**
  - h2: BUILD_FAIL
  - Model timed out at 1204s (before h4)
  - Model timed out at 1204s (before h5)
**`hustvl/yolos-small`**
  - h2 (opset 19), h4/h5 (conv fusions): not measured — agent deprioritized (yolos is 0.1% conv / 99.9% transformer, so conv-fusion and intermediate-opset hypotheses are low expected-value).
**`microsoft/resnet-18`**
  - Model timed out at 1560s (before h5)
**`sentence-transformers/all-MiniLM-L6-v2`**
  - Model timed out at 1346s (before h5)

---

## Updated Recommendations for `ep_knowledge/qnn_npu.json`

Based on this cross-architecture sweep:

- **npu-001**: Broaden scope beyond ConvNext. Architectures that benefit: facebook/dinov2-small, microsoft/resnet-18. Update `scope` field and set `gate1_statistical` confidence accordingly.
- **search_space_rules.opset.recommended_order**: Retain `[21, 17]` as default order.

### Conv Fusion Findings (h4 vs h1, h5 vs h3)

- **`apple/mobilevit-small`**: conv-fusions on opset17: -16.0% (5.6→6.5ms); conv-fusions on opset21: -25.3% (5.4→6.7ms)
- **`distilbert/distilbert-base-uncased-finetuned-sst-2-english`**: conv-fusions on opset17: -0.5% (19.5→19.6ms)
- **`microsoft/resnet-18`**: conv-fusions on opset17: -4771.1% (2.7→132.3ms)
- **`sentence-transformers/all-MiniLM-L6-v2`**: conv-fusions on opset17: -1.5% (5.9→6.0ms)
