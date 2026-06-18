# Feature Gap Issues — WinML autoconfig Research

Issues filed from the autoconfig catalog sweep research.
Updated manually after each new finding.

---

## QNN NPU / QNN GPU Findings

| Issue | Title | Status | Source Finding | Date |
|---|---|---|---|---|
| [#921](https://github.com/microsoft/winml-cli/issues/921) | analyze: detect Gemm→Reshape→Transpose hybrid-unfold pattern; warn before applying highdimRTR | OPEN | npu-010, gpu-008 | 2026-06-18 |

### #921 Context

**Root cause (npu-010 / gpu-008):** MobileViT's CNN encoder uses an unfold operation
(`Gemm→Reshape→Transpose`) that is misidentified by `highdimRTR_lowdimRTR` as an
optimizable RTR chain. The optimizer inserts ~36 spurious Reshape nodes after Gemm layers,
increasing memory traffic.

**Measured impact:**
- MobileViT QNN NPU: -19.5% regression (h9, 3×500 iters, DISCARD)
- MobileViT QNN GPU: -6.9% regression (cross-EP confirmation)
- DINOv2 QNN NPU: +38.1% gain (pure-ViT, no unfold blocks — optimization works correctly)

**Fix needed:** `analyze_insight.py` should detect `Gemm→Reshape→Transpose` patterns and
add `highdimRTR` to the skip_set for models with this signature.

---

## CPU Findings

| Issue | Title | Status | Source Finding | Date |
|---|---|---|---|---|
| *(pending)* | cpu-001: opset regression should be flagged in winml build output | — | cpu-001 | — |
| *(pending)* | cpu-008: layer_norm_fusion harmful on CNN-ViT hybrid — add guard to optimize | — | cpu-008 | — |

---

## Cross-EP / Infrastructure

| Issue | Title | Status | Source Finding | Date |
|---|---|---|---|---|
| *(pending)* | winml optimize: add FusedConv detection and unfuse path for QNN EP | — | npu-006 | — |
| *(pending)* | winml perf: add DVFS-aware protocol flag (multi-session + cool-down) | — | npu-007 | — |

---

## How to add a new entry

1. File the GitHub issue: `gh issue create --repo microsoft/winml-cli --title "..." --body "..."`
2. Note the issue number
3. Add a row to the relevant table above
4. Add a `### #NNN Context` section with:
   - Root cause
   - Measured impact (model, EP, gain/loss %, protocol)
   - Fix needed
