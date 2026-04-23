# Module: Runtime Check Rules
**Path**: `src/winml/modelkit/analyze/rules/runtime_check_rules/`
**Period**: 2026-03-23 to 2026-04-08

## 1. Module Overview
The `runtime_check_rules/` directory holds zip archives of EP operator support data used by the runtime checker to determine which ONNX operators are supported by each execution provider (QNN NPU/GPU, OpenVINO CPU/GPU/NPU, WinML NPU, VitisAI NPU) across different ONNX opsets.

## 2. Files Changed This Period
| File | PRs | Summary |
|------|-----|---------|
| `QNNExecutionProvider_NPU_ai.onnx_opset11.zip` | #15 | Added (new EP coverage) |
| `QNNExecutionProvider_NPU_ai.onnx_opset12–22.zip` | #15, #22, #213 | Expanded in #15 (batch); expanded again in #22 (p1 coverage); removed from git in #213 |
| `QNNExecutionProvider_NPU_ai.onnx_opset17.zip` | #19, #210 | Subgraph result update (#19); QNN pattern rules added (#210) |
| `QNNExecutionProvider_GPU_ai.onnx_opset17.zip` | #15, #234, #213 | Updated in #15; GPU rules expanded (#234); removed from git (#213) |
| `OpenVINOExecutionProvider_NPU_ai.onnx_opset17.zip` | #15, #19, #213 | Updated in #15; subgraph results (#19); removed (#213) |
| All other EP zips | #15, #213 | Batch updated in #15; all removed from git in #213 |
| `README.md` | #213, #251 | Developer setup guide added (#213); gh auth instructions (#251) |

## 3. Net Change Summary
- All 22 rule zip files were removed from git tracking in PR #213. They are now hosted in the external `gim-home/ModelKitArtifacts` repository and fetched via `scripts/download_rules.py`.
- Before removal, the zip files received multiple updates: QNN NPU opsets 12–22 expanded significantly with p1 model coverage in PRs #15 and #22; QNN NPU opset17 received QNN pattern rule data in PR #210; QNN GPU opset17 was updated in PR #234.
- A `README.md` was added to the directory explaining the external hosting setup and providing developer instructions for obtaining the zips.

## 4. New APIs/Functions Added
| Symbol | Description |
|--------|-------------|
| `scripts/download_rules.py` | New script to fetch rule zips from external repo (#213) |
| `README.md` | Developer setup documentation for rule zip acquisition (#213, #251) |
