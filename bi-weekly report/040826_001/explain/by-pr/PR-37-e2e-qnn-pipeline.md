# PR-37: Add E2E Eval Pipeline for QNN NPU Models (#242)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `6116e2c` |
| Date | 2026-04-08 |
| Author | Yue Sun (KayMKM) |
| PR Number | #242 |
| Files Changed | 2 |
| Insertions | +146 |
| Deletions | -54 |

## Summary
Extended the Azure DevOps E2E pipeline (`.pipelines/Modelkit E2E Test.yml`) with QNN NPU-specific stages: added date variable injection, model list continuation flag, artifact publishing, improved job naming, and refined the clean step. Extensively refactored `scripts/e2e_eval/run_eval.py` to support NPU-targeted evaluation runs, improve error-to-warning demotion, and add `--continue` behavior for model list processing. Also cleaned up `.onnx.data` external files during evaluation cleanup.

## Files Changed
- `.pipelines/Modelkit E2E Test.yml` — QNN NPU pipeline stages, artifact publish, date variable (+60 lines net)
- `scripts/e2e_eval/run_eval.py` — NPU eval support, continue flag, warning demotion, temp cleanup (+140/-54)
