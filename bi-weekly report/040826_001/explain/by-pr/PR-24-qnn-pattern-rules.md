# PR-24: QNN Pattern Rules and GPU Rules Update (#210 + #234)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commits | `6469aa6` (#210, Charles Zhang, 2026-04-01), `fd43a2f` (#234, Charles Zhang, 2026-04-02) |
| Author | Charles Zhang (chinazhangchao) |
| PR Numbers | #210, #234 |
| Files Changed | 3 total |

## Summary
PR #210 added QNN pattern rules to the runtime checker: extended `runtime_checker.py` (+5 lines) to check for pattern-based support rules when evaluating QNN EP operators, and updated the QNN NPU opset17 rule zip with expanded pattern data. PR #234 is a data-only update replacing the QNN GPU opset17 rule zip with a larger version reflecting additional GPU operator coverage.

## Files Changed
- `src/winml/modelkit/analyze/core/runtime_checker.py` — QNN pattern rule support (+5/-1) (#210)
- `src/winml/modelkit/analyze/rules/runtime_check_rules/QNNExecutionProvider_NPU_ai.onnx_opset17.zip` — expanded (#210)
- `src/winml/modelkit/analyze/rules/runtime_check_rules/QNNExecutionProvider_GPU_ai.onnx_opset17.zip` — updated (#234)
