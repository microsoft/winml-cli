# PR-09: Normalize input_constraints Before Computing Case Signature (#23)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `d436597` |
| Date | 2026-04-01 |
| Author | Hualiang Xie (xieofxie) |
| PR Number | #23 |
| Files Changed | 4 |
| Insertions | +188 |
| Deletions | -8 |

## Summary
Fixed a correctness issue in `op_input_gen.py` where `input_constraints` were not normalized before computing the case signature, which could cause duplicate or incorrect cache entries in the runtime checker. The `check_ops.py` and `result_processor.py` were updated for consistency. A new dedicated test file `tests/unit/analyze/test_check_ops.py` (146 lines) was added to verify case signature computation and normalization behavior.

## Files Changed
- `src/winml/modelkit/pattern/op_input_gen/op_input_gen.py` — normalization logic before case signature computation (+25)
- `src/winml/modelkit/analyze/runtime_checker/check_ops.py` — consistency fix (+13/-1)
- `src/winml/modelkit/analyze/runtime_checker/result_processor.py` — consistency fix (+12/-8)
- `tests/unit/analyze/test_check_ops.py` — new test file (+146)
