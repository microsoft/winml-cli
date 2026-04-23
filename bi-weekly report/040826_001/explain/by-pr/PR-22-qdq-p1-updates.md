# PR-22: QDQ Config Updates for P1 Models (#204 + #236)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commits | `b606007` (#204, Hualiang Xie, 2026-04-01), `7f6e9e5` (#236, Hualiang Xie, 2026-04-08) |
| Author | Hualiang Xie (xieofxie) |
| PR Numbers | #204, #236 |
| Files Changed | 9 total |
| Insertions | +71 |
| Deletions | -18 |

## Summary
Two batches of QDQ input generator updates targeting P1 model coverage. PR #204 extended `conv_input_generator.py` with additional QDQ type handling, fixed a return value in `matmul_input_generator.py`, added a `resize_input_generator.py` extension for QDQ, and corrected `transpose_input_generator.py`. PR #236 further extended `resize_input_generator.py` (5 lines) and added QDQ support to `unary_input_generator.py` (9 lines), along with an additional test case in `test_qdq.py`.

## Files Changed
- `src/winml/modelkit/pattern/op_input_gen/conv_input_generator.py` — QDQ type extension (#204)
- `src/winml/modelkit/pattern/op_input_gen/matmul_input_generator.py` — return fix (#204)
- `src/winml/modelkit/pattern/op_input_gen/resize_input_generator.py` — QDQ support (#204, #236)
- `src/winml/modelkit/pattern/op_input_gen/transpose_input_generator.py` — correction (#204)
- `src/winml/modelkit/pattern/op_input_gen/unary_input_generator.py` — QDQ support (#236)
- `tests/unit/analyze/core/test_qdq.py` — additional test cases (#204, #236)
- `src/winml/modelkit/analyze/runtime_checker/result_processor.py` — minor update (#204)
