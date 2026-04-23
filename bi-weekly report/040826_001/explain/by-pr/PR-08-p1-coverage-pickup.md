# PR-08: pickup p1_coverage — Final Version (#22)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `603309d` |
| Date | 2026-03-31 |
| Author | Fangyang Ci (fangyangci) |
| PR Number | #22 |
| Files Changed | 18 |
| Insertions | +131 |
| Deletions | -15 |

## Summary
Re-landed the p1 coverage work (after the prior attempt in PR #20 was reverted). Expands QNN NPU rule zips for opsets 12–22 with larger coverage data. Updates `result_processor.py` to improve p1 model result handling, and extends three input generators (`indexing_input_generator.py`, `slice_input_generator.py`, `squeeze_input_generator.py`) with additional test coverage. Adds `tests/unit/analyze/test_input_generators.py` (61 lines of new test coverage). Also updates the `runtime_checker/README.md` and fixes a test in `test_qdq.py`.

## Files Changed
- `src/winml/modelkit/analyze/runtime_checker/result_processor.py` — p1 result handling (+50/-3)
- `src/winml/modelkit/pattern/op_input_gen/indexing_input_generator.py` — additional coverage (+9/-1)
- `src/winml/modelkit/pattern/op_input_gen/slice_input_generator.py` — additional coverage (+8)
- `src/winml/modelkit/pattern/op_input_gen/squeeze_input_generator.py` — additional coverage (+9/-1)
- `tests/unit/analyze/test_input_generators.py` — new test file (+61)
- QNN NPU rule zips opset12–22 — updated with expanded p1 coverage data
