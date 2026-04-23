# PR-04: QDQParameterConfig qdq_types Refactor (#17)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `247698e` |
| Date | 2026-03-30 |
| Author | Hualiang Xie (xieofxie) |
| PR Number | #17 |
| Files Changed | 6 |
| Insertions | +113 |
| Deletions | -124 |

## Summary
Renamed the `support_weight` and `support_activation` boolean fields in `QDQParameterConfig` to a single `qdq_types` field. This consolidates the two separate flags into a unified configuration for which tensor types (weights, activations) participate in QDQ quantization. Updated all four input generators that referenced the old fields (conv, matmul, normalization, op_input_gen) and the corresponding QDQ test suite. Also touched `_pdh.py` for unrelated monitor cleanup.

## Files Changed
- `src/winml/modelkit/pattern/op_input_gen/op_input_gen.py` — core QDQParameterConfig field rename
- `src/winml/modelkit/pattern/op_input_gen/conv_input_generator.py` — updated to use qdq_types
- `src/winml/modelkit/pattern/op_input_gen/matmul_input_generator.py` — updated to use qdq_types
- `src/winml/modelkit/pattern/op_input_gen/normalization_input_generator.py` — updated to use qdq_types
- `src/winml/modelkit/session/monitor/_pdh.py` — minor monitor updates
- `tests/unit/analyze/core/test_qdq.py` — test cases updated for new field name
