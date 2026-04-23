# PR-31: Fix Slice/Split derive_properties Crash on Symbolic Dynamic Axes (#244)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `f6f9f82` |
| Date | 2026-04-03 |
| Author | Reny (vortex-captain) |
| PR Number | #244 |
| Files Changed | 3 |
| Insertions | +34 |
| Deletions | -14 |

## Summary
Fixed a `ValueError` crash in `derive_properties` for Slice and Split ops when the input data shape contains symbolic dimension names (e.g., `"time"`) instead of integer values. The fix filters to fixed-shape axes only when computing `starts_equal_shape` and `slice_all` properties in `slice_input_generator.py`. Also fixed `Split`'s `derive_properties` to use `n_outputs` from the caller when both `split_value` and `attr_num_outputs` are absent. A minor update to `runtime_checker_query.py` handles symbolic axes at the query level.

## Files Changed
- `src/winml/modelkit/pattern/op_input_gen/slice_input_generator.py` — symbolic axis filter (+37/-7)
- `src/winml/modelkit/analyze/core/runtime_checker_query.py` — symbolic axis guard (+6)
- `src/winml/modelkit/pattern/op_input_gen/indexing_input_generator.py` — Split n_outputs fix (+5/-1)
