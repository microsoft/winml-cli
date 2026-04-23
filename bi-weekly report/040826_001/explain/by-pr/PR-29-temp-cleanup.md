# PR-29: Clean .onnx.data Files in Temp (#237)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `d16c00b` |
| Date | 2026-04-03 |
| Author | Yue Sun (KayMKM) |
| PR Number | #237 |
| Files Changed | 2 |
| Insertions | +10 |
| Deletions | -7 |

## Summary
Fixed a resource cleanup issue in the optimizer pipe infrastructure where `.onnx.data` external data files were left behind in the temp directory after optimization passes. Updated `optim/pipes/fusion.py` and `optim/pipes/graph.py` to also delete companion `.onnx.data` files when cleaning up temporary ONNX model files.

## Files Changed
- `src/winml/modelkit/optim/pipes/fusion.py` — cleanup of .onnx.data external data files (+6/-2)
- `src/winml/modelkit/optim/pipes/graph.py` — cleanup of .onnx.data external data files (+11/-5)
