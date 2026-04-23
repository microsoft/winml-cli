# PR-25: Fix facebook/sam2.1-hiera-small and WinML Registration Issue (#212 + #235)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commits | `e1087e0` (#212, Charles Zhang, 2026-04-02), `4285d18` (#235, Charles Zhang, 2026-04-03) |
| Author | Charles Zhang (chinazhangchao) |
| PR Numbers | #212, #235 |
| Files Changed | 7 total |
| Insertions | +190 |
| Deletions | -118 |

## Summary
PR #212 fixed the `facebook/sam2.1-hiera-small` model failing during export: updated `export/io.py` to handle pixel mask outputs correctly, updated `models/hf/sam.py` with improved model configuration handling (60-line update), and added a guard in `optim/api.py`. PR #235 fixed a WinML EP registration issue in the runtime checker: refactored `check_ops.py` by extracting a new `src/winml/modelkit/analyze/utils/op_utils.py` utility module (97 lines) and simplifying `check_ops.py` significantly (-102 lines net), along with test updates in `test_check_ops.py`.

## Files Changed
- `src/winml/modelkit/export/io.py` — pixel mask handling fix (+21/-4) (#212)
- `src/winml/modelkit/models/hf/sam.py` — model config improvements (+60/-5) (#212)
- `src/winml/modelkit/optim/api.py` — guard addition (#212)
- `src/winml/modelkit/analyze/utils/op_utils.py` — new utility module (+97) (#235)
- `src/winml/modelkit/analyze/runtime_checker/check_ops.py` — refactored using op_utils (+6/-102) (#235)
- `tests/unit/analyze/test_check_ops.py` — updated tests (#235)
