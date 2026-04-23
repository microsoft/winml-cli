# PR-10: Move WinML EP Mock to Root conftest (#24)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `2e5af47` |
| Date | 2026-03-31 |
| Author | Zhipeng Wang (timenick) |
| PR Number | #24 |
| Files Changed | 2 |
| Insertions | +20 |
| Deletions | -17 |

## Summary
Discovered that `winml.modelkit.models.winml.base` imports `WinMLSession` at module level, which triggers WinML SDK initialization on any import of the winml.modelkit package — not just during session tests. Moved the autouse WinML SDK init mock from `tests/session/conftest.py` to the root `tests/conftest.py` so all test groups are protected against SDK initialization hangs on CI, not just the session group.

## Files Changed
- `tests/conftest.py` — mock fixture added at root scope (+20 lines)
- `tests/session/conftest.py` — fixture removed to avoid duplication (-17 lines)
