# PR-05: Mock WinML EP Init for Session Tests (#18)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `17f1b52` |
| Date | 2026-03-30 |
| Author | Zhipeng Wang (timenick) |
| PR Number | #18 |
| Files Changed | 1 |
| Insertions | +26 |
| Deletions | -3 |

## Summary
Added an autouse pytest fixture to `tests/session/conftest.py` that mocks `WinMLSession.__init__`'s call to `_init_winml_eps_once()`, which unconditionally triggers WinML SDK runtime initialization and can hang indefinitely on CI runners without the SDK installed. A companion fix filters e2e tests out of the collection stage to prevent `WinMLEPRegistry.get_instance()` from being triggered during collection before `-m` filtering takes effect. E2e tests continue to use real initialization.

## Files Changed
- `tests/session/conftest.py` — autouse fixture mocking WinML SDK init; e2e item filtering in `pytest_collection_modifyitems`
