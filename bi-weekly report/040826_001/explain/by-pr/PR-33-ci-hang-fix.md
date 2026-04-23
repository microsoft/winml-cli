# PR-33: Fix test_runtime_checker.py CI Hangs by Mocking Hardware Probing (#252)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `ba39203` |
| Date | 2026-04-06 |
| Author | Copilot |
| PR Number | #252 |
| Files Changed | 1 |
| Insertions | +4 |
| Deletions | 0 |

## Summary
Added an autouse fixture to `tests/conftest.py` that patches `RuntimeCheckerQuery._is_ep_available_locally()` to return `False`. This prevents `winml.register_execution_providers()` and `ort.get_ep_devices()` from being called during unit tests, both of which perform hardware probing that can hang indefinitely on CI runners without QNN or WinML hardware installed. The fixture was placed in the root conftest for global coverage.

## Files Changed
- `tests/conftest.py` — autouse fixture patching `_is_ep_available_locally` (+4)
