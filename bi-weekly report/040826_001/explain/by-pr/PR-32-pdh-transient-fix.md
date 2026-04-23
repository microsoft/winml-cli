# PR-32: Fix PDH Query Transient Failure in Tests (#241)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `e361726` |
| Date | 2026-04-03 |
| Author | Charles Zhang (chinazhangchao) |
| PR Number | #241 |
| Files Changed | 2 |
| Insertions | +37 |
| Deletions | -16 |

## Summary
Fixed a transient test failure caused by Windows PDH rate counters returning `None` immediately after `prime()` on busy systems. Moved the retry logic from the test into `PdhQuery.collect()` itself so all callers benefit from the robustness improvement. The internal poll loop was refactored to use `_collect_once()` to avoid recursive stalling during retries.

## Files Changed
- `src/winml/modelkit/session/monitor/_pdh.py` — retry logic moved into collect(); _collect_once() added (+49/-8)
- `tests/unit/session/test_ep_monitor.py` — simplified test now relies on collect() retry (+4/-2)
