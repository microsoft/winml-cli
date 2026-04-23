# PR-21: Suppress ep_registry Log Leaks in winml sys Output (#201)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `9bcf4ad` |
| Date | 2026-04-01 |
| Author | Qiong Wu (DingmaomaoBJTU) |
| PR Number | #201 |
| Files Changed | 2 |
| Insertions | +90 |
| Deletions | -68 |

## Summary
Fixed `winml sys` command output being polluted by EP registry log messages leaking to stdout. Rewrote `commands/sys.py` to properly suppress and restore the `winml.modelkit` logger state around `sysinfo` calls, clearing stale `RichHandler` instances and dropping underscore prefixes from internal helpers. Also added a minimal fix to `session/ep_registry.py` to prevent registry initialization logs from reaching the root logger.

## Files Changed
- `src/winml/modelkit/commands/sys.py` — logger suppression logic; command refactoring (+90/-68)
- `src/winml/modelkit/session/ep_registry.py` — log level fix to prevent leaking to root logger
