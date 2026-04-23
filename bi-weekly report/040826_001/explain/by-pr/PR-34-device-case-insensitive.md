# PR-34: Make --device Flag Case-Insensitive (#264)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `e48ad29` |
| Date | 2026-04-08 |
| Author | Qiong Wu (DingmaomaoBJTU) |
| PR Number | #264 (closes #215) |
| Files Changed | 1 |
| Insertions | +1 |
| Deletions | -1 |

## Summary
Fixed an inconsistency where `winml analyze --device` only accepted uppercase values (`CPU`, `GPU`, `NPU`) while all other commands (`compile`, `eval`, `perf`, `config`) accepted any case. Changed the Click `Choice` for the device option in `utils/cli.py` to `case_sensitive=False`. Click normalizes the value to uppercase automatically, so no downstream code changes were needed.

## Files Changed
- `src/winml/modelkit/utils/cli.py` — `case_sensitive=False` added to device Choice (+1/-1)
