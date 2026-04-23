# PR-18: Add wmk hub Command (#196)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `78442c4` |
| Date | 2026-04-01 |
| Author | Qiong Wu (DingmaomaoBJTU) |
| PR Number | #196 |
| Files Changed | 6 |
| Insertions | +1,566 |
| Deletions | 0 |

## Summary
Added a new `hub` subcommand to the CLI (at the time named `wmk`, later renamed to `winml` in PR #205). The command provides a model registry browser backed by `hub_models.json` (750 entries), supporting list and detail views via Rich console output. The implementation in `commands/hub.py` (471 lines) includes search/filter functionality, a list view, and a detail panel. A full test suite `tests/unit/commands/test_hub.py` (343 lines) was added alongside.

## Files Changed
- `src/winml/modelkit/commands/hub.py` — new command implementation (+471)
- `src/winml/modelkit/data/hub_models.json` — model registry data (+750 entries)
- `src/winml/modelkit/commands/__init__.py` — registered hub command
- `pyproject.toml` — hub command entry
- `tests/unit/commands/test_hub.py` — new test file (+343)
- `tests/unit/commands/__init__.py` — package init
