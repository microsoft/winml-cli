# PR-20: Fix charmap Codec Errors on Windows (#200 + #208)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commits | `92325bf` (#200, Qiong Wu, 2026-04-01), `903ce0a` (#208, Zhipeng Wang, 2026-04-01) |
| Authors | Qiong Wu (DingmaomaoBJTU); Zhipeng Wang (timenick) |
| PR Numbers | #200, #208 |
| Files Changed | 6 total |
| Insertions | +35 +20 |
| Deletions | -37 +9 |

## Summary
Two related PRs fixing Windows `cp1252` `UnicodeEncodeError` crashes caused by non-ASCII characters (emoji, check marks, Unicode arrows) in console output. PR #200 fixed `wmk analyze` and `wmk inspect`: removed `legacy_windows=False` from `StaticAnalyzerConsoleWriter` (which bypassed Win32 console APIs) and replaced non-ASCII symbols in `console_writer.py`, `commands/analyze.py`, and `inspect/formatter.py` with ASCII equivalents. PR #208 extended the same fix to all remaining command files that had non-ASCII characters.

## Files Changed
- `src/winml/modelkit/analyze/console_writer.py` — removed legacy_windows flag, replaced Unicode symbols (#200)
- `src/winml/modelkit/commands/analyze.py` — ASCII symbol replacements (#200)
- `src/winml/modelkit/inspect/formatter.py` — ASCII symbol replacements (#200)
- Remaining command files with non-ASCII characters fixed in (#208)
