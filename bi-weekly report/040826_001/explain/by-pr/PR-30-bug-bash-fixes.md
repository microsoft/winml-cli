# PR-30: Resolve 6 Bug Bash Issues (#246)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `7123665` |
| Date | 2026-04-03 |
| Author | Qiong Wu (DingmaomaoBJTU) |
| PR Number | #246 (closes #228, #229, #230, #231, #232, #233) |
| Files Changed | 12 |
| Insertions | +180 |
| Deletions | -59 |

## Summary
Resolved six bug bash issues found on the main branch in a single consolidated PR:

- **#228**: Replaced non-ASCII arrow (U+2192) with ASCII `->` in capability and rewrite-rule descriptions (`graph.py`, `layernorm.py`, `surgery.py`, `rewrite_rules.py`) to fix `winml optimize --help` crashes on cp1252 terminals.
- **#231**: Converted debug `print()` statements in `QDQGenerator` to `logger.debug()` to stop stdout pollution during `winml analyze`.
- **#229**: Fixed `winml sys --list-device --list-ep --format json` to emit a single valid JSON object `{devices, executionProviders}` instead of two separate arrays.
- **#232**: Fixed `winml sys --format compact` to actually produce compact single-line output instead of silently falling back to text format; removed dead `_output_device_json`/`_output_ep_json` helpers.
- **#230**: Added `stderr=subprocess.DEVNULL` to all PowerShell subprocess calls in `sysinfo/helper.py` to suppress NPU-absent error noise.
- **#233**: Changed Rich text/column overflow from `'ellipsis'` to `'crop'` in hub list view to avoid emitting U+2026 on cp1252 terminals.

Regression tests added for all 6 fixes.

## Files Changed
- `src/winml/modelkit/commands/sys.py` — JSON output fix, compact format fix
- `src/winml/modelkit/commands/hub.py` — overflow mode fix
- `src/winml/modelkit/optim/capabilities/graph.py`, `layernorm.py`, `surgery.py` — ASCII arrow fix
- `src/winml/modelkit/optim/pipes/rewrite_rules.py` — ASCII arrow fix
- `src/winml/modelkit/pattern/op_input_gen/qdq_gen.py` — print → logger.debug
- `src/winml/modelkit/sysinfo/helper.py` — stderr=DEVNULL
- `tests/regression/test_design_gaps.py` — regression tests (+46)
- `tests/unit/analyze/core/test_qdq.py` — QDQ logger test (+9)
- `tests/unit/commands/test_cli.py`, `test_hub.py` — CLI and hub tests
