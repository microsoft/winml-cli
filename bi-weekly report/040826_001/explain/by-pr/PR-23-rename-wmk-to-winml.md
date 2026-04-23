# PR-23: Rename CLI Command from wmk to winml (#205)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `2500b9b` |
| Date | 2026-04-01 |
| Author | Zhipeng Wang (timenick) |
| PR Number | #205 |
| Files Changed | 49 |
| Insertions | +531 |
| Deletions | -467 |

## Summary
Renamed the CLI entry point from `wmk` to `winml` across the entire codebase. Updated `pyproject.toml` entry point, `cli.py` prog_name, all 12 command files, source code references (error messages, cache env var `WMK_CACHE_DIR` → `WINML_CACHE_DIR`, temp dir prefix, venv name), 13 test files, README, and all `scripts/e2e_eval/` scripts and JSON config (metric key `wmk_metric_key` → `winml_metric_key`). Also renamed the e2e eval runner variable `WMK` → `WINML_CLI`.

## Files Changed (key)
- `pyproject.toml` — entry point renamed to `winml`
- `src/winml/modelkit/cli.py` — prog_name updated
- `src/winml/modelkit/commands/` — all 12 command files (analyze, build, compile, config, eval, export, hub, inspect, optimize, perf, quantize, sys)
- `src/winml/modelkit/cache/path.py` — env var renamed
- `scripts/e2e_eval/run_eval.py` — WMK → WINML_CLI; metric key rename
- `scripts/e2e_eval/testsets/models_with_acc.json` — metric key renamed in 44 entries
- 13 test files updated
