# PR-19: Fix Stale Module Paths, Import Violations, and Skipped Gelu Tests (#198/#199)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `f4135d7` |
| Date | 2026-04-01 |
| Author | Zhipeng Wang (timenick) |
| PR Numbers | #198, #199 |
| Files Changed | 39 |
| Insertions | +196 |
| Deletions | -217 |

## Summary
Cleaned up a cluster of post-restructure issues. Fixed outdated `python -m modelkit.*` module paths to `python -m winml.modelkit.*` in scripts and docs, replaced stale `static_analyzer` references with `analyze`, and converted approximately 35 absolute imports in source files (`pattern/`, `analyze/`, `optim/`) to relative imports. Also fixed a bug in Gelu rewrite tests where a double `Pattern` suffix in names caused 6 tests to be silently skipped instead of running. Removed the deprecated `_get_model_config()` private method from `models/auto.py`.

## Files Changed (key)
- `scripts/e2e_eval/run_pytorch_baseline.py` — module path fixes
- `src/winml/modelkit/pattern/` — 9 pattern files converted to relative imports
- `src/winml/modelkit/analyze/` — 6 analyze files converted to relative imports
- `src/winml/modelkit/optim/pipes/` — 5 files converted to relative imports
- `src/winml/modelkit/models/auto.py` — removed `_get_model_config()`; task class mapping comments added
- `tests/unit/optim/` — 7 test files updated for import and Gelu test fix
