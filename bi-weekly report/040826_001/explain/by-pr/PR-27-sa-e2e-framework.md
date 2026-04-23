# PR-27: SA E2E Eval Framework (#221 + #222)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commits | `199d3c5` (#221, Qiong Wu, 2026-04-02), `b8d97c8` (#222, Qiong Wu, 2026-04-02) |
| Author | Qiong Wu (DingmaomaoBJTU) |
| PR Numbers | #221, #222 |
| Files Changed | 8 total |
| Insertions | +1,726 |
| Deletions | -10 |

## Summary
PR #221 fixed a one-line bug in `analyze/core/runtime_checker.py` and added 70 lines of unit tests for that fix to `test_runtime_checker.py`. PR #222 introduced a full static analysis (SA) E2E evaluation framework in `scripts/e2e_eval/`: `run_sa_eval.py` (719 lines) orchestrates running the static analyzer across a model list and collecting support results; `sa_comparison.py` (312 lines) diffs two SA result sets to surface regressions; `sa_report.py` (613 lines) generates structured reports from SA evaluation results. Also removed `.gitignore` exclusion for `/docs/` and updated `hub_models.json` and `inspect/formatter.py`.

## Files Changed
- `scripts/e2e_eval/run_sa_eval.py` — new SA eval orchestrator (+719)
- `scripts/e2e_eval/sa_comparison.py` — SA result comparison tool (+312)
- `scripts/e2e_eval/sa_report.py` — SA report generator (+613)
- `src/winml/modelkit/analyze/core/runtime_checker.py` — one-line bug fix (#221)
- `tests/unit/analyze/core/test_runtime_checker.py` — 70 new test lines (#221)
- `src/winml/modelkit/inspect/formatter.py` — minor update (#222)
