# PR-02: Add CI Workflows for ModelKit (#14)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `97df6de` |
| Date | 2026-03-30 |
| Author | Zhipeng Wang (timenick) |
| PR Number | #14 |
| Files Changed | 5 |
| Insertions | +155 |
| Deletions | -23 |

## Summary
Introduced three GitHub Actions CI workflows: unit tests (`modelkit-ci.yml`) running on Windows with uv and Python 3.10, ruff linting (`lint.yml`), and CodeQL security scanning (`codeql.yml`). The test workflow was split into 5 parallel matrix jobs (unit, optim, models, commands, remaining) after sequential runs timed out at 30 minutes across 152 test files. Accompanying test fixes in `test_winml_session.py` and `conftest.py` guard against WinML SDK initialization hangs on CI runners without the SDK installed.

## Files Changed
- `.github/workflows/codeql.yml` — new CodeQL security analysis workflow
- `.github/workflows/lint.yml` — new ruff linting workflow
- `.github/workflows/modelkit-ci.yml` — new parallel unit test workflow
- `tests/session/conftest.py` — EP availability guard for CI environments
- `tests/session/test_winml_session.py` — fixed EP inference and registry tests for CI
