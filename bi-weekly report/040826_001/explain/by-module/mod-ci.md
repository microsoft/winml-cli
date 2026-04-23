# Module: CI / Build Configuration
**Path**: `.github/workflows/`, `pyproject.toml`
**Period**: 2026-03-23 to 2026-04-08

## 1. Module Overview
CI workflows and project build configuration. This covers GitHub Actions workflows for test, lint, and security scanning, and the `pyproject.toml` project metadata.

## 2. Files Changed This Period
| File | PRs | Summary |
|------|-----|---------|
| `.github/workflows/modelkit-ci.yml` | #14, #28, #198 | New CI workflow with 5 parallel test groups; test paths updated after directory restructure |
| `.github/workflows/lint.yml` | #14, #227 | New lint workflow; license header check added |
| `.github/workflows/codeql.yml` | #14 | New CodeQL security analysis workflow |
| `pyproject.toml` | #15, #196, #205, #213, #227 | batch update; hub command entry; wmk→winml rename; download script dep; license check config |

## 3. Net Change Summary
- Three CI workflows were introduced from scratch in PR #14: parallel unit tests on Windows (5 groups), ruff linting, and CodeQL security analysis.
- PR #227 added a license header check step to the lint workflow and fixed all pre-existing license violations in `scripts/e2e_eval/`.
- Test group paths in `modelkit-ci.yml` were updated in PR #28 to match the reorganized `tests/unit/` directory structure.
- `pyproject.toml` was updated across multiple PRs: hub command entry (#196), CLI rename to `winml` (#205), download script dependency (#213), license check tooling (#227).

## 4. New APIs/Functions Added
| Symbol | Description |
|--------|-------------|
| `.github/workflows/modelkit-ci.yml` | New parallel test CI workflow |
| `.github/workflows/lint.yml` | New lint CI workflow with license header check |
| `.github/workflows/codeql.yml` | New CodeQL security analysis workflow |
