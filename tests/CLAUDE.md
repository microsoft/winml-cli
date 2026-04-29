# Test Convention

Inherits all rules from [`/CLAUDE.md`](/CLAUDE.md). Additional test-specific rules below.

Reference: [`/docs/pytest-best-practices.md`](/docs/pytest-best-practices.md)

## Always

- Place unit tests under `tests/unit/<module>/` mirroring `src/winml/modelkit/<module>/`
- Place integration tests under `tests/integration/`, e2e under `tests/e2e/`
- Place cross-cutting CLI-surface tests (startup, import budget, arg parsing,
  command discovery, version/help output) under `tests/cli/` — these don't
  mirror any single `src/` module, so they don't fit under `tests/unit/<module>/`
- Put shared fixtures in the narrowest `conftest.py` that covers all consumers

## Never

- Place tests that mirror a `src/winml/modelkit/<module>/` directly under
  `tests/` — they belong under `tests/unit/<module>/`. The category dirs
  listed under "Always" (`tests/unit/`, `tests/integration/`, `tests/e2e/`,
  `tests/cli/`) are the only legitimate top-level test directories
- Put `test_*.py` files in `assets/`, `fixtures/`, or `mock_data/` — those are helpers only
- Duplicate fixtures across multiple `conftest.py` files
- Add a new top-level category under `tests/` without also adding it to the
  `.github/workflows/modelkit-ci.yml` path matrix — CI enumerates paths
  explicitly, so a new directory is invisible to CI until it's listed
