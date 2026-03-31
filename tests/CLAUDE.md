# Test Convention

Inherits all rules from [`/CLAUDE.md`](/CLAUDE.md). Additional test-specific rules below.

Reference: [`/docs/pytest-best-practices.md`](/docs/pytest-best-practices.md)

## Always

- Place unit tests under `tests/unit/<module>/` mirroring `src/winml/modelkit/<module>/`
- Place integration tests under `tests/integration/`, e2e under `tests/e2e/`
- Put shared fixtures in the narrowest `conftest.py` that covers all consumers

## Never

- Create module directories directly under `tests/` — use `tests/unit/<module>/` instead
- Put `test_*.py` files in `assets/`, `fixtures/`, or `mock_data/` — those are helpers only
- Duplicate fixtures across multiple `conftest.py` files
