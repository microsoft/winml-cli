# Contributing

This guide covers the development workflow for contributing to winml-cli.

---

## Prerequisites

| Component | Version |
|-----------|---------|
| Python | 3.11 (`requires-python = ">=3.11,<3.12"`) |
| Package manager | [uv](https://github.com/astral-sh/uv) |
| OS | Windows 11 (primary target) |

---

## Development Setup

```bash
git clone https://github.com/microsoft/winml-cli.git
cd winml-cli

# Install all dependencies including dev tools
uv sync --extra dev

# Enable pre-commit hooks
uv run pre-commit install
```

The pre-commit hooks automatically enforce:

- MIT license headers on all `.py` files
- Trailing whitespace removal
- End-of-file newline
- YAML syntax validation
- Ruff linting and formatting

---

## Running Tests

```bash
# All unit tests
uv run pytest tests/

# Fast CI-like run (excludes hardware-dependent tests)
uv run pytest tests/ -m "not e2e and not npu and not gpu"

# Specific module
uv run pytest tests/unit/analyze
uv run pytest tests/unit/commands

# With coverage
uv run pytest tests/ --cov=src/winml/modelkit --cov-report=html
```

**Test markers:**

| Marker | Use |
|--------|-----|
| `@pytest.mark.unit` | Fast unit tests (default) |
| `@pytest.mark.smoke` | Critical-path tests that must always pass |
| `@pytest.mark.e2e` | End-to-end tests (slow, may need hardware) |
| `@pytest.mark.npu` | Requires NPU hardware |
| `@pytest.mark.gpu` | Requires GPU |
| `@pytest.mark.slow` | Tests taking > 30 seconds |

---

## Linting and Type Checking

```bash
# Lint (check only)
uv run ruff check src/ tests/

# Lint and auto-fix
uv run ruff check src/ tests/ --fix

# Format
uv run ruff format src/ tests/

# Type check
uv run mypy src/

# Run all pre-commit hooks manually
uv run pre-commit run --all-files
```

---

## Code Structure

```text
src/winml/modelkit/
├── cli.py              # Entry point (winml command group)
├── commands/           # CLI subcommands (export, build, analyze, etc.)
├── models/             # Model loading from HuggingFace / local
├── export/             # ONNX export logic and HTP
├── optimize/           # Optimization pipelines and fusion
├── analyze/            # Analysis engine and runtime rules
├── config/             # Build config schema and constants
├── build/              # Pipeline orchestration
├── compiler/           # EP compilation (EPContext)
├── quant/              # Quantization
├── eval/               # Evaluation metrics
├── serve/              # FastAPI serving layer
├── session/            # Session management
├── core/               # Core graph abstractions
├── cache/              # Caching utilities
└── utils/              # Shared utilities

tests/
├── unit/               # Unit tests (organized by module)
├── integration/        # Integration tests
├── e2e/                # End-to-end tests
├── regression/         # Regression suite
├── fixtures/           # Test data and mock models
└── conftest.py         # Shared fixtures
```

---

## Coding Conventions

- **Line length:** 100 characters
- **Docstrings:** Google style
- **Strings:** Double quotes (enforced by Ruff)
- **Type annotations:** Required for public API functions
- **License header:** Auto-inserted by pre-commit on all `.py` files

**Import order** (enforced by Ruff isort):

1. `__future__`
2. Standard library
3. Third-party (`torch`, `transformers`, `onnx`, etc.)
4. First-party (`winml.*`)
5. Relative imports

See the internal naming convention guide for ONNX/EP/QDQ term casing rules.

---

## PR Checklist

Before submitting a pull request:

- [ ] Tests pass: `uv run pytest tests/ -m "not e2e and not npu and not gpu"`
- [ ] Linting passes: `uv run ruff check src/ tests/`
- [ ] Formatting is clean: `uv run ruff format --check src/ tests/`
- [ ] Type checking passes: `uv run mypy src/`
- [ ] New code includes unit tests (target 80%+ coverage)
- [ ] Docs updated if public API changed

**CI will run:**

1. **Lint workflow** — license headers + Ruff
2. **Test workflow** — parallelized test groups on Windows
3. **CLA bot** — Contributor License Agreement signature

---

## Documentation Development

```bash
# Live preview (auto-reloads)
uv run mkdocs serve

# Validate (strict mode, catches broken links)
uv run mkdocs build --strict
```

See [docs/README.md](https://github.com/microsoft/winml-cli/blob/main/docs/README.md)
for authoring conventions, publishing workflow, and site structure.

---

## See also

- [Installation](getting-started/installation.md) — user-facing setup
- [Commands](commands/overview.md) — CLI reference
