# Tests Directory Convention

This file defines the test directory organization rules for ModelKit. For general pytest best practices, see [`/docs/pytest-best-practices.md`](/docs/pytest-best-practices.md).

## Directory Structure

Tests are organized by **test type** at the top level, with **source module mirroring** inside each type.

```
tests/
├── conftest.py                 # Global fixtures (WinML EP mock, etc.)
├── unit/                       # Unit tests — mirror source structure inside
│   ├── analyze/
│   │   ├── core/
│   │   ├── models/
│   │   └── pattern/
│   ├── build/
│   ├── cache/
│   ├── commands/
│   ├── compiler/
│   ├── config/
│   ├── core/
│   ├── datasets/
│   ├── eval/
│   ├── export/
│   ├── inspect/
│   ├── loader/
│   ├── models/
│   ├── onnx/
│   ├── optim/
│   ├── optracing/
│   ├── quant/
│   ├── session/
│   ├── sysinfo/
│   └── utils/
├── integration/                # Integration tests — cross-module workflows
│   └── analyze/
├── e2e/                        # End-to-end tests — full CLI/pipeline
├── regression/                 # Regression tests — fixed bugs
├── assets/                     # Shared test data generators
├── fixtures/                   # Shared test model builders
└── mock_data/                  # Static mock data (JSON, etc.)
```

## Rules

### 1. Test Type Determines Top-Level Directory

| Directory | What goes here | Runs in CI | Needs hardware |
|-----------|---------------|------------|----------------|
| `unit/` | Single-module tests, no external deps | Always | No |
| `integration/` | Cross-module, may need ORT session | Gated | Sometimes |
| `e2e/` | Full CLI pipelines, real models | Nightly | Yes |
| `regression/` | Specific bug reproductions | Always | No |

### 2. Mirror Source Structure Inside `unit/`

Every source module under `src/winml/modelkit/<module>/` should have a corresponding test directory at `tests/unit/<module>/`. The directory name **must match exactly**.

```
src/winml/modelkit/analyze/    →  tests/unit/analyze/
src/winml/modelkit/build/      →  tests/unit/build/
src/winml/modelkit/datasets/   →  tests/unit/datasets/    (NOT dataset_tests/)
src/winml/modelkit/optim/      →  tests/unit/optim/
```

### 3. No Flat Module Directories at `tests/` Root

Do **not** create module-specific directories directly under `tests/`. Module tests belong inside `tests/unit/`, `tests/integration/`, or `tests/e2e/`.

```
# Wrong — flat module dir at tests root
tests/build/test_hf.py
tests/session/test_winml_session.py

# Correct — inside test type directory
tests/unit/build/test_hf.py
tests/unit/session/test_winml_session.py
```

### 4. One conftest.py Per Scope

| Location | Scope | Example fixtures |
|----------|-------|-----------------|
| `tests/conftest.py` | Global | WinML EP mock, tmp path config |
| `tests/unit/<module>/conftest.py` | Module | Module-specific builders, fake models |
| `tests/e2e/conftest.py` | E2E | Real model downloads, session setup |

Do not duplicate fixtures across conftest files. If a fixture is needed in multiple modules, promote it to `tests/conftest.py`.

### 5. Shared Test Infrastructure

| Directory | Purpose |
|-----------|---------|
| `tests/assets/` | Scripts that generate test data (e.g., ONNX model builders) |
| `tests/fixtures/` | Reusable model/graph builders imported by tests |
| `tests/mock_data/` | Static JSON/YAML files used as test input |

These are **not** test directories — they contain helpers only. No `test_*.py` files here.

## Current State vs Target

The following directories exist at `tests/` root but should be migrated into `tests/unit/`:

| Current Location | Target Location |
|---|---|
| `tests/build/` | `tests/unit/build/` |
| `tests/cache/` | `tests/unit/cache/` |
| `tests/commands/` | `tests/unit/commands/` |
| `tests/compiler/` | `tests/unit/compiler/` |
| `tests/config/` | `tests/unit/config/` |
| `tests/core/` | `tests/unit/core/` |
| `tests/dataset_tests/` | `tests/unit/datasets/` |
| `tests/eval/` | `tests/unit/eval/` |
| `tests/export/` | `tests/unit/export/` |
| `tests/inspect/` | `tests/unit/inspect/` |
| `tests/loader/` | `tests/unit/loader/` |
| `tests/models/` | `tests/unit/models/` |
| `tests/onnx/` | `tests/unit/onnx/` |
| `tests/optim/` | `tests/unit/optim/` |
| `tests/optracing/` | `tests/unit/optracing/` |
| `tests/session/` | `tests/unit/session/` |
| `tests/sysinfo/` | `tests/unit/sysinfo/` |
| `tests/utils/` | `tests/unit/utils/` |

Additionally, merge duplicated locations:

| Duplicated | Merge into |
|---|---|
| `tests/sysinfo/` + `tests/unit/sysinfo/` | `tests/unit/sysinfo/` |
| `tests/onnx/` + `tests/unit/onnx/` | `tests/unit/onnx/` |

Root-level test files (`test_cli.py`, `test_text_classification.py`) should move to appropriate `tests/unit/` subdirectories.

## Migration Notes

- Move directories incrementally — one module at a time
- After each move, run `uv run pytest tests/` to verify nothing breaks
- Update any hardcoded test paths in conftest.py or CI configs
- Merge conftest.py files when source and target directories overlap
