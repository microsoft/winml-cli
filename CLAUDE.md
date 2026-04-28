# CLAUDE.md

## Cardinal Rules

### 1. No Hardcoded Logic

Never hardcode model architecture names, node/operator names, input/output tensor names, layer naming patterns, or any model-specific logic. All solutions must be universal and architecture-agnostic.

### 2. Pytest Only

All testing uses pytest with code-generated results. Never create standalone test scripts, use LLM-generated expectations, or generate test data manually.

### 3. Mandatory Test Verification

Run `uv run pytest tests/` after every implementation or test revision. Never assume tests pass without verification.

### 4. Never Skip Failing Tests

Investigate root cause and fix the underlying issue. Never use `pytest.mark.skip` or `xfail` to hide failures. Skips are only acceptable for hardware/EP requirements (CUDA, Dml, AVX).

## Development Commands

- **Python**: Always use `uv run` or activate venv first. Never run bare python commands.
- **Temp files**: Use `temp/` folder in project root.
- **Node.js**: Available via fnm. Use `eval "$(fnm env)"` before npm/npx commands.

## Import Rules

### Source code (`src/`)

- **Relative imports** through `__init__.py`: `from ..onnx import ONNXDomain` (not `from winml.modelkit.onnx.domains import ONNXDomain`)
- **Within the same package**, sibling-relative is fine: `from .utils import EXTERNAL_DATA_THRESHOLD`
- Never use absolute `from winml.modelkit.*` paths in source code

### Test code (`tests/`)

- **Absolute imports** from package level: `from winml.modelkit.onnx import ONNXDomain`
- Never reach into internal submodules for symbols exported by `__init__.py`
- **Exception**: private `_`-prefixed functions may use internal imports for testing implementation details

### `__init__.py` public API

- Every symbol that external code needs must be exported in the package's `__init__.py`
- When adding new public classes/functions, add them to both the import and `__all__`
- Private (`_`-prefixed) symbols are never added to `__init__.py`

## Code Quality

- Run `uv run ruff check --fix` after revising Python code
- Follow naming rules in [`/docs/naming-convention.md`](/docs/naming-convention.md) (ONNX, EP, QDQ, Op acronym casing)
- Always ask clarifying questions before planning if requirements are ambiguous
- Critically evaluate proposals — challenge design decisions when warranted

## Git

- Never add `Co-Authored-By` when doing git commit
- Do not include "Test plan" section in PR descriptions
