# PR-13: Standardize Naming Conventions and Reorganize Test Directory Structure (#28)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `e120e67` |
| Date | 2026-03-31 |
| Author | Zhipeng Wang (timenick) |
| PR Number | #28 |
| Files Changed | 170 |
| Insertions | +4,479 |
| Deletions | -1,593 |

## Summary
Two major refactors in one PR. First, standardized acronym casing across 9 class names and 18 source files: `OnnxOP` became `ONNXOp`, `Ep*` became `EP*`, `Qdq*` became `QDQ*`, `OP*` became `Op*`, `OnnxConfigNotFoundError` became `ONNXConfigNotFoundError`, and `OnnxModelOutput` became `ONNXModelOutput`. Second, reorganized the test directory structure so all module tests live under `tests/unit/` following a test-type-first layout (unit/integration/e2e/regression). Flat test directories at `tests/` root were moved into `tests/unit/`; duplicate directories (`tests/onnx/` + `tests/unit/onnx/`, `tests/sysinfo/`) were merged; `tests/dataset_tests/` was renamed to `tests/unit/datasets/`. Also added `docs/naming-convention.md`, `tests/CLAUDE.md`, and `docs/pytest-best-practices.md`.

## Files Changed (key)
- `src/winml/modelkit/` — 18 source files updated for class renames (analyze, export, config, pattern, quant, sysinfo)
- `tests/unit/` — ~150 test files moved from `tests/` root into the unit/ subtree
- `docs/naming-convention.md` — new naming convention guide
- `tests/CLAUDE.md` — new test-specific rules
- `.github/workflows/modelkit-ci.yml` — test group paths updated
