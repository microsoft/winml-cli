# Module: tests
**Path**: `tests/`
**Period**: 2026-03-23 to 2026-04-08

## 1. Module Overview
The test suite for ModelKit. This period saw major structural reorganization alongside a large volume of new test coverage for new features and bug fixes.

## 2. Files Changed This Period
| File / Directory | PRs | Summary |
|-----------------|-----|---------|
| `tests/conftest.py` | #24, #252 | Root conftest created with WinML SDK mock (#24); hardware probing mock added (#252) |
| `tests/unit/` | #28, #13–#49 | All unit tests moved under `tests/unit/` from root (#28); imports updated across ~150 files in import cleanup PRs |
| `tests/integration/` | #95, #254 | New integration tier: config, loader, datasets, optim subdirs with network-dependent tests |
| `tests/e2e/` | #95 | New e2e tier: EP monitor, session e2e tests |
| `tests/regression/test_design_gaps.py` | #246 | 6 new regression tests covering bug bash fixes |
| `tests/unit/eval/` | #15, #42 | New eval test files: test_eval.py (857 lines), test_align_labels.py, test_image_segmentation_evaluator.py, test_map_metric.py, test_object_detection_evaluator.py |
| `tests/unit/eval/test_feature_extraction_evaluator.py` | #190 | New test file (+259 lines) |
| `tests/unit/models/auto/test_feature_extraction.py` | #190 | New test file (+139 lines) |
| `tests/unit/commands/test_hub.py` | #196, #205, #246 | New hub command tests (+343 lines); renamed wmk→winml; additional bug bash coverage |
| `tests/unit/analyze/test_check_ops.py` | #23, #235 | New file (+146 lines) for normalization and op_utils tests |
| `tests/unit/analyze/test_input_generators.py` | #22 | New file (+61 lines) for input generator coverage |
| `tests/unit/session/test_ep_monitor.py` | #95, #46 | Moved from tests/unit/ root to session/; imports updated |
| `tests/unit/sysinfo/test_device.py` | #28, #188 | Moved to unit/sysinfo/; NvTensorRTRTX assertion added |
| `tests/session/conftest.py` | #14, #18, #24 | Mock fixtures for WinML SDK init (moved to root in #24) |
| `tests/CLAUDE.md` | #28 | New file: concise test rules (always/never) |

## 3. Net Change Summary
- **Directory restructure (PR #28, #95, #254)**: All test files previously at the `tests/` root level were moved into `tests/unit/`; network-dependent tests moved to `tests/integration/`; hardware-dependent tests moved to `tests/e2e/`. This enforces a test-type-first organization.
- **Import hygiene (PRs #39–#49)**: ~150 test files converted from internal submodule imports to package-level imports following the policy codified in CLAUDE.md.
- **CI stability (PRs #14, #18, #24, #252)**: A layered mocking strategy was established to prevent WinML SDK and hardware probing initialization from hanging CI runners: session-level mock in #18, promoted to root conftest in #24, hardware probing mock in #252.
- **New test coverage**: Feature extraction evaluator (259 lines), hub command (343 lines), check_ops normalization (146 lines), input generators (61 lines), 6 regression tests for bug bash fixes, and large test files for the eval framework (857 lines for test_eval.py).

## 4. New Test Files Added
| File | Lines | Description |
|------|-------|-------------|
| `tests/conftest.py` | 24 | Root-level WinML mock + hardware probing mock |
| `tests/unit/eval/test_eval.py` | 857 | Evaluator framework tests |
| `tests/unit/eval/test_feature_extraction_evaluator.py` | 259 | Feature extraction evaluator tests |
| `tests/unit/models/auto/test_feature_extraction.py` | 139 | WinML feature extraction model tests |
| `tests/unit/commands/test_hub.py` | 343 | Hub command tests |
| `tests/unit/analyze/test_check_ops.py` | 146 | check_ops normalization and op_utils tests |
| `tests/unit/analyze/test_input_generators.py` | 61 | Input generator p1 coverage tests |
| `tests/regression/test_design_gaps.py` | 46+ | Regression tests for 6 bug bash fixes |
| `tests/integration/` | ~600 | New integration tier: config, loader, datasets, optim |
| `tests/e2e/` | ~237 | New e2e tier: EP monitor, session tests |
