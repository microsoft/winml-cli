# PR-15: Move Misplaced Integration/E2E Tests Out of tests/unit/ (#95)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `17b40ab` |
| Date | 2026-03-31 |
| Author | Zhipeng Wang (timenick) |
| PR Number | #95 |
| Files Changed | 25 |
| Insertions | +844 |
| Deletions | -690 |

## Summary
Moved tests that require network access, model downloads, or specific hardware (NPU/GPU) from `tests/unit/` to `tests/integration/` or `tests/e2e/`. Specifically: loader tests calling `AutoConfig.from_pretrained` moved to `tests/integration/loader/`; config/build integration tests moved to `tests/integration/config/`; dataset tests moved to `tests/integration/datasets/`; optim pipe fusion integration tests moved to `tests/integration/optim/`; EP monitor and session e2e tests moved to `tests/e2e/`. Renamed misleading test file names (`test_module_build_e2e.py` → `test_module_build.py`, `test_quantization_e2e.py` → `test_quantization.py`).

## Files Changed (key)
- `tests/e2e/` — new conftest.py, test_ep_monitor.py, test_session.py
- `tests/integration/config/test_build.py` — 218 lines of integration tests
- `tests/integration/loader/` — test_detect_task_and_class.py, test_hf_model_class_mapping.py, test_load_hf_model.py
- `tests/integration/optim/test_pipe_fusion.py` — fusion integration tests
- `tests/unit/` — removed network-dependent tests from unit directories
