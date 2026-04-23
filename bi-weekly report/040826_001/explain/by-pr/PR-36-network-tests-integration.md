# PR-36: Move Remaining Network-Dependent Tests to Integration (#254)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `7512c96` |
| Date | 2026-04-07 |
| Author | Zhipeng Wang (timenick) |
| PR Number | #254 (closes #253) |
| Files Changed | 3 |
| Insertions | +251 |
| Deletions | -246 |

## Summary
Follow-up to PR #95, catching tests that were missed in the initial integration/unit separation. Moved `test_text_classification.py` from `tests/unit/datasets/` to `tests/integration/datasets/` (every test calls `load_dataset` downloading GLUE/MRPC from the internet). Also moved `TestResolveTaskAndModelClass` and `TestConflictScenarios` from `tests/unit/loader/test_hf_model_class_mapping.py` to `tests/integration/loader/test_hf_model_class_mapping.py` (13 tests calling `AutoConfig.from_pretrained`).

## Files Changed
- `tests/integration/datasets/test_text_classification.py` — moved from unit (0 net change, file relocated)
- `tests/integration/loader/test_hf_model_class_mapping.py` — new integration file with 13 network-dependent tests (+251)
- `tests/unit/loader/test_hf_model_class_mapping.py` — network-dependent tests removed (-246)
