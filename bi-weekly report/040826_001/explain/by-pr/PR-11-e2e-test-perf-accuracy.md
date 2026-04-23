# PR-11: Add E2E Test for Perf and Accuracy (#25)

## Commit Metadata
| Field | Value |
|-------|-------|
| Commit Hash | `18ba5d4` |
| Date | 2026-03-31 |
| Author | Qiong Wu (DingmaomaoBJTU) |
| PR Number | #25 |
| Files Changed | 30 |
| Insertions | +9,296 |
| Deletions | -1 |

## Summary
Initial implementation of the `scripts/e2e_eval/` framework for end-to-end performance and accuracy evaluation. Introduces `run_eval.py` (1,214 lines), `run_pytorch_baseline.py`, `build_registry.py`, `analyze_results.py`, `find_failures.py`, and `generate_report.py`. Includes model registry JSON files (`models_all.json` with 2,162 entries, `models_with_acc.json`, `models_P0.json`), dataset build scripts (AI4Privacy, FairFace, IndonLU, PubTables), dataset label mapping JSONs, a baseline cache, and a timeout skip list. Utility modules cover accuracy computation, classification, dataset config, registry management, and reporting.

## Files Changed (key)
- `scripts/e2e_eval/run_eval.py` — main evaluation runner (+1,214)
- `scripts/e2e_eval/run_pytorch_baseline.py` — PyTorch baseline runner (+271)
- `scripts/e2e_eval/build_registry.py` — model registry builder (+451)
- `scripts/e2e_eval/utils/reporter.py` — result reporter (+710)
- `scripts/e2e_eval/testsets/models_all.json` — full model test set (+2,162 entries)
- `scripts/e2e_eval/datasets/` — 5 dataset builder scripts
- `scripts/e2e_eval/utils/` — accuracy.py, classifier.py, registry.py, dataset_config.py
