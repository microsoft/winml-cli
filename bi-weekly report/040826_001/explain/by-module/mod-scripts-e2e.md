# Module: scripts/e2e_eval
**Path**: `scripts/e2e_eval/`, `.pipelines/`
**Period**: 2026-03-23 to 2026-04-08

## 1. Module Overview
The `scripts/e2e_eval/` directory contains the end-to-end evaluation pipeline for ModelKit, covering performance and accuracy evaluation of WinML models against PyTorch baselines. The `.pipelines/` directory holds the Azure DevOps pipeline YAML.

## 2. Files Changed This Period
| File | PRs | Summary |
|------|-----|---------|
| `run_eval.py` | #25, #48, #190, #205, #242 | Initial implementation (#25); ADO pipeline integration (#48); feature extraction dispatch (#190); winml rename (#205); QNN NPU support and continue flag (#242) |
| `run_pytorch_baseline.py` | #25, #190, #198, #205 | Initial implementation; feature extraction support (#190); module path fix (#198); winml rename (#205) |
| `run_sa_eval.py` | #222 | New SA evaluation orchestrator (+719 lines) |
| `sa_comparison.py` | #222 | New SA result comparison tool (+312 lines) |
| `sa_report.py` | #222 | New SA report generator (+613 lines) |
| `build_registry.py` | #25, #227 | Initial implementation; license header (#227) |
| `analyze_results.py` | #25, #227 | Initial implementation; license header (#227) |
| `find_failures.py` | #25, #227 | Initial implementation; license header (#227) |
| `generate_report.py` | #25, #227 | Initial implementation; license header (#227) |
| `utils/accuracy.py` | #25, #190, #205 | Initial implementation; feature extraction accuracy (#190); WMK→WinML header (#205) |
| `utils/reporter.py` | #25, #227 | Initial implementation; lint fixes (#227) |
| `utils/classifier.py`, `registry.py`, `dataset_config.py` | #25, #227 | Initial implementation; license headers (#227) |
| `testsets/models_all.json` | #25 | Full model test set (2,162 entries) |
| `testsets/models_with_acc.json` | #25, #190, #205 | Accuracy model set; feature extraction entries (#190); metric key rename (#205) |
| `testsets/models_P0.json` | #25 | P0 priority model set |
| `datasets/` | #25, #205, #227 | 5 dataset builder scripts; wmk→winml rename; license fixes |
| `setup_ado_agent.ps1` | #48 | New ADO agent setup script |
| `.pipelines/Modelkit E2E Test.yml` | #27, #48, #242 | Initial stub (#27); full ADO pipeline with 5 stages (#48); QNN NPU stages (#242) |
| `cache/baseline_cache.json` | #25, #190 | Initial baseline cache; feature extraction entries (#190) |

## 3. Net Change Summary
- The entire `scripts/e2e_eval/` framework was created from scratch in PRs #25 and #27, providing infrastructure for running WinML model evaluation at scale.
- PR #48 built out the Azure DevOps pipeline with 5 parallel evaluation stages, a clean-cache flag, and an ADO agent setup script.
- PR #222 added a parallel static analysis (SA) evaluation path with `run_sa_eval.py`, `sa_comparison.py`, and `sa_report.py` for comparing SA results across code versions.
- PR #190 extended `run_eval.py` and `run_pytorch_baseline.py` to support feature extraction task evaluation.
- PR #242 added QNN NPU-specific pipeline stages, artifact publishing, and the `--continue` flag for interrupted model list processing.
- PR #205 renamed all `wmk` references to `winml` across all scripts and JSON config files.

## 4. New APIs/Functions Added
| Symbol | Description |
|--------|-------------|
| `run_eval.py` | Main E2E evaluation runner for WinML models |
| `run_pytorch_baseline.py` | PyTorch baseline evaluation runner |
| `run_sa_eval.py` | Static analysis evaluation orchestrator (#222) |
| `sa_comparison.py` | SA result diff tool (#222) |
| `sa_report.py` | SA report generator (#222) |
| `build_registry.py` | Model registry builder |
| `analyze_results.py` | Result analysis utility |
| `utils/accuracy.py` | Accuracy computation utilities |
| `utils/reporter.py` | Structured result reporter |
