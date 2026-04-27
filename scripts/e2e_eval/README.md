# E2E Evaluation Scripts

Batch-evaluate ModelKit's `winml perf` pipeline against a curated set of HuggingFace models.
Captures pass/fail, failure classification, and generates interactive reports.

## Quick Start

```bash
# 1. Build model registry from HuggingFace Hub
uv run python scripts/e2e_eval/build_registry.py

# 2. Run evaluation (P0 models only)
uv run python scripts/e2e_eval/run_eval.py --priority P0

# 3. Regenerate reports from cached results
uv run python scripts/e2e_eval/generate_report.py --input-dir eval_results/2026-02-25/
```

## Scripts

### `build_registry.py` — Generate Model Registry

Queries HuggingFace Hub for popular models per task, enriches with `model_type`,
assigns priority (P0/P1), and writes `testsets/models_all.json`.

```bash
# Default: top 10 per task, Optimum-supported models prioritized
uv run python scripts/e2e_eval/build_registry.py

# Custom top-N
uv run python scripts/e2e_eval/build_registry.py --top-n 20 --output models_large.json

# View registry stats without rebuilding
uv run python scripts/e2e_eval/build_registry.py --stats

# Dry run (preview without writing)
uv run python scripts/e2e_eval/build_registry.py --dry-run
```

| Argument | Default | Description |
|---|---|---|
| `--top-n` | 10 | Models per task |
| `--output` | `testsets/models_all.json` | Output path |
| `--curated-source` / `-s` | `testsets/models_curated.json` | Curated model list — `group`/`priority` fields applied verbatim |
| `--no-optimum-filter` | off | Disable Optimum-first soft filter |
| `--stats` | off | Print stats and exit |
| `--dry-run` | off | Preview without writing |

### `run_eval.py` — Run Evaluation

Executes `winml perf` for each model in a subprocess, classifies failures, and
generates reports (JSON, Markdown, HTML).

```bash
# Run all models in registry
uv run python scripts/e2e_eval/run_eval.py

# Filter by priority / task / group
uv run python scripts/e2e_eval/run_eval.py --priority P0
uv run python scripts/e2e_eval/run_eval.py --task image-classification
uv run python scripts/e2e_eval/run_eval.py --group "Foundry Toolkit"

# Single ad-hoc model
uv run python scripts/e2e_eval/run_eval.py --hf-model microsoft/resnet-50

# List filtered models without running
uv run python scripts/e2e_eval/run_eval.py --priority P0 --list

# Resume an interrupted run
uv run python scripts/e2e_eval/run_eval.py --continue

# Retry only ENVIRONMENT failures (disk/network issues)
uv run python scripts/e2e_eval/run_eval.py --retry-failed ENVIRONMENT UNKNOWN

# Retry ALL failed models
uv run python scripts/e2e_eval/run_eval.py --retry-failed
```

| Argument | Default | Description |
|---|---|---|
| `--registry` | `testsets/models_all.json` | Model registry file |
| `--hf-model` | — | Single model (overrides registry) |
| `--output-dir` | `eval_results/{date}` | Output directory |
| `--task` | — | Filter by HF task |
| `--priority` | — | Filter: `P0`, `P1`, `P2` |
| `--model-type` | — | Filter by model_type (e.g. `bert`) |
| `--group` | — | Filter by group (e.g. `Foundry Toolkit`) |
| `--device` | `auto` | Target device |
| `--timeout` | 600 | Per-model timeout (seconds) |
| `--list` | off | List models and exit |
| `--verbose` | off | Print stderr for failed models |
| `--continue` | off | Skip models with existing results |
| `--retry-failed [TYPE ...]` | — | Re-run failed models (implies `--continue`) |

### `generate_report.py` — Regenerate Reports

Re-reads cached `result.json` files and regenerates reports using the latest
classifier rules (no re-running needed).

```bash
uv run python scripts/e2e_eval/generate_report.py --input-dir eval_results/2026-02-25/
uv run python scripts/e2e_eval/generate_report.py --input-dir eval_results/2026-02-25/ --format html
```

| Argument | Default | Description |
|---|---|---|
| `--input-dir` | *required* | Directory with `models/*/result.json` |
| `--format` | `all` | `json`, `markdown`, `html`, `text`, or `all` |
| `--registry` | `testsets/models_all.json` | Registry for HTML enrichment |

## Key Concepts

### Priority & Groups

| Priority | Meaning |
|---|---|
| **P0** | Must-pass — core models, failures are critical |
| **P1** | Important — tracked closely, regressions flagged |
| **P2** | Extended coverage — best-effort |

Groups (`Foundry Toolkit`, `Benchmark`, `Top200`, etc.) categorize models by source/purpose.

### Failure Classification

Failures are classified from `stdout + stderr` pattern matching (ordered by pipeline stage):

| Type | Stage |
|---|---|
| `EXPORT_FAIL` | ONNX export |
| `ANALYZER_BLOCK` | Static analyzer |
| `OPT_FAIL` | Graph optimization |
| `COMPILE_FAIL` | Compilation / quantization |
| `RUNTIME_FAIL` | Inference |
| `ENVIRONMENT` | Disk / network (retryable) |
| `TIMEOUT` | Exceeded time limit |
| `UNKNOWN` | No pattern matched |

Classification is **derived on-the-fly** from stored facts — updating classifier
rules automatically reclassifies all historical results.

### Output Structure

```
eval_results/2026-02-25/
├── environment.json           # Python / package versions
├── report_20260225_103000.json
├── report_20260225_103000.txt
├── summary.md
├── perf_report.html           # Interactive HTML report
└── models/
    ├── microsoft__resnet-50__image-classification/
    │   └── result.json        # Per-model result (facts only)
    └── ...
```

## File Layout

```
scripts/e2e_eval/
├── build_registry.py          # Step 1: Generate models_all.json
├── run_eval.py                # Step 2: Run evaluation
├── generate_report.py         # Step 3: Regenerate reports
├── models_viewer.html         # Interactive registry viewer
├── models_viewer_static.html  # Standalone viewer (generated)
├── testsets/
│   ├── models_all.json        # Full model registry (generated)
│   ├── models_with_acc.json   # Models with accuracy dataset configs
│   └── models_curated.json    # Hand-curated models to be registered
├── cache/
│   ├── baseline_cache.json    # Cached PyTorch baseline accuracy results
│   └── timeout_skip_list.json # Models to skip due to known timeouts
└── utils/
    ├── __init__.py            # Shared utilities
    ├── classifier.py          # Failure classification rules
    ├── registry.py            # Registry loading & filtering
    └── reporter.py            # Result building & report generation
```
