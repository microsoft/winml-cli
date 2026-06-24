# E2E Evaluation Scripts

Batch-evaluate WinML CLI's `winml perf` pipeline against a curated set of HuggingFace models.
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
| `--priority` | `P0 P1 P2` | Filter: one or more of `P0`, `P1`, `P2`, `P3` (e.g. `--priority P0 P1`). Pass `P3` explicitly to include P3 models. |
| `--model-type` | — | Filter by model_type (e.g. `bert`) |
| `--group` | — | Filter by group (e.g. `Foundry Toolkit`) |
| `--device` | `auto` | Target device |
| `--timeout` | 600 | Per-model timeout (seconds) |
| `--list` | off | List models and exit |
| `--verbose` | off | Print stderr for failed models |
| `--continue` | off | Skip models with existing results |
| `--retry-failed [TYPE ...]` | — | Re-run failed models (implies `--continue`) |
| `--build-only` | off | Build with `--no-compile`, writing each stage's ONNX (no EP needed). Loops the EP matrix when `--ep`/`--device` omitted |

#### `--build-only` — Generate per-stage models (no EP required)

`--build-only` runs config + build with `--no-compile`, writing each stage's ONNX —
`export.onnx`, `optimized.onnx`, `quantized.onnx`. Because compile is skipped, this
needs **no execution-provider hardware** and runs on any CPU machine. Perf and accuracy
phases are skipped.

When `--ep`/`--device` are **omitted**, every model is built once per EP in the
build-only matrix, each into a `<ep>_<device>/` subdir:

| Label | EP | Device |
|---|---|---|
| `qnn_npu` | qnn | npu |
| `qnn_gpu` | qnn | gpu |
| `ov_cpu` | openvino | cpu |
| `ov_npu` | openvino | npu |
| `ov_gpu` | openvino | gpu |
| `mlas_cpu` | cpu (MLAS) | cpu |
| `dml_gpu` | dml | gpu |
| `vitisai_npu` | vitisai | npu |

Precision per combo follows the eval policy: NPU defaults to `w8a16`, CPU/GPU omit the
flag (winml auto), and native-quant EPs (VitisAI) are built unquantized (`--no-quant`).
When `--ep` or `--device` is pinned, a single build is written directly into
`<output-dir>/models/<slug>/`.

```bash
# Build all EP-matrix variants for P0 models (8 builds per model)
uv run python scripts/e2e_eval/run_eval.py --build-only --priority P0

# Pin a single EP/device (no matrix; writes directly to model dir)
uv run python scripts/e2e_eval/run_eval.py --build-only --hf-model microsoft/resnet-50 --ep qnn --device npu
```

Composite models (multiple sub-components) are built into per-component subdirectories
under each EP subdir.

**Export dedup** (without `--upload`): the `export.onnx` stage is EP/device-independent,
so it is identical across all matrix combos. It is stored once under
`<model_dir>/_shared/export.onnx` and removed from each `<ep>_<device>/` subdir,
keeping only one copy on disk. With `--upload` each combo is published and deleted on
its own, so there is nothing to share and dedup is skipped.

#### Streaming upload to the Azure Artifacts feed (`--upload`)

Running the full matrix over many models fills the local disk fast. `--upload`
publishes each **EP/device combo** to the **`Modelkit`** Azure Artifacts feed
(Universal Package) as soon as it is built, then deletes that combo's local copy —
so peak disk stays at roughly one combo, and a large/slow upload of one combo can't
fill the disk.

- **Auth**: uses `az login` (Entra ID) — no PAT. The script verifies the
  `azure-devops` az extension is installed (auto-adds it) and that you're logged in;
  if not, it aborts (so disk isn't silently filled).
- **Package**: one package `winml-cli-models`, **one version per combo**, named
  `0.0.0-<run-stamp>-<ep>-<device>-<model-slug>` where the run-stamp is a date
  (default today, `YYYYMMDD`). e.g.
  `0.0.0-20260609-qnn-npu-microsoft-resnet-50-image-classification` (the `0.0.0-`
  core keeps it valid SemVer 2.0; the rest is the pre-release segment). Uploading
  per combo keeps each package small, which lowers the per-upload timeout risk and
  lets a single combo be retried on its own.
- **Disk is always bounded**: each combo's local dir is deleted after *every*
  outcome — uploaded, version-exists, upload-failed, **timed-out**, or build-failed
  — unless `--keep-local`. A failed or timed-out combo is recorded and the run
  continues; a host-level az failure (not logged in / token expired) aborts so you
  can re-auth and resume.
- A `build_only_results.json` log (combo version → build status + upload status +
  error tail + timestamps) is written in the output dir for *every* run (with or
  without `--upload`), so you can audit which combos succeeded, failed, or timed
  out. It also drives `--continue` (skips combos already in the feed).

```bash
# Build the matrix and stream each model to the feed, deleting locals
uv run python scripts/e2e_eval/run_eval.py --build-only --upload --priority P0

# Resume an interrupted batch: same run-stamp + --continue skips combos already
# uploaded (per the results log / feed) without rebuilding them.
uv run python scripts/e2e_eval/run_eval.py --build-only --upload --continue \
  --run-stamp 20260609 --priority P0

# --upload-skip-existing: if the feed already has a version (e.g. results log lost),
# treat the publish conflict as done and delete the local copy.
uv run python scripts/e2e_eval/run_eval.py --build-only --upload --upload-skip-existing

# Upload but keep local copies (debug)
uv run python scripts/e2e_eval/run_eval.py --build-only --upload --keep-local
```

Download a specific model's specific file later with `--file-filter`:

```bash
az artifacts universal download \
  --organization https://dev.azure.com/microsoft --project windows.ai.toolkit \
  --scope project --feed Modelkit --name winml-cli-models \
  --version 0.0.0-20260609-qnn-npu-microsoft-resnet-50-image-classification \
  --path ./out --file-filter 'quantized.onnx'
```

| Upload flag | Default | Description |
|---|---|---|
| `--upload` | off | Publish each EP/device combo to the feed, then delete it locally |
| `--run-stamp` | today (`YYYYMMDD`) | Version prefix; pass the same stamp + `--continue` to resume |
| `--continue` | off | Skip combos already uploaded for this run-stamp (no rebuild) |
| `--feed` | `Modelkit` | Azure Artifacts feed name |
| `--feed-org` | `https://dev.azure.com/microsoft` | Azure DevOps org URL |
| `--feed-project` | `windows.ai.toolkit` | Project for the project-scoped feed |
| `--package-name` | `winml-cli-models` | Universal Package name |
| `--keep-local` | off | Upload but do not delete local combos (also keeps build-failed combos) |
| `--upload-skip-existing` | off | Treat an existing feed version as done (feed-based resume) |

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
| **P1** | Important — ISV models, failures are critical |
| **P2** | Important — tracked closely, regressions flagged |
| **P3** | Extended coverage — best-effort |

Groups (`Foundry Toolkit`, `Benchmark`, `ISV`, `microsoft`, `Top200`, …) categorize models by source/purpose.

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
