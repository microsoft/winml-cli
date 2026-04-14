---
name: wmk-eval
description: "Investigate model evaluation accuracy issues, develop new evaluators, run wmk eval CLI, quick POC eval, baseline comparison, regression investigation, run_eval e2e, compare eval runs, diff build configs, WinMLEvaluator"
---

# WinML Evaluation Skill

## When to Use

- Investigate accuracy regression or evaluation issues
- Develop or test a new evaluator class
- Run `winml eval` or the e2e `run_eval.py` script
- Compare eval runs across dates or builds
- Run a quick POC evaluation during development

## Running Commands

Always use `uv run` — this repo uses a venv managed by uv:

```
uv run winml eval ...
uv run winml export ...
uv run winml build ...
uv run python scripts/e2e_eval/run_eval.py ...
uv run python <any_script.py>
```

---

## 1. The `winml eval` Command

Reference: `src/winml/modelkit/commands/eval.py`

### Key Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `-m, --model` | None | HF model ID **or** path to `.onnx` file |
| `--model-id` | None | HF model ID (required when `-m` is an ONNX file) |
| `--task` | auto-detect | e.g., `image-classification`, `token-classification` |
| `--device` | `cpu` | `cpu`, `gpu`, or `npu` |
| `--samples` | `100` | Number of dataset samples |
| `--streaming` | off | Stream dataset instead of full download |
| `--dataset` | auto | HF dataset path (e.g., `imagenet-1k`, `glue`) |
| `--dataset-name` | None | Config name for multi-config datasets (e.g., `mrpc`) |
| `--split` | `validation` | Dataset split |
| `--column` | None | Column mapping as `key=value` (repeatable) |
| `--schema` | off | Print expected dataset schema for `--task` and exit |
| `-o, --output` | None | Output JSON file path |

### Two Modes

1. **Model ID mode** — builds the model with default config, then evaluates:
   ```
   uv run winml eval -m microsoft/resnet-50
   ```

2. **ONNX file mode** — skips build, evaluates the provided ONNX directly (faster):
   ```
   uv run winml eval -m model.onnx --model-id microsoft/resnet-50
   ```
   `--model-id` is always needed (even in ONNX mode) to load HF config and preprocessor.

### Quick POC During Development

When developing a new evaluator or testing changes, save time by:

1. **Use an ONNX file** instead of model ID to skip the build step:
   ```
   uv run winml eval -m path/to/model.onnx --model-id <hf-id> --samples 10 --streaming
   ```

2. **Get an ONNX file quickly** via:
   - `uv run winml export -m <hf-id> -o temp/model.onnx` — fast export with no optimization/quantization
   - Or reuse a cached artifact from `~/.cache/winml/artifacts/<model_id>/`

3. **Use `--streaming`** to avoid downloading the full dataset. Combined with `--samples 10`, this loads only 10 examples on the fly. Use this during development only. Final evaluation requires the full sample count.

4. **Use `--schema`** to check expected dataset columns:
   ```
   uv run winml eval --schema --task object-detection
   ```

---

## 2. Developing a New Evaluator

### Architecture

- Base class: `WinMLEvaluator` in `src/winml/modelkit/eval/base_evaluator.py`
- Evaluator registry: `_EVALUATOR_REGISTRY` in `src/winml/modelkit/eval/evaluate.py`
- Existing specialized evaluators in `src/winml/modelkit/eval/`:
  - `WinMLTextClassificationEvaluator`
  - `WinMLTokenClassificationEvaluator`
  - `WinMLObjectDetectionEvaluator`
  - `WinMLImageSegmentationEvaluator`
  - `WinMLFeatureExtractionEvaluator`
  - `WinMLQuestionAnsweringEvaluator`

### Key Methods to Implement

- `schema_info()` — classmethod returning `list[SchemaColumn]` for dataset schema
- `prepare_data()` — load and preprocess dataset (base handles most cases)
- `prepare_pipeline()` — create HuggingFace pipeline
- `compute()` — run evaluation and return metrics dict

### Development Workflow

1. Create evaluator class extending `WinMLEvaluator`
2. Register it in `_EVALUATOR_REGISTRY`
3. Test quickly with streaming + ONNX file:
   ```
   uv run winml eval -m temp/model.onnx --model-id <hf-id> --task <task> --samples 10 --streaming
   ```
4. Iterate until metrics look correct
5. Run with full samples for final validation
6. Candidate model is in scripts\e2e_eval\testsets\models_all.json
7. Add to `models_with_acc.json` after identifying the dataset for accuracy evaluation
8. Use HF evaluator if available. Otherwise, we need to implement our own evaluator. The implementation cares about the pipeline output and the dataset groud truth, can glue them to calculate the metric.

### POC Baseline Comparison

**IMPORTANT**: `winml eval -m <hf-id>` does NOT compute a PyTorch baseline — it builds an ONNX model and evaluates that. To correctly compare HF baseline vs ONNX:

1. **Export** the model to ONNX: `uv run winml export -m <hf-id> -o temp/model.onnx`
2. **HF baseline**: Load the native PyTorch model and pass it to the evaluator class directly:
   ```python
   from transformers import AutoConfig
   from winml.modelkit.loader.task import resolve_task_and_model_class
   from winml.modelkit.eval.evaluate import _EVALUATOR_REGISTRY
   from winml.modelkit.eval.config import WinMLEvaluationConfig
   from winml.modelkit.datasets.config import DatasetConfig

   config = AutoConfig.from_pretrained(model_id)
   _, model_cls = resolve_task_and_model_class(config, task=task)
   pytorch_model = model_cls.from_pretrained(model_id).eval()

   eval_config = WinMLEvaluationConfig(model_id=model_id, task=task, device="cpu", dataset=dataset_config)
   evaluator_cls = _EVALUATOR_REGISTRY[task]
   evaluator = evaluator_cls(eval_config, pytorch_model)
   hf_metrics = evaluator.compute()
   ```
3. **ONNX eval**: Use `_load_model` with `model_path` pointing to the exported ONNX:
   ```python
   from winml.modelkit.eval.evaluate import _load_model

   eval_config = WinMLEvaluationConfig(model_id=model_id, model_path="temp/model.onnx", task=task, device="cpu", dataset=dataset_config)
   model = _load_model(eval_config)
   evaluator = evaluator_cls(eval_config, model)
   onnx_metrics = evaluator.compute()
   ```

Reference: `scripts/e2e_eval/run_pytorch_baseline.py` for how the e2e system does this.

---

## 3. Investigating Accuracy Regressions

### Common Root Causes

1. **Quantization issues** — most common. Compare quantized vs unquantized models
2. **Build config changes** — different quant precision (e.g., `w4a16` vs `w8a16`, or `activation_type` changing)
3. **Model rebuild** — new build hash means different ONNX, check if build pipeline changed

### Investigation Steps

1. **Check eval results** — compare `eval_results/<date>/models/<model>__<task>/eval_result.json` across runs
2. **Compare build configs** — diff `build_config.json` or the cached `*_winml_build_config.json`
3. **Check cache hashes** — different hash = different model was built. Look at `~/.cache/winml/artifacts/<model_id>/`
4. **Test individual build stages** — the build pipeline has stages: export → optimize → quantize → compile. Each stage's output is cached separately. Use `winml build` with `--no-quant`, `--no-compile`, etc. to isolate which stage caused the regression
5. **Reproduce with `winml eval`** — use the cached ONNX from a specific stage:
   ```
   uv run winml eval -m ~/.cache/winml/artifacts/<model_id>/<cache_key>_optimized.onnx --model-id <hf-id>
   ```

### Cache Structure

```
~/.cache/winml/
├── artifacts/<model_id>/
│   ├── <cache_key>_export.onnx          # Exported (raw ONNX from HF)
│   ├── <cache_key>_optimized.onnx       # After ORT optimization
│   ├── <cache_key>_model.onnx           # Final (quantized + compiled)
│   └── <cache_key>_winml_build_config.json  # Build config used
└── eval_datasets/                       # Cached custom evaluation datasets
```

Cache key format: `{task_abbrev}_{config_hash}` (e.g., `imgcls_512fdf980d1793dc`).

### Comparing Two Eval Runs

Load and compare build configs programmatically:
```python
from winml.modelkit.config.build import WinMLBuildConfig
import json

with open(r"<path_to_build_config.json>") as f:
    config = WinMLBuildConfig.from_dict(json.load(f))
print(config.quant.weight_type, config.quant.activation_type)
print(config.generate_cache_key())
```

---

## 4. HuggingFace Baseline

The `WinMLEvaluator` class (and all derived evaluators) use HuggingFace `pipeline` for inference. Since both `WinMLPreTrainedModel` and HF model classes share the same call signature, the same evaluator can compute baselines with the original HF model.

- Baseline results and datasets are cached
- Baseline cache: `scripts/e2e_eval/cache/baseline_cache.json`
- The e2e script runs baseline automatically when `--eval-type accuracy` or `both`

---

## 5. E2E Evaluation (`run_eval.py`)

Reference: `scripts/e2e_eval/run_eval.py`

### Basic Command

```
uv run python scripts/e2e_eval/run_eval.py \
    --registry scripts/e2e_eval/testsets/models_with_acc.json \
    --eval-type both \
    --device npu
```

### Key Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--registry` | `models_all.json` | Model registry JSON file |
| `--eval-type` | `perf` | `perf`, `accuracy`, or `both` |
| `--device` | `auto` | Target device |
| `--hf-model` | None | Run a single model (overrides registry) |
| `--task` | None | Filter registry by task |
| `--priority` | None | Filter by priority (`P0`, `P1`, `P2`) |
| `--timeout` | 600 | Per-model timeout in seconds |
| `--continue` | off | Skip models that already have results |
| `--retry-failed` | None | Re-run failed models |
| `--list` | off | List filtered models and exit |

### What It Does

1. Loads model registry JSON
2. For each model: runs `winml perf` (build + benchmark), then `winml eval` + PyTorch baseline
3. Results go to `eval_results/<date>/models/<hf_id>__<task>/`
4. Strategy: perf runs first → populates `~/.cache/winml/artifacts/`, eval reuses the cache

### Reproducing a Single Model Issue

```
uv run python scripts/e2e_eval/run_eval.py \
    --registry scripts/e2e_eval/testsets/models_with_acc.json \
    --eval-type both \
    --device npu \
    --hf-model google/vit-base-patch16-224
```

### Registry Format

Each entry in the registry JSON:
```json
{
    "hf_id": "microsoft/resnet-50",
    "task": "image-classification",
    "model_type": "resnet",
    "group": "Top200",
    "priority": "P1",
    "dataset_config": {
        "path": "imagenet-1k",
        "split": "validation",
        "metric": "accuracy"
    }
}
```
