# Evaluation

## What is `winml eval`?

`winml eval` measures model quality by comparing metrics (e.g. accuracy) between the original HuggingFace PyTorch model and the ONNX model built by ModelKit. This helps detect accuracy regressions introduced during ONNX conversion.

| Task | Metric |
|---|---|
| `image-classification` | accuracy |
| `text-classification` | accuracy |
| `token-classification` | overall_f1 |
| `question-answering` | f1 |
| `object-detection` | map |
| `image-segmentation` | mean_iou |
| `feature-extraction` | cosine_spearman |

## How to Run

### Quick start — evaluate a HuggingFace model

This builds the ONNX model from the HF hub and evaluates it:

```bash
uv run winml eval -m microsoft/resnet-50
```

### Evaluate a pre-built ONNX model

When you already have an `.onnx` file, pass `--model-id` so the command can resolve the preprocessor and config:

```bash
uv run winml eval -m model.onnx --model-id microsoft/resnet-50 --task image-classification
```

### Specify dataset and column mappings

For models that need a specific dataset or a multi-config dataset:

```bash
uv run winml eval -m model.onnx --model-id Intel/bert-base-uncased-mrpc \
    --dataset glue --dataset-name mrpc \
    --column input_column=sentence1
```

### Check expected dataset schema

Use `--schema` to see what dataset columns a task expects before configuring `--column` mappings:

```bash
uv run winml eval --schema --task object-detection
```

Run `uv run winml eval --help` for the full list of options.

### Compute PyTorch baseline

`winml eval` only evaluates the ONNX model. To get the HuggingFace PyTorch baseline for comparison, use `run_pytorch_baseline.py`:

```bash
uv run python scripts/e2e_eval/run_pytorch_baseline.py --model microsoft/resnet-50
```

This loads the native PyTorch model and runs the same evaluator on the same dataset, printing a JSON result to stdout. The E2E pipeline (`run_eval.py --eval-type accuracy`) runs both automatically and computes the delta.

## Common Issues

### Quantization accuracy regression

The most common cause of accuracy regression is quantization. The `winml build` pipeline caches intermediate artifacts under `~/.cache/winml/artifacts/<model_id>/`, including pre-quantized (`*_optimized.onnx`) and post-quantized (`*_model.onnx`) models. To isolate whether quantization is the cause, evaluate the pre-quantized model directly:

```bash
# Evaluate the optimized (pre-quantization) model from cache
uv run winml eval -m ~/.cache/winml/artifacts/<model_id>/<cache_key>_optimized.onnx \
    --model-id <hf-id>
```

If the pre-quantized model has good accuracy but the final model does not, the issue is in quantization.

## Design Specs

| Document | Description |
|---|---|
| [3_design.md](3_design.md) | Eval CLI design |
| [3_feature_extraction_eval.md](3_feature_extraction_eval.md) | Feature extraction / sentence similarity evaluator |
| [3_fill_mask_eval.md](3_fill_mask_eval.md) | Fill-mask evaluator |
| [3_image_segmentation_eval.md](3_image_segmentation_eval.md) | Image segmentation evaluator |
| [3_object_detection_eval.md](3_object_detection_eval.md) | Object detection evaluator |
| [3_question_answering.md](3_question_answering.md) | Question answering evaluator |

## E2E Eval

The E2E eval pipeline batch-evaluates multiple models from a registry, runs PyTorch baselines, compares accuracy deltas, and generates reports.

### Model registry

The model list and dataset configuration live in JSON registry files under `scripts/e2e_eval/testsets/`. Each entry in `models_with_acc.json` specifies the model, task, and dataset to evaluate:

```json
{
    "hf_id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "task": "text-classification",
    "model_type": "roberta",
    "group": "Top200",
    "priority": "P1",
    "dataset_config": {
        "path": "tweet_eval",
        "name": "sentiment",
        "metric": "accuracy",
        "columns_mapping": {
            "input_column": "text"
        }
    }
}
```

### Run E2E accuracy evaluation

```bash
# All models in the accuracy registry
uv run python scripts/e2e_eval/run_eval.py \
    --registry scripts/e2e_eval/testsets/models_with_acc.json \
    --eval-type accuracy

# Filter by task
uv run python scripts/e2e_eval/run_eval.py \
    --registry scripts/e2e_eval/testsets/models_with_acc.json \
    --eval-type accuracy --task image-classification

# Single model
uv run python scripts/e2e_eval/run_eval.py \
    --eval-type accuracy --hf-model microsoft/resnet-50
```

## Coding Agent Skill

`docs/eval/SKILL.md` is an optional AI coding agent skill that provides domain-specific guidance for developing the eval command, writing new evaluators, and investigating accuracy regressions.

To use it, copy or symlink it to your VS Code agent skills directory:

```
~/.agents/skills/wmk-eval/SKILL.md
```

Once placed there, GitHub Copilot will automatically invoke the skill when you work on eval-related tasks.
