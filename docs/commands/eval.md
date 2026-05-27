# winml eval

> Evaluate ONNX model accuracy on a standard dataset.

## When to use this

Use `winml eval` to measure how accurately a model performs on real data — especially after quantization, where comparing the quantized model against the floating-point baseline reveals any accuracy regression introduced by precision reduction.

## Synopsis

```bash
$ winml eval [options]
```

## Flags

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--model` | `-m` | `TEXT` | — | HuggingFace model ID, or path to a local `.onnx` file. Required (unless `--model-id` is provided directly). |
| `--model-id` | | `TEXT` | — | HuggingFace model ID used for preprocessor and config resolution when `-m` points to an `.onnx` file. Required when `-m` is an ONNX file. |
| `--dataset` | | `TEXT` | task default | HuggingFace dataset path (e.g., `imagenet-1k`, `glue`). If omitted, a default dataset is selected based on the task. |
| `--dataset-name` | | `TEXT` | — | Dataset configuration name for multi-config datasets (e.g., `mrpc` within `glue`). |
| `--task` | | `TEXT` | auto-detected | Task name (e.g., `image-classification`). Auto-detected from `--model-id` when not provided. |
| `--device` | | `auto\|cpu\|gpu\|npu` | `auto` | Device to run inference on during evaluation. `auto` selects the best available device. |
| `--samples` | | `INTEGER` | `100` | Number of dataset samples to evaluate. |
| `--split` | | `TEXT` | `validation` | Dataset split to use (e.g., `validation`, `test`, `train`). |
| `--shuffle / --no-shuffle` | | flag | `shuffle` | Shuffle the dataset before sampling. Disable with `--no-shuffle` for reproducible sample ordering. |
| `--streaming` | | flag | `false` | Stream the dataset from the Hub instead of downloading the full split. Useful for large datasets. |
| `--column` | | `TEXT` (multiple) | — | Column mapping as `key=value` pairs (e.g., `--column input_column=image`). Can be specified multiple times. |
| `--label-mapping` | | `PATH` | — | Path to a JSON file mapping label names to integer IDs: `{"label_name": id}`. |
| `--output` | `-o` | `PATH` | — | Output JSON file path for the evaluation results. |
| `--schema` | | flag | `false` | Print the expected dataset schema for the given `--task` and exit. Does not run evaluation. |

## How it works

`winml eval` loads the model and runs the evaluation pipeline via the internal `evaluate` function (supporting both HuggingFace IDs and local ONNX files), then pulls the requested number of samples from a HuggingFace dataset. Each sample is preprocessed using the tokenizer or image processor associated with the model ID, passed through the ONNX Runtime session, and the output is compared against the ground-truth label. Aggregated metrics (accuracy, F1, etc.) are printed to the console and optionally written to a JSON file. When `-m` is an ONNX file, `--model-id` must be provided so the command knows which preprocessor and label vocabulary to use.

## Examples

Evaluate a HuggingFace model using the task-default dataset:

```bash
$ winml eval -m microsoft/resnet-50
```

```text
Task:     image-classification
Dataset:  imagenet-1k (validation, 100 samples)
Device:   auto

Accuracy: 76.00%

Results saved to: microsoft_resnet-50_eval.json
```

Evaluate a pre-exported ONNX file, providing the source model ID for preprocessing:

```bash
$ winml eval -m model.onnx --model-id microsoft/resnet-50 --dataset imagenet-1k
```

Evaluate a BERT model on the MRPC paraphrase task with column remapping:

```bash
$ winml eval -m bert-base-uncased --dataset glue --dataset-name mrpc \
    --column input_column=sentence1 --samples 500
```

Check what dataset columns are expected before running, then evaluate on the NPU:

```bash
$ winml eval --schema --task image-classification
$ winml eval -m facebook/convnext-tiny-224 --device npu --samples 200 --split test
```

Evaluate with a custom label mapping file and save results:

```bash
$ winml eval -m model.onnx --model-id microsoft/resnet-50 \
    --label-mapping labels.json -o results/resnet_eval.json
```

## Common pitfalls

- **ONNX file without `--model-id` fails.** When `-m` is a `.onnx` path, `--model-id` is mandatory. Without it the command cannot resolve the preprocessor or label vocabulary and will exit with a usage error.
- **Default dataset requires Hub credentials for gated datasets.** Some task defaults (e.g., `imagenet-1k`) require a HuggingFace account with accepted terms of use. Log in with `huggingface-cli login` before running eval on gated data.
- **`--shuffle` is on by default.** The random 100-sample slice changes between runs unless you pass `--no-shuffle`. Use `--no-shuffle` when comparing two model variants to ensure they see identical samples.
- **`--streaming` skips the local cache.** Streaming mode avoids downloading the full split but prevents random shuffling on large datasets. For reproducible evaluation, download the split once and omit `--streaming`.
- **Column names vary across dataset versions.** If the evaluator raises a missing-column error, run `winml eval --schema --task <task>` to inspect the expected schema and use `--column` to remap dataset field names to the expected names.

## See also

- [winml perf](perf.md) — measure latency and throughput on the same model
- [winml build](build.md) — produce the quantized artifact to evaluate
- [Quantization & QDQ](../concepts/quantization.md) — why accuracy validation after quantization matters
- [ONNX & Execution Providers](../concepts/eps-and-devices.md) — understand the `--device` option
