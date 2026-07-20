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
| `--task` | | `TEXT` | auto-detected | Task name (e.g., `image-classification`). Auto-detected from `--model-id` when not provided. Required when `-m` is an ONNX file and the task cannot be inferred. |
| `--precision` | | `TEXT` | `auto` | Precision used when building the model from a HuggingFace ID. One of `auto`, `fp32`, `fp16`, `int8`, `int16`, or a mixed `w{x}a{y}` spec (e.g., `w8a16`). `fp16`/`fp32` skip quantization. **Ignored** when `-m` is a pre-built `.onnx` file — the precision is already baked in. |
| `--device` | | choice | `auto` | Target device. Choices: `auto`, `npu`, `gpu`, `cpu`. `auto` selects the best available device. Combined with `--precision`, this drives the build when `-m` is a HuggingFace ID. |
| `--ep` / `--execution-provider` | | `TEXT` | — | Target ONNX Runtime execution provider when finer control than `--device` is needed. Full names (e.g., `QNNExecutionProvider`, `OpenVINOExecutionProvider`, `VitisAIExecutionProvider`) and aliases (`qnn`, `ov`/`openvino`, `vitis`/`vitisai`) are accepted. |
| `--dataset` | | `TEXT` | task default | HuggingFace dataset path (e.g., `imagenet-1k`, `nyu-mll/glue`). If omitted, a default dataset is selected based on the task. |
| `--dataset-name` | | `TEXT` | — | Dataset configuration name for multi-config datasets. |
| `--dataset-revision` | | `TEXT` | — | Git revision (branch, tag, or commit) of the dataset to load. Use `refs/convert/parquet` for HF datasets that are only served via the parquet mirror. |
| `--dataset-script` | | `TEXT` | — | Path to a Python script that builds the evaluation dataset locally. Requires `--trust-remote-code`. |
| `--trust-remote-code / --no-trust-remote-code` | | flag | `false` | Allow executing custom code from model repositories or dataset scripts. Required with `--dataset-script`. Use only with trusted sources. |
| `--samples` | | `INTEGER` | `100` | Number of dataset samples to evaluate. |
| `--split` | | `TEXT` | `validation` | Dataset split to use (e.g., `validation`, `test`, `train`). |
| `--shuffle / --no-shuffle` | | flag | `shuffle` | Shuffle the dataset before sampling. Disable with `--no-shuffle` for reproducible sample ordering. |
| `--streaming / --no-streaming` | | flag | `false` | Stream the dataset from the Hub instead of downloading the full split. Useful for large datasets. |
| `--column` | | `TEXT` (multiple) | — | Column mapping as `key=value` pairs (e.g., `--column input_column=image`). Can be specified multiple times. |
| `--label-mapping` | | `PATH` | — | Path to a JSON file mapping dataset label names to the integer class IDs the model emits: `{"label_name": id}`. |
| `--output` | `-o` | `PATH` | — | Output JSON file path for the evaluation results. |
| `--schema` | | flag | `false` | Print the expected dataset schema for the given `--task` and exit. Does not run evaluation. |
| `--mode` | | `onnx\|compare` | `onnx` | Evaluation mode. `onnx` evaluates the ONNX candidate on a dataset. `compare` runs the ONNX candidate and a reference on identical inputs and reports per-tensor similarity metrics — no dataset required. The reference is the HuggingFace model from `--model-id` by default, or a second ONNX file when `--reference` is given. |
| `--reference` | | `TEXT` | — | Reference `.onnx` file to compare the candidate against (used with `--mode compare`). Compares two ONNX models on identical random inputs; `--model-id` and `--task` are not required in this mode. Both models run on the same `--device` / `--ep`. |
| `--input-data` | | `PATH` | — | Path to a `.npz` file of real input tensors to compare with instead of randomly generated ones (used with `--mode compare`). Keys must match the candidate model's input names; the whole archive is treated as a single sample, so `--samples` / `--seed` are ignored. |

## How it works

`winml eval` loads the model and runs the evaluation pipeline via the internal `evaluate` function (supporting both HuggingFace IDs and local ONNX files), then pulls the requested number of samples from a HuggingFace dataset. Each sample is preprocessed using the tokenizer or image processor associated with the model ID, passed through the ONNX Runtime session, and the output is compared against the ground-truth label. Aggregated metrics (accuracy, F1, etc.) are printed to the console and optionally written to a JSON file. When `-m` is an ONNX file, `--model-id` must be provided so the command knows which preprocessor and label vocabulary to use.

## Examples

Evaluate a HuggingFace model using the task-default dataset:

```bash
$ winml eval -m microsoft/resnet-50
```

```text
Task:     image-classification
Dataset:  timm/mini-imagenet (test, 100 samples)
Device:   auto

Accuracy: 76.00%

Results saved to: microsoft_resnet-50_eval.json
```

Evaluate a pre-exported ONNX file, providing the source model ID for preprocessing:

```bash
$ winml eval -m model.onnx --model-id microsoft/resnet-50 --dataset timm/mini-imagenet
```

Evaluate a BERT model on the MRPC paraphrase task with column remapping:

```bash
$ winml eval -m Intel/bert-base-uncased-mrpc --dataset nyu-mll/glue --dataset-name mrpc --column input_column=sentence1 --column second_input_column=sentence2 --samples 500
```

Compare two ONNX files directly (e.g. an fp32 baseline vs a quantized build), reporting per-output tensor-similarity metrics on identical random inputs — no `--model-id` or dataset needed:

```bash
$ winml eval --mode compare -m quantized.onnx --reference baseline.onnx
```

Compare on real input tensors instead of random ones by passing a `.npz` archive whose keys match the candidate's input names (the whole archive is one sample):

```bash
$ winml eval --mode compare -m quantized.onnx --reference baseline.onnx --input-data inputs.npz
```

Check what dataset columns are expected before running, then remap them to match your dataset:

```bash
$ winml eval --schema --task text-classification
```

```text
Input schema for text-classification models
==================================================

--column option schema

Evaluating needs a dataset with the following columns:
  input_column
      input text (default: text)
  label_column
      class label (ClassLabel or integer) (default: label)
  second_input_column
      second text for sentence-pair tasks (optional) (default: None)

Override any default with --column:
  --column input_column=<your_text_column>
  --column label_column=<your_label_column>
  --column second_input_column=<your_pair_column>
```

The GLUE SST-2 dataset uses `sentence` instead of the default `text` column, so remap it with a single `--column` override:

```bash
$ winml eval -m distilbert/distilbert-base-uncased-finetuned-sst-2-english --dataset nyu-mll/glue --dataset-name sst2 --column input_column=sentence --samples 500
```

Evaluate against a custom dataset whose label names differ from the model's class IDs. The `--label-mapping` flag points to a JSON file whose **keys are the label name strings as they appear in the dataset** and whose **values are the integer class IDs the model emits**. For example, ResNet-50 outputs ImageNet-1k class IDs (`0`–`999`), so if your custom dataset uses readable strings like `"tabby cat"` or `"golden retriever"`, `labels.json` translates each dataset label to the corresponding ImageNet ID the model predicts:

```json
{
  "tabby cat": 281,
  "Egyptian cat": 285,
  "golden retriever": 207
}
```

```bash
$ winml eval -m microsoft/resnet-50 --dataset my-org/my-pets-dataset --label-mapping labels.json -o results/resnet_eval.json
```

Evaluate a composite model from pre-exported ONNX files. Some tasks (e.g., `image-to-text`, encoder-decoder, dual-encoder) split the model across multiple ONNX files, one per role. Pass `-m` once per role as `<role>=<path>.onnx` and supply `--model-id` so the preprocessor and tokenizer can be resolved. Run `winml eval --schema --task image-to-text` to see the expected roles for a task:

```bash
$ winml eval -m encoder=encoder.onnx -m decoder=decoder.onnx --model-id microsoft/trocr-base-printed
```

## Common pitfalls

- **ONNX file without `--model-id` fails.** When `-m` is a `.onnx` path, `--model-id` is mandatory. Without it the command cannot resolve the preprocessor or label vocabulary and will exit with a usage error.
- **The task-default dataset may not match every model.** A default dataset cannot fit every model. Classification and detection models in particular need a dataset whose label space and domain match what the model was trained on — using the default may produce misleadingly low scores, missing-label errors, or a dataset-schema error. Always pass `--dataset` (and `--label-mapping` if needed) when evaluating a model whose label space or domain differs from the task default.
- **Some dataset requires Hub credentials for gated datasets.** Some datasets (e.g., `imagenet-1k`) require a HuggingFace account with accepted terms of use. Log in with `huggingface-cli login` before running eval on gated data.
- **`--shuffle` is on by default.** The random 100-sample slice changes between runs unless you pass `--no-shuffle`. Use `--no-shuffle` when comparing two model variants to ensure they see identical samples.
- **`--streaming` skips the local cache.** Streaming mode avoids downloading the full split but prevents random shuffling on large datasets. For reproducible evaluation, download the split once and omit `--streaming`.
- **Column names vary across datasets.** If the evaluator raises a missing-column error, run `winml eval --schema --task <task>` to inspect the expected schema and use `--column` to remap dataset field names to the expected names.

## See also

- [winml perf](perf.md) — measure latency and throughput on the same model
- [winml build](build.md) — produce the quantized artifact to evaluate
- [Quantization & QDQ](../concepts/quantization.md) — why accuracy validation after quantization matters
- [ONNX & Execution Providers](../concepts/eps-and-devices.md) — understand the `--device` option
