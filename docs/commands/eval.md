# winml eval

> Evaluate ONNX or native Hugging Face PyTorch model accuracy on a standard dataset.

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
| `--shape-config` | | `PATH` | — | JSON shape overrides used while auto-generating a Hugging Face export config, for example `{"height": 480, "width": 480}`. Applies only when `-m` is a HuggingFace ID that eval builds; **ignored for pre-built `.onnx` inputs**. |
| `--input-specs` | | `PATH` | — | JSON input tensor specs to merge into the Hugging Face export config. Symbolic string dimensions infer dynamic axes. **Ignored for pre-built `.onnx` inputs**. |
| `--export-config` | | `PATH` | — | JSON ONNX export config overrides (opset version, constant folding, etc.) to merge into the Hugging Face export config. **Ignored for pre-built `.onnx` inputs**. |
| `--dynamic-axes` | | `PATH` | — | JSON dynamic axes mapping for Hugging Face ONNX export, for example `{"input_ids": {"0": "batch", "1": "sequence"}}`. **Ignored for pre-built `.onnx` inputs**. |
| `--runtime` | | `winml-ort\|pytorch` | `winml-ort` | Evaluation runtime. `winml-ort` exports Hugging Face checkpoints to ONNX; `pytorch` evaluates the original checkpoint and supports `auto`, `cpu`, or CUDA-backed `gpu` devices. |
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
| `--mode` | | `onnx\|compare` | `onnx` | Evaluation mode. `onnx` evaluates the ONNX candidate on a dataset. `compare` runs the ONNX candidate and a reference on identical random inputs and reports per-tensor similarity metrics — no dataset required. The reference is the HuggingFace model from `--model-id` by default, or a second ONNX file when `--reference` is given. |
| `--input-data` | | `PATH` | — | Path to a `.npz` file of real input tensors to compare with instead of randomly generated ones (used with `--mode compare`). Keys must match the candidate model's input names. The **leading axis of each array is the sample axis**, so an archive shaped `(N, ...)` yields `N` samples (mean/std/min/max are computed across them); all inputs must share the same `N`. Each run is shaped to the candidate's batch size — a dynamic batch runs one row per sample, a static batch `B` chunks the axis into `N // B` batches (trailing rows are dropped with a warning). Note this differs from `winml perf --input-data`, which runs the **whole archive as a single batch**. |
| `--reference` | | `TEXT` | — | Reference `.onnx` file to compare the candidate against (used with `--mode compare`). Compares two ONNX models on identical random inputs; `--model-id` and `--task` are not required in this mode. |
| `--reference-device` | | `cpu\|gpu\|npu\|auto` | `cpu` | Device used for the reference ONNX model. Only valid with `--reference`. |
| `--reference-ep` | | `TEXT` | — | Explicit execution provider used for the reference ONNX model, for example `dml`. Only valid with `--reference`. |

## How it works

`winml eval` loads the model and runs the evaluation pipeline via the internal `evaluate` function, then pulls the requested number of samples from a HuggingFace dataset. By default, Hugging Face model IDs and local checkpoints use the `winml-ort` runtime: they are exported to ONNX and evaluated through WinML. With `--runtime pytorch`, the task-resolved PyTorch model and stored dtype are preserved and the same dataset preprocessing, evaluator, and metrics run directly against that model. PyTorch `auto` selects CUDA when available and otherwise CPU; `gpu` requires CUDA. The JSON report identifies the effective runtime as `winml-ort` or `pytorch`.

Python callers can pass an existing model directly with `evaluate(config, pytorch_model=model)`. An explicit `config.model_id` selects the tokenizer or processor; otherwise evaluation infers it from `model.config._name_or_path` and reports an error if neither source is available.

PyTorch text-generation evaluation adapts the checkpoint to the existing causal-LM evaluator contract and reports perplexity without ONNX export. Pre-built ONNX files, composite `role=path` models, GenAI bundles, compare mode, references, and tensor input archives remain on their existing WinML paths. `--runtime pytorch` rejects those forms, along with ONNX build, export, EP, precision, quantization, optimization, analysis, and cache-related options.

## Examples

Evaluate a HuggingFace model using the task-default dataset:

```bash
$ winml eval -m microsoft/resnet-50
```

Evaluate the original Hugging Face PyTorch checkpoint on CPU without ONNX export:

```bash
$ winml eval -m microsoft/resnet-50 --runtime pytorch --device cpu
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

Compare an ONNX candidate against its HuggingFace reference on real input tensors instead of random ones by passing a `.npz` archive whose keys match the candidate's input names. The leading axis of each array is the sample axis, so an archive shaped `(N, ...)` runs `N` samples:

```bash
$ winml eval --mode compare -m model.onnx --model-id microsoft/resnet-50 --input-data inputs.npz
```

Compare two ONNX files directly (e.g. an fp32 baseline vs a quantized build), reporting per-output tensor-similarity metrics on identical random inputs — no `--model-id` or dataset needed:

```bash
$ winml eval --mode compare -m quantized.onnx --reference baseline.onnx
```

Compare a candidate running on DML with an ONNX reference running on CPU:

```bash
$ winml eval --mode compare -m candidate.onnx --device gpu --ep dml --reference baseline.onnx
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

## Model build cache

Evaluation reuses persistent model build artifacts by default. Pass
`--no-use-cache` for a fresh build in a temporary directory, or `--rebuild` for
a fresh build that replaces the persistent cache entry.

For a pre-built ONNX input, cache controls apply only when
`--no-skip-build` is set. The CLI warns when an explicit cache control has no
effect because no model build runs. GenAI's runtime `_compiled/` artifacts are
a separate cache and are not currently governed by these model build cache
controls. Explicit build
or cache controls on a GenAI bundle produce a warning that distinguishes the
model-build pipeline from the runtime compilation cache.

## Common pitfalls

- **ONNX file without `--model-id` fails.** When `-m` is a `.onnx` path, `--model-id` is mandatory. Without it the command cannot resolve the preprocessor or label vocabulary and will exit with a usage error.
- **The task-default dataset may not match every model.** A default dataset cannot fit every model. Classification and detection models in particular need a dataset whose label space and domain match what the model was trained on — using the default may produce misleadingly low scores, missing-label errors, or a dataset-schema error. Always pass `--dataset` (and `--label-mapping` if needed) when evaluating a model whose label space or domain differs from the task default.
- **Some dataset requires Hub credentials for gated datasets.** Some datasets (e.g., `imagenet-1k`) require a HuggingFace account with accepted terms of use. Log in with `huggingface-cli login` before running eval on gated data.
- **`--shuffle` is on by default.** The random 100-sample slice changes between runs unless you pass `--no-shuffle`. Use `--no-shuffle` when comparing two model variants to ensure they see identical samples.
- **`--streaming` skips the local cache.** Streaming mode avoids downloading the full split but prevents random shuffling on large datasets. For reproducible evaluation, download the split once and omit `--streaming`.
- **Export overrides only apply when eval builds from a HuggingFace ID.** `--shape-config`, `--input-specs`, `--export-config`, and `--dynamic-axes` shape the ONNX export that `eval` generates when `-m` is a HuggingFace model ID. When `-m` is a pre-built `.onnx` file, there is no export step, so these flags are ignored and the command prints a warning.
- **The PyTorch runtime accepts only Hugging Face checkpoints.** `--runtime pytorch` cannot be combined with ONNX files, composite models, GenAI bundles, compare/reference/input-data modes, EP selection, or ONNX build/export controls. Use `--device cpu`, `--device gpu` with CUDA, or `--device auto`.
- **Column names vary across datasets.** If the evaluator raises a missing-column error, run `winml eval --schema --task <task>` to inspect the expected schema and use `--column` to remap dataset field names to the expected names.

## See also

- [winml perf](perf.md) — measure latency and throughput on the same model
- [winml build](build.md) — produce the quantized artifact to evaluate
- [Quantization & QDQ](../concepts/quantization.md) — why accuracy validation after quantization matters
- [ONNX & Execution Providers](../concepts/eps-and-devices.md) — understand the `--device` option
