# winml export

> Convert a PyTorch / Hugging Face model to ONNX, preserving module hierarchy.

## When to use this

Use `winml export` when you have a Hugging Face model ID or a local PyTorch
checkpoint and need an ONNX file as the first step of the optimization
pipeline. This is the entry point before `winml quantize` or `winml compile`.

## Synopsis

```bash
$ winml export [options]
```

## Flags

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--model` | `-m` | string | *(required)* | Hugging Face model name or local path (e.g., `prajjwal1/bert-tiny`). |
| `--output` | `-o` | path | *(required)* | Output ONNX file path (e.g., `model.onnx`). |
| `--with-report/--no-with-report` | | flag | `false` | Generate full export reports: Markdown, JSON, and a console tree. |
| `--hierarchy/--no-hierarchy` | | flag | `true` | Preserve `hierarchy_tag` metadata in ONNX nodes (use `--no-hierarchy` for a clean ONNX file). |
| `--dynamo/--no-dynamo` | | flag | `true` | Use PyTorch's TorchDynamo ONNX exporter (default) for richer per-node module metadata. Pass `--no-dynamo` for the legacy TorchScript exporter, whose opset-17 op decomposition is the validated path for QNN/NPU compilation today. |
| `--torch-module` | | string | `None` | Comma-separated list of `torch.nn` module types to include in hierarchy (e.g., `LayerNorm,Embedding`). (Experimental — currently logs a warning.) |
| `--input-specs` | | path | `None` | JSON file with explicit input tensor specifications. Auto-generated when omitted. |
| `--task` | `-t` | string | `None` | Override auto-detected Hugging Face task (e.g., `image-feature-extraction`). |
| `--export-config` | | path | `None` | JSON file with ONNX export parameters such as `opset_version` and `do_constant_folding`. |
| `--shape-config` | | path | `None` | JSON object mapping symbolic dimension names to concrete sizes (e.g., `{"sequence_length": 2048}`). Ignored when `--input-specs` is provided. |
| `--dynamic-axes` | | path | `None` | JSON object mapping tensor names to dynamic axis names for ONNX export (e.g., `{"input_ids": {"0": "batch", "1": "sequence"}}`). |
| `--trust-remote-code/--no-trust-remote-code` | | flag | `false` | Allow executing custom code from model repositories during export. Use only with trusted sources. |
| `--allow-unsupported-nodes/--no-allow-unsupported-nodes` | | flag | `false` | Allow unsupported nodes to remain in the exported graph instead of failing export. |
| `--help` | `-h` | flag | | Show this message and exit. |

## How it works

`winml export` loads the model via Hugging Face `transformers`, then runs the
eight-step Hierarchy-preserving Tags Protocol (HTP): model preparation, input
generation, module-hierarchy tracing, TorchScript ONNX export, node-tagger
creation, per-node tagging, tag injection into ONNX `metadata_props`, and
optional report generation. The hierarchy metadata allows downstream tools to
reason about operators grouped by their originating module rather than flat
graph position. When `--no-hierarchy` is specified, hierarchy steps are bypassed
and a bare ONNX file is written, useful for third-party tools that do not
understand custom metadata.

## Examples

```bash
# Minimal export: Hugging Face model ID to ONNX file
winml export -m microsoft/resnet-50 -o resnet50.onnx
```

```text
Model: microsoft/resnet-50
Output: resnet50.onnx

Starting HTP export...
  Detected task: image-classification

Success! Model exported to: resnet50.onnx
```

```bash
# Export with verbose output and full Markdown + JSON reports
winml export -m facebook/convnext-tiny-224 -o convnext.onnx -v --with-report
```

```bash
# Export a BERT model, overriding input shapes for longer sequences
winml export -m bert-base-uncased -o bert.onnx \
  --shape-config shape.json
# shape.json: {"sequence_length": 512}
```

```bash
# Export with a hand-crafted input-spec file (skips auto-detection)
winml export -m bert-base-uncased -o bert.onnx --input-specs inputs.json
```

```bash
# Export a fully static-shaped model (the default) for NPU/QNN compilation.
# The default dummy inputs already produce static shapes; use --shape-config to
# pin any symbolic dimensions to concrete sizes, or --input-specs to fully
# specify every input tensor.
winml export -m bert-base-uncased -o bert.onnx --shape-config shape_config.json
# shape_config.json: {"sequence_length": 128}
```

```bash
# Export with dynamic batch and sequence dimensions
winml export -m bert-base-uncased -o bert.onnx --dynamic-axes dynamic_axes.json
# dynamic_axes.json:
# {"input_ids": {"0": "batch", "1": "sequence"}, "attention_mask": {"0": "batch", "1": "sequence"}}
```

`--input-specs` also accepts symbolic dimension names in `shape`; symbolic
entries are used as dynamic ONNX axis names while size `1` is used for the
dummy input tensor. For example, `{"input_ids": {"dtype": "int64", "shape":
["batch", "sequence"]}}` exports `input_ids` with dynamic `batch` and
`sequence` dimensions.

```bash
# Produce clean ONNX without hierarchy metadata (for third-party optimizers)
winml export -m microsoft/resnet-50 -o resnet50_clean.onnx --no-hierarchy
```

## See also

- [winml optimize](optimize.md) — the next pipeline stage after export
- [Supported Models](../reference/supported-models.md) — full list of validated architectures
- [Load and export concept](../concepts/load-and-export.md) — details on the export process

## Common pitfalls

- **Task detection fails on unusual model IDs.** If auto-detection picks the
  wrong task (or fails entirely), pass `-t` with the correct task string, for
  example `-t image-feature-extraction`.
- **`--shape-config` is silently ignored when `--input-specs` is set.**
  `--input-specs` takes full priority; remove it if you only want to override
  individual dimensions.
- **Dynamic dimensions can reduce QNN optimization coverage.** Static batch and
  static shapes remain the default because some QNN fusions require them. Use
  `--dynamic-axes` only when downstream runtime scenarios need variable sizes.
- **Dynamo is the default exporter.** `winml export` uses PyTorch's TorchDynamo
  ONNX exporter, which records rich per-node module metadata that drives the
  hierarchy tags. Pass `--no-dynamo` to select the legacy TorchScript exporter.
  Shape staticness is independent of the exporter: both default to static shapes
  and only emit dynamic axes when you ask for them. The QNN-relevant
  difference is op decomposition -- the dynamo exporter defaults to a newer opset
  and lowers some ops differently, which can reduce QNN fusion coverage
  (e.g. MatMul + Add fusion). Prefer `--no-dynamo` for hand exports targeting
  QNN/NPU compilation until the dynamo graph is validated for your model.
- **`--torch-module` is experimental.** The flag emits a warning and has no
  effect in the current release. Do not rely on it in automated pipelines yet.
- **Output directory must be writable.** The command creates parent directories
  automatically, but will fail with a permission error on read-only paths.
- **Model weights are downloaded to the Hugging Face cache.** Set `HF_HOME` or
  `HF_HUB_CACHE` to control the download location.

## See also

- [winml quantize](quantize.md)
- [winml compile](compile.md)
- [winml build](build.md)
- [Load and export concept](../concepts/load-and-export.md)
