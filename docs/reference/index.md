# Reference — Config Schema

This page documents the full schema for `WinMLBuildConfig`, the JSON configuration
file that drives `winml build` and related commands. Generate a config with
`winml config`, then customize it before feeding it to `winml build -c config.json`.

## Top-Level Structure

```json
{
  "loader":  { ... },
  "export":  { ... },
  "optim":   { ... },
  "quant":   { ... },
  "compile": { ... },
  "eval":    { ... },
  "auto":    true
}
```

Setting `quant` or `compile` to `null` skips that pipeline stage entirely.
Setting `auto` to `true` (default) lets winml-cli auto-configure downstream
stages based on the target device and precision.

---

## `loader` — Model Loading

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `task` | `str \| null` | `null` | HuggingFace task (e.g., `image-classification`). Auto-detected if omitted. |
| `model_class` | `str \| null` | `null` | Override model class (e.g., `AutoModelForCTC`). |
| `model_type` | `str \| null` | `null` | HuggingFace model type (e.g., `bert`, `resnet`). |
| `module_path` | `str \| null` | `null` | Dotted path to a submodule for targeted export. |
| `user_script` | `str \| null` | `null` | Path to custom model class script. |
| `trust_remote_code` | `bool` | `false` | Trust remote code from HuggingFace. |

---

## `export` — ONNX Export

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `opset_version` | `int` | `17` | ONNX opset version. |
| `batch_size` | `int` | `1` | Static batch size. Use 1 for QNN compatibility. |
| `input_tensors` | `list[InputTensorSpec] \| null` | `null` | Input tensor specifications. Auto-inferred if omitted. |
| `output_tensors` | `list[OutputTensorSpec] \| null` | `null` | Output tensor specifications. |
| `dynamic_axes` | `dict \| null` | `null` | Dynamic axes mapping. ⚠️ Breaks MatMulAddFusion on QNN. |
| `export_params` | `bool` | `true` | Include model parameters in ONNX. |
| `do_constant_folding` | `bool` | `true` | Fold constants during export. |
| `verbose` | `bool` | `false` | Verbose export logging. |
| `dynamo` | `bool` | `false` | Use PyTorch 2.x Dynamo exporter. |
| `enable_hierarchy_tags` | `bool` | `true` | Add module hierarchy tags to ONNX nodes. |
| `clean_onnx` | `bool` | `false` | Strip hierarchy tags after export. |
| `hierarchy_tag_format` | `"full" \| "module_only"` | `"full"` | Tag detail level. |

**InputTensorSpec:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str \| null` | Tensor name (e.g., `pixel_values`). |
| `dtype` | `str \| null` | Data type (e.g., `float32`, `int64`). |
| `shape` | `list[int] \| null` | Tensor shape (e.g., `[1, 3, 224, 224]`). |
| `value_range` | `[float, float] \| null` | Min/max for dummy tensor generation. |

---

## `optim` — Graph Optimization

A dictionary of boolean fusion flags. All default to `false` unless auto-configured.

| Field | Type | Description |
|-------|------|-------------|
| `gelu_fusion` | `bool` | Fuse GeLU activation patterns. |
| `layer_norm_fusion` | `bool` | Fuse LayerNorm patterns. |
| `matmul_add_fusion` | `bool` | Fuse MatMul + Add (enables BiasGelu). |

Additional fusion flags can be added as key-value pairs.

---

## `quant` — Quantization

Set to `null` to skip quantization.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `"qdq" \| "static" \| "dynamic"` | `"qdq"` | Quantization mode. |
| `weight_type` | `"uint8" \| "int8" \| "uint16" \| "int16"` | `"uint8"` | Weight data type. |
| `activation_type` | `"uint8" \| "int8" \| "uint16" \| "int16"` | `"uint8"` | Activation data type. |
| `calibration_method` | `"minmax" \| "entropy" \| "percentile"` | `"minmax"` | Scale computation method. |
| `samples` | `int` | `10` | Number of calibration samples. |
| `per_channel` | `bool` | `false` | Per-channel quantization. |
| `symmetric` | `bool` | `false` | Symmetric quantization. |
| `task` | `str \| null` | `null` | Task for dataset-aware calibration. |
| `model_name` | `str \| null` | `null` | Model ID for calibration dataset resolution. |
| `dataset_name` | `str \| null` | `null` | Override calibration dataset. |
| `distribution` | `str` | `"uniform"` | Random distribution for dummy data. |
| `seed` | `int \| null` | `null` | Random seed for reproducibility. |
| `calibration_load_path` | `str \| null` | `null` | Load pre-computed calibration scales. |
| `calibration_save_path` | `str \| null` | `null` | Save calibration scales. |
| `op_types_to_quantize` | `list[str] \| null` | `null` | Operator types to quantize (all if null). |
| `nodes_to_exclude` | `list[str] \| null` | `null` | Node names to skip. |

---

## `compile` — EP Compilation

Set to `null` to skip compilation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ep_config.provider` | `str` | `"qnn"` | EP alias: `qnn`, `cpu`, `dml`, `openvino`, `tensorrt`, `vitisai`, `migraphx`. |
| `ep_config.device` | `str` | `"auto"` | Target device: `npu`, `gpu`, `cpu`, `auto`. |
| `ep_config.enable_ep_context` | `bool` | `true` | Generate EPContext model. |
| `ep_config.embed_context` | `bool` | `false` | Embed binary in ONNX (true) or external .bin (false). |
| `ep_config.compiler` | `str` | `"ort"` | Compiler backend: `ort` or `qairt`. |
| `ep_config.provider_options` | `dict` | `{}` | EP-specific options. |
| `ep_config.qnn_sdk_root` | `str \| null` | `null` | QNN SDK path for QAIRT compiler backend. |
| `validate` | `bool` | `true` | Validate compiled model. |
| `verbose` | `bool` | `false` | Verbose compilation logging. |

---

## `eval` — Evaluation

Set to `null` (default) to skip evaluation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_id` | `str \| null` | `null` | HuggingFace model ID for config resolution. |
| `model_path` | `str \| dict[str, str] \| null` | `null` | Path to .onnx file, or a `{role: path}` dict for composite models. |
| `task` | `str \| null` | `null` | Task type. |
| `device` | `str` | `"auto"` | Inference device. |
| `precision` | `str` | `"auto"` | Precision (`fp32`, `fp16`, `w8a16`, etc.). |
| `ep` | `str \| null` | `null` | EP override. |
| `dataset.path` | `str \| null` | `null` | HuggingFace dataset path. |
| `dataset.name` | `str \| null` | `null` | Dataset config name. |
| `dataset.split` | `str` | `"validation"` | Dataset split. |
| `dataset.samples` | `int` | `100` | Evaluation sample count. |
| `dataset.shuffle` | `bool` | `true` | Shuffle before sampling. |
| `dataset.seed` | `int` | `42` | Random seed. |
| `output_path` | `str \| null` | `null` | Path for JSON results output. |

---

## Example: Full Config

```json
{
  "loader": {
    "task": "image-classification",
    "model_type": "resnet"
  },
  "export": {
    "opset_version": 17,
    "batch_size": 1
  },
  "optim": {
    "gelu_fusion": true,
    "layer_norm_fusion": true,
    "matmul_add_fusion": true
  },
  "quant": {
    "mode": "qdq",
    "weight_type": "uint8",
    "activation_type": "uint8",
    "samples": 10,
    "calibration_method": "minmax"
  },
  "compile": {
    "ep_config": {
      "provider": "qnn",
      "device": "npu",
      "enable_ep_context": true,
      "embed_context": false
    },
    "validate": true
  },
  "auto": true
}
```

### The `auto` field

The top-level `"auto"` field (default: `true`) controls whether the build pipeline runs the **autoconf loop** — an iterative analyze → discover → re-optimize cycle that automatically detects which additional graph optimizations the model needs for the target EP.

| Value | Behavior |
|-------|----------|
| `true` (default) | After initial optimization, the analyzer inspects the graph for unsupported or sub-optimal nodes and proposes additional optimization flags. The pipeline re-optimizes using the discovered flags and repeats (up to `--max-optim-iterations`, default 3). The final optimization result depends on what the analyzer discovers at runtime, so **outputs may vary** if the model or EP support changes between runs. |
| `false` | The pipeline applies only the explicit `optim` flags from the config — no autoconf discovery, no re-optimization loop. Builds are **fully deterministic** given the same config and input model. Use this for reproducible CI builds or when you have already tuned the optimization flags manually. |

When `auto` is `true` and the autoconf loop discovers additional flags, the final persisted config (written to the output directory) includes the merged result so you can inspect what was discovered.

## See also

- [winml config](../commands/config.md) — generate a config interactively
- [winml build](../commands/build.md) — run the pipeline with a config
- [Config and build](../concepts/config-and-build.md) — conceptual overview
