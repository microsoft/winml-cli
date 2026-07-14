# winml config

> Generate a reusable build configuration for a Hugging Face model or ONNX file.

## When to use this

Use `winml config` at the start of a new model project to produce a `WinMLBuildConfig` JSON file. The config captures the model identity, task, precision, and per-stage settings in one shareable artifact that you can edit, version-control, and repeatedly pass to `winml build`. Running config first lets you review and adjust pipeline settings before committing to a full build.

## Synopsis

```bash
$ winml config [options]
```

## Flags

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--model` | `-m` | `TEXT` | *(none)* | HuggingFace model ID (e.g., `microsoft/resnet-50`) or path to an existing `.onnx` file. Optional when `--model-type` or `--model-class` is provided. |
| `--task` | `-t` | `TEXT` | *(auto)* | Override the auto-detected task (e.g., `image-classification`, `text-classification`). When omitted, the first supported task for the model is selected automatically. |
| `--model-class` | | `TEXT` | *(auto)* | Override the auto-detected model class (e.g., `CLIPTextModelWithProjection`). Useful for multi-component models. |
| `--model-type` | | `TEXT` | *(auto)* | Override the auto-detected model type (e.g., `bert`, `resnet`). Can be used without `-m` to generate a config from HuggingFace default settings. |
| `--module` | | `TEXT` | *(none)* | Generate configs for every submodule whose class name matches the given string (e.g., `ResNetConvLayer`). The output is a JSON array instead of a single object. |
| `--config` | `-c` | `PATH` | *(none)* | JSON override file in `WinMLBuildConfig` format. Fields present in this file take precedence over auto-detected values. |
| `--shape-config` | | `PATH` | *(none)* | JSON file with input shape overrides for dummy input generation. Valid keys by modality — text: `sequence_length`; vision: `height`, `width`, `num_channels`; audio: `feature_size`, `nb_max_frames`, `audio_sequence_length`. |
| `--input-specs` | | `PATH` | *(none)* | JSON file with input specifications for the HuggingFace export. Fields are patched onto the auto-resolved input tensors *by name* (unlisted inputs and their `dtype`/`value_range` are preserved); unknown names are appended. Only valid when generating a HuggingFace export config. |
| `--export-config` | | `PATH` | *(none)* | ONNX export configuration JSON (`opset_version`, `do_constant_folding`, etc.) merged into the generated `export` section. Only valid when generating a HuggingFace export config. |
| `--dynamic-axes` | | `PATH` | *(none)* | JSON dynamic axes mapping for the ONNX export (e.g., `{"input_ids": {"0": "batch", "1": "sequence"}}`). Symbolic string dimensions in `--input-specs` shapes also infer dynamic axes. Only valid when generating a HuggingFace export config. |
| `--device` | `-d` | `auto\|npu\|gpu\|cpu` | `auto` | Target device. Affects the generated quantization and compilation sub-configs. `auto` leaves those sections unchanged from the kit defaults. |
| `--ep` | | `TEXT` | *(none)* | Force a specific execution provider (`qnn`, `dml`, `migraphx`, `tensorrt`, `vitisai`, `openvino`, `cpu`). Overrides the device-to-provider mapping. When used without `--device`, the device is inferred from the EP. |
| `--precision` | `-p` | `TEXT` | `auto` | Target precision: `auto`, `fp32`, `fp16`, `int8`, `int16`, or a mixed format such as `w8a16`. `auto` selects the precision based on the chosen device. |
| `--output` | `-o` | `PATH` | *(stdout)* | Write the generated JSON to this file instead of printing to stdout. |
| `--library` | | `TEXT` | `transformers` | Source library for `TasksManager` task lookup. Defaults to `transformers`; set to `diffusers` or another Optimum-supported library when needed. |
| `--quant/--no-quant` | | flag | `true` | Include quantization in the generated config (use `--no-quant` to omit it and set `quant` to `null`). |
| `--no-compile` / `--compile` | | flag | `--no-compile` (compile excluded by default) | Controls whether compilation is included in the generated config. By default compilation is **excluded** (`compile: null`). Pass `--compile` to include a compile section. |
| `--trust-remote-code/--no-trust-remote-code` | | flag | `false` | Allow execution of custom model code from the HuggingFace repository. Required for some community models. Only enable for repositories you trust. |

## How it works

`winml config` queries the HuggingFace `TasksManager` to auto-detect the model's task, class, and ONNX export specification. For known model types it looks up a per-model kit in `MODEL_BUILD_CONFIGS` and uses that as a starting point, layering in your device, precision, and override file on top. When `-m` points to an existing `.onnx` file, the export stage is skipped by setting `export` to `null` in the output. The result is a complete `WinMLBuildConfig` JSON printed to stdout or written to a file, ready to be passed to `winml build`.

## Examples

Generate a config for ResNet-50 with all auto-detected settings:

```bash
$ winml config -m microsoft/resnet-50
```

```text
Generating config for microsoft/resnet-50...
Auto-selected task: image-classification (from 'microsoft/resnet-50')
Generated config for task 'image-classification'
{
  "loader": { "task": "image-classification", ... },
  "export": { "opset_version": 17, ... },
  "optim": { ... },
  "quant": null,
  "compile": null
}
```

Target NPU with int8 quantization and save to a file:

```bash
$ winml config -m microsoft/resnet-50 --device npu --precision int8 -o resnet_npu.json
```

Generate a config for BERT and override the task:

```bash
$ winml config -m bert-base-uncased --task text-classification -o bert_cls.json
```

Generate from a model type alone (no HuggingFace download required at config time):

```bash
$ winml config --model-type bert --task fill-mask
```

Generate a config from an already-exported ONNX file, skipping quantization (compilation is already excluded by default):

```bash
$ winml config -m facebook/convnext-tiny-224.onnx --no-quant -o convnext_optim_only.json
```

Generate a config with dynamic axes so the exported model accepts a variable batch dimension:

```bash
$ winml config -m microsoft/resnet-50 --dynamic-axes dynamic_axes.json -o resnet_dyn.json
```

The `dynamic_axes.json` maps each input to its dynamic dimensions, e.g. `{"pixel_values": {"0": "batch"}}`. Symbolic string dimensions in `--input-specs` shapes (e.g. `{"input_ids": {"shape": ["batch", "sequence"]}}`) infer dynamic axes automatically without a separate `--dynamic-axes` file.

## Common pitfalls

- **At least one of `-m`, `--model-type`, or `--model-class` is required** — calling `winml config` with none of these three flags raises a usage error immediately.
- **`auto` precision does not always map to a lower-bit type** — when `--device` is also `auto`, precision stays at the kit default (usually `fp32`). Explicitly pass `--device npu` or `--device gpu` for `auto` precision to resolve to `int8` or `fp16`.
- **`--module` changes the output shape** — with `--module` the JSON output is an array of configs, not a single object. Scripts that expect a single object will fail to parse this output.
- **`--trust-remote-code` has security implications** — only use this flag with model repositories you own or explicitly trust; it allows arbitrary Python execution from the remote model card.
- **Shape overrides in `--shape-config` are modality-specific** — passing a `sequence_length` key for a vision model has no effect. Check the `--help` description for valid keys per modality.
- **Export controls require a HuggingFace export** — `--input-specs`, `--export-config`, and `--dynamic-axes` are rejected for pre-exported `.onnx` inputs (which set `export` to `null`); use them only when config generates the export section.

## See also

- [Config and build](../concepts/config-and-build.md) — structure of `WinMLBuildConfig` and how stages interact
- [Config Schema](../reference/index.md) — full field-by-field config reference
- [Supported Models](../reference/supported-models.md) — validated model architectures
- [build.md](build.md) — run the full pipeline using a generated config
- [export.md](export.md) — export a HuggingFace model to ONNX as a standalone step
- [optimize.md](optimize.md) — apply graph optimizations to an existing ONNX file
