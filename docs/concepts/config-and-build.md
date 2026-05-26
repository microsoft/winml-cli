# Config and build

`winml config` and `winml build` are a producer/consumer pair. `winml config`
inspects a Hugging Face model (or an existing ONNX file), auto-detects the task,
model class, and I/O specifications, and writes a `WinMLBuildConfig` JSON file.
`winml build` reads that file and runs the full pipeline — export, optimize,
quantize, compile — producing a Windows ML-ready ONNX artifact.

Keeping these two responsibilities separate is intentional. The config file is a
stable, human-readable description of exactly what the build will do. You can
generate it once, review or edit it, commit it to source control, and replay the
same build at any time without re-running model introspection. CI pipelines and
team workflows both benefit from treating the config file as a versioned artifact
rather than a transient intermediate.

## Generating a config

`winml config` produces a `WinMLBuildConfig` JSON with sensible defaults for the
detected model type. At minimum, provide a model identifier:

```bash
winml config -m microsoft/resnet-50 -o resnet50.json
```

Several flags shape what ends up in the config:

- `--task` overrides the auto-detected Hugging Face task when detection is
  ambiguous or when you want a specific variant (for example, `text-classification`
  vs `feature-extraction`).
- `--no-quant` sets the `quant` section to `null`, so the quantize stage is omitted
  when `winml build` consumes the config. Use this for GPU workflows where float16
  is preferred over QDQ quantization.
- `--no-compile` sets the `compile` section to `null`, producing a portable ONNX
  that the runtime compiles on first load instead of embedding a pre-compiled
  binary.
- `--trust-remote-code` allows model repositories that ship custom modeling code —
  required for some community models that define non-standard architectures outside
  the standard `transformers` library.

If `-o` is omitted, the config is printed to stdout, which is convenient for
piping or quick inspection. The generated JSON is plain text and can be edited
directly before being passed to `winml build`.

## What's in a config

A `WinMLBuildConfig` is a dataclass defined in
`src/winml/modelkit/config/build.py`. It holds five nested sub-configs, one per
pipeline stage:

| Field | Type | Purpose |
|---|---|---|
| `loader` | `WinMLLoaderConfig` | Task, model type, and model class used to load the Hugging Face model. |
| `export` | `WinMLExportConfig` | Input/output tensor specs, opset version, dynamic axes (`null` for pre-exported ONNX). |
| `optim` | `WinMLOptimizationConfig` | Graph fusion flags (GeLU, LayerNorm, MatMul+Add). |
| `quant` | `WinMLQuantizationConfig` | Precision types (`weight_type`, `activation_type`), calibration samples and method (`null` to skip). |
| `compile` | `WinMLCompileConfig` | Target EP provider, EPContext options, compiler backend (`null` to skip). |

Setting `quant` or `compile` to `null` tells the pipeline to skip that stage
entirely, equivalent to passing `--no-quant` or `--no-compile` on the command
line.

A generated config looks similar to:

```json
{
  "loader": {
    "task": "image-classification"
  },
  "export": {
    "opset_version": 17,
    "batch_size": 1
  },
  "optim": {
    "gelu_fusion": false,
    "layer_norm_fusion": false,
    "matmul_add_fusion": false
  },
  "quant": {
    "mode": "qdq",
    "weight_type": "uint8",
    "activation_type": "uint8",
    "samples": 10
  },
  "compile": {
    "ep_config": {
      "provider": "qnn",
      "enable_ep_context": true
    }
  }
}
```

The file is plain JSON. You can hand-edit any field before passing it to
`winml build` — adjust the calibration sample count, change the compile
provider, or remove a fusion flag.

## Consuming a config

Pass the config file to `winml build` with either an output directory or the
global cache flag:

```bash
# Write artifacts to a local directory
winml build -c resnet50.json -m microsoft/resnet-50 --output-dir output/

# Write to the global cache (~/.cache/winml/)
winml build -c resnet50.json -m microsoft/resnet-50 --use-cache
```

`--output-dir` and `--use-cache` are mutually exclusive; you must supply one of
the two when running `winml build` (enforced at runtime, not parse time). Within the output directory, `winml build` writes one ONNX file per
completed stage so that intermediate artifacts are available for inspection, and
it writes a copy of the resolved config so the full build parameters are recorded
alongside the outputs.

## Overrides at run time

CLI flags passed directly to `winml build` override the corresponding config
sections for that run only, without modifying the JSON file on disk. This makes
it straightforward to experiment with a variation without creating a new config:

```bash
# Skip quantization and compilation for this run only
winml build -c resnet50.json -m microsoft/resnet-50 --output-dir output/ --no-quant --no-compile

# Skip optimization (for a pre-quantized input ONNX)
winml build -c resnet50.json -m model_qdq.onnx --output-dir output/ --no-optimize
```

`--no-quant`, `--no-compile`, and `--no-optimize` each suppress the corresponding
stage regardless of what the config file specifies. Because the config file is
unchanged, re-running without the override flag reverts to the full pipeline
described in the config.

## Why version a config

Storing the `WinMLBuildConfig` JSON in source control brings three concrete
benefits:

1. **Reproducibility.** A config file pins every build decision — task, precision,
   quantization method, calibration sample count, target EP, fusion flags — in a
   single file. Running `winml build -c config.json` six months later produces the
   same artifact as it does today, regardless of how the tool's defaults evolve.

2. **CI integration.** A CI job can run `winml build -c config.json -m <model-id>
   --output-dir artifacts/` with no human intervention. Because all settings live
   in the config file, the CI script requires no per-model flag knowledge, and
   updating build parameters is a pull request to the config file, not a change to
   the pipeline script.

3. **Team sharing.** Handing a colleague a config file is enough for them to
   reproduce the exact build on their machine. There is no need to document the
   sequence of primitive commands, precision arguments, or calibration settings
   separately — the file is the documentation.

## See also

- [Primitives and pipeline](primitives-and-pipeline.md) — when to use `winml build`
  vs individual primitive commands
- [winml config command reference](../commands/config.md)
- [winml build command reference](../commands/build.md)
