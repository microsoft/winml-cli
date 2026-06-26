# BERT — Config + Build + Perf

BERT (`bert-base-uncased`) is a canonical text model that exercises every stage of the winml-cli pipeline: it has multiple input tensors, benefits from graph fusion (GeLU, LayerNorm, MatMul+Add), and produces quantizable activations that run well on NPU. That combination makes it a useful reference point for teams deploying transformer encoders on Windows.

This sample walks through the production-style workflow: generate a reusable `WinMLBuildConfig` JSON file with `winml config`, run the full export → optimize → quantize → compile pipeline in one shot with `winml build`, and measure the result with `winml perf`. If you want to understand each pipeline stage individually before running the all-in-one command, read the [Hugging Face Model to NPU tutorial](../tutorials/npu-convnext.md) first.

## Prerequisites

- winml-cli installed and `winml` on your PATH.
- A target device (NPU or GPU recommended; CPU also works).

## Step 1: Generate a build config

```bash
winml config -m bert-base-uncased -t text-classification -o bert_config.json
```

This writes a `WinMLBuildConfig` JSON file to `bert_config.json`. The file captures every pipeline setting in a single artifact that you can version-control and share. A representative excerpt looks like this:

```json
{
  "loader": {
    "task": "text-classification",
    "model_class": "AutoModelForSequenceClassification",
    "model_type": "bert"
  },
  "export": {
    "opset_version": 17,
    "batch_size": 1
    .. // truncated: input_tensors, output_tensors
  },
   "optim": {
    "clamp_constant_values": true
  },
  "quant": {
    "mode": "qdq",
    "weight_type": "uint8",
    "activation_type": "uint16",
    "samples": 10,
    "calibration_method": "minmax",
    "task": "text-classification",
    "model_id": "bert-base-uncased"
    ... // truncated: per_channel, symmetric, distribution, ...
  },
  "compile": null
}
```

!!! note
    The five top-level keys — `loader`, `export`, `optim`, `quant`, and `compile` — map directly to the five pipeline stages. Setting `quant` or `compile` to `null` skips that stage entirely. See [Config and build](../concepts/config-and-build.md) for a field-by-field description of every option.

## Step 2: Run the build

```bash
winml build -c bert_config.json -m bert-base-uncased --output-dir bert_out/
```

winml-cli reads the config, downloads the model weights once, and runs the pipeline in sequence. Terminal output shows each stage as it completes:

```text
winml build
  Config:     bert_config.json
  Model:      bert-base-uncased
  Output:     bert_out/

  export       done  (42.1s)
  optimize     done  (6.3s)
  quantize     done  (18.7s)
  compile      done  (21.4s)

  Build complete in 88.5s
  Final artifact: bert_out/model.onnx
```

!!! note
    After the optimize stage, winml-cli runs an analyzer loop that inspects the graph for nodes the target EP cannot dispatch natively and re-runs optimization with adjusted fusion flags. The loop repeats up to `--max-optim-iterations` times (default: 3). Pass `--no-optimize` to skip this stage entirely when starting from a pre-optimized ONNX file. See [How winml-cli Works](../concepts/how-it-works.md) for a full description of the autoconf loop.

## Step 3: Benchmark

```bash
winml perf -m bert_out/model.onnx --iterations 50
```

After a short warm-up, `winml perf` reports latency percentiles and throughput:

```text
Device:      npu
Task:        text-classification
Iterations:  50 (+ 10 warmup)
Batch Size:  1

Latency (ms)
  Avg    P50    P90    P95    P99    Min    Max    Std
 4.83   4.79   5.12   5.31   5.68   4.51   6.04   0.21

Throughput: 206.99 samples/sec

Results saved to: model_perf.json
```

## Customizing the config

The JSON file is plain text and can be edited before running `winml build`. Two common adjustments:

**Change precision.** To target fp16 instead of the default uint8 QDQ quantization, regenerate the config with an explicit precision flag:

```bash
winml config -m bert-base-uncased -t text-classification --precision fp16 -o bert_config.json
```

Alternatively, edit `bert_config.json` directly: set `quant.weight_type` and `quant.activation_type` to `"int8"` or `"uint16"`, or set `quant` to `null` to skip quantization entirely.

**Disable a stage at build time.** You can suppress a stage for a single run without touching the config file using the `--no-quant` flags:

```bash
winml build -c bert_config.json -m bert-base-uncased --output-dir bert_out/ --no-quant 
```

This is useful for measuring the fp32 baseline before committing to a quantized build. The `quant` section in `bert_config.json` is unchanged; the flag only affects this invocation. See [Config and build](../concepts/config-and-build.md) for the full list of configurable fields.

## What you learned

- `winml config` generates a complete, version-controllable `WinMLBuildConfig` JSON from a HuggingFace model ID in one command.
- `winml build` orchestrates the full export → optimize → quantize → compile pipeline from a single config file and model ID.
- The autoconf loop inside the optimize stage adjusts graph fusion flags automatically to maximize EP compatibility.
- `winml perf` gives a latency and throughput baseline on the built artifact in seconds.


## See also

- [winml config](../commands/config.md)
- [winml build](../commands/build.md)
- [winml perf](../commands/perf.md)
- [Config and build](../concepts/config-and-build.md)
