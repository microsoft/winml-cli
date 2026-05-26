# Datatype and Quantization

Every ONNX tensor carries data in a specific numeric type — `float32`, `float16`, `int8`, `int16` — and every winml-cli pipeline makes deliberate choices about which type to use where. This page covers both halves of that decision: the **datatype family** winml-cli understands, and the **quantization** workflow that converts a model from one datatype to another to shrink it and run it faster on integer-native hardware.

Quantization is the headline use of datatypes in winml-cli. By replacing `float32` weights and activations with `int8` or mixed precisions, you typically get a 2–4× smaller model artifact and a 2–8× latency speedup on NPU hardware. The trade-off is a potential reduction in model accuracy, the degree of which depends on the precision chosen and the sensitivity of the model.

## Datatypes

winml-cli exposes a precision shorthand on the `--precision` flag that encodes the weight/activation dtype pair as a single string. The table below lists every precision from `_KNOWN_PRECISIONS` in `_options.py`, together with the resolved quantization types from `config/precision.py`. Float precisions (`fp32`, `fp16`) carry no quantization types because weights and activations remain in floating point throughout.

| Precision | Weight dtype | Activation dtype | Notes |
|-----------|-------------|-----------------|-------|
| `auto` | device-dependent | device-dependent | Resolves to `int8` (NPU), `fp16` (GPU/CPU) at runtime |
| `fp32` | float32 | float32 | No quantization; baseline accuracy |
| `fp16` | float16 | float16 | Half-precision float; no QDQ nodes inserted |
| `int8` | uint8 | uint8 | Static quantization; default for NPU via QNN EP |
| `int16` | int16 | uint16 | Higher-accuracy quantization; larger model than int8 |
| `w8a8` | uint8 | uint8 | Equivalent to `int8`; explicit mixed-precision notation |
| `w8a16` | uint8 | uint16 | Mixed: compact weights, wider activations for accuracy |
| `w4a16` | n/a | n/a | **Planned — not yet supported.** Recognized as a precision string but raises an error at quantization time; no 4-bit weight dtype mapping exists in `precision.py` yet. |

The `--weight-type` and `--activation-type` flags on `winml quantize` accept `uint8`, `int8`, `uint16`, or `int16` and override whatever the `--precision` shorthand would have resolved. This is useful when you need an unsigned weight type for QNN compatibility but a signed activation type for a specific operator constraint. See [Weight and Activation](weight-and-activation.md) for why the two need separate flags in the first place.

## How quantization works in winml-cli

winml-cli applies quantization by inserting **QDQ** (Quantize/Dequantize) nodes into the ONNX graph. The resulting file is a standard ONNX model that any ONNX Runtime execution provider can consume and optimize for its target hardware — the EP reads the QDQ pattern and fuses adjacent operations into true integer kernels.

### Calibration

Static quantization — the kind winml-cli applies — requires a calibration pass before inserting QDQ nodes. During calibration, a small set of representative inputs runs through the original floating-point model so that winml-cli can observe the actual range of values each tensor takes at runtime. Those observed ranges are then used to choose the scale and zero-point constants baked into the QDQ nodes.

The `--samples` flag controls how many calibration inputs are used (default: `10`). More samples generally produce better range estimates but take longer. The `--method` flag selects the algorithm used to summarize the observed ranges:

- `minmax` (default) — uses the absolute minimum and maximum observed values. Fast and predictable; can be sensitive to outliers.
- `entropy` — minimizes the KL-divergence between the original and quantized distribution. Often yields better accuracy on models with heavy-tailed activation distributions.
- `percentile` — clips a small fraction of extreme values before computing the range. A practical middle ground when outliers are present but entropy calibration is slow.

Example using entropy calibration with more samples:

```bash
winml quantize -m model.onnx --precision int8 --samples 128 --method entropy
```

### The QDQ pattern

The QDQ pattern is the standard ONNX representation for static quantization. winml-cli wraps the inputs and outputs of quantizable operators with pairs of `QuantizeLinear` and `DequantizeLinear` nodes. At the graph level the model still operates in floating-point; the QDQ nodes encode the scale and zero-point metadata that a runtime needs to fuse adjacent operations into true integer kernels.

When the model runs under ONNX Runtime, the execution provider — whether CPU, DirectML, or a dedicated NPU EP — reads those QDQ patterns and performs its own graph fusion. This means the EP is free to apply hardware-specific optimizations without winml-cli needing to know anything about the target device's internal ISA or operator library. The QDQ model produced by `winml quantize` is a single portable artifact that can be deployed to any EP that supports integer execution.

## When quantization is lossy

Not all precision choices carry equal accuracy risk:

- `fp16` is usually lossless in practice. Rounding errors relative to `fp32` are small enough that most models show no measurable accuracy difference.
- `int8` and `int16` are inherently lossy. Compressing a 32-bit float into 8 or 16 bits discards information, and the magnitude of accuracy degradation depends on how well the calibration data represents the deployment distribution.
- Compound precisions like `w8a16` reduce the risk compared to full `int8` by preserving more precision in activations, but they are still lossy relative to `fp32`.

Always validate accuracy after quantizing an integer-precision model. Run `winml eval` on a representative dataset and compare the metrics against the original floating-point baseline before shipping the quantized artifact.

## See also

- [Weight and Activation](weight-and-activation.md)
- [EP and Device](eps-and-devices.md)
- [quantize command reference](../commands/quantize.md)
- [eval command reference](../commands/eval.md)
