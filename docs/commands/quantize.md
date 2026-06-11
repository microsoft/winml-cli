# winml quantize

> Quantize an ONNX model with QDQ insertion and calibration-based scaling.

## When to use this

Use `winml quantize` after `winml export` to insert
QuantizeLinear/DequantizeLinear (QDQ) node pairs into an ONNX graph. The
resulting model is ready for `winml compile` targeting an NPU or other
quantization-aware execution provider.

## Synopsis

```bash
$ winml quantize [options]
```

## Flags

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--model` | `-m` | path | *(required)* | Input ONNX model file. |
| `--output` | `-o` | path | `{input}_qdq.onnx` | Output path for the quantized model. |
| `--task` | | string | — | Task name (e.g., `image-classification`, `text-classification`) used to select a task-appropriate calibration dataset. Pair with `--model-name` so the dataset is preprocessed exactly the way the model expects. Without `--task`, calibration falls back to synthetic random data. |
| `--model-name` | | string | — | HuggingFace model ID (e.g., `microsoft/resnet-50`) used to load the matching preprocessor/tokenizer for calibration. Only used when `--task` is provided. |
| `--precision` | `-p` | string | `None` | Precision shorthand: `int8`, `int16`, or mixed-precision like `w8a16`. Overridden by explicit `--weight-type` / `--activation-type`. |
| `--samples` | | integer | `10` | Number of calibration samples used to compute quantization ranges. |
| `--method` | | choice | `minmax` | Calibration algorithm: `minmax`, `entropy`, or `percentile`. |
| `--weight-type` | | choice | `uint8` | Per-tensor type for weights: `uint8`, `int8`, `uint16`, or `int16`. Overrides `--precision`. When unset, the effective type comes from `--precision`, or `uint8` if neither is set. |
| `--activation-type` | | choice | `uint8` | Per-tensor type for activations: `uint8`, `int8`, `uint16`, or `int16`. Overrides `--precision`. When unset, the effective type comes from `--precision`, or `uint8` if neither is set. |
| `--per-channel/--no-per-channel` | | flag | `false` | Apply per-channel (rather than per-tensor) quantization to weight tensors. |
| `--symmetric/--no-symmetric` | | flag | `false` | Use symmetric quantization (zero-point fixed at 0). |
| `--help` | `-h` | flag | | Show this message and exit. |

## How it works

`winml quantize` applies static post-training quantization (PTQ) using the
ONNX Runtime quantization API. Calibration passes collect activation range
statistics, which are used to compute scale and zero-point values baked into
`QuantizeLinear` / `DequantizeLinear` node pairs around each eligible operator.
The `--method` flag controls range estimation: `minmax` uses global observed
extremes, `entropy` minimizes KL-divergence, and `percentile` clips outliers.
Precision can be set at a coarse level with `--precision` or tuned per tensor
type with `--weight-type` and `--activation-type`; explicit type flags always
override `--precision`.

Calibration data is selected from `--task` and `--model-name`. For a supported
task, a built-in default calibration dataset is loaded and preprocessed through
the model's own tokenizer or image processor, so the calibration tensors match
what the model will see at inference time. For an unsupported task — or when
`--task` is omitted entirely — calibration falls back to synthetic random data
synthesized from the ONNX input specification. Random-data calibration is fast
and always works, but the resulting scales are typically less accurate than
dataset-driven calibration, so always provide `--task` and `--model-name` when
the model task is supported.

## Examples

```bash
# Minimal quantization: defaults (10 samples, uint8 weights and activations)
winml quantize -m resnet50.onnx
```

```text
Input: resnet50.onnx
Output: resnet50_qdq.onnx
Weight type: uint8
Activation type: uint8
Samples: 10
Method: minmax

Running quantization...

Success! Model quantized
Output: resnet50_qdq.onnx
QDQ nodes inserted: 53
Total time: 4.31s
```

```bash
# Task-aware calibration: real samples preprocessed through the model's own image processor
winml quantize -m resnet50.onnx --task image-classification --model-name microsoft/resnet-50 --samples 128
```

```bash
# int8 precision shorthand (equivalent to --weight-type int8 --activation-type int8)
winml quantize -m resnet50.onnx -p int8
```

```bash
# Mixed-precision: int8 weights, uint16 activations with entropy calibration
winml quantize -m bert-base-uncased.onnx --weight-type int8 --activation-type uint16 --method entropy --samples 64
```

```bash
# Per-channel symmetric quantization to a specific output path
winml quantize -m facebook_convnext.onnx -o facebook_convnext_qdq.onnx --per-channel --symmetric --samples 32
```

```bash
# int16 precision (suitable for models sensitive to int8 accuracy loss)
winml quantize -m bert-base-uncased.onnx --precision int16
```

## Common pitfalls

- **Calibration uses synthetic random data by default.** Without `--task` and `--model-name`, scales and zero-points are computed from random tensors synthesized from the ONNX input specification — the model never sees realistic activations, so accuracy after quantization can degrade noticeably. Always pass `--task` and `--model-name` for supported tasks (e.g., `--task image-classification --model-name microsoft/resnet-50`) so calibration runs on real samples preprocessed through the model's own tokenizer or image processor.
- **`--weight-type` / `--activation-type` silently override `--precision`.** If you pass both, the explicit type flags win. Omit `--precision` when setting types explicitly to avoid confusion.
- **Low sample counts can hurt accuracy.** The default of 10 samples is sufficient for quick testing, but production models typically need 64–256 representative samples for good calibration.
- **`--per-channel` increases model size.** Per-channel quantization stores a separate scale and zero-point per output channel; this can noticeably inflate the model file size compared to per-tensor mode.
- **Output defaults to `{stem}_qdq.onnx` in the same directory as input.** Always pass `-o` when writing to a specific location to avoid accidentally overwriting or cluttering the source directory.
- **Quantizing an already-quantized model (one containing QDQ nodes) is unsupported and will produce incorrect results.** Use `winml compile --no-quant` instead if the model already contains QDQ nodes.

## See also

- [winml export](export.md)
- [winml compile](compile.md)
- [winml build](build.md)
- [Quantization concepts](../concepts/quantization.md)
