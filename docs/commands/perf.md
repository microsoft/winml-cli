# winml perf

> Benchmark an ONNX model's latency and throughput on a target device.

## When to use this

Use `winml perf` when you want a quantitative latency and throughput baseline for a model on a specific device, or when you need to compare the performance impact of different precision settings, execution providers, or batch sizes.

## Synopsis

```bash
$ winml perf [options]
```

## Flags

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--model` | `-m` | `TEXT` | — | HuggingFace model ID or path to a local `.onnx` file. Required. |
| `--task` | | `TEXT` | auto-detected | Explicit task override (e.g., `image-classification`). Inferred from the model if omitted. |
| `--iterations` | | `INTEGER` | `100` | Number of timed inference iterations used to compute statistics. |
| `--warmup` | | `INTEGER` | `10` | Number of warm-up iterations run before timing begins; excluded from statistics. |
| `--device` | `-d` | `auto\|cpu\|gpu\|npu` | `auto` | Device to run the benchmark on. `auto` selects the highest-priority available device. |
| `--precision` | | `TEXT` | `auto` | Precision mode applied during model build: `auto`, `fp32`, `fp16`, `int8`, `int16`, or compound forms such as `w8a16`. |
| `--ep` | | `TEXT` | — | Force a specific execution provider (e.g., `qnn`, `dml`, `vitisai`, `openvino`, `cpu`). Overrides the device-to-provider mapping. |
| `--ep-options` | | `KEY=VALUE` (multiple) | — | Runtime EP provider option forwarded to the inference session (e.g., `--ep-options htp_performance_mode=burst`). Repeatable. Applies to both HuggingFace model IDs and ONNX file inputs. Unlike build-time options set via `--config`, these tune the runtime session, not the compiled graph. |
| `--output` | `-o` | `PATH` | `~/.cache/winml/perf/<slug>/<timestamp>.json` | Output JSON file path for the benchmark report. |
| `--batch-size` | | `INTEGER` | `1` | Batch size used when generating synthetic input tensors. |
| `--shape-config` | | `PATH` | — | Path to a JSON file containing shape overrides (e.g., `{"height": 480, "width": 480}`). Ignored for pre-exported ONNX files and in `--module` mode. |
| `--quantize/--no-quantize` | | flag | `true` | Run quantization during model build (use `--no-quantize` to skip it). Useful for measuring the fp32 baseline. |
| `--rebuild/--no-rebuild` | | flag | `false` | Force model rebuild even if a cached artifact already exists. |
| `--ignore-cache/--no-ignore-cache` | | flag | `false` | Build from scratch in a temporary folder and discard the artifact after benchmarking. Implies `--rebuild`. |
| `--module` | | `TEXT` | — | PyTorch module class name for per-module benchmarking (e.g., `BertAttention`). Builds and times each matching instance separately. See [Load and export](../concepts/load-and-export.md). |
| `--monitor/--no-monitor` | | flag | `false` | Show a live NPU/CPU utilization chart while the benchmark runs and include hardware metrics in the JSON report. |

## How it works

`winml perf` loads the model through `WinMLAutoModel` — accepting both HuggingFace IDs and local ONNX files — then generates random input tensors from the model's I/O configuration. It runs the specified number of warm-up iterations (excluded from statistics) followed by the timed iterations, collecting per-sample latency. The final report includes mean, min, max, P50, P90, P95, P99, standard deviation, and throughput in samples per second. When `--monitor` is active, a hardware polling loop runs in parallel and records NPU / GPU utilization, CPU usage, and device memory alongside the timing data.

## Examples

Basic benchmark on the best available device:

```bash
$ winml perf -m microsoft/resnet-50
```

```text
Device:      npu
Precision:   auto
Task:        image-classification
Iterations:  100 (+ 10 warmup)
Batch Size:  1

Latency (ms)
  Avg    P50    P90    P95    P99    Min    Max    Std
 2.14   2.11   2.38   2.51   2.79   1.97   3.04   0.12

Throughput: 467.29 samples/sec

Results saved to: ~/.cache/winml/perf/microsoft_resnet-50/2026-05-27T120000.json
```

Benchmark a pre-exported ONNX file on CPU with more iterations:

```bash
$ winml perf -m model.onnx --device cpu --iterations 500
```

Benchmark a text model with an explicit task, targeting the NPU:

```bash
$ winml perf -m bert-base-uncased --task text-classification --device npu --precision w8a16
```

Benchmark with live hardware monitoring enabled:

```bash
$ winml perf -m microsoft/resnet-50 --device npu --monitor
```

Pass runtime EP provider options to tune the session (repeatable):

```bash
$ winml perf -m model.onnx --device npu \
    --ep-options htp_performance_mode=burst \
    --ep-options htp_graph_finalization_optimization_mode=3
```

Per-module benchmarking to find latency hot-spots across all attention blocks:

```bash
$ winml perf -m bert-base-uncased --module BertAttention --iterations 200
```

## Common pitfalls

- **Warm-up too low on NPU.** The first several inferences on an NPU EP can be significantly slower due to kernel compilation and caching. The default of 10 warm-up iterations is usually enough for vision models, but transformer models with many operators may need `--warmup 30` or higher to reach steady-state latency.
- **`--shape-config` is silently ignored in two cases.** It has no effect on pre-exported ONNX files (shapes are baked into the graph) and is ignored in `--module` mode. The command prints a warning in both situations.
- **Random inputs do not represent real data distributions.** Latency numbers are accurate, but memory access patterns may differ from production because the generated tensors are uniform random values. For memory-bandwidth-sensitive models this can understate real-world latency.
- **Cross-device comparison.** To compare performance across devices, run `winml perf` separately with different `--device` values and compare the resulting JSON reports.

## See also

- [winml eval](eval.md) — measure accuracy after benchmarking
- [winml build](build.md) — build the quantized artifact that `perf` benchmarks
- [Load and export concept](../concepts/load-and-export.md) — how `--module` per-instance benchmarking works
- [ONNX & Execution Providers](../concepts/eps-and-devices.md) — understand `--device` vs `--ep`
