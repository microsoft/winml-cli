# Perf and monitoring

Knowing that a model produces correct outputs is necessary but not sufficient for a production deployment. You also need to know how fast it runs, how consistently it runs, and where the time goes when it does not run fast enough. `winml perf` is the primary tool in `winml-cli` for answering those questions. It synthesises end-to-end latency numbers, per-operator timings, and live hardware utilisation into a single benchmarking workflow.

Because `winml perf` accepts both HuggingFace model IDs and local `.onnx` files, you can benchmark at any stage of the development cycle — from a freshly exported float model through to a compiled, quantized production artifact.

## What perf measures

At its core, `winml perf` runs a configurable number of inference iterations and reports latency statistics. Here is a real example benchmarking `bert-tiny` on CPU:

```
$ winml perf -m bert-tiny.onnx --device cpu --iterations 50 --warmup 5

Device:      cpu / CPUExecutionProvider
Model Precision:   fp32
Inputs:      input_ids            [1, 512]               int32
             attention_mask       [1, 512]               int32
             token_type_ids       [1, 512]               int32
Outputs:     last_hidden_state    [1, 512, 128]

Latency (ms)
┏━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┓
┃  Avg ┃  P50 ┃  P90 ┃  P95 ┃  P99 ┃  Min ┃  Max ┃  Std ┃
┡━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━┩
│ 5.53 │ 5.40 │ 6.55 │ 6.87 │ 7.65 │ 4.89 │ 7.65 │ 0.58 │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
  Warmup: 14.14 ms avg (first 5 iterations)

Throughput: 180.72 samples/sec
```

Key parameters:

| Flag | Purpose | Default |
|------|---------|---------|
| `--iterations` | Number of benchmark iterations | 100 |
| `--warmup` | Warmup iterations excluded from statistics | 10 |
| `--batch-size` | Batch size for input generation | 1 |
| `-d, --device` | Target device: `auto`, `cpu`, `gpu`, `npu` | `auto` |
| `--ep` | Specific execution provider (e.g. `qnn`, `dml`, `openvino`) | auto-resolved from device |
| `--precision` | Precision mode: `auto`, `fp32`, `fp16`, `int8`, `int16`, or `w{x}a{y}` | `auto` |
| `--quantize/--no-quantize` | Include quantization during model build | `--quantize` |
| `--skip-build/--no-skip-build` | Skip the build pipeline for ONNX inputs | `--skip-build` |

### Output format

Add `-f json` to emit structured JSON to stdout, suitable for CI pipelines or automated comparisons:

```json
{
  "benchmark_info": {
    "model_id": "bert-tiny.onnx",
    "device": "cpu",
    "ep": "CPUExecutionProvider",
    "iterations": 50,
    "warmup": 5,
    "batch_size": 1
  },
  "latency_ms": {
    "avg": 5.53, "p50": 5.40, "p90": 6.55,
    "p95": 6.87, "p99": 7.65, "min": 4.89, "max": 7.65
  },
  "throughput": { "samples_per_sec": 180.72 },
  "raw_samples_ms": [5.12, 5.40, ...]
}
```

Results are also saved automatically to `~/.cache/winml/perf/<model_slug>/<timestamp>.json` for later comparison. Override the path with `--output`.

## Live monitoring

Latency numbers alone do not tell you whether the hardware is actually being used. A slow NPU inference could mean the model is running on the NPU and hitting a memory bottleneck, or it could mean the EP silently fell back to CPU and is not using the NPU at all.

The `--monitor` flag adds a live terminal chart (powered by plotext + Rich Live) that streams hardware utilisation for whichever device is being benchmarked. The chart auto-refreshes in a background thread so you can see whether utilisation is sustained, bursty, or absent. This is particularly useful when commissioning a new model on QNN or DirectML hardware, where EP fallback can be hard to detect from latency numbers alone. If the chart stays near zero while the benchmark runs, the model is not executing on the expected device.

```
winml perf -m model.onnx --device npu --monitor
```

`--monitor` has no effect on the measured latency statistics — it is a passive observer.

## Per-operator tracing

When end-to-end latency is higher than expected, per-operator tracing lets you find the operators that are responsible. This capability is available via a hidden `--op-tracing` flag (not shown in `--help`) and requires `onnxruntime-qnn` to be installed.

Two levels are available:

- **`--op-tracing basic`** — collects cumulative time per operator type and reports a ranked list. Usually enough to identify whether a sequence of Attention nodes or a large MatMul is dominating the runtime.
- **`--op-tracing detail`** — collects timing for every individual operator node in the graph. Useful when the same operator type appears in different parts of the model with very different costs.

```
winml perf -m model.onnx --op-tracing basic
```

!!! note
    Op-tracing currently works only with the QNN execution provider. Running it on CPU or DML will produce an error indicating the requirement.

## Per-module benchmarking

Large Transformer-family models contain many repeated module instances — attention blocks, feed-forward layers, encoder stages. When you want to understand the cost of one type of block rather than the full network, `--module <ClassName>` isolates and benchmarks matching modules from the HuggingFace model hierarchy.

```
winml perf -m bert-base-uncased --module BertAttention
```

This builds and benchmarks each `BertAttention` instance separately and reports per-instance statistics. The `--module` argument must be a **class name** (e.g. `BertAttention`), not a dotted module path (e.g. not `encoder.layer.0.attention`).

The module hierarchy that `--module` navigates is built at export time: every ONNX node carries a `winml.hierarchy.tag` metadata entry recording the PyTorch module path it came from. `winml perf --module` matches against those tags, builds a separate ONNX for each match, and benchmarks them in isolation. See [Load and export](load-and-export.md) for how the metadata is written.

## See also

- [Load and export](load-and-export.md) — how the module-tree metadata that `--module` targets gets written
- [Eval and datasets](eval-and-datasets.md) — accuracy measurement to pair with performance numbers
- [perf command reference](../commands/perf.md)
