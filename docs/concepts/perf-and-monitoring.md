# Perf and monitoring

Knowing that a model produces correct outputs is necessary but not sufficient for a production deployment. You also need to know how fast it runs, how consistently it runs, and where the time goes when it does not run fast enough. `winml perf` is the primary tool in `winml-cli` for answering those questions. It synthesises end-to-end latency numbers, per-operator timings, and live hardware utilisation into a single benchmarking workflow.

Because `winml perf` accepts both HuggingFace model IDs and local `.onnx` files, you can benchmark at any stage of the development cycle — from a freshly exported float model through to a compiled, quantized production artifact.

## What perf measures

At its core, `winml perf` runs a configurable number of inference iterations and reports latency statistics. Here is a real example benchmarking `bert-tiny` on CPU:

```
$ winml perf -m bert-tiny.onnx --device cpu --iterations 50 --warmup 5

Device:      cpu / CPUExecutionProvider
Precision:   fp32
Inputs:      input_ids            [1, 512]    int32
             attention_mask       [1, 512]    int32
             token_type_ids       [1, 512]    int32
Outputs:     last_hidden_state    [1, 512, 128]
```

Output latency table:

| Avg | P50 | P90 | P95 | P99 | Min | Max | Std |
|-----|-----|-----|-----|-----|-----|-----|-----|
| 5.53 | 5.40 | 6.55 | 6.87 | 7.65 | 4.89 | 7.65 | 0.58 |

```
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

## Memory and resource metrics

When `--monitor` is active, hardware metrics are sampled throughout the benchmark and reported at the end. These metrics help answer questions like "how much device memory does this model need?" and "is the model memory-bound?".

| Category | Metrics | Description |
|----------|---------|-------------|
| **Device memory (local)** | Peak dedicated MB | VRAM or on-device memory exclusively allocated to the inference workload |
| **Device memory (shared)** | Peak shared MB | System memory shared with the device (common on integrated GPUs and NPUs) |
| **RAM** | Used MB, peak used MB | Process-level system memory consumption |
| **CPU** | Mean %, peak % | CPU utilisation during the benchmark window |
| **Device utilisation** | Mean %, peak % | NPU or GPU engine utilisation (hardware-reported) |

Example output (NPU device):

```
Hardware (during benchmark)
  NPU: 87.3% avg, 100.0% peak  |  CPU: 12.1% avg  |  Mem: 1842 MB
  Device Mem: 245/0 MB (local/shared)
```

In JSON output (`-f json`), these metrics appear under the `hw_monitor` key:

```json
"hw_monitor": {
  "device_kind": "npu",
  "device_memory": { "local_peak_mb": 245, "shared_peak_mb": 0 },
  "cpu": { "mean_pct": 12.1, "peak_pct": 34.5 },
  "ram": { "used_mb": 1842, "peak_used_mb": 1910 },
  "npu": { "mean_pct": 87.3, "peak_pct": 100.0 }
}
```

This makes it straightforward to track memory consumption across model revisions or compare devices programmatically.

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
