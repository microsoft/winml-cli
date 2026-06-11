# Perf and monitoring

Knowing that a model produces correct outputs is necessary but not sufficient for a production deployment. You also need to know how fast it runs, how consistently it runs, and where the time goes when it does not run fast enough. `winml perf` is the primary tool in `winml-cli` for answering those questions. It synthesises end-to-end latency numbers and live hardware utilisation into a single benchmarking workflow.

Because `winml perf` accepts both HuggingFace model IDs and local `.onnx` files, you can benchmark at any stage of the development cycle — from a freshly exported float model through to a compiled, quantized production artifact.

## What perf measures

At its core, `winml perf` runs a configurable number of inference iterations and reports latency statistics. Here is a real example benchmarking `bert-tiny` on CPU:

```
$ winml perf -m bert-tiny.onnx --device cpu --iterations 50 --warmup 5

Device:      cpu / CPUExecutionProvider
Task:        auto (auto-detected)
Model Precision:   fp32
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
    "task": "auto-detected",
    "device": "cpu",
    "ep": "CPUExecutionProvider",
    "precision": "auto",
    "iterations": 50,
    "warmup": 5,
    "batch_size": 1,
    "timestamp": "2026-06-11T03:27:24+00:00"
  },
  "model_info": {
    "input_names": ["input_ids", "attention_mask", "token_type_ids"],
    "input_shapes": [[1, 512], [1, 512], [1, 512]],
    "input_types": ["int32", "int32", "int32"],
    "output_names": ["last_hidden_state"],
    "output_shapes": [[1, 512, 128]]
  },
  "latency_ms": {
    "mean": 5.53, "p50": 5.40, "p90": 6.55,
    "p95": 6.87, "p99": 7.65, "min": 4.89, "max": 7.65,
    "std": 0.58, "warmup_mean": 14.14
  },
  "throughput": { "samples_per_sec": 180.72, "batches_per_sec": 180.72 },
  "raw_samples_ms": [5.12, 5.40, ...]
}
```

Results are also saved automatically to `~/.cache/winml/perf/<model_slug>/<timestamp>.json` for later comparison. Override the path with `--output`.

## Live monitoring

Latency numbers alone do not tell you whether the hardware is actually being used. A slow NPU inference could mean the model is running on the NPU and hitting a memory bottleneck, or it could mean the EP silently fell back to CPU and is not using the NPU at all.

The `--monitor` flag adds a live terminal chart (powered by plotext + Rich Live) that streams hardware utilisation for whichever device is being benchmarked. The chart updates once per iteration so you can see whether utilisation is sustained, bursty, or absent. This is particularly useful when commissioning a new model on QNN or DirectML hardware, where EP fallback can be hard to detect from latency numbers alone. If the chart stays near zero while the benchmark runs, it is a strong signal that the model may not be executing on the expected device — investigate further with EP-specific tools.

```
winml perf -m model.onnx --device npu --monitor
```

Display updates are not included in the timed inference call, but monitoring may introduce small system overhead from background PDH polling.

## Memory and resource metrics

When `--monitor` is active, hardware metrics are sampled throughout the benchmark and reported at the end. These metrics help answer questions like "how much device memory does this model need?" and "is the model memory-bound?".

The metrics collected depend on the target device:

| Metric | CPU | GPU | NPU |
|--------|:---:|:---:|:---:|
| CPU utilisation (mean/peak %) | ✓ | ✓ | ✓ |
| RAM (used MB, peak MB) | ✓ | ✓ | ✓ |
| Device utilisation (mean/peak %) | — | ✓ | ✓ |
| Device memory local (peak MB) | — | ✓ | ✓ |
| Device memory shared (peak MB) | — | ✓ | ✓ |
| Engine running time (ns) | — | ✓ | ✓ |

- **CPU**: Only system-level metrics (CPU %, RAM) are shown in terminal output. In JSON, `device_memory` and `running_time_ns` are still present but will be zero.
- **GPU**: Reports GPU engine utilisation plus dedicated VRAM (`local_peak_mb`) and shared system memory (`shared_peak_mb`) allocated by the GPU driver.
- **NPU**: Same structure as GPU. NPU adapters register as Windows GPU Engine devices, so utilisation and memory are read via the same PDH counters. `local_peak_mb` represents dedicated adapter memory; `shared_peak_mb` is system memory shared with the NPU.

### Terminal output

CPU device:

```
Hardware (during benchmark)
  CPU: 8.3% avg  |  Mem: 644 MB
```

NPU or GPU device:

```
Hardware (during benchmark)
  NPU: 87.3% avg, 100.0% peak  |  CPU: 12.1% avg  |  Mem: 1842 MB
  Device Mem: 245/0 MB (local/shared)
```

### JSON structure

In JSON output (`-f json`), these metrics appear under the `hw_monitor` key:

```json
"hw_monitor": {
  "monitor": "HWMonitor",
  "device_kind": null,
  "adapter_luid": null,
  "cpu": { "mean_pct": 15.8, "peak_pct": 16.71, "sample_count": 2 },
  "ram": { "used_mb": 640.21, "peak_mb": 640.21 },
  "device_memory": { "local_peak_mb": 0.0, "shared_peak_mb": 0.0 },
  "running_time_ns": 0
}
```

When a hardware accelerator is active, `device_kind` will be `"npu"` or `"gpu"`, and an additional key (e.g. `"npu"`) appears with device utilisation:

```json
"hw_monitor": {
  "monitor": "HWMonitor",
  "device_kind": "npu",
  "adapter_luid": "0x0000abcd12340000",
  "cpu": { "mean_pct": 12.1, "peak_pct": 34.5, "sample_count": 50 },
  "ram": { "used_mb": 1842.0, "peak_mb": 1910.0 },
  "device_memory": { "local_peak_mb": 245.0, "shared_peak_mb": 0.0 },
  "npu": { "mean_pct": 87.3, "peak_pct": 100.0, "sample_count": 50 },
  "running_time_ns": 4820000000
}
```

This makes it straightforward to track memory consumption across model revisions or compare devices programmatically.

## Per-module benchmarking

Large Transformer-family models contain many repeated module instances — attention blocks, feed-forward layers, encoder stages. When you want to understand the cost of one type of block rather than the full network, `--module <ClassName>` isolates and benchmarks matching modules from the HuggingFace model hierarchy.

```
winml perf -m bert-base-uncased --module BertAttention
```

This builds and benchmarks each `BertAttention` instance separately and reports per-instance statistics. The `--module` argument must be a **class name** (e.g. `BertAttention`), not a dotted module path (e.g. not `encoder.layer.0.attention`).

Internally, `--module` uses `torchinfo` to discover all submodule instances matching the given class name in the HuggingFace model. For each match it generates a separate build config, exports an isolated ONNX file, and benchmarks it independently. This requires a HuggingFace model ID (not a local `.onnx` file) because it needs access to the PyTorch module tree.

## See also

- [Load and export](load-and-export.md) — how the module-tree metadata that `--module` targets gets written
- [Eval and datasets](eval-and-datasets.md) — accuracy measurement to pair with performance numbers
- [perf command reference](../commands/perf.md)
