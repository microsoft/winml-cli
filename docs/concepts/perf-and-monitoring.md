# Perf and monitoring

Knowing that a model produces correct outputs is necessary but not sufficient for a production deployment. You also need to know how fast it runs, how consistently it runs, and where the time goes when it does not run fast enough. `winml perf` is the primary tool in `winml-cli` for answering those questions. It synthesises end-to-end latency numbers, per-operator timings, and live hardware utilisation into a single benchmarking workflow.

Because `winml perf` accepts both HuggingFace model IDs and local `.onnx` files, you can benchmark at any stage of the development cycle — from a freshly exported float model through to a compiled, quantized production artifact.

## What perf measures

At its core, `winml perf` runs a configurable number of inference iterations and reports latency statistics: p50, p90, and mean latency in milliseconds, plus throughput in inferences per second. Warmup iterations (controlled by `--warmup`, defaulting to 10) are excluded from the statistics so that JIT and cache effects do not skew the numbers.

You can control the run length with `--iterations` and the input shape with `--batch-size` or a `--shape-config` JSON file for models with dynamic axes. The `--device` flag selects the target EP — `cpu`, `gpu`, `npu`, or `auto` (default) — allowing you to collect numbers on each target with the same command and compare them directly. For fine-grained EP control, `--ep` lets you name a specific provider such as `qnn` or `dml`.

The results are written to a JSON file at `~/.cache/winml/perf/<slug>/<timestamp>.json` (or a custom path via `--output`) so they can be archived and compared across builds.

## Live monitoring

Latency numbers alone do not tell you whether the hardware is actually being used. A slow NPU inference could mean the model is running on the NPU and hitting a memory bottleneck, or it could mean the EP silently fell back to CPU and is not using the NPU at all.

The `--monitor` flag adds a live terminal chart that streams hardware utilisation for whichever device is being benchmarked. The chart updates in place during the iteration loop so you can see whether utilisation is sustained, bursty, or absent. This is particularly useful when commissioning a new model on QNN or DirectML hardware, where EP fallback can be hard to detect from latency numbers alone. If the chart stays near zero while the benchmark runs, the model is not executing on the expected device.

`--monitor` has no effect on the measured latency statistics — it is a passive observer.

## Per-operator tracing

When end-to-end latency is higher than expected, per-operator tracing lets you find the operators that are responsible. This capability is available via a hidden `--op-tracing` flag (not shown in `--help`) intended for advanced diagnostics. Two levels are available:

`--op-tracing basic` collects cumulative time per operator type and reports a ranked list. This is usually enough to identify whether, say, a sequence of Attention nodes or a large MatMul is dominating the runtime.

`--op-tracing detail` goes further, collecting timing for every individual operator node in the graph. This is useful when the same operator type appears in different parts of the model with very different costs — for instance, early-layer convolutions versus late-layer convolutions in a ResNet-style architecture.

If tracing is unavailable, `winml-cli` will tell you at startup rather than silently running without tracing.

## Per-module benchmarking

Large Transformer-family models contain many repeated module instances — attention blocks, feed-forward layers, encoder stages. When you want to understand the cost of one type of block rather than the full network, `--module <substring>` isolates and benchmarks matching modules from the HuggingFace model hierarchy.

`winml perf -m bert-base-uncased --module BertAttention`, for example, builds and benchmarks each `BertAttention` instance separately and reports per-instance statistics. This is faster to iterate on than benchmarking the full model when you are tuning a specific layer, and it makes the attribution of latency to architectural decisions much clearer.

The module hierarchy that `--module` navigates is built at export time: every ONNX node carries a `winml.hierarchy.tag` metadata entry recording the PyTorch module path it came from. `winml perf --module` matches against those tags, builds a separate ONNX for each match, and benchmarks them in isolation. See `winml inspect --hierarchy` to view the tree for an exported model, or [Load and export](load-and-export.md) for how the metadata is written.

## See also

- [Load and export](load-and-export.md) — how the module-tree metadata that `--module` targets gets written
- [Eval and datasets](eval-and-datasets.md) — accuracy measurement to pair with performance numbers
- [perf command reference](../commands/perf.md)
