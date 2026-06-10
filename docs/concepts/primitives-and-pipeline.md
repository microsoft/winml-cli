# Primitives and pipeline

winml-cli exposes two ways to turn a Hugging Face model or ONNX file into a
Windows ML-ready artifact. You can invoke each stage of the pipeline as an
individual primitive command — `winml export`, `winml analyze`, `winml optimize`,
`winml quantize`, `winml compile`, `winml perf`, `winml eval` — running one step
at a time with full control over inputs and outputs. Alternatively, `winml build`
wraps all of those stages into a single command driven by a `WinMLBuildConfig`
JSON file.

Understanding when to reach for a primitive versus the pipeline wrapper is the
central workflow decision in winml-cli. Both paths produce the same artifacts;
the difference is in repeatability, convenience, and how much you need to inspect
or vary individual stages.

## The primitive commands

Each primitive command corresponds to one stage of the pipeline described in
[How winml-cli works](how-it-works.md). They run in order, each producing an ONNX
file that the next stage consumes:

- **`winml export`** — loads a Hugging Face model, traces it with PyTorch and the
  Optimum exporter, and writes a portable float32 ONNX file with no EP-specific
  nodes.
- **`winml analyze`** — runs compatibility and runtime checks on the exported ONNX
  graph, detecting unsupported operators, QDQ issues, and device-specific
  constraints before further pipeline stages.
- **`winml optimize`** — applies graph transformations (operator fusion, constant
  folding, graph pruning) and runs an autoconf loop to maximize EP-compatible
  coverage.
- **`winml quantize`** — inserts QDQ nodes using calibration data, reducing weight
  and activation types to lower precision (for example, int8) for efficient
  inference.
- **`winml compile`** — invokes an EP-specific compiler (for example, QNN for NPU
  targets) to embed a pre-compiled binary cache in the ONNX graph as an EPContext
  node.
- **`winml perf`** — benchmarks latency and throughput against a Windows ML
  session; does not modify the model.
- **`winml eval`** — evaluates task-specific accuracy on a dataset; does not
  modify the model.

You can enter the pipeline at any stage. If you already have an optimized ONNX
file, pass it directly to `winml quantize` without re-exporting. Each command
writes its output to a path you specify, so all intermediate artifacts are
preserved for inspection.

## The pipeline wrapper

`winml build` orchestrates all of the above stages in order from a single
`WinMLBuildConfig` JSON file:

```bash
winml build -c config.json -m microsoft/resnet-50 -o output/
```

The config file tells `winml build` which stages to run and how to configure them.
Setting the `quant` or `compile` section to `null` in the JSON skips that stage;
passing `--no-quant`, `--no-compile`, or `--no-optimize` on the command line
achieves the same effect at runtime without editing the file.

When the model argument points to an existing ONNX file instead of a Hugging Face
ID, `winml build` detects this and skips the export stage, running
analyze → optimize → quantize → compile directly. This mirrors how each primitive
command handles the same case.

`winml build` also accepts `--use-cache` in place of `-o`/`--output-dir`, routing
artifacts to the winml-cli global cache at `~/.cache/winml/` instead of a local
directory. Use `--rebuild` to force a clean re-run even when cached artifacts
already exist.

## When to choose which

**Use primitive commands when:**

- You are learning the pipeline and want to observe each stage's output in
  isolation.
- You are debugging a specific stage — for example, inspecting the optimized graph
  before quantization, or testing a quantized model before compiling it.
- You need a one-off variation that does not warrant a versioned config, such as
  trying a different opset or a different calibration sample count.
- You are integrating winml-cli output into a larger script that already manages
  intermediate files.

**Use `winml build` when:**

- You are targeting production or CI: a single config file captures the full
  pipeline reproducibly and can be committed alongside the code that uses the
  model.
- You want to share the exact build recipe with a teammate or reproduce it later
  without reconstructing the sequence of primitive flags.
- You need the autoconf loop to propagate optimization decisions across stages,
  which only `winml build` coordinates end-to-end.
- You want stage-skipping to be declarative (`quant: null` in the config) rather
  than remembered flag-by-flag across invocations.

The two approaches are not exclusive. A common pattern is to prototype with
primitives — iterating on `winml optimize` and `winml quantize` individually to
tune fusion flags and calibration — and then encode the final settings into a
`WinMLBuildConfig` for repeatable production builds via `winml build`.

## See also

- [How winml-cli works](how-it-works.md) — pipeline stage order and internal
  architecture
- [Config and build](config-and-build.md) — generating and versioning a
  `WinMLBuildConfig`
- [winml build command reference](../commands/build.md)
- [ConvNeXT primitives sample](../samples/convnext-primitives.md) — worked example
  using primitive commands end-to-end
