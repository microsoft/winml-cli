# Compile and EPContext

When you run `winml compile`, you are not simply copying an ONNX file to a new location. You are asking an execution provider (EP) to transform the model into a form it can load and run directly, without repeating that transformation at every startup. Understanding what the compiler produces — and why — helps you decide when to compile, what output format to choose, and how to balance file size against runtime performance.

Compilation is an offline, one-time step. The artifact it creates is what you ship with your application and what `winml-cli` uses for benchmarking and evaluation.

## What compilation produces

For EPs that are fully integrated into ONNX Runtime — CPU, DirectML, and similar providers — the compile step writes a new `.onnx` file that the runtime loads directly. The ONNX graph has been prepared and, in some cases, partitioned so that the EP's session initializer has less work to do when the application starts.

For QNN-family EPs (the `--ep qnn` and `--ep vitisai` targets used for NPU inference), the compiler goes further. QNN takes the ONNX graph and produces a binary artifact — the **EP context blob** — that encodes the fully compiled, hardware-ready version of the network. This blob is then associated with the ONNX model file. On subsequent loads, the QNN EP reads the blob rather than re-compiling the graph, which makes session creation dramatically faster.

The default compiler backend is `ort` (ONNX Runtime).

## Embedded vs external EPContext

For QNN compilation, `winml-cli` gives you a choice of where the EP context blob lives. By default the blob is written as a sidecar `.bin` file alongside the `.onnx`. Passing `--embed` instead inlines the blob directly into the ONNX file.

**External (default):** The `.onnx` is small and human-inspectable; the heavy binary data lives in a separate file. You must keep the two files together — the ONNX stores a relative path back to the `.bin`. This layout is preferable for version control and for scenarios where you want to inspect or diff the model graph.

**Embedded (`--embed`):** Everything ships in a single `.onnx` file. Deployment is simpler because there is only one artifact to track. The trade-off is file size: the `.onnx` grows by the full size of the compiled context, and the file is no longer human-readable in the usual sense. Choose embedded when your deployment tooling expects a single model file, or when you want to minimize the chance of the sidecar being misplaced.

## Why pre-compile

The first time an ONNX Runtime session is created for a model on a hardware EP, the runtime must partition the graph, allocate buffers, and JIT-compile the operators. On an NPU this process can take several seconds. For applications with tight startup budgets — on-device inference in a UI flow, for example — that cold-start cost is often unacceptable.

A model produced by `winml compile` has already paid that cost. The EP context blob is the result of compilation, not its input. When the application loads the compiled model the EP reads the pre-built binary and the session is ready almost immediately. Shipping a compiled model is therefore the standard pattern for production deployments on QNN hardware.

If you are iterating on quantization settings or ONNX graphs and want to check whether the model compiles at all, pass an already-quantized (QDQ) model directly — `winml compile` compiles whatever ONNX file you supply and does not have a separate quantization pass to skip.

## Skipping validation

By default `winml compile` runs a validation pass after compilation finishes — it loads the compiled model into an inference session, feeds it dummy inputs (all-ones tensors), and checks that the outputs do not contain NaN or Inf values. This catches basic compilation failures early (e.g., the EP rejecting the graph or producing garbage outputs).

The `--no-validate` flag skips that pass. It is useful during rapid iteration when you only want to confirm that compilation succeeds without the overhead of a trial inference run.

## See also

- [EPs and devices](eps-and-devices.md) — execution provider selection and `--ep` / `--device` flags
- [Analyze and optimize](analyze-and-optimize.md) — graph-level analysis before compilation
- [compile command reference](../commands/compile.md)
- [build command reference](../commands/build.md)
