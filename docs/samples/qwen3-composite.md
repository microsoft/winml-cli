# Qwen3 — Composite Models

!!! info "Coming soon"
    Composite-model support — running models with multiple components like a text encoder + decoder, or a vision encoder + LLM, through a single winml-cli pipeline — is on an in-progress feature branch. This page will be authored once that work merges.

## What composite models are

A composite model is a system made up of two or more distinct ONNX sub-models that work together as a single inference pipeline. A common example is a vision-language model like Qwen3-VL, where a vision encoder processes an image and feeds its output into a separate language model decoder. Another pattern is an encoder-decoder pair — two ONNX files that share a tokenizer configuration and must be executed in sequence at runtime. Multi-stage pipelines generalize this further: the output tensor of one sub-model becomes the input tensor of the next, with each stage potentially targeting a different execution provider or precision. Composite models add coordination complexity beyond what a single ONNX graph requires, so they call for first-class support in the build and inference tooling rather than ad hoc stitching.

## What Qwen3 will demonstrate

The following is a forward-looking sketch of what this sample will cover once the composite-model feature branch lands:

- How to declare a composite model in a `BuildConfig` — specifying multiple sub-models, their connection points, and a shared tokenizer configuration.
- How `winml build` orchestrates export and compilation of each sub-model independently, then assembles the composite pipeline.
- How to run end-to-end inference across the composite pipeline using a single `winml` invocation.
- How to benchmark each sub-model's latency independently with `winml perf` to identify bottlenecks.
- This section is a sketch and will be revised once the implementation lands; details may change.

## Track progress

Follow development and check current status at https://github.com/microsoft/winml-cli.

## See also

- [BERT — Config + Build + Perf](../samples/bert-config-build.md)
- [Config and build](../concepts/config-and-build.md)
