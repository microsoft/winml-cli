# Analyze and optimize

Not every ONNX graph runs efficiently on every execution provider. An operator that compiles cleanly on CPU may be unsupported on an NPU, and a correct graph may still leave performance on the table because adjacent operations were not fused. winml-cli separates the concern into two commands — `winml analyze` and `winml optimize` — that together form a graph-quality loop driven automatically by `winml build`.

## What analyze does

`winml analyze` performs static analysis on an ONNX file and reports how well it will run on a target EP. It checks operator coverage, runs shape inference to catch missing or inconsistent tensor shapes, and performs runtime checks that probe actual support on the local machine.

Specify a target EP with `--ep` (e.g., `--ep qnn` or `--ep openvino`) and a device with `--device` (CPU, GPU, or NPU). Omit `--ep` to analyze against all supported EPs. Results print to the console by default; add `--output results.json` to save the report as JSON for scripting or archiving.

Exit codes carry the verdict: zero is full support, one is partial support with unsupported operators, two is a configuration error. This makes `winml analyze` suitable as a CI gate. Pass `--information` (enabled by default) to include recommendations alongside each flagged operator. Use `--save-node unsupported` or `--save-node partial` to persist node lists for further work.

## What optimize does

`winml optimize` rewrites the ONNX graph by applying fusions and structural simplifications. Fusions such as GELU, LayerNorm, and MatMul+Add collapse multi-node sequences into single operators that EPs can map to efficient kernels. Layout transformations like the NHWC transformer rearrange tensor memory order to match GPU access patterns.

Every optimization is a named capability toggled via `--enable-<name>` and `--disable-<name>` flags. Run `--list-capabilities` to see all registered optimizations and their defaults. This granularity matters when a specific fusion breaks a downstream step or when you need an exact optimization profile for a given EP.

The pattern-rewrite family is a complementary mechanism: instead of folding nodes, rewrites replace one subgraph pattern with a structurally equivalent alternative. Run `--list-rewrites` to discover available families and their flag names. Flags follow the form `--enable-<source-slug>-<target-slug>`.

Use presets (`--preset transformer-optimized`, `--preset qnn-compatible`) as a starting point, and commit a specific combination to a `--config` file for reproducible builds.

## The analyzer/optimizer loop

A single optimize pass may create fusion opportunities that were not present before, and a freshly fused graph may surface new operator compatibility issues. This is why `winml build` runs analyze and optimize in an alternating loop rather than once each.

The loop repeats up to `--max-optim-iterations` rounds (default: three), which covers most transformer and vision architectures. Convergence is checked after each round; the loop exits early when the analysis result no longer improves. Use `--no-analyze` to skip the loop and run a single optimization pass — useful for deterministic rebuilds from a fixed ONNX checkpoint where the graph is already known good.

## See also

- [Compile and EPContext](compile-and-epcontext.md)
- [Primitives and pipeline](primitives-and-pipeline.md)
- [analyze command](../commands/analyze.md)
- [optimize command](../commands/optimize.md)
