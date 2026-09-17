# winml optimize

> Apply graph optimizations and fusions to an ONNX model to reduce node count and improve inference speed.

## When to use this

Use `winml optimize` after exporting an ONNX model and before quantization or compilation. Graph fusions reduce operator count, improve memory locality, and can make downstream quantization more accurate by presenting cleaner subgraphs to the calibration pass. It is also useful as a standalone step when you want to optimize a pre-exported ONNX file without running the full build pipeline.

## Synopsis

```bash
$ winml optimize [options]
```

## Flags

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--model` | `-m` | `PATH` | *(required unless listing)* | Input ONNX model file. Not required when `--list-capabilities` or `--list-rewrites` is used. |
| `--output` | `-o` | `PATH` | `{input}_opt.onnx` | Output path for the optimized model. Defaults to the input filename with `_opt` inserted before the extension. |
| `--config` | `-c` | `PATH` | *(none)* | YAML or JSON configuration file. Fields in the file override capability defaults; CLI flags override the file. |
| `--enable-ort-graph-optimization` / `--disable-ort-graph-optimization` | | flag | enabled | Run or skip ORTGraphPipe. Disabling it does not implicitly enable compatibility rewrites or disable other pipes. Config key: `ort-graph-optimization` (boolean). |
| `--verbose` | `-v` | flag | off | Enable verbose output. |
| `--list-capabilities` | `-l` | flag | off | Print all registered optimization capabilities grouped by category and exit. Add `--verbose` for descriptions and ORT names. |
| `--list-rewrites` | | flag | off | Print all available pattern-rewrite families with their source-to-target mappings and exit. |
| `--check-optim` | | flag | off | Analyze which optimizations would apply to `--model` and the nodes/constants they affect, then exit **without writing any output**. Probes every capability independently. |
| *(dynamic)* | | flag | *(per capability)* | Each registered capability generates a `--enable-<name>` / `--disable-<name>` pair. Run `--list-capabilities` to see the full current list. Examples: `--enable-gelu-fusion`, `--disable-constant-folding`. Pattern-rewrite flags follow the form `--enable-<source-slug>-<target-slug>`; run `--list-rewrites` to discover all names. |

### Configuration precedence

When multiple sources are provided, settings are resolved in this order (highest wins):

1. Explicit CLI flags (`--enable-X` / `--disable-X`)
2. Config file (`-c`)
3. Capability defaults

## How it works

`winml optimize` loads the ONNX model, builds a final capability configuration by merging capability defaults, an optional config file, and any explicit CLI flags, then runs all enabled passes through the `Optimizer`. Each capability maps to a named optimization or fusion pipe in the `winml.modelkit.optim` registry. The capability flags are auto-generated at startup from that registry — adding a new optimization to the registry automatically makes it available as a CLI flag without any change to this command's source. After optimization, the command prints the before-and-after node count and percentage reduction so you can quantify the effect.

## Examples

Optimize a model with all capability defaults:

```bash
$ winml optimize -m microsoft/resnet-50.onnx
```

```text
Input:  microsoft/resnet-50.onnx
Output: microsoft/resnet-50_opt.onnx

Loading model...
Running optimizer...
Saving optimized model...

Success! Model optimized: microsoft/resnet-50_opt.onnx
Nodes: 312 -> 289 (7.4% reduction)
```

Enable specific fusions for a BERT model:

```bash
$ winml optimize -m bert-base-uncased.onnx \
    --enable-layer-norm-fusion \
    --enable-attention-fusion \
    -o bert_layernorm_attn.onnx
```

Use a config file to set capabilities and save the result for downstream compilation:

```bash
$ winml optimize -m facebook/convnext-tiny-224.onnx \
    -c optimize_config.yaml \
    -o convnext_opt.onnx
```

List all available optimization capabilities:

```bash
$ winml optimize --list-capabilities
```

Discover pattern-rewrite families and their flag names:

```bash
$ winml optimize --list-rewrites
```

Check which optimizations apply to a model before committing to a run (writes nothing):

```bash
$ winml optimize -m bert-base-uncased.onnx --check-optim
```

```text
Input: bert-base-uncased.onnx
--check-optim — analyzing applicable optimizations (no output written).

Loading model...
Probing 61 optimization capabilities... (this can take a while on large models)

2 applicable optimization(s):

--enable-matmul-add-fusion  (matmul)
  Fuse MatMul+Add operations into single kernel
  nodes: -1 ~1
    removed: MatMul (1)
      - MatMul 'mm'
    modified: Gemm (1)
      - Gemm 'mm/MatMulAddFusion'

--enable-clamp-constant-values  (surgery)
  Clamp extreme float constants (e.g., -inf -> -1e3) to prevent quantization issues
  constants: ~1
      - BIG

Enable any of the above with its --enable-* flag (dependencies are auto-enabled).
```

Add `-v` to list every affected node and constant instead of a sample.

## CGC compatibility rewrites

For an existing ONNX model, explicitly enable the required compatibility rules before
exporting CGIR MLIR. These rules are all **default off** and are independent of the
ordinary optimization/fusion defaults.

CGC-specific patterns are isolated in `src\winml\modelkit\pattern\cgc`, with
private matching helpers in `cgc\utils.py` and public pattern exports in
`cgc\__init__.py`. The CGIR compatibility pipe owns rule selection and execution;
the shared pattern package does not re-export these backend-specific patterns.

| Capability | Transformation and scope |
|------------|--------------------------|
| `normalize-int32-dq` | Normalize initializer-backed INT32 `DequantizeLinear` in the standard domain (opset >= 10) and `com.microsoft` (opset 1): omit immutable all-zero scalar/singleton zero points and clone singleton scales as scalars. Preserve shared initializers and domains; skip overridable parameters, per-axis vectors, unsupported attributes and nonlocal inputs. Handles nested graphs, not local functions. Enabled by CGC build configuration, disabled by default elsewhere. |
| `deduplicate-opset-imports` | Remove repeated model-level opset declarations with identical domain and version, retaining the first declaration and domain order. Reject conflicting versions for the same domain. Run before operator rewrites and opset upgrades; do not alter graph content, local functions, or the retained versions. |
| `eliminate-identity` | Remove safe internal tensor Identity aliases. Additionally replace top-level standard-domain FP32 graph-output Identities with same-shape Reshape when input/output types match exactly, all dimensions are positive static integers, opset >= 5, and the model has no subgraphs. Preserve output names/order, annotated aliases, unknown or conflicting types, scalar/dynamic/zero-size outputs and protected captures. Does not rewrite local functions. Workaround for [microsoft/ix#1198](https://github.com/microsoft/ix/issues/1198). |
| `gridsample-to-gather` | Decompose 2D `GridSample` with linear interpolation and zero padding into four `GatherND` reads, bounds masks and weighted sums. Supports both `align_corners` settings, FP16/FP32 IO and dynamic batch, with known positive channel, input spatial and grid spatial dimensions. Rank-3 indices contain explicit batch and spatial coordinates; `batch_dims=0` avoids the ORT symbolic shape inference defect tracked in [onnxruntime#24206](https://github.com/microsoft/onnxruntime/pull/24206). Batch coordinates are generated dynamically and shared across the four reads; sampled values are reshaped back to the grid layout. FP16 interpolation is computed in FP32 and cast back. Requires opset >= 16 (`bilinear` before opset 20); other modes are unchanged. Enabled by CGC builds, disabled in ordinary optimization. Floating-point rounding may differ from native sampling. |
| `omit-empty-resize-inputs` | Replace statically empty Resize ROI/scales with omitted inputs. Do not rely on graph-input defaults or rewrite crop-and-resize semantics. Requires opset 13; upgrade older matching models using ONNX version conversion. |
| `cgc-constant-folding` | Fill FoundryToolbox constant-folding gaps without an ORT Session. Fold standard `Pad.pads` constant integer chains; in graphs containing `Shape`, also fold statically known selected dimensions and bounded constant integer/boolean expressions (`Gather`, `Concat`, `Reshape`, `Slice`, `Transpose`, `Squeeze`, `Unsqueeze`, integer `Cast`, `ConstantOfShape`, arithmetic, `Equal`, `Where`). Iterate with shape inference, up to 32 rounds. Requires opset >= 11. Only the main graph is rewritten; preserve tensor names for shared uses and subgraph captures. Runtime floating-point computations and unresolved dimensions remain unchanged. This rule does not freeze inputs: specialize dimensions before optimization when needed; later Foundry `freeze-dims` does not retroactively affect this rule. Limits: 128 dependency values per traversal, 65,536 elements per operation and 1,048,576 cached elements per round. Enabled by CGC builds; disabled in ordinary optimization. `fold-constant-pad-pads` remains a compatibility alias. |
| `resize-tf-half-pixel-for-nn-to-asymmetric` | Change only the coordinate mode for nearest/floor Resize with static, non-overridable, positive integer scales. Dynamic/fractional scales and sizes-based inference are outside this rule. |
| `approximate-cubic-resize-with-linear` | **Lossy**, explicit cubic-to-linear approximation. Excludes antialiasing, outside exclusion, and crop-and-resize semantics. Prints a warning when applied. |
| `gathernd-to-reshape` | Replace GatherND only when data/indices/output ranks are not all equal and static, non-overridable int64 indices visit every input slice exactly once in storage order. Require positive static data dimensions; support batch dimensions, multi-coordinate indices, and equivalent negative indices. Dynamic data shapes or indices, overridable defaults, empty tensors, partial selection, repetition, and reordering are outside this rule. |
| `prelu-to-relu` | Decompose `PRelu(x, slope)` into `Relu(x) - slope * Relu(-x)` for FP16/FP32/FP64 and opset 7+. Require a direct, non-overridable initializer or Constant slope containing only finite values; preserve slope broadcasting and sharing. Dynamic data shapes are supported. Dynamic/non-finite slopes, integer types, and BF16 are outside this rule. |
| `dft-to-matmul` | Decompose DFT into real-valued sine/cosine matrix multiplications for FP16/FP32/FP64. Support opset 17+ axis attributes and opset 20+ scalar axis inputs; require known rank, static positive signal length, real/complex component count, and static optional axis/length parameters. Support forward/inverse full transforms, real forward onesided transforms, truncation, zero-padding, and dynamic batch dimensions. Dynamic transform parameters, BF16, and inverse onesided transforms are outside this rule. |

Configuration keys can use underscores in place of hyphens.

Add `--enable-deduplicate-opset-imports` (configuration: `"deduplicate_opset_imports": true`)
when repeated identical opset declarations leave a stale version during conversion.
This is an optimize rule, not an exporter fix: the saved optimized ONNX can be used
either for explicit CGIR export or directly by WinML Runtime's CGC backend.
It neither upgrades an opset nor resolves conflicting versions.

```text
winml optimize -m model.onnx -o model.opt.onnx --disable-ort-graph-optimization --enable-omit-empty-resize-inputs --enable-resize-tf-half-pixel-for-nn-to-asymmetric
winml export -m model.opt.onnx -o model.mlir --target cgir
winml perf -m model.mlir --runtime winml-runtime --backend cgc --device gpu
```

Add `--enable-approximate-cubic-resize-with-linear` to optimize only when an accuracy-changing
approximation is acceptable. This option is not included in the example's preserving rewrites.

Add `--enable-gathernd-to-reshape` (configuration: `"gathernd_to_reshape": true`) for
equivalent GatherND elimination. Rank mismatch alone is not sufficient: if identity
indexing cannot be proved, GatherND is retained. This bypasses CGC's DirectML descriptor
rank-alignment limitation; it does not implement general GatherND rank alignment.

Add `--enable-prelu-to-relu` (configuration: `"prelu_to_relu": true`) to bypass IX's
missing ONNX PRelu legalization. CGC already has a parameterized-ReLU implementation;
the missing mapping is in IX. This is an algebraic decomposition, not an interpolation
approximation, but it does not guarantee preservation of the sign bit of zero.
Non-finite slopes are excluded because multiplying them by zero on the inactive
branch could introduce NaNs.

Add `--enable-dft-to-matmul` (configuration: `"dft_to_matmul": true`) for DFT legalization.
The rewrite implements the DFT equation, including inverse normalization, without
requiring power-of-two lengths. It uses dense Fourier bases rather than an FFT:
constant storage and per-vector arithmetic are quadratic in the transform length.
It is intended as a compatibility workaround, not a performance replacement for a
native FFT. Different accumulation order and rounded coefficients mean floating-point
results are not bit-identical to the original DFT implementation.

The CGIR compatibility pipe applies each enabled rule once, matching the result of the
preceding rule. This lets different rewrites affect the same Resize without changing
the shared rewrite pipe's conflict policy. Rewrites retain the original model IR version;
an opset upgrade may affect the whole model. These capabilities do not provide
general GatherND replacement, dynamic-length DFT decomposition, or general control-flow folding.
An upstream conversion/runtime limitation can still prevent export or execution.

## Common pitfalls

- **`--model` is required for actual optimization** — it can be omitted only when using `--list-capabilities` or `--list-rewrites`. Missing `--model` in any other case raises a usage error.
- **Config file and CLI flags interact via precedence** — a `--disable-X` CLI flag always wins over a config file value that enables the same capability, but omitting the flag leaves the config file value in effect. To turn off a capability set by a config file, pass the explicit `--disable-X` flag.
- **Config file validation errors abort the run** — if the config file contains keys that fail capability validation or dependency checks, the command prints all errors and exits with code 1 without touching the model. Fix the config before retrying.
- **The dynamic flag list changes between releases** — new capabilities are added as the optimizer registry grows. Always use `--list-capabilities` to confirm the current set of flags rather than relying on a cached list.
- **Output path default may overwrite a sibling file** — if you run optimize twice on the same input without specifying `-o`, the second run silently overwrites `{input}_opt.onnx`. Specify an explicit output path in scripts.

## See also

- [how-it-works.md](../concepts/how-it-works.md) — where optimization fits in the full winml-cli pipeline
- [export.md](export.md) — produce an ONNX file to optimize from a HuggingFace model
- [quantize.md](quantize.md) — quantize the optimized model for lower-precision inference
- [config.md](config.md) — generate a `WinMLBuildConfig` that includes optimization settings
