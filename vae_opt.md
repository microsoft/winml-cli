# Wan 2.1 VAE Decoder ONNX Graph Optimization Report

## Scope

Model inspected directly:

`C:\Users\hualxie\ComfyUI\models\vae_onnx\wan2.1_vae_decoder_fp16.onnx`

External tensor data:

`C:\Users\hualxie\ComfyUI\models\vae_onnx\wan2.1_vae_decoder_fp16.onnx.data`

The model was analyzed without modifying either file.

## Graph summary

| Property | Value |
|---|---:|
| ONNX opset | 18 |
| Graph nodes | 9,558 |
| Initializers | 142 |
| Input | `z_tile`: FP16 `[1, 16, 21, h, w]` |
| Output | `sample_tile`: FP16 `[1, 3, 81, 8*h, 8*w]` |

The largest operator groups are:

| Operator | Count |
|---|---:|
| `Mul` | 2,359 |
| `Conv` | 927 |
| `Concat` | 845 |
| `Expand` | 794 |
| `Div` | 786 |
| `ReduceL2` | 780 |
| `Clip` | 780 |
| `Sigmoid` | 769 |
| `Add` | 381 |
| `Transpose` | 336 |
| `Reshape` | 279 |
| `Cast` | 160 |

## Recommended exact rewrites

### 1. Remove redundant `Clip` and `Expand` from channel L2 normalization

**Occurrences:** 780

Every matching block has this form:

```text
n = ReduceL2(x, axes=[1], keepdims=1)
n = Clip(n, min=0)
n = Expand(n, Shape(x))
y = Div(x, n)
```

Replace the four nodes with:

```text
n = ReduceL2(x, axes=[1], keepdims=1)
y = Div(x, n)
```

The scalar and learned-gamma multiplications after `Div` must remain in their
current order.

#### Mathematical proof

For a fixed batch and spatial location, let the channel vector be

```text
x = (x_1, x_2, ..., x_C).
```

`ReduceL2` computes

```text
n = sqrt(sum_j x_j^2) = ||x||_2.
```

Because a square is nonnegative,

```text
x_j^2 >= 0
```

and therefore

```text
n >= 0.
```

Consequently, clipping the norm to a minimum of zero cannot change it:

```text
max(n, 0) = n.
```

With `keepdims=1`, `n` already has a singleton channel dimension. `Expand`
only repeats that same norm across channels. ONNX `Div` supports
multidirectional broadcasting, so

```text
x / Expand(n, Shape(x)) = x / n.
```

#### Reduction

Each four-node block becomes two nodes: `780 * 2 = 1,560` fewer nodes.
Three shared shape-building `Concat` nodes also become dead, producing a total
reduction of **1,563 nodes**.

#### Numerical caveat

Keeping the original `ReduceL2` and `Div` kernels preserves their reduction,
rounding, and zero-norm behavior. In particular, an all-zero channel vector
still evaluates `0/0` instead of silently introducing an epsilon or returning
zero.

The following multiplications must not be folded together if the current FP16
rounding sequence must be retained:

```text
(normalized_x * channel_scale) * gamma
```

Changing it to

```text
normalized_x * (channel_scale * gamma)
```

is algebraically equal but can round differently in FP16.

#### Conditional `LpNormalization` fusion

For nonzero channel vectors, the two remaining nodes are algebraically equal
to:

```text
y = LpNormalization(x, axis=1, p=2)
```

ONNX Runtime CUDA supports this operator for FP16 at opset 18. However, its
CUDA kernel intentionally returns zeros when the norm is zero, while the
original explicit `Div` produces NaNs from `0/0`. It also uses its own FP32
reduction order for FP16 inputs. Therefore this one-node fusion is not globally
behavior-equivalent and must not be classified as exact unless nonzero norms
are proven or the zero-norm behavior change is explicitly accepted.

### 2. Replace generated `GatherND` upsampling with nearest-neighbor `Resize`

**Occurrences:** 73

Every spatial 2x upsampling block constructs integer indices and performs:

```text
FP16 input
  -> Cast(FP32)
  -> Transpose
  -> GatherND(generated indices)
  -> Transpose
  -> Cast(FP16)
```

The index-generation subgraphs are shared in three groups used by 11, 21, and
41 upsampling sites.

Replace each data path with an ONNX `Resize` operating directly on FP16:

```text
mode = "nearest"
coordinate_transformation_mode = "asymmetric"
nearest_mode = "floor"
axes = [2, 3]
scales = [2.0, 2.0] for the selected axes
```

#### Mathematical proof

For output coordinate `i`, the existing graph computes

```text
index(i) = int((i + 0.5) * (H / (2H))).
```

For nonzero `H`,

```text
H / (2H) = 1/2,
```

so

```text
index(i) = int((i + 0.5) / 2).
```

All generated coordinates are nonnegative, so conversion to `int64` is the
same as `floor`:

```text
index(i) = floor((i + 0.5) / 2) = floor(i / 2).
```

Nearest-neighbor `Resize` with `asymmetric` coordinates and `floor` selection
uses

```text
source(i) = floor(i / scale) = floor(i / 2).
```

Thus both formulations select exactly the same source element. The same proof
applies independently to height and width.

The FP32 round trip is unnecessary because nearest-neighbor sampling performs
no arithmetic on tensor values; it only copies selected elements. Every FP16
value is exactly representable in FP32, so FP16 -> FP32 -> FP16 does not alter
ordinary finite values.

#### Reduction

Replacing the 73 paths makes 433 old nodes dead, including the shared index
generation. Adding 73 `Resize` nodes gives a net reduction of **360 nodes**.

#### Required constraints

- Use `nearest`, not linear or cubic interpolation.
- Set `coordinate_transformation_mode="asymmetric"`.
- Set `nearest_mode="floor"`.
- Resize only spatial axes 2 and 3.
- Prefer constant `scales=[2.0, 2.0]`; CUDA expects the optional Resize
  configuration inputs in CPU memory, and constant initializers avoid a
  dynamic CPU shape subgraph.
- Preserve dynamic `h` and `w`; do not specialize them to one test shape.

### Combined standard-ONNX result

| Metric | Count |
|---|---:|
| Original nodes | 9,558 |
| Exact L2-normalization cleanup | 1,563 |
| Resize reduction | 360 |
| Resulting nodes | **7,635** |
| Total reduction | **1,923 (20.1%)** |

Both recommended rewrites remain in the standard ONNX domain at opset 18.
If the conditional `LpNormalization` behavior is accepted, it removes another
780 nodes and produces the previously calculated 6,855-node graph.

## Backend-specific fusions

### 3. Fuse explicit SiLU

**Occurrences:** 769

The graph expresses SiLU as:

```text
s = Sigmoid(x)
y = Mul(x, s)
```

Mathematically,

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

and therefore

```text
y = x * sigmoid(x).
```

ONNX Runtime recognizes all 769 sites as
`com.microsoft::QuickGelu(alpha=1)`. Despite the operator name, with
`alpha=1` its expression is:

```text
y = x * sigmoid(alpha*x) = x * sigmoid(x).
```

This reduces two graph nodes to one at each site. Applied after the two exact
standard rewrites, the graph would contain approximately **6,866 nodes**.
Applying the conditional `LpNormalization` fusion as well would produce the
previously calculated **6,086-node** graph.

This is mathematically equivalent but uses a Microsoft-domain operator, so it
should only be emitted when the target execution provider supports it.

### 4. Fuse attention score scaling into `MatMul`

**Occurrences:** 11

The attention blocks contain:

```text
s = MatMul(Q, K)
y = Mul(s, 0.051025390625)
```

A backend fused matrix multiplication with `alpha=0.051025390625` computes:

```text
y = alpha * (Q @ K),
```

which is the same real-number expression. ONNX Runtime can represent this as
`com.microsoft::FusedMatMul`.

This is backend-specific and may change floating-point rounding if scaling is
incorporated into the accumulation kernel. It should not be treated as a
bitwise-exact portable rewrite.

## Opportunities requiring architectural transformation

### Temporal `Concat -> Conv` unrolling

There are 780 `Concat -> Conv` sites, but only 32 distinct convolution weight
tensors. Individual weights are reused 10, 20, or 40 times. This indicates
export-time temporal unrolling.

Consolidating these repeated blocks could yield a much larger graph reduction,
but it is not a safe local fusion. The concatenations encode causal context,
zero initialization, and boundary handling. A correct rewrite must prove that
a single sequence-level convolution reproduces:

- the same temporal receptive field;
- the same zero padding at the first frame;
- the same cached-state boundaries;
- the same output-frame ordering; and
- the same residual connections.

This should be considered a separate optimization project rather than an
automatic peephole rewrite.

## Rejected local rewrites

- No adjacent inverse `Transpose -> Transpose` pairs were found.
- No adjacent redundant `Reshape -> Reshape` pairs were found.
- No adjacent redundant `Cast -> Cast` pairs were found.
- No exactly duplicated nodes were found.
- The 385 `Conv -> Add` paths are dynamic residual additions. Each convolution
  already has a bias input, so these additions cannot be folded into Conv bias.
- Combining consecutive FP16 scalar/channel multiplications changes rounding
  order and should not be called numerically exact.

## Implemented `winml optimize` surgeries

All four rewrites are implemented as independent, disabled-by-default flags:

| Flag | Replacement |
|---|---|
| `--enable-simplify-l2-normalization` | Remove only the proven redundant `Clip` and `Expand` |
| `--enable-gathernd-to-resize` | Replace the exact generated 2x nearest-neighbor path with `Resize` |
| `--enable-silu-to-quick-gelu` | Replace `x * Sigmoid(x)` with `com.microsoft::QuickGelu(alpha=1)` |
| `--enable-scaled-matmul-to-fused-matmul` | Replace `MatMul -> Mul(scalar)` with `com.microsoft::FusedMatMul(alpha=scalar)` |

The normalization, resize, and scaled-MatMul matchers run before ORT graph
optimization, while the SiLU fusion runs afterward so ORT does not inline
`QuickGelu` back into primitive operations. The matchers use graph topology,
types, shapes, constants, attributes, and consumer counts; they do not depend
on model, node, or tensor names.

### Simple before-and-after examples

#### 1. Simplify L2 normalization

For a channel vector `x = [3, 4]`, `ReduceL2(x) = 5`.

```text
Before:
  norm     = ReduceL2([3, 4])       = 5
  clipped  = Clip(5, min=0)         = 5
  expanded = Expand(5, shape=[2])   = [5, 5]
  output   = [3, 4] / [5, 5]       = [0.6, 0.8]

After:
  norm   = ReduceL2([3, 4])         = 5
  output = [3, 4] / 5              = [0.6, 0.8]
```

Broadcasting the scalar or singleton-channel norm gives the same result as
explicitly expanding it.

#### 2. Replace `GatherND` upsampling with `Resize`

For a 2x2 image, both graphs copy each value into a 2x2 output block:

```text
Input:                  2x nearest-neighbor output:

  [a b]                   [a a b b]
  [c d]                   [a a b b]
                          [c c d d]
                          [c c d d]
```

The generated `GatherND` indices and `Resize(mode="nearest",
coordinate_transformation_mode="asymmetric", nearest_mode="floor")` both select
source coordinate `floor(output_coordinate / 2)`.

#### 3. Fuse SiLU into `QuickGelu`

For `x = 2`:

```text
Before:
  output = 2 * Sigmoid(2)

After:
  output = QuickGelu(2, alpha=1)
         = 2 * Sigmoid(1 * 2)
         = 2 * Sigmoid(2)
```

`QuickGelu` is exactly the same formula when `alpha=1`.

#### 4. Fuse scaled `MatMul`

```text
Before:
  product = MatMul(A, B)
  output  = product * 0.5

After:
  output = FusedMatMul(A, B, alpha=0.5)
         = 0.5 * MatMul(A, B)
```

These are the same real-number expression. A fused kernel can change the final
floating-point rounding because it may apply `alpha` during accumulation.

The target model was processed successfully with:

```text
winml optimize -m C:\Users\hualxie\ComfyUI\models\vae_onnx\wan2.1_vae_decoder_fp16.onnx -o temp\wan2.1_vae_decoder_surgeries.onnx --overwrite --disable-constant-folding --enable-simplify-l2-normalization --enable-gathernd-to-resize --enable-silu-to-quick-gelu --enable-scaled-matmul-to-fused-matmul
```

The complete command pipeline produced 9,503 nodes. This differs from the
6,855-node direct-surgery count because the command's mandatory ORT graph pass
lowers and expands some standard operators. The requested transformations were
present in the saved graph:

| Operator | Original | Optimized |
|---|---:|---:|
| `Clip` | 780 | 0 |
| `GatherND` | 73 | 0 |
| `Resize` | 0 | 73 |
| `Sigmoid` | 769 | 0 |
| `com.microsoft::QuickGelu` | 0 | 769 |
| `MatMul` | 22 | 11 |
| `com.microsoft::FusedMatMul` | 0 | 11 |

The optimized model and its external-data sidecar pass ONNX full validation.

## Validation requirements

For each implemented rewrite:

1. Run `onnx.checker.check_model` with the external tensor data available.
2. Compare original and optimized outputs with generated test inputs covering
   multiple valid dynamic `h` and `w` values.
3. Include random inputs, all-zero inputs, large finite FP16 values, and values
   near zero.
4. Compare each rewrite independently before testing the combined graph.
5. Test on the intended execution provider, not only CPU.
6. Require exact equality for the nearest-neighbor resize rewrite where the
   provider preserves FP16 copies.
7. Use an explicitly approved FP16 tolerance for normalization and fused
   kernels because mathematically equal reductions can accumulate in a
   different order.

## Recommended implementation order

1. Implement `GatherND` upsampling to standard `Resize`; its value-selection
   equivalence is strongest and it removes index-generation overhead.
2. Remove normalization `Clip` and `Expand`, retaining `ReduceL2 -> Div`.
3. Consider `LpNormalization` only if its zero-norm behavior is acceptable.
4. Enable SiLU and scaled-MatMul fusions only through provider capability
   checks.
5. Evaluate temporal consolidation separately with full decoder equivalence
   tests.
