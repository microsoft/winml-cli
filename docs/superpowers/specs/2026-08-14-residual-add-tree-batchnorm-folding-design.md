# Residual Add-Tree BatchNormalization Folding Design

**Status:** Approved design
**Date:** 2026-08-14
**Capability:** `--enable-conv-add-batch-normalization-folding`

## Summary

Extend the existing opt-in Conv/Add/BatchNormalization folding capability to
cover safe residual sums whose BatchNormalization input is an arbitrary-depth
binary Add tree with at least two Conv leaves. The existing generated
RewritePipe rule remains responsible for the fixed
`Conv + immutable static Add -> inference BatchNormalization` topology.
AlgebraicRewritePipe becomes a second owner of the same capability and handles
the recursive topology.

The residual rewrite distributes the inference BatchNormalization channel
scale over every leaf, assigns the BatchNormalization additive shift to exactly
one deterministic Conv bias, removes the BatchNormalization node, and preserves
its output tensor name. It is architecture-agnostic and fails closed unless the
entire candidate can be proven safe and all replacement constants can be
computed before mutation.

The evidence model has four eligible residual BatchNormalization nodes, and
real-model acceptance requires all four to be removed.

## Context

The current rule is generated from
`src/winml/modelkit/pattern/rules/default.json` and implemented by
`src/winml/modelkit/pattern/conv_batchnorm_patterns.py`. It has two source
patterns for the commutative Add input orders and one folded target. Those
finite skeletons correctly cover one Conv plus one static Add operand.

PR #1301 added the generic graph indexing, cycle and definition-collision
guards, nested-graph capture detection, immutable constant loading,
copy-on-write initializer handling, bounded recursive traversal, and atomic
candidate behavior used by AlgebraicRewritePipe. It also made optimization
analysis understand one capability intentionally owned by multiple pipes:
analysis probes the capability once across the complete ordered pipeline and
reports the combined owner names and aggregate graph diff.

## Goals

- Keep the existing public CLI flag and default-off behavior.
- Preserve the fixed RewritePipe behavior.
- Fold inference BatchNormalization after a proper binary Add tree containing
  at least two eligible Conv leaves and zero or more immutable constant leaves.
- Support arbitrary tree shape and Add operand order up to a conservative depth
  limit.
- Preserve graph semantics, public outputs, and unaffected users of shared
  initializers through copy-on-write replacement.
- Reuse the hardened generic graph-rewrite and shared-capability machinery from
  PR #1301.

## Architecture

### Capability ownership

`conv-add-batch-normalization-folding` remains the single public capability.
RewritePipe continues to obtain its canonical `BoolCapability` from the
generated rewrite registry. AlgebraicRewritePipe registers that same canonical
capability object in `ALGEBRAIC_CAPABILITIES`; it does not define a second
capability with a duplicate name.

Both pipe configurations read the existing snake-case key
`conv_add_batch_normalization_folding`:

- AlgebraicRewritePipe uses it to run the residual Add-tree fold.
- RewritePipe uses it to expand the existing two fixed-topology rewrite rules.

This retains one `--enable-conv-add-batch-normalization-folding` /
`--disable-conv-add-batch-normalization-folding` pair, one config field, and one
entry in capability listings. `get_all_capabilities()` continues to return one
definition because both owners reference the same definition.

Optimization analysis relies on PR #1301's shared-owner path. It probes the
flag once from the full pipeline input, runs all owners in pipeline order, and
reports one finding with owners `algebraic_rewrite+rewrite` and the combined
node/initializer changes. It must not emit separate or duplicate findings.

### Pipeline placement

The top-level pipe order remains:

1. ORTGraphPipe, including ORT constant folding.
2. AlgebraicRewritePipe.
3. RewritePipe.
4. ORTFusionPipe.
5. SurgeryPipe.

Within AlgebraicRewritePipe, residual Add-tree BatchNormalization folding runs
before Conv channel-affine folding. Therefore:

- ORT first exposes constants that make residual candidates statically
  provable.
- Residual folding sees the original complete Add tree and consumes its
  BatchNormalization before later affine rewrites can alter eligible Conv
  branches.
- The existing fixed pattern still runs afterward in RewritePipe.

No new pipe is introduced and no unrelated pass is reordered.

### Why AlgebraicRewritePipe

A finite pattern expansion is a poor representation of this topology. Every
additional tree depth adds tree shapes and operand permutations, so a rule set
would be bounded, repetitive, and unable to express the approved
arbitrary-depth contract. It would also duplicate whole-candidate safety checks
outside the graph machinery hardened in PR #1301.

A new pipe would duplicate graph indexing, constant loading, copy-on-write
allocation, cleanup, and capability plumbing while adding another ordering
surface. AlgebraicRewritePipe already owns bounded recursive, exact algebraic
transformations and has the required fail-closed infrastructure. The fixed
one-Conv skeleton remains in RewritePipe because it is finite, established, and
independently useful.

## Candidate Model

Candidate discovery starts from each standard-domain BatchNormalization node
and recursively walks backward from its data input. Discovery produces an
immutable candidate description; it does not edit the model.

### BatchNormalization invariants

An eligible BatchNormalization must:

- use the standard ONNX domain with exactly one default-domain opset import at
  version 7 or later;
- have exactly five non-empty inputs and exactly one non-empty output;
- be inference-mode: `training_mode` is absent or integer zero;
- have a finite, non-negative `epsilon`;
- have immutable, readable `scale`, `B`, `mean`, and `var` tensors;
- use one supported floating type consistently across the data input,
  parameters, Conv parameters, and constant leaves;
- have one-dimensional parameter tensors whose length equals the channel count;
- have finite parameters and strictly positive, finite `var + epsilon`; and
- expose a statically known input rank, shape, and channel dimension sufficient
  to validate every leaf and broadcast.

An initializer that is also a graph input is overridable and therefore not
immutable. A standard-domain Constant value is immutable only when its payload
is readable under the same external-data and allocation rules as an
initializer.

Single-output inference semantics are mandatory. Training-mode outputs or any
additional BatchNormalization output make the candidate ineligible.

### Add-tree invariants

The BatchNormalization data input must be produced by a proper binary tree of
standard-domain Add nodes:

- every Add has exactly two non-empty inputs and one non-empty output;
- recursive Add nodes may occur in either input slot and at any combination of
  depths;
- every terminal leaf is either an eligible Conv output or an immutable
  constant;
- at least two distinct Conv nodes are present;
- no other operator or dynamic tensor is accepted as a leaf;
- the same computed tensor may not occupy multiple tree positions; and
- traversal depth may not exceed 64 Add edges.

Every Add output below the BatchNormalization and every Conv leaf output is
private to the candidate path. Such a tensor must not be:

- a graph output;
- consumed outside the candidate;
- consumed more than once within the candidate; or
- captured by a nested graph attribute.

These rules make the candidate a tree rather than a shared DAG and ensure that
renaming the root Add output and replacing leaf parameters cannot change an
observed intermediate.

### Conv leaf invariants

Each Conv leaf must:

- use the standard ONNX domain;
- have two or three inputs and exactly one non-empty output;
- have a readable immutable initializer for its weights;
- have a readable immutable initializer for its bias when a bias is present;
- have floating, finite weights with output-channel dimension equal to the
  BatchNormalization channel count;
- have either no bias or a finite one-dimensional bias of that channel count;
  and
- have a statically inferred output shape equal to the Add-tree result shape.

The Conv data input is unrestricted and may remain dynamic. Only parameters
that the rewrite changes must be immutable.

Shared weight and bias initializers are allowed because they are copied and the
candidate Conv inputs are rewired to the copies. Other users retain the
original tensors.

### Constant leaf invariants

A constant leaf may be an immutable initializer or the output of a
standard-domain Constant node. Its value must be floating, finite, dtype
compatible, statically shaped, and ONNX-broadcastable to the Add-tree result
shape.

Constant payloads may be shared because the fold creates a replacement
initializer for the candidate input instead of changing the original payload.
Any required broadcasted representation must remain below the repository's
external-data threshold and NumPy element-count limit. Unloaded external data,
malformed tensor payloads, or an unsafe expansion reject the whole candidate.

### Graph-wide invariants

Before candidate discovery, the graph index must establish:

- no duplicate or cross-kind tensor definitions;
- no graph cycle;
- one unambiguous producer for every traversed tensor; and
- no custom-domain node within the candidate.

A violation rejects algebraic processing without changing the input model.
Names are allocated through the existing collision-aware allocator.

## Algebra and Data Flow

For BatchNormalization input `S`, inference BatchNormalization is:

```text
Y = scale * (S - mean) / sqrt(var + epsilon) + B
```

Define channel vectors:

```text
alpha = scale / sqrt(var + epsilon)
delta = B - alpha * mean
```

For an Add tree with Conv leaves `Conv_i(X_i, W_i, b_i)` and constant leaves
`C_j`:

```text
S = sum_i Conv_i(X_i, W_i, b_i) + sum_j C_j
Y = sum_i (alpha * Conv_i(X_i, W_i, b_i))
  + sum_j (alpha * C_j)
  + delta
```

The replacement constants are:

```text
W_i' = W_i * reshape(alpha, [channels, 1, ..., 1])
b_i' = alpha * b_i
C_j' = alpha_broadcast * C_j
```

For a Conv without a bias, `b_i` is the zero channel vector and a new bias is
created. The `delta` term is added exactly once to the replacement bias of the
anchor Conv:

```text
b_anchor' = alpha * b_anchor + delta
```

The anchor is the eligible Conv leaf with the smallest original
`graph.node` index. This is deterministic, independent of Add operand order,
and requires no model-specific naming convention.

`alpha_broadcast` has shape `[1, channels, 1, ...]` at the Add-tree result rank.
For each constant leaf, multiplication uses ONNX/NumPy right-aligned
broadcasting and materializes the smallest result shape produced by
broadcasting the leaf shape with `alpha_broadcast`; it does not expand to the
full activation shape unless the leaf already requires that shape.

All calculations use a safe calculation dtype at least as wide as float32,
then cast to the original tensor dtype. Every intermediate and every
post-cast replacement is checked for finiteness, representability, expected
shape, and bounded size. A cast overflow or shape change outside the defined
broadcast result rejects the candidate.

The Add topology and operand ordering remain unchanged. After replacement
parameters are installed, the root Add takes the former BatchNormalization
output name and the BatchNormalization node is removed. Thus downstream edges
and the public output signature do not require rewiring.

## Atomic Mutation Strategy

The rewrite is atomic per BatchNormalization candidate.

1. Build and validate the full graph index.
2. Discover the complete Add tree with a bounded recursive walk.
3. Validate all BatchNormalization parameters, leaves, shapes, consumers,
   captures, domains, and tensor payloads.
4. Compute `alpha`, `delta`, every copied Conv weight, every copied Conv bias,
   and every scaled constant leaf in memory.
5. Validate every computed array and reserve collision-free names.
6. Only after all prior work succeeds, append the new initializers, rewire
   candidate leaf inputs, rename the root Add output, and remove the
   BatchNormalization node.
7. Rebuild the graph index before considering another candidate.
8. Reuse existing dead-Constant, unused-initializer, and stale-value-info
   cleanup after all enabled algebraic rewrites run.

No original initializer is modified in place. No model mutation occurs during
candidate discovery or constant computation. If any leaf or replacement fails,
the candidate contributes no initializers, rewires, renames, or removals.
Successful earlier independent candidates remain valid if a later candidate is
ineligible.

## Fail-Closed and Error Behavior

Ineligibility is expected graph content, not an optimization error. The pass
leaves that candidate unchanged and continues scanning. It must fail closed for
all of the following:

- dynamic or overridable BatchNormalization parameters;
- training mode, extra BatchNormalization outputs, invalid epsilon, or
  non-positive variance denominator;
- NaN, infinity, overflow, unsupported dtype, mixed dtype, or incompatible
  shape;
- custom-domain BatchNormalization, Add, Conv, or Constant nodes;
- fewer than two Conv leaves, unsupported leaves, non-binary Add nodes, or
  malformed node arity;
- observed, shared, multiply referenced, or nested-graph-captured computed
  intermediates;
- cycles, definition collisions, ambiguous producers, or depth above 64;
- dynamic, overridable, malformed, or non-finite Conv parameters;
- unreadable or unloaded external tensor data;
- replacement arrays that exceed safe element or external-data limits; and
- any inability to prove the complete candidate safe before mutation.

Validation uses explicit guards and narrow tensor-conversion failures. It does
not use a broad exception handler to disguise implementation defects as
ineligible candidates. Unexpected internal or serialization failures continue
through the optimizer's existing error path.

## Fixed-Pattern Coexistence

The algebraic residual matcher deliberately rejects one-Conv trees. Therefore,
the existing fixed `Conv + static Add -> BatchNormalization` cases continue to
be handled by RewritePipe without overlap. A model may contain both fixed and
residual candidates; the shared flag enables both owners in one optimizer run.

Idempotence follows from both transforms removing the eligible
BatchNormalization node. A second run under the same capability must produce
no graph or initializer changes.

## Test Design

All tests use pytest and code-generated ONNX graphs and expected results.
Production implementation tests belong primarily in
`tests/unit/optim/pipes/test_pipe_algebraic.py`; fixed-pattern regression tests
remain in `tests/unit/pattern/test_conv_batchnorm_patterns.py`; shared ownership
and probing tests belong in `tests/unit/optim/test_analysis.py`.

### Positive generated-graph coverage

- Two Conv leaves under one Add.
- Left-deep, right-deep, and balanced nested trees.
- Add operand permutations at every level.
- Every combination of existing and missing Conv biases.
- Scalar, channel, and higher-rank broadcastable constant leaves.
- Multiple constant leaves at different depths.
- Shared Conv weight/bias initializers and shared constant payloads, proving
  copy-on-write behavior leaves unrelated users unchanged.
- Deterministic shift placement on the earliest serialized Conv regardless of
  Add operand permutation.
- Multiple independent residual candidates.
- One model containing residual candidates and existing fixed-pattern
  candidates under the same flag.
- A second optimization run proving idempotence.
- Capability listing/config behavior proving no new flag was introduced.
- Shared-capability analysis proving one combined finding, both owners, and a
  combined full-pipeline diff.

Each successful case checks:

- the ONNX checker accepts the transformed model;
- every eligible BatchNormalization is removed;
- the root Add preserves the former BatchNormalization output name;
- ordered graph output names, types, ranks, and dimensions are unchanged;
- unrelated graph structure and initializer users are unchanged; and
- baseline and transformed execution agree within dtype-specific tolerances:
  `rtol=atol=3e-3` for float16, `3e-5` for float32, and `1e-10` for float64.

### Fail-closed generated-graph coverage

Parameterized tests keep the model byte-equivalent for:

- one or zero Conv leaves;
- dynamic or graph-input-overridable BatchNormalization parameters;
- training mode, additional outputs, malformed arity, invalid epsilon, and
  invalid variance;
- non-finite parameters, leaves, or computed replacements;
- integer, unsupported floating, or mixed dtypes;
- channel-count, bias, weight, inferred-shape, and broadcast mismatches;
- dynamic or overridable Conv weights/biases;
- unsupported dynamic or operator-produced leaves;
- custom-domain variants of every candidate operator;
- graph-output, externally shared, multiply referenced, and nested-captured
  Conv/Add intermediates;
- repeated leaf aliases that turn the topology into a DAG;
- cycles and all definition-collision kinds recognized by the graph index;
- depth 65 while depth 64 remains eligible;
- malformed Constant payloads, unloaded external initializer data, unloaded
  external Constant data, and unsafe replacement size; and
- ambiguous default-domain opset metadata.

## Real-Model Validation

Use:

- model:
  `D:\AI\isv_models\keen_hominy\0805\model_opset17.onnx`
- inputs:
  `D:\AI\isv_models\keen_hominy\0805\model_example_inputs.npz`
- PyTorch reference outputs:
  `D:\AI\isv_models\keen_hominy\0805\model_example_outputs_pytorch.npz`

The observed source graph contains 232 total nodes and four
BatchNormalization nodes. Two BatchNormalization nodes consume a single Add
whose leaves are two Conv nodes. The other two consume nested two-Add trees
whose leaves are three Conv nodes. All four are eligible residual candidates
under this design.

Run baseline and optimized CPU inference with identical inputs and enable only
the approved capability beyond defaults. The implementation evidence must
record:

- total graph node count before and after, with no prescribed final count
  because default ORT constant folding also runs;
- BatchNormalization node count before and after, which must be `4 -> 0`;
- exact ordered output signature before and after, including names, element
  types, ranks, and dimensions;
- per-output maximum absolute and relative error for optimized versus baseline;
- per-output maximum absolute and relative error for baseline versus PyTorch
  reference and optimized versus PyTorch reference; and
- ONNX checker success for the saved optimized model.

Optimized-versus-baseline replay must satisfy the dtype-specific test
tolerances. Optimized-versus-PyTorch error must not regress beyond those same
absolute and relative tolerances compared with baseline-versus-PyTorch error.
The optimized graph must preserve the output signature exactly and remove all
four eligible residual BatchNormalization nodes.

NPU latency, throughput, compilation success, and performance improvement are
explicitly not acceptance gates for this change.

## Non-Goals

- No new CLI flag, config key, capability alias, or default enablement.
- No replacement or removal of the existing fixed RewritePipe patterns.
- No finite expansion of recursive Add trees into generated pattern skeletons.
- No new optimization pipe.
- No support for Sub, Mul, Div, Concat, activation, projection, normalization,
  or other operator leaves.
- No dynamic BatchNormalization parameters, training BatchNormalization, or
  custom-domain operators.
- No folding across shared or publicly observed computed intermediates.
- No in-place mutation of shared initializers.
- No architecture, model-name, tensor-name, layer-name, or execution-provider
  special cases.
- No NPU performance requirement.
- No unrelated algebraic, pattern, or pipeline refactor.

## Acceptance Criteria

The future implementation is accepted when all of the following are true:

1. The existing CLI flag enables both AlgebraicRewritePipe residual folding and
   RewritePipe fixed-pattern folding, with no duplicate public capability.
2. Residual folding runs after ORT constant folding and before Conv
   channel-affine folding.
3. Every accepted candidate satisfies the documented invariants and the exact
   algebra, uses copy-on-write parameters, assigns `delta` once to the
   deterministic anchor, removes BatchNormalization, and preserves its output
   name.
4. Every documented unsafe or ambiguous condition leaves the candidate
   unchanged without partial mutation.
5. Generated pytest coverage includes positive trees, operand permutations,
   bias variants, constants, shared initializers, idempotence, fixed-pattern
   coexistence, shared-capability analysis, and all fail-closed categories.
6. Successful generated graphs pass ONNX checking and numerical equivalence at
   the documented dtype-specific tolerances.
7. The real-model evidence records all required counts, signature data, and
   replay errors; preserves the signature; passes ONNX checking; and reduces
   BatchNormalization count from four to zero by removing all four eligible
   residual nodes. The final total node count is recorded but is not prescribed.
8. Existing fixed-pattern tests and affected optimizer tests continue to pass.
9. The change contains no model-specific logic, unrelated refactor, production
   implementation outside the approved scope, or NPU acceptance dependency.
