# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from collections import Counter

import onnx


m = onnx.load(r"convnext-search\iter_00\export.onnx")
g = m.graph

out2node = {}
for n in g.node:
    for o in n.output:
        out2node[o] = n


def consumers(node):
    result = []
    for o in node.output:
        for n in g.node:
            if o in n.input:
                result.append(n)
    return result


def producer(inp):
    return out2node.get(inp)


# ── 1. Block structure ────────────────────────────────────────
print("=== ConvNext block structure (trace first DW-Conv forward) ===")
first_dw = next(
    (
        n
        for n in g.node
        if n.op_type == "Conv" and next((a.i for a in n.attribute if a.name == "group"), 1) > 1
    ),
    None,
)
cur = first_dw
for _ in range(14):
    if cur is None:
        break
    c = consumers(cur)
    c_types = [n.op_type for n in c]
    print(f"  {cur.op_type:25s} -> {c_types}")
    if len(c) == 1:
        cur = c[0]
    elif len(c) > 1:
        non_add = [n for n in c if n.op_type != "Add"]
        cur = non_add[0] if non_add else c[0]
    else:
        break

# ── 2. Transpose patterns ─────────────────────────────────────
print()
print("=== Transpose patterns (before -> Transpose -> after) ===")
trans_patterns = Counter()
for n in g.node:
    if n.op_type == "Transpose":
        c = consumers(n)
        p = producer(n.input[0])
        before = p.op_type if p else "INPUT"
        after = c[0].op_type if c else "OUTPUT"
        trans_patterns[f"{before} -> Transpose -> {after}"] += 1
for pat, cnt in trans_patterns.most_common():
    print(f"  {cnt:3d}x  {pat}")

# ── 3. GELU variants ──────────────────────────────────────────
print()
print("=== GELU sub-patterns ===")
# Standard GELU: Mul -> Div -> Erf -> Add -> Mul -> Mul
gelu_standard = 0
for n in g.node:
    if n.op_type == "Erf":
        p = producer(n.input[0])
        if p and p.op_type == "Div":
            gelu_standard += 1
print(f"  Div->Erf (Erf-based GELU): {gelu_standard}")

# Check for Sigmoid-based QuickGELU (x * sigmoid(1.702 * x))
quick_gelu = 0
for n in g.node:
    if n.op_type == "Sigmoid":
        c = consumers(n)
        if c and c[0].op_type == "Mul":
            quick_gelu += 1
print(f"  Sigmoid->Mul (QuickGELU candidate): {quick_gelu}")

# ── 4. Downsampling blocks (stage transitions) ────────────────
print()
print("=== Downsampling block pattern (LN->Conv 2x2 stride 2) ===")
down_blocks = 0
for n in g.node:
    if n.op_type == "Conv":
        stride = next((list(a.ints) for a in n.attribute if a.name == "strides"), [1, 1])
        kernel = next((list(a.ints) for a in n.attribute if a.name == "kernel_shape"), [])
        groups = next((a.i for a in n.attribute if a.name == "group"), 1)
        if stride == [2, 2] and groups == 1:
            p = producer(n.input[0])
            print(f"  stride-2 Conv kernel={kernel}  preceded_by={p.op_type if p else 'INPUT'}")
            down_blocks += 1

# ── 5. Residual branches ──────────────────────────────────────
print()
print("=== Add nodes with 2 distinct producer op-types (residual candidates) ===")
residual_counter = Counter()
for n in g.node:
    if n.op_type == "Add" and len(n.input) == 2:
        p0 = producer(n.input[0])
        p1 = producer(n.input[1])
        t0 = p0.op_type if p0 else "INIT"
        t1 = p1.op_type if p1 else "INIT"
        if t0 != t1:
            key = tuple(sorted([t0, t1]))
            residual_counter[key] += 1
for pair, cnt in residual_counter.most_common():
    print(f"  {cnt:3d}x  Add({pair[0]}, {pair[1]})")

# ── 6. Node domain analysis ───────────────────────────────────
print()
print("=== Op domains ===")
domains = Counter()
for n in g.node:
    dom = n.domain if n.domain else "ai.onnx"
    domains[dom] += 1
for d, c in domains.most_common():
    print(f"  {d}: {c} nodes")

# ── 7. analyze gaps ───────────────────────────────────────────
print()
print("=== Patterns winml analyze may miss ===")
# 1. Depthwise conv with large kernels (7x7 DW-Conv is ConvNext specific)
dw7x7 = sum(
    1
    for n in g.node
    if n.op_type == "Conv"
    and next((a.i for a in n.attribute if a.name == "group"), 1) > 1
    and next((list(a.ints) for a in n.attribute if a.name == "kernel_shape"), []) == [7, 7]
)
print(f"  7x7 DW-Conv (ConvNext pattern): {dw7x7}")
print("    -> analyze classifies as OP/ai.onnx/Conv (undifferentiated)")
print("    -> no distinction between DW-Conv and regular Conv EP support")

# 2. Transpose wrapping every layer (NCHW<->NHWC conversion)
trans_total = sum(1 for n in g.node if n.op_type == "Transpose")
print(f"  Transpose nodes total: {trans_total}")
print("    -> analyze reports as single OP/ai.onnx/Transpose")
print("    -> no detection of Transpose-sandwich (NCHW->NHWC->op->NCHW)")
print("    -> transpose-optimizer capability not reflected in analyze output")

# 3. MatMul used as dense layer (not Gemm) - different EP kernel path
matmul_count = sum(1 for n in g.node if n.op_type == "MatMul")
print(f"  MatMul (not Gemm): {matmul_count}")
print("    -> ConvNext uses MatMul for MLP (not Gemm), QNN handles differently")
print("    -> analyze does not distinguish MatMul-as-FC from MatMul-as-attention")

# 4. LayerNormalization as a single op (already fused by PyTorch export)
ln_count = sum(1 for n in g.node if n.op_type == "LayerNormalization")
print(f"  LayerNormalization (native op): {ln_count}")
print("    -> These are already fused (not the ReduceMean->Sub->... subgraph)")
print("    -> layer-norm-fusion capability targets the decomposed pattern")
print("    -> analyze should note these are ALREADY fused - no fusion needed")

# 5. Erf-based GELU (not tagged as Gelu op, appears as com.microsoft/Gelu after fusion)
print(f"  Erf-based GELU subgraphs (unfused): {gelu_standard}")
print('    -> analyze cannot detect "unfused GELU" as a pattern')
print("    -> gelu-fusion would convert these to com.microsoft/Gelu")
print('    -> no analyze rule for "fuseable_pattern: gelu_erf"')
