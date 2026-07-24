# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""analyze_insight.py — Phase 1 Insight Engine for autoconfig.

Fuses three signals to build skip_set and priority_queue:
  1. Graph analysis  : op counts, Conv%, GELU variant, dynamic axes
  2. winml analyze   : partial/unsupported op list per EP (static rule data)
  3. ep_device_knowledge KB : confirmed empirical findings (skip_passes, priority hints)

Outputs:
  InsightResult.skip_set         — set of hypothesis labels to prune
  InsightResult.priority_boosts  — {hypothesis_label: boost_score} for reordering
  InsightResult.notes            — human-readable explanation of each decision
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Optional heavy imports — gracefully degrade if not available
try:
    import onnx  # type: ignore[import-untyped]

    _ONNX_OK = True
except ImportError:
    _ONNX_OK = False

# Agent package bootstrap: make the autoconfig root importable for sibling packages.
_AGENT_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "ep_device_knowledge").is_dir()
)
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from skills.optimizer.bench_utils import run_cmd  # noqa: E402


# ── data types ────────────────────────────────────────────────────────────────


@dataclass
class GraphInfo:
    total_ops: int = 0
    op_counts: dict[str, int] = field(default_factory=dict)
    conv_pct: float = 0.0  # Conv / total_ops  (0-100)
    gemm_pct: float = 0.0  # Gemm / total_ops
    has_gelu_decomposed: bool = False  # Erf-based GELU sub-pattern
    has_dynamic_axes: bool = False
    transpose_count: int = 0
    available: bool = False  # False when onnx not installed or model not found


@dataclass
class AnalyzeResult:
    supported: list[str] = field(default_factory=list)
    partial: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    available: bool = False  # False when winml analyze failed or ep has no rule data


@dataclass
class InsightResult:
    skip_set: set[str] = field(default_factory=set)
    """Labels from HYPOTHESES that should be pruned before the search loop."""

    priority_boosts: dict[str, float] = field(default_factory=dict)
    """hypothesis_label -> boost (positive = higher priority, negative = deprioritise)."""

    notes: list[str] = field(default_factory=list)
    """Human-readable explanation for each decision."""

    graph_info: GraphInfo = field(default_factory=GraphInfo)
    analyze_result: AnalyzeResult = field(default_factory=AnalyzeResult)


# ── data types ────────────────────────────────────────────────────────────────


@dataclass
class FusionCandidate:
    """One detectable pattern that maps to a winml optimize flag."""

    flag: str
    """winml optimize flag name (e.g. 'gelu_fusion')."""

    count: int
    """How many candidate instances were found in the graph."""

    evidence: str
    """Short human-readable description of what was found."""


@dataclass
class GraphInfo:
    total_ops: int = 0
    op_counts: dict[str, int] = field(default_factory=dict)
    conv_pct: float = 0.0  # Conv / total_ops  (0-100)
    matmul_pct: float = 0.0  # MatMul / total_ops
    gemm_pct: float = 0.0  # Gemm / total_ops
    has_gelu_decomposed: bool = False  # any multi-op GELU subgraph detected
    gelu_types: list[str] = field(default_factory=list)  # 'erf', 'tanh', 'quick'
    has_dynamic_axes: bool = False
    transpose_count: int = 0
    fusion_candidates: list[FusionCandidate] = field(default_factory=list)
    """Ordered list of detected optimisation opportunities, highest-count first."""
    available: bool = False  # False when onnx not installed or model not found


@dataclass
class AnalyzeResult:
    supported: list[str] = field(default_factory=list)
    partial: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    available: bool = False  # False when winml analyze failed or ep has no rule data


@dataclass
class InsightResult:
    skip_set: set[str] = field(default_factory=set)
    """Labels from HYPOTHESES that should be pruned before the search loop."""

    priority_boosts: dict[str, float] = field(default_factory=dict)
    """hypothesis_label -> boost (positive = higher priority, negative = deprioritise)."""

    notes: list[str] = field(default_factory=list)
    """Human-readable explanation for each decision."""

    graph_info: GraphInfo = field(default_factory=GraphInfo)
    analyze_result: AnalyzeResult = field(default_factory=AnalyzeResult)


# ── graph analysis ────────────────────────────────────────────────────────────


def _build_consumer_map(graph) -> dict[str, list]:  # type: ignore[type-arg]
    """Map each output name → list of consumer nodes."""
    consumers: dict[str, list] = {}
    for node in graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)
    return consumers


def _build_producer_map(graph) -> dict[str, object]:
    """Map each output name → the node that produces it."""
    return {out: n for n in graph.node for out in n.output}


def _get_attr_float(node, name: str) -> float | None:
    """Extract a float attribute from an ONNX node."""
    for a in node.attribute:
        if a.name == name:
            return float(a.f)
    return None


def _detect_fusion_candidates(graph) -> list[FusionCandidate]:  # type: ignore[type-arg]
    """
    Scan the ONNX graph for subgraph patterns that map to winml optimize flags.

    Returns a list of FusionCandidate, ordered highest-count first.

    Detection strategy
    ------------------
    We build two lookup tables (producer_map, consumer_map) and then sweep the
    graph once per pattern family.  Each check is O(N) in the number of nodes.

    Pattern families
    ----------------
    GELU variants
        gelu_fusion         : Div → Erf → Add → Mul → Mul  (exact GELU)
        fast_gelu_fusion    : Tanh-based GELU  (Tanh node with Pow(3) ancestor)
        quick_gelu_fusion   : x * sigmoid(1.702*x)
        bias_gelu_fusion    : Add → GELU subgraph  (bias before GELU entry)
    LayerNorm variants
        layer_norm_fusion         : ReduceMean → Sub → Pow(2) → … → Add(ε)
        simplified_layer_norm_fusion : Pow(2) + ReduceMean (no Sub)
        fuse_rmsnorm              : Pow → ReduceMean → Add → Sqrt → Div → Mul
        skip_layer_norm_fusion    : Add (residual) feeds directly into LN subgraph
    Attention
        attention_fusion    : Q/K/V MatMul trio feeding a Softmax
        bias_softmax_fusion : Add immediately before Softmax
    MatMul patterns
        matmul_add_fusion         : MatMul → Add (not already counted in LN)
        matmul_activation_fusion  : MatMul → {Relu, Sigmoid, Tanh, Clip}
        matmul_transpose_fusion   : Transpose → MatMul  OR  MatMul → Transpose
        matmul_scale_fusion       : MatMul → Mul (scalar constant)
    Conv patterns
        conv_bn_fusion            : Conv → BatchNormalization
        conv_add_fusion           : Conv → Add
        conv_mul_fusion           : Conv → Mul
        conv_activation_fusion    : Conv → {Relu, LeakyRelu, Sigmoid, Tanh, Clip}
        conv_add_activation_fusion: Conv → Add → activation  (3-node chain)
        pad_fusion                : Pad → Conv
    Gemm patterns
        gemm_activation_fusion    : Gemm → {Relu, Tanh, Sigmoid}
        gemm_sum_fusion           : Gemm → Add
        gemm_transpose_fusion     : Transpose → Gemm
    Eliminations
        slice_elimination         : multiple Slice ops (potential redundancy)
        unsqueeze_elimination     : Unsqueeze of initializers
        concat_slice_elimination  : Concat → Slice (reverse of split)
        expand_elimination        : Expand nodes
    Layout
        transpose_optimizer       : Transpose count > 10
        nhwc_transformer          : Conv-heavy + Transpose → layout transform candidate
    Rewrite: highdimRTR_lowdimRTR : Reshape → Transpose → Reshape  with rank > 4
    """
    producer = _build_producer_map(graph)
    consumer = _build_consumer_map(graph)

    # Helper: get the single consumer of a node output (or None)
    def _single_consumer(node, out_idx: int = 0):
        if out_idx >= len(node.output):
            return None
        consumers = consumer.get(node.output[out_idx], [])
        return consumers[0] if len(consumers) == 1 else None

    # Helper: check if a node output feeds a specific op type
    def _consumer_is(node, op: str, out_idx: int = 0) -> bool:
        c = _single_consumer(node, out_idx)
        return c is not None and c.op_type == op

    # Helper: check if all inputs to node are exclusively from initializers (weight-only)
    init_names = {i.name for i in graph.initializer}

    def _is_initializer_input(inp_name: str) -> bool:
        return inp_name in init_names

    candidates: dict[str, FusionCandidate] = {}

    def _add(flag: str, evidence: str, n: int = 1) -> None:
        if flag in candidates:
            candidates[flag].count += n
            candidates[flag].evidence = evidence  # update to latest
        else:
            candidates[flag] = FusionCandidate(flag=flag, count=n, evidence=evidence)

    # ── GELU patterns ──────────────────────────────────────────────────────────
    erf_gelu_count = 0
    tanh_gelu_count = 0
    quick_gelu_count = 0
    bias_before_gelu = 0

    for node in graph.node:
        # Erf-based GELU: Div → Erf → (Add → Mul → Mul)
        if node.op_type == "Erf" and node.input:
            pred = producer.get(node.input[0])
            if pred and pred.op_type == "Div":
                erf_gelu_count += 1
                # Check if there's an Add feeding the Erf entry point (bias_gelu)
                # The entry to Erf-GELU is typically through the Div; check what feeds Div
                if pred.input:
                    div_pred = producer.get(pred.input[0])
                    if div_pred and div_pred.op_type in ("Add", "Gemm", "MatMul"):
                        bias_before_gelu += 1

        # Tanh-based GELU: Tanh with Pow(3) somewhere in the sub-tree
        if node.op_type == "Tanh" and node.input:
            # Check 3-hop ancestry for Pow
            cur = producer.get(node.input[0])
            for _ in range(4):
                if cur is None:
                    break
                if cur.op_type == "Pow":
                    tanh_gelu_count += 1
                    break
                cur = producer.get(cur.input[0]) if cur.input else None

        # Quick GELU: Sigmoid where predecessor is Mul with constant ≈ 1.702
        if node.op_type == "Sigmoid" and node.input:
            pred = producer.get(node.input[0])
            if pred and pred.op_type == "Mul":
                quick_gelu_count += 1

    if erf_gelu_count:
        _add("gelu_fusion", f"{erf_gelu_count} Erf-based GELU subgraph(s)", erf_gelu_count)
        _add(
            "gelu_singlegelu",
            f"{erf_gelu_count} decomposed GELU → can normalise to single Gelu op",
            erf_gelu_count,
        )
    if tanh_gelu_count:
        _add(
            "fast_gelu_fusion",
            f"{tanh_gelu_count} Tanh-based GELU subgraph(s)",
            tanh_gelu_count,
        )
    if quick_gelu_count:
        _add(
            "quick_gelu_fusion",
            f"{quick_gelu_count} Sigmoid(1.702x) quick-GELU pattern(s)",
            quick_gelu_count,
        )
    if bias_before_gelu:
        _add(
            "bias_gelu_fusion",
            f"{bias_before_gelu} Add/MatMul feeding GELU entry",
            bias_before_gelu,
        )

    # ── LayerNorm patterns ─────────────────────────────────────────────────────
    ln_full_count = 0  # ReduceMean + Sub + Pow(2)
    ln_simplified_count = 0  # Pow(2) + ReduceMean (no Sub)
    rmsnorm_count = 0  # Pow + ReduceMean (no Sub, no mean-centering)
    skip_ln_count = 0  # Add → LayerNorm subgraph

    for node in graph.node:
        if node.op_type == "Pow" and node.input:
            pred = producer.get(node.input[0])
            if pred and pred.op_type == "Sub":
                # Sub → Pow: classic LN  (ReduceMean → Sub → Pow)
                sub_pred = producer.get(pred.input[0]) if pred.input else None
                if sub_pred and sub_pred.op_type == "ReduceMean":
                    ln_full_count += 1
            elif pred and pred.op_type in ("ReduceMean", "Mul", "Add"):
                # Simplified / RMSNorm: no Sub predecessor
                ln_simplified_count += 1

        # RMSNorm: Pow → ReduceMean (direct, without Sub)
        if node.op_type == "ReduceMean" and node.input:
            pred = producer.get(node.input[0])
            if pred and pred.op_type == "Pow":
                rmsnorm_count += 1

        # skip_layer_norm: Add whose output feeds into the start of an LN subgraph
        # Heuristic: Add → ReduceMean (the mean-centering step of LN)
        if node.op_type == "Add" and _consumer_is(node, "ReduceMean"):
            skip_ln_count += 1

    if ln_full_count:
        _add(
            "layer_norm_fusion",
            f"{ln_full_count} ReduceMean→Sub→Pow LayerNorm subgraph(s)",
            ln_full_count,
        )
    if ln_simplified_count:
        _add(
            "simplified_layer_norm_fusion",
            f"{ln_simplified_count} simplified LayerNorm pattern(s) (no mean-centering)",
            ln_simplified_count,
        )
    if rmsnorm_count:
        _add("fuse_rmsnorm", f"{rmsnorm_count} RMSNorm Pow→ReduceMean pattern(s)", rmsnorm_count)
    if skip_ln_count:
        _add(
            "skip_layer_norm_fusion",
            f"{skip_ln_count} Add→ReduceMean (residual+LN) pattern(s)",
            skip_ln_count,
        )

    # ── Attention patterns ─────────────────────────────────────────────────────
    softmax_count = sum(1 for n in graph.node if n.op_type == "Softmax")
    add_before_softmax = 0
    for node in graph.node:
        if node.op_type == "Softmax" and node.input:
            pred = producer.get(node.input[0])
            if pred and pred.op_type == "Add":
                add_before_softmax += 1

    if softmax_count:
        _add(
            "attention_fusion",
            f"{softmax_count} Softmax node(s) — likely attention head(s)",
            softmax_count,
        )
    if add_before_softmax:
        _add(
            "bias_softmax_fusion",
            f"{add_before_softmax} Add→Softmax (bias+attention mask) pattern(s)",
            add_before_softmax,
        )

    # ── MatMul patterns ────────────────────────────────────────────────────────
    _ACTIVATIONS = {"Relu", "LeakyRelu", "Sigmoid", "Tanh", "Clip", "Gelu", "FastGelu"}

    mm_add = mm_act = mm_tp = mm_scale = 0
    for node in graph.node:
        if node.op_type != "MatMul":
            continue
        c = _single_consumer(node)
        if c is None:
            continue
        if c.op_type == "Add":
            mm_add += 1
        elif c.op_type in _ACTIVATIONS:
            mm_act += 1
        elif c.op_type == "Transpose":
            mm_tp += 1
        elif c.op_type == "Mul":
            # Mul with a scalar → scale fusion; heuristic: second input is initializer
            if len(c.input) > 1 and _is_initializer_input(c.input[1]):
                mm_scale += 1

    # Also check Transpose → MatMul
    tp_before_mm = sum(
        1 for node in graph.node if node.op_type == "Transpose" and _consumer_is(node, "MatMul")
    )

    if mm_add:
        _add("matmul_add_fusion", f"{mm_add} MatMul→Add pattern(s)", mm_add)
        _add(
            "matmuladd_reshapegemm",
            f"{mm_add} MatMul+Add → Reshape+Gemm rewrite candidate(s)",
            mm_add,
        )
    if mm_act:
        _add("matmul_activation_fusion", f"{mm_act} MatMul→activation pattern(s)", mm_act)
    if mm_tp + tp_before_mm:
        _add(
            "matmul_transpose_fusion",
            f"{mm_tp + tp_before_mm} MatMul↔Transpose pattern(s)",
            mm_tp + tp_before_mm,
        )
    if mm_scale:
        _add("matmul_scale_fusion", f"{mm_scale} MatMul→Mul(scalar) pattern(s)", mm_scale)

    # ── Conv patterns ──────────────────────────────────────────────────────────
    conv_bn = conv_add = conv_mul = conv_act = conv_add_act = pad_conv = 0
    for node in graph.node:
        if node.op_type == "Pad" and _consumer_is(node, "Conv"):
            pad_conv += 1

        if node.op_type != "Conv":
            continue
        c = _single_consumer(node)
        if c is None:
            continue
        if c.op_type == "BatchNormalization":
            conv_bn += 1
        elif c.op_type == "Add":
            conv_add += 1
            # Check for Conv → Add → activation chain
            cc = _single_consumer(c)
            if cc and cc.op_type in _ACTIVATIONS:
                conv_add_act += 1
        elif c.op_type == "Mul":
            conv_mul += 1
        elif c.op_type in _ACTIVATIONS:
            conv_act += 1

    if conv_bn:
        _add("conv_bn_fusion", f"{conv_bn} Conv→BN pattern(s)", conv_bn)
    if conv_add:
        _add("conv_add_fusion", f"{conv_add} Conv→Add pattern(s)", conv_add)
    if conv_mul:
        _add("conv_mul_fusion", f"{conv_mul} Conv→Mul pattern(s)", conv_mul)
    if conv_act:
        _add("conv_activation_fusion", f"{conv_act} Conv→activation pattern(s)", conv_act)
    if conv_add_act:
        _add(
            "conv_add_activation_fusion",
            f"{conv_add_act} Conv→Add→activation chain(s) (FusedConv)",
            conv_add_act,
        )
    if pad_conv:
        _add("pad_fusion", f"{pad_conv} Pad→Conv pattern(s)", pad_conv)

    # ── Gemm patterns ──────────────────────────────────────────────────────────
    gemm_act = gemm_add = gemm_tp = 0
    for node in graph.node:
        if node.op_type != "Gemm":
            continue
        c = _single_consumer(node)
        if c is None:
            continue
        if c.op_type in _ACTIVATIONS:
            gemm_act += 1
        elif c.op_type == "Add":
            gemm_add += 1
        elif c.op_type == "Transpose":
            gemm_tp += 1
    tp_before_gemm = sum(
        1 for node in graph.node if node.op_type == "Transpose" and _consumer_is(node, "Gemm")
    )
    if gemm_act:
        _add("gemm_activation_fusion", f"{gemm_act} Gemm→activation pattern(s)", gemm_act)
    if gemm_add:
        _add("gemm_sum_fusion", f"{gemm_add} Gemm→Add pattern(s)", gemm_add)
    if gemm_tp + tp_before_gemm:
        _add(
            "gemm_transpose_fusion",
            f"{gemm_tp + tp_before_gemm} Gemm↔Transpose pattern(s)",
            gemm_tp + tp_before_gemm,
        )

    # ── Elimination patterns ───────────────────────────────────────────────────
    slice_count = sum(1 for n in graph.node if n.op_type == "Slice")
    expand_count = sum(1 for n in graph.node if n.op_type == "Expand")
    unsqueeze_init = sum(
        1
        for n in graph.node
        if n.op_type == "Unsqueeze" and n.input and _is_initializer_input(n.input[0])
    )
    concat_slice = sum(1 for n in graph.node if n.op_type == "Concat" and _consumer_is(n, "Slice"))

    if slice_count > 3:
        _add("slice_elimination", f"{slice_count} Slice nodes (potential redundancy)", slice_count)
    if expand_count > 2:
        _add("expand_elimination", f"{expand_count} Expand nodes", expand_count)
    if unsqueeze_init:
        _add(
            "unsqueeze_elimination",
            f"{unsqueeze_init} Unsqueeze(initializer) node(s)",
            unsqueeze_init,
        )
    if concat_slice:
        _add(
            "concat_slice_elimination",
            f"{concat_slice} Concat→Slice pattern(s) (reverse-split)",
            concat_slice,
        )

    # ── Layout patterns ────────────────────────────────────────────────────────
    tp_count = sum(1 for n in graph.node if n.op_type == "Transpose")
    if tp_count > 10:
        _add(
            "transpose_optimizer",
            f"{tp_count} Transpose nodes — optimizer may collapse chains",
            tp_count,
        )

    # Reshape → Transpose → Reshape with high-dimensional input (rank > 4)
    rtr_highdim = 0
    for node in graph.node:
        if node.op_type == "Transpose" and node.input:
            pred = producer.get(node.input[0])
            c = _single_consumer(node)
            if pred and c and pred.op_type == "Reshape" and c.op_type == "Reshape":
                # Check if any input to the reshape has rank > 4 via shape inference
                # Approximation: count as candidate if the graph has many dims
                rtr_highdim += 1
    if rtr_highdim > 2:
        _add(
            "highdimRTR_lowdimRTR",
            f"{rtr_highdim} Reshape→Transpose→Reshape chain(s) — may reduce to lower rank",
            rtr_highdim,
        )

    # Sort by count descending
    return sorted(candidates.values(), key=lambda c: -c.count)


def run_graph_analysis(onnx_path: Path) -> GraphInfo:
    """Analyse the ONNX proto and return structural statistics."""
    info = GraphInfo()
    if not _ONNX_OK:
        return info
    if not onnx_path.exists():
        return info

    try:
        model = onnx.load(str(onnx_path))
        g = model.graph
        counts: Counter = Counter(n.op_type for n in g.node)
        total = sum(counts.values())
        info.total_ops = total
        info.op_counts = dict(counts)
        info.available = True

        if total > 0:
            info.conv_pct = counts.get("Conv", 0) / total * 100
            info.matmul_pct = counts.get("MatMul", 0) / total * 100
            info.gemm_pct = counts.get("Gemm", 0) / total * 100
            info.transpose_count = counts.get("Transpose", 0)

        # Detect GELU types
        if counts.get("Erf", 0):
            info.has_gelu_decomposed = True
            info.gelu_types.append("erf")
        if counts.get("Tanh", 0):
            info.gelu_types.append("tanh")
        if counts.get("Sigmoid", 0):
            info.gelu_types.append("sigmoid/quick")

        # Dynamic axes: any input with dim_param (string dimension)
        for inp in g.input:
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_param:
                    info.has_dynamic_axes = True
                    break

        # Full fusion candidate scan
        info.fusion_candidates = _detect_fusion_candidates(g)

    except Exception as e:
        info.available = False
        print(f"  [analyze_insight] graph analysis failed: {e}")

    return info


# ── winml analyze ─────────────────────────────────────────────────────────────


def run_winml_analyze(winml: str, onnx_path: Path, ep: str, device: str) -> AnalyzeResult:
    """Call `winml analyze -m <path> --ep <ep>` and parse JSON output."""
    result = AnalyzeResult()
    if not onnx_path.exists():
        return result

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = Path(f.name)

    try:
        rc, out, _ = run_cmd(
            [
                winml,
                "analyze",
                "-m",
                str(onnx_path),
                "--ep",
                ep,
                "--device",
                device,
                "-o",
                str(out_path),
            ],
            label=f"winml analyze --ep {ep}",
            timeout=120,
        )
        if rc not in (0, 1) or not out_path.exists():
            return result

        data = json.loads(out_path.read_text(encoding="utf-8"))
        # Output is a list; take first entry (single-EP mode)
        entry = data[0] if isinstance(data, list) and data else data
        ep_results = entry.get("results", [])
        if not ep_results:
            return result

        ep_res = ep_results[0]
        cls = ep_res.get("classification", {})

        def _extract_op_types(lst: list[str]) -> list[str]:
            """Turn 'OP/ai.onnx/Conv (QDQ)' into 'Conv'."""
            types = []
            for s in lst:
                m = re.search(r"/([A-Za-z][A-Za-z0-9_]*)(?:\s|$|\()", s)
                if m:
                    types.append(m.group(1))
            return list(dict.fromkeys(types))  # dedupe, preserve order

        result.supported = _extract_op_types(cls.get("supported", []))
        result.partial = _extract_op_types(cls.get("partial", []))
        result.unsupported = _extract_op_types(cls.get("unsupported", []))
        result.unknown = _extract_op_types(cls.get("unknown", []))
        # Consider results available only when there's actual rule data
        result.available = bool(result.supported or result.partial or result.unsupported)

    except Exception as e:
        print(f"  [analyze_insight] winml analyze failed: {e}")
    finally:
        out_path.unlink(missing_ok=True)

    return result


# ── graph-presence pruning ──────────────────────────────────────────────────

# Map each OFAT graph pass to the baseline-graph pattern it needs to fire.
# For "pattern" passes the detector emits an exact-named FusionCandidate ONLY
# when the subgraph is actually present (no count threshold), so a count of 0 is
# a confident "this pass would be a no-op" signal we can prune on.
_PASS_PATTERN_FLAG: dict[str, str] = {
    "conv_bn_fusion": "conv_bn_fusion",
    "conv_add_fusion": "conv_add_fusion",
    "conv_activation_fusion": "conv_activation_fusion",
    "gelu_fusion": "gelu_fusion",
    "layer_norm_fusion": "layer_norm_fusion",
    "skip_layer_norm_fusion": "skip_layer_norm_fusion",
    "matmul_add_fusion": "matmul_add_fusion",
    "matmul_transpose_fusion": "matmul_transpose_fusion",
    "attention_fusion": "attention_fusion",
    "bias_softmax_fusion": "bias_softmax_fusion",
}


def _pass_name_of(label: str) -> str:
    """Extract the single graph-pass name from an 'opset=NN + pass' hypothesis label."""
    return label.split("+")[-1].strip()


def _pass_can_fire(pass_name: str, g: GraphInfo, present: dict[str, int]) -> bool | None:
    """Pre-estimate, from the baseline graph, whether a single pass can change it.

    Returns True (required pattern present), False (confidently absent → the pass
    is a guaranteed no-op), or None (pass not statically estimable → leave it to
    the empirical search rather than risk a false cut).
    """
    if pass_name in _PASS_PATTERN_FLAG:
        return present.get(_PASS_PATTERN_FLAG[pass_name], 0) > 0
    # layout / rewrite passes: fall back to the primitive op the pass operates on.
    if pass_name == "transpose_optimizer":
        return g.transpose_count > 0
    if pass_name == "highdimRTR_lowdimRTR":
        return g.transpose_count > 0 and g.op_counts.get("Reshape", 0) > 0
    if pass_name == "nchwc_transformer":
        return g.op_counts.get("Conv", 0) > 0
    return None


# ── insight engine ────────────────────────────────────────────────────────────


def build_insight(
    onnx_path: Path,
    winml: str,
    ep: str,
    device: str,
    hypotheses: list[tuple[str, Any, str]],
    kb: dict,
) -> InsightResult:
    """Fuse graph + analyze + KB signals into skip_set and priority_boosts.

    Args:
        onnx_path:   Path to baseline ONNX (post-export, pre-optim).
        winml:       Path to winml executable.
        ep:          Execution provider string (e.g. "cpu", "qnn").
        device:      Device string (e.g. "cpu", "npu").
        hypotheses:  List of (label, patch_fn, dimension) from autoconfig.py.
        kb:          dict from load_ep_knowledge(ep).

    Returns:
        InsightResult with skip_set, priority_boosts, notes.
    """
    result = InsightResult()
    notes = result.notes

    print("\n=== Phase 1: Insight Engine ===")

    # ── signal 1: graph analysis ───────────────────────────────
    print("  [1/3] Graph analysis…")
    g = run_graph_analysis(onnx_path)
    result.graph_info = g
    if g.available:
        top5 = sorted(g.op_counts.items(), key=lambda x: -x[1])[:5]
        print(
            f"       total_ops={g.total_ops}  conv%={g.conv_pct:.1f}  "
            f"matmul%={g.matmul_pct:.1f}  gemm%={g.gemm_pct:.1f}  "
            f"transpose={g.transpose_count}  dynamic_axes={g.has_dynamic_axes}"
        )
        print(f"       top ops: {dict(top5)}")
        if g.fusion_candidates:
            print(f"       fusion candidates ({len(g.fusion_candidates)}):")
            for fc in g.fusion_candidates[:10]:  # top-10 only
                print(f"         [{fc.count:3d}×] {fc.flag:40s}  {fc.evidence}")
            if len(g.fusion_candidates) > 10:
                print(f"         ... and {len(g.fusion_candidates) - 10} more")
    else:
        print("       [skip] onnx not available or model not found")

    # ── signal 2: winml analyze ────────────────────────────────
    print(f"  [2/3] winml analyze --ep {ep}…")
    ar = run_winml_analyze(winml, onnx_path, ep, device)
    result.analyze_result = ar
    if ar.available:
        print(
            f"       supported={len(ar.supported)}  partial={len(ar.partial)}  "
            f"unsupported={len(ar.unsupported)}  unknown={len(ar.unknown)}"
        )
        if ar.partial:
            print(f"       partial ops: {ar.partial[:5]}")
        if ar.unsupported:
            print(f"       unsupported ops: {ar.unsupported[:5]}")
    else:
        print("       [skip] no rule data for this EP or analyze failed")

    # ── signal 3: KB confirmed rules ───────────────────────────
    print("  [3/3] Applying KB confirmed rules…")

    # ── build skip_set ─────────────────────────────────────────

    # KB-derived skips (already applied per confirmed finding)
    for note in kb.get("notes", []):
        if "[KB confirmed] Skip pass:" in note:
            pass_name = note.split("Skip pass:")[-1].strip()
            # Match against hypothesis labels that use this pass
            for label, _, _ in hypotheses:
                if pass_name.replace("_", "-") in label or pass_name in label:
                    result.skip_set.add(label)
                    notes.append(f"skip [{label}]: KB confirmed rule — {pass_name}")

    # Graph-derived skips
    if g.available:
        # npu-006: Conv% > 20% → hard-block conv fusions on QNN NPU
        if ep in ("qnn",) and device == "npu" and g.conv_pct > 20.0:
            for label, _, dim in hypotheses:
                if dim == "graph_pass" and any(kw in label for kw in ("conv", "bn", "batch")):
                    result.skip_set.add(label)
                    notes.append(
                        f"skip [{label}]: npu-006 — Conv%={g.conv_pct:.1f}%>20% on QNN NPU"
                        " (FusedConv → CPU fallback)"
                    )

        # cpu-001: opset > 17 regresses on CPU (empirical, mechanism unknown)
        if ep == "cpu":
            for label, _, dim in hypotheses:
                if dim == "opset" and "21" in label:
                    notes.append(
                        f"deprioritise [{label}]: cpu-001 — opset21 regresses on CPU"
                        " (non-monotonic, mechanism unknown)"
                    )
                    result.priority_boosts[label] = result.priority_boosts.get(label, 0) - 5

        # gpu-004: QNN GPU — skip all quantization
        if ep == "qnn" and device == "gpu":
            for label, _, dim in hypotheses:
                if dim in ("quant", "precision"):
                    result.skip_set.add(label)
                    notes.append(f"skip [{label}]: gpu-004 — quantization hangs on QNN GPU")

        # nhwc-transformer regresses p90 on DML/QNN GPU transformers
        if ep in ("dml",) or (ep == "qnn" and device == "gpu"):
            for label, _, dim in hypotheses:
                if "nhwc" in label.lower():
                    result.skip_set.add(label)
                    notes.append(
                        f"skip [{label}]: dml-002/gpu-002 — nhwc-transformer increases p90 variance"
                    )

        # graph-presence pruning (static pre-estimate): cut graph-pass hypotheses
        # whose required pattern is absent from the baseline graph. With nothing to
        # fuse the pass is a guaranteed no-op, so there is no point benchmarking it.
        # Passes we cannot statically estimate (_pass_can_fire → None) are left for
        # the empirical search rather than risk a false cut.
        present_flags = {fc.flag: fc.count for fc in g.fusion_candidates}
        for label, _, dim in hypotheses:
            if dim != "graph_pass":
                continue
            if _pass_can_fire(_pass_name_of(label), g, present_flags) is False:
                result.skip_set.add(label)
                notes.append(
                    f"skip [{label}]: graph analysis — required pattern absent in the"
                    " baseline graph (pass would be a no-op, nothing to fuse)"
                )

    # ── build priority_boosts ──────────────────────────────────

    if g.available:
        # DINOv2-family on QNN NPU: opset21 gets strong positive boost (npu-001)
        if ep == "qnn" and device == "npu":
            # Heuristic: DINOv2 has many Reshape and high attention ops
            if g.op_counts.get("Reshape", 0) > 30 and g.conv_pct < 10:
                for label, _, dim in hypotheses:
                    if dim == "opset" and "21" in label:
                        result.priority_boosts[label] = result.priority_boosts.get(label, 0) + 10
                        notes.append(
                            f"boost [{label}]: npu-001 heuristic — high Reshape count"
                            f" ({g.op_counts.get('Reshape', 0)}) + low Conv% suggests DINOv2-family"
                        )

        # Fusion-candidate-driven boosts: map detected patterns → hypothesis labels
        #
        # Strategy: for each FusionCandidate, find hypotheses whose label or dimension
        # mentions the relevant flag.  Boost proportional to log(count) so that
        # "288 MatMul→Add" doesn't overwhelm "12 GELU" by 24×.
        import math

        _FLAG_KEYWORDS: dict[str, list[str]] = {
            "gelu_fusion": ["gelu"],
            "fast_gelu_fusion": ["gelu", "fast"],
            "bias_gelu_fusion": ["gelu", "bias"],
            "quick_gelu_fusion": ["gelu", "quick"],
            "gelu_singlegelu": ["gelu"],
            "layer_norm_fusion": ["layer_norm", "layernorm", "ln"],
            "skip_layer_norm_fusion": ["skip_layer_norm", "skip_ln"],
            "simplified_layer_norm_fusion": ["layer_norm", "simplified"],
            "fuse_rmsnorm": ["rmsnorm", "rms_norm"],
            "attention_fusion": ["attention"],
            "bias_softmax_fusion": ["softmax", "attention"],
            "matmul_add_fusion": ["matmul_add", "matmul-add"],
            "matmul_activation_fusion": ["matmul_act", "matmul-act"],
            "matmul_transpose_fusion": ["matmul_transp", "matmul-transp"],
            "matmul_scale_fusion": ["matmul_scale", "matmul-scale"],
            "matmuladd_reshapegemm": ["reshape_gemm", "matmuladd"],
            "conv_bn_fusion": ["conv_bn", "conv-bn"],
            "conv_add_fusion": ["conv_add", "conv-add"],
            "conv_mul_fusion": ["conv_mul", "conv-mul"],
            "conv_activation_fusion": ["conv_act", "conv-act"],
            "conv_add_activation_fusion": ["conv_add_act", "fused_conv"],
            "pad_fusion": ["pad_conv", "pad-conv"],
            "gemm_activation_fusion": ["gemm_act", "gemm-act"],
            "gemm_sum_fusion": ["gemm_sum", "gemm-sum"],
            "gemm_transpose_fusion": ["gemm_transp"],
            "slice_elimination": ["slice_elim"],
            "unsqueeze_elimination": ["unsqueeze_elim"],
            "expand_elimination": ["expand_elim"],
            "concat_slice_elimination": ["concat_slice"],
            "transpose_optimizer": ["transpose_opt", "tp_opt"],
            "highdimRTR_lowdimRTR": ["rtr", "reshape_transpose"],
        }

        for fc in g.fusion_candidates:
            keywords = _FLAG_KEYWORDS.get(fc.flag, [fc.flag.replace("_", "-")])
            boost = round(1 + math.log(max(fc.count, 1)), 1)
            for label, _, dim in hypotheses:
                label_lower = label.lower()
                if any(kw in label_lower for kw in keywords):
                    result.priority_boosts[label] = result.priority_boosts.get(label, 0) + boost
                    notes.append(
                        f"boost [{label}] +{boost:.1f}: graph has {fc.count}× {fc.flag} candidate(s)"
                    )

        # GELU-decomposed: additional direct boost for gelu hypotheses
        if g.has_gelu_decomposed:
            for label, _, dim in hypotheses:
                if "gelu" in label.lower() and label not in {
                    n.split("]")[0].lstrip("boost [") for n in notes if "gelu" in n
                }:
                    result.priority_boosts[label] = result.priority_boosts.get(label, 0) + 2
                    notes.append(
                        f"boost [{label}]: decomposed GELU detected — fusion likely beneficial"
                    )

        # Conv-dense → conv fusions more likely to help (CPU only — not QNN NPU)
        if g.conv_pct > 40 and ep not in ("qnn",):
            for label, _, dim in hypotheses:
                if "conv" in label.lower() and dim == "graph_pass":
                    result.priority_boosts[label] = result.priority_boosts.get(label, 0) + 2
                    notes.append(
                        f"boost [{label}]: high Conv% ({g.conv_pct:.1f}%) — conv fusions promising"
                    )

    # analyze-derived: if partial ops in model → deprioritise those optims
    if ar.available and ar.partial:
        for label, _, dim in hypotheses:
            for pop in ar.partial:
                if pop.lower() in label.lower():
                    result.priority_boosts[label] = result.priority_boosts.get(label, 0) - 2
                    notes.append(
                        f"deprioritise [{label}]: op '{pop}' is partial-support on {ep.upper()}"
                    )

    # ── print summary ──────────────────────────────────────────
    print("\n  Insight Engine result:")
    print(f"    skip_set ({len(result.skip_set)}): {result.skip_set or '(none)'}")
    boosts = {k: v for k, v in result.priority_boosts.items() if v != 0}
    print(f"    priority_boosts: {boosts or '(none)'}")
    if notes:
        print("    notes:")
        for n in notes:
            print(f"      - {n}")
    print()

    return result
