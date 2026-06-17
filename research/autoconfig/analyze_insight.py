# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""analyze_insight.py — Phase 1 Insight Engine for autoconfig.

Fuses three signals to build skip_set and priority_queue:
  1. Graph analysis  : op counts, Conv%, GELU variant, dynamic axes
  2. winml analyze   : partial/unsupported op list per EP (static rule data)
  3. ep_knowledge KB : confirmed empirical findings (skip_passes, priority hints)

Outputs:
  InsightResult.skip_set         — set of hypothesis labels to prune
  InsightResult.priority_boosts  — {hypothesis_label: boost_score} for reordering
  InsightResult.notes            — human-readable explanation of each decision
"""

from __future__ import annotations

import json
import re
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

from bench_utils import run_cmd


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


# ── graph analysis ────────────────────────────────────────────────────────────


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
            info.gemm_pct = counts.get("Gemm", 0) / total * 100
            info.transpose_count = counts.get("Transpose", 0)

        # Decomposed GELU: presence of Erf node with Div predecessor
        out2node = {o: n for n in g.node for o in n.output}
        for n in g.node:
            if n.op_type == "Erf" and n.input:
                pred = out2node.get(n.input[0])
                if pred and pred.op_type == "Div":
                    info.has_gelu_decomposed = True
                    break

        # Dynamic axes: any input with dim_param (string dimension)
        for inp in g.input:
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_param:
                    info.has_dynamic_axes = True
                    break

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
            f"       total_ops={g.total_ops}  conv_pct={g.conv_pct:.1f}%  "
            f"gemm_pct={g.gemm_pct:.1f}%  transpose={g.transpose_count}"
        )
        print(f"       top ops: {dict(top5)}")
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

        # GELU-decomposed: boost gelu-fusion hypothesis
        if g.has_gelu_decomposed:
            for label, _, dim in hypotheses:
                if "gelu" in label.lower():
                    result.priority_boosts[label] = result.priority_boosts.get(label, 0) + 3
                    notes.append(
                        f"boost [{label}]: decomposed GELU detected — fusion likely beneficial"
                    )

        # High Gemm% → matmul-add-fusion more likely to help
        if g.gemm_pct > 30:
            for label, _, dim in hypotheses:
                if "matmul" in label.lower() or "gemm" in label.lower():
                    result.priority_boosts[label] = result.priority_boosts.get(label, 0) + 2
                    notes.append(
                        f"boost [{label}]: high Gemm% ({g.gemm_pct:.1f}%) — matmul fusion promising"
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
