# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""SA pre/post comparison and EPContext diff logic for SA evaluation.

Uses the Python SA API (ONNXStaticAnalyzer) directly to run analysis and
extract optimization configuration. Avoids subprocess + hardcoded flag maps.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING

import onnx


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# SA Python API invocation
# ---------------------------------------------------------------------------


def run_sa_with_info(
    onnx_path: Path,
    output_path: Path,
    ep: str = "QNNExecutionProvider",
    device: str = "NPU",
) -> tuple[dict[str, str], dict, list[dict]]:
    """Run SA using Python API with information enabled.

    Uses ONNXStaticAnalyzer with enable_information=True. Saves the full
    SA JSON result to output_path.

    Args:
        onnx_path: Input ONNX model to analyze.
        output_path: Path to write the SA JSON result.

    Returns:
        (classifications, optim_config, info_items) where:
          - classifications: {pattern_id: "SUPPORTED"/"PARTIAL"/"UNSUPPORTED"/"UNKNOWN"}
          - optim_config: WinMLOptimizationConfig dict (e.g. {"gelu_fusion": True})
          - info_items: list of {pattern_id, explanation, has_actions} for non-SUPPORTED
    """
    from winml.modelkit.analyze import AnalyzerConfig, ONNXStaticAnalyzer

    config = AnalyzerConfig(enable_information=True)
    analyzer = ONNXStaticAnalyzer(config=config)
    result = analyzer.analyze(str(onnx_path), ep=ep, device=device)

    # Save full SA JSON
    output_path.write_text(result.to_json(), encoding="utf-8")

    # Extract classifications from QNN EP result
    classifications: dict[str, str] = {}
    info_items: list[dict] = []

    ep_found = False
    for ep_result in result.output.results:
        if ep_result.ep_type != ep:
            continue
        ep_found = True
        for level_enum, pid_list in ep_result.classification.items():
            level = level_enum.value.upper()
            for pid in pid_list:
                classifications[pid] = level
        info_items.extend(
            {
                "pattern_id": info.pattern_id,
                "explanation": info.explanation or "",
                "has_actions": bool(info.actions),
            }
            for info in ep_result.information
        )
        break

    if not ep_found:
        # No rule data for this EP/device — SA skipped the EP entirely.
        # Return empty classifications so callers can proceed without SA-driven
        # optimization (perf comparison across stages still works).
        import logging

        logging.getLogger(__name__).warning(
            "SA produced no results for EP=%s — no runtime rule data available. "
            "Returning empty classifications.",
            ep,
        )

    # Get optimization config from SA recommendations
    optim_config = dict(result.get_optimization_config(ep)) if ep_found else {}

    return classifications, optim_config, info_items


# ---------------------------------------------------------------------------
# SA JSON parsing (for cached results)
# ---------------------------------------------------------------------------


def parse_sa_json(json_path: Path, ep: str = "QNNExecutionProvider") -> dict[str, str]:
    """Parse winml analyze JSON output into {pattern_id: level}.

    Works for both subprocess-written JSON (lowercase keys in classification
    dict) and Python API-written JSON (SupportLevel enum serialized as
    lowercase strings).

    Returns empty dict if file is missing or the requested EP result not found.
    """
    if not json_path.exists():
        return {}

    try:
        sa_data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    result: dict[str, str] = {}
    for ep_result in sa_data.get("results", []):
        if ep_result.get("ep_type") != ep:
            continue
        cls = ep_result.get("classification", {})
        for level, pid_list in cls.items():
            level_upper = str(level).upper().split(".")[-1]  # handle "SupportLevel.SUPPORTED"
            if isinstance(pid_list, list):
                for pid in pid_list:
                    result[pid] = level_upper
        break

    return result


# ---------------------------------------------------------------------------
# SA summary
# ---------------------------------------------------------------------------


def get_sa_summary(classifications: dict[str, str]) -> dict:
    """Compute per-level counts and supported_ratio.

    Returns {"supported", "partial", "unsupported", "unknown", "total", "supported_ratio"}
    """
    counts: dict[str, int] = {"supported": 0, "partial": 0, "unsupported": 0, "unknown": 0}
    for level in classifications.values():
        key = level.lower()
        counts[key] = counts.get(key, 0) + 1

    total = sum(counts.values())
    supported_ratio = counts["supported"] / total if total > 0 else 0.0

    return {
        "supported": counts["supported"],
        "partial": counts["partial"],
        "unsupported": counts["unsupported"],
        "unknown": counts["unknown"],
        "total": total,
        "supported_ratio": round(supported_ratio, 4),
    }


def get_level_patterns(classifications: dict[str, str], level: str) -> list[str]:
    """Return sorted list of pattern IDs at the given level."""
    return sorted(pid for pid, lv in classifications.items() if lv == level.upper())


# ---------------------------------------------------------------------------
# Pre/post delta
# ---------------------------------------------------------------------------


def compute_delta(
    sa_pre: dict[str, str],
    sa_post: dict[str, str],
) -> dict:
    """Compare pre and post SA classifications to measure optimizer effect.

    Tracks three types of improvement:
    - improved: PARTIAL/UNSUPPORTED in pre → SUPPORTED in post (same pattern ID, level changed)
    - fused_away: PARTIAL/UNSUPPORTED in pre → absent in post (optimizer fused/replaced the op)
    - regressed: SUPPORTED in pre → PARTIAL/UNSUPPORTED in post (should never happen)

    Returns delta dict with all categories and ratios.
    """
    improved: list[str] = []
    fused_away: list[str] = []  # PARTIAL/UNSUPPORTED removed by fusion — implicit improvement
    regressed: list[str] = []
    unchanged_supported: list[str] = []
    unchanged_partial_unsupported: list[str] = []

    for pattern_id, pre_level in sa_pre.items():
        post_level = sa_post.get(pattern_id)
        if pre_level in ("PARTIAL", "UNSUPPORTED"):
            if post_level is None:
                fused_away.append(pattern_id)  # op fused away — an improvement
            elif post_level == "SUPPORTED":
                improved.append(pattern_id)
            else:
                unchanged_partial_unsupported.append(pattern_id)
        elif pre_level == "SUPPORTED":
            if post_level is None:
                pass  # SUPPORTED op also fused — neutral
            elif post_level in ("PARTIAL", "UNSUPPORTED"):
                regressed.append(pattern_id)
            else:
                unchanged_supported.append(pattern_id)

    pre_summary = get_sa_summary(sa_pre)
    post_summary = get_sa_summary(sa_post)
    supported_ratio_delta = round(
        post_summary["supported_ratio"] - pre_summary["supported_ratio"], 4
    )

    return {
        "improved": sorted(improved),
        "fused_away": sorted(fused_away),
        "regressed": sorted(regressed),
        "unchanged_supported": sorted(unchanged_supported),
        "unchanged_partial_unsupported": sorted(unchanged_partial_unsupported),
        "pre_supported_ratio": pre_summary["supported_ratio"],
        "post_supported_ratio": post_summary["supported_ratio"],
        "supported_ratio_delta": supported_ratio_delta,
    }


# ---------------------------------------------------------------------------
# EPContext diff (ground truth, optional)
# ---------------------------------------------------------------------------


def get_epcontext_diff(compiled_onnx: Path) -> dict:
    """Extract fallback op info from compiled (EPContext) ONNX.

    After QNN compilation, the graph contains:
    - EPContext nodes → on NPU
    - Regular ONNX nodes → CPU fallback

    Returns {"epcontext_nodes", "fallback_nodes", "fallback_op_types",
             "fallback_pattern_ids"}
    """
    model = onnx.load(str(compiled_onnx))
    epcontext_count = 0
    fallback: list[dict] = []

    for node in model.graph.node:
        if node.op_type == "EPContext":
            epcontext_count += 1
        else:
            domain = node.domain or "ai.onnx"
            fallback.append(
                {
                    "op_type": node.op_type,
                    "pattern_id": f"OP/{domain}/{node.op_type}",
                }
            )

    fallback_counts = Counter(n["op_type"] for n in fallback)
    fallback_pattern_ids = list({n["pattern_id"] for n in fallback})

    return {
        "epcontext_nodes": epcontext_count,
        "fallback_nodes": len(fallback),
        "fallback_op_types": dict(fallback_counts.most_common()),
        "fallback_pattern_ids": sorted(fallback_pattern_ids),
    }


def compare_sa_vs_epcontext(
    sa_predictions: dict[str, str],
    compiled_onnx: Path,
) -> dict:
    """Compare SA predictions against EPContext diff ground truth.

    Uses the same TP/FP/TN/FN classification from poc_b_epcontext_diff.

    Returns comparison list and summary metrics.
    """
    diff = get_epcontext_diff(compiled_onnx)
    fallback_pids = set(diff["fallback_pattern_ids"])

    comparison = []
    for pid, sa_level in sorted(sa_predictions.items()):
        actual = "CPU" if pid in fallback_pids else "NPU"
        if sa_level == "SUPPORTED" and actual == "NPU":
            verdict = "TP"
        elif sa_level == "SUPPORTED" and actual == "CPU":
            verdict = "FP"
        elif sa_level in ("PARTIAL", "UNSUPPORTED") and actual == "CPU":
            verdict = "TN"
        elif sa_level in ("PARTIAL", "UNSUPPORTED") and actual == "NPU":
            verdict = "FN"
        else:
            verdict = "UNKNOWN"
        comparison.append(
            {
                "pattern_id": pid,
                "sa_prediction": sa_level,
                "actual_ep": actual,
                "verdict": verdict,
            }
        )

    verdicts = Counter(c["verdict"] for c in comparison)
    tp = verdicts.get("TP", 0)
    tn = verdicts.get("TN", 0)
    fp = verdicts.get("FP", 0)
    fn = verdicts.get("FN", 0)
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # False alarms: SA predicted non-SUPPORTED but EP actually handled the op
    unsupported_false_alarms = sorted(
        c["pattern_id"]
        for c in comparison
        if c["sa_prediction"] == "UNSUPPORTED" and c["actual_ep"] == "NPU"
    )
    partial_false_alarms = sorted(
        c["pattern_id"]
        for c in comparison
        if c["sa_prediction"] == "PARTIAL" and c["actual_ep"] == "NPU"
    )

    return {
        "compiled_onnx": compiled_onnx.name,
        "epcontext_nodes": diff["epcontext_nodes"],
        "fallback_nodes": diff["fallback_nodes"],
        "fallback_op_types": diff["fallback_op_types"],
        "comparison": comparison,
        "summary": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": round(accuracy, 4),
            "unsupported_false_alarms": unsupported_false_alarms,
            "partial_false_alarms": partial_false_alarms,
        },
    }
