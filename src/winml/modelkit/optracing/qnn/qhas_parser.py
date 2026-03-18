# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Parse QNN Hardware Acceleration Summary (QHAS) JSON artifacts.

QHAS files are produced by QNN EP profiling and contain per-operator
hardware metrics including cycles, DRAM/VTCM traffic, and dominant-path
information.  This module transforms the raw JSON into a normalised dict
suitable for the detail-mode op-tracing report.
"""
from __future__ import annotations


def parse_qhas(qhas_data: dict) -> dict:
    """Parse a QHAS JSON structure into normalised summary + operator list.

    Parameters
    ----------
    qhas_data:
        Deserialised QHAS JSON (must contain ``data.htp_overall_summary``
        and ``data.qnn_op_instances_nodes``).

    Returns:
    -------
    dict
        ``{"summary": {...}, "operators": [...]}``.
    """
    data = qhas_data["data"]
    summary = _extract_summary(data)

    # Derive a cycle-to-microsecond factor from the summary.
    timeline_cycles = summary["timeline_cycles"]
    cycle_to_us = summary["time_us"] / timeline_cycles if timeline_cycles else 0.0

    raw_ops = data.get("qnn_op_instances_nodes", {}).get("data", [])
    operators = [_transform_op(op, cycle_to_us) for op in raw_ops]

    return {"summary": summary, "operators": operators}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_summary(data: dict) -> dict:
    """Extract the HTP overall summary into a flat dict."""
    rows = data.get("htp_overall_summary", {}).get("data", [])
    if not rows:
        return {}
    raw = rows[0]
    return {
        "time_us": raw["time_us"],
        "graph_execute_us": raw["graph_execute_us"],
        "inf_per_s": raw["inf_per_s"],
        "timeline_cycles": raw["timeline_cycles"],
        "utilization_pct": raw["percent_utilization"],
        "total_dram_read": raw["total_dram_read"],
        "total_dram_write": raw["total_dram_write"],
        "total_vtcm_read": raw["total_vtcm_read"],
        "total_vtcm_write": raw["total_vtcm_write"],
        "peak_vtcm_alloc": raw["peak_vtcm_alloc"],
        "qnn_nodes": raw["qnn_nodes"],
        "htp_nodes": raw["htp_nodes"],
        "unique_qnn_ops": raw["unique_qnn_ops"],
        "unique_htp_ops": raw["unique_htp_ops"],
    }


def _transform_op(op: dict, cycle_to_us: float) -> dict:
    """Transform a single ``qnn_op_instances_nodes`` entry.

    Converts raw cycle counts to microseconds and computes derived
    metrics such as VTCM hit ratio and dominant-path duration.
    """
    cycles = op["cycles"]
    duration_us = cycles * cycle_to_us

    dp_cycles = op.get("num_dominant_path_cycles_htp_0")
    dominant_path_us = dp_cycles * cycle_to_us if dp_cycles else None

    vtcm_read = op.get("vtcm_read", 0)
    dram_read = op.get("dram_read", 0)

    return {
        "name": op["qnn_op"],
        "op_path": op["qnn_op"],
        "op_type": op["qnn_op_type"],
        "cycles": cycles,
        "duration_us": duration_us,
        "percent_of_total": op["percent_active_cycles"],
        "dominant_path_us": dominant_path_us,
        "num_htp_ops": op.get("num_htp_ops", 0),
        "dram_read_bytes": dram_read,
        "dram_write_bytes": op.get("dram_write", 0),
        "vtcm_read_bytes": vtcm_read,
        "vtcm_write_bytes": op.get("vtcm_write", 0),
        "vtcm_hit_ratio": _vtcm_ratio(op),
    }


def _vtcm_ratio(op: dict) -> float | None:
    """Compute VTCM hit ratio: vtcm_read / (vtcm_read + dram_read).

    Returns ``None`` when both values are zero (no read traffic).
    """
    vtcm_read = op.get("vtcm_read", 0)
    dram_read = op.get("dram_read", 0)
    total = vtcm_read + dram_read
    if total == 0:
        return None
    return vtcm_read / total
