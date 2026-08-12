# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Deprecated compatibility shim for the legacy QHAS parser."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...session.monitor.qnn import parse_qhas as _parse_current_qhas
from .._compat import warn_deprecated


def _parse_qhas(qhas_data: dict[str, Any]) -> dict[str, Any]:
    parsed = _parse_current_qhas(qhas_data)
    summary = parsed["summary"]
    raw_operators = qhas_data["data"].get("qnn_op_instances_nodes", {}).get("data", [])
    return {
        "summary": {
            "time_us": summary["inference_us"],
            "graph_execute_us": summary["execute_us"],
            "inf_per_s": summary["inf_per_s"],
            "timeline_cycles": summary["timeline_cycles"],
            "utilization_pct": summary["utilization_pct"],
            "total_dram_read": summary["dram_read_bytes"],
            "total_dram_write": summary["dram_write_bytes"],
            "total_vtcm_read": summary["vtcm_read_bytes"],
            "total_vtcm_write": summary["vtcm_write_bytes"],
            "peak_vtcm_alloc": summary["vtcm_peak_bytes"],
            "qnn_nodes": summary["qnn_nodes"],
            "htp_nodes": summary["htp_nodes"],
            "unique_qnn_ops": summary["unique_qnn_ops"],
            "unique_htp_ops": summary["unique_htp_ops"],
        },
        "operators": [
            {
                **operator,
                "name": raw_operator["qnn_op"],
                "op_path": raw_operator["qnn_op"],
                "op_type": raw_operator["qnn_op_type"],
            }
            for operator, raw_operator in zip(parsed["operators"], raw_operators, strict=True)
        ],
    }


def __getattr__(name: str) -> Any:
    if name != "parse_qhas":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    warn_deprecated(
        "qnn.qhas_parser.parse_qhas",
        "winml.modelkit.session.monitor.qnn.parse_qhas",
        stacklevel=2,
    )
    return _parse_qhas


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    parse_qhas = _parse_qhas


__all__ = ["parse_qhas"]
