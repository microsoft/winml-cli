# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Rules-based prefilter utilities for runtime-check skip decisions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .runtime_checker_query import RuntimeCheckerQuery


if TYPE_CHECKING:
    import onnx

    from ...utils.constants import EPName


logger = logging.getLogger(__name__)


class RuntimeCheckerRulesPrefilter:
    """Fixed rules-prefilter service bound to one EP+device pair."""

    def __init__(self, ep_name: EPName, device_type: str) -> None:
        self._ep_name = ep_name
        self._device_type = device_type

    def build_skip_check_result_for_rules_all_nodes_compile_run_pass(
        self,
        onnx_model: onnx.ModelProto,
    ) -> dict[str, Any] | None:
        """Build synthetic check_result when rules indicate all nodes pass."""
        return build_skip_check_result_for_rules_all_nodes_compile_run_pass(
            onnx_model,
            self._ep_name,
            self._device_type,
        )


def build_skip_check_result_for_rules_all_nodes_compile_run_pass(
    onnx_model: onnx.ModelProto,
    ep_name: EPName,
    device_type: str,
) -> dict[str, Any] | None:
    """Return synthetic check_result when all nodes have rules compile/run pass.

    Returns None when rules data is missing, any node is unsupported,
    or prefilter evaluation fails. Caller should then run real compile/run.
    """
    try:
        query = RuntimeCheckerQuery(
            model_proto=onnx_model,
            ep_name=ep_name,
            device_type=device_type,
        )
        node_results = [
            query.run_for_node(node, run_unknown_op=False) for node in query.model_proto.graph.node
        ]

        if not node_results:
            return None

        all_supported = all(
            result.result.compile and result.result.run and not result.result.no_data
            for result in node_results
        )
        if not all_supported:
            return None

        reason = f"rules_all_nodes_compile_run_pass_{len(node_results)}_nodes"
        return {
            "compile": {
                "result": {"success": True, "reason": reason},
                "stdout": "skipped_by_rules_all_nodes_compile_run_pass",
                "stderr": "",
            },
            "run": {
                "result": {"success": True, "reason": reason},
                "stdout": "skipped_by_rules_all_nodes_compile_run_pass",
                "stderr": "",
            },
        }
    except Exception as exc:
        logger.warning("Rules prefilter failed for one case (%s); fallback to ep_checker.", exc)
        return None


__all__ = [
    "RuntimeCheckerRulesPrefilter",
    "build_skip_check_result_for_rules_all_nodes_compile_run_pass",
]
