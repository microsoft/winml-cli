# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Doc Constraint Checker - Query documentation-based rules for operator constraint checking.

This module provides functionality to load and query operator constraints
extracted from documentation for any execution provider, supporting constraint
checking based on documentation specifications.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import onnx
import pandas as pd

from ...onnx import infer_onnx_shapes
from ..doc_checker.mapping_checkers import get_qnn_op_for_onnx_node
from ..doc_checker.shape_checker import ShapeConstraintChecker
from ..models.runtime_checks import PatternAlternative, PatternRuntime, RuntimeTestResult
from ..utils.model_utils import (
    collect_valueinfo_dict,
    node_to_pattern_match,
    shape_and_dtype_from_valueinfo,
)
from ..utils.node_key_utils import build_node_key_by_node_id, resolve_stable_node_key


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    import onnx


class DocConstraintChecker:
    """Query documentation-based rules for operator constraint checking.

    This class loads operator constraints from documentation JSON files
    and provides methods to check if ONNX nodes satisfy those constraints.
    Works with any execution provider that has documentation-based constraints.

    Example:
        >>> model = onnx.load("model.onnx")
        >>> checker = DocConstraintChecker(model, "QNNExecutionProvider", "NPU")
        >>> for node in model.graph.node:
        ...     result = checker.run_for_node(node)
        ...     if not result.result.compile:
        ...         print(f"Node {node.name} failed: {result.result.reason}")
    """

    # Mapping of checker function names to actual callable functions
    CHECKER_FUNCTIONS: ClassVar[dict[str, Callable]] = {
        "check_max_rank": ShapeConstraintChecker.check_max_rank,
        "check_min_rank": ShapeConstraintChecker.check_min_rank,
        "check_exact_rank": ShapeConstraintChecker.check_exact_rank,
        "check_rank_range": ShapeConstraintChecker.check_rank_range,
        "check_dimension_value": ShapeConstraintChecker.check_dimension_value,
        "check_dimension_range": ShapeConstraintChecker.check_dimension_range,
        "check_dimension_divisible": ShapeConstraintChecker.check_dimension_divisible,
        "check_dimension_multiple": ShapeConstraintChecker.check_dimension_multiple,
        # Add more checker functions here as they are implemented
    }

    def __init__(
        self,
        model_proto: onnx.ModelProto,
        ep_name: str,
        device_type: str,
        skip_shape_inference: bool = False,
        node_key_by_node_id: dict[int, str] | None = None,
    ) -> None:
        """Initialize doc constraint checker.

        Args:
            model_proto: ONNX model proto
            ep_name: Execution provider name (e.g., "QNNExecutionProvider")
            device_type: Device type (e.g., "NPU")
            skip_shape_inference: If True, assume model_proto already has shape
                inference applied (avoids expensive redundant inference).
            node_key_by_node_id: Optional sidecar map from id(node) to stable node key.
        """
        if skip_shape_inference:
            self.model_proto = model_proto
        else:
            self.model_proto = infer_onnx_shapes(model_proto)
        # Keep stable Python wrapper references for graph nodes so id(node)
        # mappings do not accidentally collide with new transient wrappers.
        self._graph_nodes: list[onnx.NodeProto] = list(self.model_proto.graph.node)
        self.ep_name = ep_name
        self.device_type = device_type
        self.valueinfo = collect_valueinfo_dict(self.model_proto)
        if node_key_by_node_id is not None:
            self._node_key_by_node_id = dict(node_key_by_node_id)
        else:
            self._node_key_by_node_id = build_node_key_by_node_id(self._graph_nodes)

        # Load ONNX to target EP operator mapping
        self.mapping_config = self._load_mapping_config()

        # Load operator constraints from JSON file
        self.op_constraints = self._load_constraints()

        # Alternatives support not yet implemented
        self.alternatives: list[PatternAlternative] = []

    def _get_stable_node_key(self, node: onnx.NodeProto) -> str:
        """Resolve stable analyzer key for a node."""
        return resolve_stable_node_key(
            node,
            node_key_by_node_id=self._node_key_by_node_id,
            graph_nodes=self._graph_nodes,
            unknown_unnamed_error=(
                "Cannot resolve stable key for unnamed node outside "
                "DocConstraintChecker model graph."
            ),
        )

    def _load_constraints(self) -> dict[str, pd.DataFrame]:
        """Load operator constraints from JSON file.

        Returns:
            Dict mapping operator name to DataFrame with constraints
        """
        rule_path = Path(__file__).parent.joinpath(
            f"../rules/information_rules/{self.ep_name}_{self.device_type}_from_doc.json"
        )

        if not rule_path.exists():
            logger.debug(f"Rule file not found: {rule_path}")
            return {}

        with rule_path.open(encoding="utf-8") as f:
            data = json.load(f)

        # Convert to DataFrames for easier querying
        op_dfs = {}
        for op_type, op_data in data.items():
            try:
                df = pd.DataFrame(op_data)
                op_dfs[op_type] = df
                logger.debug(f"Loaded constraints for operator: {op_type} ({len(df)} records)")
            except Exception as e:  # noqa: PERF203
                logger.error(f"Failed to load constraints for {op_type}: {e}")

        return op_dfs

    def _load_mapping_config(self) -> dict:
        """Load ONNX to target EP operator mapping configuration.

        Returns:
            Dict with operator mapping configuration
        """
        # For QNN, load onnx_to_qnn_mapping.json
        if "QNN" in self.ep_name:
            mapping_path = Path(__file__).parent.joinpath(
                "../rules/information_rules/onnx_to_qnn_mapping.json"
            )
        else:
            # For other EPs, could load different mapping files in the future
            logger.debug(f"No operator mapping defined for {self.ep_name}")
            return {}

        if not mapping_path.exists():
            logger.debug(f"Mapping file not found: {mapping_path}")
            return {}

        with mapping_path.open(encoding="utf-8") as f:
            mapping_config = json.load(f)

        logger.debug(f"Loaded operator mapping config for {self.ep_name}")
        return mapping_config

    def _map_onnx_to_target_op(self, node: onnx.NodeProto) -> tuple[str | None, str | None]:
        """Map ONNX operator to target EP operator using mapping config.

        Args:
            node: ONNX node

        Returns:
            Tuple of (target_op_name, error_description)
            - If mapping succeeds: (target_op, None)
            - If mapping fails: (None, error_description)
        """
        if not self.mapping_config:
            # No mapping config, use ONNX op type directly
            return node.op_type, None

        try:
            target_op, description = get_qnn_op_for_onnx_node(
                node, self.mapping_config, self.valueinfo
            )
            if target_op:
                logger.debug(f"Mapped {node.op_type} -> {target_op}: {description}")
                return target_op, None
            # Mapping failed - return error description
            logger.debug(f"Failed to map {node.op_type}: {description}")
            return None, description
        except Exception as e:
            logger.debug(f"Error mapping {node.op_type}: {e}")
            return None, str(e)

    def get_op_constraints(self, op_type: str) -> pd.DataFrame | None:
        """Get constraints DataFrame for an operator.

        Args:
            op_type: ONNX operator type

        Returns:
            DataFrame with constraints, or None if not found
        """
        return self.op_constraints.get(op_type)

    def _get_node_dtype_category(self, node: onnx.NodeProto) -> str:
        """Determine data type category for a node.

        Args:
            node: ONNX node

        Returns:
            Data type category string (e.g., "FLOAT16", "INT8", "OTHERS")
        """
        # Try to get dtype from first input
        if node.input:
            inp_name = node.input[0]
            vi = self.valueinfo.get(inp_name)
            if vi is not None:
                _, dtype = shape_and_dtype_from_valueinfo(vi)
                if dtype is not None:
                    # Map ONNX dtype to category
                    dtype_map = {
                        "float16": "FLOAT16",
                        "bfloat16": "BFLOAT16",
                        "int8": "INT8",
                        "uint8": "INT8",  # Treat uint8 as INT8 category
                        "int16": "INT16",
                        "uint16": "INT16",
                    }
                    return dtype_map.get(dtype.lower(), "OTHERS")

        return "OTHERS"

    def _get_node_actual_dtype(self, node: onnx.NodeProto) -> str | None:
        """Get the actual ONNX dtype of a node.

        Args:
            node: ONNX node

        Returns:
            ONNX dtype string (e.g., "FLOAT16", "INT32") or None if not found
        """
        if node.input:
            inp_name = node.input[0]
            vi = self.valueinfo.get(inp_name)
            if vi is not None:
                _, dtype = shape_and_dtype_from_valueinfo(vi)
                if dtype is not None:
                    return dtype.upper()
        return None

    def _get_node_shape(self, tensor_name: str) -> list[int] | None:
        """Get shape of a tensor.

        Args:
            tensor_name: Name of the tensor

        Returns:
            Shape as list of integers, or None if not found
        """
        vi = self.valueinfo.get(tensor_name)
        if vi is not None:
            shape, _ = shape_and_dtype_from_valueinfo(vi)
            return shape
        return None

    def _append_source_to_reason(self, reason: str, source_url: str | None) -> str:
        """Append source URL to the end of reason string.

        Args:
            reason: The reason string
            source_url: Optional source URL to append

        Returns:
            Reason string with source URL appended at the end
        """
        if not source_url or not reason:
            return reason
        return f"{reason} (from {source_url})"

    def _execute_checker(
        self, checker_info: dict[str, Any], node: onnx.NodeProto
    ) -> tuple[bool, str]:
        """Execute a single checker function.

        Args:
            checker_info: Dictionary with checker name, params, applies_to, and index
            node: ONNX node to check

        Returns:
            Tuple of (success, error_message)
        """
        checker_name = checker_info.get("checker")
        params = checker_info.get("params", {})
        applies_to = checker_info.get("applies_to")  # "input" or "output"
        index = checker_info.get("index", 0)

        # Get the checker function
        checker_func = self.CHECKER_FUNCTIONS.get(checker_name)
        if checker_func is None:
            logger.debug(f"Unknown checker function: {checker_name}")
            return False, f"Unknown checker function: {checker_name}"

        # Get the tensor to check
        if applies_to == "input":
            if index >= len(node.input):
                return False, f"Input index {index} out of range"
            tensor_name = node.input[index]
        elif applies_to == "output":
            if index >= len(node.output):
                return False, f"Output index {index} out of range"
            tensor_name = node.output[index]
        else:
            return False, f"Invalid applies_to value: {applies_to}"

        # Get tensor shape
        shape = self._get_node_shape(tensor_name)
        if shape is None:
            return False, f"Cannot infer shape for {applies_to} {index}"

        # Execute checker
        try:
            success, message = checker_func(shape, **params)
            if not success:
                # Add information about which input/output failed
                tensor_desc = f"{applies_to}_{index}"
                message = f"{tensor_desc}: {message}"
            return success, message
        except Exception as e:
            logger.error(f"Error executing checker {checker_name}: {e}")
            return False, f"Checker execution error: {e}"

    def run_for_node(self, node: onnx.NodeProto) -> PatternRuntime:
        """Run constraint check for a single node.

        Args:
            node: ONNX node to check

        Returns:
            PatternRuntime with check results
        """
        node_key = self._get_stable_node_key(node)
        pattern_match = node_to_pattern_match(node, node_key)
        op_type = node.op_type

        # Skip certain operators
        ignored_ops = {
            "Constant",
            "QuantizeLinear",
            "DequantizeLinear",
        }
        if op_type in ignored_ops:
            return PatternRuntime(
                pattern_id=pattern_match.pattern.pattern_id,
                result=RuntimeTestResult(run=True, compile=True, no_data=False),
                alternatives=self.alternatives,
                pattern_match=pattern_match,
            )

        # Map ONNX operator to target EP operator
        target_op, error_desc = self._map_onnx_to_target_op(node)
        if target_op is None:
            # Check if this is a condition check failure (shape incomplete) or truly unsupported
            # Condition check failures have messages like "No matching rule found for..."
            if error_desc and "No matching rule found" in error_desc:
                # Condition check failed (usually due to incomplete shape)
                # Only log, don't create doc checker result
                logger.debug(f"Condition check failed for {op_type}: {error_desc}")
                return PatternRuntime(
                    pattern_id=pattern_match.pattern.pattern_id,
                    result=RuntimeTestResult(run=True, compile=True, no_data=False),
                    alternatives=self.alternatives,
                    pattern_match=pattern_match,
                )
            # Operator truly not in DB or unsupported
            logger.debug(f"Mapping failed for {op_type}: {error_desc}")
            return PatternRuntime(
                pattern_id=pattern_match.pattern.pattern_id,
                result=RuntimeTestResult(
                    run=False,
                    compile=False,
                    no_data=True,
                    reason="No data available, SDK may not support this operator",
                ),
                alternatives=self.alternatives,
                pattern_match=pattern_match,
            )

        # Get constraints for the mapped operator
        df = self.get_op_constraints(target_op)
        if df is None or df.empty:
            return PatternRuntime(
                pattern_id=pattern_match.pattern.pattern_id,
                result=RuntimeTestResult(
                    run=True, compile=True, no_data=True, reason="constraints_not_found"
                ),
                alternatives=self.alternatives,
                pattern_match=pattern_match,
            )

        # Determine dtype category for this node
        dtype_category = self._get_node_dtype_category(node)

        # Filter constraints by dtype
        matching_rows = df[df["dtype"] == dtype_category]

        if matching_rows.empty:
            # Try OTHERS category as fallback
            matching_rows = df[df["dtype"] == "OTHERS"]

        if matching_rows.empty:
            actual_dtype = self._get_node_actual_dtype(node)
            logger.debug(
                f"No constraints found for operator "
                f"'{target_op}' with dtype category "
                f"'{dtype_category}' (actual dtype: "
                f"{actual_dtype})"
            )
            return PatternRuntime(
                pattern_id=pattern_match.pattern.pattern_id,
                result=RuntimeTestResult(
                    run=True,
                    compile=True,
                    no_data=True,
                    reason=f"no_constraints_for_dtype_{dtype_category}",
                ),
                alternatives=self.alternatives,
                pattern_match=pattern_match,
            )

        # Get the first matching row
        row = matching_rows.iloc[0]

        # Check compile_run_success
        compile_success, run_success = row["compile_run_success"]
        reason = row.get("reason", "")

        # If dtype is not supported, return immediately
        if not compile_success and not run_success and reason == "dtype_not_supported":
            # Get actual dtype and supported dtypes
            actual_dtype = self._get_node_actual_dtype(node)

            # TODO: FLOAT dtype check temporarily skipped - will follow up
            if actual_dtype == "FLOAT":
                # Skip other constraint checks for FLOAT
                return PatternRuntime(
                    pattern_id=pattern_match.pattern.pattern_id,
                    result=RuntimeTestResult(
                        run=True,
                        compile=True,
                        no_data=False,
                        reason="OK",
                    ),
                    alternatives=self.alternatives,
                    pattern_match=pattern_match,
                )

            # If we didn't skip the error above, prepare error message
            if not (compile_success and run_success):
                supported_dtypes_info = ""

                # Try to get supported dtypes from all matching rows
                all_supported_dtypes = set()
                for _, dtype_row in df.iterrows():
                    dtype_val = dtype_row.get("dtype")
                    if dtype_val and dtype_val != "OTHERS":
                        compile, run = dtype_row["compile_run_success"]
                        if compile and run:
                            all_supported_dtypes.add(dtype_val)

                if all_supported_dtypes:
                    supported_list = ", ".join(sorted(all_supported_dtypes))
                    supported_dtypes_info = f". Supported dtypes: {supported_list}"

                dtype_val = actual_dtype or dtype_category
                dtype_msg = f"Current dtype '{dtype_val}' is not supported{supported_dtypes_info}"

                return PatternRuntime(
                    pattern_id=pattern_match.pattern.pattern_id,
                    result=RuntimeTestResult(
                        run=False,
                        compile=False,
                        no_data=False,
                        reason=dtype_msg,
                    ),
                    alternatives=self.alternatives,
                    pattern_match=pattern_match,
                )

        # Execute constraint checkers
        condition_checker = row.get("condition_checker", {})
        if condition_checker and isinstance(condition_checker, list):
            # Get source_url from the row if available
            source_url = row.get("source_url")
            errors = []
            for checker_info in condition_checker:
                success, message = self._execute_checker(checker_info, node)
                if not success:
                    errors.append(message)

            if errors:
                compile_success = False
                run_success = False
                # Deduplicate errors to avoid repetitive messages
                unique_errors = list(dict.fromkeys(errors))
                reason = "; ".join(unique_errors)
                # Append source URL once at the end
                reason = self._append_source_to_reason(reason, source_url)

        return PatternRuntime(
            pattern_id=pattern_match.pattern.pattern_id,
            result=RuntimeTestResult(
                run=run_success,
                compile=compile_success,
                no_data=False,
                reason=reason if reason else "OK",
            ),
            alternatives=self.alternatives,
            pattern_match=pattern_match,
        )

    def get_operators_with_constraints(self) -> list[str]:
        """Get list of operators that have constraints defined.

        Returns:
            List of operator names
        """
        return list(self.op_constraints.keys())
