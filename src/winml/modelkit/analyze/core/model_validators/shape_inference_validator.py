# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Validator for detecting missing shape information in models.

Detects intermediate tensors with unknown shapes (dynamic or unresolvable)
after shape inference has been performed.
"""

from __future__ import annotations

import json
import logging
from collections import Counter

from ...models.information import Action, ActionLevel, Information
from ...models.runtime_checks import NodeTag
from .base import ModelValidator


logger = logging.getLogger(__name__)


class ShapeInferenceValidator(ModelValidator):
    """Validates shape inference status of model tensors.

    Detects intermediate tensors that lack shape information and provides
    recommendations to add shape inference for better optimization and
    performance analysis.
    """

    @property
    def validator_name(self) -> str:
        """Return validator name."""
        return "ShapeInferenceValidator"

    @property
    def pattern_id(self) -> str:
        """Return pattern ID for Information objects."""
        return "MODEL/ShapeInference"

    def validate(self) -> Information | None:
        """Detect operators affected by unknown shape dimensions.

        This validator identifies operators with nodes marked as having
        missing shape inference from runtime analysis.

        Returns:
            Information object if operators with unknown shapes detected, None otherwise
        """
        logger.debug(f"{self.validator_name}: Starting validation")

        affected_ops = self._collect_missing_shape_nodes_from_runtime_results()

        if not affected_ops:
            logger.debug(f"{self.validator_name}: No operators with missing shapes found")
            return None

        logger.warning(
            f"{self.validator_name}: Found {len(affected_ops)}"
            f" operator(s) with missing shape information"
        )

        # Create Information with recommendations
        return self._create_information(affected_ops)

    def _collect_missing_shape_nodes_from_runtime_results(self) -> list[dict]:
        """Collect nodes with MISSING_SHAPE_INFERENCE tag from runtime results.

        Returns:
            List of dicts with node info: {op_name, op_type}
        """
        affected_ops = []

        for runtime_result in self.op_runtime_results:
            # Check if this result has the MISSING_SHAPE_INFERENCE tag
            if NodeTag.MISSING_SHAPE_INFERENCE not in runtime_result.result.node_tags:
                continue

            # Extract node information from pattern_match
            if runtime_result.pattern_match is None:
                continue

            for matched_node in runtime_result.pattern_match.matched_node_names:
                op_info = {
                    "op_name": matched_node.node_name,
                    "op_type": matched_node.op_type,
                }
                affected_ops.append(op_info)

        # Log node names for debugging
        if affected_ops:
            node_names = [op["op_name"] for op in affected_ops]
            logger.debug(
                f"{self.validator_name}: Collected "
                f"{len(affected_ops)} operator(s) with missing "
                f"shape inference: {node_names}"
            )

        return affected_ops

    def _create_information(self, affected_ops: list[dict]) -> Information:
        """Create Information object with operator-level recommendations.

        Args:
            affected_ops: List of operators with unknown shape inputs/outputs

        Returns:
            Information object with recommendations
        """
        # Create recommendation action
        action = Action(
            pattern_from_id="",
            pattern_to_id="",
            level=ActionLevel.WARNING,
            details=json.dumps(
                [
                    {
                        "title": "Normalize model",
                        "command": "wmk optimize --model model.onnx",
                    }
                ],
                indent=2,
            ),
        )

        # Create detailed explanation
        op_types = Counter(op["op_type"] for op in affected_ops)

        op_type_summary = ", ".join(
            f"{count}x {op_type}" for op_type, count in op_types.most_common(5)
        )

        explanation = (
            f"Found {len(affected_ops)} operator(s) with unknown shape dimensions:\n"
            f"{op_type_summary}.\n"
            "These operators have inputs or outputs with:\n"
            "- Dynamic dimensions that cannot be statically determined\n"
            "- Shapes dependent on model inputs\n"
            "- Unresolvable shape information\n\n"
            "This impacts:\n"
            "- Memory pre-allocation and optimization\n"
            "- Performance prediction\n"
            "- Static analysis and validation"
        )

        # Create Information object
        return Information(
            explanation=explanation,
            actions=[action],
            pattern_id=self.pattern_id,
        )
