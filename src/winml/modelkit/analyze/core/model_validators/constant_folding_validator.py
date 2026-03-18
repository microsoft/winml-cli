"""Validator for constant folding optimization opportunities.

Detects nodes where all inputs are constants (initializers or Constant ops)
and recommends applying constant folding optimization.
"""

from __future__ import annotations

import json
import logging

from ...models.information import Action, ActionLevel, Information
from ...models.runtime_checks import NodeTag
from .base import ModelValidator


logger = logging.getLogger(__name__)


class ConstantFoldingValidator(ModelValidator):
    """Detect nodes with all-constant inputs for folding recommendations.

    This validator identifies nodes where all inputs come from initializers
    or Constant nodes, indicating they can be pre-computed at optimization
    time rather than at runtime.

    Attributes:
        model_proto: ONNX ModelProto to analyze
    """

    @property
    def validator_name(self) -> str:
        """Name of this validator."""
        return "ConstantFoldingValidator"

    @property
    def pattern_id(self) -> str:
        """Pattern ID for Information objects."""
        return "MODEL/ConstantFolding"

    def validate(self) -> Information | None:
        """Detect constant-only nodes and generate recommendation.

        Returns:
            Information object if constant-only nodes found, None otherwise
        """
        logger.debug(f"{self.validator_name}: Starting validation")

        constant_nodes = self._collect_constant_nodes_from_runtime_results()

        if not constant_nodes:
            logger.debug(f"{self.validator_name}: No constant-only nodes found")
            return None

        logger.info(f"{self.validator_name}: Found {len(constant_nodes)} constant-only node(s)")

        return self._create_information(constant_nodes)

    def _collect_constant_nodes_from_runtime_results(self) -> list[dict]:
        """Collect nodes with ALL_INPUTS_CONSTANT tag from runtime results.

        Returns:
            List of dicts with node info: {name, op_type}
        """
        constant_nodes = []

        for runtime_result in self.op_runtime_results:
            # Check if this result has the ALL_INPUTS_CONSTANT tag
            if NodeTag.ALL_INPUTS_CONSTANT not in runtime_result.result.node_tags:
                continue

            # Extract node information from pattern_match
            if runtime_result.pattern_match is None:
                continue

            for matched_node in runtime_result.pattern_match.matched_node_names:
                constant_nodes.append(
                    {
                        "name": matched_node.node_name,
                        "op_type": matched_node.op_type,
                    }
                )

        # Log node names for debugging
        if constant_nodes:
            node_names = [node["name"] for node in constant_nodes]
            logger.debug(
                f"{self.validator_name}: Collected {len(constant_nodes)} constant-only node(s) from runtime results: {node_names}"
            )

        return constant_nodes

    def _create_information(self, constant_nodes: list[dict]) -> Information:
        """Create Information recommendation for constant folding.

        Args:
            constant_nodes: List of node info dicts

        Returns:
            Information object with action recommendation
        """
        # Extract first 3 node names as examples
        example_nodes = ", ".join(node["name"] for node in constant_nodes[:3])
        examples_text = f"Examples: {example_nodes}. " if example_nodes else ""

        # Estimate size reduction
        size_reduction = self._estimate_size_reduction(constant_nodes)

        # Create action with detailed guidance
        action = Action(
            pattern_from_id="",
            pattern_to_id="",
            level=ActionLevel.OPTIONAL,
            status=None,  # Not a support status issue
            details=json.dumps(
                [
                    {
                        "title": "onnx-optimizer",
                        "command": "python -m onnxoptimizer model.onnx optimized.onnx --constant-folding",
                    },
                    {
                        "title": "onnx-simplifier",
                        "command": "python -m onnxsim model.onnx simplified.onnx",
                    },
                    {
                        "title": "ort-optimization",
                        "code_example": "session_options = onnxruntime.SessionOptions()\nsession_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL",
                    },
                ],
                indent=2,
            ),
        )

        explanation = (
            f"Model contains {len(constant_nodes)} node(s) with all-constant inputs. "
            f"{examples_text}"
            f"Models without constant folding may result to false alarms or in-accurate results in Static Analyzer. "
            f"Applying constant folding can pre-compute these operations at optimization "
            f"time rather than runtime, potentially reducing model size by ~{size_reduction}% "
            f"and improving inference speed."
        )

        return Information(
            explanation=explanation,
            actions=[action],
            pattern_id=self.pattern_id,
            status=None,
        )

    def _estimate_size_reduction(self, constant_nodes: list[dict]) -> int:
        """Rough estimate of potential model size reduction percentage.

        Args:
            constant_nodes: List of constant-only nodes

        Returns:
            Estimated reduction percentage (1-15%)
        """
        total_nodes = len(self.graph.node)
        if total_nodes == 0:
            return 1

        # Rough heuristic: each constant-only node contributes to size reduction
        # Assume each node reduces size by 0.5-1.5% depending on fraction
        reduction_pct = int((len(constant_nodes) / total_nodes) * 20)
        return min(max(reduction_pct, 1), 15)  # Clamp between 1-15%
