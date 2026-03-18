"""Validator for detecting invalid QDQ quantization parameters."""

from __future__ import annotations

import json
import logging

import numpy as np
from onnx import numpy_helper

from ....compiler import QDQ_OP_TYPES
from ...models.information import Action, ActionLevel, Information
from .base import ModelValidator


logger = logging.getLogger(__name__)


class QDQValidationValidator(ModelValidator):
    """Detects invalid QDQ quantization parameters in model initializers."""

    @property
    def validator_name(self) -> str:
        return "QDQValidationValidator"

    @property
    def pattern_id(self) -> str:
        return "MODEL/QDQValidation"

    def validate(self) -> Information | None:
        invalid_nodes = self._find_invalid_qdq_params()

        if not invalid_nodes:
            logger.debug(f"{self.validator_name}: No invalid QDQ parameters found")
            return None

        logger.warning(
            f"{self.validator_name}: Found {len(invalid_nodes)} initializer(s) with invalid values"
        )
        return self._create_information(invalid_nodes)

    def _find_invalid_qdq_params(self) -> list[str]:
        """Find QDQ nodes with invalid scale or zero_point values.

        Iterates through QuantizeLinear and DequantizeLinear nodes and checks:
        - Scale (input[1]): invalid if Inf, NaN, or 0
        - Zero point (input[2], optional): invalid if Inf or NaN
        """
        # Build initializer lookup
        initializers = {init.name: init for init in self.graph.initializer}

        invalid = set()

        for node in self.graph.node:
            if node.op_type not in QDQ_OP_TYPES:
                continue

            # Check scale (input[1])
            if len(node.input) > 1:
                scale_name = node.input[1]
                if scale_name in initializers:
                    arr = numpy_helper.to_array(initializers[scale_name])
                    if np.isinf(arr).any() or np.isnan(arr).any() or (arr == 0).any():
                        invalid.add(scale_name)

            # Check zero_point (input[2], optional)
            if len(node.input) > 2:
                zp_name = node.input[2]
                if zp_name and zp_name in initializers:
                    arr = numpy_helper.to_array(initializers[zp_name])
                    if np.isinf(arr).any() or np.isnan(arr).any():
                        invalid.add(zp_name)

        return sorted(invalid)

    def _create_information(self, invalid_nodes: list[str]) -> Information:
        examples = ", ".join(invalid_nodes[:3])
        more_text = f" and {len(invalid_nodes) - 3} more" if len(invalid_nodes) > 3 else ""

        explanation = (
            f"Model contains {len(invalid_nodes)} QDQ node(s) with invalid quantization parameters. "
            f"Examples: {examples}{more_text}. "
            "Scale is invalid if the value is Inf, NaN, or 0. Zero point is invalid if the value is Inf or NaN. "
        )

        action = Action(
            pattern_from_id="",
            pattern_to_id="",
            level=ActionLevel.WARNING,
            details=json.dumps(
                [
                    {
                        "title": "Re-quantize model",
                        "command": "wmk quantize --model model.onnx --output model-qdq.onnx",
                    }
                ],
                indent=2,
            ),
        )

        return Information(
            explanation=explanation,
            actions=[action],
            pattern_id=self.pattern_id,
            pattern_node_list=[[name] for name in invalid_nodes],
        )
