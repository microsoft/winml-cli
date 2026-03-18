"""Validator for detecting dynamic inputs in models (NPU only).

When model has dynamic inputs, warns that dynamic input parsing is not currently supported.
"""

from __future__ import annotations

import json
import logging

from ...models.information import Action, ActionLevel, Information
from .base import ModelValidator


logger = logging.getLogger(__name__)


class DynamicInputValidator(ModelValidator):
    """Detects dynamic inputs in model and warns about lack of support on NPU."""

    @property
    def validator_name(self) -> str:
        """Return validator name."""
        return "DynamicInputValidator"

    @property
    def pattern_id(self) -> str:
        """Return pattern ID for Information objects."""
        return "MODEL/DynamicInput"

    def validate(self) -> Information | None:
        """Check if model has dynamic inputs.

        Returns:
            Information object if dynamic inputs detected, None otherwise
        """
        logger.debug(f"{self.validator_name}: Starting validation")

        dynamic_inputs = self._find_dynamic_inputs()

        if not dynamic_inputs:
            logger.debug(f"{self.validator_name}: No dynamic inputs found")
            return None

        logger.warning(
            f"{self.validator_name}: Found {len(dynamic_inputs)} input(s) with dynamic shapes"
        )

        return self._create_information(dynamic_inputs)

    def _find_dynamic_inputs(self) -> list[dict]:
        """Find model inputs with dynamic dimensions.

        A dimension is considered dynamic if:
        - dim_param is set (symbolic dimension like "batch_size")
        - dim_value is <= 0 (represents dynamic dimension)

        Returns:
            List of dicts with input info: {name, shape, dynamic_dims}
        """
        dynamic_inputs = []

        for input_tensor in self.graph.input:
            # Skip if input doesn't have tensor type
            if not input_tensor.type.HasField("tensor_type"):
                continue

            tensor_type = input_tensor.type.tensor_type

            # Skip if no shape information
            if not tensor_type.HasField("shape"):
                continue

            shape = tensor_type.shape
            dynamic_dims = []
            shape_str_parts = []

            for idx, dim in enumerate(shape.dim):
                if dim.HasField("dim_param") and dim.dim_param:
                    # Symbolic dimension (e.g., "batch_size", "sequence_length")
                    dynamic_dims.append({"index": idx, "type": "symbolic", "value": dim.dim_param})
                    shape_str_parts.append(dim.dim_param)
                elif dim.HasField("dim_value"):
                    if dim.dim_value <= 0:
                        # Numeric dynamic dimension (0 or -1)
                        dynamic_dims.append(
                            {"index": idx, "type": "numeric", "value": dim.dim_value}
                        )
                        shape_str_parts.append(f"dynamic({dim.dim_value})")
                    else:
                        shape_str_parts.append(str(dim.dim_value))
                else:
                    # Unknown dimension
                    dynamic_dims.append({"index": idx, "type": "unknown", "value": None})
                    shape_str_parts.append("?")

            if dynamic_dims:
                shape_str = f"[{', '.join(shape_str_parts)}]"
                dynamic_inputs.append(
                    {
                        "name": input_tensor.name,
                        "shape": shape_str,
                        "dynamic_dims": dynamic_dims,
                    }
                )

        if dynamic_inputs:
            input_names = [inp["name"] for inp in dynamic_inputs]
            logger.debug(f"{self.validator_name}: Found dynamic inputs: {input_names}")

        return dynamic_inputs

    def _create_information(self, dynamic_inputs: list[dict]) -> Information:
        """Create Information object with warning about dynamic input support.

        Args:
            dynamic_inputs: List of inputs with dynamic dimensions

        Returns:
            Information object with recommendations
        """
        # Build input information for JSON details
        inputs_info = []
        for inp in dynamic_inputs:
            dim_details = []
            for dim in inp["dynamic_dims"]:
                if dim["type"] == "symbolic":
                    dim_details.append(f"dim_{dim['index']}: {dim['value']}")
                elif dim["type"] == "numeric":
                    dim_details.append(f"dim_{dim['index']}: dynamic({dim['value']})")
                else:
                    dim_details.append(f"dim_{dim['index']}: unknown")

            inputs_info.append(
                {
                    "input_name": inp["name"],
                    "shape": inp["shape"],
                    "dynamic_dimensions": dim_details,
                }
            )

        # Create recommendation action
        action = Action(
            pattern_from_id="",
            pattern_to_id="",
            level=ActionLevel.REQUIRED,
            details=json.dumps(
                [
                    {
                        "method": "Use static shapes",
                        "description": "Convert model to use fixed input shapes",
                        "note": "Dynamic input parsing is not currently supported on NPU. "
                        "Please use a model with static input dimensions.",
                    },
                    {
                        "method": "Freeze input shapes",
                        "description": "Use ONNX tools to freeze dynamic dimensions to specific values",
                        "command": "python -m onnxruntime.tools.symbolic_shape_infer --input model.onnx --output model_static.onnx --auto_merge",
                    },
                ],
                indent=2,
            ),
        )

        # Build summary of dynamic inputs
        input_summary = "\n".join(
            f"  - {inp['name']}: shape {inp['shape']}" for inp in dynamic_inputs[:5]
        )
        if len(dynamic_inputs) > 5:
            input_summary += f"\n  ... and {len(dynamic_inputs) - 5} more"

        explanation = (
            f"Model contains {len(dynamic_inputs)} dynamic input(s). "
            "Dynamic input parsing is not currently supported.\n\n"
            f"Detected dynamic inputs:\n{input_summary}\n\n"
            "Dynamic inputs have dimensions that are determined at runtime "
            "(e.g., batch size, sequence length).\n"
            "The NPU backend does not currently support complete static analysis "
            "of models with dynamic inputs.\n\n"
            "Recommendations:\n"
            "- Convert the model to use fixed input shapes\n"
            "- Use ONNX tools to freeze dynamic dimensions to specific values"
        )

        # Pattern node list includes input names for tracking
        pattern_node_list = [[inp["name"]] for inp in dynamic_inputs]

        return Information(
            explanation=explanation,
            actions=[action],
            pattern_id=self.pattern_id,
            pattern_node_list=pattern_node_list,
        )
