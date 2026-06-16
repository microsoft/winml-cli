# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Validator for pattern matching errors.

Reports when model validation fails during subgraph pattern matching.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import ClassVar

from ...models import ModelTag
from ...models.information import Action, ActionLevel, Information
from .base import ModelValidator


logger = logging.getLogger(__name__)


@dataclass
class PatternErrorConfig:
    """Configuration for a specific pattern matching error type.

    Attributes:
        tag: ModelTag to check for
        error_message: Short error description
        explanation_template: Detailed explanation with recommendations
        action_method: Tool or method name for fixing the issue
        action_description: Description of what the action does
        action_command: Command line to execute (if applicable)
    """

    tag: str
    error_message: str
    explanation_template: str
    action_method: str
    action_description: str
    action_command: str | None = None


class PatternMatchingValidator(ModelValidator):
    """Reports pattern matching validation errors.

    This validator checks for various model issues that prevent pattern matching.
    Error configurations are centralized for easy maintenance and extension.
    """

    # Centralized error configurations - add new error types here
    ERROR_CONFIGS: ClassVar[list] = [
        PatternErrorConfig(
            tag=ModelTag.MISSING_NODE_NAMES,
            error_message=(
                "Model has nodes with empty names - all nodes must"
                " have non-empty names for pattern matching"
            ),
            explanation_template=(
                "Model validation failed for subgraph pattern matching.\n\n"
                "Error: {error_msg}\n\n"
                "Subgraph pattern matching has been skipped. "
                "Pattern matching requires all nodes to have non-empty names.\n\n"
            ),
            # Todo: Update with actual command when available
            action_method="winml onnx_normalize",
            action_description=("Add missing node names to the model using ONNX utilities"),
            action_command=(
                "[Placeholder] winml onnx_normalize <input_model.onnx> <output_model.onnx>"
            ),
        ),
        PatternErrorConfig(
            tag=ModelTag.INVALID_PATTERN_MATCHER_MODEL,
            error_message="Model validation failed during pattern matching initialization",
            explanation_template=(
                "Model validation failed for subgraph pattern matching.\n\n"
                "Error: {error_msg}\n\n"
                "Subgraph pattern matching has been skipped. "
                "Please fix the model issues to enable pattern "
                "detection and optimization recommendations.\n\n"
            ),
            action_method="onnx.checker.check_model",
            action_description=("Validate and fix model structure using ONNX checker"),
            action_command=(
                'python -c "import onnx; model ='
                " onnx.load('input.onnx');"
                ' onnx.checker.check_model(model)"'
            ),
        ),
    ]

    @property
    def validator_name(self) -> str:
        """Return validator name."""
        return "PatternMatchingValidator"

    @property
    def pattern_id(self) -> str:
        """Return pattern ID for Information objects."""
        return "MODEL/PatternMatchingError"

    def validate(self) -> Information | None:
        """Check if pattern matching error occurred.

        Returns:
            Information object if error present, None otherwise
        """
        # Check for any configured error tags
        detected_error, actual_error_msg = self._detect_error()

        if detected_error is None:
            logger.debug(f"{self.validator_name}: No pattern matching error")
            return None

        logger.warning(
            f"{self.validator_name}: Pattern matching failed - {detected_error.error_message}"
        )

        return self._create_information(detected_error, actual_error_msg)

    def _detect_error(self) -> tuple[PatternErrorConfig, str] | tuple[None, None]:
        """Detect which pattern matching error occurred.

        Returns:
            Tuple of (PatternErrorConfig, actual_error_message) for the first detected error,
            or (None, None) if no errors
        """
        for config in self.ERROR_CONFIGS:
            if config.tag in self.model.model_tags:
                actual_msg = self.model.model_tags[config.tag]
                return config, actual_msg
        return None, None

    def _create_information(
        self, error_config: PatternErrorConfig, actual_error_msg: str | None
    ) -> Information:
        """Create Information object with pattern matching error details.

        Args:
            error_config: Configuration for the detected error
            actual_error_msg: Actual error message from the exception

        Returns:
            Information object with error explanation and recommendations
        """
        # Use actual error message if available, otherwise use config message
        error_msg = actual_error_msg or error_config.error_message

        # Create action with details - provide actionable steps for fixing
        action_detail = {
            "method": error_config.action_method,
            "description": error_config.action_description,
        }
        if error_config.action_command:
            action_detail["command"] = error_config.action_command

        action = Action(
            pattern_from_id="",
            pattern_to_id="",
            level=ActionLevel.REQUIRED,
            details=json.dumps([action_detail], indent=2),
        )

        # Format explanation using template
        explanation = error_config.explanation_template.format(error_msg=error_msg)

        return Information(
            explanation=explanation,
            actions=[action],
            pattern_id=self.pattern_id,
            pattern_node_list=[],
        )
