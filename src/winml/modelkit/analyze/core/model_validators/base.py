"""Base class for model-level validators.

Each validator performs a specific model-level check and generates Information
if issues are detected.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ...models.information import Information
    from ...models.onnx_model import ONNXModel
    from ...models.runtime_checks import PatternRuntime

logger = logging.getLogger(__name__)


class ModelValidator(ABC):
    """Base class for model-level validation checks.

    Each validator performs a specific check on the ONNX model and generates
    Information if issues are detected.

    Attributes:
        model: ONNXModel wrapper to validate
        model_proto: ONNX ModelProto extracted from model
        graph: Shorthand for model_proto.graph
        op_runtime_results: List of PatternRuntime results from runtime checker (optional)
    """

    def __init__(
        self,
        model: ONNXModel,
        op_runtime_results: list[PatternRuntime] | None = None,
    ) -> None:
        """Initialize validator with ONNX model and optional runtime results.

        Args:
            model: ONNXModel wrapper to validate
            op_runtime_results: List of PatternRuntime results from runtime checker.
                               Used to enrich validators with OP-level information.

        Raises:
            ValueError: If model is invalid
        """
        self.model = model
        self.model_proto = model.get_model()
        self.graph = self.model_proto.graph
        self.op_runtime_results = op_runtime_results or []

        logger.debug(
            f"Initialized {self.validator_name} for model with {len(self.graph.node)} nodes"
        )

    @property
    @abstractmethod
    def validator_name(self) -> str:
        """Name of this validator for logging/reporting.

        Returns:
            str: Human-readable validator name (e.g., 'ConstantFoldingValidator')
        """

    @property
    @abstractmethod
    def pattern_id(self) -> str:
        """Pattern ID for Information objects.

        Returns:
            str: Pattern ID in format 'MODEL/<Category>' (e.g., 'MODEL/ConstantFolding')
        """

    @abstractmethod
    def validate(self) -> Information | None:
        """Perform validation check.

        Returns:
            Information object if issue detected, None otherwise

        Raises:
            Exception: Exceptions should be caught by ValidatorManager
        """

    def _is_enabled(self) -> bool:
        """Check if this validator is enabled.

        Override in subclasses to add enable/disable logic.

        Returns:
            bool: True if validator should run, False otherwise
        """
        return True
