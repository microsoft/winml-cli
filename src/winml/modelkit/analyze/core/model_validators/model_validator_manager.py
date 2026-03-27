# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Manager for model validators.

Manages and executes multiple model validators, collecting their results
into Information objects.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from .constant_folding_validator import ConstantFoldingValidator
from .dynamic_input_validator import DynamicInputValidator
from .pattern_matching_validator import PatternMatchingValidator
from .qdq_validation_validator import QDQValidationValidator
from .shape_inference_validator import ShapeInferenceValidator


if TYPE_CHECKING:
    from ...models.information import Information
    from ...models.onnx_model import ONNXModel
    from ...models.runtime_checks import PatternRuntime
    from .base import ModelValidator

logger = logging.getLogger(__name__)


class ModelValidatorManager:
    """Manages and executes model validators.

    Attributes:
        model: ONNX model wrapper to validate
        op_runtime_results: List of PatternRuntime results from runtime checker
        validators: List of validator instances
        device: Device type (e.g., "NPU", "GPU", "CPU")
    """

    # Registry of available validators with device constraints
    VALIDATORS: ClassVar[dict] = {
        "constant_folding": {
            "class": ConstantFoldingValidator,
            "enabled_devices": None,  # None means enabled for all devices
        },
        "shape_inference": {
            "class": ShapeInferenceValidator,
            "enabled_devices": ["NPU"],  # Only enabled for NPU device
        },
        "qdq_validation": {
            "class": QDQValidationValidator,
            "enabled_devices": None,  # QDQ issues affect all EPs
        },
        "dynamic_input": {
            "class": DynamicInputValidator,
            "enabled_devices": ["NPU"],  # Only enabled for NPU device
        },
        "pattern_matching": {
            "class": PatternMatchingValidator,
            "enabled_devices": None,  # All devices
        },
    }

    def __init__(
        self,
        model: ONNXModel,
        enabled_validators: list[str] | None = None,
        op_runtime_results: list[PatternRuntime] | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize validator manager.

        Args:
            model: ONNX model wrapper to validate
            enabled_validators: List of validator names to enable.
                               If None, all validators are enabled (subject to device constraints).
            op_runtime_results: List of PatternRuntime results from runtime checker.
                               Used to enrich validators with OP-level information.
            device: Device type (e.g., "NPU", "GPU", "CPU").
                   Used to filter validators based on device constraints.

        Raises:
            ValueError: If model is not valid ONNXModel instance
            Warning: If unknown validator names are provided
        """
        self.model = model
        self.model_proto = model.get_model()
        self.op_runtime_results = op_runtime_results or []
        self.device = device or "NPU"
        self.device = device
        self.enabled_validators = enabled_validators or list(self.VALIDATORS.keys())

        # Instantiate enabled validators
        self.validators: list[ModelValidator] = []
        for name in self.enabled_validators:
            if name in self.VALIDATORS:
                validator_config = self.VALIDATORS[name]
                validator_class = validator_config["class"]
                enabled_devices = validator_config.get("enabled_devices")

                # Check device constraint
                if enabled_devices is not None and self.device not in enabled_devices:
                    logger.info(
                        f"Validator '{name}' is not enabled for device '{self.device}'. "
                        f"Only enabled for: {enabled_devices}"
                    )
                    continue

                try:
                    self.validators.append(
                        validator_class(self.model, op_runtime_results=self.op_runtime_results)
                    )
                    logger.debug(f"Initialized validator: {name}")
                except Exception:
                    logger.exception(f"Failed to initialize validator {name}")
            else:
                logger.warning(f"Unknown validator: {name}")

        logger.info(
            f"ModelValidatorManager initialized with {len(self.validators)} validator(s), "
            f"enriched with {len(self.op_runtime_results)} runtime result(s)"
        )

    def run_all_validators(self) -> list[Information]:
        """Run all enabled validators and collect Information.

        Returns:
            List of Information objects from validators that found issues

        Raises:
            Exception: Individual validator exceptions are caught and logged,
                      not raised to caller
        """
        information_list: list[Information] = []

        for validator in self.validators:
            try:
                logger.debug(f"Running validator: {validator.validator_name}")
                info = validator.validate()
                if info:
                    logger.info(f"{validator.validator_name} found issue: {info.pattern_id}")
                    information_list.append(info)
            except Exception as e:  # noqa: PERF203
                logger.exception(
                    f"Validator {validator.validator_name} failed with exception: "
                    f"{type(e).__name__}",
                )

        logger.info(
            f"Validation complete: {len(information_list)} issue(s) detected "
            f"by {len(self.validators)} validator(s)"
        )

        return information_list

    @classmethod
    def get_available_validators(cls) -> list[str]:
        """Get list of available validator names.

        Returns:
            List of validator names that can be used with enabled_validators
        """
        return list(cls.VALIDATORS.keys())
