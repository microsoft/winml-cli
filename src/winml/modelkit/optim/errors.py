# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Error types for ONNX optimization.

This module defines specialized exception types for different failure modes
in the optimization pipeline:

- OptimizationError: Pipeline execution failures
- ConfigurationError: Invalid configuration
- ModelValidationError: Invalid ONNX model

Example:
    from winml.modelkit.optim.errors import OptimizationError, ConfigurationError

    try:
        model = optimize_onnx("model.onnx", gelu_fusion=True)
    except OptimizationError as e:
        print(f"Optimization failed: {e.message}")
        print(f"Pipe: {e.pipe_name}")
    except ConfigurationError as e:
        print(f"Config error: {e.message}")
        for error in e.errors:
            print(f"  - {error}")
"""

from __future__ import annotations

from typing import Any


class OptimizationError(Exception):
    """Raised when optimization pipeline fails.

    Includes context about which pipe failed and model state.

    Attributes:
        message: Error description.
        pipe_name: Name of the pipe that raised the error.
        model_info: Optional information about the model being processed.
        cause: Optional underlying exception that triggered this error.

    Example:
        raise OptimizationError(
            "Failed to apply GELU fusion",
            pipe_name="ort_graph",
            model_info={"nodes": 150, "optimization_level": 2},
            cause=ort_error,
        )
    """

    def __init__(
        self,
        message: str,
        pipe_name: str | None = None,
        model_info: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize optimization error.

        Args:
            message: Error description.
            pipe_name: Name of the pipe that raised the error.
            model_info: Optional information about the model.
            cause: Optional underlying exception.
        """
        self.message = message
        self.pipe_name = pipe_name
        self.model_info = model_info or {}
        self.cause = cause
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        """Build formatted error message with context.

        Returns:
            Formatted message string with pipe name, model info, and cause.
        """
        parts = [self.message]
        if self.pipe_name:
            parts.append(f"Pipe: {self.pipe_name}")
        if self.model_info:
            parts.append(f"Model info: {self.model_info}")
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
        return " | ".join(parts)


class ConfigurationError(Exception):
    """Raised when configuration is invalid.

    Includes details about validation errors.

    Attributes:
        message: Error description.
        errors: List of specific validation errors.

    Example:
        raise ConfigurationError(
            "Invalid configuration",
            errors=[
                "Unknown capability 'invalid-name'",
                "'bias-gelu-fusion' requires 'gelu-fusion' to be enabled",
            ],
        )
    """

    def __init__(
        self,
        message: str,
        errors: list[str] | None = None,
    ) -> None:
        """Initialize configuration error.

        Args:
            message: Error description.
            errors: List of specific validation errors.
        """
        self.message = message
        self.errors = errors or []
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        """Build formatted error message with validation errors.

        Returns:
            Formatted message string with bulleted error list.
        """
        if self.errors:
            error_list = "\n  - ".join(self.errors)
            return f"{self.message}:\n  - {error_list}"
        return self.message


class ModelValidationError(Exception):
    """Raised when ONNX model is invalid.

    Wraps onnx.checker validation errors with context.

    Attributes:
        message: Error description.
        model_path: Path to the model file (if applicable).
        cause: Underlying validation exception.

    Example:
        raise ModelValidationError(
            "Input model failed ONNX validation",
            model_path="model.onnx",
            cause=validation_error,
        )
    """

    def __init__(
        self,
        message: str,
        model_path: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize model validation error.

        Args:
            message: Error description.
            model_path: Path to the model file.
            cause: Underlying validation exception.
        """
        self.message = message
        self.model_path = model_path
        self.cause = cause
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        """Build formatted error message with path and cause.

        Returns:
            Formatted message string with model path and validation error.
        """
        parts = [self.message]
        if self.model_path:
            parts.append(f"Path: {self.model_path}")
        if self.cause:
            parts.append(f"Validation error: {self.cause}")
        return " | ".join(parts)
