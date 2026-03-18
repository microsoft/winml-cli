# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Compile context for passing data between stages."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort


logger = logging.getLogger(__name__)


@dataclass
class CompileContext:
    """Shared context passed between compilation stages.

    This object carries all state through the pipeline:
    - Input model and paths
    - Configuration options
    - Output paths and metrics
    - Errors and warnings
    """

    # Input
    model_path: Path
    config: dict[str, Any]
    model: onnx.ModelProto | None = None

    # Working directory
    work_dir: Path | None = None

    # Session (set during compile)
    session: ort.InferenceSession | None = None

    # Output paths
    output_path: Path | None = None
    context_binary_path: Path | None = None

    # Metrics
    metrics: dict[str, Any] = field(default_factory=dict)

    # Errors and warnings
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    has_error: bool = False

    # Logging
    logs: list[str] = field(default_factory=list)
    verbose: bool = False

    def log(self, message: str) -> None:
        """Log a message."""
        self.logs.append(message)
        logger.info(message)

    def add_error(self, error: str) -> None:
        """Add an error and set error flag."""
        self.errors.append(error)
        self.has_error = True
        self.logs.append(f"ERROR: {error}")
        logger.error(error)

    def add_warning(self, warning: str) -> None:
        """Add a warning."""
        self.warnings.append(warning)
        self.logs.append(f"WARNING: {warning}")
        logger.warning(warning)

    def add_metric(self, name: str, value: Any) -> None:
        """Add a metric."""
        self.metrics[name] = value

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        return self.config.get(key, default)

    @property
    def execution_provider(self) -> str:
        """Get target execution provider."""
        return self.config.get("execution_provider", "qnn")

    @property
    def enable_ep_context(self) -> bool:
        """Whether to generate EPContext model."""
        return self.config.get("enable_ep_context", True)

    @property
    def validate(self) -> bool:
        """Whether to validate compiled model."""
        return self.config.get("validate", True)
