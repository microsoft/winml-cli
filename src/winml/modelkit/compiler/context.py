# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Compile context for passing data between stages."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import onnx
import onnxruntime as ort

from ..utils.constants import ORT_SESSION_COMPILER


if TYPE_CHECKING:
    from ..utils.constants import EPAlias


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

    # Multi-model / shared-EP-context compilation state (driven by Compiler).
    # n_compiled_models: how many models the Compiler has already compiled (0-based
    #   index of the current model).
    # n_total_models: total models in this compile run (>1 enables weight sharing).
    # shared_session_options: the shared ort.SessionOptions created on the first model
    #   and reused for the rest (the EP is added once and the share group lives on it).
    n_compiled_models: int = 0
    n_total_models: int = 1
    shared_session_options: ort.SessionOptions | None = None

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
    def execution_provider(self) -> EPAlias:
        """Get target execution provider."""
        return cast("EPAlias", self.config.get("execution_provider", "qnn"))

    @property
    def use_inference_session(self) -> bool:
        """Whether to use the ort.InferenceSession backend (vs ort.ModelCompiler).

        True iff the configured compiler is ``"ort_session"``.
        """
        return self.config.get("compiler") == ORT_SESSION_COMPILER

    @property
    def enable_ep_context(self) -> bool:
        """Whether to generate EPContext model."""
        return bool(self.config.get("enable_ep_context", True))

    @property
    def validate(self) -> bool:
        """Whether to validate compiled model."""
        return bool(self.config.get("validate", True))
