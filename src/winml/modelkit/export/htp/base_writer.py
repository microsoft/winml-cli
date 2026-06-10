# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Base writer class for step-aware export monitoring.

This module provides the abstract base class for all export writers,
using Python's IO protocol and decorator pattern for step handling.
"""

from __future__ import annotations

import contextlib
import io
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# datetime imports removed - following ADR-006 to use float timestamps only
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

from .step_data import (
    HierarchyData,
    InputGenData,
    ModelPrepData,
    NodeTaggingData,
    ONNXExportData,
    TagInjectionData,
)


if TYPE_CHECKING:
    from collections.abc import Callable


class ExportStep(Enum):
    """HTP export process steps."""

    MODEL_PREP = "model_preparation"  # Step 1
    INPUT_GEN = "input_generation"  # Step 2
    HIERARCHY = "hierarchy_building"  # Step 3
    ONNX_EXPORT = "onnx_export"  # Step 4
    NODE_TAGGING = "node_tagging"  # Step 5
    TAG_INJECTION = "tag_injection"  # Step 6


@dataclass
class ExportData:
    """Unified export data shared across all writers."""

    # Session tracking
    export_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Basic info
    model_name: str = ""
    output_path: str = ""
    strategy: str = "htp"
    embed_hierarchy: bool = True

    # Timing
    start_time: float = field(default_factory=lambda: time.time())
    export_time: float = 0.0

    # Typed step data
    model_prep: ModelPrepData | None = None
    input_gen: InputGenData | None = None
    hierarchy: HierarchyData | None = None
    onnx_export: ONNXExportData | None = None
    node_tagging: NodeTaggingData | None = None
    tag_injection: TagInjectionData | None = None

    def get_step_timestamp(self, step: ExportStep) -> float | None:
        """Get the Unix epoch timestamp for a step."""
        # Convert enum name to attribute name (e.g., MODEL_PREP -> model_prep)
        attr_name = step.name.lower()
        data = getattr(self, attr_name, None)
        return data.timestamp if data else None

    @property
    def timestamp(self) -> str:
        """Export start timestamp in ISO 8601 format with Z suffix."""
        from ...core.time_utils import format_timestamp_iso

        return format_timestamp_iso(self.start_time) or ""

    @property
    def elapsed_time(self) -> float:
        """Total elapsed time in seconds."""
        return time.time() - self.start_time


F = TypeVar("F", bound="Callable[..., int]")


def step(export_step: ExportStep) -> Callable[[F], F]:
    """Decorator to mark step-specific handler methods.

    Attaches ``_handles_step`` on the function so ``StepAwareWriter``'s
    discovery loop can map each handler to its declared step. The function is
    returned unchanged, so the original signature is preserved for callers and
    type checkers.
    """

    def decorator(func: F) -> F:
        func._handles_step = export_step  # type: ignore[attr-defined]
        return func

    return decorator


class StepAwareWriter(io.IOBase, ABC):
    """Base class for step-aware writers following Python's IO protocol."""

    def __init__(self) -> None:
        """Initialize the writer and discover step handlers."""
        super().__init__()
        self._step_handlers: dict[ExportStep, Callable[..., int]] = {}
        self._discover_handlers()

    def _discover_handlers(self) -> None:
        """Auto-discover step handler methods decorated with @step."""
        for name in dir(self):
            if name.startswith("_"):
                continue
            method = getattr(self, name)
            if hasattr(method, "_handles_step"):
                step_type = method._handles_step
                self._step_handlers[step_type] = method

    def write(self, export_step: ExportStep, data: ExportData) -> int:
        """Write data for a specific step.

        Args:
            export_step: The current export step
            data: The export data

        Returns:
            Number of bytes written (for IO protocol compliance)
        """
        handler = self._step_handlers.get(export_step, self._write_default)
        return handler(export_step, data)

    @abstractmethod
    def _write_default(self, export_step: ExportStep, data: ExportData) -> int:
        """Default handler for steps without specific handlers.

        Args:
            export_step: The current export step
            data: The export data

        Returns:
            Number of bytes written
        """

    def flush(self) -> None:
        """Flush any buffered data."""

    def close(self) -> None:
        """Close the writer and perform cleanup."""
        with contextlib.suppress(Exception):
            self.flush()
        super().close()

    # Required IO methods for protocol compliance
    def readable(self) -> bool:
        """This is a write-only stream."""
        return False

    def writable(self) -> bool:
        """This stream is writable."""
        return True

    def seekable(self) -> bool:
        """This stream is not seekable."""
        return False
