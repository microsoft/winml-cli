# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Result types for compiler module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CompileResult:
    """Result of compilation."""

    # Status
    success: bool

    # Output paths
    output_path: Path | None = None
    context_binary_path: Path | None = None

    # Timing metrics
    compile_time: float | None = None
    total_time: float = 0.0

    # Model info
    input_shapes: dict[str, list[int]] = field(default_factory=dict)
    output_shapes: dict[str, list[int]] = field(default_factory=dict)

    # Validation metrics
    validation_passed: bool = False
    performance_metrics: dict[str, float] = field(default_factory=dict)

    # Errors and warnings
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "output_path": str(self.output_path) if self.output_path else None,
            "context_binary_path": (
                str(self.context_binary_path) if self.context_binary_path else None
            ),
            "compile_time": self.compile_time,
            "total_time": self.total_time,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "validation_passed": self.validation_passed,
            "performance_metrics": self.performance_metrics,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def __str__(self) -> str:
        """Pretty print result."""
        lines = [
            f"CompileResult(success={self.success})",
            f"  output_path: {self.output_path}",
            f"  total_time: {self.total_time:.2f}s",
        ]
        if self.compile_time:
            lines.append(f"  compile_time: {self.compile_time:.2f}s")
        if self.errors:
            lines.append(f"  errors: {self.errors}")
        if self.warnings:
            lines.append(f"  warnings: {self.warnings}")
        return "\n".join(lines)
