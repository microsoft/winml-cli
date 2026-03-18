# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Base stage interface for compilation pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar


if TYPE_CHECKING:
    from ..context import CompileContext


class BaseStage(ABC):
    """Abstract base class for compilation stages.

    Each stage is responsible for a single transformation in the pipeline.
    Stages are stateless - they receive context and return modified context.

    Example:
        class MyStage(BaseStage):
            name = "my-stage"

            @classmethod
            def should_run(cls, context: CompileContext) -> bool:
                return context.config.get("enable_my_stage", True)

            def process(self, context: CompileContext) -> CompileContext:
                # Do transformation
                return context
    """

    # Unique identifier for this stage
    name: ClassVar[str]

    @classmethod
    @abstractmethod
    def should_run(cls, context: CompileContext) -> bool:
        """Determine if this stage should execute.

        Args:
            context: Current compilation context

        Returns:
            True if stage should run, False to skip
        """
        ...

    @abstractmethod
    def process(self, context: CompileContext) -> CompileContext:
        """Execute stage transformation.

        Args:
            context: Current compilation context

        Returns:
            Modified compilation context
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
