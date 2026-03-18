"""Base classes for optimization pipes.

A Pipe represents one optimization technology/API. Each pipe:
- Has a unique name
- Knows how to build its config from kwargs
- Knows how to process an ONNX model
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from ..errors import OptimizationError
from ..registry import CapabilityDef  # noqa: TC001 (used at runtime)


# Re-export for backward compatibility
__all__ = ["BasePipe", "OptimizationError", "PipeConfig", "caps_dict"]

if TYPE_CHECKING:
    import onnx


def caps_dict(*capabilities: CapabilityDef) -> dict[str, CapabilityDef]:
    """Build capabilities dict from capability objects.

    Helper to reduce boilerplate when defining pipe capabilities.
    Instead of writing {cap.name: cap, ...} for each capability,
    just pass them as positional arguments.

    Example:
        capabilities = caps_dict(
            gelu.GELU_FUSION,
            gelu.BIAS_GELU_FUSION,
            layernorm.LAYER_NORM_FUSION,
        )
    """
    return {cap.name: cap for cap in capabilities}


@dataclass
class PipeConfig:
    """Base configuration for optimization pipes.

    All pipe configs inherit from this class. The base class is intentionally
    minimal - each pipe defines its own config structure.
    """


class BasePipe(ABC):
    """Abstract base class for optimization pipes.

    A Pipe represents one optimization technology/API. Each pipe:
    - Has a unique name
    - Has a capabilities dict mapping names to capability definitions
    - Knows how to build its config from kwargs
    - Knows how to process an ONNX model
    """

    # Pipe identifier (must be unique)
    name: ClassVar[str]

    # Capabilities dictionary mapping capability names to definitions
    capabilities: ClassVar[dict[str, Any]] = {}

    @classmethod
    @abstractmethod
    def build_config(cls, **kwargs: Any) -> PipeConfig:
        """Build pipe configuration from user kwargs.

        This method extracts relevant kwargs for this pipe and constructs
        the appropriate PipeConfig subclass.

        Args:
            **kwargs: All user-provided configuration

        Returns:
            Configured PipeConfig for this pipe
        """
        ...

    @abstractmethod
    def process(self, model: onnx.ModelProto, config: PipeConfig) -> onnx.ModelProto:
        """Process ONNX model with given configuration.

        This method applies the pipe's optimizations to the model.
        Called unconditionally - config controls actual behavior.

        Args:
            model: Input ONNX model (will not be modified)
            config: Pipe configuration from build_config()

        Returns:
            New optimized ONNX model
        """
        ...

    @classmethod
    def should_process(cls, config: PipeConfig) -> bool:
        """Check if this pipe should process the model.

        Default implementation returns True (always process).
        Subclasses can override to skip processing based on config.

        Args:
            config: Pipe configuration from build_config()

        Returns:
            True if pipe should process, False to skip
        """
        return True
