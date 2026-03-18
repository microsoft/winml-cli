"""WinMLOptimizationConfig - Graph Optimization Configuration.

Dict-like config for capability-based optimization system.
"""

from __future__ import annotations


class WinMLOptimizationConfig(dict):
    """Dict-like optimization config for capability kwargs.

    Example:
        config = WinMLOptimizationConfig(gelu_fusion=True, matmul_add_fusion=True)
        optimize_onnx(model, **config)
    """

    def __init__(self, **kwargs: bool) -> None:
        super().__init__(kwargs)

    def to_dict(self) -> dict:
        """Convert to dictionary (sorted keys for deterministic serialization)."""
        return dict(sorted(self.items()))

    @classmethod
    def from_dict(cls, data: dict) -> WinMLOptimizationConfig:
        """Create from dictionary."""
        return cls(**data)
