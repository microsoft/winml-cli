# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
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

    @classmethod
    def for_cgc(cls) -> WinMLOptimizationConfig:
        """Enable the registered CGIR compatibility rules without ORT graph optimization."""
        from .pipes import CGIRRewritePipe

        return cls(
            ort_graph_optimization=False,
            **CGIRRewritePipe.get_compatibility_options(),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary (sorted keys for deterministic serialization)."""
        return dict(sorted(self.items()))

    @classmethod
    def from_dict(cls, data: dict) -> WinMLOptimizationConfig:
        """Create from dictionary."""
        return cls(**data)
