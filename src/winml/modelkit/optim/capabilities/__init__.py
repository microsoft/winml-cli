"""Optimization capability definitions.

Capability modules define individual optimization capabilities that can be
enabled/disabled. Import specific modules directly:

    from winml.modelkit.optim.capabilities import gelu, layernorm
"""

from __future__ import annotations

from . import (
    activation,
    attention,
    conv,
    elimination,
    gelu,
    gemm,
    graph,
    layernorm,
    layout,
    matmul,
    misc,
    surgery,
)


__all__ = [
    "activation",
    "attention",
    "conv",
    "elimination",
    "gelu",
    "gemm",
    "graph",
    "layernorm",
    "layout",
    "matmul",
    "misc",
    "surgery",
]
