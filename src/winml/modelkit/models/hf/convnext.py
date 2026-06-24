# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ConvNeXT ONNX export patch: LayerNorm fusion via PATCHING_SPECS.

Patches ConvNextLayerNorm.forward during export so that ONNX runtimes can
fuse the resulting F.layer_norm ops.  No model-specific build config is
needed; the analyzer autoconf loop discovers optimization flags
automatically from the ONNX graph structure (see issue #232).

How it works
------------
Optimum's ``ModelPatcher.patch_ops()`` does::

    setattr(spec.o, spec.name, custom_op)   # class-level replacement

Because Python's descriptor protocol turns a plain function set on a class
into an unbound method, ``_patched_layernorm_forward(self, x)`` receives
the ``ConvNextLayerNorm`` instance as ``self`` automatically -- no
``types.MethodType`` wrapping required.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from optimum.exporters.onnx.model_configs import ConvNextOnnxConfig
from optimum.exporters.onnx.model_patcher import PatchingSpec

from ...export import register_onnx_overwrite


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patched forward
# ---------------------------------------------------------------------------


def _patched_layernorm_forward(self, x: torch.Tensor) -> torch.Tensor:
    """ConvNextLayerNorm.forward replacement that enables ONNX LayerNorm fusion.

    The stock implementation branches on ``data_format`` with code paths
    that the ONNX exporter cannot fuse.  This version uses
    ``F.layer_norm`` with explicit channel permutations so the graph
    contains a single, fusible LayerNormalization node.

    Args:
        x: Input tensor (NHWC or NCHW depending on ``self.data_format``).

    Returns:
        Normalized tensor in the same layout as the input.
    """
    if self.data_format == "channels_last":
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    # channels_first: NCHW -> NHWC -> LayerNorm -> NCHW
    x = x.permute(0, 2, 3, 1)
    x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    return x.permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Build PATCHING_SPECS (guarded: transformers may lack the ConvNeXT module)
# ---------------------------------------------------------------------------


def _build_patching_specs() -> list[PatchingSpec]:
    """Return PatchingSpec list, or [] if ConvNextLayerNorm is unavailable."""
    try:
        from transformers.models.convnext.modeling_convnext import (
            ConvNextLayerNorm,
        )
    except ImportError:
        logger.debug(
            "ConvNextLayerNorm not found in transformers; "
            "LayerNorm fusion patch will not be applied during export.",
        )
        return []

    return [
        PatchingSpec(
            o=ConvNextLayerNorm,
            name="forward",
            custom_op=_patched_layernorm_forward,
        ),
    ]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@register_onnx_overwrite(
    "convnext",
    "feature-extraction",
    "image-classification",
    library_name="transformers",
)
class ConvNextIOConfig(ConvNextOnnxConfig):  # type: ignore[misc]  # optimum base is untyped
    """ConvNextOnnxConfig override that adds a LayerNorm fusion patch.

    Inherits all I/O specs from Optimum's ``ConvNextOnnxConfig``.  The only
    addition is ``PATCHING_SPECS``, which ``ModelPatcher`` applies as a
    context manager during export.  A subclass is required because
    ``register_onnx_overwrite`` needs a class to register, and mutating
    Optimum's ``ConvNextOnnxConfig.PATCHING_SPECS`` globally would be
    a side-effect visible to other users of that class.
    """

    PATCHING_SPECS = _build_patching_specs()
