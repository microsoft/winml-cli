# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Base protocol for model-type-specific quantization policies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from pathlib import Path

    from ..config import WinMLQuantizationConfig


@runtime_checkable
class QuantConfigFinalizer(Protocol):
    """Model-type-specific quant policy.

    Given the freshly exported ONNX, a finalizer populates the live
    :class:`WinMLQuantizationConfig` with the fields that can only be known
    once the graph exists — the calibration data reader, ``nodes_to_exclude``,
    and (where the scheme is fixed and reference-matched) the dtype/symmetry
    settings.

    Finalizers are registered per ``model_type`` (see
    :func:`.registry.register_quant_finalizer`). Model types without a
    registered policy fall back to the quantizer's default
    ``DatasetCalibrationReader``.
    """

    def finalize(
        self,
        quant: WinMLQuantizationConfig,
        *,
        onnx_path: Path,
        model_id: str | None = None,
    ) -> WinMLQuantizationConfig:
        """Return ``quant`` populated with the graph-derived quant settings."""
