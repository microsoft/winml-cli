# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Base class for quantization passes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path

    from ..config import QuantizeResult, WinMLQuantizationConfig


class BaseQuantPass(ABC):
    """Abstract base class for a single quantization pass.

    Each pass is constructed with a ``WinMLQuantizationConfig`` that provides
    all settings.  Passes read only the fields relevant to them and ignore the
    rest, so a single shared config object can be threaded through every pass
    in a :class:`~winml.modelkit.quant.quantizer.Quantizer` pipeline.

    Example::

        pass_ = FP16Pass(config)
        result = pass_.run(model_path, output_path)
    """

    def __init__(self, config: WinMLQuantizationConfig) -> None:
        self._config = config

    @property
    def config(self) -> WinMLQuantizationConfig:
        """Return the shared quantization configuration."""
        return self._config

    @abstractmethod
    def run(
        self,
        model_path: Path,
        output_path: Path,
        *,
        use_external_data: bool = True,
    ) -> QuantizeResult:
        """Run this quantization pass.

        Args:
            model_path: Path to the input ONNX model.
            output_path: Path where the output ONNX model should be written.
            use_external_data: Whether to write large tensors as external data.

        Returns:
            :class:`~winml.modelkit.quant.config.QuantizeResult` describing
            the outcome of this pass.

        Note:
            Passes use file-based I/O because ORT's calibration and RTN APIs
            operate on paths, and external-data models cannot be held fully in
            memory.  A future enhancement could add an optional in-memory
            fast-path for small single-pass models.
        """
