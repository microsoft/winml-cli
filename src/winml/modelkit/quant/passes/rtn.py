# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""RTN (Round-To-Nearest) weight-only quantization pass."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from .base import BaseQuantPass


if TYPE_CHECKING:
    from pathlib import Path

    from ..config import QuantizeResult, WinMLQuantizationConfig


logger = logging.getLogger(__name__)


class RTNPass(BaseQuantPass):
    """RTN weight-only quantization pass.

    Reads from :class:`~winml.modelkit.quant.config.WinMLQuantizationConfig`:

    - ``rtn_bits`` — quantization bit-width (default 4)
    - ``rtn_block_size`` — block size for quantization (default 128)
    - ``rtn_symmetric`` — symmetric quantization (default True)
    - ``rtn_accuracy_level`` — ORT accuracy level 0-4 (0 = disabled)
    - ``nodes_to_exclude`` — node names to skip

    Example::

        pass_ = RTNPass(config)
        result = pass_.run("model.onnx", "model_rtn.onnx")
    """

    def __init__(self, config: WinMLQuantizationConfig) -> None:
        super().__init__(config)

    def run(
        self,
        model_path: Path,
        output_path: Path,
        *,
        use_external_data: bool = True,
    ) -> QuantizeResult:
        """Apply RTN weight-only quantization to *model_path*."""
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

        from ...onnx import save_onnx
        from ..config import QuantizeResult

        if self._config.calibration_data is not None:
            logger.warning(
                "calibration_data is set but this is an RTNPass — calibration data will be ignored."
            )

        start_time = time.perf_counter()
        errors: list[str] = []
        warnings: list[str] = []

        logger.info(
            "Running RTN %d-bit weight-only quantization (block_size=%d, symmetric=%s)...",
            self._config.rtn_bits,
            self._config.rtn_block_size,
            self._config.rtn_symmetric,
        )

        accuracy_level = (
            self._config.rtn_accuracy_level if self._config.rtn_accuracy_level != 0 else None
        )

        quantizer = MatMulNBitsQuantizer(
            model=str(model_path),
            bits=self._config.rtn_bits,
            block_size=self._config.rtn_block_size,
            is_symmetric=self._config.rtn_symmetric,
            accuracy_level=accuracy_level,
            nodes_to_exclude=self._config.nodes_to_exclude,
        )
        quantizer.process()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        quantized_model = quantizer.model.model

        save_onnx(quantized_model, output_path, use_external_data=use_external_data)

        total_time = time.perf_counter() - start_time
        logger.info(
            "RTN quantization complete: %s -> %s (%.2fs)",
            model_path.name,
            output_path.name,
            total_time,
        )
        return QuantizeResult(
            success=True,
            output_path=output_path,
            total_time_seconds=total_time,
            errors=errors,
            warnings=warnings,
        )
