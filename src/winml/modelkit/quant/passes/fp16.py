# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""FP16 conversion pass."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from .base import BaseQuantPass


if TYPE_CHECKING:
    from pathlib import Path

    from ..config import QuantizeResult


logger = logging.getLogger(__name__)


class FP16Pass(BaseQuantPass):
    """Convert an ONNX model to FP16.

    Reads from :class:`~winml.modelkit.quant.config.WinMLQuantizationConfig`:

    - ``fp16_keep_io_types`` — keep model inputs/outputs in their original dtype
    - ``fp16_op_block_list`` — op types that must not be cast to FP16

    Example::

        pass_ = FP16Pass(config)
        result = pass_.run("model.onnx", "model_fp16.onnx")
    """

    def run(
        self,
        model_path: Path,
        output_path: Path,
        *,
        use_external_data: bool = True,
    ) -> QuantizeResult:
        """Convert *model_path* to FP16 and write the result to *output_path*."""
        from ...onnx import save_onnx
        from ..config import QuantizeResult
        from ..fp16 import convert_to_fp16

        if self._config.calibration_data is not None:
            logger.warning(
                "calibration_data is set but this is an FP16Pass"
                " — calibration data will be ignored."
            )

        start_time = time.perf_counter()
        errors: list[str] = []
        warnings: list[str] = []

        logger.info("Running FP16-only conversion (no quantization)...")
        model = convert_to_fp16(
            model_path,
            keep_io_types=self._config.fp16_keep_io_types,
            op_block_list=self._config.fp16_op_block_list,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_onnx(model, output_path, use_external_data=use_external_data)

        total_time = time.perf_counter() - start_time
        logger.info(
            "FP16 conversion complete: %s -> %s (%.2fs)",
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
