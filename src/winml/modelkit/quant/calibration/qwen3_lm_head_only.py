# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Weight-only int4 (RTN) quant policy for the ``qwen3_lm_head_only`` build.

The LM head export (``models.hf.qwen3.qwen_lm_head_only``) emits a single
``MatMul`` (the vocab projection). It is quantized weight-only to 4 bits with
``MatMulNBitsQuantizer`` (RTN, symmetric int4 weights).

The build pipeline's device/precision policy only sets ``mode="rtn"`` and
``rtn_bits`` (from the precision string, e.g. ``w4a32``/``int4``); this finalizer
is keyed on ``model_type`` and pins the remaining knobs (block size, symmetry,
accuracy level) so they are authoritative regardless of the defaults baked into
:class:`WinMLQuantizationConfig`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path

    from ..config import WinMLQuantizationConfig


logger = logging.getLogger(__name__)

# Weight-only int4 RTN scheme: block_size=32, accuracy_level=4, symmetric.
LM_HEAD_RTN_BITS = 4
LM_HEAD_RTN_BLOCK_SIZE = 32
LM_HEAD_RTN_ACCURACY_LEVEL = 4
LM_HEAD_RTN_SYMMETRIC = True


def finalize_lm_head_quant_config(
    quant: WinMLQuantizationConfig,
    *,
    onnx_path: Path,
    model_id: str | None = None,
) -> WinMLQuantizationConfig:
    """Pin the weight-only int4 RTN scheme for the LM head.

    The precision policy must already have resolved a weight-only precision
    (``mode="rtn"``); this hook only tightens the RTN knobs (symmetric int4
    weights, block_size=32, accuracy_level=4).
    """
    # ``mode`` must be "rtn": the precision-driven flow keys the quantizer
    # dispatch on ``config.mode``. A build whose precision resolved to "static"
    # or "fp16" would otherwise bypass the MatMulNBits path entirely.
    quant.mode = "rtn"
    quant.rtn_bits = LM_HEAD_RTN_BITS
    quant.rtn_block_size = LM_HEAD_RTN_BLOCK_SIZE
    quant.rtn_symmetric = LM_HEAD_RTN_SYMMETRIC
    quant.rtn_accuracy_level = LM_HEAD_RTN_ACCURACY_LEVEL

    logger.info(
        "Finalizing lm-head quant config for %s "
        "(RTN int%d, block_size=%d, symmetric=%s, accuracy_level=%d)",
        onnx_path.name,
        quant.rtn_bits,
        quant.rtn_block_size,
        quant.rtn_symmetric,
        quant.rtn_accuracy_level,
    )
    return quant


class Qwen3LMHeadOnlyQuantFinalizer:
    """Quant policy for the ``qwen3_lm_head_only`` model_type.

    Named in :data:`~winml.modelkit.quant.calibration.registry.QUANT_FINALIZERS`
    and resolved by :func:`~winml.modelkit.quant.get_quant_finalizer`. Adapts
    :func:`finalize_lm_head_quant_config` to the
    :class:`~winml.modelkit.quant.calibration.base.QuantConfigFinalizer`
    protocol so the build pipeline applies the int4 RTN scheme (keyed on
    ``model_type``).
    """

    def finalize(
        self,
        quant: WinMLQuantizationConfig,
        *,
        onnx_path: Path,
        model_id: str | None = None,
    ) -> WinMLQuantizationConfig:
        """Populate ``quant`` with the LM-head weight-only int4 RTN scheme."""
        return finalize_lm_head_quant_config(quant, onnx_path=onnx_path, model_id=model_id)
