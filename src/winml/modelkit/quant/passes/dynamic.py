# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Dynamic quantization pass."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .base import BaseQuantPass


if TYPE_CHECKING:
    from ..config import QuantizeResult, WinMLQuantizationConfig


logger = logging.getLogger(__name__)


# ONNX operator types introduced by dynamic quantization (QOperator format).
# Counting these gives a universal, architecture-agnostic measure of how many
# nodes were quantized — analogous to how the static pass counts QDQ ops. These
# are all standard ai.onnx operators, so this stays model-agnostic (no
# architecture-specific names):
#   - DynamicQuantizeLinear: runtime activation quantization.
#   - MatMulInteger / ConvInteger: integer compute over quantized operands.
#   - DequantizeLinear: restores a statically-quantized weight/embedding back to
#     float when it is consumed by a non-integer op (e.g. an embedding Gather
#     feeding an Add). This is empirically confirmed on embedding models.
# Including DequantizeLinear makes the count a heuristic: a DQL node could in
# principle originate from something other than a quantized-weight restore, so
# ``nodes_quantized`` may slightly over-count. That is acceptable because it is a
# reporting metric only (not correctness-critical), analogous to — if fuzzier
# than — the static pass's QDQ-op count.
# QuantizeLinear is intentionally excluded: quantize_dynamic emits
# DynamicQuantizeLinear (not QuantizeLinear) for activations and stores weights
# as pre-quantized initializers, so a static QuantizeLinear never appears in its
# output.
_DYNAMIC_QUANT_OP_TYPES: frozenset[str] = frozenset(
    {
        "DynamicQuantizeLinear",
        "MatMulInteger",
        "ConvInteger",
        "DequantizeLinear",
    }
)


class DynamicPass(BaseQuantPass):
    """Dynamic quantization pass.

    Statically quantizes weights while activation quantization parameters are
    computed dynamically at inference time, so **no calibration data is
    required**. Uses ONNX Runtime's ``quantize_dynamic`` which emits QOperator
    nodes (``DynamicQuantizeLinear`` / ``MatMulInteger`` / ``ConvInteger``).

    Reads from :class:`~winml.modelkit.quant.config.WinMLQuantizationConfig`:

    - ``weight_type`` — ``uint8`` → ``QUInt8``, ``int8`` → ``QInt8``. Dynamic
      quantization only supports 8-bit weights; 16-bit values fall back to
      ``int8`` with a warning.
    - ``per_channel`` — per-channel weight quantization.
    - ``reduce_range`` — quantize weights with 7 bits to avoid saturation on
      pre-VNNI CPUs.
    - ``weight_symmetric`` / ``symmetric`` — symmetric weight quantization
      (activations are always dynamic/asymmetric).
    - ``op_types_to_quantize`` — op types to quantize (``None`` = ORT default).
    - ``nodes_to_exclude`` — node names to skip.

    Example::

        pass_ = DynamicPass(config)
        result = pass_.run("model.onnx", "model_dynamic.onnx")
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
        """Apply dynamic quantization to *model_path*."""
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from onnxruntime.quantization.quant_utils import add_pre_process_metadata

        from ...onnx import (
            capture_metadata,
            infer_shapes,
            load_onnx,
            restore_metadata,
            save_onnx,
        )
        from ..config import QuantizeResult

        if self._config.calibration_data is not None:
            logger.warning(
                "calibration_data is set but this is a DynamicPass"
                " — calibration data will be ignored."
            )

        start_time = time.perf_counter()
        errors: list[str] = []
        warnings: list[str] = []

        # Dynamic quantization only supports 8-bit weights.
        weight_type_map = {
            "uint8": QuantType.QUInt8,
            "int8": QuantType.QInt8,
        }
        weight_type = weight_type_map.get(self._config.weight_type)
        if weight_type is None:
            msg = (
                "Dynamic quantization supports only 8-bit weights; "
                f"weight_type={self._config.weight_type!r} is unsupported "
                "— falling back to int8."
            )
            logger.warning(msg)
            warnings.append(msg)
            weight_type = QuantType.QInt8

        weight_symmetric = (
            self._config.weight_symmetric
            if self._config.weight_symmetric is not None
            else self._config.symmetric
        )
        extra_options = {"WeightSymmetric": weight_symmetric}

        logger.info(
            "Running dynamic quantization (weight_type=%s, per_channel=%s, reduce_range=%s)...",
            weight_type,
            self._config.per_channel,
            self._config.reduce_range,
        )

        input_model = load_onnx(model_path, validate=False)
        metadata_snapshot = capture_metadata(input_model)
        add_pre_process_metadata(input_model)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        abs_model_output = str(Path(output_path).resolve())
        if output_path.exists():
            output_path.unlink()
        stale_sidecar = output_path.parent / f"{output_path.name}.data"
        if stale_sidecar.exists():
            stale_sidecar.unlink()

        # ORT writes external-data sidecars relative to the cwd, so run from the
        # output directory (mirrors StaticPass) to keep the .data file adjacent.
        original_cwd = Path.cwd()
        try:
            os.chdir(output_path.parent)
            quantize_dynamic(
                model_input=input_model,
                model_output=abs_model_output,
                weight_type=weight_type,
                per_channel=self._config.per_channel,
                reduce_range=self._config.reduce_range,
                op_types_to_quantize=self._config.op_types_to_quantize,
                nodes_to_exclude=self._config.nodes_to_exclude or [],
                use_external_data_format=use_external_data,
                extra_options=extra_options,
            )
        finally:
            os.chdir(original_cwd)

        quantized_model = load_onnx(output_path, validate=False)

        logger.info("Running shape inference on quantized model...")
        quantized_model = infer_shapes(quantized_model)

        if metadata_snapshot.node_count > 0:
            logger.info("Restoring metadata from pre-quantization model...")
            restore_metadata(quantized_model, metadata_snapshot)

        nodes_quantized = sum(
            1 for node in quantized_model.graph.node if node.op_type in _DYNAMIC_QUANT_OP_TYPES
        )

        save_onnx(quantized_model, output_path, use_external_data=use_external_data)

        total_time = time.perf_counter() - start_time
        logger.info(
            "Dynamic quantization complete: %s -> %s (%.2fs)",
            model_path.name,
            output_path.name,
            total_time,
        )
        return QuantizeResult(
            success=True,
            output_path=output_path,
            total_time_seconds=total_time,
            nodes_quantized=nodes_quantized,
            errors=errors,
            warnings=warnings,
        )
