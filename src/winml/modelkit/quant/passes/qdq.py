# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""QDQ (Quantize-Dequantize) calibrated quantization pass."""

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


class QDQPass(BaseQuantPass):
    """QDQ (static/dynamic) calibrated quantization pass.

    Reads all QDQ-relevant fields from
    :class:`~winml.modelkit.quant.config.WinMLQuantizationConfig`:
    ``samples``, ``calibration_method``, ``calibration_data``, ``task``,
    ``model_name``, ``dataset_name``, ``weight_type``, ``activation_type``,
    ``per_channel``, ``symmetric``, ``op_types_to_quantize``,
    ``nodes_to_exclude``.

    Example::

        pass_ = QDQPass(config)
        result = pass_.run("model.onnx", "model_qdq.onnx")
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
        """Apply QDQ calibrated quantization to *model_path*."""
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantType,
            get_qdq_config,
            quantize,
        )

        from ..config import QuantizeResult

        weight_type_map = {
            "uint8": QuantType.QUInt8,
            "int8": QuantType.QInt8,
            "uint16": QuantType.QUInt16,
            "int16": QuantType.QInt16,
        }
        activation_type_map = {
            "uint8": QuantType.QUInt8,
            "int8": QuantType.QInt8,
            "uint16": QuantType.QUInt16,
            "int16": QuantType.QInt16,
        }
        calibration_method_map = {
            "minmax": CalibrationMethod.MinMax,
            "entropy": CalibrationMethod.Entropy,
            "percentile": CalibrationMethod.Percentile,
        }

        start_time = time.perf_counter()
        errors: list[str] = []
        warnings: list[str] = []

        cal_start = time.perf_counter()

        if self._config.calibration_data is not None:
            data_reader = self._config.calibration_data
            logger.info("Using custom calibration data")
        else:
            from ...datasets import DatasetCalibrationReader

            task = self._config.task or "random"
            data_reader = DatasetCalibrationReader(
                model_name=self._config.model_name or "random",
                task=task,
                max_samples=self._config.samples,
                dataset_name=self._config.dataset_name,
                model_path=model_path,
            )
            logger.info(
                "Using calibration: task=%s, samples=%d",
                task,
                self._config.samples,
            )

        cal_time = time.perf_counter() - cal_start

        qdq_start = time.perf_counter()

        weight_type = weight_type_map[self._config.weight_type]
        activation_type = activation_type_map[self._config.activation_type]
        calibrate_method = calibration_method_map[self._config.calibration_method]

        extra_options = {
            "ActivationSymmetric": self._config.symmetric,
            "WeightSymmetric": self._config.symmetric,
        }

        logger.info("Generating QDQ config...")
        qdq_config = get_qdq_config(
            model_input=str(model_path),
            calibration_data_reader=data_reader,
            weight_type=weight_type,
            activation_type=activation_type,
            per_channel=self._config.per_channel,
            calibrate_method=calibrate_method,
            op_types_to_quantize=self._config.op_types_to_quantize,
            nodes_to_exclude=self._config.nodes_to_exclude or [],
            extra_options=extra_options,
        )

        from onnxruntime.quantization.quant_utils import add_pre_process_metadata

        from ...onnx import capture_metadata, load_onnx, restore_metadata, save_onnx
        from ..qdq_fix import fix_qdq_dtype_info

        input_model = load_onnx(model_path, validate=False)
        metadata_snapshot = capture_metadata(input_model)
        add_pre_process_metadata(input_model)

        if use_external_data:
            qdq_config.use_external_data_format = True
        logger.info("Applying quantization...")
        abs_model_output = str(Path(output_path).resolve())
        if output_path.exists():
            output_path.unlink()
        stale_sidecar = output_path.parent / f"{output_path.name}.data"
        if stale_sidecar.exists():
            stale_sidecar.unlink()
        original_cwd = Path.cwd()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chdir(output_path.parent)
            quantize(
                model_input=input_model,
                model_output=abs_model_output,
                quant_config=qdq_config,
            )
        finally:
            os.chdir(original_cwd)

        qdq_time = time.perf_counter() - qdq_start

        postproc_start = time.perf_counter()

        quantized_model = load_onnx(output_path, validate=False)

        logger.info("Fixing QDQ node dtype info...")
        fix_result = fix_qdq_dtype_info(quantized_model)
        warnings.extend(fix_result.warnings)

        from ...onnx import infer_shapes

        logger.info("Running shape inference on quantized model...")
        quantized_model = infer_shapes(quantized_model)

        if metadata_snapshot.node_count > 0:
            logger.info("Restoring metadata from pre-quantization model...")
            restore_metadata(quantized_model, metadata_snapshot)

        postproc_time = time.perf_counter() - postproc_start

        from ...compiler import QDQ_OP_TYPES

        nodes_quantized = sum(
            1 for node in quantized_model.graph.node if node.op_type in QDQ_OP_TYPES
        )

        save_onnx(quantized_model, output_path)

        total_time = time.perf_counter() - start_time

        logger.info(
            "Quantization complete: %s -> %s (%.2fs)",
            model_path.name,
            output_path.name,
            total_time,
        )

        return QuantizeResult(
            success=True,
            output_path=output_path,
            calibration_time_seconds=cal_time,
            qdq_insertion_time_seconds=qdq_time,
            postproc_time_seconds=postproc_time,
            total_time_seconds=total_time,
            nodes_quantized=nodes_quantized,
            errors=errors,
            warnings=warnings,
        )
