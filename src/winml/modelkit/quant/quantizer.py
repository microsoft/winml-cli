# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Main quantizer implementation."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from .config import QuantizeResult, WinMLQuantizationConfig


logger = logging.getLogger(__name__)


def quantize_onnx(
    model_path: str | Path,
    output_path: str | Path | None = None,
    config: WinMLQuantizationConfig | None = None,
    **kwargs: Any,
) -> QuantizeResult:
    """Quantize ONNX model by inserting QDQ nodes.

    Args:
        model_path: Path to input float32 ONNX model
        output_path: Path for output quantized model (defaults to {model_stem}_qdq.onnx)
        config: Quantization configuration (uses defaults if None)

    Returns:
        QuantizeResult with path to quantized model and metrics

    Examples:
        # Quick quantize with defaults (10 samples, uint8)
        result = quantize_onnx("model.onnx")

        # Quantize with explicit output path
        result = quantize_onnx("model.onnx", "model_quantized.onnx")

        # Quantize with custom config
        result = quantize_onnx("model.onnx", config=WinMLQuantizationConfig(samples=100))
    """
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantType,
        get_qdq_config,
        quantize,
    )

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

    # TODO: Move to global env config
    use_external_data: bool = kwargs.pop("use_external_data", True)

    start_time = time.perf_counter()
    model_path = Path(model_path)
    config = config or WinMLQuantizationConfig()

    # Validate input
    if not model_path.exists():
        return QuantizeResult(
            success=False,
            output_path=None,
            errors=[f"Model not found: {model_path}"],
        )

    # Determine output path
    if output_path is not None:
        output_path = Path(output_path)
    else:
        output_path = model_path.parent / f"{model_path.stem}_qdq.onnx"

    errors: list[str] = []
    warnings: list[str] = []

    try:
        # Create calibration data reader
        cal_start = time.perf_counter()

        if config.calibration_data is not None:
            # User provided explicit calibration data
            data_reader = config.calibration_data
            logger.info("Using custom calibration data")
        else:
            # Use DatasetCalibrationReader for all cases:
            # - task-aware: auto-selects TextDataset, ImageDataset, etc.
            # - fallback: unsupported tasks → RandomDataset (reads ONNX metadata)
            # - no task: task="random" → RandomDataset directly
            from ..datasets import DatasetCalibrationReader

            task = config.task or "random"
            data_reader = DatasetCalibrationReader(
                model_name=config.model_name or "random",
                task=task,
                max_samples=config.samples,
                dataset_name=config.dataset_name,
                model_path=model_path,
            )
            logger.info(
                "Using calibration: task=%s, samples=%d",
                task,
                config.samples,
            )

        cal_time = time.perf_counter() - cal_start

        # Apply QDQ quantization
        qdq_start = time.perf_counter()

        # Map config to ORT types
        weight_type = weight_type_map[config.weight_type]
        activation_type = activation_type_map[config.activation_type]
        calibrate_method = calibration_method_map[config.calibration_method]

        # Build extra options
        extra_options = {
            "ActivationSymmetric": config.symmetric,
            "WeightSymmetric": config.symmetric,
        }

        # Step 1: Generate QDQ config
        logger.info("Generating QDQ config...")
        qdq_config = get_qdq_config(
            model_input=str(model_path),
            calibration_data_reader=data_reader,
            weight_type=weight_type,
            activation_type=activation_type,
            per_channel=config.per_channel,
            calibrate_method=calibrate_method,
            op_types_to_quantize=config.op_types_to_quantize,
            nodes_to_exclude=config.nodes_to_exclude or [],
            extra_options=extra_options,
        )

        # Step 2: Capture metadata before ORT quantization (it rebuilds the graph)
        from ..onnx import capture_metadata, load_onnx, restore_metadata, save_onnx
        from .qdq_fix import fix_qdq_dtype_info

        pre_quant_model = load_onnx(model_path, load_weights=False, validate=False)
        metadata_snapshot = capture_metadata(pre_quant_model)
        del pre_quant_model

        # Step 3: Apply quantization
        if use_external_data:
            qdq_config.use_external_data_format = True
        logger.info("Applying quantization...")
        # Temporarily change CWD to the output directory so that ORT's
        # save_model_to_file() — which passes a bare filename
        # (e.g. "quantized.onnx.data") to onnx.convert_model_to_external_data —
        # resolves its CWD-relative os.path.exists() check against the actual
        # output directory rather than the process CWD.  Without this, a stale
        # .onnx.data sidecar in the process CWD from a previous build triggers
        # a false-positive FileExistsError even when the output dir is clean.
        # Use absolute paths so the chdir does not break relative input/output
        # resolution.  output_path.parent is guaranteed to exist (caller mkdir).
        abs_model_input = str(Path(model_path).resolve())
        abs_model_output = str(Path(output_path).resolve())
        # Ensure output parent exists; subsequent os.chdir requires it.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove stale output artifacts from a previous build.  ORT/onnx refuse
        # to overwrite an existing external-data sidecar (e.g. quantized.onnx.data),
        # raising FileExistsError, so we proactively clear them here.
        if output_path.exists():
            output_path.unlink()
        stale_sidecar = output_path.parent / f"{output_path.name}.data"
        if stale_sidecar.exists():
            stale_sidecar.unlink()
        original_cwd = Path.cwd()
        try:
            os.chdir(output_path.parent)
            quantize(
                model_input=abs_model_input,
                model_output=abs_model_output,
                quant_config=qdq_config,
            )
        finally:
            os.chdir(original_cwd)

        qdq_time = time.perf_counter() - qdq_start

        # Post-processing: fix QDQ dtype + shape inference + restore metadata
        postproc_start = time.perf_counter()

        # Step 4: Load quantized model for post-processing
        quantized_model = load_onnx(output_path, validate=False)

        # Step 5: Fix QDQ node dtype info (scale/zero_point may have UNDEFINED types)
        logger.info("Fixing QDQ node dtype info...")
        fix_result = fix_qdq_dtype_info(quantized_model)
        warnings.extend(fix_result.warnings)

        # Step 6: Run shape inference (defensive — propagates shapes through QDQ nodes)
        # Uses the shared infer_shapes which tries symbolic first (handles
        # com.microsoft ops like QLinearConv) then falls back to ONNX standard.
        # Does NOT run graph optimization pipes that could break quantized models.
        from ..onnx import infer_shapes

        logger.info("Running shape inference on quantized model...")
        quantized_model = infer_shapes(quantized_model)

        # Step 7: Restore metadata lost during ORT quantization
        if metadata_snapshot.node_count > 0:
            logger.info("Restoring metadata from pre-quantization model...")
            restore_metadata(quantized_model, metadata_snapshot)

        # Step 8: Save the fixed model back
        save_onnx(quantized_model, output_path)

        postproc_time = time.perf_counter() - postproc_start

        # Count quantized nodes from in-memory model
        from ..compiler import QDQ_OP_TYPES

        nodes_quantized = sum(
            1 for node in quantized_model.graph.node if node.op_type in QDQ_OP_TYPES
        )

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

    except Exception:
        total_time = time.perf_counter() - start_time
        logger.exception("Quantization failed")

        import traceback

        return QuantizeResult(
            success=False,
            output_path=None,
            total_time_seconds=total_time,
            errors=[traceback.format_exc()],
            warnings=warnings,
        )
