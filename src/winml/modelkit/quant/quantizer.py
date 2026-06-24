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
    *,
    precision: str | None = None,
    **kwargs: Any,
) -> QuantizeResult:
    """Quantize ONNX model with optional multi-pass precision support.

    When ``precision`` is provided (e.g., "w4a16"), the function internally
    expands it into sequential passes (e.g., ["int4", "fp16"]) and runs each
    in order, managing intermediate files automatically. The caller only
    needs to invoke this function once.

    When ``precision`` is None, falls back to single-pass execution based on
    ``config.algorithm``.

    Args:
        model_path: Path to input ONNX model.
        output_path: Path for output model (defaults to {model_stem}_qdq.onnx).
        config: Quantization configuration (uses defaults if None).
        precision: Optional precision string (e.g., "fp16", "int4", "w4a16").
            When set, overrides config.algorithm routing with multi-pass
            expansion logic.

    Returns:
        QuantizeResult with path to final output model and aggregated metrics.

    Examples:
        # Single-pass RTN int4
        result = quantize_onnx("model.onnx", precision="int4", config=rtn_config)

        # Multi-pass w4a16 (int4 + fp16)
        result = quantize_onnx("model.onnx", precision="w4a16", config=rtn_config)

        # Single-pass FP16 only
        result = quantize_onnx("model.onnx", precision="fp16")

        # Legacy: use config.algorithm directly
        result = quantize_onnx("model.onnx", config=WinMLQuantizationConfig(samples=100))
    """
    from ..config.precision import expand_precision

    model_path = Path(model_path)
    config = config or WinMLQuantizationConfig()

    if output_path is not None:
        output_path = Path(output_path)
    else:
        output_path = model_path.parent / f"{model_path.stem}_qdq.onnx"

    # If precision is provided, expand and run multi-pass
    if precision is not None:
        passes = expand_precision(precision)
        return _run_multi_pass(
            model_path=model_path,
            output_path=output_path,
            config=config,
            passes=passes,
            **kwargs,
        )

    # Single-pass: delegate to internal implementation
    return _quantize_single_pass(
        model_path=model_path,
        output_path=output_path,
        config=config,
        **kwargs,
    )


def _run_multi_pass(
    *,
    model_path: Path,
    output_path: Path,
    config: WinMLQuantizationConfig,
    passes: list[str],
    **kwargs: Any,
) -> QuantizeResult:
    """Run a sequence of quantization passes, managing intermediate files.

    Each pass produces a QuantizeResult; the final result aggregates timing
    from all passes.
    """
    current_path = model_path
    total_time = 0.0
    all_warnings: list[str] = []
    intermediates: list[Path] = []

    for step_idx, step_prec in enumerate(passes):
        # Determine output for this step
        if step_idx == len(passes) - 1:
            step_output = output_path
        else:
            step_output = output_path.parent / (
                f"{output_path.stem}_pass{step_idx}{output_path.suffix}"
            )
            intermediates.append(step_output)

        # Build config for this step
        step_config = _make_pass_config(step_prec, config)

        result = _quantize_single_pass(
            model_path=current_path,
            output_path=step_output,
            config=step_config,
            **kwargs,
        )

        if not result.success:
            # Clean up intermediates on failure
            _cleanup_intermediates(intermediates)
            return result

        total_time += result.total_time_seconds
        all_warnings.extend(result.warnings)
        current_path = step_output

    # Clean up intermediate files
    _cleanup_intermediates(intermediates)

    return QuantizeResult(
        success=True,
        output_path=output_path,
        total_time_seconds=total_time,
        nodes_quantized=result.nodes_quantized,
        errors=[],
        warnings=all_warnings,
    )


def _make_pass_config(
    step_prec: str, base_config: WinMLQuantizationConfig
) -> WinMLQuantizationConfig:
    """Build a config for a single pass based on precision string."""
    if step_prec == "fp16":
        return WinMLQuantizationConfig(
            algorithm="fp16",
            fp16_keep_io_types=base_config.fp16_keep_io_types,
            fp16_op_block_list=base_config.fp16_op_block_list,
        )
    # int4, w4a32, or any RTN/QDQ pass — use base config as-is
    return base_config


def _cleanup_intermediates(intermediates: list[Path]) -> None:
    """Remove intermediate pass files and their external data sidecars."""
    for path in intermediates:
        if path.exists():
            path.unlink()
        ext_data = path.parent / f"{path.name}.data"
        if ext_data.exists():
            ext_data.unlink()


def _quantize_single_pass(
    *,
    model_path: Path,
    output_path: Path,
    config: WinMLQuantizationConfig,
    **kwargs: Any,
) -> QuantizeResult:
    """Run a single quantization pass (FP16, RTN, or QDQ).

    This is the internal workhorse — callers should use ``quantize_onnx()``
    which handles multi-pass expansion and path resolution.
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

    use_external_data: bool = kwargs.pop("use_external_data", True)

    start_time = time.perf_counter()

    # Validate input
    if not model_path.exists():
        return QuantizeResult(
            success=False,
            output_path=None,
            errors=[f"Model not found: {model_path}"],
        )

    errors: list[str] = []
    warnings: list[str] = []

    try:
        # ── Pure FP16 fast path (no quantization, only FP16 conversion) ──
        if config.algorithm == "fp16":
            from ..onnx import load_onnx, save_onnx
            from ..optim.fp16 import convert_to_fp16

            if config.calibration_data is not None:
                logger.warning(
                    "calibration_data is set but algorithm='fp16' — "
                    "calibration data will be ignored."
                )

            logger.info("Running FP16-only conversion (no quantization)...")
            model = load_onnx(model_path, validate=False)
            model = convert_to_fp16(
                model,
                keep_io_types=config.fp16_keep_io_types,
                op_block_list=config.fp16_op_block_list,
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

        # ── RTN weight-only path ──────────────────────────────────────
        if config.algorithm == "rtn":
            from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

            from ..onnx import load_onnx, save_onnx

            if config.calibration_data is not None:
                logger.warning(
                    "calibration_data is set but algorithm='rtn' — "
                    "calibration data will be ignored."
                )

            logger.info(
                "Running RTN %d-bit weight-only quantization (block_size=%d, symmetric=%s)...",
                config.rtn_bits,
                config.rtn_block_size,
                config.rtn_symmetric,
            )

            accuracy_level = config.rtn_accuracy_level if config.rtn_accuracy_level != 0 else None

            quantizer = MatMulNBitsQuantizer(
                model=str(model_path),
                bits=config.rtn_bits,
                block_size=config.rtn_block_size,
                is_symmetric=config.rtn_symmetric,
                accuracy_level=accuracy_level,
                nodes_to_exclude=config.nodes_to_exclude,
            )
            quantizer.process()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            # MatMulNBitsQuantizer.model is an ONNXModel wrapper; extract the proto
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

        # Step 2: Load the input model, capture its metadata snapshot (ORT
        # rebuilds the graph during quantization, so we restore afterwards),
        # and tag it as pre-processed so quantize_static() does not emit the
        # "run pre-processing before quantization" warning.  We hand this
        # in-memory ModelProto to ORT directly rather than mutating the user's
        # input file on disk.
        from onnxruntime.quantization.quant_utils import add_pre_process_metadata

        from ..onnx import capture_metadata, load_onnx, restore_metadata, save_onnx
        from .qdq_fix import fix_qdq_dtype_info

        input_model = load_onnx(model_path, validate=False)
        metadata_snapshot = capture_metadata(input_model)
        add_pre_process_metadata(input_model)

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
        # Use an absolute output path so the chdir does not break its
        # resolution.  output_path.parent is guaranteed to exist (caller mkdir).
        abs_model_output = str(Path(output_path).resolve())
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
                model_input=input_model,
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

        postproc_time = time.perf_counter() - postproc_start

        # Count quantized nodes from in-memory model
        from ..compiler import QDQ_OP_TYPES

        nodes_quantized = sum(
            1 for node in quantized_model.graph.node if node.op_type in QDQ_OP_TYPES
        )

        # Step 8: Save the final model
        save_onnx(quantized_model, output_path, use_external_data=use_external_data)

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
