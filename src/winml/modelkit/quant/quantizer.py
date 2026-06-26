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
from typing import TYPE_CHECKING, Any

from .config import QuantizeResult, WinMLQuantizationConfig


if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


def _check_input_model_opset(model_path: Path) -> str | None:
    """Return a clear error message if *model_path* is empty/corrupt, else None.

    Mirrors ORT's ``get_opset_version`` requirement: a usable model must declare
    a default (``""`` / ``ai.onnx``) opset import. A zero-byte or truncated file
    parses into an (almost) empty ModelProto with no such opset import — the
    signature of a previous stage that failed to finish writing (most commonly
    because it ran out of disk space). Detecting it here lets us surface the
    real cause instead of ORT's opaque "Failed to find proper ai.onnx domain".

    Reads only the graph (no external weights) directly via ``onnx.load_model``
    so the check stays cheap and never trips over a missing ``.data`` sidecar.
    """
    from onnx import load_model

    try:
        model = load_model(str(model_path), load_external_data=False)
    except Exception as e:
        return (
            f"Input ONNX model could not be parsed: {model_path} ({e}). "
            "The file may be truncated or corrupt — for example, a previous "
            "build stage may have run out of disk space. Free up disk space "
            "and rebuild."
        )

    has_default_opset = any(opset.domain in ("", "ai.onnx") for opset in model.opset_import)
    if not has_default_opset:
        return (
            f"Input ONNX model is empty or corrupt (no ai.onnx opset import): "
            f"{model_path}. It may have been truncated by a previous failed "
            "write (e.g. insufficient disk space). Free up disk space and rebuild."
        )
    return None


def quantize_onnx(
    model_path: str | Path,
    output_path: str | Path | None = None,
    config: WinMLQuantizationConfig | None = None,
    **kwargs: Any,
) -> QuantizeResult:
    """Quantize an ONNX model using a single quantization pass.

    The quantization mode is driven by ``config.mode``:
    - "fp16": FP16 conversion (no quantization)
    - "rtn": RTN weight-only quantization
    - "static"/"dynamic": QDQ calibrated quantization

    Note: Composite precisions like "w4a16" (requiring multiple sequential
    passes) are not yet supported here — see #964 for the planned
    Quantizer pipeline that will handle multi-pass orchestration.

    Args:
        model_path: Path to input ONNX model.
        output_path: Path for output model (defaults to {model_stem}_qdq.onnx).
        config: Quantization configuration (uses defaults if None).

    Returns:
        QuantizeResult with path to final output model and metrics.

    Examples:
        # Single-pass RTN int4
        result = quantize_onnx("model.onnx", config=WinMLQuantizationConfig(mode="rtn"))

        # Single-pass FP16 only
        result = quantize_onnx("model.onnx", config=WinMLQuantizationConfig(mode="fp16"))

        # QDQ with defaults
        result = quantize_onnx("model.onnx", config=WinMLQuantizationConfig(samples=100))
    """
    model_path = Path(model_path)
    config = config or WinMLQuantizationConfig()

    if output_path is not None:
        output_path = Path(output_path)
    else:
        output_path = model_path.parent / f"{model_path.stem}_qdq.onnx"

    return _quantize_single_pass(
        model_path=model_path,
        output_path=output_path,
        config=config,
        **kwargs,
    )


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
    use_external_data: bool = kwargs.pop("use_external_data", True)

    start_time = time.perf_counter()

    # Validate input
    if not model_path.exists():
        return QuantizeResult(
            success=False,
            output_path=None,
            errors=[f"Model not found: {model_path}"],
        )

    # Guard against an empty/corrupt input model. A previous stage that ran out
    # of disk space can leave a truncated/zero-byte .onnx behind; without this
    # check ORT fails deep inside quantization with the opaque
    # "Failed to find proper ai.onnx domain". Surface the real cause instead.
    opset_error = _check_input_model_opset(model_path)
    if opset_error is not None:
        return QuantizeResult(
            success=False,
            output_path=None,
            errors=[opset_error],
        )

    errors: list[str] = []
    warnings: list[str] = []

    try:
        # Dispatch to the appropriate single-mode handler
        _mode_handlers: dict[str, Callable[..., QuantizeResult]] = {
            "fp16": _quantize_fp16,
            "rtn": _quantize_rtn,
        }
        handler = _mode_handlers.get(config.mode, _quantize_qdq)
        return handler(
            model_path=model_path,
            output_path=output_path,
            config=config,
            start_time=start_time,
            use_external_data=use_external_data,
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


def _quantize_fp16(
    *,
    model_path: Path,
    output_path: Path,
    config: WinMLQuantizationConfig,
    start_time: float,
    use_external_data: bool,
    errors: list[str],
    warnings: list[str],
) -> QuantizeResult:
    """Run FP16 conversion (no quantization)."""
    from ..onnx import load_onnx, save_onnx
    from .fp16 import convert_to_fp16

    if config.calibration_data is not None:
        logger.warning(
            "calibration_data is set but mode='fp16' — calibration data will be ignored."
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


def _quantize_rtn(
    *,
    model_path: Path,
    output_path: Path,
    config: WinMLQuantizationConfig,
    start_time: float,
    use_external_data: bool,
    errors: list[str],
    warnings: list[str],
) -> QuantizeResult:
    """Run RTN weight-only quantization."""
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    from ..onnx import save_onnx

    if config.calibration_data is not None:
        logger.warning("calibration_data is set but mode='rtn' — calibration data will be ignored.")

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


def _quantize_qdq(
    *,
    model_path: Path,
    output_path: Path,
    config: WinMLQuantizationConfig,
    start_time: float,
    use_external_data: bool,
    errors: list[str],
    warnings: list[str],
) -> QuantizeResult:
    """Run QDQ (static/dynamic) calibrated quantization."""
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

    cal_start = time.perf_counter()

    if config.calibration_data is not None:
        data_reader = config.calibration_data
        logger.info("Using custom calibration data")
    else:
        from ..datasets import DatasetCalibrationReader

        task = config.task or "random"
        data_reader = DatasetCalibrationReader(
            model_name=config.model_id or "random",
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

    qdq_start = time.perf_counter()

    weight_type = weight_type_map[config.weight_type]
    activation_type = activation_type_map[config.activation_type]
    calibrate_method = calibration_method_map[config.calibration_method]

    extra_options = {
        "ActivationSymmetric": config.symmetric,
        "WeightSymmetric": config.symmetric,
    }

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

    # Load the input model, capture its metadata snapshot (ORT rebuilds the
    # graph during quantization, so we restore afterwards), and tag it as
    # pre-processed so quantize_static() does not emit a warning.
    from onnxruntime.quantization.quant_utils import add_pre_process_metadata

    from ..onnx import capture_metadata, load_onnx, restore_metadata, save_onnx
    from .qdq_fix import fix_qdq_dtype_info

    input_model = load_onnx(model_path, validate=False)
    metadata_snapshot = capture_metadata(input_model)
    add_pre_process_metadata(input_model)

    if use_external_data:
        qdq_config.use_external_data_format = True
    logger.info("Applying quantization...")
    # Temporarily change CWD to output directory so ORT's save_model_to_file()
    # resolves its CWD-relative os.path.exists() check correctly.
    abs_model_output = str(Path(output_path).resolve())
    # Remove stale output artifacts from a previous build
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

    quantized_model = load_onnx(output_path, validate=False)

    logger.info("Fixing QDQ node dtype info...")
    fix_result = fix_qdq_dtype_info(quantized_model)
    warnings.extend(fix_result.warnings)

    from ..onnx import infer_shapes

    logger.info("Running shape inference on quantized model...")
    quantized_model = infer_shapes(quantized_model)

    if metadata_snapshot.node_count > 0:
        logger.info("Restoring metadata from pre-quantization model...")
        restore_metadata(quantized_model, metadata_snapshot)

    postproc_time = time.perf_counter() - postproc_start

    from ..compiler import QDQ_OP_TYPES

    nodes_quantized = sum(1 for node in quantized_model.graph.node if node.op_type in QDQ_OP_TYPES)

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
