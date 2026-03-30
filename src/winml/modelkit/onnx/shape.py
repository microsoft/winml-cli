# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNX shape inference with metadata preservation.

Provides a reusable shape inference function that:
1. Tries ORT SymbolicShapeInference first (handles com.microsoft ops)
2. Falls back to onnx.shape_inference (handles standard ops)
3. Automatically preserves all node/model metadata through the operation

This is extracted from Optimizer._infer_shapes() so it can be reused by
any pipeline stage (quantize, compile, etc.) without depending on the
full optimizer.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import onnx

from winml.modelkit.onnx.utils import EXTERNAL_DATA_THRESHOLD, get_model_size

from .metadata import capture_metadata, restore_metadata


logger = logging.getLogger(__name__)


def infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Run shape inference with metadata preservation.

    Strategy:
    1. Try ORT SymbolicShapeInference (handles com.microsoft ops like
       QLinearConv, QLinearAdd from quantization)
    2. Fall back to onnx.shape_inference (handles standard ai.onnx ops)
    3. If both fail, return model unchanged with a warning

    Metadata (node metadata_props, winml.* attributes, model metadata_props)
    is automatically captured before and restored after shape inference,
    since these operations may create new ModelProto objects.

    Args:
        model: ONNX ModelProto to infer shapes for.

    Returns:
        Model with shape information propagated. If shape inference fails
        entirely, the original model is returned unchanged.
    """
    snapshot = capture_metadata(model)

    result = _run_inference(model)

    if snapshot.node_count > 0 or snapshot.model_prop_count > 0:
        restore_metadata(result, snapshot)

    return result


def _run_inference(model: onnx.ModelProto) -> onnx.ModelProto:
    """Execute shape inference without metadata handling.

    Tries symbolic first (handles com.microsoft domain ops),
    falls back to ONNX standard inference.
    """
    # Try symbolic first (handles com.microsoft ops from ORT fusion/quantization)
    try:
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

        return SymbolicShapeInference.infer_shapes(
            model,
            int_max=2**31 - 1,
            auto_merge=False,
            guess_output_rank=False,
            verbose=0,
        )
    except Exception as e:
        logger.debug("Symbolic shape inference failed: %s", e)

    # Fallback to ONNX (handles standard ops, fails on com.microsoft)
    try:
        return infer_onnx_shapes(
            model,
            strict_mode=False,
            data_prop=True,
        )
    except Exception as e:
        logger.warning("Shape inference failed: %s", e)
        return model


def infer_onnx_shapes(
    model: onnx.ModelProto,
    check_type: bool = False,
    strict_mode: bool = False,
    data_prop: bool = False,
) -> onnx.ModelProto:
    """Same as ``onnx.shape_inference.infer_shapes``, but handles large models.

    Uses a temp file on disk for models exceeding the protobuf size limit.

    For models exceeding the external-data threshold, a temporary directory is
    used so that both the ``.onnx`` file and its ``.data`` sidecar are cleaned
    up automatically.  ``onnx.save(..., save_as_external_data=True)`` mutates
    the in-memory *model* (strips ``raw_data``, sets ``data_location`` to
    EXTERNAL).  If the subsequent shape-inference or reload step fails, the
    data is re-internalised from the sidecar before the exception propagates,
    so the caller never receives a model with dangling external-data refs.
    """
    if get_model_size(model) >= EXTERNAL_DATA_THRESHOLD:
        with tempfile.TemporaryDirectory(prefix="modelkit_compat_") as tmp_dir:
            tmp_path = str(Path(tmp_dir) / "model.onnx")
            # onnx.save mutates model in-place; restore immediately
            onnx.save(model, tmp_path, save_as_external_data=True)
            onnx.load_external_data_for_model(model, tmp_dir)
            onnx.shape_inference.infer_shapes_path(
                tmp_path, check_type=check_type, strict_mode=strict_mode, data_prop=data_prop
            )
            return onnx.load(tmp_path)

    return onnx.shape_inference.infer_shapes(
        model, check_type=check_type, strict_mode=strict_mode, data_prop=data_prop
    )
