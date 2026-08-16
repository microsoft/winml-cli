# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""FP16 conversion utility for ONNX models.

Provides a single entry point for FP32→FP16 model conversion, used by
the quantizer's ``mode="fp16"`` path.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

from google.protobuf.message import EncodeError


if TYPE_CHECKING:
    import onnx

logger = logging.getLogger(__name__)


def convert_to_fp16(
    model: onnx.ModelProto | str | Path,
    *,
    keep_io_types: bool = True,
    op_block_list: list[str] | None = None,
) -> onnx.ModelProto:
    """Convert an ONNX model from FP32 to FP16 precision.

    Uses onnxruntime.transformers.float16.convert_float_to_float16 internally.
    No new dependencies — ORT is already a project dependency.

    Note: ORT's converter mutates the model in-place and returns the same object.

    Args:
        model: Input ONNX ModelProto or path. ModelProto inputs are mutated
            in-place by ORT. Path inputs support large external-data models.
        keep_io_types: If True, preserve FP32 model inputs/outputs by inserting
            Cast nodes at boundaries. Recommended for CPU-safe inference.
        op_block_list: Op types to keep in FP32 (e.g., ["LayerNorm", "Softmax"]).
            When None, ORT uses its DEFAULT_OP_BLOCK_LIST which includes ops
            known to be numerically unsafe in FP16 (e.g., TopK, CumSum, etc.).

    Returns:
        The converted model (same object as input due to ORT in-place mutation).
    """
    import onnx
    from onnx import TensorProto
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model_path = Path(model) if isinstance(model, str | Path) else None
    if model_path is not None:
        inspection_model = onnx.load(str(model_path), load_external_data=False)
    else:
        inspection_model = cast("onnx.ModelProto", model)

    # Skip if model is already FP16 (check floating-point initializer dtypes)
    fp32_types = {TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.BFLOAT16}
    initializers = inspection_model.graph.initializer
    if initializers:
        float_inits = [t for t in initializers if t.data_type in fp32_types | {TensorProto.FLOAT16}]
        if float_inits and all(t.data_type == TensorProto.FLOAT16 for t in float_inits):
            logger.info("Model is already FP16 — skipping conversion.")
            if model_path is not None:
                onnx.load_external_data_for_model(inspection_model, str(model_path.parent))
            return inspection_model

    original_nodes = len(inspection_model.graph.node)

    logger.info("Converting model to FP16...")
    if keep_io_types:
        logger.info("  Keeping I/O types as FP32")
    if op_block_list:
        logger.info("  Keeping ops in FP32: %s", op_block_list)

    if model_path is not None:
        with tempfile.NamedTemporaryFile(
            dir=model_path.parent, suffix=".shape_inferred.onnx", delete=False
        ) as temporary:
            inferred_path = Path(temporary.name)
        try:
            onnx.shape_inference.infer_shapes_path(str(model_path), str(inferred_path))
            inferred_model = onnx.load(str(inferred_path))
        finally:
            inferred_path.unlink(missing_ok=True)
        converted = cast(
            "onnx.ModelProto",
            convert_float_to_float16(
                inferred_model,
                keep_io_types=keep_io_types,
                disable_shape_infer=True,
                op_block_list=op_block_list,
            ),
        )
    else:
        try:
            converted = cast(
                "onnx.ModelProto",
                convert_float_to_float16(
                    model,
                    keep_io_types=keep_io_types,
                    op_block_list=op_block_list,
                ),
            )
        except EncodeError:
            logger.warning(
                "FP16 conversion shape inference could not serialize the model; "
                "retrying with shape inference disabled. This can happen for "
                "large ONNX models that use external data."
            )
            converted = cast(
                "onnx.ModelProto",
                convert_float_to_float16(
                    model,
                    keep_io_types=keep_io_types,
                    disable_shape_infer=True,
                    op_block_list=op_block_list,
                ),
            )

    # ORT's converter appends Cast nodes at the end of the node list (for
    # keep_io_types), which breaks topological ordering. Re-sort the graph
    # using ORT's own topological sort utility.
    if keep_io_types:
        from onnxruntime.transformers.onnx_model import OnnxModel

        OnnxModel.graph_topological_sort(converted.graph)

    converted_nodes = len(converted.graph.node)
    if converted_nodes != original_nodes:
        logger.info("FP16 conversion complete: %d -> %d nodes", original_nodes, converted_nodes)
    else:
        logger.info("FP16 conversion complete: %d nodes", converted_nodes)

    return converted
