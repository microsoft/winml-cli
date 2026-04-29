# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ONNX external data utilities.

Handles copying and managing ONNX models that use external data files
(models > 2GB where weights are stored in separate .data sidecar files).

Based on patterns from Microsoft Olive (olive/passes/onnx/common.py).

Example:
    >>> from winml.modelkit.onnx.external_data import copy_onnx_model
    >>> copy_onnx_model("src/model.onnx", "dst/model.onnx")
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import external_data_helper

from .persistence import load_onnx, save_onnx


logger = logging.getLogger(__name__)


# Pattern matching and runtime-query helpers later materialize numpy arrays.
# Keep external initializer loading bounded until those pipelines stay lazy.
MAX_EXTERNAL_INITIALIZER_BYTES_FOR_QUERY = 1024 * 1024


def _get_external_tensor_info(tensor: onnx.TensorProto) -> tuple[str | None, int, int | None]:
    """Extract location, offset, and length metadata for an external tensor."""
    info = {entry.key: entry.value for entry in tensor.external_data}
    location = info.get("location")
    offset = int(info.get("offset", "0"))
    length = int(info["length"]) if "length" in info else None
    return location, offset, length


def _tensor_proto_dtype_to_np_dtype(tensor_type: int) -> np.dtype[Any]:
    """Convert a TensorProto dtype enum to a numpy dtype."""
    try:
        from onnx.helper import tensor_dtype_to_np_dtype as onnx_tensor_dtype_to_np_dtype
    except ImportError:
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

        return np.dtype(TENSOR_TYPE_TO_NP_TYPE[tensor_type])

    return np.dtype(onnx_tensor_dtype_to_np_dtype(tensor_type))


def try_load_external_initializer_array(
    tensor: onnx.TensorProto,
    model_path: str | Path | None,
) -> np.ndarray | None:
    """Load a small external initializer from disk.

    Returns None when the model path is unavailable, metadata is incomplete,
    the sidecar file is missing, or the tensor is too large for the current
    value-materialization pipeline.
    """
    if tensor.data_location != onnx.TensorProto.EXTERNAL or model_path is None:
        return None

    location, offset, length = _get_external_tensor_info(tensor)
    if location is None:
        return None

    model_path = Path(model_path)
    data_path = Path(location)
    if not data_path.is_absolute():
        data_path = model_path.parent / data_path
    if not data_path.exists():
        return None

    try:
        np_dtype = _tensor_proto_dtype_to_np_dtype(tensor.data_type)
        shape = tuple(tensor.dims)
        numel = int(np.prod(shape)) if shape else 1
        expected_bytes = numel * np_dtype.itemsize
        if length is not None and length < expected_bytes:
            logger.debug(
                "External initializer %s length %s is smaller than expected %s",
                tensor.name,
                length,
                expected_bytes,
            )
            return None
        if expected_bytes > MAX_EXTERNAL_INITIALIZER_BYTES_FOR_QUERY:
            logger.debug(
                "Skipping external initializer %s because %s bytes exceeds %s",
                tensor.name,
                expected_bytes,
                MAX_EXTERNAL_INITIALIZER_BYTES_FOR_QUERY,
            )
            return None

        with data_path.open("rb") as f:
            f.seek(offset)
            arr = np.fromfile(f, dtype=np_dtype, count=numel)

        if arr.size != numel:
            logger.debug(
                "External initializer %s only yielded %s elements, expected %s",
                tensor.name,
                arr.size,
                numel,
            )
            return None

        return arr.reshape(shape)
    except Exception as e:
        logger.debug(
            "Failed to read external initializer %s from %s: %s",
            tensor.name,
            data_path,
            e,
        )
        return None


def get_external_data_files(model_path: str | Path) -> list[str]:
    """Get list of external data filenames referenced by an ONNX model.

    Loads only the graph structure (no weights) for speed.

    Args:
        model_path: Path to the ONNX model file.

    Returns:
        List of unique external data filenames (relative to model dir).
        Empty list if model has no external data.
    """
    file_names: set[str] = set()
    model = load_onnx(model_path, load_weights=False, validate=False)
    for tensor in external_data_helper._get_all_tensors(model):
        if external_data_helper.uses_external_data(tensor):
            file_names.add(
                external_data_helper.ExternalDataInfo(tensor).location,
            )
    return sorted(file_names)


def has_external_data(model_path: str | Path) -> bool:
    """Check if an ONNX model uses external data files."""
    return len(get_external_data_files(model_path)) > 0


def copy_onnx_model(
    src: str | Path,
    dst: str | Path,
) -> None:
    """Copy an ONNX model and all its external data files.

    Handles three cases:
    1. No external data: simple file copy.
    2. Single external data file: copy .data file + patch .onnx location field.
    3. Multiple external data files: load full model and re-save consolidated.

    Args:
        src: Source ONNX model path.
        dst: Destination ONNX model path.
    """
    src = Path(src).resolve()
    dst = Path(dst).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        external_files = get_external_data_files(src)
    except Exception:
        # Not a valid ONNX file or can't parse — fall back to simple copy
        shutil.copy2(src, dst)
        return

    if not external_files:
        # No external data — simple copy
        shutil.copy2(src, dst)
        return

    if len(external_files) == 1:
        # Single external data file — copy .data + patch .onnx
        _copy_single_external(src, dst, external_files[0])
    else:
        # Multiple files — consolidate into one
        _copy_consolidate(src, dst)

    logger.debug(
        "Copied ONNX model with external data: %s -> %s (%d data files)",
        src.name,
        dst.name,
        len(external_files),
    )


def _copy_single_external(
    src: Path,
    dst: Path,
    data_filename: str,
) -> None:
    """Copy model with a single external data file efficiently.

    Copies the .data file directly (no memory load of weights),
    then patches the .onnx protobuf to point to the new data filename.
    """
    src_data = src.parent / data_filename
    dst_data_name = f"{dst.name}.data"
    dst_data = dst.parent / dst_data_name

    # Copy the data file (no weight loading)
    shutil.copy2(src_data, dst_data)

    # Load .onnx structure only, patch location, save
    model = load_onnx(src, load_weights=False, validate=False)
    for tensor in external_data_helper._get_all_tensors(model):
        if external_data_helper.uses_external_data(tensor):
            info = external_data_helper.ExternalDataInfo(tensor)
            # set_external_data requires raw_data field to exist (Olive pattern)
            tensor.raw_data = b""
            external_data_helper.set_external_data(
                tensor,
                location=dst_data_name,
                offset=info.offset,
                length=info.length,
            )
            tensor.ClearField("raw_data")
    # Write patched proto as-is. Cannot use save_onnx here because its
    # has_existing_external check would force re-externalization on a
    # graph-only model (no loaded weights), corrupting the output.
    import onnx

    onnx.save_model(model, str(dst))


def _copy_consolidate(src: Path, dst: Path) -> None:
    """Copy model with multiple external data files by consolidating.

    Loads the full model into memory and re-saves with a single
    external data file.
    """
    logger.info("Consolidating %s external data files", src.name)
    model = load_onnx(src, validate=False)  # loads all external data
    save_onnx(model, dst, threshold_size=0, location=f"{dst.name}.data")
