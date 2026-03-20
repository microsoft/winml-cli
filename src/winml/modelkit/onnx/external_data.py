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

from onnx import external_data_helper

from .persistence import load_onnx, save_onnx


logger = logging.getLogger(__name__)


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
        src.name, dst.name, len(external_files),
    )


def _copy_single_external(
    src: Path, dst: Path, data_filename: str,
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
