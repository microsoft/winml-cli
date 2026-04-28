# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import tempfile
from pathlib import Path

import onnx
from onnx.external_data_helper import _get_all_tensors


EXTERNAL_DATA_THRESHOLD = 100 * 1024 * 1024  # 100 MiB


def has_unloaded_external_data(model: onnx.ModelProto) -> bool:
    """Return True if the model contains external-data tensors whose bytes are not in memory.

    When a model is loaded with ``load_external_data=False``, tensors keep their
    ``data_location == EXTERNAL`` annotation but ``raw_data`` stays empty.
    The ONNX checker needs the sidecar ``.data`` file on disk to validate those
    tensors, which may not be available in the current working directory.
    """
    return any(
        t.data_location == onnx.TensorProto.EXTERNAL and not t.raw_data
        for t in _get_all_tensors(model)
    )


def get_model_size(model: onnx.ModelProto) -> int:
    """Calculate the total size of an ONNX model in bytes.

    This includes the size of all initializers and any external data tensors.

    Args:
        model: The ONNX model to calculate the size of.

    Returns:
        Total size of the model in bytes.
    """
    # Sum raw tensor data sizes directly to avoid protobuf
    # serialization, which fails for models exceeding ~2 GB.

    return sum(len(t.raw_data) for t in _get_all_tensors(model))


def check_onnx_model(
    model: onnx.ModelProto,
    full_check: bool = False,
    skip_opset_compatibility_check: bool = False,
    check_custom_domain: bool = False,
    skip_if_unloaded_external_data: bool = False,
) -> None:
    """Same as ``onnx.checker.check_model``, but handles >2GiB models.

    Uses a temp file on disk for large models.

    Args:
        skip_if_unloaded_external_data: When True, skip validation if the model
            has tensors with ``data_location == EXTERNAL`` but no ``raw_data``
            (i.e. loaded with ``load_external_data=False``).  The ONNX checker
            needs the sidecar ``.data`` file on disk to validate those tensors,
            which is not available in that case.
    """
    if skip_if_unloaded_external_data and has_unloaded_external_data(model):
        return

    tmp_dir = None

    if get_model_size(model) >= EXTERNAL_DATA_THRESHOLD:
        try:
            with tempfile.TemporaryDirectory(prefix="modelkit_compat_") as tmp_dir:
                tmp_path = str(Path(tmp_dir) / "model.onnx")
                # onnx.save mutates model in-place; restore immediately
                onnx.save(model, tmp_path, save_as_external_data=True)
                onnx.load_external_data_for_model(model, tmp_dir)
                onnx.checker.check_model(
                    tmp_path,
                    full_check,
                    skip_opset_compatibility_check,
                    check_custom_domain,
                )
            return
        finally:
            # Clean up temporary files (always execute)
            if tmp_dir and Path(tmp_dir).exists():
                for file in Path(tmp_dir).glob("*"):
                    file.unlink(missing_ok=True)
                Path(tmp_dir).rmdir()

    onnx.checker.check_model(
        model,
        full_check,
        skip_opset_compatibility_check,
        check_custom_domain,
    )
