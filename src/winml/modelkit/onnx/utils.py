# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import tempfile
from pathlib import Path

import onnx
from onnx.external_data_helper import _get_all_tensors


EXTERNAL_DATA_THRESHOLD = 100 * 1024 * 1024  # 100 MiB


def strip_node_attrs(
    model: onnx.ModelProto,
    op_type: str,
    keep_attrs: frozenset[str] | set[str],
    domain: str = "",
) -> onnx.ModelProto:
    """Remove all attributes from matching nodes except those in *keep_attrs*.

    Useful for stripping default-valued optional attributes that an exporter
    fills in automatically but that are not needed at inference time.

    Modifies *model* **in-place** and also returns it for convenient chaining.

    Args:
        model: ONNX model proto to modify.
        op_type: Operator type string (e.g. ``"GroupQueryAttention"``).
        keep_attrs: Attribute names to retain; every other attribute is removed.
        domain: Operator domain to match (e.g. ``"com.microsoft"``).  The
            empty string matches the default ONNX domain.

    Returns:
        The same *model* object (mutated in-place).
    """
    for node in model.graph.node:
        if node.op_type != op_type or node.domain != domain:
            continue
        to_remove = [a.name for a in node.attribute if a.name not in keep_attrs]
        for name in to_remove:
            for i, a in enumerate(node.attribute):
                if a.name == name:
                    del node.attribute[i]
                    break
    return model


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
) -> None:
    """Same as ``onnx.checker.check_model``, but handles >2GiB models.

    Uses a temp file on disk for large models.
    """
    tmp_dir = None

    if get_model_size(model) >= EXTERNAL_DATA_THRESHOLD:
        try:
            with tempfile.TemporaryDirectory(prefix="winmlcli_compat_") as tmp_dir:
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
