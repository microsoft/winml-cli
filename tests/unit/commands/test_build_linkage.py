# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

from winml.modelkit.commands.build import _write_model_linkage_file


if TYPE_CHECKING:
    from pathlib import Path


def test_write_model_linkage_file_collects_linked_files(tmp_path: Path) -> None:
    final_model = tmp_path / "model.onnx"
    final_model.write_bytes(b"onnx")

    ep_info = {
        "ep_contexts": [
            {"cache_context_path": "quantized_qnn_ctx_qnn.bin"},
            {"cache_context_path": "model.onnx.data"},
        ]
    }

    with (
        patch(
            "winml.modelkit.onnx.external_data.get_external_data_files",
            return_value=["model.onnx.data"],
        ),
        patch("winml.modelkit.core.onnx_utils.get_epcontext_info", return_value=ep_info),
    ):
        linkage_path = _write_model_linkage_file(final_model)

    assert linkage_path == tmp_path / "model_linkage.json"
    data = json.loads(linkage_path.read_text())
    assert data["schema_version"] == 1
    assert data["model"] == "model.onnx"
    assert data["linked_files"] == ["model.onnx.data", "quantized_qnn_ctx_qnn.bin"]


def test_write_model_linkage_file_skips_missing_model(tmp_path: Path) -> None:
    linkage_path = _write_model_linkage_file(tmp_path / "model.onnx")
    assert linkage_path is None
    assert not (tmp_path / "model_linkage.json").exists()
