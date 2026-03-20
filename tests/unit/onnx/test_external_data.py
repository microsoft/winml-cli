# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for modelkit.onnx.external_data utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from winml.modelkit.onnx.external_data import (
    copy_onnx_model,
    get_external_data_files,
    has_external_data,
)


if TYPE_CHECKING:
    from pathlib import Path


def _make_small_model() -> onnx.ModelProto:
    """Create a minimal ONNX model (no external data)."""
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
    weight = numpy_helper.from_array(
        np.random.randn(4, 2).astype(np.float32), name="W"
    )
    node = helper.make_node("MatMul", ["X", "W"], ["Y"])
    graph = helper.make_graph([node], "test", [x_info], [y_info], [weight])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class TestGetExternalDataFiles:
    """Tests for get_external_data_files()."""

    def test_no_external_data(self, tmp_path: Path) -> None:
        """Model without external data returns empty list."""
        model = _make_small_model()
        path = tmp_path / "small.onnx"
        onnx.save(model, str(path))

        assert get_external_data_files(path) == []

    def test_with_external_data(self, tmp_path: Path) -> None:
        """Model with external data returns the data filename."""
        model = _make_small_model()
        path = tmp_path / "ext.onnx"
        onnx.save_model(
            model, str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="ext.onnx.data",
            size_threshold=0,
        )

        assert get_external_data_files(path) == ["ext.onnx.data"]


class TestHasExternalData:
    """Tests for has_external_data()."""

    def test_no_external(self, tmp_path: Path) -> None:
        model = _make_small_model()
        path = tmp_path / "small.onnx"
        onnx.save(model, str(path))
        assert has_external_data(path) is False

    def test_with_external(self, tmp_path: Path) -> None:
        model = _make_small_model()
        path = tmp_path / "ext.onnx"
        onnx.save_model(
            model, str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="ext.onnx.data",
            size_threshold=0,
        )
        assert has_external_data(path) is True


class TestCopyOnnxModel:
    """Tests for copy_onnx_model()."""

    def test_copy_no_external_data(self, tmp_path: Path) -> None:
        """Copy model without external data — simple file copy."""
        model = _make_small_model()
        src = tmp_path / "src" / "model.onnx"
        src.parent.mkdir()
        onnx.save(model, str(src))

        dst = tmp_path / "dst" / "model.onnx"
        copy_onnx_model(src, dst)

        assert dst.exists()
        loaded = onnx.load(str(dst))
        assert len(loaded.graph.node) == 1

    def test_copy_with_external_data(self, tmp_path: Path) -> None:
        """Copy model with external data — .onnx + .data both copied."""
        model = _make_small_model()
        src = tmp_path / "src" / "model.onnx"
        src.parent.mkdir()
        onnx.save_model(
            model, str(src),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.onnx.data",
            size_threshold=0,
        )
        assert (src.parent / "model.onnx.data").exists()

        dst = tmp_path / "dst" / "copied.onnx"
        copy_onnx_model(src, dst)

        assert dst.exists()
        assert (dst.parent / "copied.onnx.data").exists()
        # Verify loadable with weights
        loaded = onnx.load(str(dst))
        assert len(loaded.graph.node) == 1
        weight = numpy_helper.to_array(loaded.graph.initializer[0])
        assert weight.shape == (4, 2)

    def test_copy_creates_dst_dir(self, tmp_path: Path) -> None:
        """Destination directory is created if it doesn't exist."""
        model = _make_small_model()
        src = tmp_path / "model.onnx"
        onnx.save(model, str(src))

        dst = tmp_path / "deep" / "nested" / "dir" / "model.onnx"
        copy_onnx_model(src, dst)
        assert dst.exists()

    def test_copy_invalid_file_falls_back(self, tmp_path: Path) -> None:
        """Non-ONNX file falls back to simple copy."""
        src = tmp_path / "fake.onnx"
        src.write_text("not a real onnx file")

        dst = tmp_path / "dst" / "fake.onnx"
        copy_onnx_model(src, dst)

        assert dst.exists()
        assert dst.read_text() == "not a real onnx file"
