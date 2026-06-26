# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for modelkit.onnx.external_data utilities."""

from __future__ import annotations

import errno
import shutil
from typing import TYPE_CHECKING

import numpy as np
import onnx
import pytest
from onnx import TensorProto, external_data_helper, helper, numpy_helper

from winml.modelkit.onnx import ONNXSaveError
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
    weight = numpy_helper.from_array(np.random.randn(4, 2).astype(np.float32), name="W")
    node = helper.make_node("MatMul", ["X", "W"], ["Y"])
    graph = helper.make_graph([node], "test", [x_info], [y_info], [weight])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_filled_model(value: float, shape: tuple[int, ...]) -> onnx.ModelProto:
    """Create a deterministic ONNX model with a constant-filled initializer.

    Used by overwrite tests where two distinguishable models are needed.
    """
    weight = numpy_helper.from_array(np.full(shape, value, dtype=np.float32), name="W")
    inp = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(shape))
    out = helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(shape))
    node = helper.make_node("Add", ["X", "W"], ["Y"])
    graph = helper.make_graph([node], "g", [inp], [out], [weight])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _serialize_without_external_location(model: onnx.ModelProto) -> bytes:
    """Serialize the model with the `location` entry stripped from every
    external_data tensor — for comparing two models that point to different
    sidecar filenames but are otherwise identical."""
    clone = onnx.ModelProto()
    clone.CopyFrom(model)
    for tensor in external_data_helper._get_all_tensors(clone):
        if tensor.data_location == TensorProto.EXTERNAL:
            for entry in list(tensor.external_data):
                if entry.key == "location":
                    tensor.external_data.remove(entry)
    return clone.SerializeToString(deterministic=True)


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
            model,
            str(path),
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
            model,
            str(path),
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
            model,
            str(src),
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

    def test_copy_overwrites_existing_dst_no_external_data(self, tmp_path: Path) -> None:
        """Pre-existing dst (no external data) is overwritten byte-for-byte by src."""
        src = tmp_path / "src.onnx"
        dst = tmp_path / "dst.onnx"

        onnx.save(_make_filled_model(1.0, (4, 4)), str(src))
        onnx.save(_make_filled_model(99.0, (8, 8)), str(dst))  # pre-existing, different

        pre_dst_bytes = dst.read_bytes()
        src_bytes = src.read_bytes()
        assert pre_dst_bytes != src_bytes

        copy_onnx_model(src, dst)

        post_dst_bytes = dst.read_bytes()
        assert post_dst_bytes == src_bytes
        assert post_dst_bytes != pre_dst_bytes
        assert not (tmp_path / "dst.onnx.data").exists()

    def test_copy_overwrites_existing_dst_with_external_data(self, tmp_path: Path) -> None:
        """Pre-existing dst + sidecar (external data) are both overwritten.

        Verifies:
        - dst.onnx.data is byte-identical to src.onnx.data
        - dst.onnx matches src.onnx except for the external_data.location field
        - dst.onnx's location field points at dst.onnx.data
        - Loaded initializer arrays are equal
        """
        src = tmp_path / "src.onnx"
        dst = tmp_path / "dst.onnx"
        src_data = tmp_path / "src.onnx.data"
        dst_data = tmp_path / "dst.onnx.data"

        onnx.save_model(
            _make_filled_model(2.0, (64, 64)),
            str(src),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="src.onnx.data",
            size_threshold=0,
        )
        onnx.save_model(
            _make_filled_model(999.0, (32, 32)),
            str(dst),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="dst.onnx.data",
            size_threshold=0,
        )

        src_data_bytes = src_data.read_bytes()
        pre_dst_data_bytes = dst_data.read_bytes()
        pre_dst_onnx_bytes = dst.read_bytes()
        assert src_data_bytes != pre_dst_data_bytes

        copy_onnx_model(src, dst)

        # .data file byte-identical to src's sidecar
        post_dst_data_bytes = dst_data.read_bytes()
        assert post_dst_data_bytes == src_data_bytes
        assert post_dst_data_bytes != pre_dst_data_bytes

        # .onnx file no longer matches old dst
        assert dst.read_bytes() != pre_dst_onnx_bytes

        # .onnx matches src modulo external_data.location field
        src_model = onnx.load(str(src), load_external_data=False)
        dst_model = onnx.load(str(dst), load_external_data=False)
        assert _serialize_without_external_location(
            src_model
        ) == _serialize_without_external_location(dst_model)

        # dst.onnx's location must point at dst.onnx.data
        for tensor in external_data_helper._get_all_tensors(dst_model):
            if tensor.data_location == TensorProto.EXTERNAL:
                info = external_data_helper.ExternalDataInfo(tensor)
                assert info.location == "dst.onnx.data"

        # Semantic check: loaded initializer arrays are equal
        src_full = onnx.load(str(src), load_external_data=True)
        dst_full = onnx.load(str(dst), load_external_data=True)
        src_arr = numpy_helper.to_array(src_full.graph.initializer[0])
        dst_arr = numpy_helper.to_array(dst_full.graph.initializer[0])
        assert np.array_equal(src_arr, dst_arr)


class TestCopyOnnxModelDiskFull:
    """copy_onnx_model surfaces a clear error and cleans up on a failed write."""

    def test_copy_disk_full_raises_and_cleans_dst(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = tmp_path / "src.onnx"
        dst = tmp_path / "out" / "dst.onnx"
        onnx.save(_make_small_model(), str(src))  # valid, no external data

        def _failing_copy2(_s: object, d: object, *_a: object, **_k: object) -> None:
            from pathlib import Path as _Path

            _Path(d).write_bytes(b"")  # partial/truncated destination
            raise OSError(errno.ENOSPC, "simulated write failure")

        monkeypatch.setattr(shutil, "copy2", _failing_copy2)

        with pytest.raises(ONNXSaveError) as exc_info:
            copy_onnx_model(src, dst)

        err = exc_info.value
        assert err.disk_full is True
        assert isinstance(err, OSError)
        assert "disk space" in str(err).lower()
        # The truncated destination must not be left behind.
        assert not dst.exists()
