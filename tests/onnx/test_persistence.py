# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for modelkit.onnx.persistence module.

Covers load_onnx, save_onnx, and cleanup_onnx with inline and external-data
models, error paths, and edge cases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.external_data_helper import _get_all_tensors, uses_external_data

from winml.modelkit.onnx.persistence import (
    _EXTERNAL_DATA_THRESHOLD,
    cleanup_onnx,
    load_onnx,
    save_onnx,
)


if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_model() -> onnx.ModelProto:
    """Create a minimal valid ONNX model (Relu, no initializers)."""
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Relu", ["X"], ["Y"])
    graph = helper.make_graph([node], "test", [x_info], [y_info])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_model_with_initializer(
    weight_shape: tuple[int, ...] = (512,),
) -> onnx.ModelProto:
    """Create a model with an initializer (Add node with constant weights).

    Default shape (512,) = 2048 bytes, above ONNX's per-tensor
    size_threshold=1024 so it can be externalized when needed.
    """
    rng = np.random.RandomState(0)
    weights = rng.randn(*weight_shape).astype(np.float32)

    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, list(weight_shape))
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(weight_shape))
    w_tensor = numpy_helper.from_array(weights, name="W")
    node = helper.make_node("Add", ["X", "W"], ["Y"])
    graph = helper.make_graph(
        [node], "test_init", [x_info], [y_info], initializer=[w_tensor]
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_model_with_two_initializers() -> onnx.ModelProto:
    """Create a model with two initializers (Add + Mul with separate weights).

    Shapes (512,) = 2048 bytes each, above ONNX per-tensor size_threshold=1024.
    """
    rng = np.random.RandomState(42)
    w1 = rng.randn(512).astype(np.float32)
    w2 = rng.randn(512).astype(np.float32)

    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [512])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [512])
    w1_tensor = numpy_helper.from_array(w1, name="W1")
    w2_tensor = numpy_helper.from_array(w2, name="W2")
    add_node = helper.make_node("Add", ["X", "W1"], ["mid"])
    mul_node = helper.make_node("Mul", ["mid", "W2"], ["Y"])
    graph = helper.make_graph(
        [add_node, mul_node],
        "test_two_init",
        [x_info],
        [y_info],
        initializer=[w1_tensor, w2_tensor],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


# ============================================================================
# load_onnx tests
# ============================================================================


class TestLoadOnnx:
    """Tests for load_onnx."""

    def test_basic_load(self, tmp_path: Path) -> None:
        """Load a saved model and verify graph structure is intact."""
        model = _make_tiny_model()
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        loaded = load_onnx(model_path)

        assert len(loaded.graph.node) == 1
        assert loaded.graph.node[0].op_type == "Relu"
        assert loaded.graph.input[0].name == "X"
        assert loaded.graph.output[0].name == "Y"

    def test_validates_by_default(self, tmp_path: Path) -> None:
        """Validation is on by default; a corrupt file should raise."""
        corrupt_path = tmp_path / "corrupt.onnx"
        corrupt_path.write_bytes(b"NOT_AN_ONNX_MODEL")

        with pytest.raises((onnx.checker.ValidationError, Exception)):
            load_onnx(corrupt_path)

    def test_validate_false_skips_validation(self, tmp_path: Path) -> None:
        """validate=False skips onnx.checker and loads whatever is there."""
        model = _make_tiny_model()
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        loaded = load_onnx(model_path, validate=False)
        assert len(loaded.graph.node) == 1

    def test_load_weights_false(self, tmp_path: Path) -> None:
        """load_weights=False loads graph structure without weight data."""
        model = _make_model_with_initializer()
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        loaded = load_onnx(model_path, load_weights=False, validate=False)

        # Graph structure is intact
        assert len(loaded.graph.node) == 1
        assert loaded.graph.node[0].op_type == "Add"

    def test_file_not_found(self) -> None:
        """FileNotFoundError for a path that does not exist."""
        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            load_onnx("/nonexistent/path/model.onnx")

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        """Accepts pathlib.Path as well as str."""
        model = _make_tiny_model()
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        loaded = load_onnx(model_path)
        assert isinstance(loaded, onnx.ModelProto)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        """Accepts a plain string path."""
        model = _make_tiny_model()
        model_path = tmp_path / "model.onnx"
        onnx.save(model, str(model_path))

        loaded = load_onnx(str(model_path))
        assert isinstance(loaded, onnx.ModelProto)

    def test_load_external_data_model(self, tmp_path: Path) -> None:
        """Loading a model saved with external data resolves the sidecar weights."""
        model = _make_model_with_initializer()
        model_path = tmp_path / "ext_model.onnx"
        save_onnx(model, model_path, threshold_size=0)

        loaded = load_onnx(model_path, validate=False)

        # Weight data should be loaded (non-empty raw_data on initializer)
        assert len(loaded.graph.initializer) == 1
        w = numpy_helper.to_array(loaded.graph.initializer[0])
        assert w.shape == (512,)
        assert w.dtype == np.float32

    def test_validate_external_data_model(self, tmp_path: Path) -> None:
        """Path-based validation works for models with external data sidecar."""
        model = _make_model_with_initializer()
        model_path = tmp_path / "val_ext.onnx"
        save_onnx(model, model_path, threshold_size=0)

        # validate=True (default) should succeed using path-based check
        loaded = load_onnx(model_path)
        assert len(loaded.graph.node) == 1


# ============================================================================
# save_onnx tests
# ============================================================================


class TestSaveOnnx:
    """Tests for save_onnx."""

    def test_small_model_saves_inline(self, tmp_path: Path) -> None:
        """A small model should save inline with no .data sidecar."""
        model = _make_tiny_model()
        model_path = tmp_path / "small.onnx"

        save_onnx(model, model_path)

        assert model_path.exists()
        data_path = tmp_path / "small.onnx.data"
        assert not data_path.exists()

    def test_threshold_zero_forces_external(self, tmp_path: Path) -> None:
        """threshold_size=0 forces external data regardless of model size."""
        model = _make_model_with_initializer()
        model_path = tmp_path / "forced.onnx"

        save_onnx(model, model_path, threshold_size=0)

        assert model_path.exists()
        data_path = tmp_path / "forced.onnx.data"
        assert data_path.exists()

    def test_use_external_data_false_forces_inline(self, tmp_path: Path) -> None:
        """use_external_data=False forces inline, even with threshold_size=0."""
        model = _make_model_with_initializer()
        model_path = tmp_path / "inline.onnx"

        save_onnx(model, model_path, use_external_data=False, threshold_size=0)

        assert model_path.exists()
        data_path = tmp_path / "inline.onnx.data"
        assert not data_path.exists()

    def test_large_model_exceeds_threshold(self, tmp_path: Path) -> None:
        """Model exceeding threshold_size saves with external data."""
        model = _make_model_with_initializer()
        model_path = tmp_path / "large.onnx"

        # Use a tiny threshold so our small model triggers external data
        save_onnx(model, model_path, threshold_size=1)

        assert model_path.exists()
        data_path = tmp_path / "large.onnx.data"
        assert data_path.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Parent directories are created automatically."""
        model = _make_tiny_model()
        model_path = tmp_path / "a" / "b" / "c" / "model.onnx"

        save_onnx(model, model_path)

        assert model_path.exists()

    def test_custom_location(self, tmp_path: Path) -> None:
        """Custom location parameter controls the .data filename."""
        model = _make_model_with_initializer()
        model_path = tmp_path / "custom.onnx"

        save_onnx(model, model_path, threshold_size=0, location="weights.bin")

        assert model_path.exists()
        custom_data = tmp_path / "weights.bin"
        assert custom_data.exists()
        # Default name should NOT exist
        default_data = tmp_path / "custom.onnx.data"
        assert not default_data.exists()

    def test_threshold_negative_forces_external(self, tmp_path: Path) -> None:
        """Negative threshold_size triggers external data (same as zero)."""
        model = _make_model_with_initializer()
        model_path = tmp_path / "neg_thresh.onnx"

        save_onnx(model, model_path, threshold_size=-1)

        assert model_path.exists()
        data_path = tmp_path / "neg_thresh.onnx.data"
        assert data_path.exists()

    def test_model_below_threshold_saves_inline(self, tmp_path: Path) -> None:
        """A model with initializers but below threshold saves inline."""
        model = _make_model_with_initializer()
        model_path = tmp_path / "below.onnx"

        # Default threshold is 100 MiB; our tiny model is well below
        save_onnx(model, model_path)

        assert model_path.exists()
        data_path = tmp_path / "below.onnx.data"
        assert not data_path.exists()

    def test_large_model_via_mocked_bytesize(self, tmp_path: Path) -> None:
        """Model with ByteSize >= threshold triggers external data path.

        Mocks ByteSize to simulate a >2GiB model without actually
        allocating that much memory.
        """
        model = _make_model_with_initializer()
        model_path = tmp_path / "mocked_large.onnx"

        two_gib = 2 * 1024 * 1024 * 1024  # 2 GiB
        with patch.object(type(model), "ByteSize", return_value=two_gib):
            save_onnx(model, model_path)

        assert model_path.exists()
        data_path = tmp_path / "mocked_large.onnx.data"
        assert data_path.exists()

    def test_respects_existing_external_data_markers(self, tmp_path: Path) -> None:
        """When model already has external data markers, always save external.

        When a model is loaded graph-only (load_external_data=False) it retains
        external data markers.  Re-saving with use_external_data=False should
        still preserve those markers because the implementation detects them
        and forces the external-data code path.
        """
        model = _make_model_with_initializer()

        # First save with external data to set markers
        step1_dir = tmp_path / "step1"
        step1_dir.mkdir()
        step1_path = step1_dir / "model.onnx"
        save_onnx(model, step1_path, threshold_size=0)

        # Reload WITHOUT weights to preserve external data markers
        marked_model = onnx.load(str(step1_path), load_external_data=False)

        # Verify it has external markers
        has_markers = any(
            uses_external_data(t) for t in _get_all_tensors(marked_model)
        )
        assert has_markers, "Model should have external data markers after step1"

        # Save again with use_external_data=False; markers should override
        step2_dir = tmp_path / "step2"
        step2_dir.mkdir()
        step2_path = step2_dir / "model.onnx"
        save_onnx(marked_model, step2_path, use_external_data=False)

        assert step2_path.exists()

        # Verify that external data markers were preserved in the re-saved model
        resaved = onnx.load(str(step2_path), load_external_data=False)
        has_markers_after = any(
            uses_external_data(t) for t in _get_all_tensors(resaved)
        )
        assert has_markers_after, (
            "External data markers should be preserved because the model "
            "already had them, overriding use_external_data=False"
        )


# ============================================================================
# cleanup_onnx tests
# ============================================================================


class TestCleanupOnnx:
    """Tests for cleanup_onnx."""

    def test_cleanup_inline_model(self, tmp_path: Path) -> None:
        """Cleaning up an inline model deletes just the .onnx file."""
        model = _make_tiny_model()
        model_path = tmp_path / "inline.onnx"
        onnx.save(model, str(model_path))

        deleted = cleanup_onnx(model_path)

        assert not model_path.exists()
        assert model_path in deleted
        assert len(deleted) == 1

    def test_cleanup_external_data(self, tmp_path: Path) -> None:
        """Cleaning up a model with external data deletes both files."""
        model = _make_model_with_initializer()
        model_path = tmp_path / "ext.onnx"
        save_onnx(model, model_path, threshold_size=0)

        data_path = tmp_path / "ext.onnx.data"
        assert data_path.exists(), "Precondition: data file should exist"

        deleted = cleanup_onnx(model_path)

        assert not model_path.exists()
        assert not data_path.exists()
        assert model_path in deleted
        assert data_path in deleted
        assert len(deleted) == 2

    def test_file_not_found(self) -> None:
        """FileNotFoundError for a path that does not exist."""
        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            cleanup_onnx("/nonexistent/path/model.onnx")

    def test_cleanup_custom_location(self, tmp_path: Path) -> None:
        """Cleanup discovers and deletes custom-named external data files."""
        model = _make_model_with_initializer()
        model_path = tmp_path / "custom_loc.onnx"
        save_onnx(model, model_path, threshold_size=0, location="weights.bin")

        custom_data = tmp_path / "weights.bin"
        assert custom_data.exists(), "Precondition: custom data file should exist"

        deleted = cleanup_onnx(model_path)

        assert not model_path.exists()
        assert not custom_data.exists()
        assert model_path in deleted
        assert custom_data in deleted
        assert len(deleted) == 2

    def test_cleanup_multiple_external_files(self, tmp_path: Path) -> None:
        """Cleanup handles models with tensors pointing to multiple data files.

        Manually crafts external_data entries pointing to separate files
        to simulate a model with scattered external data.
        """
        model = _make_model_with_two_initializers()
        model_path = tmp_path / "multi.onnx"

        # Save with external data first (all tensors in one file)
        save_onnx(model, model_path, threshold_size=0)

        # Now load graph-only and manually rewrite external_data locations
        # to simulate per-tensor files (scattered data)
        graph_only = onnx.load(str(model_path), load_external_data=False)

        # Read the original single data file content to split
        orig_data = (tmp_path / "multi.onnx.data").read_bytes()

        # Clear existing external data entries and reassign per-tensor
        offset = 0
        data_files = {}
        for tensor in _get_all_tensors(graph_only):
            if not uses_external_data(tensor):
                continue
            # Determine tensor byte length from external_data metadata
            tensor_len = 0
            for entry in tensor.external_data:
                if entry.key == "length":
                    tensor_len = int(entry.value)
            if tensor_len == 0:
                # fallback: compute from shape
                dims = [d.dim_value for d in tensor.type.tensor_type.shape.dim]
                tensor_len = int(np.prod(dims)) * 4

            fname = f"{tensor.name}.bin"
            data_files[fname] = orig_data[offset : offset + tensor_len]
            offset += tensor_len

            # Rewrite external_data entries
            del tensor.external_data[:]
            tensor.external_data.add(key="location", value=fname)
            tensor.external_data.add(key="offset", value="0")
            tensor.external_data.add(key="length", value=str(tensor_len))

        # Delete the original single-file data and .onnx, write new artifacts
        (tmp_path / "multi.onnx.data").unlink()
        model_path.unlink()

        # Save the modified graph (no external data save -- just the proto)
        onnx.save_model(graph_only, str(model_path))

        # Write per-tensor data files
        for fname, content in data_files.items():
            (tmp_path / fname).write_bytes(content)

        # Verify preconditions
        for fname in data_files:
            assert (tmp_path / fname).exists()

        # Cleanup should discover and delete all per-tensor files
        deleted = cleanup_onnx(model_path)

        assert not model_path.exists()
        for fname in data_files:
            assert not (tmp_path / fname).exists(), f"{fname} should be deleted"
        # Model + N data files
        assert len(deleted) == 1 + len(data_files)

    def test_missing_data_file_does_not_crash(self, tmp_path: Path) -> None:
        """If the .data file is already gone, cleanup still succeeds."""
        model = _make_model_with_initializer()
        model_path = tmp_path / "partial.onnx"
        save_onnx(model, model_path, threshold_size=0)

        # Manually delete the .data file before cleanup
        data_path = tmp_path / "partial.onnx.data"
        assert data_path.exists()
        data_path.unlink()

        # Should not raise
        deleted = cleanup_onnx(model_path)

        assert not model_path.exists()
        # Only the .onnx file was deleted (data was already gone)
        assert model_path in deleted
        assert data_path not in deleted


# ============================================================================
# Integration / round-trip tests
# ============================================================================


class TestRoundTrip:
    """Integration tests: load -> modify -> save -> verify."""

    def test_load_modify_save_verify_inline(self, tmp_path: Path) -> None:
        """Round-trip: load inline model, add a node, save, reload, verify."""
        model = _make_model_with_initializer()
        orig_path = tmp_path / "original.onnx"
        save_onnx(model, orig_path)

        # Load
        loaded = load_onnx(orig_path, validate=False)
        assert len(loaded.graph.node) == 1

        # Modify: append a Relu after the Add
        relu_node = helper.make_node("Relu", ["Y"], ["Z"])
        loaded.graph.node.append(relu_node)
        z_info = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [512])
        # Replace output
        del loaded.graph.output[:]
        loaded.graph.output.append(z_info)

        # Save modified
        modified_path = tmp_path / "modified.onnx"
        save_onnx(loaded, modified_path)

        # Reload and verify
        reloaded = load_onnx(modified_path, validate=False)
        assert len(reloaded.graph.node) == 2
        assert reloaded.graph.node[0].op_type == "Add"
        assert reloaded.graph.node[1].op_type == "Relu"
        assert reloaded.graph.output[0].name == "Z"

        # Verify weights survived the round trip
        orig_w = numpy_helper.to_array(model.graph.initializer[0])
        reloaded_w = numpy_helper.to_array(reloaded.graph.initializer[0])
        np.testing.assert_array_equal(orig_w, reloaded_w)

    def test_load_modify_save_verify_external(self, tmp_path: Path) -> None:
        """Round-trip with external data: load, modify, save, reload, verify."""
        model = _make_model_with_initializer()
        orig_dir = tmp_path / "orig"
        orig_dir.mkdir()
        orig_path = orig_dir / "model.onnx"
        save_onnx(model, orig_path, threshold_size=0)

        # Load with weights
        loaded = load_onnx(orig_path, validate=False)
        original_weight = numpy_helper.to_array(loaded.graph.initializer[0]).copy()

        # Modify: scale the weights by 2
        w = numpy_helper.to_array(loaded.graph.initializer[0])
        scaled = w * 2.0
        new_tensor = numpy_helper.from_array(scaled, name="W")
        del loaded.graph.initializer[:]
        loaded.graph.initializer.append(new_tensor)

        # Save to new directory with external data
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        out_path = out_dir / "model.onnx"
        save_onnx(loaded, out_path, threshold_size=0)

        # Reload and verify
        reloaded = load_onnx(out_path, validate=False)
        reloaded_w = numpy_helper.to_array(reloaded.graph.initializer[0])
        np.testing.assert_allclose(reloaded_w, original_weight * 2.0)

    def test_save_cleanup_leaves_no_artifacts(self, tmp_path: Path) -> None:
        """Save then cleanup should leave the directory empty."""
        model = _make_model_with_initializer()
        model_path = tmp_path / "ephemeral.onnx"
        save_onnx(model, model_path, threshold_size=0)

        # Directory should have model + data
        files_before = set(tmp_path.iterdir())
        assert len(files_before) == 2

        cleanup_onnx(model_path)

        files_after = set(tmp_path.iterdir())
        assert len(files_after) == 0, f"Leftover files: {files_after}"


# ============================================================================
# Constant verification
# ============================================================================


class TestConstants:
    """Verify module-level constants."""

    def test_threshold_is_100mib(self) -> None:
        assert _EXTERNAL_DATA_THRESHOLD == 100 * 1024 * 1024
