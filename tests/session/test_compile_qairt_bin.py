"""Tests for compile_qairt_bin.py subprocess script.

These tests verify the ONNX input extraction and error handling logic
without requiring the actual QAIRT SDK.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper


def create_onnx_with_dynamic_batch(output_path: Path) -> Path:
    """Create ONNX model with dynamic batch dimension (dim_value=0)."""
    # Input with dynamic batch (represented as dim_param or dim_value=0)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 4])

    # Output
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 4])

    # Identity node
    node = helper.make_node("Identity", ["X"], ["Y"])

    graph = helper.make_graph([node], "dynamic_batch_model", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return output_path


class TestExtractInputSpecs:
    """Test extract_input_specs function."""

    def test_filters_initializers(self, simple_matmul_onnx: Path):
        """Test that initializers are filtered out from input specs.

        The simple_matmul_onnx model has:
        - Input "A" (1, 4) - should be included
        - Initializer "B" (4, 4) - should NOT be included
        """
        from winml.modelkit.session.qairt.compile_qairt_bin import extract_input_specs

        specs = extract_input_specs(simple_matmul_onnx)

        # Should only have one input (A), not the initializer (B)
        assert len(specs) == 1
        assert specs[0]["name"] == "A"
        assert specs[0]["shape"] == (1, 4)
        assert specs[0]["dtype"] == np.dtype(np.float32)

    def test_replaces_dynamic_dims_with_1(self, tmp_path: Path):
        """Test that dynamic dimensions (<=0) are replaced with 1.

        Key branch: dim.dim_value if dim.dim_value > 0 else 1
        """
        from winml.modelkit.session.qairt.compile_qairt_bin import extract_input_specs

        model_path = create_onnx_with_dynamic_batch(tmp_path / "dynamic.onnx")
        specs = extract_input_specs(model_path)

        assert len(specs) == 1
        assert specs[0]["name"] == "X"
        # Dynamic batch dim should be replaced with 1
        assert specs[0]["shape"] == (1, 4)


class TestMain:
    """Test main() entry point error handling."""

    def test_returns_1_on_compile_error(self, tmp_path: Path, monkeypatch):
        """Test that main() returns 1 when compile_model raises exception.

        Key branch: except Exception -> return 1
        """
        from winml.modelkit.session.qairt import compile_qairt_bin

        # Create a minimal model for the test
        model_path = tmp_path / "model.onnx"
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
        node = helper.make_node("Identity", ["X"], ["Y"])
        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.save(model, str(model_path))

        # Create mock SDK root
        sdk_root = tmp_path / "sdk"
        (sdk_root / "lib" / "python").mkdir(parents=True)
        (sdk_root / "lib" / "x86_64-windows-msvc").mkdir(parents=True)
        (sdk_root / "lib" / "aarch64-windows-msvc").mkdir(parents=True)

        # Mock compile_model to raise an exception
        def mock_compile_model(*args, **kwargs):
            raise RuntimeError("Simulated QAIRT compile error")

        monkeypatch.setattr(compile_qairt_bin, "compile_model", mock_compile_model)

        # Simulate command line args
        test_args = [
            "compile_qairt_bin.py",
            "--qairt-root", str(sdk_root),
            "--model", str(model_path),
            "--output-dir", str(tmp_path / "output"),
        ]
        monkeypatch.setattr(sys, "argv", test_args)

        # Call main and check return code
        result = compile_qairt_bin.main()
        assert result == 1
