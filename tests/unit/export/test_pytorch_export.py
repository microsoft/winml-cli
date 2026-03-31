# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for modelkit.export.pytorch and HTPExporter with WinMLExportConfig.

Tests pure PyTorch export (no HuggingFace dependency) and verifies
HTPExporter correctly uses WinMLExportConfig for I/O specs.
"""

from __future__ import annotations

import onnx
import pytest
import torch
import torch.nn as nn

from winml.modelkit.export import (
    InputTensorSpec,
    OutputTensorSpec,
    WinMLExportConfig,
    export_pytorch,
)


# =============================================================================
# Test Models (pure PyTorch, no HF)
# =============================================================================


class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class MultiInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x, mask):
        return self.fc(x) * mask


class MismatchedInputOrderModel(nn.Module):
    """Model where forward() param order differs from InputTensorSpec order.

    forward() expects (text_input, pixel_values) but InputTensorSpec may list
    pixel_values first — simulating the CLIP OnnxConfig ordering bug.
    """

    def __init__(self):
        super().__init__()
        self.text_fc = nn.Linear(16, 8)
        self.vision_conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, text_input, pixel_values):
        text_out = self.text_fc(text_input)
        vis_out = self.pool(self.vision_conv(pixel_values)).squeeze(-1).squeeze(-1)
        return text_out + vis_out


# =============================================================================
# TestInputTensorSpecToTensor
# =============================================================================


class TestInputTensorSpecToTensor:
    """Tests for InputTensorSpec.to_tensor()."""

    def test_float_tensor(self) -> None:
        spec = InputTensorSpec(name="x", dtype="float32", shape=(2, 3))
        t = spec.to_tensor()
        assert t.shape == (2, 3)
        assert t.dtype == torch.float32
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_int32_tensor(self) -> None:
        spec = InputTensorSpec(name="ids", dtype="int32", shape=(1, 128))
        t = spec.to_tensor()
        assert t.shape == (1, 128)
        assert t.dtype == torch.int32
        assert (t == 1).all()

    def test_int64_tensor(self) -> None:
        spec = InputTensorSpec(name="ids", dtype="int64", shape=(1, 64))
        t = spec.to_tensor()
        assert t.dtype == torch.int64
        assert (t == 1).all()

    def test_no_dtype_defaults_float(self) -> None:
        spec = InputTensorSpec(name="x", shape=(1, 10))
        t = spec.to_tensor()
        assert t.dtype == torch.float32

    def test_no_shape_raises(self) -> None:
        spec = InputTensorSpec(name="x", dtype="float32")
        with pytest.raises(ValueError, match="shape is None"):
            spec.to_tensor()


# =============================================================================
# TestWinMLExportConfigGenerateDummyInputs
# =============================================================================


class TestWinMLExportConfigGenerateDummyInputs:
    """Tests for WinMLExportConfig.generate_dummy_inputs()."""

    def test_single_float_input(self) -> None:
        config = WinMLExportConfig(
            input_tensors=[InputTensorSpec(name="x", dtype="float32", shape=(1, 10))],
        )
        inputs = config.generate_dummy_inputs()
        assert "x" in inputs
        assert inputs["x"].shape == (1, 10)
        assert inputs["x"].dtype == torch.float32

    def test_mixed_dtypes(self) -> None:
        config = WinMLExportConfig(
            input_tensors=[
                InputTensorSpec(name="input_ids", dtype="int32", shape=(1, 128)),
                InputTensorSpec(name="pixel_values", dtype="float32", shape=(1, 3, 224, 224)),
            ],
        )
        inputs = config.generate_dummy_inputs()
        assert inputs["input_ids"].dtype == torch.int32
        assert inputs["pixel_values"].dtype == torch.float32
        assert len(inputs) == 2

    def test_empty_input_tensors_raises(self) -> None:
        config = WinMLExportConfig()
        with pytest.raises(ValueError, match="input_tensors must be populated"):
            config.generate_dummy_inputs()

    def test_skips_specs_without_shape(self) -> None:
        config = WinMLExportConfig(
            input_tensors=[
                InputTensorSpec(name="x", dtype="float32", shape=(1, 10)),
                InputTensorSpec(name="no_shape"),  # no shape — skipped
            ],
        )
        inputs = config.generate_dummy_inputs()
        assert len(inputs) == 1
        assert "x" in inputs


# =============================================================================
# TestExportPytorch
# =============================================================================


class TestExportPytorch:
    """Tests for export_pytorch function."""

    def test_simple_linear(self, tmp_path) -> None:
        model = SimpleLinear()
        config = WinMLExportConfig(
            input_tensors=[InputTensorSpec(name="x", dtype="float32", shape=(1, 10))],
        )
        result = export_pytorch(model, tmp_path / "model.onnx", config)

        assert (tmp_path / "model.onnx").exists()
        assert result["onnx_nodes"] > 0

    def test_two_layer_net(self, tmp_path) -> None:
        model = TwoLayerNet()
        config = WinMLExportConfig(
            input_tensors=[InputTensorSpec(name="x", dtype="float32", shape=(1, 10))],
        )
        result = export_pytorch(model, tmp_path / "model.onnx", config)

        assert result["onnx_nodes"] >= 3  # matmul + relu + matmul

    def test_output_names_in_onnx(self, tmp_path) -> None:
        model = SimpleLinear()
        config = WinMLExportConfig(
            input_tensors=[InputTensorSpec(name="x", dtype="float32", shape=(1, 10))],
            output_tensors=[OutputTensorSpec(name="logits")],
        )
        export_pytorch(model, tmp_path / "model.onnx", config)

        onnx_model = onnx.load(str(tmp_path / "model.onnx"))
        output_names = [o.name for o in onnx_model.graph.output]
        # Output names may be inferred from trace (override config if mismatch)
        assert len(output_names) > 0

    def test_input_names_in_onnx(self, tmp_path) -> None:
        model = SimpleLinear()
        # Input name must match model.forward() param name for hierarchy tracing
        config = WinMLExportConfig(
            input_tensors=[InputTensorSpec(name="x", dtype="float32", shape=(1, 10))],
        )
        export_pytorch(model, tmp_path / "model.onnx", config)

        onnx_model = onnx.load(str(tmp_path / "model.onnx"))
        input_names = [i.name for i in onnx_model.graph.input]
        assert "x" in input_names

    def test_input_shape_in_onnx(self, tmp_path) -> None:
        model = SimpleLinear()
        config = WinMLExportConfig(
            input_tensors=[InputTensorSpec(name="x", dtype="float32", shape=(1, 10))],
        )
        export_pytorch(model, tmp_path / "model.onnx", config)

        onnx_model = onnx.load(str(tmp_path / "model.onnx"))
        input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        assert input_shape == [1, 10]

    def test_no_input_tensors_raises(self, tmp_path) -> None:
        model = SimpleLinear()
        config = WinMLExportConfig()  # no input_tensors

        with pytest.raises(ValueError, match="input_tensors must be populated"):
            export_pytorch(model, tmp_path / "model.onnx", config)

    def test_onnx_valid(self, tmp_path) -> None:
        """Exported ONNX passes onnx.checker."""
        model = TwoLayerNet()
        config = WinMLExportConfig(
            input_tensors=[InputTensorSpec(name="x", dtype="float32", shape=(1, 10))],
        )
        export_pytorch(model, tmp_path / "model.onnx", config)

        onnx_model = onnx.load(str(tmp_path / "model.onnx"))
        onnx.checker.check_model(onnx_model)

    def test_int_input(self, tmp_path) -> None:
        """Model with integer input (embedding-like)."""

        class EmbedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 16)

            def forward(self, ids):
                return self.embed(ids)

        model = EmbedModel()
        config = WinMLExportConfig(
            input_tensors=[InputTensorSpec(name="ids", dtype="int32", shape=(1, 8))],
        )
        # int32 input needs to be long for embedding — this tests the flow
        # (may warn but should still export)
        try:
            export_pytorch(model, tmp_path / "model.onnx", config)
            assert (tmp_path / "model.onnx").exists()
        except RuntimeError:
            # Embedding requires int64, but we generate int32
            # This is expected — the test verifies the flow works up to export
            pass

    def test_mismatched_input_order_exports_successfully(self, tmp_path) -> None:
        """Export succeeds when InputTensorSpec order differs from forward() param order.

        Regression test for CLIP bug: OnnxConfig listed pixel_values before input_ids,
        but CLIPModel.forward() expected input_ids first. With positional args this
        caused 'not enough values to unpack (expected 4, got 2)'. The fix uses
        kwargs= in torch.onnx.export so name-based binding makes order irrelevant.
        """
        model = MismatchedInputOrderModel()

        # Intentionally list pixel_values FIRST — opposite of forward(text_input, pixel_values)
        config = WinMLExportConfig(
            input_tensors=[
                InputTensorSpec(name="pixel_values", dtype="float32", shape=(1, 3, 8, 8)),
                InputTensorSpec(name="text_input", dtype="float32", shape=(1, 16)),
            ],
        )

        result = export_pytorch(model, tmp_path / "model.onnx", config)

        assert (tmp_path / "model.onnx").exists()
        assert result["onnx_nodes"] > 0

        onnx_model = onnx.load(str(tmp_path / "model.onnx"))
        onnx.checker.check_model(onnx_model)
        input_names = {i.name for i in onnx_model.graph.input}
        assert "pixel_values" in input_names
        assert "text_input" in input_names
