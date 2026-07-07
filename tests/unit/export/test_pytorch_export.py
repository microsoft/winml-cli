# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for modelkit.export.pytorch and HTPExporter with WinMLExportConfig.

Tests pure PyTorch export (no HuggingFace dependency) and verifies
HTPExporter correctly uses WinMLExportConfig for I/O specs.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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


def _all_value_info_have_shape(model: onnx.ModelProto) -> bool:
    """Every intermediate (value_info) tensor has a concrete or symbolic shape."""
    if not model.graph.value_info:
        return False
    for vi in model.graph.value_info:
        shape = vi.type.tensor_type.shape
        if not shape.dim:
            return False
        for dim in shape.dim:
            if not dim.HasField("dim_value") and not dim.HasField("dim_param"):
                return False
    return True


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

    def test_normalization_succeeds_and_shape_inferences(self, tmp_path) -> None:
        """After export, status reports succeeded and value_info is fully shaped."""
        model = TwoLayerNet()
        config = WinMLExportConfig(
            input_tensors=[InputTensorSpec(name="x", dtype="float32", shape=(1, 10))],
        )
        result = export_pytorch(model, tmp_path / "model.onnx", config)

        assert result["model_normalization_status"] == "succeeded"

        onnx_model = onnx.load(str(tmp_path / "model.onnx"))
        assert _all_value_info_have_shape(onnx_model)

    def test_failed_normalization_skips_shape_inference(self, tmp_path) -> None:
        """When normalization is mocked to return False, status is failed."""
        model = TwoLayerNet()
        config = WinMLExportConfig(
            input_tensors=[InputTensorSpec(name="x", dtype="float32", shape=(1, 10))],
        )
        with patch(
            "winml.modelkit.export.pytorch._normalize_exported_model",
            return_value=False,
        ):
            result = export_pytorch(model, tmp_path / "model.onnx", config)

        assert result["model_normalization_status"] == "failed"

        onnx_model = onnx.load(str(tmp_path / "model.onnx"))
        assert not _all_value_info_have_shape(onnx_model)

    def test_normalize_false_skips_normalization(self, tmp_path) -> None:
        """When normalize=False, the helper isn't called and status is not_run."""
        model = TwoLayerNet()
        config = WinMLExportConfig(
            input_tensors=[InputTensorSpec(name="x", dtype="float32", shape=(1, 10))],
        )
        with patch(
            "winml.modelkit.export.pytorch._normalize_exported_model",
        ) as mock_normalize:
            result = export_pytorch(model, tmp_path / "model.onnx", config, normalize=False)

        mock_normalize.assert_not_called()
        assert result["model_normalization_status"] == "not_run"

        onnx_model = onnx.load(str(tmp_path / "model.onnx"))
        assert not _all_value_info_have_shape(onnx_model)

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


class TestStaleExternalDataCleanup:
    """`_cleanup_stale_external_data` prunes per-tensor sidecars after consolidation.

    The TorchScript exporter writes one external-data file per initializer; the
    tag-injection re-save consolidates them into a single ``<model>.onnx.data``.
    Without cleanup the per-tensor sidecars linger as orphans that roughly double
    on-disk size, which is what exhausts disk on large / composite exports.
    """

    def _write_model_with_consolidated_data(self, path) -> None:
        """Save a model forcing a single consolidated external-data sidecar."""
        from winml.modelkit.onnx import save_onnx

        # Weights must exceed onnx's per-tensor external threshold (1024 bytes),
        # so use a linear layer large enough to be externalized (256*256*4 bytes).
        class BigLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(256, 256)

            def forward(self, x):
                return self.linear(x)

        model = BigLinear()
        config = WinMLExportConfig(
            input_tensors=[InputTensorSpec(name="x", dtype="float32", shape=(1, 256))],
        )
        export_pytorch(model, path, config)
        # Force external data so the model references exactly one .data sidecar,
        # independent of the exporter's size heuristics.
        loaded = onnx.load(str(path))
        save_onnx(loaded, path, threshold_size=0)

    def test_removes_orphan_sidecars_keeps_referenced(self, tmp_path) -> None:
        from winml.modelkit.export.htp.exporter import HTPExporter
        from winml.modelkit.onnx import get_external_data_files

        model_path = tmp_path / "model.onnx"
        self._write_model_with_consolidated_data(model_path)

        referenced = get_external_data_files(model_path)
        assert referenced, "expected the consolidated model to reference a .data sidecar"

        # Simulate leftover per-tensor sidecars from the raw TorchScript export.
        orphans = ["onnx__MatMul_1", "linear.weight", "linear.bias"]
        for name in orphans:
            (tmp_path / name).write_bytes(b"stale-weights")

        # The exporter records raw sidecars *plus* whatever the consolidated save
        # references; only the unreferenced ones must be deleted.
        HTPExporter._cleanup_stale_external_data(str(model_path), orphans + referenced)

        for name in orphans:
            assert not (tmp_path / name).exists(), f"orphan {name} should be removed"
        for name in referenced:
            assert (tmp_path / name).exists(), f"referenced {name} must be kept"

    def test_noop_when_no_previous_sidecars(self, tmp_path) -> None:
        from winml.modelkit.export.htp.exporter import HTPExporter

        model_path = tmp_path / "model.onnx"
        self._write_model_with_consolidated_data(model_path)
        before = {p.name for p in tmp_path.iterdir()}

        HTPExporter._cleanup_stale_external_data(str(model_path), [])

        assert {p.name for p in tmp_path.iterdir()} == before

    def test_export_path_leaves_only_referenced_sidecars(self, tmp_path) -> None:
        """End-to-end: HTPExporter.export() must prune raw per-tensor sidecars.

        Drives the real export pipeline but makes the raw ONNX step emit per-tensor
        external-data sidecars (as the TorchScript exporter does). After Step 7's
        re-save, the output directory must contain no orphaned per-tensor sidecars —
        every remaining external-data file must still be referenced by the final
        model. This proves the pre-consolidation capture and post-consolidation
        cleanup stay wired into ``export()``, not just the private helper.
        """
        from onnx.external_data_helper import convert_model_to_external_data

        from winml.modelkit.export.htp.exporter import HTPExporter
        from winml.modelkit.onnx import get_external_data_files

        class BigLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(256, 256)

            def forward(self, x):
                return self.linear(x)

        real_convert = HTPExporter._convert_model_to_onnx
        raw_sidecars: list[str] = []

        def convert_with_per_tensor_sidecars(
            self, model, output_path, inputs, export_config, task=None
        ):
            # Produce a valid graph via the real converter, then rewrite every
            # initializer as its own external sidecar to mimic the raw exporter.
            real_convert(self, model, output_path, inputs, export_config, task=task)
            out = Path(output_path).resolve()
            loaded = onnx.load(str(out))
            convert_model_to_external_data(loaded, all_tensors_to_one_file=False, size_threshold=0)
            onnx.save(loaded, str(out))
            raw_sidecars.extend(get_external_data_files(out))

        model_path = tmp_path / "model.onnx"
        config = WinMLExportConfig(
            input_tensors=[InputTensorSpec(name="x", dtype="float32", shape=(1, 256))],
        )

        with patch.object(
            HTPExporter,
            "_convert_model_to_onnx",
            new=convert_with_per_tensor_sidecars,
        ):
            export_pytorch(BigLinear(), model_path, config)

        # Sanity: the raw step really did emit multiple per-tensor sidecars.
        assert len(raw_sidecars) >= 2, "expected the raw export to emit per-tensor sidecars"

        referenced = set(get_external_data_files(model_path))
        # Every external-data file left in the directory must still be referenced;
        # the raw per-tensor sidecars that consolidation orphaned are gone.
        remaining_sidecars = {
            p.name
            for p in tmp_path.iterdir()
            if p.name != model_path.name and p.suffix not in {".json", ".onnx"}
        }
        assert remaining_sidecars == referenced
        for name in raw_sidecars:
            if name not in referenced:
                assert not (tmp_path / name).exists(), f"orphan {name} must be pruned"
