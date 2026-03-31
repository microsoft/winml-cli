# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for WinMLExportConfig validation paths and roundtrip serialization.

Tests all validation logic in WinMLExportConfig.__post_init__(), plus
InputTensorSpec / OutputTensorSpec roundtrip serialization.
"""

from __future__ import annotations

import logging

import pytest

from winml.modelkit.export.config import (
    InputTensorSpec,
    OutputTensorSpec,
    WinMLExportConfig,
)


# =============================================================================
# 1-2. batch_size validation (ValueError)
# =============================================================================


class TestBatchSizeValidation:
    """batch_size must be positive; <= 0 raises ValueError."""

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            WinMLExportConfig(batch_size=0)

    def test_batch_size_negative_one_raises(self):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            WinMLExportConfig(batch_size=-1)

    def test_batch_size_negative_large_raises(self):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            WinMLExportConfig(batch_size=-100)

    def test_batch_size_positive_ok(self):
        cfg = WinMLExportConfig(batch_size=1)
        assert cfg.batch_size == 1

    def test_batch_size_large_positive_ok(self):
        cfg = WinMLExportConfig(batch_size=32)
        assert cfg.batch_size == 32


# =============================================================================
# 3. opset_version < 11 logs warning
# =============================================================================


class TestOpsetVersionWarning:
    """opset_version < 11 emits a logger.warning about being very old."""

    def test_opset_below_11_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.export.config"):
            WinMLExportConfig(opset_version=10)
        assert "opset_version 10 is very old" in caplog.text

    def test_opset_7_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.export.config"):
            WinMLExportConfig(opset_version=7)
        assert "opset_version 7 is very old" in caplog.text

    def test_opset_11_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.export.config"):
            WinMLExportConfig(opset_version=11)
        assert "very old" not in caplog.text

    def test_opset_17_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.export.config"):
            WinMLExportConfig(opset_version=17)
        assert "very old" not in caplog.text


# =============================================================================
# 4. Input tensor shape[0] != batch_size logs warning
# =============================================================================


class TestInputTensorShapeMismatchWarning:
    """When input_tensors have shape[0] != batch_size, a warning is logged."""

    def test_shape_mismatch_warns(self, caplog):
        tensors = [InputTensorSpec(name="pixel_values", shape=(4, 3, 224, 224))]
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.export.config"):
            WinMLExportConfig(batch_size=1, input_tensors=tensors)
        assert "shape[0]=4 doesn't match batch_size=1" in caplog.text

    def test_shape_mismatch_unnamed_tensor(self, caplog):
        tensors = [InputTensorSpec(shape=(2, 3, 224, 224))]
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.export.config"):
            WinMLExportConfig(batch_size=1, input_tensors=tensors)
        assert "unnamed" in caplog.text
        assert "shape[0]=2 doesn't match batch_size=1" in caplog.text

    def test_shape_matches_no_warning(self, caplog):
        tensors = [InputTensorSpec(name="pixel_values", shape=(1, 3, 224, 224))]
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.export.config"):
            WinMLExportConfig(batch_size=1, input_tensors=tensors)
        assert "doesn't match batch_size" not in caplog.text

    def test_no_shape_no_warning(self, caplog):
        tensors = [InputTensorSpec(name="pixel_values")]
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.export.config"):
            WinMLExportConfig(batch_size=1, input_tensors=tensors)
        assert "doesn't match batch_size" not in caplog.text

    def test_multiple_tensors_mismatch(self, caplog):
        tensors = [
            InputTensorSpec(name="input_ids", shape=(1, 128)),
            InputTensorSpec(name="attention_mask", shape=(2, 128)),
        ]
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.export.config"):
            WinMLExportConfig(batch_size=1, input_tensors=tensors)
        assert "attention_mask" in caplog.text
        assert "input_ids" not in caplog.text  # shape[0]=1 matches batch_size=1


# =============================================================================
# 5. dynamic_axes with axis 0 logs BiasGelu warning
# =============================================================================


class TestDynamicAxesWarning:
    """dynamic_axes with axis 0 warns about BiasGelu."""

    def test_dynamic_batch_axis_warns(self, caplog):
        dynamic_axes = {"input_ids": {0: "batch_size"}}
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.export.config"):
            WinMLExportConfig(dynamic_axes=dynamic_axes)
        assert "Dynamic batch detected for input 'input_ids'" in caplog.text
        assert "BiasGelu" in caplog.text

    def test_dynamic_non_batch_axis_no_warning(self, caplog):
        dynamic_axes = {"input_ids": {1: "sequence_length"}}
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.export.config"):
            WinMLExportConfig(dynamic_axes=dynamic_axes)
        assert "BiasGelu" not in caplog.text

    def test_multiple_inputs_dynamic_batch(self, caplog):
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch"},
        }
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.export.config"):
            WinMLExportConfig(dynamic_axes=dynamic_axes)
        assert "input_ids" in caplog.text
        assert "attention_mask" in caplog.text

    def test_no_dynamic_axes_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.export.config"):
            WinMLExportConfig(dynamic_axes=None)
        assert "BiasGelu" not in caplog.text


# =============================================================================
# 6. hierarchy_tag_format validation (ValueError)
# =============================================================================


class TestHierarchyTagFormatValidation:
    """hierarchy_tag_format must be 'full' or 'module_only'."""

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid hierarchy_tag_format"):
            WinMLExportConfig(hierarchy_tag_format="invalid")

    def test_full_format_ok(self):
        cfg = WinMLExportConfig(hierarchy_tag_format="full")
        assert cfg.hierarchy_tag_format == "full"

    def test_module_only_format_ok(self):
        cfg = WinMLExportConfig(hierarchy_tag_format="module_only")
        assert cfg.hierarchy_tag_format == "module_only"


# =============================================================================
# 7. clean_onnx=True with enable_hierarchy_tags=False logs warning
# =============================================================================


class TestCleanOnnxConflictWarning:
    """clean_onnx=True has no effect when enable_hierarchy_tags=False."""

    def test_clean_without_hierarchy_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.export.config"):
            WinMLExportConfig(clean_onnx=True, enable_hierarchy_tags=False)
        assert "clean_onnx=True has no effect" in caplog.text

    def test_clean_with_hierarchy_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.export.config"):
            WinMLExportConfig(clean_onnx=True, enable_hierarchy_tags=True)
        assert "clean_onnx=True has no effect" not in caplog.text

    def test_no_clean_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="winml.modelkit.export.config"):
            WinMLExportConfig(clean_onnx=False, enable_hierarchy_tags=False)
        assert "clean_onnx=True has no effect" not in caplog.text


# =============================================================================
# 8. InputTensorSpec roundtrip: to_dict -> from_dict preserves all fields
# =============================================================================


class TestInputTensorSpecRoundtrip:
    """InputTensorSpec serialization roundtrip."""

    def test_full_roundtrip(self):
        original = InputTensorSpec(name="pixel_values", dtype="float32", shape=(1, 3, 224, 224))
        restored = InputTensorSpec.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.dtype == original.dtype
        assert restored.shape == original.shape

    def test_minimal_roundtrip(self):
        original = InputTensorSpec()
        d = original.to_dict()
        assert d == {}
        restored = InputTensorSpec.from_dict(d)
        assert restored.name is None
        assert restored.dtype is None
        assert restored.shape is None

    def test_name_only_roundtrip(self):
        original = InputTensorSpec(name="input_ids")
        restored = InputTensorSpec.from_dict(original.to_dict())
        assert restored.name == "input_ids"
        assert restored.dtype is None
        assert restored.shape is None


# =============================================================================
# 9. InputTensorSpec.from_dict converts list shape to tuple
# =============================================================================


class TestInputTensorSpecListToTuple:
    """from_dict must convert list shape to tuple."""

    def test_list_shape_converted_to_tuple(self):
        data = {"name": "pixel_values", "shape": [1, 3, 224, 224]}
        spec = InputTensorSpec.from_dict(data)
        assert isinstance(spec.shape, tuple)
        assert spec.shape == (1, 3, 224, 224)

    def test_tuple_shape_preserved(self):
        data = {"name": "pixel_values", "shape": (1, 3, 224, 224)}
        spec = InputTensorSpec.from_dict(data)
        assert isinstance(spec.shape, tuple)
        assert spec.shape == (1, 3, 224, 224)

    def test_none_shape_preserved(self):
        data = {"name": "pixel_values"}
        spec = InputTensorSpec.from_dict(data)
        assert spec.shape is None


# =============================================================================
# 10. OutputTensorSpec roundtrip
# =============================================================================


class TestOutputTensorSpecRoundtrip:
    """OutputTensorSpec serialization roundtrip."""

    def test_full_roundtrip(self):
        original = OutputTensorSpec(name="logits")
        restored = OutputTensorSpec.from_dict(original.to_dict())
        assert restored.name == original.name

    def test_minimal_roundtrip(self):
        original = OutputTensorSpec()
        d = original.to_dict()
        assert d == {}
        restored = OutputTensorSpec.from_dict(d)
        assert restored.name is None


# =============================================================================
# 11. WinMLExportConfig roundtrip: to_dict -> from_dict
# =============================================================================


class TestWinMLExportConfigRoundtrip:
    """WinMLExportConfig serialization roundtrip."""

    def test_defaults_roundtrip(self):
        original = WinMLExportConfig()
        restored = WinMLExportConfig.from_dict(original.to_dict())
        assert restored.opset_version == original.opset_version
        assert restored.batch_size == original.batch_size
        assert restored.export_params == original.export_params
        assert restored.do_constant_folding == original.do_constant_folding
        assert restored.verbose == original.verbose
        assert restored.enable_hierarchy_tags == original.enable_hierarchy_tags
        assert restored.clean_onnx == original.clean_onnx
        assert restored.hierarchy_tag_format == original.hierarchy_tag_format
        assert restored.input_tensors is None
        assert restored.output_tensors is None
        assert restored.dynamic_axes is None

    def test_full_roundtrip(self):
        original = WinMLExportConfig(
            opset_version=14,
            batch_size=2,
            input_tensors=[
                InputTensorSpec(name="pixel_values", dtype="float32", shape=(2, 3, 224, 224)),
                InputTensorSpec(name="attention_mask", dtype="int64", shape=(2, 128)),
            ],
            output_tensors=[
                OutputTensorSpec(name="logits"),
                OutputTensorSpec(name="hidden_states"),
            ],
            dynamic_axes={"pixel_values": {1: "channels"}},
            export_params=False,
            do_constant_folding=False,
            verbose=True,
            enable_hierarchy_tags=True,
            clean_onnx=True,
            hierarchy_tag_format="module_only",
        )
        restored = WinMLExportConfig.from_dict(original.to_dict())

        assert restored.opset_version == 14
        assert restored.batch_size == 2
        assert restored.export_params is False
        assert restored.do_constant_folding is False
        assert restored.verbose is True
        assert restored.enable_hierarchy_tags is True
        assert restored.clean_onnx is True
        assert restored.hierarchy_tag_format == "module_only"

        assert len(restored.input_tensors) == 2
        assert restored.input_tensors[0].name == "pixel_values"
        assert restored.input_tensors[0].dtype == "float32"
        assert restored.input_tensors[0].shape == (2, 3, 224, 224)
        assert restored.input_tensors[1].name == "attention_mask"

        assert len(restored.output_tensors) == 2
        assert restored.output_tensors[0].name == "logits"
        assert restored.output_tensors[1].name == "hidden_states"

        assert restored.dynamic_axes == {"pixel_values": {1: "channels"}}

    def test_from_dict_ignores_unknown_fields(self):
        data = {"opset_version": 17, "batch_size": 1, "unknown_field": "ignored"}
        cfg = WinMLExportConfig.from_dict(data)
        assert cfg.opset_version == 17
        assert not hasattr(cfg, "unknown_field")


# =============================================================================
# 12. Backward compatibility: legacy InitVar parameters
# =============================================================================


class TestLegacyParameterBackwardCompat:
    """Legacy input_shape_, input_names_, output_names_ are converted to tensor specs."""

    def test_legacy_input_shape_converted(self):
        cfg = WinMLExportConfig(input_shape_=(1, 3, 224, 224))
        assert cfg.input_tensors is not None
        assert len(cfg.input_tensors) == 1
        assert cfg.input_tensors[0].name == "input"
        assert cfg.input_tensors[0].shape == (1, 3, 224, 224)

    def test_legacy_input_shape_with_names(self):
        cfg = WinMLExportConfig(
            input_shape_=(1, 3, 224, 224),
            input_names_=["pixel_values", "attention_mask"],
        )
        assert cfg.input_tensors is not None
        assert len(cfg.input_tensors) == 2
        assert cfg.input_tensors[0].name == "pixel_values"
        assert cfg.input_tensors[0].shape == (1, 3, 224, 224)
        assert cfg.input_tensors[1].name == "attention_mask"
        assert cfg.input_tensors[1].shape is None

    def test_legacy_output_names_converted(self):
        cfg = WinMLExportConfig(output_names_=["logits", "hidden_states"])
        assert cfg.output_tensors is not None
        assert len(cfg.output_tensors) == 2
        assert cfg.output_tensors[0].name == "logits"
        assert cfg.output_tensors[1].name == "hidden_states"

    def test_legacy_ignored_when_input_tensors_provided(self):
        explicit = [InputTensorSpec(name="explicit", shape=(1, 3, 32, 32))]
        cfg = WinMLExportConfig(
            input_tensors=explicit,
            input_shape_=(1, 3, 224, 224),
        )
        assert len(cfg.input_tensors) == 1
        assert cfg.input_tensors[0].name == "explicit"
        assert cfg.input_tensors[0].shape == (1, 3, 32, 32)

    def test_legacy_ignored_when_output_tensors_provided(self):
        explicit = [OutputTensorSpec(name="explicit")]
        cfg = WinMLExportConfig(
            output_tensors=explicit,
            output_names_=["legacy_logits"],
        )
        assert len(cfg.output_tensors) == 1
        assert cfg.output_tensors[0].name == "explicit"

    def test_legacy_all_params_together(self):
        cfg = WinMLExportConfig(
            input_shape_=(1, 3, 224, 224),
            input_names_=["pixel_values"],
            output_names_=["logits"],
        )
        assert len(cfg.input_tensors) == 1
        assert cfg.input_tensors[0].name == "pixel_values"
        assert cfg.input_tensors[0].shape == (1, 3, 224, 224)
        assert len(cfg.output_tensors) == 1
        assert cfg.output_tensors[0].name == "logits"
