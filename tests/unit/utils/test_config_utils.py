# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Tests for merge_config utility function.

Tests cover:
1. Basic dict overrides on dataclass
2. Nested config merging (WinMLBuildConfig with nested export/optim/quant/compile)
3. Dict-subclass configs (WinMLOptimizationConfig is a dict subclass)
4. None handling (explicit None to unset optional fields)
5. Merging two config objects (not just dict)
6. Base config unchanged (immutability)
7. Unknown fields ignored
8. List replacement (not merged)
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from winml.modelkit.config import WinMLBuildConfig, merge_config
from winml.modelkit.export.config import InputTensorSpec, OutputTensorSpec, WinMLExportConfig
from winml.modelkit.optim.config import WinMLOptimizationConfig
from winml.modelkit.quant import WinMLQuantizationConfig


class TestMergeConfigBasic:
    """Test basic dict overrides on dataclass."""

    def test_simple_field_override(self) -> None:
        """Test overriding a simple field with a dict."""
        base = WinMLExportConfig(opset_version=17, batch_size=1)
        merged = merge_config(base, {"opset_version": 18})

        assert merged.opset_version == 18
        assert merged.batch_size == 1  # unchanged

    def test_multiple_field_override(self) -> None:
        """Test overriding multiple fields at once."""
        base = WinMLExportConfig(opset_version=17, batch_size=1, verbose=False)
        merged = merge_config(base, {"opset_version": 18, "batch_size": 4, "verbose": True})

        assert merged.opset_version == 18
        assert merged.batch_size == 4
        assert merged.verbose is True

    def test_empty_override_returns_equivalent(self) -> None:
        """Test that empty dict override returns equivalent config."""
        base = WinMLExportConfig(opset_version=17, batch_size=1)
        merged = merge_config(base, {})

        assert merged.opset_version == base.opset_version
        assert merged.batch_size == base.batch_size

    def test_none_override_returns_base(self) -> None:
        """Test that None override returns base config unchanged."""
        base = WinMLExportConfig(opset_version=17, batch_size=1)
        merged = merge_config(base, None)

        # Should return the same object when override is None
        assert merged is base


class TestMergeConfigNested:
    """Test nested config merging with WinMLBuildConfig."""

    def test_nested_export_override(self) -> None:
        """Test overriding nested export config."""
        base = WinMLBuildConfig()
        merged = merge_config(base, {"export": {"opset_version": 18}})

        assert merged.export.opset_version == 18
        # Other export fields unchanged
        assert merged.export.batch_size == base.export.batch_size

    def test_nested_optim_override(self) -> None:
        """Test overriding nested optim config (dict subclass)."""
        base = WinMLBuildConfig(
            optim=WinMLOptimizationConfig(gelu_fusion=True, matmul_add_fusion=False)
        )
        merged = merge_config(base, {"optim": {"matmul_add_fusion": True}})

        assert merged.optim["matmul_add_fusion"] is True
        assert merged.optim["gelu_fusion"] is True  # unchanged

    def test_nested_quant_override(self) -> None:
        """Test overriding nested quant config."""
        base = WinMLBuildConfig()
        merged = merge_config(base, {"quant": {"samples": 100, "weight_type": "int8"}})

        assert merged.quant is not None
        assert merged.quant.samples == 100
        assert merged.quant.weight_type == "int8"

    def test_nested_compile_override(self) -> None:
        """Test overriding nested compile config."""
        base = WinMLBuildConfig()
        merged = merge_config(base, {"compile": {"validate": False, "verbose": True}})

        assert merged.compile is not None
        assert merged.compile.validate is False
        assert merged.compile.verbose is True

    def test_multiple_nested_overrides(self) -> None:
        """Test overriding multiple nested configs at once."""
        base = WinMLBuildConfig()
        merged = merge_config(
            base,
            {
                "export": {"opset_version": 18},
                "optim": {"gelu_fusion": True},
                "quant": {"samples": 50},
            },
        )

        assert merged.export.opset_version == 18
        assert merged.optim["gelu_fusion"] is True
        assert merged.quant is not None
        assert merged.quant.samples == 50


class TestMergeConfigDictSubclass:
    """Test merging dict-subclass configs like WinMLOptimizationConfig."""

    def test_optim_config_update(self) -> None:
        """Test updating WinMLOptimizationConfig (dict subclass)."""
        base = WinMLOptimizationConfig(gelu_fusion=True, matmul_add_fusion=False)
        merged = merge_config(base, {"matmul_add_fusion": True, "new_option": True})

        assert merged["gelu_fusion"] is True  # unchanged
        assert merged["matmul_add_fusion"] is True  # updated
        assert merged["new_option"] is True  # added

    def test_optim_config_preserves_type(self) -> None:
        """Test that merged optim config is still WinMLOptimizationConfig."""
        base = WinMLOptimizationConfig(gelu_fusion=True)
        merged = merge_config(base, {"matmul_add_fusion": True})

        assert isinstance(merged, WinMLOptimizationConfig)
        assert isinstance(merged, dict)

    def test_optim_config_in_model_config(self) -> None:
        """Test optim config merge within WinMLBuildConfig."""
        base = WinMLBuildConfig(
            optim=WinMLOptimizationConfig(gelu_fusion=False, matmul_add_fusion=False)
        )
        merged = merge_config(base, {"optim": {"gelu_fusion": True}})

        assert isinstance(merged.optim, WinMLOptimizationConfig)
        assert merged.optim["gelu_fusion"] is True
        assert merged.optim["matmul_add_fusion"] is False


class TestMergeConfigNoneHandling:
    """Test None handling to unset optional fields."""

    def test_explicit_none_unsets_field(self) -> None:
        """Test that explicit None in override unsets an optional field."""
        base = WinMLBuildConfig()
        assert base.quant is not None  # default is not None

        merged = merge_config(base, {"quant": None})

        assert merged.quant is None

    def test_explicit_none_unsets_compile(self) -> None:
        """Test that explicit None unsets compile config."""
        base = WinMLBuildConfig()
        assert base.compile is not None

        merged = merge_config(base, {"compile": None})

        assert merged.compile is None

    def test_none_to_value_transition(self) -> None:
        """Test transitioning from None to a value.

        Note: When base field is None and override provides a dict, merge_config
        attempts to use from_dict if available on the field type to reconstruct.
        The type resolver uses typing.get_type_hints() to resolve PEP 563
        string annotations, then reconstructs via from_dict() when available.
        """
        base = WinMLBuildConfig(quant=None)
        assert base.quant is None

        merged = merge_config(base, {"quant": {"samples": 100}})

        assert merged.quant is not None
        # Type is correctly resolved and reconstructed via from_dict()
        assert isinstance(merged.quant, WinMLQuantizationConfig)
        assert merged.quant.samples == 100

    def test_none_to_value_transition_with_config_object(self) -> None:
        """Test transitioning from None to a value using a config object.

        This is the recommended way to set a field that was previously None.
        """
        base = WinMLBuildConfig(quant=None)
        assert base.quant is None

        # Use a proper config object for reliable behavior
        merged = merge_config(base, {"quant": WinMLQuantizationConfig(samples=100)})

        assert merged.quant is not None
        assert isinstance(merged.quant, WinMLQuantizationConfig)
        assert merged.quant.samples == 100


class TestMergeConfigObjects:
    """Test merging two config objects (not just dict)."""

    def test_merge_two_export_configs(self) -> None:
        """Test merging two WinMLExportConfig objects."""
        base = WinMLExportConfig(opset_version=17, batch_size=1, verbose=False)
        override = WinMLExportConfig(opset_version=18, batch_size=1, verbose=True)

        merged = merge_config(base, override)

        assert merged.opset_version == 18
        assert merged.verbose is True

    def test_merge_two_model_configs(self) -> None:
        """Test merging two WinMLBuildConfig objects."""
        base = WinMLBuildConfig(
            export=WinMLExportConfig(opset_version=17),
            optim=WinMLOptimizationConfig(gelu_fusion=True),
        )
        override = WinMLBuildConfig(
            export=WinMLExportConfig(opset_version=18),
            optim=WinMLOptimizationConfig(matmul_add_fusion=True),
        )

        merged = merge_config(base, override)

        assert merged.export.opset_version == 18
        # Note: optim is a dict, so it gets replaced/merged
        assert "matmul_add_fusion" in merged.optim

    def test_merge_config_with_to_dict_method(self) -> None:
        """Test merging config objects that have to_dict method."""
        base = WinMLQuantizationConfig(samples=10, weight_type="uint8")
        override = WinMLQuantizationConfig(samples=100, weight_type="int8")

        merged = merge_config(base, override)

        assert merged.samples == 100
        assert merged.weight_type == "int8"


class TestMergeConfigImmutability:
    """Test that base config is not modified (immutability)."""

    def test_base_dataclass_unchanged(self) -> None:
        """Test that base dataclass is not modified after merge."""
        base = WinMLExportConfig(opset_version=17, batch_size=1)
        original_opset = base.opset_version
        original_batch = base.batch_size

        _ = merge_config(base, {"opset_version": 18, "batch_size": 4})

        assert base.opset_version == original_opset
        assert base.batch_size == original_batch

    def test_base_model_config_unchanged(self) -> None:
        """Test that base WinMLBuildConfig is not modified after merge."""
        base = WinMLBuildConfig(
            export=WinMLExportConfig(opset_version=17),
            optim=WinMLOptimizationConfig(gelu_fusion=True),
        )
        original_opset = base.export.opset_version
        original_gelu = base.optim.get("gelu_fusion")

        _ = merge_config(
            base,
            {
                "export": {"opset_version": 18},
                "optim": {"gelu_fusion": False},
            },
        )

        assert base.export.opset_version == original_opset
        assert base.optim.get("gelu_fusion") == original_gelu

    def test_base_dict_config_unchanged(self) -> None:
        """Test that base dict config is not modified after merge."""
        base = WinMLOptimizationConfig(gelu_fusion=True, matmul_add_fusion=False)
        original = dict(base)

        _ = merge_config(base, {"gelu_fusion": False, "new_key": True})

        assert dict(base) == original

    def test_nested_config_immutability(self) -> None:
        """Test that nested configs in base are not modified."""
        base = WinMLBuildConfig()
        original_quant_samples = base.quant.samples if base.quant else None

        _ = merge_config(base, {"quant": {"samples": 999}})

        assert base.quant is not None
        assert base.quant.samples == original_quant_samples


class TestMergeConfigUnknownFields:
    """Test that unknown fields are ignored."""

    def test_unknown_fields_ignored_dataclass(self) -> None:
        """Test that unknown fields are ignored for dataclass."""
        base = WinMLExportConfig(opset_version=17)
        merged = merge_config(
            base,
            {
                "opset_version": 18,
                "unknown_field": "should_be_ignored",
                "another_unknown": 123,
            },
        )

        assert merged.opset_version == 18
        assert not hasattr(merged, "unknown_field")
        assert not hasattr(merged, "another_unknown")

    def test_unknown_nested_fields_ignored(self) -> None:
        """Test that unknown fields in nested configs are ignored."""
        base = WinMLBuildConfig()
        merged = merge_config(
            base,
            {
                "export": {"opset_version": 18, "fake_field": "ignored"},
                "unknown_section": {"data": 123},
            },
        )

        assert merged.export.opset_version == 18
        assert not hasattr(merged.export, "fake_field")
        assert not hasattr(merged, "unknown_section")


class TestMergeConfigListReplacement:
    """Test that lists are replaced, not merged."""

    def test_list_replacement_with_spec_objects(self) -> None:
        """Test that input_tensors list is replaced, not merged (using InputTensorSpec objects)."""
        base = WinMLExportConfig(
            input_tensors=[
                InputTensorSpec(name="input1", shape=(1, 3, 224, 224)),
                InputTensorSpec(name="input2", shape=(1, 128)),
            ]
        )
        merged = merge_config(
            base,
            {
                "input_tensors": [
                    InputTensorSpec(name="new_input", shape=(1, 512)),
                ]
            },
        )

        # Should be replaced, not merged
        assert len(merged.input_tensors) == 1
        assert merged.input_tensors[0].name == "new_input"
        assert merged.input_tensors[0].shape == (1, 512)

    def test_list_replacement_with_raw_dicts_on_simple_list(self) -> None:
        """Test list replacement with raw dicts on a config without validation.

        Note: merge_config replaces lists directly without converting dict items
        to their proper types. This works for configs that don't validate list
        contents in __post_init__.

        For WinMLExportConfig specifically, using raw dicts in input_tensors
        will fail because __post_init__ validates that items are InputTensorSpec
        objects with 'shape' attribute. Use InputTensorSpec objects instead.
        """
        # Test with WinMLQuantizationConfig's optional list fields
        base = WinMLQuantizationConfig(
            nodes_to_exclude=["node1", "node2"],
        )
        merged = merge_config(
            base,
            {
                "nodes_to_exclude": ["new_node"],
            },
        )

        # List is replaced entirely
        assert merged.nodes_to_exclude == ["new_node"]

    def test_list_replacement_output_tensors_with_spec_objects(self) -> None:
        """Test that output_tensors list is replaced using OutputTensorSpec objects."""
        base = WinMLExportConfig(
            output_tensors=[
                OutputTensorSpec(name="output1"),
                OutputTensorSpec(name="output2"),
            ]
        )
        merged = merge_config(
            base,
            {
                "output_tensors": [
                    OutputTensorSpec(name="single_output"),
                ]
            },
        )

        assert len(merged.output_tensors) == 1
        assert merged.output_tensors[0].name == "single_output"

    def test_empty_list_replacement(self) -> None:
        """Test that an empty list replaces existing list."""
        base = WinMLExportConfig(
            input_tensors=[
                InputTensorSpec(name="input1", shape=(1, 3, 224, 224)),
            ]
        )
        merged = merge_config(base, {"input_tensors": []})

        assert merged.input_tensors == []

    def test_list_count_verification(self) -> None:
        """Test that list replacement changes the list length correctly."""
        base = WinMLExportConfig(
            input_tensors=[
                InputTensorSpec(name="input1", shape=(1, 3, 224, 224)),
                InputTensorSpec(name="input2", shape=(1, 128)),
                InputTensorSpec(name="input3", shape=(1, 64)),
            ]
        )
        # Replace 3 items with 1
        merged = merge_config(
            base,
            {
                "input_tensors": [
                    InputTensorSpec(name="single_input"),
                ]
            },
        )

        assert len(merged.input_tensors) == 1
        assert base.input_tensors is not None
        assert len(base.input_tensors) == 3  # Original unchanged


class TestMergeConfigEdgeCases:
    """Test edge cases and special scenarios."""

    def test_merge_with_override_dict_copied(self) -> None:
        """Test that the override dict is copied and not mutated."""
        base = WinMLExportConfig(opset_version=17)
        override: dict[str, Any] = {"opset_version": 18}
        original_override = copy.deepcopy(override)

        _ = merge_config(base, override)

        assert override == original_override

    def test_invalid_override_type_raises(self) -> None:
        """Test that invalid override type raises TypeError."""
        base = WinMLExportConfig(opset_version=17)

        with pytest.raises(TypeError, match="overrides must be dict or config"):
            merge_config(base, "invalid")  # type: ignore

        with pytest.raises(TypeError, match="overrides must be dict or config"):
            merge_config(base, 123)  # type: ignore

    def test_deeply_nested_merge(self) -> None:
        """Test merging with deeply nested structure."""
        base = WinMLBuildConfig()
        # WinMLCompileConfig has nested EPConfig, QDQConfig, CalibrationConfig
        merged = merge_config(
            base,
            {
                "compile": {
                    "validate": False,
                    "verbose": True,
                },
            },
        )

        assert merged.compile is not None
        assert merged.compile.validate is False
        assert merged.compile.verbose is True

    def test_preserve_default_factory_values(self) -> None:
        """Test that default factory values are preserved when not overridden."""
        base = WinMLBuildConfig()

        # Only override export, others should keep defaults
        merged = merge_config(base, {"export": {"opset_version": 18}})

        assert merged.export.opset_version == 18
        # These should still have their default factory values
        assert merged.optim is not None
        assert isinstance(merged.optim, WinMLOptimizationConfig)
        assert merged.quant is not None
        assert merged.compile is not None


class TestMergeConfigRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_preset_override_pattern(self) -> None:
        """Test the common pattern of overriding a preset config."""
        # This mimics the usage shown in the config docstring
        preset = WinMLBuildConfig(
            export=WinMLExportConfig(opset_version=17, batch_size=1),
            optim=WinMLOptimizationConfig(gelu_fusion=True),
            quant=WinMLQuantizationConfig(samples=10),
        )

        user_config = merge_config(
            preset,
            {
                "quant": {"samples": 100, "weight_type": "int8"},
                "export": {"opset_version": 18},
            },
        )

        # User overrides applied
        assert user_config.quant is not None
        assert user_config.quant.samples == 100
        assert user_config.quant.weight_type == "int8"
        assert user_config.export.opset_version == 18

        # Preset values preserved
        assert user_config.optim["gelu_fusion"] is True
        assert user_config.export.batch_size == 1

    def test_disable_quantization_pattern(self) -> None:
        """Test the pattern of disabling quantization via merge."""
        base = WinMLBuildConfig()  # Has quant enabled by default

        disabled = merge_config(base, {"quant": None})

        assert disabled.quant is None
        # Other configs unchanged
        assert disabled.export is not None
        assert disabled.compile is not None

    def test_quick_config_adjustment(self) -> None:
        """Test quick config adjustments for testing/debugging."""
        base = WinMLBuildConfig()

        # Quick adjustment for debugging
        debug_config = merge_config(
            base,
            {
                "export": {"verbose": True},
                "compile": {"verbose": True},
            },
        )

        assert debug_config.export.verbose is True
        assert debug_config.compile is not None
        assert debug_config.compile.verbose is True
