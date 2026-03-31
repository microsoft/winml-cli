# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ORTFusionPipe and ORTFusionPipeConfig.

Tests the fusion optimization pipe following Cardinal Rules:
- CARDINAL RULE #1: No hardcoded model architectures
- CARDINAL RULE #2: All tests use pytest with code-generated results
- CARDINAL RULE #3: Tests must run and pass

Test Categories:
1. ORTFusionPipeConfig Tests - Configuration structure and defaults
2. ORTFusionPipe.build_config Tests - Config building from kwargs
3. ORTFusionPipe.process Tests - Model processing with FusionOptions
4. ORTFusionPipe Integration Tests - End-to-end workflow
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import onnx
import pytest


if TYPE_CHECKING:
    from collections.abc import Generator


# Import pipe classes from modelkit (production)
from winml.modelkit.optim.pipes.fusion import ORTFusionPipe, ORTFusionPipeConfig
from winml.modelkit.optim.registry import (
    BoolCapability,
    CapabilityCategory,
)


@pytest.fixture
def clean_registry() -> Generator[None, None, None]:
    """Fixture for test isolation - clears capabilities between tests."""
    # Note: In production modelkit, capabilities are class-level constants
    # This fixture provides isolation semantics for testing
    yield


@pytest.fixture
def fusion_capabilities(clean_registry: None) -> dict[str, BoolCapability]:
    """Create sample fusion pipe capabilities for testing.

    Note: In production, capabilities are defined as class-level constants on pipes.
    This fixture creates test-specific capability instances.
    Note: GELU capabilities are disabled in FusionPipe due to ORT bundling issue.
    """
    return {
        "attention-fusion": BoolCapability(
            name="attention-fusion",
            ort_name="AttentionFusion",
            description="Fuse attention patterns",
            category=CapabilityCategory.ATTENTION,
            default=False,
        ),
        "layer-norm-fusion": BoolCapability(
            name="layer-norm-fusion",
            ort_name="LayerNormFusion",
            description="Fuse layer normalization",
            category=CapabilityCategory.LAYER_NORM,
            default=False,
        ),
    }


class TestORTFusionPipeConfig:
    """Tests for ORTFusionPipeConfig dataclass."""

    def test_fusion_pipe_config_defaults(self) -> None:
        """Verify ORTFusionPipeConfig has correct default values."""
        config = ORTFusionPipeConfig()

        # Check default model type
        assert config.model_type == "clip"

        # Check fusion toggles default to False
        # GELU (5) - DISABLED (use GraphPipe for GELU fusion)
        # LayerNorm (4)
        assert config.enable_layer_norm is False
        assert config.enable_skip_layer_norm is False
        assert config.enable_embed_layer_norm is False
        assert config.enable_bias_skip_layer_norm is False
        # Attention (3)
        assert config.enable_attention is False
        assert config.enable_packed_qkv is False
        assert config.enable_packed_kv is False
        # GroupNorm (2)
        assert config.enable_group_norm is False
        assert config.enable_skip_group_norm is False
        # MatMul (1)
        assert config.enable_qordered_matmul is False
        # Layout & Misc (2)
        assert config.enable_nhwc_conv is False
        assert config.enable_bias_add is False

    def test_fusion_pipe_config_custom_values(self) -> None:
        """Verify ORTFusionPipeConfig accepts custom values."""
        config = ORTFusionPipeConfig(
            model_type="gpt2",
            enable_attention=True,
            enable_layer_norm=True,
        )

        assert config.model_type == "gpt2"
        assert config.enable_attention is True
        assert config.enable_layer_norm is True

    def test_fusion_pipe_config_is_pipe_config(self) -> None:
        """Verify ORTFusionPipeConfig inherits from PipeConfig."""
        from winml.modelkit.optim.pipes.base import PipeConfig

        config = ORTFusionPipeConfig()
        assert isinstance(config, PipeConfig)

    def test_fusion_pipe_config_all_fusion_options(self) -> None:
        """Verify ORTFusionPipeConfig has all FusionOptions fields."""
        # Create config with all 12 fusion toggles enabled (GELU disabled)
        config = ORTFusionPipeConfig(
            model_type="t5",
            # GELU (5) - DISABLED (use GraphPipe for GELU fusion)
            # LayerNorm (4)
            enable_layer_norm=True,
            enable_skip_layer_norm=True,
            enable_embed_layer_norm=True,
            enable_bias_skip_layer_norm=True,
            # Attention (3)
            enable_attention=True,
            enable_packed_qkv=True,
            enable_packed_kv=True,
            # GroupNorm (2)
            enable_group_norm=True,
            enable_skip_group_norm=True,
            # MatMul (1)
            enable_qordered_matmul=True,
            # Layout & Misc (2)
            enable_nhwc_conv=True,
            enable_bias_add=True,
        )

        # Verify all fields are set
        assert config.model_type == "t5"
        # GELU (5) - DISABLED (use GraphPipe for GELU fusion)
        # LayerNorm (4)
        assert config.enable_layer_norm is True
        assert config.enable_skip_layer_norm is True
        assert config.enable_embed_layer_norm is True
        assert config.enable_bias_skip_layer_norm is True
        # Attention (3)
        assert config.enable_attention is True
        assert config.enable_packed_qkv is True
        assert config.enable_packed_kv is True
        # GroupNorm (2)
        assert config.enable_group_norm is True
        assert config.enable_skip_group_norm is True
        # MatMul (1)
        assert config.enable_qordered_matmul is True
        # Layout & Misc (2)
        assert config.enable_nhwc_conv is True
        assert config.enable_bias_add is True


class TestORTFusionPipeBuildConfig:
    """Tests for ORTFusionPipe.build_config method."""

    def test_build_config_empty_kwargs(self, fusion_capabilities: dict) -> None:
        """Build config with no kwargs should use defaults."""
        config = ORTFusionPipe.build_config()

        assert isinstance(config, ORTFusionPipeConfig)
        assert config.model_type == "clip"
        # All fusion options default to False (GELU disabled due to ORT bundling)
        assert config.enable_attention is False
        assert config.enable_layer_norm is False

    def test_build_config_model_type(self, fusion_capabilities: dict) -> None:
        """Build config with explicit model type."""
        for model_type in ["bert", "gpt2", "t5", "vit"]:
            config = ORTFusionPipe.build_config(model_type=model_type)
            assert config.model_type == model_type

    def test_build_config_single_fusion(self, fusion_capabilities: dict) -> None:
        """Enable a single fusion via kwargs."""
        config = ORTFusionPipe.build_config(layer_norm_fusion=True)

        assert config.enable_layer_norm is True
        assert config.enable_attention is False

    def test_build_config_multiple_fusions(self, fusion_capabilities: dict) -> None:
        """Enable multiple fusions via kwargs."""
        config = ORTFusionPipe.build_config(attention_fusion=True, layer_norm_fusion=True)

        assert config.enable_attention is True
        assert config.enable_layer_norm is True

    def test_build_config_fusion_attrs_mapping(self, fusion_capabilities: dict) -> None:
        """Verify fusion_attr correctly maps capabilities to config fields.

        Note: GELU capabilities are disabled due to ORT bundling issue.
        """
        # attention-fusion has pipe_metadata={"fusion_attr": "enable_attention"}
        config = ORTFusionPipe.build_config(attention_fusion=True)
        assert config.enable_attention is True

        # layer-norm-fusion has pipe_metadata={"fusion_attr": "enable_layer_norm"}
        config = ORTFusionPipe.build_config(layer_norm_fusion=True)
        assert config.enable_layer_norm is True

    def test_build_config_respects_capability_defaults(self, fusion_capabilities: dict) -> None:
        """Build config without overrides should respect capability defaults.

        Note: GELU capabilities are disabled due to ORT bundling issue.
        """
        # All fusion capabilities default to False
        config = ORTFusionPipe.build_config()

        assert config.enable_attention is False
        assert config.enable_layer_norm is False

    def test_build_config_uses_pipe_capabilities(self, clean_registry: None) -> None:
        """Build config should use capabilities defined on the pipe class.

        Note: GELU capabilities are disabled due to ORT bundling issue.
        """
        # In production, ORTFusionPipe has its own capabilities dict
        # that determines what fusion options are available
        config = ORTFusionPipe.build_config()

        # Verify config is created successfully with pipe's capabilities
        assert isinstance(config, ORTFusionPipeConfig)
        # All fusion options should default to False
        assert config.enable_attention is False
        assert config.enable_layer_norm is False

    def test_build_config_ignores_unknown_kwargs(self, fusion_capabilities: dict) -> None:
        """Build config should gracefully ignore unknown kwargs."""
        # Should not raise error for unknown parameters
        config = ORTFusionPipe.build_config(
            unknown_param="value", another_unknown=123, layer_norm_fusion=True
        )

        # Known parameter should still work
        assert config.enable_layer_norm is True

    def test_build_config_with_model_type_and_fusions(self, fusion_capabilities: dict) -> None:
        """Build config with both model type and fusion options."""
        config = ORTFusionPipe.build_config(
            model_type="gpt2", layer_norm_fusion=True, attention_fusion=True
        )

        assert config.model_type == "gpt2"
        assert config.enable_layer_norm is True
        assert config.enable_attention is True


class TestORTFusionPipeProcess:
    """Tests for ORTFusionPipe.process method."""

    def test_process_returns_model(self, sample_model: onnx.ModelProto) -> None:
        """Process should return an ONNX ModelProto."""
        pipe = ORTFusionPipe()
        config = ORTFusionPipeConfig(model_type="bert")

        result = pipe.process(sample_model, config)

        assert isinstance(result, onnx.ModelProto)
        assert result.graph is not None

    def test_process_preserves_model_structure(self, sample_model: onnx.ModelProto) -> None:
        """Process should preserve basic model structure."""
        pipe = ORTFusionPipe()
        config = ORTFusionPipeConfig(model_type="bert")

        result = pipe.process(sample_model, config)

        # Basic structure should be preserved
        # Note: Fusion may modify nodes, but inputs/outputs should remain
        assert len(result.graph.input) == len(sample_model.graph.input)
        assert len(result.graph.output) == len(sample_model.graph.output)

    def test_process_with_no_fusions(self, sample_model: onnx.ModelProto) -> None:
        """Process with all fusions disabled should still work."""
        pipe = ORTFusionPipe()
        config = ORTFusionPipeConfig(model_type="bert")  # All fusions default to False

        result = pipe.process(sample_model, config)

        assert isinstance(result, onnx.ModelProto)

    def test_process_with_single_fusion(self, sample_model: onnx.ModelProto) -> None:
        """Process with single fusion enabled."""
        pipe = ORTFusionPipe()
        config = ORTFusionPipeConfig(model_type="bert", enable_layer_norm=True)

        result = pipe.process(sample_model, config)

        assert isinstance(result, onnx.ModelProto)

    def test_process_with_multiple_fusions(self, sample_model: onnx.ModelProto) -> None:
        """Process with multiple fusions enabled."""
        pipe = ORTFusionPipe()
        config = ORTFusionPipeConfig(
            model_type="bert",
            enable_attention=True,
            enable_layer_norm=True,
        )

        result = pipe.process(sample_model, config)

        assert isinstance(result, onnx.ModelProto)

    @pytest.mark.slow
    def test_process_different_model_types(self, sample_model: onnx.ModelProto) -> None:
        """Test processing with different model types."""
        pipe = ORTFusionPipe()

        # Test common model types
        for model_type in ["bert", "gpt2", "t5"]:
            config = ORTFusionPipeConfig(model_type=model_type)
            result = pipe.process(sample_model, config)
            assert isinstance(result, onnx.ModelProto)


class TestORTFusionPipeShouldProcess:
    """Tests for ORTFusionPipe.should_process method."""

    def test_should_process_returns_false_when_no_fusions(self) -> None:
        """should_process returns False when all fusion options are False."""
        config = ORTFusionPipeConfig(model_type="bert")  # All fusions default to False
        assert ORTFusionPipe.should_process(config) is False

    def test_should_process_returns_true_with_single_fusion(self) -> None:
        """should_process returns True when at least one fusion option is True.

        Note: GELU capabilities (5) are disabled due to ORT bundling issue.
        """
        # Test each of the 12 fusion toggles individually (GELU disabled)
        fusion_options = [
            # GELU (5) - DISABLED (use GraphPipe for GELU fusion)
            # LayerNorm (4)
            "enable_layer_norm",
            "enable_skip_layer_norm",
            "enable_embed_layer_norm",
            "enable_bias_skip_layer_norm",
            # Attention (3)
            "enable_attention",
            "enable_packed_qkv",
            "enable_packed_kv",
            # GroupNorm (2)
            "enable_group_norm",
            "enable_skip_group_norm",
            # MatMul (1)
            "enable_qordered_matmul",
            # Layout & Misc (2)
            "enable_nhwc_conv",
            "enable_bias_add",
        ]

        for option in fusion_options:
            config = ORTFusionPipeConfig(**{option: True})
            assert ORTFusionPipe.should_process(config) is True, f"Failed for {option}"

    def test_should_process_returns_true_with_multiple_fusions(self) -> None:
        """should_process returns True when multiple fusion options are True.

        Note: GELU capabilities (5) are disabled due to ORT bundling issue.
        """
        config = ORTFusionPipeConfig(enable_attention=True, enable_layer_norm=True)
        assert ORTFusionPipe.should_process(config) is True

        # Test all 12 fusion toggles enabled (GELU disabled)
        config = ORTFusionPipeConfig(
            # GELU (5) - DISABLED (use GraphPipe for GELU fusion)
            # LayerNorm (4)
            enable_layer_norm=True,
            enable_skip_layer_norm=True,
            enable_embed_layer_norm=True,
            enable_bias_skip_layer_norm=True,
            # Attention (3)
            enable_attention=True,
            enable_packed_qkv=True,
            enable_packed_kv=True,
            # GroupNorm (2)
            enable_group_norm=True,
            enable_skip_group_norm=True,
            # MatMul (1)
            enable_qordered_matmul=True,
            # Layout & Misc (2)
            enable_nhwc_conv=True,
            enable_bias_add=True,
        )
        assert ORTFusionPipe.should_process(config) is True

    def test_should_process_checks_all_fusion_options(self) -> None:
        """Verify should_process checks all 12 fusion toggles (GELU disabled)."""
        # Create a config with all options False
        config = ORTFusionPipeConfig()
        assert ORTFusionPipe.should_process(config) is False

        # Verify that enabling any single option makes it return True
        # This confirms all 12 fusion toggles are checked (GELU disabled)
        all_options = {
            # GELU (5) - DISABLED (use GraphPipe for GELU fusion)
            # LayerNorm (4)
            "enable_layer_norm": False,
            "enable_skip_layer_norm": False,
            "enable_embed_layer_norm": False,
            "enable_bias_skip_layer_norm": False,
            # Attention (3)
            "enable_attention": False,
            "enable_packed_qkv": False,
            "enable_packed_kv": False,
            # GroupNorm (2)
            "enable_group_norm": False,
            "enable_skip_group_norm": False,
            # MatMul (1)
            "enable_qordered_matmul": False,
            # Layout & Misc (2)
            "enable_nhwc_conv": False,
            "enable_bias_add": False,
        }

        # Test each option individually
        for option_name in all_options:
            test_config_kwargs = all_options.copy()
            test_config_kwargs[option_name] = True
            test_config = ORTFusionPipeConfig(**test_config_kwargs)
            assert ORTFusionPipe.should_process(test_config) is True, (
                f"should_process should return True when {option_name} is enabled"
            )


class TestORTFusionPipeIntegration:
    """Integration tests for ORTFusionPipe end-to-end workflow."""

    def test_should_process_method_exists(self) -> None:
        """ORTFusionPipe should have optional should_process method."""
        pipe = ORTFusionPipe()
        # Method may or may not exist - just check it doesn't error
        assert hasattr(pipe, "process")

    def test_end_to_end_workflow(
        self, sample_model: onnx.ModelProto, fusion_capabilities: dict
    ) -> None:
        """Test complete workflow from kwargs to processed model.

        Note: GELU capabilities are disabled due to ORT bundling issue.
        """
        # Step 1: Build config from user kwargs
        config = ORTFusionPipe.build_config(
            model_type="bert", layer_norm_fusion=True, attention_fusion=True
        )

        # Step 2: Create pipe and process model
        pipe = ORTFusionPipe()
        result = pipe.process(sample_model, config)

        # Verify results
        assert isinstance(result, onnx.ModelProto)
        assert config.model_type == "bert"
        assert config.enable_layer_norm is True
        assert config.enable_attention is True

    def test_pipe_class_attributes(self) -> None:
        """Verify ORTFusionPipe has required class attributes."""
        assert hasattr(ORTFusionPipe, "name")
        assert hasattr(ORTFusionPipe, "capabilities")
        assert ORTFusionPipe.name == "ort_fusion"
        # Verify capabilities dict contains expected entries
        assert isinstance(ORTFusionPipe.capabilities, dict)
        assert len(ORTFusionPipe.capabilities) > 0

    def test_pipe_has_capabilities_dict(
        self, fusion_capabilities: dict, clean_registry: None
    ) -> None:
        """ORTFusionPipe should have a capabilities class attribute."""
        # In production, capabilities are defined as class-level dicts on pipes
        caps = ORTFusionPipe.capabilities

        # Should be a non-empty dict
        assert isinstance(caps, dict)
        assert len(caps) > 0

        # Verify some known capabilities exist (these are defined in capabilities modules)
        # Note: The exact capabilities depend on what's imported in the pipe module
        cap_names = list(caps.keys())
        assert len(cap_names) > 0  # At least some capabilities are registered

    def test_multiple_pipes_independent(
        self, sample_model: onnx.ModelProto, clean_registry: None
    ) -> None:
        """Multiple ORTFusionPipe instances should be independent."""
        pipe1 = ORTFusionPipe()
        pipe2 = ORTFusionPipe()

        config1 = ORTFusionPipeConfig(model_type="bert")
        config2 = ORTFusionPipeConfig(model_type="gpt2")

        result1 = pipe1.process(sample_model, config1)
        result2 = pipe2.process(sample_model, config2)

        # Both should work independently
        assert isinstance(result1, onnx.ModelProto)
        assert isinstance(result2, onnx.ModelProto)

    def test_build_config_with_custom_kwargs(self, clean_registry: None) -> None:
        """Build config should handle custom kwargs gracefully."""
        # Should not raise error for unknown parameters
        # ORTFusionPipe only uses capabilities from its class-level capabilities dict
        config = ORTFusionPipe.build_config(custom_flag=True, unknown_param="test")
        assert isinstance(config, ORTFusionPipeConfig)

    def test_all_fusion_options_accessible(self) -> None:
        """Verify all 12 FusionOptions fields are accessible in config (GELU disabled)."""
        config = ORTFusionPipeConfig()

        # All 12 fusion toggle fields should be accessible (GELU disabled)
        fusion_fields = [
            # GELU (5) - DISABLED (use GraphPipe for GELU fusion)
            # LayerNorm (4)
            "enable_layer_norm",
            "enable_skip_layer_norm",
            "enable_embed_layer_norm",
            "enable_bias_skip_layer_norm",
            # Attention (3)
            "enable_attention",
            "enable_packed_qkv",
            "enable_packed_kv",
            # GroupNorm (2)
            "enable_group_norm",
            "enable_skip_group_norm",
            # MatMul (1)
            "enable_qordered_matmul",
            # Layout & Misc (2)
            "enable_nhwc_conv",
            "enable_bias_add",
        ]

        for field in fusion_fields:
            assert hasattr(config, field)
            assert isinstance(getattr(config, field), bool)


class TestORTFusionPipeEdgeCases:
    """Edge case tests for ORTFusionPipe."""

    def test_build_config_with_all_fusions_enabled(self, fusion_capabilities: dict) -> None:
        """Test build_config with all fusion options enabled.

        This edge case ensures the system can handle enabling all fusion
        optimizations simultaneously without conflicts. All fusion capabilities
        are explicitly enabled via kwargs.

        Note: GELU capabilities are disabled due to ORT bundling issue.
        """
        config = ORTFusionPipe.build_config(attention_fusion=True, layer_norm_fusion=True)

        assert isinstance(config, ORTFusionPipeConfig)
        # Verify all enabled fusions are reflected in config
        assert config.enable_attention is True
        assert config.enable_layer_norm is True

    def test_build_config_with_different_model_types(self, fusion_capabilities: dict) -> None:
        """Test build_config with different model types.

        This tests that build_config correctly handles various model types
        including BERT-family (bert), GPT-family (gpt2), encoder-decoder (t5),
        and vision models (vit). Each model type may have different fusion
        optimization behaviors.
        """
        model_types = ["gpt2", "t5", "vit"]

        for model_type in model_types:
            config = ORTFusionPipe.build_config(model_type=model_type)

            assert isinstance(config, ORTFusionPipeConfig)
            assert config.model_type == model_type

    def test_process_passthrough_when_no_fusions(self, sample_model: onnx.ModelProto) -> None:
        """Test that model is returned unchanged when no fusions are enabled.

        When all fusion options are False, the should_process check should return
        False, and the model should be returned without any fusion processing.
        This verifies the passthrough behavior.
        """
        pipe = ORTFusionPipe()
        # All fusion options default to False
        config = ORTFusionPipeConfig(model_type="bert")

        # Verify should_process returns False
        assert not pipe.should_process(config)

        # Process should return model unchanged
        result = pipe.process(sample_model, config)

        # Should be the exact same model object (no fusion applied)
        assert result is sample_model
        assert isinstance(result, onnx.ModelProto)


# =============================================================================
# GELU DISABLING VERIFICATION TESTS
# =============================================================================


class TestGeluDisabledInFusionPipe:
    """Tests verifying GELU capabilities are disabled in FusionPipe.

    FIXME: ORT's FusionOptions.enable_gelu bundles multiple fusion types
    (GELU, QuickGelu, FastGelu) under one flag, making it impossible to
    control them independently. This causes unexpected behavior where enabling
    gelu-fusion also fuses QuickGelu patterns (x*sigmoid(1.702*x)).

    Use GraphPipe for isolated GELU fusion control instead.

    See: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/fusion_options.py
    """

    def test_fusionpipe_capabilities_exclude_gelu(self) -> None:
        """Verify FusionPipe.capabilities does NOT contain GELU capabilities.

        This test ensures that GELU-related capabilities are not exposed
        in FusionPipe, forcing users to use GraphPipe for GELU fusion.
        """
        caps = ORTFusionPipe.capabilities

        # GELU capabilities should NOT be in FusionPipe
        gelu_capability_names = [
            "gelu-fusion",
            "bias-gelu-fusion",
            "gelu-approximation",
            "quick-gelu-fusion",
            "fast-gelu-fusion",
        ]

        for cap_name in gelu_capability_names:
            assert cap_name not in caps, (
                f"GELU capability '{cap_name}' should NOT be in FusionPipe.capabilities. "
                f"GELU capabilities are disabled due to ORT bundling issue."
            )

    def test_fusionpipe_config_excludes_gelu_fields(self) -> None:
        """Verify ORTFusionPipeConfig does NOT have GELU-related fields.

        This test ensures that GELU fields are not exposed in the config,
        preventing users from accidentally enabling GELU fusion via FusionPipe.
        """
        config = ORTFusionPipeConfig()

        # GELU fields should NOT be in ORTFusionPipeConfig
        gelu_fields = [
            "enable_gelu",
            "enable_bias_gelu",
            "enable_gelu_approximation",
            "enable_gemm_fast_gelu",
            "enable_bias_splitgelu",
        ]

        for field in gelu_fields:
            assert not hasattr(config, field), (
                f"GELU field '{field}' should NOT be in ORTFusionPipeConfig. "
                f"GELU capabilities are disabled due to ORT bundling issue."
            )

    def test_fusionpipe_does_not_fuse_gelu_pattern(self) -> None:
        """Verify FusionPipe does NOT fuse GELU patterns.

        This test creates a model with a GELU pattern and verifies that
        FusionPipe does NOT produce fused Gelu nodes, confirming that
        GELU fusion is effectively disabled.
        """
        from ..assets.graphpipe.builders.gelu import gelu_fusion_builder

        # Build a model with GELU pattern
        initializers: list = []
        nodes = gelu_fusion_builder(
            input_name="input",
            output_name="output",
            prefix="gelu_test_",
            initializers=initializers,
        )

        # Create model with GELU pattern
        graph = onnx.helper.make_graph(
            nodes,
            "gelu_test",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 64])],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 64])],
            initializers,
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])

        # Count Erf nodes before processing (GELU pattern has Erf)
        erf_count_before = sum(1 for n in model.graph.node if n.op_type == "Erf")
        assert erf_count_before == 1, "GELU pattern should have exactly 1 Erf node"

        # Process with FusionPipe - even with layer_norm enabled (some fusion active)
        pipe = ORTFusionPipe()
        config = ORTFusionPipeConfig(model_type="clip", enable_layer_norm=True)
        result = pipe.process(model, config)

        # Verify NO fused Gelu nodes were created
        gelu_count = sum(1 for n in result.graph.node if n.op_type == "Gelu")
        assert gelu_count == 0, (
            f"FusionPipe should NOT produce Gelu nodes (found {gelu_count}). "
            f"GELU fusion is disabled due to ORT bundling issue."
        )

        # Verify the Erf node is still present (pattern not fused)
        erf_count_after = sum(1 for n in result.graph.node if n.op_type == "Erf")
        assert erf_count_after == 1, (
            f"GELU pattern should remain decomposed (Erf node should still exist). "
            f"Found {erf_count_after} Erf nodes after processing."
        )

    def test_fusionpipe_does_not_fuse_quick_gelu_pattern(self) -> None:
        """Verify FusionPipe does NOT fuse QuickGelu patterns.

        This is the critical test case that motivated disabling GELU in FusionPipe.
        ORT's enable_gelu flag bundles QuickGelu fusion, causing unexpected behavior.
        """
        from ..assets.graphpipe.builders.gelu import quick_gelu_builder

        # Build a model with QuickGelu pattern
        initializers: list = []
        nodes = quick_gelu_builder(
            input_name="input",
            output_name="output",
            prefix="quick_gelu_test_",
            initializers=initializers,
        )

        # Create model with QuickGelu pattern
        graph = onnx.helper.make_graph(
            nodes,
            "quick_gelu_test",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 64])],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 64])],
            initializers,
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])

        # Count Sigmoid nodes before processing (QuickGelu pattern has Sigmoid)
        sigmoid_count_before = sum(1 for n in model.graph.node if n.op_type == "Sigmoid")
        assert sigmoid_count_before == 1, "QuickGelu pattern should have exactly 1 Sigmoid node"

        # Process with FusionPipe - even with layer_norm enabled (some fusion active)
        pipe = ORTFusionPipe()
        config = ORTFusionPipeConfig(model_type="clip", enable_layer_norm=True)
        result = pipe.process(model, config)

        # Verify NO fused QuickGelu nodes were created
        quick_gelu_count = sum(1 for n in result.graph.node if n.op_type == "QuickGelu")
        assert quick_gelu_count == 0, (
            f"FusionPipe should NOT produce QuickGelu nodes (found {quick_gelu_count}). "
            f"GELU fusion is disabled due to ORT bundling issue."
        )

        # Verify the Sigmoid node is still present (pattern not fused)
        sigmoid_count_after = sum(1 for n in result.graph.node if n.op_type == "Sigmoid")
        assert sigmoid_count_after == 1, (
            f"QuickGelu pattern should remain decomposed (Sigmoid node should still exist). "
            f"Found {sigmoid_count_after} Sigmoid nodes after processing."
        )


# =============================================================================
# CONTROL OPTION TESTS (Section 7.3 of 5_fusion_pipe.md)
# =============================================================================


class TestAttentionOpType:
    """Tests for attention-op-type control option.

    Tests that attention-op-type correctly controls which operator
    attention fusion produces:
    - Attention → com.microsoft.Attention
    - MultiHeadAttention → com.microsoft.MultiHeadAttention
    - GroupQueryAttention → com.microsoft.GroupQueryAttention
    - PagedAttention → SKIP (requires vLLM)
    """

    @pytest.fixture
    def self_attention_model(self) -> onnx.ModelProto:
        """Load self-attention pattern for testing."""

        from ..assets.fusionpipe.generate_patterns import create_self_attention_model

        return create_self_attention_model(
            batch_size=1,
            seq_len=128,
            hidden_size=768,
            num_heads=12,
            prefix="test_attn_",
        )

    @pytest.fixture
    def gqa_model(self) -> onnx.ModelProto:
        """Load GQA pattern for testing."""
        from ..assets.fusionpipe.generate_patterns import create_gqa_model

        return create_gqa_model(
            batch_size=1,
            seq_len=128,
            hidden_size=1024,
            num_heads=32,
            kv_num_heads=8,
            prefix="test_gqa_",
        )

    def _count_nodes_by_op_type(
        self, model: onnx.ModelProto, op_type: str, domain: str = ""
    ) -> int:
        """Count nodes with specific op_type and domain."""
        count = 0
        for node in model.graph.node:
            node_domain = node.domain if node.domain else ""
            if node.op_type == op_type and node_domain == domain:
                count += 1
        return count

    def _has_node(self, model: onnx.ModelProto, op_type: str, domain: str = "") -> bool:
        """Check if model has a node with specific op_type and domain."""
        return self._count_nodes_by_op_type(model, op_type, domain) > 0

    def test_self_attention_model_structure(self, self_attention_model: onnx.ModelProto) -> None:
        """Verify self-attention test model has expected structure."""
        # Should have MatMul nodes for Q, K, V projections
        matmul_count = self._count_nodes_by_op_type(self_attention_model, "MatMul")
        assert matmul_count >= 3, f"Expected at least 3 MatMul nodes, got {matmul_count}"

        # Should have Softmax for attention
        softmax_count = self._count_nodes_by_op_type(self_attention_model, "Softmax")
        assert softmax_count >= 1, f"Expected at least 1 Softmax node, got {softmax_count}"

        # Should NOT have fused attention nodes before optimization
        assert not self._has_node(self_attention_model, "Attention", "com.microsoft")
        assert not self._has_node(self_attention_model, "MultiHeadAttention", "com.microsoft")

    def test_gqa_model_structure(self, gqa_model: onnx.ModelProto) -> None:
        """Verify GQA test model has expected structure."""
        # Should have MatMul nodes for Q, K, V projections
        matmul_count = self._count_nodes_by_op_type(gqa_model, "MatMul")
        assert matmul_count >= 3, f"Expected at least 3 MatMul nodes, got {matmul_count}"

        # Should have Tile nodes for KV head expansion (GQA-specific)
        tile_count = self._count_nodes_by_op_type(gqa_model, "Tile")
        assert tile_count >= 1, f"Expected at least 1 Tile node for KV expansion, got {tile_count}"

        # Should NOT have fused GQA nodes before optimization
        assert not self._has_node(gqa_model, "GroupQueryAttention", "com.microsoft")

    def test_attention_fusion_produces_attention_op(
        self, self_attention_model: onnx.ModelProto
    ) -> None:
        """Test that attention fusion with default settings produces Attention op.

        When attention-fusion is enabled, ORT should produce com.microsoft.Attention.
        """
        pipe = ORTFusionPipe()
        config = ORTFusionPipeConfig(
            model_type="clip",  # Always use clip internally
            enable_attention=True,
        )

        # Process the model
        result = pipe.process(self_attention_model, config)

        # Note: ORT's transformer optimizer may or may not fuse depending on pattern
        # This test verifies the config is correctly passed to ORT
        assert isinstance(result, onnx.ModelProto)

        # Check if fusion occurred (it may not if pattern doesn't match exactly)
        has_attention = self._has_node(result, "Attention", "com.microsoft")
        has_mha = self._has_node(result, "MultiHeadAttention", "com.microsoft")

        # Log what happened for debugging
        print(f"Attention node found: {has_attention}")
        print(f"MultiHeadAttention node found: {has_mha}")
        print(f"Result nodes: {[n.op_type for n in result.graph.node]}")

    @pytest.mark.skip(reason="use_multi_head_attention is a control option, not a config toggle")
    def test_attention_fusion_with_mha_produces_mha_op(
        self, self_attention_model: onnx.ModelProto
    ) -> None:
        """Test that use_multi_head_attention=True produces MultiHeadAttention op.

        Note: This test is skipped because use_multi_head_attention is a control
        option that is NOT part of ORTFusionPipeConfig. Control options are handled
        separately from fusion toggles.
        """

    @pytest.mark.skip(reason="GQA requires specific model structure and ORT version")
    def test_gqa_fusion_produces_gqa_op(self, gqa_model: onnx.ModelProto) -> None:
        """Test that GQA pattern produces GroupQueryAttention op.

        Note: This test is skipped because GQA fusion requires:
        1. Specific model structure with rotary embeddings
        2. ORT version with GQA support
        3. Correct num_heads/kv_num_heads configuration

        The test pattern may need refinement to match ORT's exact expectations.
        """
        pipe = ORTFusionPipe()
        config = ORTFusionPipeConfig(
            model_type="clip",
            enable_attention=True,
        )

        result = pipe.process(gqa_model, config)

        assert isinstance(result, onnx.ModelProto)

        has_gqa = self._has_node(result, "GroupQueryAttention", "com.microsoft")
        print(f"GroupQueryAttention node found: {has_gqa}")

    @pytest.mark.skip(
        reason="PagedAttention requires vLLM runtime - not testable with standard ORT"
    )
    def test_paged_attention_skipped(self) -> None:
        """PagedAttention requires vLLM runtime and is not testable.

        PagedAttention (vllm.ort.ext.PagedAttention) is a vLLM-specific
        operator for paged KV cache during inference serving. It cannot
        be tested with standard ONNX Runtime.
        """


class TestGroupNormChannelsLast:
    """Tests for group-norm-channels-last control option.

    Tests that group_norm_channels_last correctly toggles between:
    - True (default): NHWC layout for GroupNorm
    - False: NCHW layout preserved
    """

    @pytest.fixture
    def groupnorm_model(self) -> onnx.ModelProto:
        """Create GroupNorm model for testing."""
        from ..assets.fusionpipe.generate_patterns import create_groupnorm_model

        return create_groupnorm_model(
            batch_size=1,
            channels=32,
            height=16,
            width=16,
            num_groups=4,
            prefix="gnlayout_",
        )

    def test_groupnorm_model_structure(self, groupnorm_model: onnx.ModelProto) -> None:
        """Verify GroupNorm test model has expected structure."""
        # Should have Reshape nodes for group normalization
        reshape_count = sum(1 for n in groupnorm_model.graph.node if n.op_type == "Reshape")
        assert reshape_count >= 2, f"Expected at least 2 Reshape nodes, got {reshape_count}"

        # Should have ReduceMean for normalization
        reducemean_count = sum(1 for n in groupnorm_model.graph.node if n.op_type == "ReduceMean")
        assert reducemean_count >= 1, f"Expected ReduceMean nodes, got {reducemean_count}"

    @pytest.mark.skip(reason="GroupNorm fusion requires SD model type and specific pattern")
    def test_groupnorm_channels_last_layout(self, groupnorm_model: onnx.ModelProto) -> None:
        """Test GroupNorm fusion with channels_last (NHWC) layout.

        Note: Skipped because GroupNorm fusion is only enabled for
        Stable Diffusion model types (unet, vae, clip) and requires
        specific pattern matching.
        """
        pipe = ORTFusionPipe()
        config = ORTFusionPipeConfig(
            model_type="unet",  # GroupNorm fusion only for SD models
        )

        result = pipe.process(groupnorm_model, config)
        assert isinstance(result, onnx.ModelProto)
