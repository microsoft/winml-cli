# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ORTGraphPipeConfig.

Tests configuration logic for the graph optimization pipe.
All tests use capabilities that exist in GRAPH_CAPABILITIES (default=False only).

Test Capabilities Used:
- gelu_fusion (GeluFusionL2) - GELU activation fusion
- matmul_add_fusion (MatMulAddFusion) - MatMul + Add fusion
- layer_norm_fusion (LayerNormFusionL2) - Layer normalization fusion

Design Principle:
- GRAPH_CAPABILITIES only contains default=False items (advanced optimizations)
- Basic optimizations (ConstantFolding, IdentityElimination) are handled by ORT Level 2
- ORTGraphPipeConfig enables specific advanced optimizations users want
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from winml.modelkit.optim.pipes import (
    GRAPH_CAPABILITIES,
    ORTGraphPipe,
    ORTGraphPipeConfig,
    PipeConfig,
)


if TYPE_CHECKING:
    import pytest


# =============================================================================
# TEST CONSTANTS - Capabilities used for testing (all default=False)
# =============================================================================

# Primary test capabilities from different categories
# NOTE: attention_fusion is in FusionPipe, not GraphPipe
CAP_GELU = "gelu_fusion"  # ORT: GeluFusionL2
CAP_MATMUL = "matmul_add_fusion"  # ORT: MatMulAddFusion
CAP_LAYERNORM = "layer_norm_fusion"  # ORT: LayerNormFusionL2

ORT_GELU = "GeluFusionL2"
ORT_MATMUL = "MatMulAddFusion"
ORT_LAYERNORM = "LayerNormFusionL2"

# L1 variants that are ALSO added to disabled_optimizers for isolation
# These are L1 versions of L2 optimizers that run at optimization level 2
# See graph.py lines 172-178 for why these are needed
L1_VARIANTS = ["GeluFusion", "LayerNormFusion"]

# Optimizers that are ALWAYS disabled - not handled by GraphPipe
# These require optimize_model() API, not available through SessionOptions
ALWAYS_DISABLED = ["AttentionFusion", "EmbedLayerNormFusion"]


class TestORTGraphPipeConfigInit:
    """Tests for ORTGraphPipeConfig.__init__."""

    def test_defaults_all_disabled(self) -> None:
        """Default config has default=False optimizers disabled (conservative baseline)."""
        config = ORTGraphPipeConfig()

        assert config.optimization_level == 2
        assert isinstance(config.disabled_optimizers, list)
        assert len(config.disabled_optimizers) > 0

        # Count should match GRAPH_CAPABILITIES with ort_name AND default=False,
        # PLUS L1 variants, PLUS always-disabled optimizers
        # Capabilities with default=True (like ConstantFolding) stay enabled
        # L1 variants (GeluFusion, LayerNormFusion) are added for proper isolation
        # Always-disabled (AttentionFusion, EmbedLayerNormFusion) are never enabled
        caps_count = len(
            [
                c
                for c in GRAPH_CAPABILITIES.values()
                if hasattr(c, "ort_name") and c.ort_name and not c.default
            ]
        )
        expected_count = caps_count + len(L1_VARIANTS) + len(ALWAYS_DISABLED)
        assert len(config.disabled_optimizers) == expected_count

    def test_defaults_includes_test_caps(self) -> None:
        """Default config should have test capabilities in disabled list."""
        config = ORTGraphPipeConfig()

        # All our test caps should be disabled by default
        assert ORT_GELU in config.disabled_optimizers
        assert ORT_MATMUL in config.disabled_optimizers
        assert ORT_LAYERNORM in config.disabled_optimizers

    def test_enabled_single_cap(self) -> None:
        """Enable single capability removes it from disabled list."""
        config = ORTGraphPipeConfig(enabled=[CAP_GELU])

        assert ORT_GELU not in config.disabled_optimizers
        # Others still disabled
        assert ORT_MATMUL in config.disabled_optimizers
        assert ORT_LAYERNORM in config.disabled_optimizers

    def test_enabled_multiple_caps(self) -> None:
        """Enable multiple capabilities removes them from disabled list."""
        config = ORTGraphPipeConfig(enabled=[CAP_GELU, CAP_MATMUL, CAP_LAYERNORM])

        assert ORT_GELU not in config.disabled_optimizers
        assert ORT_MATMUL not in config.disabled_optimizers
        assert ORT_LAYERNORM not in config.disabled_optimizers

    def test_enabled_reduces_disabled_count(self) -> None:
        """Enabling caps reduces the disabled_optimizers count."""
        baseline = ORTGraphPipeConfig()
        enabled_one = ORTGraphPipeConfig(enabled=[CAP_GELU])
        enabled_three = ORTGraphPipeConfig(enabled=[CAP_GELU, CAP_MATMUL, CAP_LAYERNORM])

        # gelu_fusion enables both GeluFusionL2 AND GeluFusion (L1 variant) = -2
        assert len(enabled_one.disabled_optimizers) == len(baseline.disabled_optimizers) - 2
        # gelu_fusion(-2) + matmul_add_fusion(-1) + layer_norm_fusion(-2) = -5
        # (layer_norm_fusion enables both LayerNormFusionL2 AND LayerNormFusion L1 variant)
        assert len(enabled_three.disabled_optimizers) == len(baseline.disabled_optimizers) - 5

    def test_verbose_parameter(self) -> None:
        """Verbose parameter is stored correctly."""
        config_verbose = ORTGraphPipeConfig(verbose=True)
        config_quiet = ORTGraphPipeConfig(verbose=False)

        assert config_verbose.verbose is True
        assert config_quiet.verbose is False

    def test_always_disabled_optimizers(self) -> None:
        """AttentionFusion and EmbedLayerNormFusion are ALWAYS disabled.

        These optimizers require optimize_model() API with transformer-specific
        analysis, not available through SessionOptions. They must always be
        disabled in GraphPipe regardless of configuration.
        """
        config = ORTGraphPipeConfig()

        # These must always be in the disabled list
        assert "AttentionFusion" in config.disabled_optimizers
        assert "EmbedLayerNormFusion" in config.disabled_optimizers

    def test_always_disabled_cannot_be_enabled(self) -> None:
        """Always-disabled optimizers cannot be enabled through config.

        Even if someone tries to enable these through a capability name,
        they should remain disabled because they're not in GRAPH_CAPABILITIES.
        """
        # These are not valid capability names for GraphPipe
        config = ORTGraphPipeConfig(enabled=["attention_fusion", "embed_layer_norm_fusion"])

        # They should still be in disabled list (enable() silently ignores unknown names)
        assert "AttentionFusion" in config.disabled_optimizers
        assert "EmbedLayerNormFusion" in config.disabled_optimizers

    def test_optimization_level_fixed_at_2(self) -> None:
        """Optimization level is always 2 (not configurable)."""
        config = ORTGraphPipeConfig()
        assert config.optimization_level == 2


class TestORTGraphPipeConfigEnable:
    """Tests for ORTGraphPipeConfig.enable() method."""

    def test_enable_returns_self(self) -> None:
        """enable() returns self for method chaining."""
        config = ORTGraphPipeConfig()
        result = config.enable(CAP_GELU)

        assert result is config

    def test_enable_method_chaining(self) -> None:
        """enable() supports method chaining."""
        config = ORTGraphPipeConfig()
        initial_count = len(config.disabled_optimizers)

        result = config.enable(CAP_GELU).enable(CAP_MATMUL).enable(CAP_LAYERNORM)

        assert result is config
        # gelu_fusion enables both GeluFusionL2 AND GeluFusion (L1 variant) = -2
        # matmul_add_fusion = -1
        # layer_norm_fusion enables both LayerNormFusionL2 AND LayerNormFusion (L1 variant) = -2
        # total = -5
        assert len(config.disabled_optimizers) == initial_count - 5
        assert ORT_GELU not in config.disabled_optimizers
        assert ORT_MATMUL not in config.disabled_optimizers
        assert ORT_LAYERNORM not in config.disabled_optimizers

    def test_enable_unknown_cap_is_noop(self) -> None:
        """Enabling unknown capability is a no-op (no error)."""
        config = ORTGraphPipeConfig()
        initial_count = len(config.disabled_optimizers)

        result = config.enable("unknown_capability_xyz")

        assert result is config
        assert len(config.disabled_optimizers) == initial_count

    def test_enable_already_enabled_is_noop(self) -> None:
        """Enabling already-enabled cap is idempotent."""
        config = ORTGraphPipeConfig(enabled=[CAP_GELU])
        initial_count = len(config.disabled_optimizers)

        config.enable(CAP_GELU)

        assert len(config.disabled_optimizers) == initial_count


class TestORTGraphPipeConfigInheritance:
    """Tests for ORTGraphPipeConfig class structure."""

    def test_inherits_from_pipe_config(self) -> None:
        """ORTGraphPipeConfig inherits from PipeConfig."""
        config = ORTGraphPipeConfig()
        assert isinstance(config, PipeConfig)

    def test_has_required_attributes(self) -> None:
        """Config has all required attributes."""
        config = ORTGraphPipeConfig()

        assert hasattr(config, "optimization_level")
        assert hasattr(config, "disabled_optimizers")
        assert hasattr(config, "verbose")


class TestORTGraphPipeBuildConfig:
    """Tests for ORTGraphPipe.build_config() method."""

    def test_returns_config_instance(self) -> None:
        """build_config returns ORTGraphPipeConfig instance."""
        config = ORTGraphPipe.build_config()

        assert isinstance(config, ORTGraphPipeConfig)

    def test_empty_kwargs_all_disabled(self) -> None:
        """Empty kwargs means all caps disabled (all are default=False)."""
        config = ORTGraphPipe.build_config()

        # All test caps should be disabled
        assert ORT_GELU in config.disabled_optimizers
        assert ORT_MATMUL in config.disabled_optimizers
        assert ORT_LAYERNORM in config.disabled_optimizers

    def test_enable_via_true_kwarg(self) -> None:
        """Setting cap=True enables it (isolation mode)."""
        config = ORTGraphPipe.build_config(gelu_fusion=True)

        assert ORT_GELU not in config.disabled_optimizers
        # Other caps still disabled (isolation mode)
        assert ORT_MATMUL in config.disabled_optimizers

    def test_enable_multiple_via_kwargs(self) -> None:
        """Setting multiple caps=True enables them all."""
        config = ORTGraphPipe.build_config(
            gelu_fusion=True,
            matmul_add_fusion=True,
            layer_norm_fusion=True,
        )

        assert ORT_GELU not in config.disabled_optimizers
        assert ORT_MATMUL not in config.disabled_optimizers
        assert ORT_LAYERNORM not in config.disabled_optimizers

    def test_explicit_false_keeps_disabled(self) -> None:
        """Setting cap=False keeps it disabled."""
        config = ORTGraphPipe.build_config(gelu_fusion=False)

        assert ORT_GELU in config.disabled_optimizers

    def test_isolation_mode_only_enabled_run(self) -> None:
        """When any cap is enabled, ONLY enabled caps run (isolation mode)."""
        config = ORTGraphPipe.build_config(gelu_fusion=True)

        # Only gelu should be enabled (both L2 and L1 variants)
        assert ORT_GELU not in config.disabled_optimizers
        assert "GeluFusion" not in config.disabled_optimizers  # L1 variant also enabled

        # All others should be disabled (except default=True caps like ConstantFolding)
        # gelu_fusion enables 2 items (GeluFusionL2 + GeluFusion L1 variant)
        disabled_count = len(config.disabled_optimizers)
        caps_count = len(
            [
                c
                for c in GRAPH_CAPABILITIES.values()
                if hasattr(c, "ort_name") and c.ort_name and not c.default
            ]
        )
        # Total disabled = caps(default=False) + L1_variants + always_disabled - enabled(2)
        expected_disabled = caps_count + len(L1_VARIANTS) + len(ALWAYS_DISABLED) - 2
        assert disabled_count == expected_disabled

    def test_unknown_kwargs_ignored(self) -> None:
        """Unknown kwargs are gracefully ignored."""
        config = ORTGraphPipe.build_config(
            unknown_param="value",
            another_unknown=123,
            gelu_fusion=True,
        )

        assert isinstance(config, ORTGraphPipeConfig)
        assert ORT_GELU not in config.disabled_optimizers

    def test_optimization_level_kwarg_ignored(self) -> None:
        """graph_optimization_level kwarg is ignored (level fixed at 2)."""
        config = ORTGraphPipe.build_config(graph_optimization_level=99)

        assert config.optimization_level == 2

    def test_verbose_kwarg_passed_through(self) -> None:
        """verbose kwarg is passed to config."""
        config = ORTGraphPipe.build_config(verbose=True)

        assert config.verbose is True


class TestGraphCapabilitiesIntegrity:
    """Tests for GRAPH_CAPABILITIES module-level constant."""

    def test_graph_capabilities_exists(self) -> None:
        """GRAPH_CAPABILITIES is available at module level."""
        assert isinstance(GRAPH_CAPABILITIES, dict)
        assert len(GRAPH_CAPABILITIES) > 0

    def test_graph_capabilities_same_as_pipe(self) -> None:
        """GRAPH_CAPABILITIES is same object as pipe's capabilities."""
        assert ORTGraphPipe.capabilities is GRAPH_CAPABILITIES

    def test_all_caps_have_ort_name(self) -> None:
        """All capabilities in GRAPH_CAPABILITIES have ort_name."""
        for name, cap in GRAPH_CAPABILITIES.items():
            assert hasattr(cap, "ort_name"), f"{name} missing ort_name"
            assert cap.ort_name, f"{name} has empty ort_name"

    def test_most_bool_caps_are_default_false(self) -> None:
        """Most BoolCapability in GRAPH_CAPABILITIES are default=False.

        Design principle: GRAPH_CAPABILITIES contains advanced optimizations
        that users must explicitly enable (default=False), with exceptions
        like ConstantFolding (default=True) which is enabled by default but
        can be disabled for size-sensitive models.
        """
        from winml.modelkit.optim.registry import BoolCapability

        # Capabilities that are allowed to have default=True
        allowed_default_true = {"constant-folding"}

        for name, cap in GRAPH_CAPABILITIES.items():
            if isinstance(cap, BoolCapability):
                if name in allowed_default_true:
                    assert cap.default is True, f"{name} should be default=True"
                else:
                    assert cap.default is False, (
                        f"{name} should be default=False, got {cap.default}"
                    )

    def test_test_caps_exist_in_graph_capabilities(self) -> None:
        """Our test capabilities exist in GRAPH_CAPABILITIES."""
        cap_python_names = [cap.python_name for cap in GRAPH_CAPABILITIES.values()]

        assert CAP_GELU in cap_python_names
        assert CAP_MATMUL in cap_python_names
        assert CAP_LAYERNORM in cap_python_names


class TestORTGraphPipeConfigDependencies:
    """Tests for automatic dependency handling in ORTGraphPipeConfig.enable().

    When certain capabilities are enabled, their dependencies must also be enabled.
    These tests verify the dependency chains defined in graph.py lines 188-200.
    """

    def test_bias_gelu_fusion_enables_gelu_fusion(self) -> None:
        """bias_gelu_fusion requires gelu_fusion (GeluFusionL2)."""
        config = ORTGraphPipeConfig()

        # Both should be disabled initially
        assert "GeluFusionL2" in config.disabled_optimizers
        assert "BiasGeluFusion" in config.disabled_optimizers

        config.enable("bias_gelu_fusion")

        # Both should be enabled (dependency auto-enabled)
        assert "GeluFusionL2" not in config.disabled_optimizers
        assert "BiasGeluFusion" not in config.disabled_optimizers

    def test_gelu_approximation_enables_gelu_fusion(self) -> None:
        """gelu_approximation requires gelu_fusion (GeluFusionL2)."""
        config = ORTGraphPipeConfig()

        assert "GeluFusionL2" in config.disabled_optimizers

        config.enable("gelu_approximation")

        # GeluFusionL2 should be auto-enabled
        assert "GeluFusionL2" not in config.disabled_optimizers
        # GeluApproximation uses special flag, not disabled_optimizers
        assert config.enable_gelu_approximation is True

    def test_skip_layer_norm_fusion_enables_layer_norm_fusion(self) -> None:
        """skip_layer_norm_fusion requires layer_norm_fusion (LayerNormFusionL2)."""
        config = ORTGraphPipeConfig()

        assert "LayerNormFusionL2" in config.disabled_optimizers
        assert "SkipLayerNormFusion" in config.disabled_optimizers

        config.enable("skip_layer_norm_fusion")

        # Dependency should be auto-enabled
        assert "LayerNormFusionL2" not in config.disabled_optimizers
        assert "SkipLayerNormFusion" not in config.disabled_optimizers

    def test_bias_skip_layer_norm_fusion_enables_both_dependencies(self) -> None:
        """bias_skip_layer_norm_fusion requires BOTH LayerNormFusionL2 AND SkipLayerNormFusion.

        This tests the previously uncovered lines 195-196 in graph.py.
        """
        config = ORTGraphPipeConfig()

        # All three should be disabled initially
        assert "LayerNormFusionL2" in config.disabled_optimizers
        assert "SkipLayerNormFusion" in config.disabled_optimizers

        config.enable("bias_skip_layer_norm_fusion")

        # Both dependencies should be auto-enabled
        assert "LayerNormFusionL2" not in config.disabled_optimizers
        assert "SkipLayerNormFusion" not in config.disabled_optimizers

    def test_matmul_activation_fusion_enables_matmul_transpose_fusion(self) -> None:
        """matmul_activation_fusion requires matmul_transpose_fusion (MatmulTransposeFusion)."""
        config = ORTGraphPipeConfig()

        assert "MatmulTransposeFusion" in config.disabled_optimizers
        assert "MatMulActivationFusion" in config.disabled_optimizers

        config.enable("matmul_activation_fusion")

        # Dependency should be auto-enabled
        assert "MatmulTransposeFusion" not in config.disabled_optimizers
        assert "MatMulActivationFusion" not in config.disabled_optimizers

    def test_dependencies_via_build_config(self) -> None:
        """Dependencies are also enabled when using build_config()."""
        config = ORTGraphPipe.build_config(bias_gelu_fusion=True)

        # Both BiasGeluFusion and its dependency GeluFusionL2 should be enabled
        assert "GeluFusionL2" not in config.disabled_optimizers
        assert "BiasGeluFusion" not in config.disabled_optimizers

    def test_chained_dependency_enabling(self) -> None:
        """Multiple capabilities with dependencies can be enabled together."""
        config = ORTGraphPipeConfig()
        config.enable("bias_gelu_fusion").enable("skip_layer_norm_fusion")

        # All should be enabled
        assert "GeluFusionL2" not in config.disabled_optimizers
        assert "BiasGeluFusion" not in config.disabled_optimizers
        assert "LayerNormFusionL2" not in config.disabled_optimizers
        assert "SkipLayerNormFusion" not in config.disabled_optimizers


# =============================================================================
# VERBOSE OUTPUT TESTS
# =============================================================================


class TestLogBuildConfigVerbose:
    """Tests for _log_build_config_verbose() method output."""

    def test_build_config_verbose_header(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verbose output includes proper header."""
        with caplog.at_level(logging.DEBUG):
            ORTGraphPipe._log_build_config_verbose(["gelu_fusion"], ["GeluFusionL2"])

        assert "ORTGraphPipe BUILD_CONFIG VERBOSE OUTPUT" in caplog.text
        assert "=" * 70 in caplog.text

    def test_build_config_verbose_enabled_capabilities(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verbose output shows enabled capabilities."""
        enabled = ["gelu_fusion", "matmul_add_fusion"]
        with caplog.at_level(logging.DEBUG):
            ORTGraphPipe._log_build_config_verbose(enabled, [])

        assert "[Enabled Capabilities] (2)" in caplog.text
        assert "[enabled] gelu_fusion" in caplog.text
        assert "[enabled] matmul_add_fusion" in caplog.text

    def test_build_config_verbose_enabled_overflow(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verbose output truncates enabled list at 10 items."""
        enabled = [f"cap_{i}" for i in range(15)]
        with caplog.at_level(logging.DEBUG):
            ORTGraphPipe._log_build_config_verbose(enabled, [])

        assert "[Enabled Capabilities] (15)" in caplog.text
        assert "[enabled] cap_0" in caplog.text
        assert "[enabled] cap_9" in caplog.text
        assert "... and 5 more" in caplog.text

    def test_build_config_verbose_empty_enabled(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verbose output handles empty enabled list."""
        with caplog.at_level(logging.DEBUG):
            ORTGraphPipe._log_build_config_verbose([], [])

        assert "[Enabled Capabilities] (0)" in caplog.text
        assert "(none)" in caplog.text

    def test_build_config_verbose_disabled_grouping(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verbose output groups disabled optimizers by type."""
        disabled = [
            "GeluFusion",
            "LayerNormFusion",
            "NhwcTransformer",
            "ConstantFolding",
        ]
        with caplog.at_level(logging.DEBUG):
            ORTGraphPipe._log_build_config_verbose([], disabled)

        assert "Fusions (2):" in caplog.text
        assert "[disabled] GeluFusion" in caplog.text
        assert "[disabled] LayerNormFusion" in caplog.text
        assert "Transformers (1):" in caplog.text
        assert "[disabled] NhwcTransformer" in caplog.text
        assert "Others (1):" in caplog.text
        assert "[disabled] ConstantFolding" in caplog.text

    def test_build_config_verbose_disabled_overflow(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verbose output truncates disabled lists properly."""
        # 15 fusions should show 10 + "... and 5 more"
        disabled = [f"Fusion{i}Fusion" for i in range(15)]
        with caplog.at_level(logging.DEBUG):
            ORTGraphPipe._log_build_config_verbose([], disabled)

        assert "Fusions (15):" in caplog.text
        assert "... and 5 more" in caplog.text

    def test_build_config_verbose_no_disabled(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verbose output handles empty disabled list."""
        with caplog.at_level(logging.DEBUG):
            ORTGraphPipe._log_build_config_verbose(["gelu_fusion"], [])

        assert "(none - all optimizers enabled)" in caplog.text


class TestLogProcessVerbose:
    """Tests for _log_process_verbose() method output."""

    def test_process_verbose_header(
        self, caplog: pytest.LogCaptureFixture, tmp_path: object
    ) -> None:
        """Verbose output includes proper header."""
        from onnx import TensorProto, helper

        # Create minimal model
        input_t = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
        output_t = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])
        node = helper.make_node("Relu", ["input"], ["output"])
        graph = helper.make_graph([node], "test", [input_t], [output_t])
        model = helper.make_model(graph)

        config = ORTGraphPipeConfig()
        pipe = ORTGraphPipe()
        input_file = tmp_path / "in.onnx"
        output_file = tmp_path / "out.onnx"

        with caplog.at_level(logging.DEBUG):
            pipe._log_process_verbose(config, model, input_file, output_file, "")

        assert "ORTGraphPipe PROCESS VERBOSE OUTPUT - ORT SESSION OPTIONS" in caplog.text
        assert "=" * 70 in caplog.text

    def test_process_verbose_model_info(
        self, caplog: pytest.LogCaptureFixture, tmp_path: object
    ) -> None:
        """Verbose output shows model info."""
        from onnx import TensorProto, helper

        input_t = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
        output_t = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])
        node = helper.make_node("Relu", ["input"], ["output"])
        graph = helper.make_graph([node], "test", [input_t], [output_t])
        model = helper.make_model(graph)

        config = ORTGraphPipeConfig()
        pipe = ORTGraphPipe()
        input_file = tmp_path / "in.onnx"
        output_file = tmp_path / "out.onnx"

        with caplog.at_level(logging.DEBUG):
            pipe._log_process_verbose(config, model, input_file, output_file, "")

        assert "[Input Model]" in caplog.text
        assert "Nodes: 1" in caplog.text
        assert "Inputs: 1" in caplog.text
        assert "Outputs: 1" in caplog.text

    def test_process_verbose_session_options(
        self, caplog: pytest.LogCaptureFixture, tmp_path: object
    ) -> None:
        """Verbose output shows ORT session options."""
        from onnx import TensorProto, helper

        input_t = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
        output_t = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])
        node = helper.make_node("Relu", ["input"], ["output"])
        graph = helper.make_graph([node], "test", [input_t], [output_t])
        model = helper.make_model(graph)

        config = ORTGraphPipeConfig()
        pipe = ORTGraphPipe()
        input_file = tmp_path / "in.onnx"
        output_file = tmp_path / "out.onnx"

        with caplog.at_level(logging.DEBUG):
            pipe._log_process_verbose(config, model, input_file, output_file, "")

        assert "[ORT SessionOptions]" in caplog.text
        assert "graph_optimization_level: 2" in caplog.text
        assert "CPUExecutionProvider" in caplog.text

    def test_process_verbose_disable_list(
        self, caplog: pytest.LogCaptureFixture, tmp_path: object
    ) -> None:
        """Verbose output shows disabled optimizers list."""
        from onnx import TensorProto, helper

        input_t = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
        output_t = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])
        node = helper.make_node("Relu", ["input"], ["output"])
        graph = helper.make_graph([node], "test", [input_t], [output_t])
        model = helper.make_model(graph)

        config = ORTGraphPipeConfig()
        pipe = ORTGraphPipe()
        input_file = tmp_path / "in.onnx"
        output_file = tmp_path / "out.onnx"

        disable_list = "GeluFusion;SkipLayerNormFusion;LayerNormFusion"
        with caplog.at_level(logging.DEBUG):
            pipe._log_process_verbose(config, model, input_file, output_file, disable_list)

        assert "[Session Config Entries]" in caplog.text
        assert "optimization.disable_specified_optimizers:" in caplog.text
        assert "- GeluFusion" in caplog.text
        assert "- SkipLayerNormFusion" in caplog.text
        assert "- LayerNormFusion" in caplog.text

    def test_process_verbose_disable_list_overflow(
        self, caplog: pytest.LogCaptureFixture, tmp_path: object
    ) -> None:
        """Verbose output truncates long disable lists."""
        from onnx import TensorProto, helper

        input_t = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
        output_t = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])
        node = helper.make_node("Relu", ["input"], ["output"])
        graph = helper.make_graph([node], "test", [input_t], [output_t])
        model = helper.make_model(graph)

        config = ORTGraphPipeConfig()
        pipe = ORTGraphPipe()
        input_file = tmp_path / "in.onnx"
        output_file = tmp_path / "out.onnx"

        # 12 items should show 8 + "... (12 total)"
        disable_list = ";".join([f"Optimizer{i}" for i in range(12)])
        with caplog.at_level(logging.DEBUG):
            pipe._log_process_verbose(config, model, input_file, output_file, disable_list)

        assert "- Optimizer0" in caplog.text
        assert "- Optimizer7" in caplog.text
        assert "(12 total)" in caplog.text

    def test_process_verbose_empty_disable_list(
        self, caplog: pytest.LogCaptureFixture, tmp_path: object
    ) -> None:
        """Verbose output handles empty disable list."""
        from onnx import TensorProto, helper

        input_t = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
        output_t = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])
        node = helper.make_node("Relu", ["input"], ["output"])
        graph = helper.make_graph([node], "test", [input_t], [output_t])
        model = helper.make_model(graph)

        config = ORTGraphPipeConfig()
        pipe = ORTGraphPipe()
        input_file = tmp_path / "in.onnx"
        output_file = tmp_path / "out.onnx"

        with caplog.at_level(logging.DEBUG):
            pipe._log_process_verbose(config, model, input_file, output_file, "")

        assert "(no disabled optimizers)" in caplog.text

    def test_process_verbose_ort_version(
        self, caplog: pytest.LogCaptureFixture, tmp_path: object
    ) -> None:
        """Verbose output shows ORT version."""
        from onnx import TensorProto, helper

        input_t = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 64])
        output_t = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64])
        node = helper.make_node("Relu", ["input"], ["output"])
        graph = helper.make_graph([node], "test", [input_t], [output_t])
        model = helper.make_model(graph)

        config = ORTGraphPipeConfig()
        pipe = ORTGraphPipe()
        input_file = tmp_path / "in.onnx"
        output_file = tmp_path / "out.onnx"

        with caplog.at_level(logging.DEBUG):
            pipe._log_process_verbose(config, model, input_file, output_file, "")

        assert "[ORT Runtime]" in caplog.text
        assert "Version:" in caplog.text
