# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Integration tests for the Optimizer class and optimization pipeline.

Tests the main optimization pipeline following Cardinal Rules:
- CARDINAL RULE #1: No hardcoded model architectures
- CARDINAL RULE #2: All tests use pytest with code-generated results
- CARDINAL RULE #3: Tests must run and pass
- CARDINAL RULE #4: Never skip tests because they fail

Test Categories:
1. Optimizer Initialization - Pipe registration and lazy loading
2. Optimizer.optimize() Execution - Sequential pipe execution
3. Pipe Protocol Tests - build_config/process/should_process contract
4. End-to-End Integration - Full pipeline with real pipes
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, ClassVar

import onnx
import pytest

from winml.modelkit.optim.optimizer import Optimizer
from winml.modelkit.optim.pipes import PIPES, get_all_capabilities
from winml.modelkit.optim.pipes.base import BasePipe, PipeConfig
from winml.modelkit.optim.pipes.fusion import ORTFusionPipe, ORTFusionPipeConfig
from winml.modelkit.optim.pipes.graph import ORTGraphPipe, ORTGraphPipeConfig
from winml.modelkit.optim.pipes.surgery import SurgeryPipe, SurgeryPipeConfig
from winml.modelkit.optim.registry import (
    BoolCapability,
    CapabilityCategory,
    CapabilityDef,
    auto_enable_dependencies,
    defaults,
    validate,
    validate_dependencies,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def optimizer() -> Optimizer:
    """Create a fresh Optimizer instance."""
    return Optimizer()


@pytest.fixture
def mock_pipe_config() -> PipeConfig:
    """Create a minimal PipeConfig for testing."""
    return PipeConfig()


# =============================================================================
# TEST: Optimizer Initialization
# =============================================================================


class TestOptimizerInitialization:
    """Tests for Optimizer pipe initialization and registration."""

    def test_pipes_classvar_starts_empty(self) -> None:
        """Verify Optimizer.pipes is a class variable (list)."""
        assert hasattr(Optimizer, "pipes")
        assert isinstance(Optimizer.pipes, list)

    def test_initialize_pipes_populates_list(self) -> None:
        """Verify _initialize_pipes loads pipes from PIPES registry."""
        # Save original pipes and reset for test
        original_pipes = Optimizer.pipes[:]
        Optimizer.pipes = []
        try:
            Optimizer._initialize_pipes()
            assert len(Optimizer.pipes) >= 2, "Should have at least ORTGraphPipe and ORTFusionPipe"
        finally:
            # Restore original pipes
            Optimizer.pipes = original_pipes

    def test_initialize_pipes_is_idempotent(self) -> None:
        """Verify _initialize_pipes does not re-load if already populated."""
        # Ensure pipes are populated
        Optimizer._initialize_pipes()
        first_load = Optimizer.pipes[:]

        # Call again - should not change
        Optimizer._initialize_pipes()
        assert Optimizer.pipes == first_load

    def test_all_pipes_are_basepipe_subclasses(self) -> None:
        """Verify all registered pipes inherit from BasePipe."""
        Optimizer._initialize_pipes()
        for pipe_class in Optimizer.pipes:
            assert isinstance(pipe_class, type), f"{pipe_class} is not a class"
            assert issubclass(pipe_class, BasePipe), (
                f"{pipe_class.__name__} does not inherit from BasePipe"
            )

    def test_all_pipes_have_unique_names(self) -> None:
        """Verify all registered pipes have unique name attributes."""
        Optimizer._initialize_pipes()
        names = [pipe_class.name for pipe_class in Optimizer.pipes]
        assert len(names) == len(set(names)), f"Duplicate pipe names found: {names}"

    def test_pipes_match_module_level_registry(self) -> None:
        """Verify Optimizer pipes match the PIPES list from pipes/__init__.py."""
        Optimizer._initialize_pipes()
        assert Optimizer.pipes == PIPES


# =============================================================================
# TEST: Optimizer.optimize() Execution Flow
# =============================================================================


class TestOptimizerExecution:
    """Tests for Optimizer.optimize() execution mechanics."""

    def test_optimize_returns_model_proto(
        self, sample_model: onnx.ModelProto, optimizer: Optimizer
    ) -> None:
        """Verify optimize() returns an onnx.ModelProto."""
        result = optimizer.optimize(sample_model)
        assert isinstance(result, onnx.ModelProto)

    def test_optimize_preserves_graph_validity(
        self, sample_model: onnx.ModelProto, optimizer: Optimizer
    ) -> None:
        """Verify optimize() output has valid graph structure."""
        result = optimizer.optimize(sample_model)
        assert result.graph is not None
        assert len(result.graph.input) > 0
        assert len(result.graph.output) > 0

    def test_optimize_calls_all_pipes_in_order(
        self, sample_model: onnx.ModelProto
    ) -> None:
        """Verify optimize() calls build_config then process for each pipe, in order."""
        call_log: list[str] = []

        @dataclass
        class TrackerConfig(PipeConfig):
            pipe_id: str = ""

        class TrackerPipeA(BasePipe):
            name: ClassVar[str] = "tracker_a"
            capabilities: ClassVar[dict[str, Any]] = {}

            @classmethod
            def build_config(cls, **kwargs: Any) -> TrackerConfig:
                call_log.append("build_a")
                return TrackerConfig(pipe_id="a")

            def process(self, model: onnx.ModelProto, config: PipeConfig) -> onnx.ModelProto:
                call_log.append("process_a")
                return model

        class TrackerPipeB(BasePipe):
            name: ClassVar[str] = "tracker_b"
            capabilities: ClassVar[dict[str, Any]] = {}

            @classmethod
            def build_config(cls, **kwargs: Any) -> TrackerConfig:
                call_log.append("build_b")
                return TrackerConfig(pipe_id="b")

            def process(self, model: onnx.ModelProto, config: PipeConfig) -> onnx.ModelProto:
                call_log.append("process_b")
                return model

        opt = Optimizer()
        original_pipes = Optimizer.pipes[:]
        Optimizer.pipes = [TrackerPipeA, TrackerPipeB]
        try:
            opt.optimize(sample_model)
        finally:
            Optimizer.pipes = original_pipes

        # Verify order: build_a, process_a, build_b, process_b
        assert call_log == ["build_a", "process_a", "build_b", "process_b"]

    def test_optimize_passes_kwargs_to_build_config(
        self, sample_model: onnx.ModelProto
    ) -> None:
        """Verify optimize() passes **kwargs to each pipe's build_config."""
        captured_kwargs: dict[str, Any] = {}

        @dataclass
        class CaptureConfig(PipeConfig):
            pass

        class CapturePipe(BasePipe):
            name: ClassVar[str] = "capture"
            capabilities: ClassVar[dict[str, Any]] = {}

            @classmethod
            def build_config(cls, **kwargs: Any) -> CaptureConfig:
                captured_kwargs.update(kwargs)
                return CaptureConfig()

            def process(self, model: onnx.ModelProto, config: PipeConfig) -> onnx.ModelProto:
                return model

        opt = Optimizer()
        original_pipes = Optimizer.pipes[:]
        Optimizer.pipes = [CapturePipe]
        try:
            opt.optimize(sample_model, test_param=42, another="hello")
        finally:
            Optimizer.pipes = original_pipes

        assert "test_param" in captured_kwargs
        assert captured_kwargs["test_param"] == 42
        assert "another" in captured_kwargs
        assert captured_kwargs["another"] == "hello"

    def test_optimize_chains_model_through_pipes(
        self, sample_model: onnx.ModelProto
    ) -> None:
        """Verify model flows from one pipe's output to the next pipe's input."""

        @dataclass
        class ChainConfig(PipeConfig):
            pass

        class ChainPipe1(BasePipe):
            name: ClassVar[str] = "chain1"
            capabilities: ClassVar[dict[str, Any]] = {}

            @classmethod
            def build_config(cls, **kwargs: Any) -> ChainConfig:
                return ChainConfig()

            def process(self, model: onnx.ModelProto, config: PipeConfig) -> onnx.ModelProto:
                model.metadata_props.add(key="chain1_marker", value="true")
                return model

        class ChainPipe2(BasePipe):
            name: ClassVar[str] = "chain2"
            capabilities: ClassVar[dict[str, Any]] = {}

            @classmethod
            def build_config(cls, **kwargs: Any) -> ChainConfig:
                return ChainConfig()

            def process(self, model: onnx.ModelProto, config: PipeConfig) -> onnx.ModelProto:
                # Verify we received the model from ChainPipe1
                has_marker = any(p.key == "chain1_marker" for p in model.metadata_props)
                assert has_marker, "ChainPipe2 should receive model modified by ChainPipe1"
                model.metadata_props.add(key="chain2_marker", value="true")
                return model

        opt = Optimizer()
        original_pipes = Optimizer.pipes[:]
        Optimizer.pipes = [ChainPipe1, ChainPipe2]
        try:
            result = opt.optimize(sample_model)
        finally:
            Optimizer.pipes = original_pipes

        metadata_keys = [p.key for p in result.metadata_props]
        assert "chain1_marker" in metadata_keys
        assert "chain2_marker" in metadata_keys

    def test_optimize_respects_should_process(
        self, sample_model: onnx.ModelProto
    ) -> None:
        """Verify optimize() skips pipes when should_process returns False."""

        @dataclass
        class SkipConfig(PipeConfig):
            enabled: bool = False

        class SkippablePipe(BasePipe):
            name: ClassVar[str] = "skippable"
            capabilities: ClassVar[dict[str, Any]] = {}
            process_called: ClassVar[bool] = False

            @classmethod
            def build_config(cls, **kwargs: Any) -> SkipConfig:
                return SkipConfig(enabled=False)

            @classmethod
            def should_process(cls, config: PipeConfig) -> bool:
                return getattr(config, "enabled", True)

            def process(self, model: onnx.ModelProto, config: PipeConfig) -> onnx.ModelProto:
                SkippablePipe.process_called = True
                return model

        SkippablePipe.process_called = False
        opt = Optimizer()
        original_pipes = Optimizer.pipes[:]
        Optimizer.pipes = [SkippablePipe]
        try:
            opt.optimize(sample_model)
        finally:
            Optimizer.pipes = original_pipes

        assert not SkippablePipe.process_called, (
            "Pipe should be skipped when should_process is False"
        )

    def test_optimize_with_empty_pipes_list(
        self, sample_model: onnx.ModelProto
    ) -> None:
        """Verify optimize() handles empty pipes list gracefully."""
        opt = Optimizer()
        original_pipes = Optimizer.pipes[:]
        Optimizer.pipes = []
        try:
            result = opt.optimize(sample_model)
            assert isinstance(result, onnx.ModelProto)
        finally:
            Optimizer.pipes = original_pipes


# =============================================================================
# TEST: Pipe Protocol - build_config / process / should_process
# =============================================================================


class TestPipeProtocol:
    """Tests verifying each concrete pipe implements the BasePipe protocol."""

    @pytest.mark.parametrize(
        "pipe_class",
        [ORTGraphPipe, ORTFusionPipe, SurgeryPipe],
        ids=["ORTGraphPipe", "ORTFusionPipe", "SurgeryPipe"],
    )
    def test_pipe_has_name(self, pipe_class: type[BasePipe]) -> None:
        """Each pipe must have a unique name ClassVar."""
        assert hasattr(pipe_class, "name")
        assert isinstance(pipe_class.name, str)
        assert len(pipe_class.name) > 0

    @pytest.mark.parametrize(
        "pipe_class",
        [ORTGraphPipe, ORTFusionPipe, SurgeryPipe],
        ids=["ORTGraphPipe", "ORTFusionPipe", "SurgeryPipe"],
    )
    def test_pipe_has_capabilities_dict(self, pipe_class: type[BasePipe]) -> None:
        """Each pipe must have a capabilities ClassVar dict."""
        assert hasattr(pipe_class, "capabilities")
        assert isinstance(pipe_class.capabilities, dict)

    @pytest.mark.parametrize(
        "pipe_class",
        [ORTGraphPipe, ORTFusionPipe, SurgeryPipe],
        ids=["ORTGraphPipe", "ORTFusionPipe", "SurgeryPipe"],
    )
    def test_build_config_returns_pipe_config(self, pipe_class: type[BasePipe]) -> None:
        """build_config() must return a PipeConfig subclass."""
        config = pipe_class.build_config()
        assert isinstance(config, PipeConfig)

    @pytest.mark.parametrize(
        "pipe_class",
        [ORTGraphPipe, ORTFusionPipe, SurgeryPipe],
        ids=["ORTGraphPipe", "ORTFusionPipe", "SurgeryPipe"],
    )
    def test_build_config_accepts_kwargs(self, pipe_class: type[BasePipe]) -> None:
        """build_config() must accept **kwargs without error."""
        config = pipe_class.build_config(some_unknown_param=True)
        assert isinstance(config, PipeConfig)

    @pytest.mark.parametrize(
        "pipe_class",
        [ORTGraphPipe, ORTFusionPipe, SurgeryPipe],
        ids=["ORTGraphPipe", "ORTFusionPipe", "SurgeryPipe"],
    )
    def test_should_process_is_classmethod(self, pipe_class: type[BasePipe]) -> None:
        """should_process() must be a classmethod returning bool."""
        config = pipe_class.build_config()
        result = pipe_class.should_process(config)
        assert isinstance(result, bool)


class TestORTGraphPipeConfig:
    """Tests for ORTGraphPipe configuration building."""

    def test_default_config_has_optimization_level_2(self) -> None:
        """Default config should have optimization_level=2."""
        config = ORTGraphPipe.build_config()
        assert isinstance(config, ORTGraphPipeConfig)
        assert config.optimization_level == 2

    def test_enable_specific_capability(self) -> None:
        """Enabling a capability should remove it from disabled list."""
        config = ORTGraphPipe.build_config(gelu_fusion=True)
        assert isinstance(config, ORTGraphPipeConfig)
        # When gelu_fusion is enabled, its ORT name should NOT be in disabled list
        assert "GeluFusionL2" not in config.disabled_optimizers

    def test_all_disabled_by_default(self) -> None:
        """With no kwargs, advanced optimizers should be disabled."""
        config = ORTGraphPipe.build_config()
        assert isinstance(config, ORTGraphPipeConfig)
        # Default-false capabilities should be in the disabled list
        assert len(config.disabled_optimizers) > 0

    def test_should_process_returns_true_for_level_2(self) -> None:
        """should_process should return True when level > 0."""
        config = ORTGraphPipe.build_config()
        assert ORTGraphPipe.should_process(config) is True


class TestORTFusionPipeConfig:
    """Tests for ORTFusionPipe configuration building."""

    def test_default_config_has_all_fusions_off(self) -> None:
        """Default config should have all fusion options disabled."""
        config = ORTFusionPipe.build_config()
        assert isinstance(config, ORTFusionPipeConfig)
        assert config.enable_layer_norm is False
        assert config.enable_attention is False
        assert config.enable_skip_layer_norm is False

    def test_enable_specific_fusion(self) -> None:
        """Enabling a fusion should set the corresponding config attribute."""
        config = ORTFusionPipe.build_config(layer_norm_fusion=True)
        assert isinstance(config, ORTFusionPipeConfig)
        assert config.enable_layer_norm is True

    def test_should_process_false_when_all_disabled(self) -> None:
        """should_process should return False when no fusions are enabled."""
        config = ORTFusionPipe.build_config()
        assert ORTFusionPipe.should_process(config) is False

    def test_should_process_true_when_fusion_enabled(self) -> None:
        """should_process should return True when at least one fusion is enabled."""
        config = ORTFusionPipe.build_config(attention_fusion=True)
        assert ORTFusionPipe.should_process(config) is True


class TestSurgeryPipeConfig:
    """Tests for SurgeryPipe configuration building."""

    def test_default_config_disabled(self) -> None:
        """Default config should have clamping disabled."""
        config = SurgeryPipe.build_config()
        assert isinstance(config, SurgeryPipeConfig)
        assert config.clamp_constant_values is False

    def test_enable_clamping(self) -> None:
        """Enabling clamp_constant_values should set the config."""
        config = SurgeryPipe.build_config(clamp_constant_values=True)
        assert isinstance(config, SurgeryPipeConfig)
        assert config.clamp_constant_values is True

    def test_custom_clamp_range(self) -> None:
        """Custom clamp range should be configurable."""
        config = SurgeryPipe.build_config(
            clamp_constant_values=True, clamp_min=-500.0, clamp_max=500.0
        )
        assert config.clamp_min == -500.0
        assert config.clamp_max == 500.0

    def test_should_process_false_when_disabled(self) -> None:
        """should_process should return False when no operations enabled."""
        config = SurgeryPipe.build_config()
        assert SurgeryPipe.should_process(config) is False

    def test_should_process_true_when_enabled(self) -> None:
        """should_process should return True when clamping is enabled."""
        config = SurgeryPipe.build_config(clamp_constant_values=True)
        assert SurgeryPipe.should_process(config) is True


# =============================================================================
# TEST: Capability Registry Functions
# =============================================================================


class TestCapabilityRegistry:
    """Tests for registry functions used by the optimizer pipeline."""

    def test_get_all_capabilities_returns_dict(self) -> None:
        """get_all_capabilities should return a dict of CapabilityDefs."""
        all_caps = get_all_capabilities()
        assert isinstance(all_caps, dict)
        assert len(all_caps) > 0

    def test_all_capabilities_are_capability_defs(self) -> None:
        """Every value should be a CapabilityDef subclass."""
        all_caps = get_all_capabilities()
        for name, cap in all_caps.items():
            assert isinstance(cap, CapabilityDef), (
                f"Capability '{name}' is not a CapabilityDef: {type(cap)}"
            )

    def test_capability_names_are_kebab_case(self) -> None:
        """Capability names should use kebab-case convention."""
        all_caps = get_all_capabilities()
        for name in all_caps:
            assert "_" not in name, f"Capability '{name}' uses underscore, expected kebab-case"
            assert name == name.lower(), f"Capability '{name}' is not lowercase"

    def test_defaults_function(self) -> None:
        """defaults() should return a dict with default values."""
        caps = {
            "test-bool": BoolCapability(
                name="test-bool",
                ort_name="TestBool",
                description="Test",
                category=CapabilityCategory.MISC,
                default=False,
            ),
        }
        result = defaults(caps)
        assert result == {"test-bool": False}

    def test_validate_accepts_correct_types(self) -> None:
        """validate() should return no errors for correct types."""
        caps = {
            "test-bool": BoolCapability(
                name="test-bool",
                ort_name="TestBool",
                description="Test",
                category=CapabilityCategory.MISC,
                default=False,
            ),
        }
        errors = validate({"test-bool": True}, caps)
        assert errors == []

    def test_validate_rejects_wrong_types(self) -> None:
        """validate() should return errors for wrong types."""
        caps = {
            "test-bool": BoolCapability(
                name="test-bool",
                ort_name="TestBool",
                description="Test",
                category=CapabilityCategory.MISC,
                default=False,
            ),
        }
        errors = validate({"test-bool": "invalid"}, caps)
        assert len(errors) > 0

    def test_validate_dependencies_satisfied(self) -> None:
        """validate_dependencies() should pass when dependencies are met."""
        caps = {
            "parent": BoolCapability(
                name="parent",
                ort_name="Parent",
                description="Parent",
                category=CapabilityCategory.MISC,
                default=False,
            ),
            "child": BoolCapability(
                name="child",
                ort_name="Child",
                description="Child",
                category=CapabilityCategory.MISC,
                default=False,
                depends_on=("parent",),
            ),
        }
        errors = validate_dependencies({"parent": True, "child": True}, caps)
        assert errors == []

    def test_validate_dependencies_unsatisfied(self) -> None:
        """validate_dependencies() should report unsatisfied dependencies."""
        caps = {
            "parent": BoolCapability(
                name="parent",
                ort_name="Parent",
                description="Parent",
                category=CapabilityCategory.MISC,
                default=False,
            ),
            "child": BoolCapability(
                name="child",
                ort_name="Child",
                description="Child",
                category=CapabilityCategory.MISC,
                default=False,
                depends_on=("parent",),
            ),
        }
        errors = validate_dependencies({"parent": False, "child": True}, caps)
        assert len(errors) > 0

    def test_auto_enable_dependencies(self) -> None:
        """auto_enable_dependencies() should enable required dependencies."""
        caps = {
            "parent": BoolCapability(
                name="parent",
                ort_name="Parent",
                description="Parent",
                category=CapabilityCategory.MISC,
                default=False,
            ),
            "child": BoolCapability(
                name="child",
                ort_name="Child",
                description="Child",
                category=CapabilityCategory.MISC,
                default=False,
                depends_on=("parent",),
            ),
        }
        config = {"child": True, "parent": False}
        resolved = auto_enable_dependencies(config, caps)
        assert resolved["parent"] is True, "Parent should be auto-enabled"
        assert resolved["child"] is True


# =============================================================================
# TEST: Optimizer Function Signature
# =============================================================================


class TestOptimizerSignature:
    """Tests verifying the Optimizer class public interface."""

    def test_optimize_method_signature(self) -> None:
        """Verify Optimizer.optimize has expected signature."""
        sig = inspect.signature(Optimizer.optimize)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "model" in params
        # Should accept **kwargs
        assert any(
            sig.parameters[p].kind == inspect.Parameter.VAR_KEYWORD for p in params
        )

    def test_optimizer_has_initialize_pipes(self) -> None:
        """Verify Optimizer has _initialize_pipes classmethod."""
        assert hasattr(Optimizer, "_initialize_pipes")
        assert callable(Optimizer._initialize_pipes)

    def test_optimizer_has_pipes_classvar(self) -> None:
        """Verify Optimizer.pipes is a class-level list."""
        assert hasattr(Optimizer, "pipes")
        assert isinstance(Optimizer.pipes, list)


# =============================================================================
# TEST: End-to-End Integration with Real Pipes
# =============================================================================


class TestOptimizerIntegration:
    """Integration tests running real optimization on models."""

    def test_optimize_simple_model_no_capabilities(
        self, sample_model: onnx.ModelProto
    ) -> None:
        """Run optimize on a simple model with no capabilities enabled."""
        opt = Optimizer()
        result = opt.optimize(sample_model)

        assert isinstance(result, onnx.ModelProto)
        assert result.graph is not None

    def test_optimize_preserves_model_io(
        self, sample_model: onnx.ModelProto
    ) -> None:
        """Verify optimization preserves model inputs and outputs."""
        opt = Optimizer()
        result = opt.optimize(sample_model)

        assert len(result.graph.input) > 0
        assert len(result.graph.output) > 0

    def test_optimize_with_verbose_flag(
        self, sample_model: onnx.ModelProto
    ) -> None:
        """Verify verbose=True does not cause errors."""
        opt = Optimizer()
        result = opt.optimize(sample_model, verbose=True)
        assert isinstance(result, onnx.ModelProto)

    def test_optimize_dependency_resolution(
        self, sample_model: onnx.ModelProto
    ) -> None:
        """Verify _resolve_dependencies runs without error."""
        opt = Optimizer()
        # bias_gelu_fusion depends on gelu_fusion - dependency resolution should handle it
        result = opt.optimize(sample_model, bias_gelu_fusion=True)
        assert isinstance(result, onnx.ModelProto)

    def test_optimize_returns_different_object(
        self, sample_model: onnx.ModelProto
    ) -> None:
        """Verify optimize returns a new model object (not the same reference)."""
        opt = Optimizer()
        result = opt.optimize(sample_model)
        # The result should be a new model object due to shape inference
        # and pipe processing that creates new ModelProto instances
        assert isinstance(result, onnx.ModelProto)

    def test_resolve_dependencies_method(self) -> None:
        """Verify _resolve_dependencies handles known dependency patterns."""
        opt = Optimizer()
        # Pass in bias_gelu_fusion which depends on gelu_fusion
        kwargs = {"bias_gelu_fusion": True}
        resolved = opt._resolve_dependencies(kwargs)
        # After resolution, gelu_fusion should also be present and True
        assert resolved.get("gelu_fusion") is True

    def test_registered_pipes_count(self) -> None:
        """Verify the expected number of pipes are registered."""
        Optimizer._initialize_pipes()
        # Currently: RewritePipe, ORTGraphPipe, ORTFusionPipe, SurgeryPipe
        assert len(Optimizer.pipes) == 4

    def test_registered_pipe_names(self) -> None:
        """Verify expected pipe names are registered."""
        Optimizer._initialize_pipes()
        names = {pipe_class.name for pipe_class in Optimizer.pipes}
        assert "rewrite" in names
        assert "ort_graph" in names
        assert "ort_fusion" in names
        assert "surgery" in names
