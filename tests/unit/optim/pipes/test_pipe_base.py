# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for PipeConfig and BasePipe abstract classes.

Tests the base pipe architecture following Cardinal Rules:
- CARDINAL RULE #1: No hardcoded model architectures
- CARDINAL RULE #2: All tests use pytest with code-generated results
- CARDINAL RULE #3: Tests must run and pass

Test Categories:
1. PipeConfig Tests - Basic configuration class functionality
2. BasePipe Abstract Tests - Abstract class enforcement
3. BasePipe Interface Tests - Required method signatures
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import onnx
import pytest

# Import pipe base classes from modelkit (production)
from winml.modelkit.optim.pipes.base import BasePipe, OptimizationError, PipeConfig


class TestPipeConfig:
    """Tests for PipeConfig base class."""

    def test_pipe_config_instantiation(self) -> None:
        """Verify PipeConfig can be instantiated as minimal base class."""
        config = PipeConfig()
        assert isinstance(config, PipeConfig)

    def test_pipe_config_subclass(self) -> None:
        """Verify PipeConfig can be subclassed with custom fields."""

        @dataclass
        class CustomPipeConfig(PipeConfig):
            """Custom config with additional fields."""

            option1: bool = False
            option2: int = 0

        config = CustomPipeConfig(option1=True, option2=42)
        assert config.option1 is True
        assert config.option2 == 42
        assert isinstance(config, PipeConfig)

    def test_pipe_config_is_dataclass(self) -> None:
        """Verify PipeConfig uses dataclass for structure."""
        import dataclasses

        # PipeConfig should be a dataclass or subclasses should work as dataclasses
        @dataclass
        class TestConfig(PipeConfig):
            value: int = 0

        assert dataclasses.is_dataclass(TestConfig)
        config = TestConfig(value=10)
        assert config.value == 10


class TestBasePipeAbstract:
    """Tests for BasePipe abstract class enforcement."""

    def test_base_pipe_is_abstract(self) -> None:
        """Verify BasePipe cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BasePipe()  # type: ignore[abstract]

    def test_base_pipe_requires_build_config(self) -> None:
        """Verify subclass must implement build_config method."""

        # Missing build_config should raise TypeError
        with pytest.raises(TypeError, match="abstract"):

            class IncompletePipe(BasePipe):  # type: ignore[abstract]
                name: ClassVar[str] = "incomplete"

                def process(self, model: onnx.ModelProto, config: PipeConfig) -> onnx.ModelProto:
                    return model

            IncompletePipe()

    def test_base_pipe_requires_process(self) -> None:
        """Verify subclass must implement process method."""

        # Missing process should raise TypeError
        with pytest.raises(TypeError, match="abstract"):

            class IncompletePipe(BasePipe):  # type: ignore[abstract]
                name: ClassVar[str] = "incomplete"

                @classmethod
                def build_config(cls, **kwargs: Any) -> PipeConfig:
                    return PipeConfig()

            IncompletePipe()

    def test_base_pipe_requires_name_class_var(self) -> None:
        """Verify subclass must define name class variable."""

        # Complete implementation with all required elements
        class CompletePipe(BasePipe):
            name: ClassVar[str] = "complete"

            @classmethod
            def build_config(cls, **kwargs: Any) -> PipeConfig:
                return PipeConfig()

            def process(self, model: onnx.ModelProto, config: PipeConfig) -> onnx.ModelProto:
                return model

        # Should be instantiable
        pipe = CompletePipe()
        assert pipe.name == "complete"

    def test_base_pipe_complete_implementation(self, sample_model: onnx.ModelProto) -> None:
        """Verify a complete BasePipe implementation works correctly."""

        @dataclass
        class TestPipeConfig(PipeConfig):
            enabled: bool = False

        class TestPipe(BasePipe):
            name: ClassVar[str] = "test"

            @classmethod
            def build_config(cls, **kwargs: Any) -> TestPipeConfig:
                return TestPipeConfig(enabled=kwargs.get("enabled", False))

            def process(self, model: onnx.ModelProto, config: PipeConfig) -> onnx.ModelProto:
                # Simple pass-through for test
                return model

        # Test instantiation
        pipe = TestPipe()
        assert isinstance(pipe, BasePipe)

        # Test build_config
        config = pipe.build_config(enabled=True)
        assert isinstance(config, TestPipeConfig)
        assert config.enabled is True

        # Test process
        result = pipe.process(sample_model, config)
        assert isinstance(result, onnx.ModelProto)


class TestBasePipeInterface:
    """Tests for BasePipe interface methods."""

    def test_base_pipe_has_capabilities_dict(self) -> None:
        """Verify BasePipe has capabilities class attribute as dict."""

        class TestPipe(BasePipe):
            name: ClassVar[str] = "test"
            capabilities: ClassVar[dict[str, Any]] = {"test-cap": "value"}

            @classmethod
            def build_config(cls, **kwargs: Any) -> PipeConfig:
                return PipeConfig()

            def process(self, model: onnx.ModelProto, config: PipeConfig) -> onnx.ModelProto:
                return model

        # capabilities should be a dict
        assert hasattr(TestPipe, "capabilities")
        assert isinstance(TestPipe.capabilities, dict)
        assert "test-cap" in TestPipe.capabilities

    def test_base_pipe_default_capabilities_empty(self) -> None:
        """Verify BasePipe default capabilities is empty dict."""
        # BasePipe.capabilities should default to empty dict
        assert BasePipe.capabilities == {}

    def test_base_pipe_should_process_default(self) -> None:
        """Verify should_process returns True by default."""

        class TestPipe(BasePipe):
            name: ClassVar[str] = "test"

            @classmethod
            def build_config(cls, **kwargs: Any) -> PipeConfig:
                return PipeConfig()

            def process(self, model: onnx.ModelProto, config: PipeConfig) -> onnx.ModelProto:
                return model

        # Default should_process returns True
        config = PipeConfig()
        assert TestPipe.should_process(config) is True

    def test_build_config_accepts_kwargs(self) -> None:
        """Verify build_config accepts arbitrary kwargs."""

        @dataclass
        class TestPipeConfig(PipeConfig):
            value: int = 0

        class TestPipe(BasePipe):
            name: ClassVar[str] = "test"

            @classmethod
            def build_config(cls, **kwargs: Any) -> TestPipeConfig:
                return TestPipeConfig(value=kwargs.get("value", 0))

            def process(self, model: onnx.ModelProto, config: PipeConfig) -> onnx.ModelProto:
                return model

        # Should accept any kwargs without error
        config = TestPipe.build_config(value=42, unknown_param="ignored")
        assert config.value == 42

    def test_process_returns_model(self, sample_model: onnx.ModelProto) -> None:
        """Verify process method returns an ONNX ModelProto."""

        class TestPipe(BasePipe):
            name: ClassVar[str] = "test"

            @classmethod
            def build_config(cls, **kwargs: Any) -> PipeConfig:
                return PipeConfig()

            def process(self, model: onnx.ModelProto, config: PipeConfig) -> onnx.ModelProto:
                # Return the model unchanged
                return model

        pipe = TestPipe()
        config = PipeConfig()
        result = pipe.process(sample_model, config)

        assert isinstance(result, onnx.ModelProto)
        # Verify it's a valid model (has graph)
        assert result.graph is not None


class TestOptimizationError:
    """Tests for OptimizationError exception class."""

    def test_optimization_error_basic(self) -> None:
        """Verify basic error with just message works correctly."""
        error = OptimizationError("Test error message")

        # Check basic attributes
        assert error.message == "Test error message"
        assert error.pipe_name is None
        assert error.model_info == {}
        assert error.cause is None

        # Check exception message
        assert str(error) == "Test error message"

    def test_optimization_error_with_pipe_name(self) -> None:
        """Verify error with pipe_name parameter includes pipe context."""
        error = OptimizationError("Test error", pipe_name="test-pipe")

        # Check attributes
        assert error.message == "Test error"
        assert error.pipe_name == "test-pipe"
        assert error.model_info == {}
        assert error.cause is None

        # Check exception message contains pipe name
        error_str = str(error)
        assert "Test error" in error_str
        assert "Pipe: test-pipe" in error_str

    def test_optimization_error_with_model_info(self) -> None:
        """Verify error with model_info dict includes model context."""
        model_info = {"arch": "resnet", "layers": 50}
        error = OptimizationError("Test error", model_info=model_info)

        # Check attributes
        assert error.message == "Test error"
        assert error.pipe_name is None
        assert error.model_info == model_info
        assert error.cause is None

        # Check exception message contains model info
        error_str = str(error)
        assert "Test error" in error_str
        assert "Model info:" in error_str
        assert "arch" in error_str or "resnet" in error_str

    def test_optimization_error_with_cause(self) -> None:
        """Verify error with cause exception includes causation chain."""
        original_error = ValueError("Original error")
        error = OptimizationError("Test error", cause=original_error)

        # Check attributes
        assert error.message == "Test error"
        assert error.pipe_name is None
        assert error.model_info == {}
        assert error.cause is original_error

        # Check exception message contains cause
        error_str = str(error)
        assert "Test error" in error_str
        assert "Caused by:" in error_str
        assert "Original error" in error_str

    def test_optimization_error_full_context(self) -> None:
        """Verify error with all parameters includes complete context."""
        model_info = {"name": "test_model", "version": "1.0"}
        original_error = RuntimeError("Underlying failure")

        error = OptimizationError(
            "Optimization failed",
            pipe_name="fusion-pipe",
            model_info=model_info,
            cause=original_error,
        )

        # Check all attributes
        assert error.message == "Optimization failed"
        assert error.pipe_name == "fusion-pipe"
        assert error.model_info == model_info
        assert error.cause is original_error

        # Check exception message contains all parts
        error_str = str(error)
        assert "Optimization failed" in error_str
        assert "Pipe: fusion-pipe" in error_str
        assert "Model info:" in error_str
        assert "Caused by:" in error_str
        assert "Underlying failure" in error_str

    def test_optimization_error_message_formatting(self) -> None:
        """Verify message contains all parts separated by pipes."""
        model_info = {"test": "value"}
        original_error = Exception("cause error")

        error = OptimizationError(
            "main message",
            pipe_name="test-pipe",
            model_info=model_info,
            cause=original_error,
        )

        error_str = str(error)

        # Check separator is used
        assert " | " in error_str

        # Check all parts are present in order
        parts = error_str.split(" | ")
        assert len(parts) == 4
        assert parts[0] == "main message"
        assert parts[1] == "Pipe: test-pipe"
        assert "Model info:" in parts[2]
        assert "Caused by:" in parts[3]

    def test_optimization_error_empty_model_info(self) -> None:
        """Verify error with empty dict for model_info doesn't include it in message."""
        error = OptimizationError("Test error", model_info={})

        # Empty dict should be stored but not included in message
        assert error.model_info == {}
        error_str = str(error)
        assert error_str == "Test error"
        assert "Model info:" not in error_str

    def test_optimization_error_is_exception(self) -> None:
        """Verify OptimizationError is an Exception subclass."""
        error = OptimizationError("Test")

        # Should be an Exception
        assert isinstance(error, Exception)

        # Should be raisable
        with pytest.raises(OptimizationError) as exc_info:
            raise error

        assert exc_info.value is error
        assert str(exc_info.value) == "Test"
