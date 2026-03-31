# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for error handling paths in optim module.

This module focuses on covering defensive error paths that are not hit
during normal operation:
- OptimizationError with all optional fields
- ConfigurationError without errors list
- Optimizer exception handlers for validation/pipe failures
- Shape inference fallbacks

These tests use mocking to force error conditions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import onnx
import pytest
from onnx import TensorProto, helper

from winml.modelkit.optim.errors import (
    ConfigurationError,
    ModelValidationError,
    OptimizationError,
)
from winml.modelkit.optim.optimizer import Optimizer


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_model() -> onnx.ModelProto:
    """Create a minimal valid ONNX model for testing."""
    x_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 10])
    y_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])

    identity_node = helper.make_node("Identity", ["X"], ["Y"], name="identity")

    graph = helper.make_graph([identity_node], "test_graph", [x_input], [y_output])

    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


# =============================================================================
# TESTS: OptimizationError (errors.py lines 64-68, 76-83)
# =============================================================================


class TestOptimizationError:
    """Test OptimizationError with all optional fields."""

    def test_init_message_only(self) -> None:
        """Test with just message."""
        err = OptimizationError("Simple error")

        assert err.message == "Simple error"
        assert err.pipe_name is None
        assert err.model_info == {}
        assert err.cause is None
        assert str(err) == "Simple error"

    def test_init_with_pipe_name(self) -> None:
        """Test with pipe_name included."""
        err = OptimizationError("Pipe failed", pipe_name="ort_graph")

        assert err.message == "Pipe failed"
        assert err.pipe_name == "ort_graph"
        assert "Pipe: ort_graph" in str(err)

    def test_init_with_model_info(self) -> None:
        """Test with model_info included."""
        info = {"nodes": 150, "optimization_level": 2}
        err = OptimizationError("Error with info", model_info=info)

        assert err.model_info == info
        assert "Model info:" in str(err)
        assert "150" in str(err)

    def test_init_with_cause(self) -> None:
        """Test with cause exception included."""
        cause = ValueError("underlying error")
        err = OptimizationError("Wrapped error", cause=cause)

        assert err.cause is cause
        assert "Caused by:" in str(err)
        assert "underlying error" in str(err)

    def test_init_with_all_fields(self) -> None:
        """Test with all optional fields set."""
        cause = RuntimeError("ORT crash")
        info = {"nodes": 100}
        err = OptimizationError(
            "Complete error",
            pipe_name="fusion_pipe",
            model_info=info,
            cause=cause,
        )

        assert err.message == "Complete error"
        assert err.pipe_name == "fusion_pipe"
        assert err.model_info == info
        assert err.cause is cause

        msg = str(err)
        assert "Complete error" in msg
        assert "Pipe: fusion_pipe" in msg
        assert "Model info:" in msg
        assert "Caused by:" in msg

    def test_build_message_format(self) -> None:
        """Test _build_message uses pipe separator."""
        err = OptimizationError("msg", pipe_name="pipe1", cause=ValueError("x"))

        # Should use " | " as separator
        assert " | " in str(err)
        parts = str(err).split(" | ")
        assert len(parts) == 3  # message, pipe, cause


# =============================================================================
# TESTS: ConfigurationError (errors.py line 129)
# =============================================================================


class TestConfigurationErrorNoBullets:
    """Test ConfigurationError when no errors list provided."""

    def test_no_errors_list(self) -> None:
        """Test message without errors list (line 129)."""
        err = ConfigurationError("Config invalid")

        assert err.message == "Config invalid"
        assert err.errors == []
        # Should return just the message, no bullet list
        assert str(err) == "Config invalid"
        assert "- " not in str(err)

    def test_empty_errors_list(self) -> None:
        """Test with explicitly empty errors list."""
        err = ConfigurationError("No details", errors=[])

        assert str(err) == "No details"

    def test_with_errors_list(self) -> None:
        """Test with errors list (for comparison)."""
        err = ConfigurationError("Validation failed", errors=["error1", "error2"])

        msg = str(err)
        assert "Validation failed:" in msg
        assert "- error1" in msg
        assert "- error2" in msg


# =============================================================================
# TESTS: ModelValidationError (complete coverage)
# =============================================================================


class TestModelValidationError:
    """Test ModelValidationError with all fields."""

    def test_init_message_only(self) -> None:
        """Test with just message."""
        err = ModelValidationError("Invalid model")

        assert err.message == "Invalid model"
        assert err.model_path is None
        assert err.cause is None
        assert str(err) == "Invalid model"

    def test_init_with_path(self) -> None:
        """Test with model_path."""
        err = ModelValidationError("Bad model", model_path="/path/to/model.onnx")

        assert err.model_path == "/path/to/model.onnx"
        assert "Path: /path/to/model.onnx" in str(err)

    def test_init_with_cause(self) -> None:
        """Test with cause exception."""
        cause = onnx.checker.ValidationError("schema mismatch")
        err = ModelValidationError("Validation failed", cause=cause)

        assert err.cause is cause
        assert "Validation error:" in str(err)

    def test_init_with_all_fields(self) -> None:
        """Test with all fields."""
        cause = ValueError("bad tensor")
        err = ModelValidationError(
            "Complete validation error",
            model_path="model.onnx",
            cause=cause,
        )

        msg = str(err)
        assert "Complete validation error" in msg
        assert "Path: model.onnx" in msg
        assert "Validation error:" in msg


# =============================================================================
# TESTS: Optimizer error paths (optimizer.py)
# =============================================================================


class TestOptimizerInputValidationFailure:
    """Test that Optimizer no longer validates input (load_onnx handles it)."""

    def test_no_validation_in_optimizer(self, simple_model: onnx.ModelProto) -> None:
        """Test that Optimizer.optimize() does not call check_model at all.

        Validation is handled by load_onnx() on input and load_onnx() when
        the consumer loads the saved output. The optimizer itself does not
        validate (in-memory check_model fails on >2GiB models).
        """
        optimizer = Optimizer()

        # Mock _initialize_pipes to set empty pipes list
        def mock_init_pipes(cls: type) -> None:
            cls.pipes = []

        with (
            patch.object(Optimizer, "_initialize_pipes", mock_init_pipes),
            patch("onnx.checker.check_model") as mock_check,
            patch("winml.modelkit.onnx.shape.infer_shapes", return_value=simple_model),
        ):
            optimizer.optimize(simple_model)

            # check_model should not be called at all — validation is
            # handled by load_onnx (path-based, safe for any model size)
            mock_check.assert_not_called()


class TestOptimizerPipeFailure:
    """Test Optimizer when a pipe fails (lines 133-135)."""

    def test_pipe_processing_failure_raises(self, simple_model: onnx.ModelProto) -> None:
        """Test that pipe failure raises and logs error."""
        optimizer = Optimizer()

        # Mock the pipe to fail during process()
        mock_pipe_class = MagicMock()
        mock_pipe_instance = MagicMock()
        mock_pipe_class.return_value = mock_pipe_instance
        mock_pipe_class.name = "mock_pipe"

        # Configure mock pipe behavior
        mock_pipe_instance.build_config.return_value = {}
        mock_pipe_instance.should_process.return_value = True
        mock_pipe_instance.process.side_effect = RuntimeError("ORT optimization failed")

        with (
            patch("onnx.checker.check_model"),  # Skip validation
            patch.object(Optimizer, "pipes", [mock_pipe_class]),
            patch("winml.modelkit.onnx.shape.infer_shapes", return_value=simple_model),
            pytest.raises(RuntimeError, match="ORT optimization failed"),
        ):
            optimizer.optimize(simple_model)


class TestOptimizerNoPostValidation:
    """Test Optimizer does not perform post-optimization validation.

    Post-optimization check_model was removed because in-memory validation
    fails on >2GiB models and models with custom domains (com.microsoft).
    Validation is handled by load_onnx (path-based, safe for any size).
    """

    def test_no_post_validation_check(self, simple_model: onnx.ModelProto) -> None:
        """Optimizer.optimize() completes without calling check_model."""
        optimizer = Optimizer()

        def mock_init_pipes(cls: type) -> None:
            cls.pipes = []

        with (
            patch.object(Optimizer, "_initialize_pipes", mock_init_pipes),
            patch("onnx.checker.check_model") as mock_check,
            patch("winml.modelkit.onnx.shape.infer_shapes", return_value=simple_model),
        ):
            result = optimizer.optimize(simple_model)
            assert isinstance(result, onnx.ModelProto)
            mock_check.assert_not_called()


class TestOptimizerResolveDependencies:
    """Test _resolve_dependencies default value path (line 195)."""

    def test_resolve_dependencies_uses_defaults(self, simple_model: onnx.ModelProto) -> None:
        """Test that missing kwargs use capability defaults."""
        optimizer = Optimizer()

        # Call with empty kwargs - should use defaults for all capabilities
        result = optimizer._resolve_dependencies({})

        # Result should contain snake_case keys with default values
        # (the actual values depend on capability definitions)
        assert isinstance(result, dict)

    def test_resolve_dependencies_partial_kwargs(self, simple_model: onnx.ModelProto) -> None:
        """Test with some kwargs provided, others use defaults."""
        optimizer = Optimizer()

        # Provide only one capability
        result = optimizer._resolve_dependencies({"gelu_fusion": True})

        # Should have gelu_fusion=True and others at defaults
        assert result.get("gelu_fusion") is True


class TestShapeInferenceFallback:
    """Test shape inference fallback paths in modelkit.onnx.shape."""

    def test_symbolic_failure_falls_back_to_onnx(self, simple_model: onnx.ModelProto) -> None:
        """Test that symbolic failure falls back to ONNX shape inference."""
        from winml.modelkit.onnx.shape import infer_shapes

        with (
            patch(
                "onnxruntime.tools.symbolic_shape_infer.SymbolicShapeInference.infer_shapes",
                side_effect=RuntimeError("SymPy error"),
            ),
            patch(
                "onnx.shape_inference.infer_shapes",
                return_value=simple_model,
            ),
        ):
            result = infer_shapes(simple_model)
            assert result is simple_model

    def test_both_inference_failures_returns_original(self, simple_model: onnx.ModelProto) -> None:
        """Test that both failures return original model."""
        from winml.modelkit.onnx.shape import infer_shapes

        with (
            patch(
                "onnxruntime.tools.symbolic_shape_infer.SymbolicShapeInference.infer_shapes",
                side_effect=RuntimeError("SymPy error"),
            ),
            patch(
                "onnx.shape_inference.infer_shapes",
                side_effect=RuntimeError("ONNX error"),
            ),
        ):
            result = infer_shapes(simple_model)
            assert result is simple_model

    def test_symbolic_success_skips_onnx(self, simple_model: onnx.ModelProto) -> None:
        """Test that symbolic success skips ONNX inference."""
        from winml.modelkit.onnx.shape import infer_shapes

        mock_onnx = MagicMock()

        with (
            patch(
                "onnxruntime.tools.symbolic_shape_infer.SymbolicShapeInference.infer_shapes",
                return_value=simple_model,
            ),
            patch("winml.modelkit.onnx.shape.onnx.shape_inference.infer_shapes", mock_onnx),
        ):
            result = infer_shapes(simple_model)
            assert result is simple_model
            mock_onnx.assert_not_called()


class TestOptimizerPipeSkipping:
    """Test pipe skipping when should_process returns False."""

    def test_pipe_skipped_when_no_capabilities(self, simple_model: onnx.ModelProto) -> None:
        """Test that pipes are skipped when should_process returns False."""
        optimizer = Optimizer()

        mock_pipe_class = MagicMock()
        mock_pipe_instance = MagicMock()
        mock_pipe_class.return_value = mock_pipe_instance
        mock_pipe_class.name = "skipped_pipe"

        mock_pipe_instance.build_config.return_value = {}
        mock_pipe_instance.should_process.return_value = False
        # process should NOT be called

        with (
            patch("onnx.checker.check_model"),
            patch.object(Optimizer, "pipes", [mock_pipe_class]),
            patch("winml.modelkit.onnx.shape.infer_shapes", return_value=simple_model),
        ):
            optimizer.optimize(simple_model)

            # Verify process was NOT called
            mock_pipe_instance.process.assert_not_called()
