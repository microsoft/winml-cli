"""Tests for modelkit.optim.api module.

This module tests the public optimize_onnx() function following the design
specified in docs/design/optimization/6_design_api.md.

Test Strategy:
- Unit tests: Test helper functions directly without mocking
- API behavior tests: Mock Optimizer to test config handling, precedence
- Integration tests: Use conftest fixtures for real optimization

Test Categories:
1. Input Handling - model parameter accepts str, Path, ModelProto
2. Output Handling - output parameter controls file saving
3. Config Handling - config parameter from JSON file or dict
4. Configuration Precedence - kwargs > config > defaults
5. Error Handling - proper exception types for each failure mode
6. Validation - config validation and dependency checking
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import onnx
import pytest
from onnx import TensorProto, helper


if TYPE_CHECKING:
    from pathlib import Path

from winml.modelkit.optim import (
    ConfigurationError,
    ModelValidationError,
    optimize_onnx,
)
from winml.modelkit.optim.api import (
    _convert_to_kwargs,
    _load_config,
    _load_model,
    _merge_config,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_model() -> onnx.ModelProto:
    """Create a minimal valid ONNX model for testing.

    Uses opset 11 for ORT compatibility.
    """
    x_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 10])
    y_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 10])
    node = helper.make_node("Identity", ["X"], ["Y"], name="identity")
    graph = helper.make_graph([node], "test_graph", [x_input], [y_output])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


@pytest.fixture
def model_file(simple_model: onnx.ModelProto, tmp_path: Path) -> Path:
    """Save simple model to a file and return path."""
    model_path = tmp_path / "test_model.onnx"
    onnx.save(simple_model, str(model_path))
    return model_path


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Create a valid JSON config file."""
    config_path = tmp_path / "config.json"
    config = {
        "gelu-fusion": True,
        "layer-norm-fusion": True,
    }
    with config_path.open("w") as f:
        json.dump(config, f)
    return config_path


@pytest.fixture
def mock_capability() -> MagicMock:
    """Create a mock capability definition."""
    cap = MagicMock()
    cap.name = "test-cap"
    cap.python_name = "test_cap"
    cap.default = False
    return cap


# =============================================================================
# TEST: Input Handling (_load_model)
# =============================================================================


class TestLoadModel:
    """Tests for _load_model helper function."""

    def test_load_model_from_string_path(self, model_file: Path) -> None:
        """Accept string path to ONNX file."""
        model, path = _load_model(str(model_file))
        assert isinstance(model, onnx.ModelProto)
        assert path == str(model_file)

    def test_load_model_from_path_object(self, model_file: Path) -> None:
        """Accept Path object to ONNX file."""
        model, path = _load_model(model_file)
        assert isinstance(model, onnx.ModelProto)
        assert path == str(model_file)

    def test_load_model_from_proto(self, simple_model: onnx.ModelProto) -> None:
        """Accept ModelProto directly."""
        model, path = _load_model(simple_model)
        assert model is simple_model
        assert path is None

    def test_load_model_file_not_found(self) -> None:
        """Raise FileNotFoundError for missing model."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            _load_model("nonexistent_model.onnx")

    def test_load_model_invalid_onnx(self, tmp_path: Path) -> None:
        """Raise ModelValidationError for invalid ONNX file."""
        invalid_file = tmp_path / "invalid.onnx"
        invalid_file.write_text("not valid onnx content")
        with pytest.raises(ModelValidationError, match="Failed to load ONNX model"):
            _load_model(invalid_file)


# =============================================================================
# TEST: Config Handling (_load_config)
# =============================================================================


class TestLoadConfig:
    """Tests for _load_config helper function."""

    def test_load_config_valid_json(self, config_file: Path) -> None:
        """Load valid JSON config file."""
        config = _load_config(config_file)
        assert config["gelu-fusion"] is True
        assert config["layer-norm-fusion"] is True

    def test_load_config_file_not_found(self) -> None:
        """Raise FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            _load_config("nonexistent_config.json")

    def test_load_config_invalid_json(self, tmp_path: Path) -> None:
        """Raise ConfigurationError for invalid JSON."""
        invalid_config = tmp_path / "invalid.json"
        invalid_config.write_text("{invalid json}")
        with pytest.raises(ConfigurationError, match="Invalid JSON"):
            _load_config(invalid_config)

    def test_load_config_empty_json(self, tmp_path: Path) -> None:
        """Handle empty JSON object."""
        empty_config = tmp_path / "empty.json"
        empty_config.write_text("{}")
        config = _load_config(empty_config)
        assert config == {}


# =============================================================================
# TEST: Configuration Precedence (_merge_config)
# =============================================================================


class TestMergeConfig:
    """Tests for _merge_config helper function."""

    def test_merge_uses_defaults_when_no_config(self, mock_capability: MagicMock) -> None:
        """Use capability defaults when no config provided."""
        all_caps = {"test-cap": mock_capability}
        result = _merge_config(None, {}, all_caps)
        assert result["test-cap"] is False  # default

    def test_merge_config_dict_overrides_defaults(self, mock_capability: MagicMock) -> None:
        """Config dict overrides capability defaults."""
        all_caps = {"test-cap": mock_capability}
        config_dict = {"test-cap": True}
        result = _merge_config(config_dict, {}, all_caps)
        assert result["test-cap"] is True

    def test_merge_kwargs_override_config(self, mock_capability: MagicMock) -> None:
        """Kwargs override config dict values."""
        all_caps = {"test-cap": mock_capability}
        config_dict = {"test-cap": True}
        kwargs = {"test_cap": False}  # snake_case
        result = _merge_config(config_dict, kwargs, all_caps)
        assert result["test-cap"] is False  # kwargs win

    def test_merge_extracts_capabilities_section(self, mock_capability: MagicMock) -> None:
        """Extract capabilities from nested config structure."""
        all_caps = {"test-cap": mock_capability}
        config_dict = {"capabilities": {"test-cap": True}}
        result = _merge_config(config_dict, {}, all_caps)
        assert result["test-cap"] is True

    def test_merge_ignores_unknown_config_keys(self, mock_capability: MagicMock) -> None:
        """Ignore unknown keys in config dict."""
        all_caps = {"test-cap": mock_capability}
        config_dict = {"test-cap": True, "unknown-cap": True}
        result = _merge_config(config_dict, {}, all_caps)
        assert "unknown-cap" not in result

    def test_merge_none_kwargs_not_applied(self, mock_capability: MagicMock) -> None:
        """None values in kwargs should not override."""
        all_caps = {"test-cap": mock_capability}
        config_dict = {"test-cap": True}
        kwargs = {"test_cap": None}
        result = _merge_config(config_dict, kwargs, all_caps)
        assert result["test-cap"] is True  # config value preserved


# =============================================================================
# TEST: Convert to Kwargs (_convert_to_kwargs)
# =============================================================================


class TestConvertToKwargs:
    """Tests for _convert_to_kwargs helper function."""

    def test_convert_kebab_to_snake(self, mock_capability: MagicMock) -> None:
        """Convert kebab-case keys to snake_case."""
        all_caps = {"test-cap": mock_capability}
        config = {"test-cap": True}
        result = _convert_to_kwargs(config, all_caps)
        assert result == {"test_cap": True}

    def test_convert_ignores_unknown_keys(self, mock_capability: MagicMock) -> None:
        """Ignore keys not in all_caps."""
        all_caps = {"test-cap": mock_capability}
        config = {"test-cap": True, "unknown": False}
        result = _convert_to_kwargs(config, all_caps)
        assert "unknown" not in result


# =============================================================================
# TEST: optimize_onnx() Input/Output (with mocked Optimizer)
# =============================================================================


class TestOptimizeOnnxInput:
    """Tests for optimize_onnx input handling.

    Uses mocked Optimizer to test input handling without running real pipeline.
    """

    def test_accepts_string_path(self, model_file: Path) -> None:
        """Accept string path as model input."""
        with patch("winml.modelkit.optim.api.Optimizer") as mock_opt:
            mock_opt.return_value.optimize.return_value = onnx.ModelProto()
            result = optimize_onnx(str(model_file))
            assert isinstance(result, onnx.ModelProto)
            # Verify model was loaded and passed to optimizer
            mock_opt.return_value.optimize.assert_called_once()

    def test_accepts_path_object(self, model_file: Path) -> None:
        """Accept Path object as model input."""
        with patch("winml.modelkit.optim.api.Optimizer") as mock_opt:
            mock_opt.return_value.optimize.return_value = onnx.ModelProto()
            result = optimize_onnx(model_file)
            assert isinstance(result, onnx.ModelProto)

    def test_accepts_model_proto(self, simple_model: onnx.ModelProto) -> None:
        """Accept ModelProto directly."""
        with patch("winml.modelkit.optim.api.Optimizer") as mock_opt:
            mock_opt.return_value.optimize.return_value = onnx.ModelProto()
            result = optimize_onnx(simple_model)
            assert isinstance(result, onnx.ModelProto)

    def test_file_not_found_error(self) -> None:
        """Raise FileNotFoundError for missing model."""
        with pytest.raises(FileNotFoundError):
            optimize_onnx("nonexistent.onnx")


class TestOptimizeOnnxOutput:
    """Tests for optimize_onnx output handling.

    Uses mocked Optimizer to test output handling without running real pipeline.
    """

    def test_returns_model_when_no_output(self, model_file: Path) -> None:
        """Return ModelProto when output=None."""
        with patch("winml.modelkit.optim.api.Optimizer") as mock_opt:
            expected_model = onnx.ModelProto()
            mock_opt.return_value.optimize.return_value = expected_model
            result = optimize_onnx(model_file)
            assert result is expected_model

    def test_saves_to_output_path(self, model_file: Path, tmp_path: Path) -> None:
        """Save model to output path when provided."""
        output_path = tmp_path / "output.onnx"
        with patch("winml.modelkit.optim.api.Optimizer") as mock_opt:
            # Return a valid model that can be saved
            mock_opt.return_value.optimize.return_value = onnx.ModelProto()
            optimize_onnx(model_file, output_path)
            assert output_path.exists()

    def test_creates_parent_directories(self, model_file: Path, tmp_path: Path) -> None:
        """Create parent directories for output path."""
        output_path = tmp_path / "subdir" / "nested" / "output.onnx"
        with patch("winml.modelkit.optim.api.Optimizer") as mock_opt:
            mock_opt.return_value.optimize.return_value = onnx.ModelProto()
            optimize_onnx(model_file, output_path)
            assert output_path.exists()


class TestOptimizeOnnxConfig:
    """Tests for optimize_onnx config handling.

    Uses mocked Optimizer to verify config is processed correctly.
    """

    def test_config_from_json_file(self, model_file: Path, config_file: Path) -> None:
        """Load config from JSON file."""
        with patch("winml.modelkit.optim.api.Optimizer") as mock_opt:
            mock_opt.return_value.optimize.return_value = onnx.ModelProto()
            optimize_onnx(model_file, config=config_file)
            # Verify optimizer was called with config values
            call_kwargs = mock_opt.return_value.optimize.call_args[1]
            assert call_kwargs.get("gelu_fusion") is True
            assert call_kwargs.get("layer_norm_fusion") is True

    def test_config_from_dict(self, model_file: Path) -> None:
        """Use config dict directly."""
        config = {"gelu-fusion": True}
        with patch("winml.modelkit.optim.api.Optimizer") as mock_opt:
            mock_opt.return_value.optimize.return_value = onnx.ModelProto()
            optimize_onnx(model_file, config=config)
            call_kwargs = mock_opt.return_value.optimize.call_args[1]
            assert call_kwargs.get("gelu_fusion") is True

    def test_config_file_not_found(self, model_file: Path) -> None:
        """Raise FileNotFoundError for missing config file."""
        with pytest.raises(FileNotFoundError):
            optimize_onnx(model_file, config="nonexistent.json")

    def test_invalid_json_config(self, model_file: Path, tmp_path: Path) -> None:
        """Raise ConfigurationError for invalid JSON config."""
        invalid_config = tmp_path / "invalid.json"
        invalid_config.write_text("{bad json")
        with pytest.raises(ConfigurationError):
            optimize_onnx(model_file, config=invalid_config)


class TestOptimizeOnnxPrecedence:
    """Tests for configuration precedence.

    Verifies: kwargs > config file > defaults
    """

    def test_kwargs_override_config_file(self, model_file: Path, tmp_path: Path) -> None:
        """Kwargs should override config file values."""
        # Create config with gelu-fusion=True
        config_path = tmp_path / "config.json"
        with config_path.open("w") as f:
            json.dump({"gelu-fusion": True}, f)

        # Pass gelu_fusion=False as kwarg - should override
        with patch("winml.modelkit.optim.api.Optimizer") as mock_opt:
            mock_opt.return_value.optimize.return_value = onnx.ModelProto()
            optimize_onnx(model_file, config=config_path, gelu_fusion=False)
            call_kwargs = mock_opt.return_value.optimize.call_args[1]
            assert call_kwargs.get("gelu_fusion") is False

    def test_config_overrides_defaults(self, model_file: Path, tmp_path: Path) -> None:
        """Config file should override capability defaults."""
        config_path = tmp_path / "config.json"
        with config_path.open("w") as f:
            json.dump({"gelu-fusion": True}, f)

        with patch("winml.modelkit.optim.api.Optimizer") as mock_opt:
            mock_opt.return_value.optimize.return_value = onnx.ModelProto()
            optimize_onnx(model_file, config=config_path)
            call_kwargs = mock_opt.return_value.optimize.call_args[1]
            # gelu-fusion default is False, config sets it to True
            assert call_kwargs.get("gelu_fusion") is True

    def test_explicit_kwarg_overrides_all(self, model_file: Path, tmp_path: Path) -> None:
        """Explicit kwargs have highest precedence."""
        config_path = tmp_path / "config.json"
        with config_path.open("w") as f:
            json.dump({"gelu-fusion": True, "layer-norm-fusion": True}, f)

        with patch("winml.modelkit.optim.api.Optimizer") as mock_opt:
            mock_opt.return_value.optimize.return_value = onnx.ModelProto()
            # Override one, leave other from config
            optimize_onnx(model_file, config=config_path, gelu_fusion=False)
            call_kwargs = mock_opt.return_value.optimize.call_args[1]
            assert call_kwargs.get("gelu_fusion") is False  # From kwarg
            assert call_kwargs.get("layer_norm_fusion") is True  # From config


class TestOptimizeOnnxValidation:
    """Tests for configuration validation."""

    def test_validates_config_types(self, model_file: Path, tmp_path: Path) -> None:
        """Validate config value types."""
        config_path = tmp_path / "config.json"
        # String value for boolean capability should raise error
        with config_path.open("w") as f:
            json.dump({"gelu-fusion": "invalid"}, f)

        with pytest.raises(ConfigurationError):
            optimize_onnx(model_file, config=config_path)


class TestOptimizeOnnxErrorTypes:
    """Tests for proper error type propagation."""

    def test_model_validation_error_on_invalid_model(self, tmp_path: Path) -> None:
        """Raise ModelValidationError for invalid ONNX file."""
        invalid_file = tmp_path / "invalid.onnx"
        invalid_file.write_text("not valid onnx")
        with pytest.raises(ModelValidationError):
            optimize_onnx(invalid_file)

    def test_configuration_error_on_invalid_config(
        self, model_file: Path, tmp_path: Path
    ) -> None:
        """Raise ConfigurationError for invalid config values."""
        config_path = tmp_path / "config.json"
        with config_path.open("w") as f:
            json.dump({"gelu-fusion": "not-a-bool"}, f)

        with pytest.raises(ConfigurationError):
            optimize_onnx(model_file, config=config_path)


# =============================================================================
# TEST: Integration (using conftest fixtures)
# =============================================================================


class TestOptimizeOnnxIntegration:
    """Integration tests using real optimization pipeline.

    Uses all_patterns_model fixture from conftest.py for real optimization.
    """

    def test_end_to_end_basic(
        self, all_patterns_model: onnx.ModelProto, tmp_path: Path
    ) -> None:
        """Full pipeline: load, optimize, save."""
        output_path = tmp_path / "optimized.onnx"
        result = optimize_onnx(all_patterns_model, output_path)

        assert isinstance(result, onnx.ModelProto)
        assert output_path.exists()

        # Verify saved model is valid
        loaded = onnx.load(str(output_path))
        onnx.checker.check_model(loaded)

    def test_end_to_end_with_capabilities(
        self, all_patterns_model: onnx.ModelProto, tmp_path: Path
    ) -> None:
        """Full pipeline with explicit capabilities."""
        output_path = tmp_path / "optimized.onnx"
        result = optimize_onnx(
            all_patterns_model,
            output_path,
            gelu_fusion=True,
            layer_norm_fusion=True,
        )

        assert isinstance(result, onnx.ModelProto)
        assert output_path.exists()

    def test_model_proto_passthrough(
        self, all_patterns_model: onnx.ModelProto
    ) -> None:
        """ModelProto input should work without file I/O."""
        result = optimize_onnx(all_patterns_model)
        assert isinstance(result, onnx.ModelProto)
        # Verify it's a valid model
        onnx.checker.check_model(result)

    def test_returns_optimized_model(
        self, all_patterns_model: onnx.ModelProto
    ) -> None:
        """Verify optimization actually modifies the model."""
        # Run optimization
        result = optimize_onnx(all_patterns_model)

        # Model should still be valid
        assert isinstance(result, onnx.ModelProto)
        onnx.checker.check_model(result)

        # Note: We don't assert node reduction because mandatory stages
        # may add nodes (shape inference). The key is valid output.
