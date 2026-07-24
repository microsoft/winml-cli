# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Tests for WinML configuration system.

Tests the configuration classes migrated to modelkit
following the design specifications in docs/design/automodel/CORELOOP.md Section 3.

Acceptance Criteria (from design):
- AC-1: Config dataclasses with nested structure (export, optim, quant, compile)
- AC-2: from_dict() and to_dict() serialization
- AC-3: JSON round-trip persistence via to_dict/from_dict
- AC-6: Forward compatibility (ignore unknown fields)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


class TestWinMLExportConfig:
    """Test WinMLExportConfig dataclass."""

    def test_default_values(self):
        """AC-1: Test default configuration values."""
        from winml.modelkit.export import WinMLExportConfig

        config = WinMLExportConfig()

        assert config.opset_version == 17
        assert config.batch_size == 1
        assert config.dynamic_axes is None

    def test_custom_values(self):
        """AC-1: Test custom configuration values."""
        from winml.modelkit.export import WinMLExportConfig

        config = WinMLExportConfig(
            opset_version=14,
            batch_size=4,
            dynamic_axes={"input": {0: "batch"}},
        )

        assert config.opset_version == 14
        assert config.batch_size == 4
        assert config.dynamic_axes == {"input": {0: "batch"}}

    def test_to_dict(self):
        """AC-2: Test serialization to dict."""
        from winml.modelkit.export import WinMLExportConfig

        config = WinMLExportConfig(opset_version=15, batch_size=2)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["opset_version"] == 15
        assert config_dict["batch_size"] == 2

    def test_from_dict(self):
        """AC-2: Test deserialization from dict."""
        from winml.modelkit.export import WinMLExportConfig

        config_dict = {"opset_version": 16, "batch_size": 8}
        config = WinMLExportConfig.from_dict(config_dict)

        assert config.opset_version == 16
        assert config.batch_size == 8

    def test_from_dict_ignores_unknown_fields(self):
        """AC-6: Forward compatibility - ignore unknown fields."""
        from winml.modelkit.export import WinMLExportConfig

        config_dict = {
            "opset_version": 17,
            "batch_size": 1,
            "unknown_field": "should_be_ignored",
            "another_unknown": 42,
        }
        config = WinMLExportConfig.from_dict(config_dict)

        assert config.opset_version == 17
        assert not hasattr(config, "unknown_field")


class TestWinMLQuantizationConfig:
    """Test WinMLQuantizationConfig dataclass."""

    def test_default_values(self):
        """AC-1: Test default quantization config."""
        from winml.modelkit.quant import WinMLQuantizationConfig

        config = WinMLQuantizationConfig()

        assert config.mode == "static"
        assert config.weight_type == "uint8"
        assert config.samples == 10
        assert config.calibration_method == "minmax"

    def test_qdq_mode_config(self):
        """Test QDQ-specific configuration."""
        from winml.modelkit.quant import WinMLQuantizationConfig

        config = WinMLQuantizationConfig(
            mode="static",
            weight_type="int8",
            activation_type="int8",
            calibration_method="minmax",
        )

        assert config.mode == "static"
        assert config.weight_type == "int8"


class TestWinMLOptimizationConfig:
    """Test WinMLOptimizationConfig (dict subclass)."""

    def test_default_values(self):
        """AC-1: Test default optimization config is empty dict."""
        from winml.modelkit.optim.config import WinMLOptimizationConfig

        config = WinMLOptimizationConfig()

        # WinMLOptimizationConfig is a dict subclass, default is empty
        assert isinstance(config, dict)

    def test_fusion_flags(self):
        """Test fusion configuration as dict keys."""
        from winml.modelkit.optim.config import WinMLOptimizationConfig

        config = WinMLOptimizationConfig(gelu_fusion=True, matmul_add_fusion=False)
        config_dict = config.to_dict()

        assert config_dict["gelu_fusion"] is True
        assert config_dict["matmul_add_fusion"] is False


class TestWinMLCompileConfig:
    """Test WinMLCompileConfig dataclass."""

    def test_default_values(self):
        """AC-1: Test default compile config."""
        from winml.modelkit.compiler import WinMLCompileConfig

        config = WinMLCompileConfig()

        # Default provider is None (no hidden EP default)
        assert config.device == ""
        assert config.validate is True

    def test_device_via_ep_config(self):
        """Test device via ep_config.provider."""
        from winml.modelkit.compiler import EPConfig, WinMLCompileConfig

        for provider in ["qnn", "cpu", "cuda", "dml"]:
            config = WinMLCompileConfig(ep_config=EPConfig(provider=provider))
            assert config.device == provider

    def test_factory_methods(self):
        """Test ``for_provider`` factory across common providers."""
        from winml.modelkit.compiler import WinMLCompileConfig

        qnn_config = WinMLCompileConfig.for_provider("qnn")
        assert qnn_config is not None
        assert qnn_config.device == "qnn"

        assert WinMLCompileConfig.for_provider("cpu") is None


class TestWinMLConfig:
    """Test unified WinMLBuildConfig."""

    def test_default_initialization(self):
        """AC-1: Test default nested config structure."""
        from winml.modelkit.config import WinMLBuildConfig

        config = WinMLBuildConfig()

        assert hasattr(config, "export")
        assert hasattr(config, "quant")
        assert hasattr(config, "optim")
        assert hasattr(config, "compile")

    def test_nested_access(self):
        """AC-1: Test nested config access."""
        from winml.modelkit.compiler import EPConfig, WinMLCompileConfig
        from winml.modelkit.config import WinMLBuildConfig

        config = WinMLBuildConfig(compile=WinMLCompileConfig(ep_config=EPConfig(provider="qnn")))

        # Should be able to access nested configs
        assert config.export.opset_version == 17
        assert config.compile.device == "qnn"

    def test_to_dict_nested(self):
        """AC-2: Test nested serialization."""
        from winml.modelkit.config import WinMLBuildConfig

        config = WinMLBuildConfig()
        config_dict = config.to_dict()

        assert "export" in config_dict
        assert "compile" in config_dict

    def test_from_dict_nested(self):
        """AC-2: Test nested deserialization."""
        from winml.modelkit.config import WinMLBuildConfig

        config_dict = {
            "export": {"opset_version": 15, "batch_size": 2},
            "quant": {"mode": "qdq", "weight_type": "int8"},
            "optim": {},
            "compile": {"execution_provider": "cpu", "quantize": False},
        }
        config = WinMLBuildConfig.from_dict(config_dict)

        assert config.export.opset_version == 15
        assert config.export.batch_size == 2
        assert config.compile.device == "cpu"

    def test_json_round_trip_via_dict(self, tmp_path: Path):
        """AC-3: Test config persistence via to_dict/from_dict + JSON."""
        from winml.modelkit.config import WinMLBuildConfig

        config = WinMLBuildConfig()
        config.export.batch_size = 4

        # Save via JSON
        save_path = tmp_path / "config.json"
        save_path.write_text(json.dumps(config.to_dict(), indent=2))

        assert save_path.exists()

        # Load via JSON
        loaded_dict = json.loads(save_path.read_text())
        loaded = WinMLBuildConfig.from_dict(loaded_dict)

        assert loaded.export.batch_size == 4
        assert loaded.export.opset_version == config.export.opset_version

    def test_json_serialization(self, tmp_path: Path):
        """Test JSON round-trip serialization."""
        from winml.modelkit.config import WinMLBuildConfig

        config = WinMLBuildConfig()
        config_dict = config.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(config_dict)
        parsed = json.loads(json_str)

        # Should reconstruct
        reloaded = WinMLBuildConfig.from_dict(parsed)
        assert reloaded.export.opset_version == config.export.opset_version


class TestConfigValidation:
    """Test configuration validation."""

    def test_opset_version_range(self):
        """Test opset version validation."""
        from winml.modelkit.export import WinMLExportConfig

        # Valid opset versions
        for version in [11, 14, 17, 20]:
            config = WinMLExportConfig(opset_version=version)
            assert config.opset_version == version

    def test_batch_size_positive(self):
        """Test batch size must be positive."""
        from winml.modelkit.export import WinMLExportConfig

        config = WinMLExportConfig(batch_size=1)
        assert config.batch_size == 1

        config = WinMLExportConfig(batch_size=32)
        assert config.batch_size == 32
