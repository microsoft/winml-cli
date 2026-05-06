# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for compiler configuration classes."""

import warnings

import pytest

from winml.modelkit.compiler import (
    EPConfig,
    WinMLCompileConfig,
)


class TestEPConfig:
    """Test EPConfig dataclass."""

    def test_default_values(self):
        """Test default EP configuration."""
        config = EPConfig()
        assert config.provider == "qnn"
        assert config.provider_options == {}
        assert config.enable_ep_context is True
        assert config.embed_context is False
        assert config.compiler == "ort"
        assert config.qnn_sdk_root is None

    def test_custom_values(self):
        """Test custom EP configuration."""
        config = EPConfig(
            provider="cuda",
            provider_options={"device_id": "1"},
            enable_ep_context=False,
            embed_context=True,
        )
        assert config.provider == "cuda"
        assert config.provider_options == {"device_id": "1"}
        assert config.enable_ep_context is False
        assert config.embed_context is True


class TestCompileConfig:
    """Test WinMLCompileConfig dataclass."""

    def test_default_values(self):
        """Test default config has only EP settings, no quant fields."""
        config = WinMLCompileConfig()
        assert config.ep_config.provider == "qnn"
        assert config.validate is True
        assert config.verbose is False
        assert not hasattr(config, "qdq_config")
        assert not hasattr(config, "calibration_config")

    def test_device_property(self):
        """Test device property returns provider name."""
        config = WinMLCompileConfig.for_qnn()
        assert config.device == "qnn"

        config = WinMLCompileConfig.for_cpu()
        assert config.device == "cpu"

    def test_for_qnn(self):
        """Test QNN factory method creates correct config."""
        config = WinMLCompileConfig.for_qnn()
        assert config.ep_config.provider == "qnn"
        assert config.ep_config.enable_ep_context is True

    def test_for_qnn_no_qdq_config(self):
        """Test QNN factory does not create any qdq_config attribute."""
        config = WinMLCompileConfig.for_qnn()
        assert not hasattr(config, "qdq_config")

    def test_for_cpu(self):
        """Test CPU factory method."""
        config = WinMLCompileConfig.for_cpu()
        assert config.ep_config.provider == "cpu"
        assert config.ep_config.enable_ep_context is False

    def test_for_cuda(self):
        """Test CUDA factory method."""
        config = WinMLCompileConfig.for_cuda()
        assert config.ep_config.provider == "cuda"
        assert config.ep_config.enable_ep_context is False

    def test_for_dml(self):
        """Test DirectML factory method."""
        config = WinMLCompileConfig.for_dml()
        assert config.ep_config.provider == "dml"
        assert config.ep_config.enable_ep_context is False

    def test_for_nv_tensorrt_rtx(self):
        """Test NvTensorRTRTX factory method."""
        config = WinMLCompileConfig.for_nv_tensorrt_rtx()
        assert config.ep_config.provider == "nv_tensorrt_rtx"
        assert config.ep_config.enable_ep_context is False

    def test_for_openvino(self):
        """Test OpenVINO factory method."""
        config = WinMLCompileConfig.for_openvino()
        assert config.ep_config.provider == "openvino"
        assert config.ep_config.enable_ep_context is True

    def test_for_vitisai(self):
        """Test Vitis AI factory method."""
        config = WinMLCompileConfig.for_vitisai()
        assert config.ep_config.provider == "vitisai"
        assert config.ep_config.enable_ep_context is False

    def test_for_migraphx(self):
        """Test MIGraphX factory method."""
        config = WinMLCompileConfig.for_migraphx()
        assert config.ep_config.provider == "migraphx"
        assert config.ep_config.enable_ep_context is False

    def test_to_dict(self):
        """Test serialization contains only EP fields, no quant fields."""
        config = WinMLCompileConfig.for_qnn()
        d = config.to_dict()

        # EP fields present
        assert d["execution_provider"] == "qnn"
        assert d["provider_options"] == {}
        assert d["enable_ep_context"] is True
        assert d["embed_context"] is False
        assert d["compiler"] == "ort"
        assert d["qnn_sdk_root"] is None
        assert d["validate"] is True

        # No quant fields
        assert "quantize" not in d
        assert "weight_type" not in d
        assert "activation_type" not in d
        assert "per_channel" not in d
        assert "calibration_method" not in d
        assert "calibration_samples" not in d
        assert "calibration_load_path" not in d
        assert "calibration_save_path" not in d

    def test_to_dict_cpu(self):
        """Test serialization for CPU config."""
        config = WinMLCompileConfig.for_cpu()
        d = config.to_dict()

        assert d["execution_provider"] == "cpu"
        assert d["enable_ep_context"] is False
        assert "quantize" not in d

    def test_from_dict_basic(self):
        """Test deserialization of EP-only dict."""
        data = {
            "execution_provider": "qnn",
            "provider_options": {"htp_performance_mode": "default"},
            "enable_ep_context": True,
            "embed_context": False,
            "compiler": "ort",
            "validate": True,
        }
        config = WinMLCompileConfig.from_dict(data)
        assert config.ep_config.provider == "qnn"
        assert config.ep_config.provider_options == {"htp_performance_mode": "default"}
        assert config.ep_config.enable_ep_context is True
        assert config.validate is True

    def test_from_dict_ignores_legacy_fields(self):
        """Test from_dict silently ignores legacy quant fields."""
        data = {
            "execution_provider": "qnn",
            "quantize": True,
            "weight_type": "uint8",
            "activation_type": "uint8",
            "per_channel": False,
            "calibration_method": "minmax",
            "calibration_samples": 100,
            "calibration_load_path": "calibration_data.json",
            "calibration_save_path": "calibration_out.json",
            "validate": True,
        }
        config = WinMLCompileConfig.from_dict(data)

        # EP fields parsed correctly
        assert config.ep_config.provider == "qnn"
        assert config.validate is True

        # No quant attributes created
        assert not hasattr(config, "qdq_config")
        assert not hasattr(config, "calibration_config")

    def test_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        original = WinMLCompileConfig.for_qnn()
        d = original.to_dict()
        restored = WinMLCompileConfig.from_dict(d)

        assert restored.ep_config.provider == original.ep_config.provider
        assert restored.ep_config.enable_ep_context == original.ep_config.enable_ep_context
        assert restored.validate == original.validate


class TestDeprecationWarnings:
    """Test deprecation warnings for quantize parameter."""

    @pytest.mark.parametrize(
        "factory_method",
        [
            "for_qnn",
            "for_cpu",
            "for_cuda",
            "for_dml",
            "for_nv_tensorrt_rtx",
            "for_openvino",
            "for_vitisai",
            "for_migraphx",
        ],
    )
    def test_quantize_true_emits_deprecation(self, factory_method: str):
        """Passing quantize=True emits DeprecationWarning."""
        factory = getattr(WinMLCompileConfig, factory_method)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = factory(quantize=True)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "quantize" in str(w[0].message).lower()
            assert "deprecated" in str(w[0].message).lower()
            # Config is still created successfully
            assert config is not None

    @pytest.mark.parametrize(
        "factory_method",
        [
            "for_qnn",
            "for_cpu",
            "for_cuda",
            "for_dml",
            "for_nv_tensorrt_rtx",
            "for_openvino",
            "for_vitisai",
            "for_migraphx",
        ],
    )
    def test_quantize_false_emits_deprecation(self, factory_method: str):
        """Passing quantize=False also emits DeprecationWarning."""
        factory = getattr(WinMLCompileConfig, factory_method)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            factory(quantize=False)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    @pytest.mark.parametrize(
        "factory_method",
        [
            "for_qnn",
            "for_cpu",
            "for_cuda",
            "for_dml",
            "for_nv_tensorrt_rtx",
            "for_openvino",
            "for_vitisai",
            "for_migraphx",
        ],
    )
    def test_no_quantize_no_warning(self, factory_method: str):
        """Not passing quantize param emits no warning."""
        factory = getattr(WinMLCompileConfig, factory_method)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            factory()
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0


class TestCompileConfigUsagePatterns:
    """Test real-world usage patterns."""

    def test_custom_provider_options(self):
        """Test setting custom provider options."""
        config = WinMLCompileConfig.for_qnn()
        config.ep_config.provider_options["htp_performance_mode"] = "default"
        assert config.ep_config.provider_options["htp_performance_mode"] == "default"

    def test_set_qairt_compiler(self):
        """Test setting compiler to qairt with SDK root."""
        from pathlib import Path

        config = WinMLCompileConfig.for_qnn()
        config.ep_config.compiler = "qairt"
        config.ep_config.qnn_sdk_root = Path("/opt/qairt")
        assert config.ep_config.compiler == "qairt"
        assert config.ep_config.qnn_sdk_root == Path("/opt/qairt")


class TestForProvider:
    """Parametrized tests for WinMLCompileConfig.for_provider() factory."""

    @pytest.mark.parametrize(
        "provider,expect_provider",
        [
            (None, None),
            # EPs that produce EPContext → compile config returned
            ("qnn", "qnn"),
            ("openvino", "openvino"),
            # EPs with enable_ep_context=False → no offline compile step → None
            ("dml", None),
            ("cpu", None),
            ("cuda", None),
            ("nv_tensorrt_rtx", None),
            ("vitisai", None),
            ("migraphx", None),
            # Unknown/custom EPs use the generic fallback (enable_ep_context=False
            # in the fallback does NOT apply the None rule — only known factories do)
            ("custom_ep", "custom_ep"),
        ],
    )
    def test_for_provider(
        self,
        provider: str | None,
        expect_provider: str | None,
    ) -> None:
        """for_provider() returns correct config or None."""
        result = WinMLCompileConfig.for_provider(provider)
        if expect_provider is None:
            assert result is None
        else:
            assert result is not None
            assert result.ep_config.provider == expect_provider

    @pytest.mark.parametrize(
        "factory_name",
        ["for_dml", "for_cpu", "for_cuda", "for_vitisai", "for_migraphx", "for_nv_tensorrt_rtx"],
    )
    def test_direct_factory_still_works(self, factory_name: str) -> None:
        """Low-level for_* factories are still callable directly even though
        for_provider() returns None for these EPs."""
        config = getattr(WinMLCompileConfig, factory_name)()
        assert config is not None
        assert config.ep_config.enable_ep_context is False

    def test_for_provider_custom_ep_no_context(self):
        """Custom EP fallback disables EP context."""
        result = WinMLCompileConfig.for_provider("custom_ep")
        assert result is not None
        assert result.ep_config.enable_ep_context is False
