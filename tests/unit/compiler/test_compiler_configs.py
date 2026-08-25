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
from winml.modelkit.config import merge_config


class TestEPConfig:
    """Test EPConfig dataclass."""

    def test_default_values(self):
        """Test default EP configuration."""
        config = EPConfig()
        assert config.provider is None
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

    def test_file_backed_provider_options_are_explicit(self) -> None:
        """EP configs expose architecture-agnostic file dependency metadata."""
        assert "provider_option_file_keys" in EPConfig.__dataclass_fields__


class TestCompileConfig:
    """Test WinMLCompileConfig dataclass."""

    def test_default_values(self):
        """Test default config has only EP settings, no quant fields."""
        config = WinMLCompileConfig()
        assert config.ep_config.provider is None
        assert config.validate is True
        assert config.verbose is False
        assert not hasattr(config, "qdq_config")
        assert not hasattr(config, "calibration_config")

    def test_device_property(self):
        """Test device property returns provider name."""
        config = WinMLCompileConfig.for_provider("qnn")
        assert config is not None
        assert config.device == "qnn"

        config = WinMLCompileConfig.for_cpu()
        assert config.device == "cpu"

    def test_for_provider_no_qdq_config(self):
        """``for_provider`` does not create any ``qdq_config`` attribute."""
        config = WinMLCompileConfig.for_provider("qnn")
        assert config is not None
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
        assert config.ep_config.provider == "nvtensorrtrtx"
        assert config.ep_config.enable_ep_context is True

    def test_for_openvino(self):
        """Test OpenVINO factory method."""
        config = WinMLCompileConfig.for_openvino()
        assert config.ep_config.provider == "openvino"
        assert config.ep_config.enable_ep_context is True

    def test_for_vitisai(self):
        """Test Vitis AI factory method."""
        config = WinMLCompileConfig.for_vitisai()
        assert config.ep_config.provider == "vitisai"
        assert config.ep_config.enable_ep_context is True

    def test_for_vitisai_declares_discovered_xclbin_as_file_backed(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The discovered compiler input participates in EPContext identity."""
        xclbin = tmp_path / "voe-4.0-win_amd64" / "xclbins" / "phoenix" / "4x4.xclbin"
        xclbin.parent.mkdir(parents=True)
        xclbin.write_bytes(b"xclbin")
        monkeypatch.setenv("RYZEN_AI_INSTALLATION_PATH", str(tmp_path))

        config = WinMLCompileConfig.for_vitisai()

        assert config.ep_config.provider_option_file_keys == {"xclbin"}

    def test_for_migraphx(self):
        """Test MIGraphX factory method."""
        config = WinMLCompileConfig.for_migraphx()
        assert config.ep_config.provider == "migraphx"
        assert config.ep_config.enable_ep_context is False

    def test_to_dict(self):
        """Test serialization contains only EP fields, no quant fields."""
        config = WinMLCompileConfig.for_provider("qnn")
        assert config is not None
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
        assert config.ep_device is None

    def test_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        original = WinMLCompileConfig.for_provider("qnn")
        assert original is not None
        d = original.to_dict()
        restored = WinMLCompileConfig.from_dict(d)

        assert restored.ep_config.provider == original.ep_config.provider
        assert restored.ep_config.enable_ep_context == original.ep_config.enable_ep_context
        assert restored.validate == original.validate

    def test_roundtrip_preserves_provider_option_file_keys(self) -> None:
        """Explicit file-backed option keys survive config serialization."""
        original = WinMLCompileConfig(
            ep_config=EPConfig(
                provider="qnn",
                provider_options={"compiler_input": "inputs.bin"},
                provider_option_file_keys={"compiler_input"},
            )
        )

        serialized = original.to_dict()

        assert serialized.get("provider_option_file_keys") == ["compiler_input"]
        restored = WinMLCompileConfig.from_dict(serialized)
        assert restored.ep_config.provider_option_file_keys == {"compiler_input"}

    def test_roundtrip_preserves_ep_device(self) -> None:
        """Round-trip retains the resolved EP/device/source binding."""
        from winml.modelkit.session import EPDeviceTarget

        target = EPDeviceTarget(ep="QNNExecutionProvider", device="npu", source="bundled")
        config = WinMLCompileConfig.for_ep_device(target)

        assert config is not None
        serialized = config.to_dict()
        assert serialized["ep_device"] == target.to_dict()

        restored = WinMLCompileConfig.from_dict(serialized)
        assert restored.ep_device == target

    def test_merge_config_deserializes_ep_device_override(self) -> None:
        """Dictionary overrides retain a typed resolved EP/device/source binding."""
        from winml.modelkit.session import EPDeviceTarget

        target = EPDeviceTarget(
            ep="QNNExecutionProvider",
            device="npu",
            source="bundled",
        )
        merged = merge_config(
            WinMLCompileConfig.for_qnn(),
            {"ep_device": target.to_dict()},
        )

        assert merged.ep_device == target
        assert merged.to_dict()["ep_device"] == target.to_dict()


class TestCompileConfigUsagePatterns:
    """Test real-world usage patterns."""

    def test_custom_provider_options(self):
        """Test setting custom provider options."""
        config = WinMLCompileConfig.for_provider("qnn")
        assert config is not None
        config.ep_config.provider_options["htp_performance_mode"] = "default"
        assert config.ep_config.provider_options["htp_performance_mode"] == "default"

    def test_set_qairt_compiler(self):
        """Test setting compiler to qairt with SDK root."""
        from pathlib import Path

        config = WinMLCompileConfig.for_provider("qnn")
        assert config is not None
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
            ("vitisai", "vitisai"),
            ("nv_tensorrt_rtx", "nvtensorrtrtx"),
            # EPs with enable_ep_context=False → no offline compile step → None
            ("dml", None),
            ("cpu", None),
            ("cuda", None),
            ("migraphx", None),
            # Unknown/custom EPs: no EPContext support → None (same as known non-EPContext EPs)
            ("custom_ep", None),
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
        ["for_dml", "for_cpu", "for_cuda", "for_migraphx"],
    )
    def test_direct_factory_still_works(self, factory_name: str) -> None:
        """Low-level for_* factories are still callable directly even though
        for_provider() returns None for these EPs."""
        config = getattr(WinMLCompileConfig, factory_name)()
        assert config is not None
        assert config.ep_config.enable_ep_context is False

    def test_for_provider_custom_ep_returns_none(self):
        """Unknown/custom EPs return None — no EPContext support assumed."""
        result = WinMLCompileConfig.for_provider("custom_ep")
        assert result is None

    @pytest.mark.parametrize(
        "provider,expected_provider",
        [
            ("qnn", "qnn"),
            ("cpu", None),
            ("cuda", None),
            ("dml", None),
            ("nv_tensorrt_rtx", "nvtensorrtrtx"),
            ("openvino", "openvino"),
            ("vitisai", "vitisai"),
            ("migraphx", None),
        ],
    )
    @pytest.mark.parametrize("quantize_value", [True, False])
    def test_for_provider_quantize_emits_deprecation(
        self,
        provider: str,
        expected_provider: str | None,
        quantize_value: bool,
    ) -> None:
        """``for_provider(p, quantize=<any non-None>)`` emits ``DeprecationWarning``.

        Pins the consolidated deprecation surface introduced by T-09: the
        eight per-EP factories that each carried their own ``quantize=``
        deprecation block are collapsed into a single ``for_provider``
        entry point. Both ``True`` and ``False`` warn (only ``None`` /
        omitted is silent).
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = WinMLCompileConfig.for_provider(provider, quantize=quantize_value)
            if expected_provider is None:
                assert config is None
            else:
                assert config is not None
                assert config.ep_config.provider == expected_provider
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1
            assert "quantize" in str(deprecation_warnings[0].message).lower()

    def test_for_provider_no_quantize_no_warning(self) -> None:
        """``for_provider(p)`` without ``quantize=`` emits no warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            WinMLCompileConfig.for_provider("qnn")
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

    def test_for_ep_device_preserves_device_provider_options(self) -> None:
        """A resolved target must reach the device-specific factory options."""
        from winml.modelkit.session import EPDeviceTarget

        config = WinMLCompileConfig.for_ep_device(
            EPDeviceTarget(ep="QNNExecutionProvider", device="npu")
        )

        assert config is not None
        assert config.ep_config.device == "npu"
        assert config.ep_config.provider_options["device_type"] == "NPU"

    def test_for_ep_device_normalizes_duck_typed_target_for_serialization(self) -> None:
        """Duck-typed targets accepted by the factory remain serializable."""
        from types import SimpleNamespace

        config = WinMLCompileConfig.for_ep_device(
            SimpleNamespace(ep="QNNExecutionProvider", device="npu", source="pypi")
        )

        assert config is not None
        assert config.to_dict()["ep_device"] == {
            "ep": "QNNExecutionProvider",
            "device": "npu",
            "source": "pypi",
        }
