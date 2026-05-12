# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for WinMLAutoModel.from_onnx() classmethod.

Verifies:
- from_onnx() auto-generates config via generate_build_config(onnx_path=...)
- from_onnx() uses explicit config when provided
- from_pretrained() delegates ONNX files to from_onnx()
- from_onnx passes ep and device through to build_onnx_model()
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.models.auto import WinMLAutoModel
from winml.modelkit.session.ep_device import EPDevice


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def cpu_ep_device() -> EPDevice:
    """Minimal EPDevice for CPU used across from_onnx/from_pretrained tests."""
    return EPDevice(
        ep="CPUExecutionProvider",
        device="cpu",
        vendor_id=0x1234,
        device_id=0x0001,
    )


@pytest.fixture()
def fake_onnx(tmp_path: Path) -> Path:
    """Create a fake ONNX file for testing."""
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"fake-onnx")
    return onnx_file


def _make_build_result(tmp_path: Path) -> MagicMock:
    """Create a mock BuildResult with the expected attributes."""
    result = MagicMock()
    result.final_onnx_path = tmp_path / "model.onnx"
    result.output_dir = tmp_path
    return result


class TestFromOnnx:
    """Test WinMLAutoModel.from_onnx()."""

    def test_auto_generates_config_when_none(
        self, fake_onnx: Path, tmp_path: Path, cpu_ep_device: EPDevice
    ):
        """from_onnx() without config auto-generates via generate_build_config."""
        mock_config = MagicMock()
        mock_config.export = None
        mock_config.loader = None
        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.config.generate_onnx_build_config", return_value=mock_config),
            patch("winml.modelkit.build.build_onnx_model") as mock_build,
            patch("winml.modelkit.models.auto.get_winml_class") as mock_get_class,
        ):
            mock_build.return_value = _make_build_result(tmp_path)
            mock_instance = MagicMock()
            mock_get_class.return_value = lambda **kw: mock_instance

            WinMLAutoModel.from_onnx(
                str(fake_onnx),
                ep_device=cpu_ep_device,
                task="image-classification",
            )

        mock_build.assert_called_once()
        call_kwargs = mock_build.call_args.kwargs
        config = call_kwargs["config"]
        # ONNX builds have export=None (no HF export needed)
        assert config.export is None

    def test_uses_explicit_config_as_override(
        self, fake_onnx: Path, tmp_path: Path, cpu_ep_device: EPDevice
    ):
        """from_onnx() with explicit config merges it as override on generated config."""
        from winml.modelkit.config import WinMLBuildConfig
        from winml.modelkit.optim.config import WinMLOptimizationConfig

        # Override with specific optim flags (export=None inherited from base)
        explicit_config = WinMLBuildConfig(
            export=None,  # preserve ONNX sentinel
            optim=WinMLOptimizationConfig(gelu_fusion=True),
            quant=None,
        )

        # generate_onnx_build_config applies the override and returns a merged config.
        # Simulate that by returning the explicit_config directly (the merged result).
        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch(
                "winml.modelkit.config.generate_onnx_build_config",
                return_value=explicit_config,
            ),
            patch("winml.modelkit.build.build_onnx_model") as mock_build,
            patch("winml.modelkit.models.auto.get_winml_class") as mock_get_class,
        ):
            mock_build.return_value = _make_build_result(tmp_path)
            mock_instance = MagicMock()
            mock_get_class.return_value = lambda **kw: mock_instance

            WinMLAutoModel.from_onnx(
                fake_onnx,
                ep_device=cpu_ep_device,
                task="image-classification",
                config=explicit_config,
            )

        call_kwargs = mock_build.call_args.kwargs
        # Config is generated with override applied
        assert call_kwargs["config"].export is None  # ONNX sentinel preserved
        assert call_kwargs["config"].quant is None  # from override
        assert call_kwargs["config"].optim.get("gelu_fusion") is True  # from override

    def test_passes_ep_and_device_to_build(self, fake_onnx: Path, tmp_path: Path):
        """from_onnx() extracts ep and device from EPDevice and forwards to build_onnx_model."""
        npu_ep_device = EPDevice(
            ep="QNNExecutionProvider",
            device="npu",
            vendor_id=0x17CB,
            device_id=0x0001,
        )
        mock_config = MagicMock()
        mock_config.loader = None
        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch(
                "winml.modelkit.config.generate_onnx_build_config",
                return_value=mock_config,
            ),
            patch("winml.modelkit.build.build_onnx_model") as mock_build,
            patch("winml.modelkit.models.auto.get_winml_class") as mock_get_class,
        ):
            mock_build.return_value = _make_build_result(tmp_path)
            mock_instance = MagicMock()
            mock_get_class.return_value = lambda **kw: mock_instance

            WinMLAutoModel.from_onnx(
                fake_onnx,
                ep_device=npu_ep_device,
                task="image-classification",
            )

        # from_onnx converts ep_device.ep to short form via short_ep_name() before build
        call_kwargs = mock_build.call_args.kwargs
        assert call_kwargs["ep"] == "qnn"
        assert call_kwargs["device"] == "npu"

    def test_returns_winml_pretrained_model(
        self, fake_onnx: Path, tmp_path: Path, cpu_ep_device: EPDevice
    ):
        """from_onnx() returns the inference wrapper from get_winml_class."""
        mock_config = MagicMock()
        mock_config.loader = None
        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch(
                "winml.modelkit.config.generate_onnx_build_config",
                return_value=mock_config,
            ),
            patch("winml.modelkit.build.build_onnx_model") as mock_build,
            patch("winml.modelkit.models.auto.get_winml_class") as mock_get_class,
        ):
            mock_build.return_value = _make_build_result(tmp_path)
            mock_instance = MagicMock()
            mock_get_class.return_value = lambda **kw: mock_instance

            result = WinMLAutoModel.from_onnx(
                fake_onnx,
                ep_device=cpu_ep_device,
                task="image-classification",
            )

        assert result is mock_instance


class TestFromPretrainedDelegatesToFromOnnx:
    """Test that from_pretrained delegates .onnx files to from_onnx."""

    def test_delegates_onnx_to_from_onnx(
        self, fake_onnx: Path, tmp_path: Path, cpu_ep_device: EPDevice
    ):
        """from_pretrained with .onnx file delegates to from_onnx."""
        with patch.object(WinMLAutoModel, "from_onnx") as mock_from_onnx:
            mock_from_onnx.return_value = MagicMock()

            WinMLAutoModel.from_pretrained(
                str(fake_onnx),
                cpu_ep_device,
                task="image-classification",
                precision="fp32",
            )

        mock_from_onnx.assert_called_once()
        call_kwargs = mock_from_onnx.call_args.kwargs
        assert call_kwargs["task"] == "image-classification"
        assert call_kwargs["ep_device"] is cpu_ep_device
        assert call_kwargs["precision"] == "fp32"

    def test_passes_ep_from_kwargs(self, fake_onnx: Path, tmp_path: Path, cpu_ep_device: EPDevice):
        """from_pretrained passes ep_device through to from_onnx."""
        with patch.object(WinMLAutoModel, "from_onnx") as mock_from_onnx:
            mock_from_onnx.return_value = MagicMock()

            WinMLAutoModel.from_pretrained(
                str(fake_onnx),
                cpu_ep_device,
                task="image-classification",
            )

        call_kwargs = mock_from_onnx.call_args.kwargs
        assert call_kwargs["ep_device"].ep == "CPUExecutionProvider"
