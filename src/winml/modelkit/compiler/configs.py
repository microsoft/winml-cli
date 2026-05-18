# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Configuration classes for compiler module.

Design follows the automodel pattern:
- Single source of truth (WinMLCompileConfig)
- Explicit over implicit
- Factory methods for common configurations
- No capability registry - just dataclasses

Quantization concerns (QDQ, calibration) have been moved to
WinMLQuantizationConfig in modelkit.quant.config (#241).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ..session import EPDevice  # noqa: TC004


@dataclass
class EPConfig:
    """Configuration for Execution Provider compilation.

    Controls how the model is compiled for the target EP.

    Attributes:
        provider: Target execution provider (qnn, cpu, cuda, dml)
        provider_options: EP-specific options as key=value dict
        enable_ep_context: Generate EPContext model with pre-compiled graph
        embed_context: Embed context in ONNX (True) or external .bin file (False)
        compiler: Compiler backend ("ort" or "qairt")
        qnn_sdk_root: Path to QAIRT SDK root (required when compiler is "qairt")
    """

    provider: str = "qnn"
    provider_options: dict[str, str] = field(default_factory=dict)
    enable_ep_context: bool = True
    embed_context: bool = False
    compiler: str = "ort"
    qnn_sdk_root: Path | None = None


@dataclass
class WinMLCompileConfig:
    """Configuration for ONNX compilation pipeline.

    This is the single source of truth for compile (EP) settings.
    Users create this config and pass it to compile_onnx().

    Quantization concerns (QDQ insertion, calibration) are handled
    separately by WinMLQuantizationConfig.

    Core Loop:
        [model.onnx] -> [compile] -> [model_ctx.onnx]

    Attributes:
        ep_config: Execution provider settings
        validate: Validate compiled model
        verbose: Enable verbose logging

    Examples:
        # Default: QNN compilation
        config = WinMLCompileConfig.for_qnn()

        # CPU (no EPContext)
        config = WinMLCompileConfig.for_cpu()

        # Custom provider options
        config = WinMLCompileConfig.for_qnn()
        config.ep_config.provider_options["htp_performance_mode"] = "default"
    """

    # Target EP settings
    ep_config: EPConfig = field(default_factory=EPConfig)

    # Resolved EP+device pair (set by CLI or API callers; None means compile
    # stage will infer from ep_config.provider via resolve_device()).
    ep_device: EPDevice | None = None

    # Behavior
    validate: bool = True
    verbose: bool = False

    @property
    def device(self) -> str:
        """Get device/provider name for backward compatibility."""
        return self.ep_config.provider

    @classmethod
    def for_ep_device(cls, ep_device: EPDevice) -> WinMLCompileConfig:
        """Factory that creates a config from a fully-resolved EPDevice.

        The ep_device is stored on the config and threaded to the compile
        stage so that resolve_device() is only called once at the CLI boundary.

        Args:
            ep_device: Fully-resolved (EP, device) binding.

        Returns:
            WinMLCompileConfig bound to the given EPDevice.
        """
        from ..session import short_ep_name

        provider = short_ep_name(ep_device.ep)
        base = cls.for_provider(provider) or cls(ep_config=EPConfig(provider=provider))
        base.ep_device = ep_device
        return base

    @classmethod
    def for_provider(cls, provider: str | None) -> WinMLCompileConfig | None:
        """Factory that dispatches to a known for_* method or creates a generic config.

        Args:
            provider: Provider name (e.g., "qnn", "dml", "openvino") or None.

        Returns:
            WinMLCompileConfig for the provider, or None if provider is None.
        """
        if provider is None:
            return None
        factories: dict[str, Any] = {
            "qnn": cls.for_qnn,
            "dml": cls.for_dml,
            "cuda": cls.for_cuda,
            "nv_tensorrt_rtx": cls.for_nv_tensorrt_rtx,
            "openvino": cls.for_openvino,
            "vitisai": cls.for_vitisai,
            "migraphx": cls.for_migraphx,
            "cpu": cls.for_cpu,
        }
        factory = factories.get(provider)
        if factory:
            return factory()
        # Generic fallback for unknown/custom providers
        return cls(ep_config=EPConfig(provider=provider, enable_ep_context=False))

    @classmethod
    def for_qnn(cls, quantize: bool | None = None) -> WinMLCompileConfig:
        """Factory for QNN compilation.

        Args:
            quantize: Deprecated. Quantization is now handled by
                WinMLQuantizationConfig. This parameter is ignored.

        Returns:
            WinMLCompileConfig configured for QNN EP.
        """
        if quantize is not None:
            warnings.warn(
                "The 'quantize' parameter is deprecated and ignored. "
                "Use WinMLQuantizationConfig for quantization settings.",
                DeprecationWarning,
                stacklevel=2,
            )
        return cls(ep_config=EPConfig(provider="qnn"))

    @classmethod
    def for_cpu(cls, quantize: bool | None = None) -> WinMLCompileConfig:
        """Factory for CPU compilation (no EPContext)."""
        if quantize is not None:
            warnings.warn(
                "The 'quantize' parameter is deprecated and ignored. "
                "Use WinMLQuantizationConfig for quantization settings.",
                DeprecationWarning,
                stacklevel=2,
            )
        return cls(
            ep_config=EPConfig(provider="cpu", enable_ep_context=False),
        )

    @classmethod
    def for_cuda(cls, quantize: bool | None = None) -> WinMLCompileConfig:
        """Factory for CUDA compilation."""
        if quantize is not None:
            warnings.warn(
                "The 'quantize' parameter is deprecated and ignored. "
                "Use WinMLQuantizationConfig for quantization settings.",
                DeprecationWarning,
                stacklevel=2,
            )
        return cls(
            ep_config=EPConfig(provider="cuda", enable_ep_context=False),
        )

    @classmethod
    def for_dml(cls, quantize: bool | None = None) -> WinMLCompileConfig:
        """Factory for DirectML compilation."""
        if quantize is not None:
            warnings.warn(
                "The 'quantize' parameter is deprecated and ignored. "
                "Use WinMLQuantizationConfig for quantization settings.",
                DeprecationWarning,
                stacklevel=2,
            )
        return cls(
            ep_config=EPConfig(provider="dml", enable_ep_context=False),
        )

    @classmethod
    def for_nv_tensorrt_rtx(cls, quantize: bool | None = None) -> WinMLCompileConfig:
        """Factory for NvTensorRTRTX compilation."""
        if quantize is not None:
            warnings.warn(
                "The 'quantize' parameter is deprecated and ignored. "
                "Use WinMLQuantizationConfig for quantization settings.",
                DeprecationWarning,
                stacklevel=2,
            )
        return cls(
            ep_config=EPConfig(provider="nv_tensorrt_rtx", enable_ep_context=False),
        )

    @classmethod
    def for_openvino(cls, quantize: bool | None = None) -> WinMLCompileConfig:
        """Factory for OpenVINO compilation."""
        if quantize is not None:
            warnings.warn(
                "The 'quantize' parameter is deprecated and ignored. "
                "Use WinMLQuantizationConfig for quantization settings.",
                DeprecationWarning,
                stacklevel=2,
            )
        return cls(
            ep_config=EPConfig(provider="openvino", enable_ep_context=True),
        )

    @classmethod
    def for_vitisai(cls, quantize: bool | None = None) -> WinMLCompileConfig:
        """Factory for Vitis AI (AMD/Xilinx NPU) compilation."""
        if quantize is not None:
            warnings.warn(
                "The 'quantize' parameter is deprecated and ignored. "
                "Use WinMLQuantizationConfig for quantization settings.",
                DeprecationWarning,
                stacklevel=2,
            )
        return cls(
            ep_config=EPConfig(provider="vitisai", enable_ep_context=False),
        )

    @classmethod
    def for_migraphx(cls, quantize: bool | None = None) -> WinMLCompileConfig:
        """Factory for MIGraphX (AMD ROCm GPU) compilation."""
        if quantize is not None:
            warnings.warn(
                "The 'quantize' parameter is deprecated and ignored. "
                "Use WinMLQuantizationConfig for quantization settings.",
                DeprecationWarning,
                stacklevel=2,
            )
        return cls(
            ep_config=EPConfig(provider="migraphx", enable_ep_context=False),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for internal use.

        Returns only EP-related fields. Quantization settings are
        serialized separately by WinMLQuantizationConfig.
        """
        d: dict[str, Any] = {
            "execution_provider": self.ep_config.provider,
            "provider_options": self.ep_config.provider_options,
            "enable_ep_context": self.ep_config.enable_ep_context,
            "embed_context": self.ep_config.embed_context,
            "compiler": self.ep_config.compiler,
            "qnn_sdk_root": (
                str(self.ep_config.qnn_sdk_root) if self.ep_config.qnn_sdk_root else None
            ),
            "validate": self.validate,
        }
        if self.ep_device is not None:
            d["ep_device"] = self.ep_device.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WinMLCompileConfig:
        """Create from dictionary, ignoring unknown and legacy quant fields.

        Legacy quantization fields (quantize, weight_type, activation_type,
        per_channel, calibration_method, calibration_samples, etc.) are
        silently ignored for backward compatibility.

        Args:
            data: Configuration dictionary.

        Returns:
            WinMLCompileConfig instance.
        """
        from ..session import EPDevice as _EPDevice

        ep_config = EPConfig(
            provider=data.get("execution_provider", "qnn"),
            provider_options=data.get("provider_options", {}),
            enable_ep_context=data.get("enable_ep_context", True),
            embed_context=data.get("embed_context", False),
            compiler=data.get("compiler", "ort"),
            qnn_sdk_root=(Path(data["qnn_sdk_root"]) if data.get("qnn_sdk_root") else None),
        )

        ep_device = None
        if "ep_device" in data and data["ep_device"] is not None:
            ep_device = _EPDevice.from_dict(data["ep_device"])

        return cls(
            ep_config=ep_config,
            ep_device=ep_device,
            validate=data.get("validate", True),
            verbose=data.get("verbose", False),
        )
