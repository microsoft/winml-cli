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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils.constants import EPAlias, EPName


if TYPE_CHECKING:
    from ..utils.constants import EPNameOrAlias


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

    provider: EPAlias = "qnn"
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

    # Behavior
    validate: bool = True
    verbose: bool = False

    @property
    def device(self) -> str:
        """Get device/provider name for backward compatibility."""
        return self.ep_config.provider

    @classmethod
    def for_provider(
        cls, provider: EPNameOrAlias | None, device: str | None = None
    ) -> WinMLCompileConfig | None:
        """Factory that dispatches to a known for_* method or creates a generic config.

        Args:
            provider: Canonical EP name (e.g., "QNNExecutionProvider") or alias
                (e.g., "qnn"). Aliases are normalized to canonical form before
                dispatch. ``None`` short-circuits to ``None``.
            device: Target device ("cpu", "gpu", "npu"). Used by EPs like OpenVINO
                that compile device-specific EPContext blobs and need device_type
                in provider_options so CPU and GPU builds get different cache keys.

        Returns:
            WinMLCompileConfig for the provider, or None if provider is None.
        """
        from ..utils.constants import EP_NAME_TO_ALIAS, normalize_ep_name

        if provider is None:
            return None
        canonical = normalize_ep_name(provider)
        if canonical is None:
            return None
        factories: dict[EPName, Any] = {
            "QNNExecutionProvider": lambda: cls.for_qnn(device=device),
            "DmlExecutionProvider": cls.for_dml,
            "CUDAExecutionProvider": cls.for_cuda,
            "NvTensorRTRTXExecutionProvider": cls.for_nv_tensorrt_rtx,
            "OpenVINOExecutionProvider": lambda: cls.for_openvino(device=device),
            "VitisAIExecutionProvider": lambda: cls.for_vitisai(device=device),
            "MIGraphXExecutionProvider": cls.for_migraphx,
            "CPUExecutionProvider": cls.for_cpu,
        }
        factory = factories.get(canonical)
        if factory:
            config = factory()
            # EPs that don't produce EPContext have no offline compile step
            if not config.ep_config.enable_ep_context:
                return None
            return config
        # Generic fallback: ``canonical`` is in the EPName Literal set, so the
        # reverse lookup is total (no cast).
        return cls(
            ep_config=EPConfig(provider=EP_NAME_TO_ALIAS[canonical], enable_ep_context=False)
        )

    @classmethod
    def for_qnn(cls, device: str | None = None) -> WinMLCompileConfig:
        """Factory for QNN compilation.

        Args:
            device: Target device ("npu", "gpu"). Sets device_type in
                provider_options so NPU and GPU builds get different cache keys.

        Returns:
            WinMLCompileConfig configured for QNN EP.
        """
        provider_options: dict[str, str] = {}
        if device:
            provider_options["device_type"] = device.upper()
        return cls(ep_config=EPConfig(provider="qnn", provider_options=provider_options))

    @classmethod
    def for_cpu(cls) -> WinMLCompileConfig:
        """Factory for CPU compilation (no EPContext)."""
        return cls(
            ep_config=EPConfig(provider="cpu", enable_ep_context=False),
        )

    @classmethod
    def for_cuda(cls) -> WinMLCompileConfig:
        """Factory for CUDA compilation."""
        return cls(
            ep_config=EPConfig(provider="cuda", enable_ep_context=False),
        )

    @classmethod
    def for_dml(cls) -> WinMLCompileConfig:
        """Factory for DirectML compilation."""
        return cls(
            ep_config=EPConfig(provider="dml", enable_ep_context=False),
        )

    @classmethod
    def for_nv_tensorrt_rtx(cls) -> WinMLCompileConfig:
        """Factory for NvTensorRTRTX compilation."""
        return cls(
            ep_config=EPConfig(provider="nv_tensorrt_rtx", enable_ep_context=False),
        )

    @classmethod
    def for_openvino(cls, device: str | None = None) -> WinMLCompileConfig:
        """Factory for OpenVINO compilation."""
        provider_options: dict[str, str] = {}
        if device:
            # OV EPContext blobs are device-specific (CPU vs GPU).
            # Embedding device_type ensures CPU and GPU builds get different
            # cache keys and don't accidentally share the wrong EPContext.
            provider_options["device_type"] = device.upper()
        return cls(
            ep_config=EPConfig(
                provider="openvino", enable_ep_context=True, provider_options=provider_options
            ),
        )

    @classmethod
    def for_vitisai(cls, device: str | None = None) -> WinMLCompileConfig:
        """Factory for Vitis AI (AMD NPU/GPU) compilation."""
        provider_options: dict[str, str] = {}
        if device:
            provider_options["device_type"] = device.upper()
        return cls(
            ep_config=EPConfig(
                provider="vitisai", enable_ep_context=False, provider_options=provider_options
            ),
        )

    @classmethod
    def for_migraphx(cls) -> WinMLCompileConfig:
        """Factory for MIGraphX (AMD ROCm GPU) compilation."""
        return cls(
            ep_config=EPConfig(provider="migraphx", enable_ep_context=False),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for internal use.

        Returns only EP-related fields. Quantization settings are
        serialized separately by WinMLQuantizationConfig.
        """
        return {
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WinMLCompileConfig:
        """Create from dictionary. Unknown keys are ignored."""
        ep_config = EPConfig(
            provider=data.get("execution_provider", "qnn"),
            provider_options=data.get("provider_options", {}),
            enable_ep_context=data.get("enable_ep_context", True),
            embed_context=data.get("embed_context", False),
            compiler=data.get("compiler", "ort"),
            qnn_sdk_root=(Path(data["qnn_sdk_root"]) if data.get("qnn_sdk_root") else None),
        )

        return cls(
            ep_config=ep_config,
            validate=data.get("validate", True),
            verbose=data.get("verbose", False),
        )
