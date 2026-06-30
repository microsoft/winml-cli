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
    from ..session import EPDeviceTarget  # noqa: TC004


# Per-EP defaults driving :meth:`WinMLCompileConfig.for_provider`. The only
# non-default field today is ``enable_ep_context`` (True for EPs that consume
# the pre-compiled EPContext graph, False for the rest). Unknown / custom
# providers fall through to ``enable_ep_context=False``.
_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "qnn": {"enable_ep_context": True},
    "cpu": {"enable_ep_context": False},
    "cuda": {"enable_ep_context": False},
    "dml": {"enable_ep_context": False},
    "nv_tensorrt_rtx": {"enable_ep_context": False},
    "openvino": {"enable_ep_context": True},
    "vitisai": {"enable_ep_context": False},
    "migraphx": {"enable_ep_context": False},
}


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
        # Resolve hardware target, then build the compile config.
        ep_device = resolve_device(EPDeviceTarget(ep="qnn", device="npu"))
        config = WinMLCompileConfig.for_ep_device(ep_device)

        # Or construct from a short EP name directly.
        config = WinMLCompileConfig.for_provider("qnn")

        # Custom provider options after construction.
        config.ep_config.provider_options["htp_performance_mode"] = "default"
    """

    # Target EP settings
    ep_config: EPConfig = field(default_factory=EPConfig)

    # Resolved EP+device pair (set by CLI or API callers; None means compile
    # stage will infer from ep_config.provider via resolve_device()).
    ep_device: EPDeviceTarget | None = None

    # Behavior
    validate: bool = True
    verbose: bool = False

    @property
    def device(self) -> str:
        """Get device/provider name for backward compatibility."""
        return self.ep_config.provider

    @classmethod
    def for_ep_device(cls, ep_device: EPDeviceTarget) -> WinMLCompileConfig:
        """Factory that creates a config from a fully-resolved EPDeviceTarget.

        The ep_device is stored on the config and threaded to the compile
        stage so that resolve_device() is only called once at the CLI boundary.

        Args:
            ep_device: Fully-resolved (EP, device) binding.

        Returns:
            WinMLCompileConfig bound to the given EPDeviceTarget.
        """
        from ..session import short_ep_name

        provider = short_ep_name(ep_device.ep)
        base = cls.for_provider(provider)
        assert base is not None  # provider is non-None — for_provider only returns None on None input
        base.ep_device = ep_device
        return base

    @classmethod
    def for_provider(
        cls,
        provider: str | None,
        quantize: bool | None = None,
    ) -> WinMLCompileConfig | None:
        """Factory driven by :data:`_PROVIDER_DEFAULTS`.

        Args:
            provider: Provider short name (e.g., ``"qnn"``, ``"dml"``,
                ``"openvino"``) or ``None``. Unknown / custom names fall back
                to ``enable_ep_context=False``.
            quantize: Deprecated. Quantization is now handled by
                :class:`WinMLQuantizationConfig` — passing any non-``None``
                value emits a :class:`DeprecationWarning` and is otherwise
                ignored. Retained as a transitional surface so callers that
                still thread ``quantize=`` continue to receive the warning
                in one place.

        Returns:
            ``WinMLCompileConfig`` for the provider, or ``None`` if
            ``provider`` is ``None``.
        """
        if quantize is not None:
            warnings.warn(
                "The 'quantize' parameter is deprecated and ignored. "
                "Use WinMLQuantizationConfig for quantization settings.",
                DeprecationWarning,
                stacklevel=2,
            )
        if provider is None:
            return None
        defaults = _PROVIDER_DEFAULTS.get(provider, {"enable_ep_context": False})
        return cls(ep_config=EPConfig(provider=provider, **defaults))

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
        from ..session import EPDeviceTarget as _EPDeviceTarget

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
            ep_device = _EPDeviceTarget.from_dict(data["ep_device"])

        return cls(
            ep_config=ep_config,
            ep_device=ep_device,
            validate=data.get("validate", True),
            verbose=data.get("verbose", False),
        )
