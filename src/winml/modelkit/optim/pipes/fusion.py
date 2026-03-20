# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Fusion optimization pipe using ONNX Runtime transformer optimizer.

This pipe applies transformer-specific fusion optimizations using ORT's
FusionOptions API. It detects and fuses multi-operation patterns into
single optimized operations.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar


if TYPE_CHECKING:
    import onnx

from winml.modelkit.onnx import save_onnx

# Import capability modules for FusionPipe
# Note: gelu import removed - GELU capabilities disabled due to ORT bundling issue
from ..capabilities import attention, layernorm
from .base import BasePipe, OptimizationError, PipeConfig, caps_dict


@dataclass
class ORTFusionPipeConfig(PipeConfig):
    """Configuration for ORT fusion optimization pipe."""

    model_type: str = "clip"  # Always use "clip" internally per design doc

    # Fusion toggles (from FusionOptions) - Total: 12 capabilities (GELU disabled)
    # Note: ORT defaults these to True, but we default to False for QNN compatibility

    # GELU Capabilities (5) - DISABLED (use GraphPipe for GELU fusion)
    # enable_gelu: bool = False
    # enable_bias_gelu: bool = False
    # enable_gelu_approximation: bool = False
    # enable_gemm_fast_gelu: bool = False
    # enable_bias_splitgelu: bool = False  # SD only

    # LayerNorm Capabilities (4)
    enable_layer_norm: bool = False
    enable_skip_layer_norm: bool = False
    enable_embed_layer_norm: bool = False
    enable_bias_skip_layer_norm: bool = False

    # Attention Capabilities (3) - rotary_embeddings removed (Llama/GPT-NeoX only)
    enable_attention: bool = False
    enable_packed_qkv: bool = False  # SD only
    enable_packed_kv: bool = False  # SD only

    # GroupNorm Capabilities (2) - SD only
    enable_group_norm: bool = False
    enable_skip_group_norm: bool = False

    # MatMul Capabilities (1)
    enable_qordered_matmul: bool = False

    # Layout & Misc Capabilities (2) - SD only
    enable_nhwc_conv: bool = False
    enable_bias_add: bool = False

    # Custom fusions (applied after ORT built-in fusions)
    fuse_rmsnorm: bool = False


class ORTFusionPipe(BasePipe):
    """Fusion optimization pipe using ORT FusionOptions."""

    name: ClassVar[str] = "ort_fusion"

    # Capabilities for ORTFusionPipe - transformer-specific fusion capabilities
    # FIXME: GELU capabilities disabled - ORT's FusionOptions.enable_gelu bundles
    # multiple fusion types (GELU, QuickGelu, FastGelu) under one flag, making it
    # impossible to control them independently. This causes unexpected behavior
    # where enabling gelu-fusion also fuses QuickGelu patterns (x*sigmoid(1.702*x)).
    # Use GraphPipe for isolated GELU fusion control instead.
    # See: https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/fusion_options.py
    capabilities: ClassVar[dict[str, Any]] = caps_dict(
        # GELU capabilities - DISABLED (not independently controllable via FusionOptions)
        # gelu.GELU_FUSION,
        # gelu.FAST_GELU_FUSION,
        # gelu.BIAS_GELU_FUSION,
        # gelu.QUICK_GELU_FUSION,
        # gelu.GELU_APPROXIMATION,
        # Attention capabilities
        attention.ATTENTION_FUSION,
        # Layer normalization capabilities
        layernorm.LAYER_NORM_FUSION,
        layernorm.SKIP_LAYER_NORM_FUSION,
        layernorm.SIMPLIFIED_LAYER_NORM_FUSION,
        layernorm.FUSE_RMSNORM,
        layernorm.EMBED_LAYER_NORM_FUSION,
        layernorm.BIAS_SKIP_LAYER_NORM_FUSION,
    )

    # Mapping from capability name to FusionOptions attribute name
    # Total: 12 fusion toggles/settings (GELU disabled, controls NOT included)
    FUSION_ATTR_MAP: ClassVar[dict[str, str]] = {
        # GELU (5) - DISABLED (bundled under enable_gelu, not independently controllable)
        # "gelu-fusion": "enable_gelu",
        # "bias-gelu-fusion": "enable_bias_gelu",
        # "gelu-approximation": "enable_gelu_approximation",
        # "gemm-fast-gelu": "enable_gemm_fast_gelu",
        # "bias-splitgelu-fusion": "enable_bias_splitgelu",
        # LayerNorm (4)
        "layer-norm-fusion": "enable_layer_norm",
        "skip-layer-norm-fusion": "enable_skip_layer_norm",
        "embed-layer-norm-fusion": "enable_embed_layer_norm",
        "bias-skip-layer-norm-fusion": "enable_bias_skip_layer_norm",
        # Attention (3) - rotary-embeddings removed
        "attention-fusion": "enable_attention",
        "packed-qkv-fusion": "enable_packed_qkv",
        "packed-kv-fusion": "enable_packed_kv",
        # GroupNorm (2) - SD only
        "group-norm-fusion": "enable_group_norm",
        "skip-group-norm-fusion": "enable_skip_group_norm",
        # MatMul (1)
        "qordered-matmul": "enable_qordered_matmul",
        # Layout & Misc (2) - SD only
        "nhwc-conv": "enable_nhwc_conv",
        "bias-add-fusion": "enable_bias_add",
    }

    @classmethod
    def build_config(cls, **kwargs: Any) -> ORTFusionPipeConfig:
        """Build fusion pipe config from kwargs.

        Args:
            **kwargs: User-provided configuration

        Returns:
            Configured ORTFusionPipeConfig
        """
        import dataclasses

        # Get valid field names from ORTFusionPipeConfig
        valid_fields = {f.name for f in dataclasses.fields(ORTFusionPipeConfig)}

        config_kwargs = {
            "model_type": kwargs.get("model_type", "clip"),
        }

        # Map capabilities to FusionPipeConfig fields using FUSION_ATTR_MAP
        # Note: Multiple capabilities can map to the same fusion_attr (e.g.,
        # layer-norm-fusion and simplified-layer-norm-fusion both map to enable_layer_norm).
        # We only set a value if explicitly provided via kwargs, otherwise we set the
        # default only if not already set (to avoid later capabilities overwriting earlier ones).
        for cap_name, fusion_attr in cls.FUSION_ATTR_MAP.items():
            cap = cls.capabilities.get(cap_name)
            if cap and fusion_attr in valid_fields:
                # Convert kebab-case name to snake_case for kwargs lookup
                python_name = cap_name.replace("-", "_")
                if python_name in kwargs:
                    # Explicit kwarg always takes precedence
                    config_kwargs[fusion_attr] = kwargs[python_name]
                elif fusion_attr not in config_kwargs:
                    # Only set default if this attr hasn't been set yet
                    config_kwargs[fusion_attr] = cap.default

        # Custom fusions (not in FUSION_ATTR_MAP — direct config fields).
        # Unlike ORT-mapped capabilities, custom fusions must be explicitly
        # enabled. Their defaults (False) are not auto-applied from the
        # capability registry — they only activate via explicit kwargs.
        config_kwargs["fuse_rmsnorm"] = kwargs.get("fuse_rmsnorm", False)

        return ORTFusionPipeConfig(**config_kwargs)

    @classmethod
    def should_process(cls, config: ORTFusionPipeConfig) -> bool:
        """Check if any fusion options are enabled.

        Args:
            config: Pipe configuration

        Returns:
            True if any fusion option is enabled, False otherwise
        """
        return any(
            [
                # GELU (5) - DISABLED (use GraphPipe for GELU fusion)
                # LayerNorm (4)
                config.enable_layer_norm,
                config.enable_skip_layer_norm,
                config.enable_embed_layer_norm,
                config.enable_bias_skip_layer_norm,
                # Attention (3)
                config.enable_attention,
                config.enable_packed_qkv,
                config.enable_packed_kv,
                # GroupNorm (2)
                config.enable_group_norm,
                config.enable_skip_group_norm,
                # MatMul (1)
                config.enable_qordered_matmul,
                # Layout & Misc (2)
                config.enable_nhwc_conv,
                config.enable_bias_add,
                # Custom fusions
                config.fuse_rmsnorm,
            ]
        )

    def process(self, model: onnx.ModelProto, config: ORTFusionPipeConfig) -> onnx.ModelProto:
        """Apply fusion optimizations using ORT transformer optimizer.

        Args:
            model: Input ONNX model (will not be modified)
            config: Pipe configuration from build_config()

        Returns:
            New optimized ONNX model

        Raises:
            OptimizationError: If optimization fails
        """
        # Skip if no fusion options are enabled
        if not self.should_process(config):
            return model

        # Import ORT inside method to avoid import errors
        try:
            from onnxruntime.transformers import optimizer
            from onnxruntime.transformers.fusion_options import FusionOptions
        except ImportError as e:
            raise OptimizationError(
                "Failed to import onnxruntime.transformers",
                pipe_name=self.name,
                cause=e,
            ) from e

        # Build FusionOptions from config
        fusion_opts = FusionOptions(config.model_type)

        # Shape inference always enabled (ORT default) - fusion passes need it
        # for on-demand pattern validation
        fusion_opts.enable_shape_inference = True

        # Fusion toggles - 12 capabilities (GELU disabled)
        # GELU (5) - DISABLED: Force all GELU fusions OFF to prevent bundled behavior
        # ORT's enable_gelu bundles GELU, QuickGelu, and FastGelu under one flag
        fusion_opts.enable_gelu = False
        fusion_opts.enable_bias_gelu = False
        fusion_opts.enable_gelu_approximation = False
        fusion_opts.enable_gemm_fast_gelu = False
        fusion_opts.enable_bias_splitgelu = False
        # LayerNorm (4)
        fusion_opts.enable_layer_norm = config.enable_layer_norm
        fusion_opts.enable_skip_layer_norm = config.enable_skip_layer_norm
        fusion_opts.enable_embed_layer_norm = config.enable_embed_layer_norm
        fusion_opts.enable_bias_skip_layer_norm = config.enable_bias_skip_layer_norm
        # Attention (3)
        fusion_opts.enable_attention = config.enable_attention
        fusion_opts.enable_packed_qkv = config.enable_packed_qkv
        fusion_opts.enable_packed_kv = config.enable_packed_kv
        # GroupNorm (2)
        fusion_opts.enable_group_norm = config.enable_group_norm
        fusion_opts.enable_skip_group_norm = config.enable_skip_group_norm
        # MatMul (1)
        fusion_opts.enable_qordered_matmul = config.enable_qordered_matmul
        # Layout & Misc (2)
        fusion_opts.enable_nhwc_conv = config.enable_nhwc_conv
        fusion_opts.enable_bias_add = config.enable_bias_add

        # ORT optimizer requires file path
        input_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                input_path = f.name
                save_onnx(model, input_path)

            # Run optimizer
            optimized = optimizer.optimize_model(
                input_path,
                model_type=config.model_type,
                optimization_options=fusion_opts,
                opt_level=0,  # We handle graph opt in GraphPipe
            )

            # Custom fusions — applied on the same OnnxModel instance
            # after ORT built-in fusions, using ORT Fusion base class
            if getattr(config, "fuse_rmsnorm", False):
                from ..fusions import FusionRMSNorm

                FusionRMSNorm(optimized).apply()

            return optimized.model

        except Exception as e:
            raise OptimizationError(
                "Failed to apply fusion optimizations",
                pipe_name=self.name,
                model_info={"model_type": config.model_type},
                cause=e,
            ) from e

        finally:
            # Clean up temporary file
            if input_path:
                Path(input_path).unlink(missing_ok=True)
