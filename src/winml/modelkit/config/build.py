# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML Build Configuration - Dataclass and Generation.

This module provides:
- WinMLBuildConfig: Combined config dataclass for WinML pipeline
- generate_build_config(): Backward-compatible dispatcher
- generate_hf_build_config(): Config from HuggingFace model (Scenarios A/B/C)
- generate_onnx_build_config(): Config from pre-exported ONNX (Scenario D)

Configuration Hierarchy:
    WinMLBuildConfig (Top-level aggregator)
    ├── loader: WinMLLoaderConfig       # from modelkit/loader/config.py
    ├── export: WinMLExportConfig       # from modelkit/export/config.py
    ├── optim: WinMLOptimizationConfig  # from modelkit/optim/config.py
    ├── quant: WinMLQuantizationConfig  # from modelkit/quant/config.py
    ├── compile: WinMLCompileConfig     # from modelkit/compiler/configs.py
    └── eval: WinMLEvaluationConfig     # from modelkit/eval/config.py

Design Principles (P1 FUNDAMENTAL):
- CALLS existing APIs from loader/, export/, models/hf/
- Does NOT reimplement their logic
- Only NEW logic is assembly and submodule specialization
- NO HARDCODED VALUES - all shapes from parameters, all defaults from dataclasses

Usage:
    from winml.modelkit.config import WinMLBuildConfig, generate_build_config

    # Auto-generate complete config
    config = generate_build_config("microsoft/resnet-50")

    # Use dataclass directly
    config = WinMLBuildConfig()

    # From dictionary
    config = WinMLBuildConfig.from_dict({
        "loader": {"task": "image-classification"}
    })
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from ..compiler.configs import WinMLCompileConfig
from ..export.config import (
    InputTensorSpec,
    OutputTensorSpec,
    WinMLExportConfig,
    _resolve_export_config_from_specs,
)
from ..loader.config import WinMLLoaderConfig, resolve_loader_config
from ..optim.config import WinMLOptimizationConfig
from ..quant.config import WinMLQuantizationConfig
from ..utils.config_utils import merge_config


# NOTE: WinMLEvaluationConfig is imported lazily to avoid pulling
# eval/__init__.py which imports heavy deps (torch, sklearn, etc.).
# NOTE: MODEL_BUILD_CONFIGS is imported lazily inside generate_build_config()
# to avoid circular import: config -> models.hf -> config


if TYPE_CHECKING:
    import torch
    from torch import nn

    from ..eval.config import WinMLEvaluationConfig  # noqa: TC004

__all__ = [
    "WinMLBuildConfig",
    "generate_build_config",
    "generate_hf_build_config",
    "generate_onnx_build_config",
    "resolve_quant_compile_config",
]

logger = logging.getLogger(__name__)


# =============================================================================
# WINML BUILD CONFIG DATACLASS
# =============================================================================


@dataclass
class WinMLBuildConfig:
    """Combined configuration for WinML model pipeline.

    Attributes:
        loader: Loader configuration (task, model_class, user_script)
        export: Export configuration
        optim: Optimization configuration
        quant: Quantization configuration
        compile: Compilation configuration
        eval: Evaluation configuration

    Example:
        from winml.modelkit.config import WinMLBuildConfig
        from ..optim import WinMLOptimizationConfig

        # Default config
        config = WinMLBuildConfig()

        # With explicit optim
        config = WinMLBuildConfig(
            optim=WinMLOptimizationConfig(
                gelu_fusion=True,
                matmul_add_fusion=True,
            ),
        )

        # With loader config for explicit model class
        config = WinMLBuildConfig.from_dict({
            "loader": {
                "task": "feature-extraction",
                "model_class": "CLIPTextModelWithProjection"
            }
        })
    """

    loader: WinMLLoaderConfig = field(default_factory=WinMLLoaderConfig)
    export: WinMLExportConfig | None = field(default_factory=WinMLExportConfig)
    optim: WinMLOptimizationConfig = field(default_factory=WinMLOptimizationConfig)
    quant: WinMLQuantizationConfig | None = field(default_factory=WinMLQuantizationConfig)
    compile: WinMLCompileConfig | None = field(default_factory=WinMLCompileConfig)
    eval: WinMLEvaluationConfig | None = None

    def __post_init__(self) -> None:
        # Lazy import: inject into module globals so typing.get_type_hints()
        # can resolve the eval field annotation (used by merge_config).
        from ..eval.config import WinMLEvaluationConfig

        globals().setdefault("WinMLEvaluationConfig", WinMLEvaluationConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> WinMLBuildConfig:
        """Create config from nested dictionary."""
        from ..eval.config import WinMLEvaluationConfig

        loader_data = config_dict.get("loader", {})
        export_data = config_dict.get("export", {})
        quant_data = config_dict.get("quant")
        compile_data = config_dict.get("compile")
        eval_data = config_dict.get("eval")
        eval_cfg = None
        if eval_data is not None:
            eval_cfg = WinMLEvaluationConfig.from_dict(eval_data)
        return cls(
            loader=WinMLLoaderConfig.from_dict(loader_data),
            export=(WinMLExportConfig.from_dict(export_data) if export_data is not None else None),
            optim=WinMLOptimizationConfig.from_dict(config_dict.get("optim", {})),
            quant=(
                WinMLQuantizationConfig.from_dict(quant_data) if quant_data is not None else None
            ),
            compile=(
                WinMLCompileConfig.from_dict(compile_data) if compile_data is not None else None
            ),
            eval=eval_cfg,
        )

    def to_dict(self) -> dict:
        """Convert config to nested dictionary."""
        result: dict = {
            "export": self.export.to_dict() if self.export is not None else None,
            "optim": self.optim.to_dict(),
            "quant": self.quant.to_dict() if self.quant is not None else None,
            "compile": self.compile.to_dict() if self.compile is not None else None,
        }
        # Only include loader if it has non-default values
        loader_dict = self.loader.to_dict()
        if loader_dict:
            result["loader"] = loader_dict
        if self.eval is not None:
            result["eval"] = self.eval.to_dict()
        return result

    def validate(self) -> None:
        """Validate config completeness for a build pipeline.

        Checks that all required sections and fields are set. Collects ALL
        validation errors before raising, so the user sees every problem at once.

        Build types:
            - HF build (export is not None): requires loader.task, quant.task,
              quant.model_name when quant is enabled
            - ONNX build (export is None): relaxed — loader.task and quant
              fields are optional since the ONNX model is pre-exported

        Raises:
            ValueError: If one or more validation checks fail. The message
                lists every failure found.
        """
        errors: list[str] = []
        is_onnx_build = self.export is None

        # 1. Loader/export requirements differ by build type
        is_submodule = bool(self.loader and self.loader.module_path)
        if not is_submodule and not is_onnx_build and (not self.loader or not self.loader.task):
            errors.append("loader.task is required for full model builds")
        # export=None is valid for ONNX builds

        # 2. optim config always required
        if self.optim is None:
            errors.append("optim config is required")

        # 3. quant validation (when present)
        # Exceptions: ONNX builds (export=None) don't need quant.task/model_name
        # because the ONNX model is pre-exported. Submodule builds (module_path
        # set) use RandomDataset which only needs the ONNX model_path.
        if self.quant is not None:
            is_submodule = bool(self.loader and self.loader.module_path)
            needs_quant_ids = not is_onnx_build and not is_submodule
            if needs_quant_ids and not self.quant.task:
                errors.append("quant.task is required when quant is enabled for HF builds")
            if needs_quant_ids and not self.quant.model_name:
                errors.append("quant.model_name is required when quant is enabled for HF builds")

        # 4. compile validation (when present)
        if self.compile is not None and (
            not self.compile.ep_config or not self.compile.ep_config.provider
        ):
            errors.append("compile.ep_config.provider is required when compile is enabled")

        if errors:
            raise ValueError("Invalid WinMLBuildConfig:\n" + "\n".join(f"  - {e}" for e in errors))

    def generate_cache_key(self) -> str:
        """Generate deterministic cache key for caching pipeline outputs."""
        components = self.to_dict()
        json_str = json.dumps(components, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]


# =============================================================================
# SUBMODULE INFO DATACLASS
# =============================================================================


class SubmoduleClassNotFoundError(LookupError):
    """Raised when no submodule matches the requested class name.

    Attributes:
        class_name: The class name that was requested.
        available_classes: Sorted list of submodule class names actually
            present (and executed) in the traced model — used by callers to
            render "did you mean…?" suggestions.
    """

    def __init__(self, class_name: str, available_classes: list[str]) -> None:
        self.class_name = class_name
        self.available_classes = available_classes
        super().__init__(f"No submodule with class '{class_name}' found")


@dataclass
class SubmoduleInfo:
    """Info about a discovered submodule from torchinfo.

    All fields are derived from torchinfo's LayerInfo during summary() trace.
    No hardcoded values — shapes and dtypes come from the actual forward pass.

    Attributes:
        class_name: Module class name (e.g., "Conv2d", "ResNetConvLayer")
        module_path: Full dotted path matching named_modules()
        input_shapes: Shape of each input tensor (e.g., [[1,16,64], [1,16,64]])
        output_shapes: Shape of each output tensor (e.g., [[1,16,64]])
        input_dtypes: Dtype of each input tensor (e.g., ["float32", "float32"])
        output_dtypes: Dtype of each output tensor (e.g., ["float32"])
        input_names: Forward-arg names for each input (e.g., ["hidden_state"]
            or ["pixel_values"]). Empty when hook capture didn't run; callers
            then fall back to generic ``input_{i}`` names.
    """

    class_name: str
    module_path: str
    input_shapes: list[list[int]]
    output_shapes: list[list[int]]
    input_dtypes: list[str]
    output_dtypes: list[str]
    input_names: list[str] = field(default_factory=list)


# =============================================================================
# DEVICE / PRECISION POLICY (shared by HF and ONNX paths)
# =============================================================================
def resolve_quant_compile_config(
    *,
    device: str = "auto",
    precision: str = "auto",
    ep: str | None = None,
    task: str | None = None,
) -> tuple[WinMLQuantizationConfig | None, WinMLCompileConfig | None]:
    """Resolve quantization and compilation config from device/precision policy.

    Detects hardware and resolves optimal precision. Returns the appropriate
    quant and compile configs as a tuple. The caller decides how to use them
    (e.g., whether to skip stages based on model state).

    Args:
        device: Target device ("auto", "npu", "gpu", "cpu").
        precision: Target precision ("auto", "fp32", "fp16", "int8",
            "int16", or "w{x}a{y}" e.g. "w8a16").
        ep: Explicit execution provider override.
        task: Model task (used for precision heuristics, e.g., LLM on GPU).

    Returns:
        Tuple of (quant_config, compile_config). Either may be None when the
        policy does not require that stage (e.g., CPU with fp32).
    """
    from ..sysinfo import resolve_device
    from .precision import resolve_precision

    resolved_device, available_devices = resolve_device(device=device)
    logger.info(
        "Device resolved: %s (available: %s)",
        resolved_device,
        ", ".join(available_devices),
    )

    policy = resolve_precision(
        device=resolved_device,
        precision=precision,
        ep=ep,
        available_devices=available_devices,
        task=task,
    )

    if policy.device == "auto":
        return None, None

    # Quant config
    quant_config: WinMLQuantizationConfig | None = None
    if policy.weight_type is not None:
        quant_config = WinMLQuantizationConfig()
        quant_config.weight_type = policy.weight_type
        quant_config.activation_type = policy.activation_type

    # Compile config
    compile_config = WinMLCompileConfig.for_provider(policy.compile_provider)

    return quant_config, compile_config


# =============================================================================
# GENERATE ONNX BUILD CONFIG (Scenario D)
# =============================================================================


def generate_onnx_build_config(
    onnx_path: str | Path,
    *,
    task: str | None = None,
    device: str = "auto",
    precision: str = "auto",
    ep: str | None = None,
    override: WinMLBuildConfig | None = None,
) -> WinMLBuildConfig:
    """Generate build config for a pre-exported ONNX model (Scenario D).

    Skips loader resolution, export, and registry lookup. Assembles a minimal
    config with ``export=None``, auto-detects whether the model is already
    quantized, and applies device/precision policy.

    Args:
        onnx_path: Path to the pre-exported ONNX file.
        task: Optional task name (e.g., "image-classification").
        device: Target device ("auto", "npu", "gpu", "cpu").
        precision: Target precision ("auto", "fp32", "fp16", "int8",
            "int16", or "w{x}a{y}" e.g. "w8a16").
        ep: Explicit execution provider override.
        override: Partial WinMLBuildConfig to merge on top of auto-detected.

    Returns:
        WinMLBuildConfig with export=None and device/precision applied.
    """
    from ..onnx import is_compiled_onnx, is_quantized_onnx

    onnx_path_resolved = Path(onnx_path)
    if not onnx_path_resolved.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path_resolved}. "
            f"Provide a valid path to an existing ONNX file."
        )

    # Start with full pipeline config
    config = WinMLBuildConfig(
        loader=WinMLLoaderConfig(task=task),
        export=None,  # sentinel: already ONNX
        optim=WinMLOptimizationConfig(),
    )

    # Detect model state and apply resolved configs accordingly
    # Priority: compiled > quantized > raw (default)
    if is_compiled_onnx(onnx_path_resolved):
        # Skip all stages — quant=None, compile=None
        config.quant = None
        config.compile = None
        logger.info("Compiled model (EPContext) detected")
    else:
        # Resolve quant + compile from device/precision policy
        resolved_quant, resolved_compile = resolve_quant_compile_config(
            device=device,
            precision=precision,
            ep=ep,
            task=task,
        )

        if is_quantized_onnx(onnx_path_resolved):
            # Skip optimize+quantize, compile with resolved policy
            config.quant = None
            config.compile = resolved_compile
            logger.info("Quantized model (QDQ) detected")
        else:
            # Raw/optimized: apply full resolved policy
            config.quant = resolved_quant
            config.compile = resolved_compile

    # User override has highest priority — applied last
    if override:
        config = merge_config(config, override)
        # Preserve export=None sentinel for ONNX builds.
        # merge_config may reconstruct a default WinMLExportConfig from the
        # override's default field, but ONNX builds use export=None to signal
        # "already exported, skip export stage".
        config.export = None

    return config


# =============================================================================
# GENERATE HF BUILD CONFIG (Scenarios A/B/C)
# =============================================================================


@overload
def generate_hf_build_config(
    model_id: str | None = None,
    *,
    task: str | None = None,
    model_class: str | None = None,
    model_type: str | None = None,
    module: None = None,
    override: WinMLBuildConfig | None = None,
    shape_config: dict | None = None,
    library_name: str = "transformers",
    device: str = "auto",
    precision: str = "auto",
    trust_remote_code: bool = False,
    ep: str | None = None,
) -> WinMLBuildConfig: ...


@overload
def generate_hf_build_config(
    model_id: str | None = None,
    *,
    task: str | None = None,
    model_class: str | None = None,
    model_type: str | None = None,
    module: str,
    override: WinMLBuildConfig | None = None,
    shape_config: dict | None = None,
    library_name: str = "transformers",
    device: str = "auto",
    precision: str = "auto",
    trust_remote_code: bool = False,
    ep: str | None = None,
) -> list[WinMLBuildConfig]: ...


def generate_hf_build_config(
    model_id: str | None = None,
    *,
    task: str | None = None,
    model_class: str | None = None,
    model_type: str | None = None,
    module: str | None = None,
    override: WinMLBuildConfig | None = None,
    shape_config: dict | None = None,
    library_name: str = "transformers",
    device: str = "auto",
    precision: str = "auto",
    trust_remote_code: bool = False,
    ep: str | None = None,
) -> WinMLBuildConfig | list[WinMLBuildConfig]:
    """Generate WinMLBuildConfig for a HuggingFace model (Scenarios A/B/C).

    Orchestrates loader resolution, export config, registry lookup, optional
    user override, device/precision policy, and optional submodule
    specialization.

    Resolution Priority (Three-Tier):
        Tier 1 (HIGHEST): override parameter (user-specified WinMLBuildConfig)
        Tier 2 (MIDDLE):  MODEL_BUILD_CONFIGS registry
        Tier 3 (LOWEST):  Optimum/HF defaults via loader/export modules

    Orchestration Flow:
        1. loader.resolve_loader_config()   -> (WinMLLoaderConfig, hf_config, resolved_class)
           (includes sub-config consolidation for multimodal)
        2. MODEL_BUILD_CONFIGS.get() — registry lookup (may short-circuit step 3)
        3. export._resolve_export_config_from_specs() OR registered export config
        4. _assemble_config() + merge -> WinMLBuildConfig
        5. If module: specialize for each matching submodule

    Args:
        model_id: HuggingFace model ID (e.g., "bert-base-uncased") or local path.
                  Optional when model_type is provided.
        task: Override auto-detected task (e.g., "text-classification").
        model_class: Override auto-detected model class.
        model_type: Override auto-detected model type (e.g., "bert", "resnet").
        module: If specified, generate configs for submodules matching this
                class name. Uses torchinfo to discover submodules and infer
                I/O shapes.
        override: Partial WinMLBuildConfig to merge on top of auto-detected.
        shape_config: Shape overrides passed to resolve_export_config().
        library_name: Source library for TasksManager lookup.
        device: Target device ("auto", "npu", "gpu", "cpu").
        precision: Target precision ("auto", "fp32", "fp16", "int8",
            "int16", or "w{x}a{y}" e.g. "w8a16").
        trust_remote_code: Allow running custom code from model repository.
        ep: Explicit execution provider override.

    Returns:
        - When module=None: WinMLBuildConfig (single config)
        - When module=str: list[WinMLBuildConfig] (one per matching submodule)

    Raises:
        ValueError: If neither model_id nor model_type is provided, task
                    detection fails, or model_type has no supported tasks.
    """
    # STEP 1: Resolve loader config (ALL loader concerns)
    _trust_remote_code = trust_remote_code or (
        override.loader.trust_remote_code if override and override.loader else False
    )
    loader_config, hf_config, resolved_class = resolve_loader_config(
        model_id,
        task=task,
        model_class=model_class,
        model_type=model_type,
        trust_remote_code=_trust_remote_code,
        library_name=library_name,
    )

    # =========================================================================
    # STEP 2: Lookup registered config FIRST (may short-circuit Optimum)
    # =========================================================================
    # Lazy import to avoid circular dependency: config -> models.hf -> config
    from ..models.hf import MODEL_BUILD_CONFIGS

    _registry_key = loader_config.model_type.replace("_", "-")
    registered = MODEL_BUILD_CONFIGS.get(_registry_key)

    # =========================================================================
    # STEP 3: Generate export config
    # =========================================================================
    # Priority: registered config with I/O specs > Optimum lookup.
    # Models not in Optimum's TasksManager (e.g., BLIP) crash at
    # _resolve_export_config_from_specs(). If the registry already has
    # input_tensors, use them directly and skip the Optimum path.
    # Note: None means "not configured" (fall through to Optimum);
    # [] would mean "explicitly no inputs" (use as-is, skip Optimum).
    _registered_export = registered.export if registered else None
    if _registered_export is not None and _registered_export.input_tensors is not None:
        # deepcopy to avoid mutating the shared registry singleton
        export_config = copy.deepcopy(_registered_export)
        logger.info(
            "Using registered export config for '%s' (skipping Optimum lookup)",
            _registry_key,
        )
    else:
        # Standard path: resolve I/O specs from Optimum's OnnxConfig
        logger.debug(
            "No registered export config for '%s'; resolving via Optimum",
            _registry_key,
        )
        export_config = _resolve_export_config_from_specs(
            model_type=loader_config.model_type,
            task=loader_config.task,
            hf_config=hf_config,
            library_name=library_name,
            model_id=model_id,
            batch_size=WinMLExportConfig().batch_size,
            **(shape_config or {}),
        )

    # =========================================================================
    # STEP 4: Assemble config + merge override
    # =========================================================================
    parent_config = _assemble_config(
        loader_config=loader_config,
        export_config=export_config,
        registered=registered,
        model_id=model_id,
        model_type=hf_config.model_type,
    )
    if override:
        parent_config = merge_config(parent_config, override)

    # =========================================================================
    # STEP 4.5: Apply device/precision policy (affects quant + compile only)
    # =========================================================================
    from ..sysinfo import resolve_device
    from .precision import resolve_precision

    # ALWAYS detect hardware — even when device="auto" — so we don't
    # blindly default to QNN on machines without an NPU (#412).
    resolved_device, available_devices = resolve_device(device=device)
    logger.info(
        "Device resolved: %s (available: %s)",
        resolved_device,
        ", ".join(available_devices),
    )

    policy = resolve_precision(
        device=resolved_device,
        precision=precision,
        ep=ep,
        available_devices=available_devices,
        task=parent_config.loader.task,
    )

    # Apply policy: set compile provider from detected hardware
    if policy.device != "auto":
        # Quant config (weight_type and activation_type are always both-None or both-set)
        if policy.weight_type is not None:
            if parent_config.quant is None:
                parent_config.quant = WinMLQuantizationConfig()
            parent_config.quant.weight_type = policy.weight_type
            parent_config.quant.activation_type = policy.activation_type
        else:
            parent_config.quant = None

        # Compile config
        parent_config.compile = WinMLCompileConfig.for_provider(
            policy.compile_provider,
        )
    else:
        # Even in auto/auto mode, set compile provider from detected hardware
        # instead of preserving the hardcoded EPConfig default (#412).
        from .precision import get_provider_for_device

        hw_provider = get_provider_for_device(resolved_device)
        if hw_provider is not None:
            parent_config.compile = WinMLCompileConfig.for_provider(
                hw_provider,
            )
        # When hw_provider is None (CPU-only), keep the default compile config
        # so the pipeline still has a valid compile section.

    # =========================================================================
    # STEP 5: Specialize for submodules if requested
    # =========================================================================
    if module:
        # Instantiate model with RANDOM WEIGHTS -- torchinfo only needs architecture.
        # Concrete classes (BertForMaskedLM, etc.) accept config as constructor arg.
        # Auto classes (AutoModelForMaskedLM, etc.) reject direct construction
        # and require .from_config(). Try direct first, fall back to from_config.
        try:
            model = resolved_class(hf_config)
        except OSError as e:
            logger.debug("Direct construction failed (%s), using from_config()", e)
            model = resolved_class.from_config(hf_config)

        # Extract input shapes and dtypes from export_config -- NO HARDCODED VALUES
        input_tensors = [t for t in (export_config.input_tensors or []) if t.shape is not None]
        input_shapes = [t.shape for t in input_tensors]
        input_dtypes = [t.dtype for t in input_tensors]
        if not input_shapes:
            raise ValueError(
                "Cannot extract input shapes for submodule discovery. "
                "Ensure export config has input_tensors with shapes populated, "
                "or provide shapes explicitly."
            )
        submodules = _find_submodules_by_class(
            model,
            module,
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
        )
        logger.info("Found %d submodules matching '%s'", len(submodules), module)

        return [_build_submodule_config(sub_info, parent_config) for sub_info in submodules]

    return parent_config


# =============================================================================
# GENERATE BUILD CONFIG - DISPATCHER (backward compat)
# =============================================================================


@overload
def generate_build_config(
    model_id: str | None = None,
    *,
    task: str | None = None,
    model_class: str | None = None,
    model_type: str | None = None,
    module: None = None,
    override: WinMLBuildConfig | None = None,
    shape_config: dict | None = None,
    library_name: str = "transformers",
    device: str = "auto",
    precision: str = "auto",
    trust_remote_code: bool = False,
    ep: str | None = None,
    onnx_path: str | Path | None = None,
) -> WinMLBuildConfig: ...


@overload
def generate_build_config(
    model_id: str | None = None,
    *,
    task: str | None = None,
    model_class: str | None = None,
    model_type: str | None = None,
    module: str,
    override: WinMLBuildConfig | None = None,
    shape_config: dict | None = None,
    library_name: str = "transformers",
    device: str = "auto",
    precision: str = "auto",
    trust_remote_code: bool = False,
    ep: str | None = None,
    onnx_path: str | Path | None = None,
) -> list[WinMLBuildConfig]: ...


def generate_build_config(
    model_id: str | None = None,
    *,
    task: str | None = None,
    model_class: str | None = None,
    model_type: str | None = None,
    module: str | None = None,
    override: WinMLBuildConfig | None = None,
    shape_config: dict | None = None,
    library_name: str = "transformers",
    device: str = "auto",
    precision: str = "auto",
    trust_remote_code: bool = False,
    ep: str | None = None,
    onnx_path: str | Path | None = None,
) -> WinMLBuildConfig | list[WinMLBuildConfig]:
    """Generate WinMLBuildConfig by orchestrating existing modules.

    Thin dispatcher that routes to :func:`generate_onnx_build_config` (when
    ``onnx_path`` is provided) or :func:`generate_hf_build_config` (otherwise).
    Kept for backward compatibility -- new code should call the specific
    function directly.

    Args:
        model_id: HuggingFace model ID or local path (forwarded to HF path).
        task: Override auto-detected task.
        model_class: Override auto-detected model class.
        model_type: Override auto-detected model type.
        module: If specified, generate configs for submodules matching this
                class name (HF path only).
        override: Partial WinMLBuildConfig to merge on top of auto-detected.
        shape_config: Shape overrides for dummy input generation.
        library_name: Source library for TasksManager lookup.
        device: Target device ("auto", "npu", "gpu", "cpu").
        precision: Target precision ("auto", "fp32", "fp16", "int8",
            "int16", or "w{x}a{y}" e.g. "w8a16").
        trust_remote_code: Allow running custom code from model repository.
        ep: Explicit execution provider override.
        onnx_path: Path to a pre-exported ONNX file (Scenario D).

    Returns:
        - When module=None: WinMLBuildConfig (single config)
        - When module=str: list[WinMLBuildConfig] (one per matching submodule)
    """
    if onnx_path is not None:
        return generate_onnx_build_config(
            onnx_path,
            task=task,
            device=device,
            precision=precision,
            ep=ep,
            override=override,
        )
    return generate_hf_build_config(
        model_id,
        task=task,
        model_class=model_class,
        model_type=model_type,
        module=module,
        override=override,
        shape_config=shape_config,
        library_name=library_name,
        device=device,
        precision=precision,
        trust_remote_code=trust_remote_code,
        ep=ep,
    )


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _build_submodule_config(
    sub_info: SubmoduleInfo,
    parent_config: WinMLBuildConfig,
) -> WinMLBuildConfig:
    """Build a WinMLBuildConfig for a single discovered submodule.

    Submodules are intermediate nn.Module layers (e.g., ResNetConvLayer) that
    have no OnnxConfig registration and no standard ONNX tensor names.
    All I/O specs (shapes, dtypes, counts) come from torchinfo traces — no
    hardcoded values.

    Args:
        sub_info: Submodule metadata from torchinfo (class_name, shapes, dtypes)
        parent_config: Parent model config to inherit optim/compile from

    Returns:
        WinMLBuildConfig for the submodule with:
        - Generic I/O names ("input_0", "input_1", ...) since no OnnxConfig exists
        - Shapes, dtypes, and tensor count from torchinfo trace
        - Inherited model_type from parent; task intentionally omitted
        - module_path and model_class from sub_info
        - Inherited optim/compile from parent
        - Quant with task=None, model_name=None (RandomDataset fallback)
    """

    # Build InputTensorSpec for EACH input tensor (not just the first).
    # Use the submodule's actual forward-arg names so build_hf_model can
    # call submodule(**kwargs) correctly — submodule forward args may be
    # positional (e.g. `input`) or keyword (e.g. `hidden_state`). Fall back
    # to generic input_{i} only when names were not discovered.
    def _input_name(i: int) -> str:
        if i < len(sub_info.input_names) and sub_info.input_names[i]:
            return sub_info.input_names[i]
        return f"input_{i}"

    input_tensors = [
        InputTensorSpec(
            name=_input_name(i),
            shape=tuple(shape),
            dtype=sub_info.input_dtypes[i] if i < len(sub_info.input_dtypes) else None,
        )
        for i, shape in enumerate(sub_info.input_shapes)
    ]

    # Build OutputTensorSpec for EACH output tensor
    output_tensors = [
        OutputTensorSpec(name=f"output_{i}") for i in range(len(sub_info.output_shapes))
    ]

    return WinMLBuildConfig(
        loader=WinMLLoaderConfig(
            # task intentionally omitted — submodules don't have tasks
            model_type=parent_config.loader.model_type,
            model_class=sub_info.class_name,
            module_path=sub_info.module_path,
        ),
        export=WinMLExportConfig(
            input_tensors=input_tensors or None,
            output_tensors=output_tensors or None,
            dynamic_axes={},  # Static shapes for submodules
            # opset_version and batch_size use dataclass defaults from WinMLExportConfig
        ),
        optim=copy.deepcopy(parent_config.optim),
        # Submodule builds use RandomDataset for calibration:
        # quantize_onnx() falls back to "random" when task/model_name are None,
        # and RandomDataset reads input specs from the ONNX model file.
        quant=WinMLQuantizationConfig(
            samples=1,
            task=None,
            model_name=None,
        ),
        compile=copy.deepcopy(parent_config.compile),
    )


def _merge_export_config(
    base: WinMLExportConfig,
    override: WinMLExportConfig,
) -> WinMLExportConfig:
    """Merge registered export config on top of Optimum-resolved config.

    Override fields replace base fields when non-None.
    Handles InputTensorSpec/OutputTensorSpec lists correctly
    (unlike generic merge_config which converts them to dicts).

    Args:
        base: Optimum-resolved export config (or empty placeholder).
        override: Registered export config from MODEL_BUILD_CONFIGS.

    Returns:
        New WinMLExportConfig with override fields applied.
    """
    # Pick input/output tensors: override wins when non-None.
    # Deep-copy lists to avoid sharing references with the registry singleton.
    input_tensors = (
        override.input_tensors if override.input_tensors is not None else base.input_tensors
    )
    output_tensors = (
        override.output_tensors if override.output_tensors is not None else base.output_tensors
    )

    return WinMLExportConfig(
        opset_version=(
            override.opset_version
            if override.opset_version != WinMLExportConfig().opset_version
            else base.opset_version
        ),
        batch_size=(
            override.batch_size
            if override.batch_size != WinMLExportConfig().batch_size
            else base.batch_size
        ),
        input_tensors=(copy.deepcopy(input_tensors) if input_tensors is not None else None),
        output_tensors=(copy.deepcopy(output_tensors) if output_tensors is not None else None),
        dynamic_axes=(
            override.dynamic_axes if override.dynamic_axes is not None else base.dynamic_axes
        ),
        dynamo=override.dynamo if override.dynamo else base.dynamo,
    )


def _assemble_config(
    loader_config: WinMLLoaderConfig,
    export_config: WinMLExportConfig,
    registered: WinMLBuildConfig | None,
    *,
    model_id: str | None = None,
    model_type: str | None = None,
) -> WinMLBuildConfig:
    """Assemble WinMLBuildConfig from resolved loader and export configs.

    Handles optim/quant/compile from the registry or defaults,
    and populates quant config with task and model_name.

    Args:
        loader_config: Resolved WinMLLoaderConfig (from resolve_loader_config).
        export_config: Resolved WinMLExportConfig
            (from registry or _resolve_export_config_from_specs).
        registered: Registered config from MODEL_BUILD_CONFIGS (or None).
        model_id: HuggingFace model ID (for quant model_name), or None.
        model_type: Parent HF model type (for quant fallback name).

    Returns:
        Assembled WinMLBuildConfig.
    """
    # Get optim/quant/compile from registry if available, else use defaults
    # IMPORTANT: Match WinMLBuildConfig() default behavior - always have quant/compile
    optim_config = (
        copy.deepcopy(registered.optim)
        if registered and registered.optim
        else WinMLOptimizationConfig()
    )
    quant_config = (
        copy.deepcopy(registered.quant)
        if registered and registered.quant
        else WinMLQuantizationConfig()
    )
    compile_config = (
        copy.deepcopy(registered.compile)
        if registered and registered.compile
        else WinMLCompileConfig()
    )

    # Populate quant config with task and model_name for task-aware calibration
    if quant_config:
        quant_config.task = loader_config.task
        if model_id is None and model_type is not None:
            logger.warning(
                "Quantization model_name set to '%s' (model type). "
                "For calibration datasets, provide --model with a full model ID.",
                model_type,
            )
        quant_config.model_name = model_id or model_type

    return WinMLBuildConfig(
        loader=loader_config,
        export=export_config,
        optim=optim_config,
        quant=quant_config,
        compile=compile_config,
    )


def _get_dtype_map() -> dict[str, torch.dtype]:
    """Return mapping from dtype string names to torch.dtype.

    Lazy helper: imports torch and builds the dict on each call.
    Both ``_find_submodules_by_class`` and ``_build_dummy_inputs``
    share this single definition to avoid duplication.
    """
    import torch

    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }


def _find_submodules_by_class(
    model: nn.Module,
    class_name: str,
    *,
    input_shapes: list[tuple[int, ...]],
    input_dtypes: list[str | None] | None = None,
) -> list[SubmoduleInfo]:
    """Find all submodules matching a class name using torchinfo.

    IMPORTANT: NO HARDCODED INPUT SHAPES OR DTYPES. The caller must provide
    input_shapes and input_dtypes obtained from io_specs or other configuration
    sources.

    Args:
        model: PyTorch model instance
        class_name: Class name to match (e.g., "ResNetConvLayer")
        input_shapes: List of input tensor shapes for torchinfo.summary().
                     Must be provided by caller (from io_specs).
        input_dtypes: Optional list of dtype strings (e.g., ["int32", "float32"])
                     for each input tensor. When provided, torchinfo uses these
                     instead of defaulting to float32. Required for models with
                     integer inputs (e.g., BERT's input_ids).

    Returns:
        List of SubmoduleInfo with I/O shapes from torchinfo

    Raises:
        ValueError: If input_shapes is empty (caller must provide shapes)

    Example:
        # Shapes and dtypes come from io_specs, via resolve_io_specs()
        submodules = _find_submodules_by_class(
            model,
            "ResNetConvLayer",
            input_shapes=[(1, 3, 224, 224)],
            input_dtypes=["float32"],
        )
    """
    if not input_shapes:
        raise ValueError(
            "input_shapes must be provided. NO HARDCODED SHAPES allowed. "
            "Pass shapes from io_specs obtained via resolve_io_specs()."
        )

    import torch
    from torchinfo import summary

    # Use the first input shape for torchinfo (most models have single input)
    # For multi-input models, torchinfo accepts a list
    input_size = input_shapes[0] if len(input_shapes) == 1 else input_shapes

    # Map dtype strings to torch.dtype for torchinfo
    torch_dtypes = None
    if input_dtypes:
        dtype_map = _get_dtype_map()
        torch_dtypes = [
            dtype_map.get(d, torch.float32) if d else torch.float32 for d in input_dtypes
        ]

    # Run torchinfo to get module hierarchy with shapes
    model_info = summary(
        model,
        input_size=input_size,
        dtypes=torch_dtypes,
        verbose=0,
        depth=10,
    )

    # Collect torchinfo-discovered modules matching class_name, plus the
    # full set of executed class names — surfaced via SubmoduleClassNotFoundError
    # so the CLI can suggest valid alternatives on a typo.
    torchinfo_modules: list[tuple[str, Any]] = []  # (full_path, layer_info)
    executed_class_names: set[str] = set()
    for layer_info in model_info.summary_list:
        if not layer_info.executed:
            continue
        executed_class_names.add(layer_info.class_name)
        if layer_info.class_name != class_name:
            continue

        # Build full dotted path by walking parent chain (matches named_modules())
        parts = []
        node = layer_info
        while node.parent_info is not None:
            parts.append(node.var_name or "")
            node = node.parent_info
        full_path = ".".join(reversed(parts))
        torchinfo_modules.append((full_path, layer_info))

    if not torchinfo_modules:
        raise SubmoduleClassNotFoundError(class_name, sorted(executed_class_names))

    # Second pass: hook-based capture for complete multi-input I/O data.
    # torchinfo only captures the first input tensor per module; our hooks
    # capture ALL positional args AND keyword args.
    from ..inspect.module_io_capture import capture_module_io

    dummy_inputs = _build_dummy_inputs(input_shapes, input_dtypes)
    hook_data = capture_module_io(model, dummy_inputs, target_class=class_name)

    results = []
    for full_path, layer_info in torchinfo_modules:
        io_info = hook_data.get(full_path)
        layer_input_names: list[str] = []
        if io_info and io_info.input_shapes:
            # Prefer hook-captured data (has complete multi-input info)
            layer_input_shapes = io_info.input_shapes
            layer_output_shapes = io_info.output_shapes
            layer_input_dtypes = io_info.input_dtypes
            layer_output_dtypes = io_info.output_dtypes
            layer_input_names = io_info.input_names
        else:
            # Fall back to torchinfo data (single input only)
            layer_input_shapes = [layer_info.input_size] if layer_info.input_size else []
            layer_output_shapes = [layer_info.output_size] if layer_info.output_size else []

            # torchinfo does not expose per-layer dtypes; infer from module
            # parameters, falling back to "float32" for parameter-free layers.
            param_dtype = "float32"
            params = list(layer_info.module.parameters())
            if params:
                param_dtype = str(params[0].dtype).replace("torch.", "")

            layer_input_dtypes = [param_dtype] * len(layer_input_shapes)
            layer_output_dtypes = [param_dtype] * len(layer_output_shapes)

            # Without hook data, derive names from the forward signature so
            # build_hf_model can invoke the submodule with the correct kwargs.
            try:
                sig = inspect.signature(layer_info.module.forward)
                layer_input_names = [p.name for p in sig.parameters.values() if p.name != "self"][
                    : len(layer_input_shapes)
                ]
            except (TypeError, ValueError):
                layer_input_names = []

        results.append(
            SubmoduleInfo(
                class_name=layer_info.class_name,
                module_path=full_path,
                input_shapes=layer_input_shapes,
                output_shapes=layer_output_shapes,
                input_dtypes=layer_input_dtypes,
                output_dtypes=layer_output_dtypes,
                input_names=layer_input_names,
            )
        )

    return results


def _build_dummy_inputs(
    input_shapes: list[tuple[int, ...]],
    input_dtypes: list[str | None] | None = None,
) -> dict[str, torch.Tensor]:
    """Build dummy input tensors for hook capture forward pass.

    Args:
        input_shapes: List of input tensor shapes.
        input_dtypes: Optional list of dtype strings per tensor.

    Returns:
        Dictionary of named dummy tensors matching the given shapes and dtypes.
    """
    import torch

    dtype_map = _get_dtype_map()

    inputs = {}
    for i, shape in enumerate(input_shapes):
        dtype_str = input_dtypes[i] if input_dtypes and i < len(input_dtypes) else None
        torch_dtype = dtype_map.get(dtype_str, torch.float32) if dtype_str else torch.float32
        if torch_dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            inputs[f"input_{i}"] = torch.ones(shape, dtype=torch_dtype)
        else:
            inputs[f"input_{i}"] = torch.randn(shape, dtype=torch_dtype)
    return inputs
