# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Public API for ONNX model optimization.

This module provides the main `optimize_onnx` function, which is the primary
entry point for optimizing ONNX models. It wraps the internal `Optimizer` class
and provides a simple, high-level interface.

Example:
    from winml.modelkit.optim import optimize_onnx

    # Basic usage
    model = optimize_onnx("model.onnx", gelu_fusion=True)

    # With config file
    model = optimize_onnx("model.onnx", config="optimize.json")

    # Save to file
    model = optimize_onnx("model.onnx", "optimized.onnx", gelu_fusion=True)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import onnx

from ..onnx import load_onnx, save_onnx
from .errors import ConfigurationError, ModelValidationError
from .optimizer import Optimizer
from .pipes import get_all_capabilities
from .registry import auto_enable_dependencies, validate, validate_dependencies


# Configure module logger
logger = logging.getLogger(__name__)


def _load_model(model: str | Path | onnx.ModelProto) -> tuple[onnx.ModelProto, str | None]:
    """Load ONNX model from path or return if already loaded.

    Args:
        model: Input model - path to ONNX file or ModelProto object.

    Returns:
        Tuple of (loaded ModelProto, original path string or None).

    Raises:
        FileNotFoundError: If model path does not exist.
        ModelValidationError: If model cannot be loaded.
    """
    if isinstance(model, onnx.ModelProto):
        return model, None

    model_path = Path(model)
    if not model_path.exists():
        msg = f"Model file not found: {model_path}"
        raise FileNotFoundError(msg)

    try:
        loaded = load_onnx(model_path)
    except Exception as e:
        raise ModelValidationError(
            "Failed to load ONNX model",
            model_path=str(model_path),
            cause=e,
        ) from e

    return loaded, str(model_path)


def _load_config(config: str | Path) -> dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        config: Path to JSON configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        ConfigurationError: If JSON is invalid.
    """
    config_path = Path(config)
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    try:
        with config_path.open(encoding="utf-8") as f:
            result: dict[str, Any] = json.load(f)
            return result
    except json.JSONDecodeError as e:
        raise ConfigurationError(
            f"Invalid JSON in config file: {config_path}",
            errors=[str(e)],
        ) from e


def _merge_config(
    config_dict: dict[str, Any] | None,
    capabilities: dict[str, Any],
    all_caps: dict[str, Any],
) -> dict[str, Any]:
    """Merge configuration with proper precedence.

    Precedence (highest to lowest):
    1. **capabilities kwargs (snake_case)
    2. config file/dict (kebab-case)
    3. Capability defaults

    Args:
        config_dict: Configuration from file or dict (kebab-case keys).
        capabilities: Explicit capability overrides (snake_case keys).
        all_caps: All capability definitions from registry.

    Returns:
        Merged configuration dict with kebab-case keys.
    """
    # Start with defaults
    result: dict[str, Any] = {name: cap.default for name, cap in all_caps.items()}

    # Layer 2: Apply config file/dict (kebab-case keys)
    if config_dict is not None:
        # Extract capabilities section if present, otherwise use entire dict
        caps_config = config_dict.get("capabilities", config_dict)
        # Only apply known capability keys
        result.update({key: value for key, value in caps_config.items() if key in all_caps})

    # Layer 3: Apply kwargs (snake_case keys, highest precedence)
    for cap_name, cap_def in all_caps.items():
        python_name = cap_def.python_name  # snake_case
        if python_name in capabilities and capabilities[python_name] is not None:
            result[cap_name] = capabilities[python_name]

    return result


def _convert_to_kwargs(config: dict[str, Any], all_caps: dict[str, Any]) -> dict[str, Any]:
    """Convert kebab-case config to snake_case kwargs for Optimizer.

    Args:
        config: Configuration with kebab-case keys.
        all_caps: All capability definitions.

    Returns:
        Configuration with snake_case keys for Optimizer.optimize().
    """
    result: dict[str, Any] = {}
    for cap_name, value in config.items():
        if cap_name in all_caps:
            python_name = all_caps[cap_name].python_name
            result[python_name] = value
    return result


def optimize_onnx(
    model: str | Path | onnx.ModelProto,
    output: str | Path | None = None,
    *,
    config: str | Path | dict[str, Any] | None = None,
    **capabilities: Any,
) -> onnx.ModelProto:
    """Optimize an ONNX model with capability-based control.

    This is the main public API for ONNX optimization. It provides a simple,
    high-level interface that wraps the internal Optimizer class.

    Args:
        model: Input model - path to ONNX file or ModelProto object.
        output: Output path. If provided, saves optimized model to this path.
            If None, only returns ModelProto without saving.
        config: Configuration source - JSON file path or capabilities dict.
            Values from config are overridden by **capabilities kwargs.
            JSON keys use kebab-case (e.g., "gelu-fusion").
        **capabilities: Individual capability overrides using snake_case names.
            e.g., gelu_fusion=True, layer_norm_fusion=True.
            These have the highest precedence over config values.

    Returns:
        Optimized ModelProto.

    Raises:
        ModelValidationError: If input model is invalid ONNX.
        ConfigurationError: If config file/dict is invalid.
        OptimizationError: If optimization pipeline fails.
        FileNotFoundError: If model or config file path doesn't exist.

    Example:
        >>> from winml.modelkit.optim import optimize_onnx
        >>>
        >>> # Minimal usage - runs mandatory stages only
        >>> model = optimize_onnx("model.onnx")
        >>>
        >>> # Enable specific capabilities
        >>> model = optimize_onnx(
        ...     "model.onnx",
        ...     "optimized.onnx",
        ...     gelu_fusion=True,
        ...     layer_norm_fusion=True,
        ... )
        >>>
        >>> # Load from JSON config
        >>> model = optimize_onnx("model.onnx", config="optimize.json")
        >>>
        >>> # Override config with kwargs (kwargs take precedence)
        >>> model = optimize_onnx(
        ...     "model.onnx",
        ...     config="optimize.json",
        ...     gelu_fusion=False,  # Overrides config file
        ... )
    """
    # Step 1: Load model
    logger.info("Loading ONNX model...")
    loaded_model, _model_path = _load_model(model)

    # Step 2: Load config if path provided
    config_dict: dict[str, Any] | None = None
    if config is not None:
        if isinstance(config, str | Path):
            logger.info("Loading config from %s...", config)
            config_dict = _load_config(config)
        else:
            config_dict = config

    # Step 3: Get all capabilities from registry
    all_caps = get_all_capabilities()

    # Step 4: Merge configuration with precedence
    merged_config = _merge_config(config_dict, capabilities, all_caps)

    # Step 5: Validate configuration
    logger.debug("Validating configuration...")
    validation_errors = validate(merged_config, all_caps)
    if validation_errors:
        raise ConfigurationError("Invalid configuration", errors=validation_errors)

    # Step 6: Auto-enable dependencies
    resolved_config = auto_enable_dependencies(merged_config, all_caps)

    # Step 7: Validate dependencies after resolution
    dep_errors = validate_dependencies(resolved_config, all_caps)
    if dep_errors:
        raise ConfigurationError("Capability dependency error", errors=dep_errors)

    # Step 8: Convert to kwargs format for Optimizer
    optimizer_kwargs = _convert_to_kwargs(resolved_config, all_caps)

    # Step 9: Run optimization
    logger.info("Starting optimization pipeline...")
    optimizer = Optimizer()
    optimized_model = optimizer.optimize(loaded_model, **optimizer_kwargs)

    # Step 10: Save if output path provided
    if output is not None:
        output_path = Path(output)
        logger.info("Saving optimized model to %s...", output_path)
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_onnx(optimized_model, output_path)
        logger.info("Model saved successfully")

    return optimized_model
