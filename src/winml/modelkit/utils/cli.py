# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CLI utilities for ModelKit commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import click

from ..session import VALID_DEVICES, VALID_EPS


# Sorted lowercase device choices consistent with the rest of the codebase.
# Previously SUPPORTED_DEVICES = ["CPU", "GPU", "NPU"] (uppercase — bug).
_DEVICE_CHOICES = sorted(VALID_DEVICES)

# Sorted short EP names sourced from the session facade (single source of truth).
_EP_CHOICES = sorted(VALID_EPS)


if TYPE_CHECKING:
    from ..config import WinMLBuildConfig


def model_path_option(required=True):
    """Add --model option that accepts a local ONNX file path.

    The path is validated for existence on disk.

    Args:
        required: Whether the model option is required (default: True)

    Returns:
        Decorator function
    """
    return click.option(
        "--model",
        "-m",
        required=required,
        type=click.Path(exists=True, path_type=Path),
        help="Path to ONNX model file to analyze",
    )


def model_option(required=True):
    """Add --model option that accepts any model reference.

    Accepts a HuggingFace model ID, build output directory, or .onnx file path.
    No path existence validation is performed.

    Args:
        required: Whether the model option is required (default: True)

    Returns:
        Decorator function
    """
    return click.option(
        "--model",
        "-m",
        required=required,
        default=None,
        help="Model: HF model ID, build output directory, or .onnx file path",
    )


def ep_option(required=True, optional_message=None):
    """Add --ep (execution provider) option to a Click command.

    Args:
        required: Whether the EP option is required (default: True)
        optional_message: Message to append to help text when
            optional (e.g., "If not specified, analyzes all
            supported EPs.")

    Returns:
        Decorator function
    """
    help_text = (
        "Target execution provider. "
        "Full names: QNNExecutionProvider, OpenVINOExecutionProvider, VitisAIExecutionProvider. "
        "Aliases: qnn, ov/openvino, vitis/vitisai"
    )
    if optional_message:
        help_text = f"{help_text}. {optional_message}"

    return click.option(
        "--ep",
        required=required,
        default=None,
        type=click.Choice(_EP_CHOICES, case_sensitive=False),
        help=help_text,
    )


def device_option(required=True, optional_message=None, default="npu"):
    """Add --device option to a Click command.

    Args:
        required: Whether the device option is required (default: True)
        optional_message: Message to append to help text when
            optional (e.g., "If not specified, uses npu as
            default.")
        default: Default value when optional (default: "npu")

    Returns:
        Decorator function
    """
    help_text = "Target device type (cpu, gpu, npu)"
    if optional_message:
        help_text = f"{help_text}. {optional_message}"

    return click.option(
        "--device",
        required=required,
        default=default if not required else None,
        type=click.Choice(_DEVICE_CHOICES, case_sensitive=False),
        help=help_text,
    )


def verbosity_options(f):
    """Add verbose and quiet logging options to a Click command.

    Adds --verbose/-v (stackable: -v, -vv, -vvv) and --quiet/-q flags.
    The decorated function receives ``verbose`` (int, count of -v flags)
    and ``quiet`` (bool).

    See :mod:`winml.modelkit.utils.logging` for the verbosity convention.

    Args:
        f: Click command function to decorate

    Returns:
        Decorated function with verbose and quiet options
    """
    f = click.option(
        "--quiet",
        "-q",
        is_flag=True,
        default=False,
        help="Quiet mode - errors only to stderr",
    )(f)
    f = click.option(
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (-v=INFO, -vv=DEBUG)",
    )(f)
    return f  # noqa: RET504


def build_config_option(func):
    """Add -c/--config option for WinMLBuildConfig JSON file."""
    return click.option(
        "-c",
        "--config",
        "config_file",
        type=click.Path(exists=True, path_type=Path),
        default=None,
        help="WinMLBuildConfig JSON file (from winml config). "
        "Provides defaults; explicit CLI options take precedence.",
    )(func)


def load_build_config(config_path: Path) -> WinMLBuildConfig:
    """Load a WinMLBuildConfig from a JSON file.

    Args:
        config_path: Path to JSON config file.

    Returns:
        Parsed WinMLBuildConfig.

    Raises:
        click.UsageError: If file is empty or invalid JSON.
    """
    from ..config import WinMLBuildConfig

    try:
        content = config_path.read_text()
        if not content.strip():
            raise click.UsageError(f"Config file is empty: {config_path}")
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise click.UsageError(f"Invalid JSON in build config: {e}") from e

    if not isinstance(data, dict):
        raise click.UsageError(f"Build config must be a JSON object, got {type(data).__name__}")

    return WinMLBuildConfig.from_dict(data)


def is_cli_provided(ctx: click.Context, param_name: str) -> bool:
    """Check whether a CLI parameter was explicitly provided by the user.

    Args:
        ctx: Click context.
        param_name: The parameter name (Python name, e.g. 'model').

    Returns:
        True if the user explicitly passed the option on the command line.
    """
    source = ctx.get_parameter_source(param_name)
    return source == click.core.ParameterSource.COMMANDLINE
