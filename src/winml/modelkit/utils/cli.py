# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CLI utilities for ModelKit commands."""

from pathlib import Path

import click

from .constants import ALL_EP_NAMES, SUPPORTED_DEVICES


def model_option(required=True):
    """Add --model option to a Click command.

    Args:
        required: Whether the model option is required (default: True)

    Returns:
        Decorator function
    """
    return click.option(
        "--model",
        required=required,
        type=click.Path(exists=True, path_type=Path),
        help="Path to ONNX model file to analyze",
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
        type=click.Choice(ALL_EP_NAMES, case_sensitive=False),
        help=help_text,
    )


def device_option(required=True, optional_message=None, default="NPU"):
    """Add --device option to a Click command.

    Args:
        required: Whether the device option is required (default: True)
        optional_message: Message to append to help text when
            optional (e.g., "If not specified, uses NPU as
            default.")
        default: Default value when optional (default: "NPU")

    Returns:
        Decorator function
    """
    help_text = "Target device type (CPU, GPU, NPU)"
    if optional_message:
        help_text = f"{help_text}. {optional_message}"

    return click.option(
        "--device",
        required=required,
        default=default if not required else None,
        type=click.Choice(SUPPORTED_DEVICES, case_sensitive=True),
        help=help_text,
    )


def verbosity_options(f):
    """Add verbose and quiet logging options to a Click command.

    Adds --verbose/-v and --quiet/-q flags that control logging verbosity.
    These options are automatically passed to the decorated function.

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
        is_flag=True,
        default=False,
        help="Enable verbose logging to stderr",
    )(f)
    return f  # noqa: RET504
