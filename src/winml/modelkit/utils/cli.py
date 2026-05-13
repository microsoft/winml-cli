# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CLI utilities for ModelKit commands."""

from pathlib import Path

import click

from ..session import _VALID_DEVICES, VALID_EPS


# Sorted lowercase device choices consistent with the rest of the codebase.
# Previously SUPPORTED_DEVICES = ["CPU", "GPU", "NPU"] (uppercase — bug).
_DEVICE_CHOICES = sorted(_VALID_DEVICES)

# Sorted short EP names sourced from the session facade (single source of truth).
_EP_CHOICES = sorted(VALID_EPS)


def model_option(required=True):
    """Add --model option to a Click command.

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
