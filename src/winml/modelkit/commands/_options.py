# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Shared CLI option decorators.

One source of truth for options used across multiple wmk subcommands.
See docs/design/cli/3_cli_args_spec.md for the full specification.
"""

from __future__ import annotations

import click


def _normalize_uppercase(
    ctx: click.Context,
    param: click.Parameter,
    value: str | None,
) -> str | None:
    """Normalize value to uppercase (for device choices)."""
    return value.upper() if value else value


_KNOWN_PRECISIONS = {
    "auto",
    "fp32",
    "fp16",
    "int8",
    "int16",
    "w8a8",
    "w8a16",
    "w4a16",
}


def _validate_precision(
    ctx: click.Context,
    param: click.Parameter,
    value: str | None,
) -> str | None:
    """Warn on unknown precision values but don't reject them."""
    if value and value.lower() not in _KNOWN_PRECISIONS:
        click.echo(
            f"Warning: unknown precision '{value}'. "
            f"Known values: {', '.join(sorted(_KNOWN_PRECISIONS))}",
            err=True,
        )
    return value


def model_option(required: bool = True):
    """Model identifier: HF model ID, local path, or .onnx file."""
    return click.option(
        "-m",
        "--model",
        required=required,
        type=str,
        help="HuggingFace model ID, local path, or .onnx file",
    )


def device_option():
    """Target device for inference/compilation."""
    return click.option(
        "-d",
        "--device",
        default="auto",
        type=click.Choice(
            ["auto", "cpu", "gpu", "npu"],
            case_sensitive=False,
        ),
        callback=_normalize_uppercase,
        expose_value=True,
        is_eager=False,
        help="Target device",
    )


def ep_option():
    """Force specific execution provider (overrides --device)."""
    return click.option(
        "--ep",
        default=None,
        type=str,
        help="Force specific execution provider (overrides --device)",
    )


def output_file_option(required: bool = False):
    """Output file path (single artifact)."""
    from pathlib import Path

    return click.option(
        "-o",
        "--output",
        required=required,
        type=click.Path(path_type=Path),
        help="Output file path",
    )


def output_dir_option(required: bool = False):
    """Output directory (multi-artifact builds)."""
    from pathlib import Path

    return click.option(
        "--output-dir",
        required=required,
        type=click.Path(path_type=Path),
        help="Output directory for artifacts",
    )


def task_option():
    """Override auto-detected task."""
    return click.option(
        "-t",
        "--task",
        default=None,
        type=str,
        help="Override auto-detected task (e.g., image-classification)",
    )


def precision_option(default: str | None = "auto"):
    """Target precision (string + validator per ADR-8)."""
    return click.option(
        "-p",
        "--precision",
        default=default,
        type=str,
        callback=_validate_precision,
        help="Target precision (e.g., fp32, fp16, int8, w8a16)",
    )
