# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Quantize command for winml CLI.

This module provides the quantize command that inserts QDQ (Quantize-Dequantize)
nodes into ONNX models for quantization-aware inference.

Usage:
    winml quantize --model MODEL [OPTIONS]

Examples:
    winml quantize -m model.onnx
    winml quantize -m model.onnx --precision int8
    winml quantize -m model.onnx -o model_qdq.onnx --samples 100
    winml quantize -m model.onnx --weight-type int8 --activation-type uint8
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import click
from rich.console import Console

from ..utils import cli as cli_utils
from ..utils.logging import configure_logging


if TYPE_CHECKING:
    from typing import Literal


logger = logging.getLogger(__name__)
console = Console()


def _warn_ignored_calibration_options(ctx: click.Context, reason: str) -> None:
    """Warn if the user passed calibration-related CLI options that are ignored."""
    cli_utils.warn_ignored_calibration_options(ctx, reason, console=console)


@click.command()
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input ONNX model file",
)
@cli_utils.output_option("Output path (default: {input}_qdq.onnx)")
@cli_utils.precision_option(
    default=None,
    help_text="Quantization precision: auto, fp16, int4, int8, int16, or w{x}a{y} where "
    "x in {4,8,16}, y in {8,16} (e.g., w4a16, w8a8, w8a16). "
    "int4/w4a16 uses RTN weight-only quantization; "
    "fp16 converts all FP32 tensors to FP16 (no QDQ)",
    optional_message="Overridden by explicit --weight-type/--activation-type",
)
@click.option(
    "--samples",
    type=int,
    default=10,
    help="Number of calibration samples (default: 10)",
)
@click.option(
    "--method",
    type=click.Choice(["minmax", "entropy", "percentile"]),
    default="minmax",
    help="Calibration method (default: minmax)",
)
@click.option(
    "--weight-type",
    type=click.Choice(["uint8", "int8", "uint16", "int16"]),
    default=None,
    help="Weight quantization type. Overrides --precision.",
)
@click.option(
    "--activation-type",
    type=click.Choice(["uint8", "int8", "uint16", "int16"]),
    default=None,
    help="Activation quantization type. Overrides --precision.",
)
@click.option(
    "--per-channel/--no-per-channel",
    default=False,
    show_default=True,
    help="Use per-channel quantization",
)
@click.option(
    "--symmetric/--no-symmetric",
    default=False,
    show_default=True,
    help="Use symmetric quantization",
)
@click.option(
    "--task",
    type=str,
    default=None,
    help="Task for calibration dataset selection (e.g., 'image-classification').",
)
@click.option(
    "--model-name",
    type=str,
    default=None,
    help="HuggingFace model name (e.g., 'microsoft/resnet-50'). When provided "
    "with --task, enables task-aware calibration datasets using the model's preprocessor.",
)
@cli_utils.build_config_option()
@cli_utils.verbosity_options()
@click.pass_context
def quantize(
    ctx: click.Context,
    model: Path,
    output: Path | None,
    precision: str | None,
    samples: int,
    method: str,
    weight_type: str | None,
    activation_type: str | None,
    per_channel: bool,
    symmetric: bool,
    task: str | None,
    model_name: str | None,
    verbose: int,
    quiet: bool,
    config_file: Path | None,
) -> None:
    r"""Quantize ONNX model by inserting QDQ nodes, RTN weight-only, or convert to FP16.

    This command applies quantization to an ONNX model. The algorithm is
    auto-selected from the precision: int4/w4a16 → RTN weight-only,
    int8/int16/w8a8 → static QDQ, fp16 → FP16 conversion.

    \b
    Examples:
        # Basic quantization with defaults (10 samples, uint8)
        winml quantize -m model.onnx

        # Use precision shorthand (same as --weight-type uint8 --activation-type uint8)
        winml quantize -m model.onnx --precision int8

        # RTN 4-bit weight-only quantization (no calibration data needed)
        winml quantize -m model.onnx --precision int4

        # Int16 quantization
        winml quantize -m model.onnx --precision int16

        # Convert model to FP16 (no QDQ, full-model conversion)
        winml quantize -m model.onnx --precision fp16

        # Custom output path and more samples
        winml quantize -m model.onnx -o quantized.onnx --samples 100

        # Explicit types with entropy calibration
        winml quantize -m model.onnx --weight-type int8 --method entropy
    """
    # Merge top-level -v/-q with subcommand-level flags so either position works.
    verbose, quiet = cli_utils.resolve_verbosity(ctx, verbose, quiet)
    configure_logging(verbosity=verbose, quiet=quiet)

    # Apply build config defaults (CLI explicit options take precedence).
    # Only read the JSON for what explicitly specified in config file.
    if config_file is not None:
        _, raw_cfg = cli_utils.load_build_config(config_file)
        qc = raw_cfg.get("quant") or {}
        if not cli_utils.is_cli_provided(ctx, "samples") and "samples" in qc:
            samples = qc["samples"]
        if not cli_utils.is_cli_provided(ctx, "method") and "calibration_method" in qc:
            method = qc["calibration_method"]
        if not cli_utils.is_cli_provided(ctx, "weight_type") and "weight_type" in qc:
            weight_type = qc["weight_type"]
        if not cli_utils.is_cli_provided(ctx, "activation_type") and "activation_type" in qc:
            activation_type = qc["activation_type"]
        if not cli_utils.is_cli_provided(ctx, "per_channel") and "per_channel" in qc:
            per_channel = qc["per_channel"]
        if not cli_utils.is_cli_provided(ctx, "symmetric") and "symmetric" in qc:
            symmetric = qc["symmetric"]
        if not cli_utils.is_cli_provided(ctx, "task") and "task" in qc:
            task = qc["task"]
        if not cli_utils.is_cli_provided(ctx, "model_name") and "model_name" in qc:
            model_name = qc["model_name"]

    # Import quantizer (late import to speed up CLI)
    from ..quant import WinMLQuantizationConfig, quantize_onnx

    # ── Build config based on precision ──────────────────────────
    precision_lower = precision.lower() if precision else None

    if precision_lower == "fp16":
        # FP16 conversion
        _warn_ignored_calibration_options(ctx, "FP16 conversion does not use calibration data.")
        if output is None:
            output = model.parent / f"{model.stem}_fp16.onnx"
        config = WinMLQuantizationConfig(mode="fp16")
        label = "FP16 conversion"

    elif precision_lower and _is_weight_only(precision_lower):
        # RTN weight-only
        from ..config.precision import extract_weight_bits

        _warn_ignored_calibration_options(
            ctx, "RTN weight-only quantization does not use calibration data."
        )
        rtn_bits = extract_weight_bits(precision_lower)
        if output is None:
            output = model.parent / f"{model.stem}_int{rtn_bits}.onnx"
        config = WinMLQuantizationConfig(mode="rtn", rtn_bits=rtn_bits)
        label = f"RTN {rtn_bits}-bit quantization"

    else:
        # QDQ calibrated quantization
        resolved_weight, resolved_activation = _resolve_quant_types(
            precision, weight_type, activation_type
        )
        if output is None:
            output = model.parent / f"{model.stem}_qdq.onnx"
        config = WinMLQuantizationConfig(
            samples=samples,
            calibration_method=cast('Literal["minmax", "entropy", "percentile"]', method),
            weight_type=cast('Literal["uint8", "int8", "uint16", "int16"]', resolved_weight),
            activation_type=cast(
                'Literal["uint8", "int8", "uint16", "int16"]', resolved_activation
            ),
            per_channel=per_channel,
            symmetric=symmetric,
            task=task,
            model_name=model_name,
        )
        label = "Quantization"

        # Display QDQ-specific info
        console.print(f"[bold blue]Weight type:[/bold blue] {resolved_weight}")
        console.print(f"[bold blue]Activation type:[/bold blue] {resolved_activation}")
        console.print(f"[bold blue]Samples:[/bold blue] {samples}")
        console.print(f"[bold blue]Method:[/bold blue] {method}")
        if config.dataset_name:
            _dataset_display = config.dataset_name
        elif config.task and config.task != "random":
            _dataset_display = f"Default for task '{config.task}'"
        else:
            _dataset_display = "Random data (synthetic from ONNX I/O specs)"
        console.print(f"[bold blue]Dataset:[/bold blue] {_dataset_display}")

    # ── Shared execution: print header, run, report ──────────────
    output.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold blue]Input:[/bold blue] {model}")
    console.print(f"[bold blue]Output:[/bold blue] {output}")
    console.print(f"[bold blue]Precision:[/bold blue] {precision or 'auto'}")

    try:
        console.print(f"\n[bold]Running {label.lower()}...[/bold]")
        result = quantize_onnx(model, output_path=output, config=config)

        if result.success:
            console.print(f"\n[bold green]Success![/bold green] {label} complete")
            console.print(f"[dim]Output: {result.output_path}[/dim]")
            if result.nodes_quantized:
                console.print(f"[dim]QDQ nodes inserted: {result.nodes_quantized}[/dim]")
            console.print(f"[dim]Total time: {result.total_time_seconds:.2f}s[/dim]")
        else:
            console.print(f"\n[bold red]{label} failed:[/bold red]")
            for error in result.errors:
                console.print(f"  {error}")
            raise click.ClickException(f"{label} failed")

    except click.ClickException:
        raise
    except Exception as e:
        console.print(f"\n[bold red]{label} failed:[/bold red] {e}")
        logger.exception("%s failed", label)
        raise click.ClickException(f"{label} failed: {e}") from e


def _is_weight_only(precision: str) -> bool:
    """Check if precision requires weight-only (RTN) quantization."""
    from ..config.precision import is_weight_only_precision

    return is_weight_only_precision(precision)


def _resolve_quant_types(
    precision: str | None,
    weight_type: str | None,
    activation_type: str | None,
) -> tuple[str, str]:
    """Resolve weight and activation types from precision and explicit flags.

    Priority: explicit flags > --precision > defaults.
    Aligned with config/precision.py _WEIGHT_TYPE/_ACTIVATION_TYPE mapping.

    Returns:
        Tuple of (weight_type, activation_type).
    """
    from ..config import is_quantized_precision, resolve_quant_types

    if precision and _is_weight_only(precision.lower()):
        # Should not reach here — RTN path returns early above.
        raise click.BadParameter(
            f"'{precision}' is a weight-only precision (use RTN path).",
            param_hint="'-p' / '--precision'",
        )
    if precision and is_quantized_precision(precision):
        default_w, default_a = resolve_quant_types(precision)
    elif precision is None or precision.lower() == "auto":
        default_w, default_a = "uint8", "uint8"
    else:
        raise click.BadParameter(
            f"'{precision}' is not a supported quantization precision. "
            "Accepted: auto, int8, int16, or w{x}a{y} with x,y in {8,16} "
            "(e.g., w8a8, w8a16, w16a16).",
            param_hint="'-p' / '--precision'",
        )

    # Explicit flags override precision defaults
    resolved_w = weight_type if weight_type else default_w
    resolved_a = activation_type if activation_type else default_a

    return resolved_w, resolved_a
