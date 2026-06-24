# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
r"""Measure per-op quantization error (local and cumulative SQNR).

Usage:
    winml debug --float-model float.onnx --quant-model quantized.onnx

Examples:
    # Random inputs (self-contained, no downloads)
    winml debug --float-model model_optimized.onnx --quant-model model_quantized.onnx

    # Real, task-aware calibration inputs
    winml debug --float-model float.onnx --quant-model qdq.onnx \\
        --model-id microsoft/swinv2-tiny-patch4-window16-256 \\
        --task image-classification --samples 16
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ..utils import cli as cli_utils
from ..utils.logging import configure_logging


logger = logging.getLogger(__name__)
console = Console()


@click.command("debug")
@click.option(
    "--float-model",
    "float_model",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Float (pre-quantization) ONNX model.",
)
@click.option(
    "--quant-model",
    "quant_model",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Quantized (QDQ) ONNX model — the build artifact to debug.",
)
@click.option(
    "--samples",
    type=int,
    default=2,
    show_default=True,
    help="Number of input samples to average over.",
)
@click.option(
    "--model-id",
    type=str,
    default=None,
    help="HuggingFace model id for real, task-aware calibration inputs.",
)
@click.option(
    "--task",
    type=str,
    default=None,
    help="Task for task-aware calibration (e.g. 'image-classification'). "
    "Falls back to random inputs when omitted.",
)
@cli_utils.output_option("Write the full per-tensor results to this JSON file.")
@cli_utils.verbosity_options()
@click.pass_context
def debug(
    ctx: click.Context,
    float_model: Path,
    quant_model: Path,
    samples: int,
    model_id: str | None,
    task: str | None,
    output: Path | None,
    verbose: int,
    quiet: bool,
) -> None:
    """Measure per-op quantization error, op by op.

    Runs the float and quantized models over the same inputs and reports, per
    activation, the local SQNR and the cumulative SQNR. Lower dB == more damage.

    Local SQNR is the error from quantizing this tensor alone, excluding
    upstream. Cumulative SQNR is the error at this tensor, including error
    inherited from upstream.
    """
    verbose, quiet = cli_utils.resolve_verbosity(ctx, verbose, quiet)
    configure_logging(verbosity=verbose, quiet=quiet)

    from ..debug import debug_quantization

    console.print(f"[bold blue]Float model:[/bold blue] {float_model}")
    console.print(f"[bold blue]Quant model:[/bold blue] {quant_model}")
    console.print(f"[bold blue]Samples:[/bold blue] {samples}\n")

    result = debug_quantization(
        float_model,
        quant_model,
        samples=samples,
        model_id=model_id,
        task=task,
    )

    print_result(result)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        console.print(f"\n[dim]Full per-tensor results written to {output}[/dim]")

# Number of worst-ranked rows shown per table.
TOP_N = 10


def print_result(result: dict) -> None:
    """Render the debug result dict as console tables."""
    activations = result["activations"]
    weights = result["weights"]
    model_outputs = result["model_outputs"]
    summary = result["summary"]

    console.print(
        "Local SQNR      = error from quantizing this tensor alone, excluding upstream."
    )
    console.print(
        "Cumulative SQNR = error at this tensor, including error inherited from upstream."
    )

    # Model outputs: cumulative SQNR at every graph output (shown in full).
    _render_table(
        "Outputs cumulative SQNR",
        "Output",
        [(o["output_name"], o["cumulative_sqnr_db"]) for o in model_outputs],
    )

    local_sorted = sorted(activations, key=lambda a: a["local_sqnr_db"])
    _render_table(
        f"Top {TOP_N} worst local SQNR",
        "Tensor",
        [(a["tensor_name"], a["local_sqnr_db"]) for a in local_sorted],
        top=TOP_N,
    )
    _print_stats(summary["local"])

    cumulative_sorted = sorted(
        activations,
        key=lambda a: (a["cumulative_sqnr_db"] is None, a["cumulative_sqnr_db"] or 0.0),
    )
    _render_table(
        f"Top {TOP_N} worst cumulative SQNR",
        "Tensor",
        [(a["tensor_name"], a["cumulative_sqnr_db"]) for a in cumulative_sorted],
        top=TOP_N,
    )
    _print_stats(summary["cumulative"])

    weights_sorted = sorted(weights, key=lambda w: w["weight_sqnr_db"])
    _render_table(
        f"Top {TOP_N} worst weight SQNR",
        "Weight",
        [(w["weight_name"], w["weight_sqnr_db"]) for w in weights_sorted],
        top=TOP_N,
    )
    _print_stats(summary["weight"])


def _print_stats(stats: dict) -> None:
    # One-line SQNR summary printed below a table.
    def _fmt(value: float | None) -> str:
        return f"{value:.2f}" if value is not None else "n/a"

    console.print(
        f"(count = {stats['count']}, mean = {_fmt(stats['mean'])}, "
        f"std = {_fmt(stats['std'])}, min = {_fmt(stats['min'])}, "
        f"max = {_fmt(stats['max'])})\n"
    )


def _render_table(
    title: str,
    name_header: str,
    rows: list[tuple[str, float | None]],
    *,
    top: int | None = None,
) -> None:
    table = Table(title=title, title_style="bold", title_justify="left", header_style="bold cyan")
    table.add_column("#", justify="right", style="dim")
    table.add_column("SQNR (dB)", justify="right")
    table.add_column(name_header, overflow="fold")

    shown = rows if top is None else rows[:top]
    for i, (name, sqnr) in enumerate(shown, 1):
        table.add_row(str(i), _fmt_sqnr(sqnr), name)
    console.print(table)



def _fmt_sqnr(value: float | None) -> str:
    if value is None:
        return "[dim]n/a[/dim]"
    color = "red" if value < 20 else "yellow" if value < 40 else "green"
    return f"[{color}]{value:7.2f}[/{color}]"
