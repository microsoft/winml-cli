# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CLI for compiler module."""

from __future__ import annotations

from pathlib import Path

import click

from .configs import (
    CalibrationConfig,
    EPConfig,
    QDQConfig,
    WinMLCompileConfig,
)


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """ONNX model compiler for execution providers."""


@cli.command()
@click.option(
    "-m",
    "--model",
    "model_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input ONNX model path",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    help="Output path for compiled model",
)
@click.option(
    "-w",
    "--work-dir",
    type=click.Path(path_type=Path),
    help="Working directory for intermediate files",
)
@click.option(
    "--ep",
    "--execution-provider",
    "execution_provider",
    type=click.Choice(["qnn", "cpu", "cuda", "dml"]),
    default="qnn",
    help="Target execution provider",
    show_default=True,
)
@click.option(
    "--quantize/--no-quantize",
    default=True,
    help="Apply QDQ quantization",
    show_default=True,
)
@click.option(
    "--calibration-method",
    type=click.Choice(["minmax", "entropy", "percentile"]),
    default="minmax",
    help="Calibration algorithm",
    show_default=True,
)
@click.option(
    "--calibration-samples",
    type=int,
    default=100,
    help="Number of random calibration samples",
    show_default=True,
)
@click.option(
    "--calibration-load",
    type=click.Path(exists=True, path_type=Path),
    help="Load calibration data from file",
)
@click.option(
    "--calibration-save",
    type=click.Path(path_type=Path),
    help="Save calibration data to file",
)
@click.option(
    "--weight-type",
    type=click.Choice(["int8", "uint8", "int16", "uint16"]),
    default="int8",
    help="Weight quantization type",
    show_default=True,
)
@click.option(
    "--activation-type",
    type=click.Choice(["int8", "uint8", "int16", "uint16"]),
    default="uint8",
    help="Activation quantization type",
    show_default=True,
)
@click.option(
    "--per-channel/--no-per-channel",
    default=False,
    help="Use per-channel quantization",
    show_default=True,
)
@click.option(
    "--ep-context/--no-ep-context",
    "enable_ep_context",
    default=True,
    help="Generate EPContext model",
    show_default=True,
)
@click.option(
    "--provider-option",
    multiple=True,
    help="EP-specific option (key=value), can be used multiple times",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def compile(
    model_path: Path,
    output_path: Path | None,
    work_dir: Path | None,
    execution_provider: str,
    quantize: bool,
    calibration_method: str,
    calibration_samples: int,
    calibration_load: Path | None,
    calibration_save: Path | None,
    weight_type: str,
    activation_type: str,
    per_channel: bool,
    enable_ep_context: bool,
    provider_option: tuple[str, ...],
    verbose: bool,
) -> None:
    r"""Compile ONNX model to EP-specific format.

    Examples:
        # Basic QNN compile with defaults
        python -m modelkit.compiler compile -m model.onnx --ep qnn

        # Skip quantization (pre-quantized model)
        python -m modelkit.compiler compile -m model_qdq.onnx --no-quantize

        # Custom calibration
        python -m modelkit.compiler compile -m model.onnx \
            --calibration-samples 500 \
            --calibration-save calibration.json

        # Load pre-computed calibration
        python -m modelkit.compiler compile -m model.onnx \
            --calibration-load calibration.json
    """
    from .compiler import compile_onnx

    # Parse provider options
    provider_options = {}
    for opt in provider_option:
        if "=" in opt:
            key, value = opt.split("=", 1)
            provider_options[key] = value
        else:
            click.echo(f"Warning: Invalid provider option '{opt}' (expected key=value)")

    # Build config
    ep_config = EPConfig(
        provider=execution_provider,
        provider_options=provider_options,
        enable_ep_context=enable_ep_context,
    )

    qdq_config = None
    calibration_config = None

    if quantize:
        qdq_config = QDQConfig(
            weight_type=weight_type,
            activation_type=activation_type,
            per_channel=per_channel,
        )
        calibration_config = CalibrationConfig(
            method=calibration_method,
            samples=calibration_samples,
            load_path=calibration_load,
            save_path=calibration_save,
        )

    config = WinMLCompileConfig(
        ep_config=ep_config,
        qdq_config=qdq_config,
        calibration_config=calibration_config,
        verbose=verbose,
    )

    # Run compilation (output_path passed as separate argument)
    try:
        result = compile_onnx(model_path=model_path, output_path=output_path, config=config)

        # Print result
        if result.success:
            click.echo(click.style("Compilation successful!", fg="green"))
            click.echo(f"Output: {result.output_path}")
            click.echo(f"Total time: {result.total_time:.2f}s")

            if result.calibration_time:
                click.echo(f"  Calibration: {result.calibration_time:.2f}s")
            if result.qdq_time:
                click.echo(f"  QDQ: {result.qdq_time:.2f}s")
            if result.compile_time:
                click.echo(f"  Compile: {result.compile_time:.2f}s")

            if result.warnings:
                click.echo(click.style("\nWarnings:", fg="yellow"))
                for warning in result.warnings:
                    click.echo(f"  - {warning}")
        else:
            click.echo(click.style("Compilation failed!", fg="red"))
            for error in result.errors:
                click.echo(f"  Error: {error}")
            raise SystemExit(1)

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))
        if verbose:
            import traceback

            traceback.print_exc()
        raise SystemExit(1) from None


@cli.command()
@click.option(
    "-m",
    "--model",
    "model_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input ONNX model path",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for calibration data (JSON)",
)
@click.option(
    "--method",
    type=click.Choice(["minmax", "entropy", "percentile"]),
    default="minmax",
    help="Calibration method",
)
@click.option(
    "--samples",
    type=int,
    default=100,
    help="Number of calibration samples",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def calibrate(
    model_path: Path,
    output_path: Path,
    method: str,
    samples: int,
    verbose: bool,
) -> None:
    """Run calibration only and save to file.

    This command runs only the calibration stage and saves the
    calibration data to a JSON file for later use.

    Examples:
        # Generate calibration data
        python -m modelkit.compiler calibrate -m model.onnx -o calibration.json

        # Use more samples
        python -m modelkit.compiler calibrate -m model.onnx -o calibration.json --samples 500
    """
    import tempfile

    from .context import CompileContext
    from .stages.calibrate import CalibrateStage
    from .stages.detect import DetectStage

    click.echo(f"Calibrating {model_path}...")
    click.echo(f"Method: {method}, Samples: {samples}")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            context = CompileContext(
                model_path=model_path,
                config={
                    "quantize": True,
                    "calibration_method": method,
                    "calibration_samples": samples,
                    "calibration_save_path": str(output_path),
                },
                work_dir=Path(temp_dir),
                verbose=verbose,
            )

            # Run detect
            detect = DetectStage()
            if detect.should_run(context):
                context = detect.process(context)

            # Run calibrate
            calibrate_stage = CalibrateStage()
            if calibrate_stage.should_run(context):
                context = calibrate_stage.process(context)

        if context.has_error:
            click.echo(click.style("Calibration failed!", fg="red"))
            for error in context.errors:
                click.echo(f"  Error: {error}")
            raise SystemExit(1)

        click.echo(click.style("Calibration successful!", fg="green"))
        click.echo(f"Saved to: {output_path}")

    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"))
        if verbose:
            import traceback

            traceback.print_exc()
        raise SystemExit(1) from None


@cli.command()
@click.option(
    "-m",
    "--model",
    "model_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input ONNX model path",
)
def info(model_path: Path) -> None:
    """Show model information.

    Displays input/output shapes and basic model statistics.
    """
    from ..onnx import load_onnx

    model = load_onnx(model_path, validate=False)

    click.echo(f"Model: {model_path}")
    click.echo(f"IR Version: {model.ir_version}")
    click.echo(f"Opset: {model.opset_import[0].version}")

    click.echo("\nInputs:")
    for inp in model.graph.input:
        shape = [d.dim_value or "?" for d in inp.type.tensor_type.shape.dim]
        click.echo(f"  {inp.name}: {shape}")

    click.echo("\nOutputs:")
    for out in model.graph.output:
        shape = [d.dim_value or "?" for d in out.type.tensor_type.shape.dim]
        click.echo(f"  {out.name}: {shape}")

    # Count ops
    op_counts: dict[str, int] = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    click.echo(f"\nNodes: {len(model.graph.node)}")
    click.echo("Top operations:")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1])[:10]:
        click.echo(f"  {op}: {count}")

    # Check for Q/DQ
    from .utils import QDQ_OP_TYPES

    has_qdq = any(node.op_type in QDQ_OP_TYPES for node in model.graph.node)
    click.echo(f"\nQuantized: {has_qdq}")


@cli.command("list-providers")
def list_providers() -> None:
    """List available execution providers."""
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        click.echo("Available execution providers:")
        for provider in available:
            click.echo(f"  - {provider}")
    except ImportError:
        click.echo("Error: onnxruntime not installed")


if __name__ == "__main__":
    cli()
