# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Compile command for winml CLI.

This module provides the compile command that compiles ONNX models to
EP-specific formats (e.g., QNN EPContext) with optional quantization.

Usage:
    winml compile --model MODEL [OPTIONS]

Examples:
    winml compile -m model.onnx
    winml compile -m model.onnx --device npu
    winml compile -m model.onnx --device gpu --ep migraphx
    winml compile -m model_qdq.onnx --no-quantize
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
from rich.console import Console

from ..onnx import is_compiled_onnx
from ..session import VALID_DEVICES, EPDeviceTarget, resolve_device
from ..session.ep_device import DeviceNotFound, WinMLEPNotDiscovered, WinMLEPRegistrationFailed
from ..utils import cli as cli_utils
from ..utils.logging import configure_logging
from ._ep_arg import EpAtSourceParamType


logger = logging.getLogger(__name__)
console = Console()


@click.command()
@click.option(
    "--model",
    "-m",
    required=False,
    type=click.Path(exists=True, path_type=Path),
    help="Input ONNX model file (required unless --list)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: same as input model)",
)
@click.option(
    "--device",
    "-d",
    type=click.Choice(["auto", *sorted(VALID_DEVICES)], case_sensitive=False),
    default=None,
    help="Target device (default: deduced from --ep, or 'npu' if neither given)",
)
@click.option(
    "--ep",
    type=EpAtSourceParamType(),
    default=None,
    help="Force specific EP, optionally pinned to a source (e.g. 'openvino@pypi'). "
    "Overrides device-to-provider mapping.",
)
@click.option(
    "--quantize/--no-quantize",
    default=True,
    help="Enable/disable quantization (default: enabled)",
)
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Validate compiled model (default: enabled)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose output",
)
@click.option(
    "--compiler",
    type=click.Choice(["ort", "qairt"]),
    default="ort",
    help="Compiler backend (default: ort)",
)
@click.option(
    "--qnn-sdk-root",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to QAIRT SDK root",
)
@click.option(
    "--embed",
    is_flag=True,
    default=False,
    help="Embed EP context in ONNX file (default: external .bin file)",
)
@click.option(
    "--list",
    "list_compilers_flag",
    is_flag=True,
    default=False,
    help="List available compilers for the selected device and exit",
)
@cli_utils.build_config_option
@click.pass_context
def compile(
    ctx: click.Context,
    model: Path | None,
    output_dir: Path | None,
    device: str | None,
    ep: str | None,
    quantize: bool,
    validate: bool,
    verbose: bool,
    compiler: str,
    qnn_sdk_root: Path | None,
    embed: bool,
    list_compilers_flag: bool,
    config_file: Path | None,
) -> None:
    r"""Compile ONNX model to EP-specific format.

    This command compiles an ONNX model to an EP-specific format (e.g., QNN
    EPContext) with optional quantization. For pre-quantized models (containing
    QDQ nodes), use --no-quantize.

    \b
    Examples:
        # Compile for NPU (default, uses QNN/VitisAI)
        winml compile -m model.onnx

        # Compile for NPU with explicit VitisAI EP
        winml compile -m model.onnx --ep vitisai

        # Compile for GPU with MIGraphX
        winml compile -m model.onnx --device gpu --ep migraphx

        # Compile pre-quantized model
        winml compile -m model_qdq.onnx --no-quantize

        # Compile using QAIRT SDK
        winml compile -m model.onnx --compiler qairt --qnn-sdk-root /path/to/sdk
    """
    # Inherit debug mode from parent
    if ctx.obj and ctx.obj.get("debug"):
        verbose = True

    # Apply build config defaults (CLI explicit options take precedence)
    if config_file is not None:
        build_cfg = cli_utils.load_build_config(config_file)
        if build_cfg.compile:
            cc = build_cfg.compile
            if not cli_utils.is_cli_provided(ctx, "ep"):
                ep = cc.ep_config.provider
            if not cli_utils.is_cli_provided(ctx, "compiler"):
                compiler = cc.ep_config.compiler
            if not cli_utils.is_cli_provided(ctx, "embed"):
                embed = cc.ep_config.embed_context
            if not cli_utils.is_cli_provided(ctx, "validate"):
                validate = cc.validate
            if not cli_utils.is_cli_provided(ctx, "verbose"):
                verbose = cc.verbose

    configure_logging(verbose=verbose)

    # Resolve EP+device at the CLI boundary (plan §C / Decision B).
    # device=None or "auto" both signal auto-detect via the resolver.
    _device_arg = "auto" if (device is None or device.lower() == "auto") else device.lower()
    # --ep is parsed by EpAtSourceParamType at click parse time and
    # arrives as (ep, source) or None — feeds the EPDeviceTarget's
    # `source` axis (Scenarios A.5/A.6 per 2_coreloop.md §6.2).
    ep_part, source_part = ep if ep else (None, None)
    try:
        ep_device_resolved = resolve_device(
            EPDeviceTarget(
                ep=ep_part or "auto",
                device=_device_arg,
                source=source_part,
            )
        )
    except DeviceNotFound as e:
        raise click.ClickException(str(e)) from e
    except WinMLEPNotDiscovered as e:
        raise click.ClickException(
            f"EP plugin not found: {e}. Install the required EP package (e.g. onnxruntime-qnn)."
        ) from e
    except WinMLEPRegistrationFailed as e:
        raise click.ClickException(f"EP registration failed: {e}") from e
    except ValueError as e:
        raise click.UsageError(str(e)) from e
    logger.info("Resolved to: %s", ep_device_resolved)

    # Handle --list
    if list_compilers_flag:
        from ..compiler import list_compilers

        click.echo(list_compilers(ep_device_resolved.device))
        return

    # Validate model is provided when not listing
    if model is None:
        raise click.UsageError("Missing option '--model' / '-m'.")

    if is_compiled_onnx(model):
        raise click.ClickException(
            f"{model} is already a compiled EPContext model and cannot be re-compiled. "
            "Run 'winml compile' on the original ONNX model."
        )

    # Import compiler (late import to speed up CLI)
    from ..compiler import WinMLCompileConfig, compile_onnx

    # Build config from the already-resolved EPDeviceTarget (ep_device is never None here).
    config = WinMLCompileConfig.for_ep_device(ep_device_resolved)

    config.validate = validate
    config.verbose = verbose

    # Set compiler options
    config.ep_config.compiler = compiler
    config.ep_config.qnn_sdk_root = qnn_sdk_root
    config.ep_config.embed_context = embed

    # Deprecation notice for --no-quantize
    if not quantize:
        console.print(
            "[yellow]Note:[/yellow] --no-quantize has no effect. "
            "Quantization is no longer performed during compile. "
            "Use 'winml quantize' before 'winml compile' to control quantization."
        )

    # Show info — device and provider come directly from the resolved EPDeviceTarget.
    console.print(f"[bold blue]Input:[/bold blue] {model}")
    console.print(f"[bold blue]Device:[/bold blue] {ep_device_resolved.device}")
    if ep:
        console.print(f"[bold blue]EP:[/bold blue] {ep}")
    console.print(f"[bold blue]Provider:[/bold blue] {ep_device_resolved.ep}")
    console.print(f"[bold blue]Compiler:[/bold blue] {compiler}")
    if qnn_sdk_root:
        console.print(f"[bold blue]SDK root:[/bold blue] {qnn_sdk_root}")
    if output_dir:
        console.print(f"[bold blue]Output dir:[/bold blue] {output_dir}")

    try:
        console.print("\n[bold]Compiling model...[/bold]")
        result = compile_onnx(model, output_path=output_dir, config=config)

        if result.success:
            if config.ep_config.enable_ep_context and not result.output_path:
                console.print(
                    "\n[bold yellow]Warning:[/bold yellow] Compilation finished "
                    "but no output file was written to the output directory."
                )
                raise click.ClickException(
                    "No output file produced. Check EP context support for "
                    f"provider '{config.ep_config.provider}'."
                )
            console.print("\n[bold green]Success![/bold green] Model compiled")
            if result.output_path:
                console.print(f"[dim]Output: {result.output_path}[/dim]")
            if result.compile_time:
                console.print(f"[dim]Compile time: {result.compile_time:.2f}s[/dim]")
            if result.total_time:
                console.print(f"[dim]Total time: {result.total_time:.2f}s[/dim]")
        else:
            console.print("\n[bold red]Compilation failed:[/bold red]")
            for error in result.errors:
                console.print(f"  {error}")
            raise click.ClickException("Compilation failed")

    except click.ClickException:
        raise
    except Exception as e:
        console.print(f"\n[bold red]Compilation failed:[/bold red] {e}")
        logger.exception("Compilation failed")
        raise click.ClickException(f"Compilation failed: {e}") from e
