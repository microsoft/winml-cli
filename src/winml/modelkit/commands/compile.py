# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Compile command for winml CLI.

This module provides the compile command that compiles ONNX models to
EP-specific formats (e.g., QNN EPContext).

Usage:
    winml compile --model MODEL [OPTIONS]

Examples:
    winml compile -m model.onnx
    winml compile -m model.onnx --device npu
    winml compile -m model.onnx --device gpu --ep migraphx
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console

from ..onnx import is_compiled_onnx
from ..sysinfo import resolve_device, resolve_eps
from ..utils import cli as cli_utils
from ..utils.constants import EP_SUPPORTED_DEVICES, normalize_ep_name


if TYPE_CHECKING:
    from ..utils.constants import EPName, EPNameOrAlias
from ..utils.logging import configure_logging


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
@cli_utils.output_option("Output file path (e.g., model_compiled.onnx)")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: same as input model)",
)
@click.option(
    "--device",
    "-d",
    type=click.Choice(["auto", "npu", "gpu", "cpu"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Target device",
)
@cli_utils.ep_option(
    required=False,
    optional_message="Overrides device-to-provider mapping.",
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
    output: Path | None,
    output_dir: Path | None,
    device: str,
    ep: EPNameOrAlias | None,
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
    EPContext).

    \b
    Examples:
        # Compile for NPU (default, uses QNN/VitisAI)
        winml compile -m model.onnx

        # Compile for NPU with explicit VitisAI EP
        winml compile -m model.onnx --ep vitisai

        # Compile for GPU with MIGraphX
        winml compile -m model.onnx --device gpu --ep migraphx

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

    resolved_device, _ = resolve_device(device, ep=ep)

    # Handle --list
    if list_compilers_flag:
        from ..compiler import list_compilers

        provider = _resolve_compile_provider(resolved_device, ep)
        click.echo(list_compilers(provider))
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

    # Resolve EP from device + ep flags
    provider = _resolve_compile_provider(resolved_device, ep)
    config = WinMLCompileConfig.for_provider(provider, device=resolved_device)

    if config is None:
        raise click.ClickException(
            f"Provider '{provider}' does not support EPContext compilation. "
            "Compile is only supported for providers that produce EPContext models "
            "(e.g. qnn, openvino)."
        )

    config.validate = validate
    config.verbose = verbose

    # Set compiler options
    config.ep_config.compiler = compiler
    config.ep_config.qnn_sdk_root = qnn_sdk_root
    config.ep_config.embed_context = embed

    # Show info
    console.print(f"[bold blue]Input:[/bold blue] {model}")
    console.print(f"[bold blue]Device:[/bold blue] {resolved_device}")
    if ep:
        console.print(f"[bold blue]EP:[/bold blue] {ep}")
    console.print(f"[bold blue]Provider:[/bold blue] {provider}")
    console.print(f"[bold blue]Compiler:[/bold blue] {compiler}")
    if qnn_sdk_root:
        console.print(f"[bold blue]SDK root:[/bold blue] {qnn_sdk_root}")
    # Resolve output path: -o (file) takes precedence over --output-dir
    resolved_output = output or output_dir
    if output:
        console.print(f"[bold blue]Output:[/bold blue] {output}")
    elif output_dir:
        console.print(f"[bold blue]Output dir:[/bold blue] {output_dir}")

    try:
        console.print("\n[bold]Compiling model...[/bold]")
        result = compile_onnx(model, output_path=resolved_output, config=config)

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


def _resolve_compile_provider(resolved_device: str, ep: EPNameOrAlias | None) -> EPName:
    """Resolve the compile provider from device + ep flags.

    ``ep`` overrides the device mapping. Returns
    the canonical EP name (e.g., ``"QNNExecutionProvider"``).
    """
    if ep:
        canonical = normalize_ep_name(ep)
        if canonical is None:
            raise click.UsageError(f"Unknown EP: {ep}")
        supported = EP_SUPPORTED_DEVICES[canonical]
        if resolved_device.lower() not in supported:
            raise click.UsageError(
                f"--ep {ep} cannot run on --device {resolved_device}. "
                f"{canonical} supports: {', '.join(supported)}."
            )
        from ..session.ep_registry import WinMLEPRegistry

        registry = WinMLEPRegistry.get_instance()
        if not registry.is_ep_available(canonical):
            available = [e for e in EP_SUPPORTED_DEVICES if registry.is_ep_available(e)]
            raise click.UsageError(
                f"--ep {ep} ({canonical}) is not registered on this host. "
                f"Available EPs: {', '.join(available) if available else 'none'}."
            )
        return canonical

    eps = resolve_eps(resolved_device)
    if not eps:
        return "CPUExecutionProvider"
    return eps[0]
