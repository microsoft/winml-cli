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
from typing import TYPE_CHECKING, cast

import click
from rich.console import Console

from ..onnx import is_compiled_onnx
from ..session import (
    VALID_DEVICES,
    DeviceNotFound,
    EPDeviceTarget,
    WinMLEPNotDiscovered,
    WinMLEPRegistrationFailed,
    available_eps_for_device,
    resolve_device,
)
from ..utils import cli as cli_utils
from ..utils.constants import COMPILER_NAMES, ORT_SESSION_COMPILER, normalize_ep_name


if TYPE_CHECKING:
    from ..utils.constants import CompilerName, EPName, EPNameOrAlias
from ..utils.logging import configure_logging
from ._ep_arg import EpAtSourceParamType


def _resolve_compile_provider(resolved_device: str, ep: EPNameOrAlias | None) -> EPName:
    """Resolve the compile provider from device + ep flags.

    ``ep`` overrides the device mapping. Returns the canonical EP name.
    Device/EP compatibility is enforced upstream; this only normalizes.
    """
    if ep is not None:
        normalized = normalize_ep_name(ep)
        if normalized is not None:
            return normalized
    return cast("EPName", available_eps_for_device(resolved_device)[0])


logger = logging.getLogger(__name__)
console = Console()


@click.command()
@cli_utils.model_path_option(
    required=False,
    multiple=True,
    help_text="Input ONNX model file. Repeat -m to compile multiple models with a shared "
    "EP context (weight sharing). Required unless --list.",
)
@cli_utils.output_option("Output file path (e.g., model_compiled.onnx)")
@cli_utils.overwrite_option()
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
    "--validate/--no-validate",
    default=True,
    help="Validate compiled model (default: enabled)",
)
@click.option(
    "--compiler",
    type=click.Choice(list(COMPILER_NAMES)),
    default="ort",
    help="Compiler backend (default: ort). 'ort_session' compiles via "
    "ort.InferenceSession (ep.context_enable) — required for shared-context multi-model.",
)
@click.option(
    "--qnn-sdk-root",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to QAIRT SDK root",
)
@click.option(
    "--embed/--no-embed",
    default=False,
    show_default=True,
    help="Embed EP context in ONNX file (default: external .bin file)",
)
@click.option(
    "--list",
    "list_compilers_flag",
    is_flag=True,
    default=False,
    help="List available compilers for the selected device and exit",
)
@cli_utils.build_config_option()
@cli_utils.verbosity_options()
@cli_utils.no_color_option()
@click.pass_context
def compile(
    ctx: click.Context,
    model: tuple[Path, ...],
    output: Path | None,
    output_dir: Path | None,
    overwrite: bool,
    device: str | None,
    ep: tuple[str, str | None] | None,
    validate: bool,
    verbose: int,
    quiet: bool,
    compiler: CompilerName,
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
    # Merge top-level -v/-q with subcommand-level flags so either position works.
    verbose, quiet = cli_utils.resolve_verbosity(ctx, verbose, quiet)

    # --ep is parsed by EpAtSourceParamType at click parse time and
    # arrives as (ep, source) or None — feeds the EPDeviceTarget's
    # `source` axis (Scenarios A.5/A.6 per 2_coreloop.md §6.2). Unpack
    # at entry so the local ``ep`` is a bare ``str | None`` for the rest
    # of the function — this keeps the Rich pre-run block from leaking
    # the raw tuple into user-visible output (T-01 regression pin).
    ep_part, source_part = ep if ep else (None, None)
    ep_name: str | None = ep_part

    # Apply build config defaults (CLI explicit options take precedence).
    # Read raw JSON so missing keys are distinguishable from dataclass defaults.
    config_provider_options: dict[str, str] = {}
    if config_file is not None:
        _, raw_cfg = cli_utils.load_build_config(config_file)
        cc = raw_cfg.get("compile") or {}
        # EP provider options (e.g. QNN htp_arch/soc_model/vtcm_mb) for the compile session.
        if "provider_options" in cc:
            config_provider_options = dict(cc["provider_options"])
        if not cli_utils.is_cli_provided(ctx, "ep") and "execution_provider" in cc:
            ep_name = cc["execution_provider"]
        if not cli_utils.is_cli_provided(ctx, "compiler") and "compiler" in cc:
            compiler = cc["compiler"]
        if not cli_utils.is_cli_provided(ctx, "embed") and "embed_context" in cc:
            embed = cc["embed_context"]
        if not cli_utils.is_cli_provided(ctx, "validate") and "validate" in cc:
            validate = cc["validate"]
        # Config-file verbosity fallback. CLI flags always win: only honor the
        # build config's `verbose` when the user gave no verbosity on either CLI
        # position (resolve_verbosity above already merged top-level + subcommand
        # -v, so a merged 0 means "none on the CLI") and did not ask for --quiet.
        # ``int`` maps both `true`->1 (INFO) and an explicit count (e.g. 2->DEBUG).
        # Currently compile-only; tracked for all commands in
        # https://github.com/microsoft/winml-cli/issues/799
        if verbose == 0 and not quiet and "verbose" in cc:
            verbose = int(cc["verbose"])

    configure_logging(verbosity=verbose, quiet=quiet)

    # Resolve EP+device at the CLI boundary (plan §C / Decision B).
    # device=None or "auto" both signal auto-detect via the resolver.
    _device_arg = "auto" if (device is None or device.lower() == "auto") else device.lower()
    try:
        ep_device_resolved = resolve_device(
            EPDeviceTarget(
                ep=ep_name or "auto",
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

        click.echo(list_compilers(cast("EPName", ep_device_resolved.device)))
        return

    # Validate model(s) provided when not listing
    if not model:
        raise click.UsageError("Missing option '--model' / '-m'.")
    models = list(model)

    for m in models:
        if is_compiled_onnx(m):
            raise click.ClickException(
                f"{m} is already a compiled EPContext model and cannot be re-compiled. "
                "Run 'winml compile' on the original ONNX model."
            )

    # Multiple models share one EP context and are written by filename into a
    # directory, so a single -o/--output file path is ambiguous: require --output-dir
    # (and forbid -o/--output).
    if len(models) > 1 and (output is not None or output_dir is None):
        raise click.UsageError(
            "Multiple --model inputs are written by filename into a directory; "
            "pass --output-dir (and not -o/--output)."
        )

    # Import compiler (late import to speed up CLI)
    from ..compiler import WinMLCompileConfig, compile_multiple_onnx, compile_onnx
    from ..session import expand_ep_name

    # Build config from the already-resolved EPDeviceTarget (ep_device is never None here).
    config = WinMLCompileConfig.for_ep_device(ep_device_resolved)
    if config is None:
        provider = expand_ep_name(ep_device_resolved.ep)
        raise click.ClickException(
            f"Provider '{provider}' does not support EPContext compilation. "
            "Compile is only supported for providers that produce EPContext models "
            "(e.g. qnn, openvino)."
        )

    config.validate = validate
    config.verbose = bool(verbose)

    # Set compiler options. The compiler choice selects the backend:
    # "ort_session" -> ort.InferenceSession, else ort.ModelCompiler / qairt.
    config.ep_config.compiler = compiler
    config.ep_config.qnn_sdk_root = qnn_sdk_root
    config.ep_config.embed_context = embed
    # EP provider options supplied via --config (compile.provider_options).
    if config_provider_options:
        config.ep_config.provider_options.update(config_provider_options)

    # Show info — device and provider come directly from the resolved EPDeviceTarget.
    console.print(f"[bold blue]Input:[/bold blue] {', '.join(str(m) for m in models)}")
    console.print(f"[bold blue]Device:[/bold blue] {ep_device_resolved.device}")
    if ep_name:
        console.print(f"[bold blue]EP:[/bold blue] {ep_name}")
    console.print(f"[bold blue]Provider:[/bold blue] {ep_device_resolved.ep}")
    console.print(f"[bold blue]Compiler:[/bold blue] {compiler}")
    if len(models) > 1:
        console.print(f"[bold blue]Shared EP context:[/bold blue] yes ({len(models)} models)")
    if qnn_sdk_root:
        console.print(f"[bold blue]SDK root:[/bold blue] {qnn_sdk_root}")
    # Resolve output path: -o (file) takes precedence over --output-dir
    resolved_output = output or output_dir
    # Refuse to clobber an existing output unless the user opted in. A file
    # blocks when it exists; a directory blocks only when non-empty.
    cli_utils.guard_output(resolved_output, overwrite)
    if output:
        console.print(f"[bold blue]Output:[/bold blue] {output}")
    elif output_dir:
        console.print(f"[bold blue]Output dir:[/bold blue] {output_dir}")

    try:
        console.print("\n[bold]Compiling model(s)...[/bold]")
        if len(models) == 1 and compiler != ORT_SESSION_COMPILER:
            # Default path: single model via ort.ModelCompiler (staged pipeline).
            results = [compile_onnx(models[0], output_path=resolved_output, config=config)]
        else:
            # Multi-model (shared EP context) and/or inference-session backend.
            # Multiple models require --output-dir (a directory, enforced above); a
            # single inference_session model may use -o (a file) or --output-dir.
            results = compile_multiple_onnx(models, resolved_output, config)

        # Report every model's result (not just the first failure).
        multi = len(results) > 1
        failures = 0
        for model_path, result in zip(models, results, strict=True):
            label = f" — {model_path.name}" if multi else ""
            if result.success:
                if config.ep_config.enable_ep_context and not result.output_path:
                    # Compiled but no artifact landed: a warning, not a failure.
                    console.print(
                        "\n[bold yellow]Warning:[/bold yellow] Compilation finished but "
                        f"no output file was written to the output directory.{label}"
                    )
                    continue
                console.print(f"\n[bold green]Success![/bold green] Model compiled{label}")
                if result.output_path:
                    console.print(f"[dim]Output: {result.output_path}[/dim]")
                if result.compile_time:
                    console.print(f"[dim]Compile time: {result.compile_time:.2f}s[/dim]")
                if result.total_time:
                    console.print(f"[dim]Total time: {result.total_time:.2f}s[/dim]")
            else:
                failures += 1
                console.print(f"\n[bold red]Compilation failed:[/bold red]{label}")
                for error in result.errors:
                    console.print(f"  {error}")

        if failures:
            raise click.ClickException(
                f"Compilation failed for {failures} of {len(results)} model(s)."
                if multi
                else "Compilation failed"
            )

    except click.ClickException:
        raise
    except Exception as e:
        console.print(f"\n[bold red]Compilation failed:[/bold red] {e}")
        logger.exception("Compilation failed")
        raise click.ClickException(f"Compilation failed: {e}") from e
