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

import json
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
    resolve_device,
)
from ..utils import cli as cli_utils
from ..utils.constants import COMPILER_NAMES, ORT_SESSION_COMPILER, normalize_ep_name


if TYPE_CHECKING:
    from typing import Literal, TypeGuard

    from ..utils.constants import CompilerName, EPAlias, EPName
from ..utils.logging import configure_logging
from ._ep_arg import EpAtSourceParamType


logger = logging.getLogger(__name__)
console = Console()


def _is_epcontext_compiler(value: str) -> TypeGuard[CompilerName]:
    return value in COMPILER_NAMES


@click.command()
@cli_utils.model_path_option(
    required=False,
    multiple=True,
    help_text="Input model file. "
    "Repeat -m to compile multiple ONNX models with a shared "
    "EP context (weight sharing). Required unless listing compilers, EPs, or targets.",
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
@cli_utils.device_luid_option(
    optional_message="Only supported for CGIR to CG Output IR compilation."
)
@click.option(
    "--ep",
    type=EpAtSourceParamType(),
    default=None,
    help="Force specific EP, optionally pinned to a source (e.g. 'openvino@pypi'). "
    "Overrides device-to-provider mapping.",
)
@cli_utils.ep_options_option(
    optional_message="Applied to the compilation session; overrides matching "
    "compile.provider_options keys from --config. "
    "VitisAI off-device compilation requires target=X2.",
)
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Request post-compilation validation. With --compiler offdevice, "
    "embedded-artifact checks always run, including with --no-validate; "
    "target-device inference is skipped.",
)
@click.option(
    "--compiler",
    type=click.Choice([*COMPILER_NAMES, "cgc"]),
    default="ort",
    help="Compiler backend (default: ort). 'ort_session' compiles via "
    "ort.InferenceSession (ep.context_enable) — required for shared-context multi-model. "
    "'offdevice' compiles without matching target hardware using ORT ModelCompiler. "
    "'cgc' compiles CGIR to CG Output IR.",
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
    help="Embed EP context in ONNX file (default: external .bin file). "
    "Off-device compilation always embeds.",
)
@click.option(
    "--list",
    "list_compilers_flag",
    is_flag=True,
    default=False,
    help="List available compilers for the selected device and exit",
)
@click.option(
    "--target-platform",
    "target",
    type=str,
    default=None,
    help="Deployment platform: OpenVINO virtual NPU platform ID or QNN numeric "
    "soc_model ID from QAIRT. Requires --compiler offdevice.",
)
@click.option(
    "--list-eps",
    is_flag=True,
    help="List implemented off-device compilation EPs and payload availability as JSON.",
)
@click.option(
    "--list-targets",
    is_flag=True,
    help="List advertised deployment platforms for the selected off-device EP, "
    "or explain how to specify targets that cannot be enumerated.",
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
    device_luid: str | None,
    ep: tuple[str, str | None] | None,
    ep_options: tuple[str, ...],
    validate: bool,
    verbose: int,
    quiet: bool,
    compiler: CompilerName | Literal["cgc"],
    qnn_sdk_root: Path | None,
    embed: bool,
    list_compilers_flag: bool,
    target: str | None,
    list_eps: bool,
    list_targets: bool,
    config_file: Path | None,
) -> None:
    r"""Compile ONNX to EPContext, or CGIR to CG Output IR.

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

        # Compile off-device for a platform returned by --list-targets
        winml compile --compiler offdevice --ep openvino --target-platform 4000 -m model.onnx

        # Compile CGIR to device-targeted CG Output IR
        winml compile -m input.mlir --compiler cgc --device gpu -o output.mlir
    """
    # Merge top-level -v/-q with subcommand-level flags so either position works.
    verbose, quiet = cli_utils.resolve_verbosity(ctx, verbose, quiet)

    compiler_was_provided = cli_utils.is_cli_provided(ctx, "compiler")
    inferred_cgc = (
        not compiler_was_provided
        and len(model) == 1
        and model[0].suffix.casefold() in {".mlir", ".mlirbc"}
    )
    ir_mode = compiler == "cgc" or inferred_cgc
    if ir_mode:
        incompatible = {
            "--ep": ep is not None,
            "--ep-options": bool(ep_options),
            "--qnn-sdk-root": qnn_sdk_root is not None,
            "--embed/--no-embed": cli_utils.is_cli_provided(ctx, "embed"),
            "--list": list_compilers_flag,
            "--target-platform": target is not None,
            "--list-eps": list_eps,
            "--list-targets": list_targets,
            "--config": config_file is not None,
        }
        rejected = [name for name, present in incompatible.items() if present]
        if rejected:
            raise click.UsageError(
                f"CG Output IR compilation does not support: {', '.join(rejected)}."
            )
        if len(model) != 1:
            raise click.UsageError("CG Output IR compilation requires exactly one --model.")
        if device not in {None, "gpu"}:
            raise click.UsageError("CG Output IR compilation currently requires --device gpu.")
        from ..compiler import compile_cgc_ir

        source = model[0]
        destination = output or (output_dir or source.parent) / f"{source.stem}_output.mlir"
        configure_logging(verbosity=verbose, quiet=quiet)
        try:
            result_path = compile_cgc_ir(
                source,
                destination,
                device_luid=device_luid,
                validate=validate,
                overwrite=overwrite,
            )
        except OSError as e:
            raise click.ClickException(f"CGC compilation failed: {e}") from e
        console.print(f"Output: {result_path}")
        console.print("Format: MLIR text; form: DEVICE_TARGETED (CG Output IR).")
        console.print(
            "Validation: reload, pipeline build and I/O schema passed (no inference)."
            if validate
            else "Validation: skipped by --no-validate."
        )
        return
    if device_luid is not None:
        raise click.UsageError("--device-luid is only supported for CG Output IR compilation.")

    # --ep is parsed by EpAtSourceParamType at click parse time and
    # arrives as (ep, source) or None — feeds the EPDeviceTarget's
    # `source` axis (Scenarios A.5/A.6 per 2_coreloop.md §6.2). Unpack
    # at entry so the local ``ep`` is a bare ``str | None`` for the rest
    # of the function — this keeps the Rich pre-run block from leaking
    # the raw tuple into user-visible output (T-01 regression pin).
    ep_part, source_part = ep if ep else (None, None)
    ep_name: str | None = ep_part
    cli_provider_options = cli_utils.parse_ep_options(ep_options)

    # Apply build config defaults (CLI explicit options take precedence).
    # Read raw JSON so missing keys are distinguishable from dataclass defaults.
    config_provider_options: dict[str, str] = {}
    config_provider_option_file_keys: set[str] = set()
    config_compile_keys: set[str] = set()
    if config_file is not None:
        try:
            build_cfg, raw_cfg = cli_utils.load_build_config(config_file)
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            raise click.UsageError(f"Invalid compile configuration: {e}") from e
        cc = raw_cfg.get("compile") or {}
        config_compile_keys.update(cc)
        configured_target = (
            build_cfg.compile.ep_device
            if build_cfg.compile is not None and "ep_device" in cc
            else None
        )
        # EP provider options (e.g. QNN htp_arch/soc_model/vtcm_mb) for the compile session.
        if "provider_options" in cc:
            config_provider_options = dict(cc["provider_options"])
        if "provider_option_file_keys" in cc:
            config_provider_option_file_keys = set(cc["provider_option_file_keys"])
        if not cli_utils.is_cli_provided(ctx, "device"):
            if configured_target is not None:
                device = configured_target.device
            if "device" in cc:
                device = cc["device"]
        if not cli_utils.is_cli_provided(ctx, "ep"):
            if configured_target is not None:
                ep_name = configured_target.ep
                source_part = configured_target.source
            if "execution_provider" in cc:
                ep_name = cc["execution_provider"]
        if not cli_utils.is_cli_provided(ctx, "compiler") and "compiler" in cc:
            compiler = cc["compiler"]
        if not cli_utils.is_cli_provided(ctx, "target") and "target_platform" in cc:
            target = cc["target_platform"]
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

    off_device = compiler == "offdevice"
    if (list_eps or list_targets) and not off_device:
        raise click.UsageError(
            "--list-eps and --list-targets require --compiler offdevice."
        )
    if list_eps and list_targets:
        raise click.UsageError("--list-eps and --list-targets are mutually exclusive.")
    if target is not None and not off_device:
        raise click.UsageError("--target-platform requires --compiler offdevice.")
    if target is not None and not isinstance(target, str):
        raise click.UsageError("compile.target_platform must be a string.")

    if not _is_epcontext_compiler(compiler):
        raise click.UsageError(
            "CGC compilation requires a single .mlir/.mlirbc model or explicit "
            "--compiler cgc; it cannot be selected through EPContext configuration."
        )

    configure_logging(verbosity=verbose, quiet=quiet)

    # Resolve EP+device at the CLI boundary (plan §C / Decision B).
    # device=None or "auto" both signal auto-detect via the resolver.
    if device is not None and not isinstance(device, str):
        raise click.UsageError(f"compile.device must be a string, got {type(device).__name__}")
    if ep_name is not None and not isinstance(ep_name, str):
        raise click.UsageError(
            f"compile.execution_provider must be a string, got {type(ep_name).__name__}"
        )
    if off_device:
        if list_compilers_flag:
            raise click.UsageError("Off-device mode cannot be combined with --list.")
        if device is not None and device.lower() not in {"auto", "npu"}:
            raise click.UsageError("Off-device mode requires --device npu.")
        if qnn_sdk_root is not None:
            raise click.UsageError(
                "Off-device mode uses ORT ModelCompiler and does not accept "
                "--qnn-sdk-root."
            )
        is_vitisai = normalize_ep_name(ep_name) == "VitisAIExecutionProvider"
        provider_options = {**config_provider_options, **(cli_provider_options or {})}
        if config_provider_option_file_keys or (
            provider_options
            and (
                list_eps
                or list_targets
                or not is_vitisai
                or set(provider_options) - {"target"}
            )
        ):
            raise click.UsageError("Off-device mode does not accept provider options.")
        if list_eps or list_targets:
            if model:
                raise click.UsageError("Off-device enumeration does not accept --model.")
            if target is not None:
                raise click.UsageError(
                    "Off-device enumeration does not accept --target-platform."
                )
            if list_targets and ep_name in {None, "auto"}:
                raise click.UsageError("--list-targets requires --ep.")
            from ..compiler import list_compilation_eps, list_compilation_targets

            try:
                listing = (
                    list_compilation_targets(ep_name, source=source_part)
                    if list_targets and ep_name is not None
                    else list_compilation_eps(ep_name, source=source_part)
                )
            except Exception as e:
                logger.exception("Off-device enumeration failed")
                raise click.ClickException(str(e)) from e
            click.echo(json.dumps(listing, indent=2))
            return

        if len(model) != 1:
            raise click.UsageError("Off-device compilation requires exactly one --model.")
        if not ep_name or ep_name.lower() == "auto":
            raise click.UsageError("Off-device compilation requires --ep.")
        if is_vitisai:
            if target is not None:
                raise click.UsageError(
                    "VitisAI selects a compiler backend, not a deployment platform. "
                    "Use --ep-options target=X2 instead of --target-platform."
                )
            target = provider_options.get("target")
            if target != "X2":
                raise click.UsageError(
                    "VitisAI off-device compilation requires --ep-options target=X2."
                )
        elif target is None or not target.strip():
            raise click.UsageError(
                "Off-device compilation requires --target-platform: "
                "an OpenVINO virtual platform ID "
                "or a QNN numeric soc_model ID from QAIRT."
            )
        if not embed and (
            cli_utils.is_cli_provided(ctx, "embed") or "embed_context" in config_compile_keys
        ):
            raise click.UsageError("Off-device compilation requires embedded context. Use --embed.")
        from ..compiler import EPConfig, WinMLCompileConfig, compile_onnx

        destination = output or (output_dir or model[0].parent) / f"{model[0].stem}_ctx.onnx"
        cli_utils.guard_output(destination, overwrite)
        config = WinMLCompileConfig(
            ep_config=EPConfig(
                provider=cast("EPAlias", ep_name),
                provider_options=provider_options,
                compiler=compiler,
                device="npu",
                embed_context=True,
                target_platform=None if is_vitisai else target,
            ),
            ep_device=EPDeviceTarget(ep=ep_name, device="npu", source=source_part),
            validate=validate,
            verbose=bool(verbose),
        )
        try:
            result = compile_onnx(
                model[0],
                output_path=destination,
                config=config,
                overwrite=overwrite,
            )
            if not result.success or result.output_path is None:
                detail = "; ".join(result.errors) or "No compiled artifact was produced."
                raise RuntimeError(detail)
        except Exception as e:
            logger.exception("Off-device compilation failed")
            raise click.ClickException(f"Off-device compilation failed: {e}") from e
        console.print("[bold green]Success![/bold green] Model compiled off-device")
        target_label = "Compiler backend" if is_vitisai else "Target platform"
        console.print(f"[dim]{target_label}: {target}[/dim]")
        console.print(f"[dim]Output: {result.output_path}[/dim]")
        console.print("Artifact validation: passed (embedded-context checks)")
        for warning in result.warnings:
            console.print(warning, markup=False)
        return

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
            f"EP plugin not found: {e}. Verify that the requested EP is available "
            "through Windows ML EP Catalog or install a compatible BYO plugin."
        ) from e
    except WinMLEPRegistrationFailed as e:
        raise click.ClickException(f"EP registration failed: {e}") from e
    except ValueError as e:
        raise click.UsageError(str(e)) from e
    logger.info("Resolved to: %s", ep_device_resolved)

    # Handle --list
    if list_compilers_flag:
        from ..compiler import list_compilers

        click.echo(list_compilers(cast("EPName", ep_device_resolved.ep)))
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
    # Merge config and CLI provider options, with explicit CLI values winning
    # for duplicate keys.
    if config_provider_options:
        config.ep_config.provider_options.update(config_provider_options)
    if config_provider_option_file_keys:
        config.ep_config.provider_option_file_keys.update(config_provider_option_file_keys)
    if cli_provider_options:
        config.ep_config.provider_options.update(cli_provider_options)

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
