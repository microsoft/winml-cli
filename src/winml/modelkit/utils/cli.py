# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CLI utilities for WinML CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar

import click
from rich.console import Console

from .constants import ALL_EP_NAMES, SUPPORTED_DEVICES


if TYPE_CHECKING:
    from collections.abc import Callable

    from ..config import WinMLBuildConfig


# TypeVar for signature-preserving Click decorators.
F = TypeVar("F", bound="Callable[..., Any]")

# Allowed values for ``--format`` / ``-f``.
OutputFormat: TypeAlias = Literal["text", "json", "table", "compact"]


# Shared stderr console for security/diagnostic messages emitted from utils.
# Mirrors the module-level ``console = Console()`` pattern used by individual
# command modules, but targets stderr so messages survive ``-q/--quiet``.
_stderr_console = Console(stderr=True)

# Per-process flag so the warning surfaces at most once per CLI run / API call.
# Multiple instrumented entry points along a single call chain (e.g. CLI flag
# -> generate_hf_build_config -> resolve_loader_config -> load_hf_model)
# would otherwise emit the same warning several times.
_trust_remote_code_warned = False


def warn_trust_remote_code() -> None:
    """Print the ``trust_remote_code`` security warning to stderr.

    Uses the shared stderr ``rich.Console`` so the warning renders in bold red
    and matches the rest of the CLI's output style; bypassing the ``logging``
    module also means it is **not** suppressed by ``-q/--quiet``. Emitted at
    most once per process so a single CLI run or API call surfaces the
    warning exactly once, even when several instrumented entry points (CLI
    flag, ``load_hf_model``, ``generate_hf_build_config``, ...) are reached
    along the same call chain.
    """
    global _trust_remote_code_warned
    if _trust_remote_code_warned:
        return
    _trust_remote_code_warned = True
    _stderr_console.print(
        "[bold red]WARNING:[/bold red] trust_remote_code is enabled - "
        "custom Python from the model repository will be downloaded and "
        "executed. Proceed only if you trust the publisher."
    )


def model_path_option(required: bool = True) -> Callable[[F], F]:
    """Add --model option that accepts a local ONNX file path.

    The path is validated for existence on disk.

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


def model_option(required: bool = True, optional_message: str | None = None) -> Callable[[F], F]:
    """Add --model option that accepts any model reference.

    Accepts a HuggingFace model ID, build output directory, or .onnx file path.
    No path existence validation is performed.

    Args:
        required: Whether the model option is required (default: True)

    Returns:
        Decorator function
    """
    help = "Model: HF model ID, build output directory, or .onnx file path"
    if optional_message:
        help = f"{help}. {optional_message}"
    return click.option(
        "--model",
        "-m",
        required=required,
        default=None,
        help=help,
    )


def output_option(help_text: str, required: bool = False) -> Callable[[F], F]:
    """Add ``-o/--output`` option that accepts a file path.

    The path is delivered to the callback as a :class:`pathlib.Path`.

    Args:
        help_text: Command-specific help string for the option.
        required: Whether the option is required (default: False).

    Returns:
        Decorator function.
    """
    kwargs: dict = {"type": click.Path(path_type=Path), "help": help_text}
    if required:
        kwargs["required"] = True
    else:
        kwargs["default"] = None
    return click.option("--output", "-o", **kwargs)


def format_option(
    choices: list[OutputFormat] | None = None,
    default: OutputFormat = "text",
    short_flag: bool = True,
) -> Callable[[F], F]:
    """Add ``--format`` option to a Click command.

    The option is exposed as the ``output_format`` parameter in the
    decorated function (type: :data:`OutputFormat`).

    Args:
        choices: Allowed format values. Defaults to ``["text", "json"]``.
        default: Default format value. Defaults to ``"text"``.
        short_flag: Whether to include ``-f`` short alias. Set to False
            when another option already uses ``-f``.
    """
    if choices is None:
        choices = ["text", "json"]
    args = ["-f", "--format"] if short_flag else ["--format"]
    return click.option(
        *args,
        "output_format",
        type=click.Choice(choices, case_sensitive=False),
        default=default,
        help=f"Output format (default: {default}). 'json' prints structured JSON to stdout.",
    )


def ep_option(required: bool = True, optional_message: str | None = None) -> Callable[[F], F]:
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

    ep_choices = [name for name in ALL_EP_NAMES if name not in ("cuda", "CUDAExecutionProvider")]

    return click.option(
        "--ep",
        "--execution-provider",
        required=required,
        default=None,
        type=click.Choice(ep_choices, case_sensitive=False),
        help=help_text,
    )


def device_option(
    required: bool = True,
    optional_message: str | None = None,
    default: str | None = "NPU",
    include_auto: bool = False,
) -> Callable[[F], F]:
    """Add --device option to a Click command.

    Args:
        required: Whether the device option is required (default: True)
        optional_message: Message to append to help text when
            optional (e.g., "If not specified, uses NPU as
            default.")
        default: Default value when optional (default: "NPU")
        include_auto: Whether to include "auto" as a valid choice
            (default: False).

    Returns:
        Decorator function
    """
    device_choices = [device.lower() for device in SUPPORTED_DEVICES]
    choices = ["auto", *device_choices] if include_auto else device_choices
    help_text = f"Target device type ({', '.join(choices)})"
    if optional_message:
        help_text = f"{help_text}. {optional_message}"

    return click.option(
        "-d",
        "--device",
        required=required,
        default=default if not required else None,
        show_default=True,
        type=click.Choice(choices, case_sensitive=False),
        help=help_text,
    )


def verbosity_options() -> Callable[[F], F]:
    """Add verbose and quiet logging options to a Click command.

    Adds --verbose/-v (stackable: -v, -vv, -vvv) and --quiet/-q flags.
    The decorated function receives ``verbose`` (int, count of -v flags)
    and ``quiet`` (bool).

    See :mod:`winml.modelkit.utils.logging` for the verbosity convention.

    Returns:
        Decorator function adding verbose and quiet options.
    """

    def decorator(f: F) -> F:
        f = click.option(
            "--quiet",
            "-q",
            is_flag=True,
            default=False,
            help="Quiet mode - errors only to stderr",
        )(f)
        return click.option(
            "--verbose",
            "-v",
            count=True,
            help="Increase verbosity (-v=INFO, -vv=DEBUG)",
        )(f)

    return decorator


def resolve_verbosity(ctx: click.Context, verbose: int, quiet: bool) -> tuple[int, bool]:
    """Merge subcommand ``--verbose``/``--quiet`` with the parent group's values.

    The top-level ``winml`` group also accepts ``-v``/``-q`` and stores the
    resolved values in ``ctx.obj``. Both positions are equally valid:
    ``winml -v export …`` and ``winml export -v …`` should behave the same.
    This helper takes the max verbosity and OR of quiet so users can supply
    the flag at either level (or both).

    Precedence: ``-q``/``--quiet`` always wins over verbosity, including the
    ``--debug`` alias — ``winml --debug export -q …`` runs at ERROR. ``-q`` is
    an explicit "shut up" signal and trumps any verbosity raise, so the user
    is never surprised by debug spam after they asked for quiet.

    Args:
        ctx: Click context for the current subcommand.
        verbose: Subcommand-level ``-v`` count.
        quiet: Subcommand-level ``--quiet`` flag.

    Returns:
        Tuple ``(verbose, quiet)`` ready to pass to ``configure_logging``.
    """
    if ctx.obj:
        verbose = max(verbose, int(ctx.obj.get("verbosity", 0)))
        # ``debug`` is the historical backward-compat alias for ``-vv``; keep
        # honoring it so tests that bypass ``main()`` and stuff ``debug=True``
        # straight into ctx.obj still raise the verbosity floor.
        if ctx.obj.get("debug"):
            verbose = max(verbose, 2)
        quiet = quiet or bool(ctx.obj.get("quiet", False))
    return verbose, quiet


def build_config_option(help: str | None = None) -> Callable[[F], F]:
    """Add -c/--config option for WinMLBuildConfig JSON file."""
    if help is None:
        help = (
            "WinMLBuildConfig JSON file (from winml config). "
            "Provides defaults; explicit CLI options take precedence."
        )
    return click.option(
        "-c",
        "--config",
        "config_file",
        type=click.Path(exists=True, path_type=Path),
        default=None,
        help=help,
    )


def skip_build_option(
    default: bool = True,
    optional_message: str | None = None,
) -> Callable[[F], F]:
    """Add --skip-build/--no-skip-build toggle for commands that accept ONNX inputs.

    When skip-build is on, the build pipeline (optimize -> [quantize] -> [compile])
    is bypassed and the ONNX file is used as-is. Applies only to ONNX inputs.

    Args:
        default: Default value (True = skip build by default, use --no-skip-build
            to run the full build pipeline on the ONNX file).
        optional_message: Extra command-specific guidance appended to help text.

    Returns:
        Decorator function.
    """
    help_text = (
        "Skip the build pipeline (optimize/quantize/compile) and use the ONNX "
        "file as-is. Use --no-skip-build to run the full build pipeline. "
        "Applies only to ONNX inputs."
    )
    if optional_message:
        help_text = f"{help_text} {optional_message}"

    return click.option(
        "--skip-build/--no-skip-build",
        default=default,
        show_default=True,
        help=help_text,
    )


def trust_remote_code_option(optional_message: str | None = None) -> Callable[[F], F]:
    """Add shared --trust-remote-code option to a Click command.

    Args:
        optional_message: Extra command-specific guidance appended to help text.

    Returns:
        Decorator function.
    """
    help_text = (
        "Allow executing custom code from model repositories or dataset scripts. "
        "Use only with trusted sources."
    )
    if optional_message:
        help_text = f"{help_text} {optional_message}"

    def _warn_callback(ctx: click.Context, param: click.Parameter, value: bool) -> bool:
        if value:
            warn_trust_remote_code()
        return value

    return click.option(
        "--trust-remote-code/--no-trust-remote-code",
        default=False,
        show_default=True,
        help=help_text,
        callback=_warn_callback,
    )


def allow_unsupported_nodes_option(optional_message: str | None = None) -> Callable[[F], F]:
    """Add shared --allow-unsupported-nodes option to a Click command.

    When set, the build's optimize/analyze loop logs a warning instead of
    raising when unsupported nodes persist after analysis, so the build
    proceeds (the EP may fall back to another device for those nodes).

    Args:
        optional_message: Extra command-specific guidance appended to help text.

    Returns:
        Decorator function.
    """
    help_text = (
        "Continue the build instead of failing when the analyzer reports "
        "unsupported nodes (the EP may fall back to another device for them)."
    )
    if optional_message:
        help_text = f"{help_text} {optional_message}"

    return click.option(
        "--allow-unsupported-nodes/--no-allow-unsupported-nodes",
        default=False,
        show_default=True,
        help=help_text,
    )


def precision_option(
    required: bool = False,
    optional_message: str | None = None,
) -> Callable[[F], F]:
    """Add shared --precision option to a Click command.

    Consistent with winml perf, winml eval, winml config. Values: fp32, fp16.

    Args:
        required: Whether the option is required.
        optional_message: Extra guidance appended to help text.

    Returns:
        Decorator function.
    """
    help_text = "Model precision: fp32 (default) or fp16."
    if optional_message:
        help_text = f"{help_text} {optional_message}"

    return click.option(
        "--precision",
        type=click.Choice(["fp32", "fp16"]),
        default=None,
        required=required,
        help=help_text,
    )


def load_build_config(config_path: Path) -> tuple[WinMLBuildConfig, dict]:
    """Load a WinMLBuildConfig from a JSON file.

    Args:
        config_path: Path to JSON config file.

    Returns:
        Tuple ``(build_cfg, raw_dict)``. ``raw_dict`` is the unmodified
        parsed JSON object, returned alongside the dataclass so callers can
        distinguish "key explicitly set in JSON" from "key absent" — a
        distinction the dataclass alone cannot preserve, because
        ``from_dict`` substitutes dataclass defaults for missing keys.

    Raises:
        click.UsageError: If file is empty or invalid JSON.
    """
    from ..config import WinMLBuildConfig

    try:
        content = config_path.read_text()
        if not content.strip():
            raise click.UsageError(f"Config file is empty: {config_path}")
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise click.UsageError(f"Invalid JSON in build config: {e}") from e

    if not isinstance(data, dict):
        raise click.UsageError(f"Build config must be a JSON object, got {type(data).__name__}")

    return WinMLBuildConfig.from_dict(data), data


def is_onnx_file_path(model_input: str) -> bool:
    """Check if input is a path to an existing ``.onnx`` file.

    Shared helper for CLI commands that accept either a HuggingFace model ID
    or a local ``.onnx`` file path for the ``-m/--model`` option.
    """
    path = Path(model_input)
    return path.suffix == ".onnx" and path.exists()


def is_cli_provided(ctx: click.Context, param_name: str) -> bool:
    """Check whether a CLI parameter was explicitly provided by the user.

    Args:
        ctx: Click context.
        param_name: The parameter name (Python name, e.g. 'model').

    Returns:
        True if the user explicitly passed the option on the command line.
    """
    source = ctx.get_parameter_source(param_name)
    return source == click.core.ParameterSource.COMMANDLINE


def collect_cli_overrides(ctx: click.Context, cls: type) -> dict[str, Any]:
    """Collect CLI-provided values that match fields on a dataclass.

    Iterates ``ctx.params`` and returns ``{field_name: value}`` for every
    CLI param that was explicitly provided AND maps to a field on *cls*.

    Name mapping uses ``field(metadata={"cli_name": ...})`` on the
    dataclass.  Fields without ``cli_name`` metadata match by name.

    Args:
        ctx: Click context.
        cls: Target dataclass whose fields define the valid key set.

    Returns:
        Dict of ``{field_name: value}`` for CLI-provided params.
    """
    import dataclasses

    # Build reverse map: cli_name -> field_name
    rename: dict[str, str] = {}
    valid_fields: set[str] = set()
    for f in dataclasses.fields(cls):
        valid_fields.add(f.name)
        cli_name = f.metadata.get("cli_name")
        if cli_name:
            rename[cli_name] = f.name

    overrides: dict[str, Any] = {}
    for cli_name, value in ctx.params.items():
        field_name = rename.get(cli_name, cli_name)
        if field_name in valid_fields and is_cli_provided(ctx, cli_name):
            overrides[field_name] = value
    return overrides
