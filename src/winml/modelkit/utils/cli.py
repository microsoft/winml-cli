# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CLI utilities for WinML CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console

from .constants import ALL_EP_NAMES, SUPPORTED_DEVICES


if TYPE_CHECKING:
    from ..config import WinMLBuildConfig


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


def model_path_option(required=True):
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


def model_option(required=True, optional_message=None):
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


def output_option(help_text: str, required: bool = False):
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

    ep_choices = [name for name in ALL_EP_NAMES if name not in ("cuda", "CUDAExecutionProvider")]

    return click.option(
        "--ep",
        "--execution-provider",
        required=required,
        default=None,
        type=click.Choice(ep_choices, case_sensitive=False),
        help=help_text,
    )


def device_option(required=True, optional_message=None, default="NPU", include_auto=False):
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


def verbosity_options():
    """Add verbose and quiet logging options to a Click command.

    Adds --verbose/-v (stackable: -v, -vv, -vvv) and --quiet/-q flags.
    The decorated function receives ``verbose`` (int, count of -v flags)
    and ``quiet`` (bool).

    See :mod:`winml.modelkit.utils.logging` for the verbosity convention.

    Returns:
        Decorator function adding verbose and quiet options.
    """

    def decorator(f):
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


def build_config_option(help: str | None = None):
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


def trust_remote_code_option(optional_message: str | None = None):
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
        "--trust-remote-code",
        is_flag=True,
        default=False,
        help=help_text,
        callback=_warn_callback,
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


def collect_cli_overrides(ctx: click.Context, cls: type) -> dict[str, object]:
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

    overrides: dict[str, object] = {}
    for cli_name, value in ctx.params.items():
        field_name = rename.get(cli_name, cli_name)
        if field_name in valid_fields and is_cli_provided(ctx, cli_name):
            overrides[field_name] = value
    return overrides
