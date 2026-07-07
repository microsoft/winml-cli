# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CLI utilities for WinML CLI commands."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
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


class ModelLoadError(click.ClickException):
    """Exit code 3: model could not be loaded onto the device/EP.

    Use for failures loading a model onto a device/EP, missing accelerators,
    or session creation that fails for hardware reasons. The message is printed
    verbatim to stderr (no ``Error:`` prefix) so callers control the wording.
    """

    exit_code = 3

    def show(self, file: Any = None) -> None:
        """Print the message verbatim to stderr (no ``Error:`` prefix)."""
        click.echo(self.format_message(), err=True)


class InferenceError(click.ClickException):
    """Exit code 4: inference/prediction failed at runtime.

    Use for prediction failures after the model loaded successfully. The
    message is printed verbatim to stderr (no ``Error:`` prefix).
    """

    exit_code = 4

    def show(self, file: Any = None) -> None:
        """Print the message verbatim to stderr (no ``Error:`` prefix)."""
        click.echo(self.format_message(), err=True)


class PartialSupportError(click.exceptions.Exit):
    """Exit code 1: a valid negative result, not an error.

    Raised silently (no ``Error:`` prefix) so commands can signal an
    actionable-but-non-fatal outcome (e.g. analyze: model not fully supported).
    """

    def __init__(self) -> None:
        super().__init__(1)


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


def warn_ignored_calibration_options(
    ctx: click.Context, reason: str, *, console: Console | None = None
) -> None:
    """Warn if the user passed calibration-related CLI options that are ignored.

    Checks whether ``--samples``, ``--method``, ``--weight-type``, or
    ``--activation-type`` were explicitly provided on the command line and
    emits a yellow warning listing the ignored options.

    Args:
        ctx: Click context (used to detect explicitly-provided params).
        reason: Human-readable explanation (e.g., "FP16 does not use
            calibration data.").
        console: Optional Rich console for output. Defaults to stderr.
    """
    ignored = []
    if is_cli_provided(ctx, "samples"):
        ignored.append("--samples")
    if is_cli_provided(ctx, "method"):
        ignored.append("--method")
    if is_cli_provided(ctx, "weight_type"):
        ignored.append("--weight-type")
    if is_cli_provided(ctx, "activation_type"):
        ignored.append("--activation-type")
    if ignored:
        out = console or _stderr_console
        out.print(f"[yellow]Warning:[/yellow] {', '.join(ignored)} ignored — {reason}")


def model_path_option(
    required: bool = True,
    multiple: bool = False,
    help_text: str | None = None,
) -> Callable[[F], F]:
    """Add ``-m/--model`` option that accepts a local ONNX file path.

    The path is validated for existence on disk and delivered as a
    :class:`pathlib.Path`. Shared by the ONNX-only commands (``analyze``,
    ``compile``, ``optimize``, ``quantize``) so the flag spelling, ``Path``
    type, and existence check stay identical. The decorated function receives
    the value as the ``model`` parameter (a tuple when ``multiple=True``).

    Args:
        required: Whether the model option is required (default: True).
        multiple: Accept the flag repeatably; the value becomes a tuple
            (default: False).
        help_text: Override for the help string (default: a generic
            ONNX-file description).

    Returns:
        Decorator function.
    """
    return click.option(
        "--model",
        "-m",
        required=required,
        multiple=multiple,
        type=click.Path(exists=True, path_type=Path),
        help=help_text or "Path to ONNX model file to analyze",
    )


def model_option(
    required: bool = True,
    optional_message: str | None = None,
    multiple: bool = False,
    help_text: str | None = None,
) -> Callable[[F], F]:
    """Add ``-m/--model`` option that accepts any model reference.

    Accepts a HuggingFace model ID, build output directory, or .onnx file path.
    No path existence validation is performed. Shared by the flexible-input
    commands (``build``, ``config``, ``eval``, ``export``, ``inspect``,
    ``perf``, ``run``, ``serve``) so the flag spelling stays identical. The
    decorated function receives the value as the ``model`` parameter (a tuple
    when ``multiple=True``).

    Args:
        required: Whether the model option is required (default: True).
        optional_message: Command-specific note appended after the help text.
        multiple: Accept the flag repeatably; the value becomes a tuple
            (default: False).
        help_text: Override for the base help string. Commands whose accepted
            inputs are narrower (e.g. ``inspect`` takes only an HF ID) supply
            their own; ``optional_message`` is still appended to it.

    Returns:
        Decorator function.
    """
    help = help_text or "Model: HF model ID, build output directory, or .onnx file path"
    if optional_message:
        help = f"{help}. {optional_message}"
    # ``multiple`` options default to an empty tuple; single-valued ones to None.
    kwargs: dict[str, Any] = {"multiple": True} if multiple else {"default": None}
    return click.option(
        "--model",
        "-m",
        required=required,
        help=help,
        **kwargs,
    )


def model_id_option(help_text: str | None = None) -> Callable[[F], F]:
    """Add ``--model-id`` option for a HuggingFace model ID.

    Shared by commands (e.g. ``quantize`` and ``eval``) that take an ONNX model
    path via ``-m/--model`` and need a separate HuggingFace model ID, for example
    to resolve the matching preprocessor/tokenizer or calibration datasets.

    Args:
        help_text: Optional override for the help string.

    Returns:
        Decorator function.
    """
    help = help_text or "HuggingFace model ID (e.g., 'microsoft/resnet-50')."
    return click.option(
        "--model-id",
        type=str,
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


def overwrite_option(optional_message: str | None = None) -> Callable[[F], F]:
    """Add the shared ``--overwrite/--no-overwrite`` toggle (default: no-overwrite).

    Output-producing commands default to *not* clobbering an existing output so
    a re-run can't silently destroy a previous result. Pair this with
    :func:`guard_output`, which performs the actual existence check. The
    decorated function receives the value as the ``overwrite`` parameter.

    Args:
        optional_message: Command-specific note appended after the help text.

    Returns:
        Decorator function.
    """
    help_text = "Overwrite an existing output instead of erroring out"
    if optional_message:
        help_text = f"{help_text}. {optional_message}"
    return click.option(
        "--overwrite/--no-overwrite",
        "overwrite",
        default=False,
        show_default=True,
        help=help_text,
    )


def guard_output(
    path: str | Path | None,
    overwrite: bool,
    *,
    label: str = "Output",
) -> None:
    """Fail fast when an output path already exists and ``--overwrite`` was not set.

    Shared safety check for every output-producing command so a re-run can't
    silently clobber a previous result. Call this *before* any ``mkdir`` /
    cleanup / work, with the fully resolved output path (including defaulted
    paths like ``{stem}_qdq.onnx``). A ``None`` path (e.g. output goes to
    stdout) is a no-op.

    Files block when they exist. Directories block only when they exist *and*
    are non-empty, so a freshly-created or empty output directory does not
    false-trigger.

    Args:
        path: Resolved output file or directory path, or ``None``.
        overwrite: When ``True``, the check is skipped (user opted in).
        label: Human-readable noun for the error message (e.g. ``"Output dir"``).

    Raises:
        click.ClickException: If the path exists (non-empty, for directories)
            and ``overwrite`` is ``False``.
    """
    if path is None or overwrite:
        return
    resolved = Path(path)
    if not resolved.exists():
        return
    if resolved.is_dir():
        if any(resolved.iterdir()):
            raise click.ClickException(
                f"{label} directory '{resolved}' already exists and is not empty. "
                "Re-run with --overwrite to replace its contents."
            )
        return
    raise click.ClickException(
        f"{label} '{resolved}' already exists. Re-run with --overwrite to replace it."
    )


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


def ep_option(
    required: bool = True,
    optional_message: str | None = None,
    default: str | None = None,
    include_auto: bool = False,
    include_all: bool = False,
) -> Callable[[F], F]:
    """Add --ep (execution provider) option to a Click command.

    Args:
        required: Whether the EP option is required (default: True)
        optional_message: Message to append to help text when
            optional (e.g., "If not specified, analyzes all
            supported EPs.")
        default: Default value when optional (default: None)
        include_auto: Whether to include "auto" as a valid choice
            (default: False).
        include_all: Whether to include "all" as a valid choice
            (default: False).

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
    choices = ["auto", *ep_choices] if include_auto else ep_choices
    choices = ["all", *choices] if include_all else choices

    return click.option(
        "--ep",
        "--execution-provider",
        required=required,
        default=default if not required else None,
        show_default=True,
        type=click.Choice(choices, case_sensitive=False),
        help=help_text,
    )


def ep_options_option(optional_message: str | None = None) -> Callable[[F], F]:
    """Add a repeatable ``--ep-options KEY=VALUE`` option to a Click command.

    Collects runtime EP provider options (e.g. QNN ``htp_performance_mode``)
    that are forwarded to ``add_provider_for_devices`` when the inference
    session is created. Distinct from build-time provider options set via
    ``--config``: these affect the runtime session, not the compiled graph.

    Use :func:`parse_ep_options` to turn the collected tuple into a dict.

    Args:
        optional_message: Extra command-specific guidance appended to help text.

    Returns:
        Decorator function.
    """
    help_text = (
        "Runtime EP provider option as KEY=VALUE (repeatable). Forwarded to the "
        "inference session's execution provider (e.g. "
        "--ep-options htp_performance_mode=burst). Duplicate keys: later "
        "occurrence wins."
    )
    if optional_message:
        help_text = f"{help_text} {optional_message}"

    return click.option(
        "--ep-options",
        "ep_options",
        multiple=True,
        help=help_text,
    )


def parse_ep_options(values: tuple[str, ...]) -> dict[str, str] | None:
    """Parse ``--ep-options KEY=VALUE`` tuples into a provider-options dict.

    Args:
        values: Raw values collected by a ``multiple=True`` Click option.

    Surrounding whitespace is stripped from both key and value. Duplicate
    keys follow last-write-wins semantics (the later occurrence wins).

    Returns:
        Mapping of option name to value, or ``None`` when nothing was provided
        (so callers can leave the session default untouched).

    Raises:
        click.BadParameter: If any value is missing the ``=`` separator or has
            an empty key.
    """
    if not values:
        return None
    options: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise click.BadParameter(
                f"Invalid EP option format: '{item}'. Use KEY=VALUE.",
                param_hint="--ep-options",
            )
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise click.BadParameter(
                f"Invalid EP option format: '{item}'. Key cannot be empty.",
                param_hint="--ep-options",
            )
        options[key] = value.strip()
    return options


def device_option(
    required: bool = True,
    optional_message: str | None = None,
    default: str | None = "NPU",
    include_auto: bool = False,
    include_all: bool = False,
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
        include_all: Whether to include "all" as a valid choice
            (default: False).

    Returns:
        Decorator function
    """
    device_choices = [device.lower() for device in SUPPORTED_DEVICES]
    choices = ["auto", *device_choices] if include_auto else device_choices
    choices = ["all", *choices] if include_all else choices
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


def precision_option(
    default: str | tuple[str, ...] | None = "auto",
    optional_message: str | None = None,
    include_short: bool = True,
    help_text: str | None = None,
    multiple: bool = False,
) -> Callable[[F], F]:
    """Add --precision option to a Click command.

    Shared across ``build``, ``config``, ``eval``, ``perf``, and ``quantize`` so
    the flag spelling (``-p``/``--precision``) and parsing stay consistent. Uses
    ``type=str`` (not ``click.Choice``) so the ``w{x}a{y}`` mixed-precision
    format (e.g. ``w8a16``) is accepted; invalid values are rejected downstream
    (``resolve_precision`` for build-path commands, ``_resolve_quant_types`` for
    ``quantize``).

    Args:
        default: Default precision value (default: "auto"). Pass ``None`` for
            commands like ``quantize`` that treat "no precision" distinctly.
        optional_message: Command-specific note appended after the help text
            (e.g., "Ignored for pre-built ONNX inputs.").
        include_short: Whether to also register the ``-p`` short alias
            (default: True).
        help_text: Override for the base help text. Commands whose accepted
            values differ from the default float+int set (e.g. ``quantize``,
            which has no fp16/fp32) supply their own; ``optional_message`` is
            still appended to it.
        multiple: Allow the flag to be specified multiple times to compose a
            pass pipeline (e.g. ``-p int4 -p fp16``). When True the parameter
            receives a ``tuple[str, ...]`` and ``default`` should be ``()``
            (default: False).

    Returns:
        Decorator function.
    """
    base_help = help_text or (
        "Precision: auto, fp32, fp16, int8, int16, or w{x}a{y} (e.g., w8a16). "
        "auto resolves from --device (npu->w8a16, gpu/cpu->fp16); "
        "fp16/fp32 skip quantization"
    )
    if optional_message:
        base_help = f"{base_help}. {optional_message}"

    param_decls = ["--precision", "precision"]
    if include_short:
        param_decls.insert(0, "-p")
    return click.option(
        *param_decls,
        type=str,
        default=default,
        multiple=multiple,
        show_default=not multiple,
        help=base_help,
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


def no_color_option() -> Callable[[F], F]:
    """Add a ``--no-color`` flag that disables colored output.

    Rich honors the ``NO_COLOR`` environment variable for every Console, so the
    flag's callback just sets ``NO_COLOR=1`` for the remainder of the run — this
    covers all consoles regardless of how they are constructed and matches the
    existing ``NO_COLOR=1`` / ``CI=true`` environment behavior. The change lives
    only in the current process, so the next invocation is colored again.

    Returns:
        Decorator function adding the ``--no-color`` flag (no exposed param).
    """

    def _disable_color(ctx: click.Context, param: click.Parameter, value: bool) -> bool:
        if value:
            os.environ["NO_COLOR"] = "1"
        return value

    return click.option(
        "--no-color",
        is_flag=True,
        default=False,
        expose_value=False,
        callback=_disable_color,
        help="Disable colored output (also via NO_COLOR=1 or CI=true).",
    )


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


def compile_option(
    default: bool | None = None,
    help_text: str | None = None,
) -> Callable[[F], F]:
    """Add shared ``--no-compile/--compile`` toggle to a Click command.

    The flag is exposed as the ``no_compile`` parameter. Note the inverted
    sense — ``--no-compile`` maps to ``no_compile=True``:

        * ``--no-compile`` -> ``no_compile=True``  (force skip compilation)
        * ``--compile``    -> ``no_compile=False`` (force enable compilation)

    Args:
        default: Value for ``no_compile`` when neither flag is passed.
            ``None`` -> tri-state inherit (e.g. ``winml build`` inherits from
            the config file); ``True`` -> exclude compilation by default
            (e.g. ``winml config`` omits the compile section).
        help_text: Command-specific help string. Falls back to a generic
            description when not provided.

    Returns:
        Decorator function.
    """
    if help_text is None:
        help_text = "Override compilation. --compile forces enable; --no-compile forces skip."

    return click.option(
        "--no-compile/--compile",
        "no_compile",
        default=default,
        help=help_text,
    )


def quant_option(
    default: bool = True,
    optional_message: str | None = None,
    help_text: str | None = None,
) -> Callable[[F], F]:
    """Add the shared ``--quant/--no-quant`` quantization toggle.

    Shared across ``build``, ``config``, ``perf``, and ``eval`` so the flag
    spelling and default stay consistent. ``--quantize/--no-quantize`` is kept
    as an alias so existing ``perf`` invocations keep working. The decorated
    function receives the value as the ``quant`` parameter (``True`` = run
    quantization, ``--no-quant`` overrides the config's quant section).

    Args:
        default: Default value (default: True = quantize).
        optional_message: Command-specific note appended after the help text.
        help_text: Override for the base help text. ``config`` phrases it in
            terms of the emitted config section; ``optional_message`` is still
            appended to it.

    Returns:
        Decorator function.
    """
    base_help = help_text or "Enable quantization (use --no-quant to skip, overrides config)"
    if optional_message:
        base_help = f"{base_help}. {optional_message}"
    return click.option(
        "--quant/--no-quant",
        "--quantize/--no-quantize",
        "quant",
        default=default,
        show_default=True,
        help=base_help,
    )


def optimize_option(
    default: bool = True,
    optional_message: str | None = None,
) -> Callable[[F], F]:
    """Add the shared ``--optimize/--no-optimize`` toggle.

    Controls whether the build pipeline runs graph optimization. The decorated
    function receives the value as the ``optimize`` parameter; ``--no-optimize``
    maps to ``skip_optimize=True`` downstream (see
    :func:`build_pipeline_extra_kwargs`).

    Args:
        default: Default value (default: True = optimize).
        optional_message: Command-specific note appended after the help text.

    Returns:
        Decorator function.
    """
    base_help = "Run optimization (use --no-optimize to skip for pre-quantized ONNX models)"
    if optional_message:
        base_help = f"{base_help}. {optional_message}"
    return click.option(
        "--optimize/--no-optimize",
        "optimize",
        default=default,
        show_default=True,
        help=base_help,
    )


def analyze_option(
    default: bool = True,
    optional_message: str | None = None,
) -> Callable[[F], F]:
    """Add the shared ``--analyze/--no-analyze`` toggle.

    Controls whether the build runs the autoconf analyzer loop. The decorated
    function receives the value as the ``analyze`` parameter; ``--no-analyze``
    forces ``max_optim_iterations`` to 0 (see
    :func:`build_pipeline_extra_kwargs`).

    Args:
        default: Default value (default: True = analyze).
        optional_message: Command-specific note appended after the help text.

    Returns:
        Decorator function.
    """
    base_help = "Run analyzer loop during build (use --no-analyze to skip)"
    if optional_message:
        base_help = f"{base_help}. {optional_message}"
    return click.option(
        "--analyze/--no-analyze",
        "analyze",
        default=default,
        show_default=True,
        help=base_help,
    )


def max_optim_iterations_option(optional_message: str | None = None) -> Callable[[F], F]:
    """Add the shared ``--max-optim-iterations`` option.

    The decorated function receives the value as the ``max_optim_iterations``
    parameter (``None`` = use the pipeline default of 3). ``--no-analyze`` wins
    over an explicit value (see :func:`build_pipeline_extra_kwargs`).

    Args:
        optional_message: Command-specific note appended to the help text.

    Returns:
        Decorator function.
    """
    base_help = "Maximum autoconf re-optimization rounds (default: 3). --no-analyze sets this to 0"
    if optional_message:
        base_help = f"{base_help}. {optional_message}"
    return click.option(
        "--max-optim-iterations",
        "max_optim_iterations",
        type=int,
        default=None,
        help=base_help,
    )


def build_pipeline_extra_kwargs(
    *,
    optimize: bool = True,
    analyze: bool = True,
    max_optim_iterations: int | None = None,
) -> dict[str, Any]:
    """Translate the shared optimize/analyze/max-optim flags into build kwargs.

    Centralizes the mapping shared by ``build``, ``perf``, and ``eval`` so the
    semantics stay identical:

    * ``--no-optimize`` -> ``skip_optimize=True``
    * ``--no-analyze``  -> ``hack_max_optim_iterations=0``
    * ``--max-optim-iterations N`` -> ``hack_max_optim_iterations=N`` (only when
      analysis is enabled; ``--no-analyze`` takes precedence).

    Keys are omitted when they would carry the pipeline default, so callers can
    splat the result unconditionally onto ``build_hf_model`` /
    ``build_onnx_model`` (or ``WinMLAutoModel``, which forwards them).

    Returns:
        Mapping of build-control kwargs.
    """
    extra: dict[str, Any] = {}
    if not optimize:
        extra["skip_optimize"] = True
    if not analyze:
        extra["hack_max_optim_iterations"] = 0
    elif max_optim_iterations is not None:
        extra["hack_max_optim_iterations"] = max_optim_iterations
    return extra


def ignored_build_flags_warning(
    *,
    skip_build_onnx: bool,
    quant: bool = True,
    optimize: bool = True,
    analyze: bool = True,
    max_optim_iterations: int | None = None,
) -> str | None:
    """Build a warning for build-pipeline flags that are no-ops on a pre-built ONNX.

    Commands that accept a pre-built ``.onnx`` input (``eval``, ``perf``) forward
    ``--no-quant``/``--no-optimize``/``--no-analyze``/``--max-optim-iterations`` to
    ``from_onnx``, but with ``skip_build`` (the default) no build runs, so those
    toggles silently take no effect. This returns a message naming the flags the
    user actually set (or ``None`` when nothing was set or a build will run), so
    callers can surface it through their own logger/console — mirroring the
    ``--precision``-ignored warning.

    Args:
        skip_build_onnx: True when the input is a pre-built ONNX *and* the build
            is skipped (the precondition under which the flags are no-ops).
        quant/optimize/analyze: Enabled-semantics toggles (False = user passed
            the ``--no-*`` form).
        max_optim_iterations: Explicit value, or ``None`` when left at default.

    Returns:
        Warning message, or ``None`` if no ignored flags apply.
    """
    if not skip_build_onnx:
        return None
    ignored = [
        flag
        for flag, was_set in (
            ("--no-quant", not quant),
            ("--no-optimize", not optimize),
            ("--no-analyze", not analyze),
            ("--max-optim-iterations", max_optim_iterations is not None),
        )
        if was_set
    ]
    if not ignored:
        return None
    return (
        f"{', '.join(ignored)} ignored for pre-built ONNX inputs "
        "(no build runs; pass --no-skip-build to rebuild)."
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


# ---------------------------------------------------------------------------
# ``-m/--model`` input classification
# ---------------------------------------------------------------------------

# Local model-file extensions used only to give a "path does not exist" error
# (instead of "not a valid HF id") when a path-shaped, non-existent value is
# passed.  ``.onnx`` is handled separately so it gets its own message.
_LOCAL_MODEL_FILE_EXTS = frozenset({".onnx", ".pt", ".pth", ".safetensors", ".bin", ".gguf"})

# HuggingFace repo id: an optional ``org/`` prefix plus a repo name, each
# segment built from ``[A-Za-z0-9]`` plus ``._-`` (must start alphanumeric).
_HF_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")


class ModelInputKind(Enum):
    """What a ``-m/--model`` value resolves to.

    - ``HF_ID``: not present on disk; a HuggingFace hub reference (``org/name``).
    - ``ONNX_FILE``: an existing local ``.onnx`` file.
    - ``FOLDER``: an existing local directory. The ``is_*``/``folder_has_onnx``
      flags on :class:`ModelInput` describe what the folder contains so each
      command can apply its own policy (build-family treats it as a transformers
      source; run/serve load the ONNX the engine discovers inside it).
    """

    HF_ID = "hf_id"
    ONNX_FILE = "onnx_file"
    FOLDER = "folder"


@dataclass(frozen=True)
class ModelInput:
    """Classification result for a ``-m/--model`` value.

    The folder flags are only meaningful when ``kind is FOLDER`` (all ``False``
    otherwise). ``is_winml_cli_folder`` implies ``folder_has_onnx``.
    """

    kind: ModelInputKind
    raw: str
    is_hf_folder: bool = False
    folder_has_onnx: bool = False
    is_winml_cli_folder: bool = False


def _looks_like_path(raw: str) -> bool:
    """Return True when *raw* is clearly a filesystem path (not an HF id).

    Used only for a non-existent value to choose between a "path does not
    exist" error and a "not a valid HF id" error. HF ids carry at most one
    ``/`` and none of these path indicators.
    """
    if "\\" in raw:
        return True
    if raw.startswith(("./", "../", "~/", ".\\", "..\\", "~\\")):
        return True
    p = Path(raw)
    if p.is_absolute():
        return True
    # Windows drive-letter prefix, e.g. ``C:models``.
    if len(raw) >= 2 and raw[1] == ":":
        return True
    if raw.count("/") > 1:
        return True
    return p.suffix.lower() in _LOCAL_MODEL_FILE_EXTS


def classify_model_input(raw: str) -> ModelInput:
    """Classify a ``-m/--model`` value into a :class:`ModelInput`.

    Existence-first: a value present on disk is a local file or directory; a
    value absent from disk is a HuggingFace hub id (or an error). The engine
    remains responsible for locating the actual ``*.onnx`` inside a folder.

    Raises:
        click.UsageError: for a non-existent ``.onnx`` path, a non-existent
            path-shaped value, a syntactically invalid HF id, or an existing
            non-``.onnx`` file.
    """
    path = Path(raw)

    if not path.exists():
        if path.suffix.lower() == ".onnx":
            raise click.UsageError(f"ONNX file not found: {raw}")
        if _looks_like_path(raw):
            raise click.UsageError(f"Model path does not exist: {raw}")
        if _HF_ID_RE.match(raw):
            return ModelInput(kind=ModelInputKind.HF_ID, raw=raw)
        raise click.UsageError(
            f"'{raw}' is not a valid HuggingFace model id, an existing .onnx "
            "file, or an existing directory."
        )

    if path.is_file():
        if path.suffix.lower() == ".onnx":
            return ModelInput(kind=ModelInputKind.ONNX_FILE, raw=raw)
        raise click.UsageError(
            f"Unsupported model file: {raw} (expected a .onnx file or a directory)."
        )

    # Existing directory: report what it contains; callers apply policy.
    folder_has_onnx = any(path.glob("*.onnx"))
    is_winml_cli_folder = folder_has_onnx and any(path.glob("*build_manifest.json"))
    return ModelInput(
        kind=ModelInputKind.FOLDER,
        raw=raw,
        is_hf_folder=(path / "config.json").is_file(),
        folder_has_onnx=folder_has_onnx,
        is_winml_cli_folder=is_winml_cli_folder,
    )


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


def resolve_model_path(
    *,
    model: tuple[str, ...],
    model_id: str | None,
    require_model_id_for_onnx: bool = True,
) -> tuple[str | dict[str, str] | None, str | None]:
    """Turn repeated ``-m`` values + ``--model-id`` into ``(model_path, model_id)``.

    Shared by ``eval`` and ``perf`` commands for consistent composite-model CLI
    syntax.  ``-m role=path`` pairs require ``--model-id`` for preprocessor and
    config resolution.

    Args:
        require_model_id_for_onnx: When *True* (default, used by ``eval``),
            a bare ONNX file requires ``--model-id``.  ``perf`` passes
            *False* because it can benchmark without HF config.

    Returns:
        A 2-tuple ``(model_path, hf_model_id)`` where *model_path* is
        ``None`` (HF id only), a single ONNX path string, or a
        ``{role: path}`` dict for composite models.
    """
    if not model:
        if model_id is not None:
            return None, model_id
        raise click.UsageError(
            "A model is required. Provide -m with a HuggingFace model ID, "
            "a path to an .onnx file, or role=path pairs for composite models."
        )

    role_assigned = [v for v in model if "=" in v]
    plain = [v for v in model if "=" not in v]

    if role_assigned and plain:
        raise click.UsageError(
            "Cannot mix plain `-m <value>` and `-m role=path` forms. "
            "Use `role=path` consistently for composite models."
        )

    if role_assigned:
        if model_id is None:
            raise click.UsageError(
                "--model-id is required when using composite `-m role=path` options."
            )
        sub_model_paths: dict[str, str] = {}
        for v in role_assigned:
            role, _, path = v.partition("=")
            role, path = role.strip(), path.strip()
            if not role or not path:
                raise click.BadParameter(
                    f"Invalid role=path: {v!r}. Both role and path are required.",
                    param_hint="-m/--model",
                )
            if role in sub_model_paths:
                raise click.BadParameter(
                    f"Duplicate role {role!r} in -m options.",
                    param_hint="-m/--model",
                )
            if not Path(path).exists():
                raise click.BadParameter(
                    f"ONNX file not found: {path}",
                    param_hint="-m/--model",
                )
            sub_model_paths[role] = path
        return sub_model_paths, model_id

    if len(plain) > 1:
        raise click.UsageError(
            "Multiple -m values require `role=path` syntax for composite models."
        )

    value = plain[0]
    try:
        _mi = classify_model_input(value)
    except click.UsageError as e:
        # Preserve param-scoped BadParameter contract (tests + hint).
        raise click.BadParameter(str(e), param_hint="-m/--model") from e
    if _mi.kind is ModelInputKind.ONNX_FILE:
        if require_model_id_for_onnx and model_id is None:
            raise click.UsageError(
                "When using an ONNX file, --model-id is required "
                "for preprocessor and config resolution."
            )
        return value, model_id
    if model_id is not None and model_id != value:
        raise click.UsageError(
            "Cannot pass both `-m <hf_id>` and `--model-id`. "
            "Use `--model-id` only together with an ONNX file path in `-m`."
        )
    return None, model_id or value
