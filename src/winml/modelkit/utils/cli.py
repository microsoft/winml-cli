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


def precision_option(
    default: str | None = "auto",
    optional_message: str | None = None,
    include_short: bool = True,
    help_text: str | None = None,
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
        show_default=True,
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
