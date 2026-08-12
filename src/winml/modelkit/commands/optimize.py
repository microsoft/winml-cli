# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Optimize command for winml CLI.

This module provides the optimize command that uses the capability-driven
optimizer for ONNX model optimization with fusion and graph optimizations.

CLI options are auto-generated from the capability registry, following
the Open-Closed Principle from the design documentation.

Usage:
    winml optimize --model MODEL --output OUTPUT [OPTIONS]

Examples:
    winml optimize -m model.onnx -o model_opt.onnx
    winml optimize -m model.onnx -o model_opt.onnx --enable-gelu-fusion
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import click
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from ..onnx import load_onnx, save_onnx
from ..utils import cli as cli_utils
from ..utils.logging import configure_logging


if TYPE_CHECKING:
    from collections.abc import Callable

F = TypeVar("F", bound="Callable[..., Any]")


logger = logging.getLogger(__name__)
console = Console()


# =============================================================================
# CONFIG FILE LOADING
# =============================================================================


def load_config(path: Path) -> dict[str, Any]:
    """Load configuration from config file (YAML/JSON).

    Automatically detects format based on file extension.

    Args:
        path: Path to configuration file

    Returns:
        Dictionary with configuration values

    Raises:
        click.ClickException: If file cannot be loaded or parsed
    """
    suffix = path.suffix.lower()

    try:
        if suffix in (".yaml", ".yml"):
            return _load_yaml(path)
        if suffix == ".json":
            return _load_json(path)
        raise click.ClickException(f"Unsupported config format: {suffix}. Use .yaml/.yml or .json")
    except FileNotFoundError as e:
        raise click.ClickException(f"Config file not found: {path}") from e


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    try:
        import yaml
    except ImportError as e:
        raise click.ClickException(
            "YAML support requires 'pyyaml' package. Install with: pip install pyyaml"
        ) from e

    try:
        with path.open() as f:
            result = yaml.safe_load(f) or {}
            if not isinstance(result, dict):
                raise click.ClickException(
                    f"Config file must contain a YAML mapping, got {type(result).__name__}"
                )
            return result
    except yaml.YAMLError as e:
        raise click.ClickException(f"Invalid YAML in config file: {e}") from e


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON configuration file."""
    try:
        with path.open() as f:
            result = json.load(f)
            if not isinstance(result, dict):
                raise click.ClickException(
                    f"Config file must contain a JSON object, got {type(result).__name__}"
                )
            return result
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON in config file: {e}") from e


def capability_options(func: F) -> F:
    """Decorator that adds CLI options for all registered capabilities.

    This decorator auto-generates CLI options from the capability registry,
    following the design pattern from modelkit/optim/cli.py.
    """
    # Late import to speed up CLI startup
    from ..optim import BoolCapability, ChoiceCapability, IntCapability, get_all_capabilities

    # Get all capabilities and reverse for correct Click ordering
    all_caps = list(get_all_capabilities().values())
    all_caps.reverse()

    for cap in all_caps:
        if isinstance(cap, BoolCapability):
            default_str = "enabled" if cap.default else "disabled"
            func = click.option(
                f"--enable-{cap.name}/--disable-{cap.name}",
                cap.python_name,
                default=None,  # Use None to detect explicit vs default
                help=f"{cap.description} (Default: {default_str})",
            )(func)
        elif isinstance(cap, IntCapability):
            func = click.option(
                f"--{cap.name}",
                cap.python_name,
                type=click.IntRange(min=cap.min_value, max=cap.max_value),
                default=None,
                help=f"{cap.description} (Default: {cap.default})",
            )(func)
        elif isinstance(cap, ChoiceCapability):
            func = click.option(
                f"--{cap.name}",
                cap.python_name,
                type=click.Choice(cap.choices),
                default=None,
                help=f"{cap.description} (Default: {cap.default})",
            )(func)

    return func


def _render_check_optim(console: Console, findings: list[Any], verbose: bool) -> None:
    """Render the applicability report produced by a ``--check-optim`` probe.

    Args:
        console: Rich console for output.
        findings: Applicable ``CapabilityFinding`` objects in pipeline order.
        verbose: When True, list every affected node instead of a sample.
    """
    if not findings:
        console.print("\n[yellow]No registered optimizations would change this model.[/yellow]")
        return

    console.print(f"\n[bold]{len(findings)} applicable optimization(s):[/bold]\n")

    sample_limit = 6
    for finding in findings:
        console.print(
            f"[bold green]{finding.enable_flag}[/bold green]  [dim]({finding.category})[/dim]"
        )
        console.print(f"  {escape(finding.description)}")

        segments: list[str] = []
        if finding.removed_nodes:
            segments.append(f"[red]-{len(finding.removed_nodes)}[/red]")
        if finding.added_nodes:
            segments.append(f"[green]+{len(finding.added_nodes)}[/green]")
        if finding.modified_nodes:
            segments.append(f"[yellow]~{len(finding.modified_nodes)}[/yellow]")
        if segments:
            console.print(f"  nodes: {' '.join(segments)}")

        for kind, nodes in (
            ("removed", finding.removed_nodes),
            ("added", finding.added_nodes),
            ("modified", finding.modified_nodes),
        ):
            if not nodes:
                continue
            histogram = ", ".join(
                f"{escape(op)} ({count})" for op, count in finding.op_histogram(kind)
            )
            console.print(f"    {kind}: {histogram}")
            shown = nodes if verbose else nodes[:sample_limit]
            for node in shown:
                console.print(f"      [dim]- {escape(node.label())}[/dim]")
            if not verbose and len(nodes) > sample_limit:
                console.print(f"      [dim]… and {len(nodes) - sample_limit} more (use -v)[/dim]")

        const_segments: list[str] = []
        if finding.modified_initializers:
            const_segments.append(f"[yellow]~{len(finding.modified_initializers)}[/yellow]")
        if finding.added_initializers:
            const_segments.append(f"[green]+{len(finding.added_initializers)}[/green]")
        if finding.removed_initializers:
            const_segments.append(f"[red]-{len(finding.removed_initializers)}[/red]")
        if const_segments:
            console.print(f"  constants: {' '.join(const_segments)}")
            touched = (
                finding.modified_initializers
                + finding.added_initializers
                + finding.removed_initializers
            )
            shown_inits = touched if verbose else touched[:sample_limit]
            for name in shown_inits:
                console.print(f"      [dim]- {escape(name)}[/dim]")
            if not verbose and len(touched) > sample_limit:
                console.print(f"      [dim]… and {len(touched) - sample_limit} more (use -v)[/dim]")

        console.print()

    console.print(
        "[dim]Enable any of the above with its --enable-* flag "
        "(dependencies are auto-enabled).[/dim]"
    )


def _resolve_optimization_target(ep: str | None, device: str | None) -> tuple[Any, Any]:
    """Resolve an explicit optimization EP/device request."""
    from ..session import (
        EPDeviceTarget,
        WinMLEPRegistry,
        resolve_device,
    )

    target = resolve_device(EPDeviceTarget(ep=ep or "auto", device=(device or "auto").lower()))
    return target, WinMLEPRegistry.instance().auto_device(target)


def _run_check_optim(
    model: Path,
    all_caps: dict[str, Any],
    verbose: bool,
    *,
    target: Any | None = None,
    ep_device: Any | None = None,
) -> None:
    """Probe which optimizations apply to the model and print a report.

    No output file is written. Every boolean capability that is off by default
    is evaluated independently against the model.

    Args:
        model: Path to the input ONNX model.
        all_caps: The full capability registry.
        verbose: Whether to show every affected node/constant.
        target: Resolved EP/device target, if explicitly requested.
        ep_device: Resolved runtime EP device used by each optimization probe.
    """
    from ..optim import BoolCapability, analyze_model

    probe_count = sum(
        1 for cap in all_caps.values() if isinstance(cap, BoolCapability) and not cap.default
    )

    console.print(f"[bold blue]Input:[/bold blue] {model}")
    if target is not None:
        console.print(f"[bold blue]Target:[/bold blue] {target.ep} on {target.device.upper()}")
    console.print(
        "[dim]--check-optim — analyzing applicable optimizations (no output written).[/dim]"
    )
    console.print("\n[bold]Loading model...[/bold]")
    onnx_model = load_onnx(model)

    console.print(
        f"[bold]Probing {probe_count} optimization capabilities...[/bold] "
        "[dim](this can take a while on large models)[/dim]"
    )
    with Progress(
        TextColumn("[bold]Checking[/bold] [cyan]{task.description}[/cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("preparing capabilities", total=probe_count)
        findings = analyze_model(
            onnx_model,
            all_caps,
            ep_device=ep_device,
            on_probe_start=lambda name: progress.update(task_id, description=name),
            on_probe_complete=lambda _: progress.advance(task_id),
        )

    _render_check_optim(console, findings, verbose)


@click.command()
@click.option(
    "--list-capabilities",
    "-l",
    is_flag=True,
    default=False,
    help="List all registered optimization capabilities and exit",
)
@click.option(
    "--list-rewrites",
    is_flag=True,
    default=False,
    help="List available pattern rewrite families and exit",
)
@click.option(
    "--check-optim",
    is_flag=True,
    default=False,
    help="Analyze which optimizations apply to the model (and the nodes they "
    "affect) without writing any output",
)
@cli_utils.model_path_option(
    # Not required when --list-capabilities/--list-rewrites is used
    required=False,
    help_text="Input ONNX model file",
)
@cli_utils.output_option("Output path (default: {input}_opt.onnx)")
@cli_utils.overwrite_option()
@cli_utils.ep_option(
    required=False,
    default=None,
    include_auto=True,
    include_cuda=True,
    optional_message="If omitted with --device, selects a compatible EP automatically.",
)
@cli_utils.device_option(
    required=False,
    default=None,
    include_auto=True,
    optional_message="If both --ep and --device are omitted, preserves CPU optimization.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Configuration file (YAML/JSON)",
)
@cli_utils.verbosity_options()
@cli_utils.no_color_option()
@capability_options
@click.pass_context  # type: ignore[arg-type]  # capability_options widens the signature; click stubs want positional-only ctx but we keep it keyword-callable for back-compat
def optimize(
    ctx: click.Context,
    list_capabilities: bool,
    list_rewrites: bool,
    check_optim: bool,
    model: Path | None,
    output: Path | None,
    overwrite: bool,
    ep: str | None,
    device: str | None,
    config: Path | None,
    verbose: int,
    quiet: bool,
    **kwargs: Any,
) -> None:
    r"""Optimize ONNX model with capability-driven optimizer.

    This command applies graph optimizations and operator fusions to an ONNX model
    using the capability-driven optimizer from winml.modelkit.optim.

    CLI options are auto-generated from registered capabilities.

    Configuration precedence (highest to lowest):
        1. CLI options (--enable-X, --disable-X)
        2. Config file options (-c/--config)
        3. Capability defaults

    \b
    Examples:
        # List available capabilities
        winml optimize --list-capabilities

        # List available rewrite pattern families
        winml optimize --list-rewrites

        # Check which optimizations apply to a model (and the nodes they
        # affect) without writing any output
        winml optimize -m model.onnx --check-optim

        # Pattern rewrite flags follow: --enable-{source-slug}-{target-slug}
        # Run --list-rewrites to discover all available flag names.
        # Example (all GELU variants -> single Gelu node):
        winml optimize -m model.onnx -o out.onnx --enable-gelu-singlegelu
        # Example (only Gelu1 variant -> single Gelu node):
        winml optimize -m model.onnx -o out.onnx --enable-gelu1-singlegelu

        # Basic optimization with GELU fusion
        winml optimize -m model.onnx -o model_opt.onnx --enable-gelu-fusion

        # Use config file
        winml optimize -m model.onnx -c config.toml
    """
    # Import capabilities (late import to speed up CLI)
    from ..optim import (
        BoolCapability,
        auto_enable_dependencies,
        get_all_capabilities,
        validate,
        validate_dependencies,
    )

    all_caps = get_all_capabilities()

    # Handle --list-capabilities
    if list_capabilities:
        from ..optim import ChoiceCapability, IntCapability

        if not all_caps:
            console.print("[yellow]No capabilities registered.[/yellow]")
            return

        # Group by category
        categories = sorted(
            {cap.category for cap in all_caps.values()},
            key=lambda c: c.value,
        )

        if verbose:
            # Verbose mode: full details with descriptions
            for category in categories:
                caps_in_category = {
                    name: cap for name, cap in all_caps.items() if cap.category == category
                }

                console.print(f"\n{category.value.upper()} ({len(caps_in_category)} capabilities)")
                console.print("-" * 80)

                for name, cap in sorted(caps_in_category.items()):
                    # Type-specific default formatting
                    if isinstance(cap, BoolCapability):
                        default_str = "enabled" if cap.default else "disabled"
                    else:
                        default_str = str(cap.default)

                    console.print(f"  {name}")
                    console.print(f"    Default: {default_str}")
                    console.print(f"    {cap.description}")
                    console.print(f"    [ORT: {cap.ort_name}]")
        else:
            # Compact mode: just list flags
            console.print(f"\n[bold]Available optimization flags ({len(all_caps)} total):[/bold]\n")

            for category in categories:
                caps_in_category = {
                    name: cap for name, cap in all_caps.items() if cap.category == category
                }

                console.print(f"[dim]{category.value}:[/dim]")
                for name, cap in sorted(caps_in_category.items()):
                    if isinstance(cap, BoolCapability):
                        # Show the flag to change from default
                        if cap.default:
                            console.print(f"  --disable-{name}")
                        else:
                            console.print(f"  --enable-{name}")
                    elif isinstance(cap, IntCapability):
                        console.print(f"  --{name} <{cap.min_value}..{cap.max_value}>")
                    elif isinstance(cap, ChoiceCapability):
                        console.print(f"  --{name} <{'|'.join(cap.choices)}>")
                console.print()

            console.print("[dim]Use --list-capabilities --verbose for detailed descriptions.[/dim]")

        return

    # Handle --list-rewrites
    if list_rewrites:
        from ..optim.pipes.rewrite_rules import (
            REWRITE_GROUPS,
            source_flag_name,
        )

        if not REWRITE_GROUPS:
            console.print("[yellow]No rewrite capabilities discovered.[/yellow]")
            return

        console.print("\n[bold]Rewrite capabilities (source -> target):[/bold]\n")
        for group in REWRITE_GROUPS:
            rule_file = Path(group.rule_file).name
            is_multi = len(group.sources) > 1
            group_flag = f"--enable-{group.flag_name}"
            suffix = "  [dim](group: all variants)[/dim]" if is_multi else ""
            console.print(f"  [green]{group_flag}[/green]{suffix}")
            if is_multi:
                console.print(f"      sources: {', '.join(group.sources)}")
            else:
                console.print(f"      source:  {group.sources[0]}")
            console.print(f"      target:  {group.target_class}")
            console.print(f"      rule:    winml/modelkit/pattern/rules/{rule_file}")
            if is_multi:
                for src in group.sources:
                    src_flag = f"--enable-{source_flag_name(src, group.target_class)}"
                    console.print(f"  [green]{src_flag}[/green]  [dim](per-source: {src})[/dim]")
            console.print()
        return

    # Require model if not listing capabilities/rewrites
    if model is None:
        raise click.UsageError("Missing option '--model' / '-m'.")

    # Merge top-level -v/-q with subcommand-level flags so either position works.
    verbose, quiet = cli_utils.resolve_verbosity(ctx, verbose, quiet)
    configure_logging(verbosity=verbose, quiet=quiet)

    target = None
    ep_device = None
    if ep is not None or device is not None:
        from ..session import (
            DeviceNotFound,
            UnknownListingPick,
            WinMLEPNotDiscovered,
            WinMLEPRegistrationFailed,
        )

        try:
            target, ep_device = _resolve_optimization_target(ep, device)
        except (
            DeviceNotFound,
            RuntimeError,
            UnknownListingPick,
            ValueError,
            WinMLEPNotDiscovered,
            WinMLEPRegistrationFailed,
        ) as e:
            raise click.UsageError(f"Could not resolve optimization target: {e}") from e

    # Handle --check-optim: report which optimizations apply, write nothing.
    if check_optim:
        _run_check_optim(
            model,
            all_caps,
            bool(verbose),
            target=target,
            ep_device=ep_device,
        )
        return

    # Import optimizer
    from ..optim import Optimizer

    # Determine output path
    if output is None:
        output = model.parent / f"{model.stem}_opt.onnx"

    # Refuse to clobber an existing output unless the user opted in.
    cli_utils.guard_output(output, overwrite)

    # Show info
    console.print(f"[bold blue]Input:[/bold blue] {model}")
    console.print(f"[bold blue]Output:[/bold blue] {output}")
    if config:
        console.print(f"[bold blue]Config:[/bold blue] {config}")

    # Build optimization config with proper precedence:
    # 1. Start with capability defaults
    final_config: dict[str, Any] = {}
    for cap_name, cap_def in all_caps.items():
        final_config[cap_name] = cap_def.default

    # 2. Apply config file if specified (overrides defaults)
    if config:
        file_config = load_config(config)
        # Normalize snake_case keys to kebab-case (accept both formats)
        file_config = {k.replace("_", "-"): v for k, v in file_config.items()}
        final_config.update(file_config)
        console.print(f"[dim]Loaded config from: {config}[/dim]")

    # 3. Override with explicit CLI options (highest precedence)
    # kwargs contains python_name -> value mappings from capability_options
    for cap_name, cap_def in all_caps.items():
        python_name = cap_def.python_name
        if python_name in kwargs and kwargs[python_name] is not None:
            final_config[cap_name] = kwargs[python_name]

    # 4. Auto-enable dependencies (e.g., bias-gelu-fusion requires gelu-fusion)
    final_config = auto_enable_dependencies(final_config, all_caps)

    # 5. Validate configuration (especially important for config files)
    errors = validate(final_config, all_caps)
    dep_errors = validate_dependencies(final_config, all_caps)
    all_errors = errors + dep_errors

    if all_errors:
        header = click.style("Configuration validation errors:", fg="red", bold=True)
        bullets = "\n".join(click.style(f"  * {error}", fg="red") for error in all_errors)
        raise click.UsageError(f"{header}\n{bullets}")

    # Convert capability names (kebab-case) to python names (snake_case) for optimizer
    optimizer_kwargs: dict[str, Any] = {}
    for cap_name, value in final_config.items():
        if cap_name in all_caps:
            python_name = all_caps[cap_name].python_name
            optimizer_kwargs[python_name] = value

    # Pass verbose to pipes
    if verbose:
        optimizer_kwargs["verbose"] = True

    if ep_device is not None:
        if target is None:
            raise RuntimeError("Resolved optimization EP device has no target metadata.")
        optimizer_kwargs["ep_device"] = ep_device
        console.print(
            f"[bold blue]Target:[/bold blue] {target.ep} on {target.device.upper()}"
        )

    try:
        console.print("\n[bold]Loading model...[/bold]")
        onnx_model = load_onnx(model)
        original_nodes = len(onnx_model.graph.node)

        console.print("[bold]Running optimizer...[/bold]")
        optimizer = Optimizer()
        optimized_model = optimizer.optimize(onnx_model, **optimizer_kwargs)

        console.print("[bold]Saving optimized model...[/bold]")
        save_onnx(optimized_model, output)

        # Report results
        optimized_nodes = len(optimized_model.graph.node)
        node_change = (
            abs(optimized_nodes - original_nodes) / original_nodes * 100 if original_nodes else 0
        )

        console.print(f"\n[bold green]Success![/bold green] Model optimized: {output}")
        if optimized_nodes < original_nodes:
            change_label = "reduction"
        elif optimized_nodes > original_nodes:
            change_label = "increase"
        else:
            change_label = None
        node_info = f"Nodes: {original_nodes} -> {optimized_nodes}"
        if change_label is None:
            node_info += " (no change)"
        else:
            node_info += f" ({node_change:.1f}% {change_label})"
        console.print(f"[dim]{node_info}[/dim]")

    except Exception as e:
        console.print(f"\n[bold red]Optimization failed:[/bold red] {e}")
        logger.exception("Optimization failed")
        raise click.ClickException(f"Optimization failed: {e}") from e
