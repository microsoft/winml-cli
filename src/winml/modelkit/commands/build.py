# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Build command for ModelKit CLI.

Thin CLI wrapper around build_hf_model() and build_onnx_model() APIs.
The build module owns the pipeline. This command parses flags, loads config,
auto-detects ONNX vs HF input, calls the appropriate API, and reports results.

Usage:
    winml build -c config.json -m microsoft/resnet-50 -o output/
    winml build -c config.json -m model.onnx -o output/
    winml build -c config.json -m bert-base-uncased -o output/ --no-quant
    winml build -c config.json -o output/ --use-cache
    winml build -c config.json -m microsoft/resnet-50 -o output/ --rebuild -v
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.logging import RichHandler

from ..utils import cli as cli_utils
from ..utils.console import (
    detect_model_source,
    get_console,
    print_error,
    print_final,
    print_setup,
    print_stage_skip,
    print_stages_header,
)


if TYPE_CHECKING:
    from typing import Any

    from torch import nn

    from ..build import BuildResult
    from ..config import WinMLBuildConfig

logger = logging.getLogger(__name__)
console = get_console()


# =============================================================================
# CLI HELPERS
# =============================================================================


def _load_config(
    config_file: str,
    *,
    no_quant: bool = False,
    no_compile: bool | None = None,
) -> WinMLBuildConfig | list[WinMLBuildConfig]:
    """Load WinMLBuildConfig from JSON file with CLI overrides.

    Supports both single config (JSON object) and module mode (JSON array).

    Args:
        config_file: Path to JSON config file.
        no_quant: If True, set config.quant = None (skip quantization).
        no_compile: ``bool | None`` compile override.
            ``True``  → ``--no-compile``: force skip compilation.
            ``False`` → ``--compile``: force enable compilation; raises UsageError if
                        config has no compile section.
            ``None``  → neither flag passed: inherit compile settings from config file.

    Returns:
        Single WinMLBuildConfig for normal mode, or list for module mode.

    Raises:
        click.UsageError: If config file is invalid or --compile is forced
            without a compile section in the config.
    """
    from ..config import WinMLBuildConfig

    config_path = Path(config_file)
    try:
        content = config_path.read_text()
        if not content.strip():
            raise click.UsageError(f"Config file is empty: {config_path}")
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise click.UsageError(f"Invalid JSON in config: {e}") from e

    def _apply_overrides(cfg: WinMLBuildConfig) -> WinMLBuildConfig:
        if no_quant:
            cfg.quant = None
        if no_compile is True:
            cfg.compile = None
        elif no_compile is False and cfg.compile is None:
            raise click.UsageError(
                "Cannot enable compilation: no compile section found in the config file. "
                "Re-run `winml config --compile` to generate a compile-enabled config."
            )
        return cfg

    if isinstance(data, dict):
        return _apply_overrides(WinMLBuildConfig.from_dict(data))

    if isinstance(data, list):
        for i, d in enumerate(data):
            if not isinstance(d, dict):
                raise click.UsageError(
                    f"Module config [{i}] must be a JSON object, got {type(d).__name__}"
                )
        return [_apply_overrides(WinMLBuildConfig.from_dict(d)) for d in data]

    raise click.UsageError(f"Config must be a JSON object or array, got {type(data).__name__}")


def _instantiate_parent_model(model_type: str, task: str | None = None) -> nn.Module:
    """Instantiate parent model with init weights for submodule extraction.

    Uses resolve_loader_config to get the task-specific model class (e.g.,
    BertForMaskedLM instead of BertModel), then instantiates with init weights
    via from_config(). This ensures module paths from torchinfo (which trace
    the task-specific model) match the model structure.

    NO pretrained weights are downloaded (design principle P1).

    Args:
        model_type: HuggingFace model type (e.g., "bert", "resnet").
        task: HuggingFace task (e.g., "fill-mask"). Used to resolve the
            correct model class. If None, uses the first supported task.

    Returns:
        PyTorch model in eval mode with random/init weights.
    """
    from ..loader import resolve_loader_config

    _, hf_config, resolved_class = resolve_loader_config(
        model_type=model_type,
        task=task,
    )

    try:
        model = resolved_class(hf_config)
    except OSError as e:
        logger.debug("Direct construction failed (%s), using from_config()", e)
        model = resolved_class.from_config(hf_config)

    model.eval()
    return model


def _build_modules(
    configs: list[WinMLBuildConfig],
    output_dir: Path,
    *,
    rebuild: bool = False,
    ep: str | None = None,
    device: str | None = None,
) -> list[BuildResult]:
    """Build each module config using init-weight parent for submodule extraction.

    Iterates configs in original order, caching parent models by model_type
    so each unique type is instantiated only once. For each config, extracts
    the submodule via parent.get_submodule(module_path) and calls
    build_hf_model with the extracted pytorch_model.

    Args:
        configs: List of WinMLBuildConfig, each with loader.model_type and
            loader.module_path set.
        output_dir: Base output directory. Each module gets a subdirectory
            named ``{class_name}_{index}``.
        rebuild: If True, overwrite existing artifacts.
        ep: Target execution provider for analyzer.
        device: Target device for analyzer.

    Returns:
        List of BuildResult in the same order as *configs*.

    Raises:
        click.UsageError: If any config is missing model_type or module_path.
    """
    from ..build import build_hf_model

    parents: dict[str, Any] = {}
    results: list[BuildResult] = []

    for i, cfg in enumerate(configs):
        model_type = cfg.loader.model_type
        module_path = cfg.loader.module_path
        class_name = cfg.loader.model_class or "module"

        if not model_type:
            raise click.UsageError(f"Config #{i} missing loader.model_type")
        if not module_path:
            raise click.UsageError(f"Config #{i} missing loader.module_path")

        task = cfg.loader.task
        parent_key = f"{model_type}:{task}"
        if parent_key not in parents:
            console.print(f"  [dim]Instantiating {model_type} parent (init weights)...[/dim]")
            parents[parent_key] = _instantiate_parent_model(model_type, task=task)

        parent = parents[parent_key]
        submodule = parent.get_submodule(module_path)

        subdir = output_dir / f"{class_name}_{i}"

        result = build_hf_model(
            config=cfg,
            output_dir=subdir,
            pytorch_model=submodule,
            rebuild=rebuild,
            ep=ep,
            device=device,
        )
        results.append(result)

    return results


# =============================================================================
# CLI COMMAND
# =============================================================================


@click.command("build")
@click.option(
    "-c",
    "--config",
    "config_file",
    type=click.Path(exists=True),
    required=True,
    help="WinMLBuildConfig JSON file (from winml config)",
)
@click.option(
    "-m",
    "--model",
    "model_id",
    default=None,
    help="HuggingFace model ID or path to .onnx file. Omit for random-weight build.",
)
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    type=click.Path(),
    default=None,
    help="Output directory for all build artifacts",
)
@click.option(
    "--use-cache",
    is_flag=True,
    default=False,
    help="Use ModelKit global cache (~/.cache/winml/). Mutually exclusive with -o.",
)
@click.option(
    "--rebuild",
    is_flag=True,
    default=False,
    help="Overwrite existing artifacts and rebuild",
)
@click.option(
    "--no-quant",
    is_flag=True,
    default=False,
    help="Skip quantization (overrides config)",
)
@click.option(
    "--no-compile/--compile",
    "no_compile",
    default=None,
    help="Override compilation from config. --no-compile forces skip; "
    "--compile forces enable (config must have a compile section). "
    "Default: inherit from config file.",
)
@click.option(
    "--ep",
    default=None,
    help="Target execution provider for analyzer (e.g., 'qnn'). "
    "Falls back to compile config EP if not set.",
)
@click.option(
    "--device",
    default=None,
    help="Target device for analyzer (e.g., 'NPU', 'GPU'). Default: NPU.",
)
@click.option(
    "--no-analyze",
    is_flag=True,
    default=False,
    help="Skip analyzer loop during build",
)
@click.option(
    "--no-optimize",
    is_flag=True,
    default=False,
    help="Skip optimization (for pre-quantized ONNX models)",
)
@click.option(
    "--max-optim-iterations",
    "max_optim_iterations",
    type=int,
    default=None,
    help="Maximum autoconf re-optimization rounds (default: 3). --no-analyze sets this to 0.",
)
@cli_utils.trust_remote_code_option(
    optional_message="Trust remote code for custom model architectures (e.g., Mu2)."
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose logging",
)
@click.pass_context
def build(
    ctx: click.Context,
    config_file: str,
    model_id: str | None,
    output_dir: str | None,
    use_cache: bool,
    rebuild: bool,
    no_quant: bool,
    no_compile: bool | None,
    no_optimize: bool,
    ep: str | None,
    device: str | None,
    no_analyze: bool,
    max_optim_iterations: int | None,
    trust_remote_code: bool,
    verbose: bool,
) -> None:
    r"""Build a WinML-optimized ONNX model from a HuggingFace model or .onnx file.

    Requires a config file generated by 'winml config'. The config file already
    contains device/precision settings (applied during 'winml config' generation).
    Specify either --output-dir or --use-cache for artifact destination.

    If -m points to an existing .onnx file, the build skips export and runs
    optimize -> quantize -> compile directly (ONNX build path).

    \b
    Examples:
        # Full pipeline with pretrained weights
        winml build -c config.json -m microsoft/resnet-50 -o output/

        # Build from pre-exported ONNX file
        winml build -c config.json -m model.onnx -o output/

        # Export + optimize only (config must have compile=null, or pass --no-compile to force skip)
        winml build -c config.json -m bert-base-uncased -o output/ --no-quant --no-compile

        # Random-weight build (no download)
        winml build -c config.json -o output/

        # Use global cache
        winml build -c config.json -m microsoft/resnet-50 --use-cache

        # Force rebuild
        winml build -c config.json -m microsoft/resnet-50 -o output/ --rebuild
    """
    # Inherit debug flag from parent context
    if ctx.obj and ctx.obj.get("debug"):
        verbose = True

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Validate mutual exclusion
    if output_dir and use_cache:
        raise click.UsageError("--output-dir and --use-cache are mutually exclusive.")
    if not output_dir and not use_cache:
        raise click.UsageError("One of --output-dir or --use-cache is required.")

    # If ep unspecified, attempt to auto-select a suitable EP from the registry
    if ep is None:
        from ..session import WinMLEPRegistry

        registry = WinMLEPRegistry.get_instance()
        candidate_eps = [
            "QNNExecutionProvider",
            "OpenVINOExecutionProvider",
            "VitisAIExecutionProvider",
        ]
        for candidate_ep in candidate_eps:
            if registry.is_ep_available(candidate_ep):
                ep = candidate_ep
                logger.info("EP unspecified for build, auto-selecting: %s", ep)
                break
    if ep is None:
        logger.warning(
            "EP unspecified for build, and auto-selection failed. Proceeding without EP hints."
        )

    try:
        # Hub-hosted ONNX (e.g. ``onnx-community/sam3-tracker-ONNX/onnx/...``)
        # is downloaded once and treated as a local .onnx file thereafter.
        if model_id is not None:
            from ..loader import is_hf_onnx_path, resolve_hf_onnx_path

            if is_hf_onnx_path(model_id):
                model_id = str(resolve_hf_onnx_path(model_id))

        # Load config first (needed for both output modes)
        config_or_configs = _load_config(
            config_file,
            no_quant=no_quant,
            no_compile=no_compile,
        )
        is_module_mode = isinstance(config_or_configs, list)

        # Build extra kwargs for pipeline control
        extra_kwargs: dict[str, Any] = {}
        if no_optimize:
            extra_kwargs["skip_optimize"] = True
        if no_analyze:
            extra_kwargs["hack_max_optim_iterations"] = 0
        elif max_optim_iterations is not None:
            extra_kwargs["hack_max_optim_iterations"] = max_optim_iterations
        if trust_remote_code:
            extra_kwargs["trust_remote_code"] = True

        if is_module_mode:
            # ---- MODULE MODE: array config, one build per submodule ----
            if use_cache:
                raise click.UsageError(
                    "--use-cache is not supported for module mode (array config). "
                    "Use --output-dir instead."
                )
            if not output_dir:
                raise click.UsageError("--output-dir is required for module mode (array config).")

            resolved_dir = Path(output_dir)
            configs = config_or_configs

            if not configs:
                raise click.UsageError("Module config array is empty -- nothing to build.")

            print_setup(
                console,
                model=model_id or "random-init",
                config=Path(config_file).name,
                output=str(resolved_dir),
                source="HuggingFace",
            )
            print_stages_header(console)
            console.print(f"   \U0001f9e9 [bold]Modules:[/bold] {len(configs)}")
            console.print()

            results = _build_modules(
                configs=configs,
                output_dir=resolved_dir,
                rebuild=rebuild,
                ep=ep,
                device=device,
            )

            # Report per-module results
            for i, result in enumerate(results):
                module_path = configs[i].loader.module_path
                if result.reused:
                    console.print(f"  [{i}] {module_path}: reused")
                else:
                    console.print(
                        f"  [{i}] {module_path}: {result.elapsed:.1f}s -> {result.final_onnx_path}"
                    )

            # Write module summary
            from ..build import write_module_summary

            summary_instances = []
            for cfg, result in zip(configs, results, strict=True):
                summary_instances.append(
                    {
                        "module_path": cfg.loader.module_path,
                        "class_name": cfg.loader.model_class,
                        "output_dir": str(result.output_dir),
                        "build_elapsed_s": round(result.elapsed, 2),
                    }
                )

            write_module_summary(
                output_path=resolved_dir / "module_summary.json",
                model_id=model_id or "random-init",
                module_class=configs[0].loader.model_class or "unknown",
                instances=summary_instances,
            )
            console.print(f"  Summary: {resolved_dir / 'module_summary.json'}")

            console.print()

        else:
            # ---- SINGLE MODEL MODE ----
            config = config_or_configs

            # Resolve output directory and cache_key
            cache_key: str | None = None
            if use_cache:
                from ..cache import get_cache_dir, get_cache_key, get_model_dir
                from ..loader import get_task_abbrev

                task = config.loader.task if config.loader else None
                resolved_dir = get_model_dir(
                    model_id or "random-init",
                    cache_dir=get_cache_dir(),
                )
                if not task:
                    raise click.UsageError(
                        "--use-cache requires loader.task in config. "
                        "The cache key is prefixed by the task abbreviation."
                    )
                cache_key = get_cache_key(
                    get_task_abbrev(task),
                    config.generate_cache_key(),
                )
            else:
                resolved_dir = Path(output_dir)

            _run_single_build(
                config=config,
                config_file=config_file,
                model_id=model_id,
                resolved_dir=resolved_dir,
                rebuild=rebuild,
                cache_key=cache_key,
                ep=ep,
                device=device,
                extra_kwargs=extra_kwargs,
            )

    except click.UsageError:
        raise  # Let click handle its own errors
    except ValueError as e:
        raise click.UsageError(str(e)) from e
    except Exception as e:
        if verbose:
            logger.exception("Build failed")

        # Map common errors to actionable hints
        err_str = str(e)
        hint = None
        if "Quantization failed" in err_str:
            hint = "Try: --no-quant to skip quantization"
        elif "Compilation failed" in err_str:
            hint = "Try: --no-compile to skip compilation"
        elif "Black nodes persist" in err_str:
            hint = "Try: winml analyze -m <model> --ep <ep> to investigate operator support"
        elif isinstance(e, FileNotFoundError):
            hint = "Check: model path or HuggingFace model ID"

        if hint:
            console.print()
            print_error(console, f"Build failed: {e}", hint=hint)
            console.print()

        raise click.ClickException(f"Build failed: {e}") from e


# =============================================================================
# SINGLE MODEL BUILD — CLI-level stage orchestration
# =============================================================================


def _run_single_build(
    *,
    config: WinMLBuildConfig,
    config_file: str,
    model_id: str | None,
    resolved_dir: Path,
    rebuild: bool,
    cache_key: str | None,
    ep: str | None,
    device: str | None,
    extra_kwargs: dict[str, Any],
) -> None:
    """Run single-model build with Rich Live progress per stage."""
    from .config import _is_onnx_file

    _is_onnx = model_id is not None and _is_onnx_file(model_id)
    # Derive source from _is_onnx to guarantee header label matches pipeline
    source = "ONNX" if _is_onnx else detect_model_source(model_id)

    # Gap 1: (pretrained) suffix; Gap 2: ONNX file size
    if model_id is None:
        model_label = "random-init"
    elif _is_onnx:
        _sz = _safe_size(Path(model_id))
        from ..utils.console import fmt_size

        model_label = f"{model_id}  [dim]({fmt_size(_sz)})[/dim]" if _sz else model_id
    else:
        model_label = f"{model_id}  [dim](pretrained)[/dim]"

    # ── 🔧 Setup section ────────────────────────────────────────
    print_setup(
        console,
        model=model_label,
        config=Path(config_file).name,
        output=str(resolved_dir),
        source=source,
    )
    print_stages_header(console)

    # ── Redirect logging + warnings through Rich during Live stages ──
    # This ensures log messages and warnings.warn() render above the
    # Live area instead of breaking it (same pattern as winml analyze).
    root_logger = logging.getLogger()
    old_handlers = root_logger.handlers[:]
    rich_handler = RichHandler(
        console=console,
        show_path=False,
        show_time=True,
        rich_tracebacks=False,
    )
    rich_handler.setLevel(root_logger.level)
    root_logger.handlers = [rich_handler]
    # Route warnings.warn() (e.g., TracerWarning) through logging → Rich
    logging.captureWarnings(True)

    start_time = time.monotonic()

    try:
        if _is_onnx:
            stage_timings = _build_onnx_pipeline(
                config=config,
                onnx_path=Path(model_id),
                output_dir=resolved_dir,
                rebuild=rebuild,
                ep=ep,
                device=device,
                extra_kwargs=extra_kwargs,
            )
        else:
            stage_timings = _build_hf_pipeline(
                config=config,
                model_id=model_id,
                output_dir=resolved_dir,
                rebuild=rebuild,
                cache_key=cache_key,
                ep=ep,
                device=device,
                extra_kwargs=extra_kwargs,
            )

        elapsed = time.monotonic() - start_time
        final_path = resolved_dir / "model.onnx"
        if final_path.exists() and stage_timings:
            print_final(
                console,
                elapsed,
                str(final_path),
                stage_timings=stage_timings,
            )
    finally:
        logging.captureWarnings(False)
        root_logger.handlers = old_handlers


def _print_reused(artifact_path: Path) -> None:
    """Print reused artifact message."""
    console.print()
    console.print(
        f"   \u267b\ufe0f  [bold cyan]Existing artifact found:[/bold cyan] {artifact_path}"
    )
    console.print("   \U0001f4a1 [dim]Use --rebuild to force rebuild.[/dim]")
    console.print()


def _safe_size(path: Path) -> int:
    """Get file size including ONNX external data, return 0 if unavailable."""
    try:
        if path.suffix == ".onnx":
            from ..utils.console import get_onnx_total_size

            return get_onnx_total_size(path)
        return path.stat().st_size
    except OSError:
        return 0


def _show_io(sl: Any, config: WinMLBuildConfig) -> None:
    """Show I/O tensors in a StageLive."""
    export_cfg = config.export
    if not export_cfg:
        return
    inputs = export_cfg.input_tensors or []
    outputs = export_cfg.output_tensors or []
    for i, t in enumerate(inputs):
        name = t.name or "(unnamed)"
        shape = str(list(t.shape)) if getattr(t, "shape", None) else "dynamic"
        dtype = getattr(t, "dtype", None) or "?"
        sl.io_input(name, shape, dtype, first=(i == 0))
    for i, t in enumerate(outputs):
        name = t.name or "(unnamed)"
        # OutputTensorSpec has name only — show name, no shape/dtype
        label = "Output:       " if i == 0 else "              "
        sl.detail(f"{label}[cyan]{name}[/cyan]")


# =============================================================================
# SHARED PIPELINE STAGE HELPERS
# =============================================================================


def _run_optimize_stage(
    *,
    config: WinMLBuildConfig,
    model_path: Path,
    optimized_path: Path,
    ep: str | None,
    device: str | None,
    max_iters: int,
    stage_timings: list[tuple[str, float | None]],
    show_io_first: bool = False,
) -> tuple[Path, float]:
    """Run the optimize stage inside a StageLive context.

    Creates all 5 analyzer callbacks bound to the live display, calls
    run_optimize_analyze_loop, shows convergence message and artifact.

    Args:
        config: Build configuration.
        model_path: Input model path.
        optimized_path: Output path for optimized model.
        ep: Execution provider for analyzer.
        device: Target device for analyzer.
        max_iters: Maximum analyzer iterations.
        stage_timings: List to append (stage_name, elapsed) tuple to.
        show_io_first: If True, show I/O tensors at the start of the stage
            (used in ONNX mode where there is no export stage).

    Returns:
        Tuple of (current_path, opt_elapsed).
    """
    from ..build import run_optimize_analyze_loop
    from ..utils.console import StageLive

    with StageLive("optimize", console) as sl:
        sl.set_status("Optimizing ONNX graph...")

        if show_io_first:
            _show_io(sl, config)

        # Analyzer callback state for live EP bars
        _ep_bars: dict[str, int] = {}
        _ep_counts: dict[str, dict[str, int]] = {}
        _ep_totals: dict[str, int] = {}
        _current_ep = [""]
        _current_iter = [0, 0]  # [iteration, max_iter]
        _header_shown = [False]

        def _on_iteration_start(iteration: int, max_iter: int) -> None:
            _ep_bars.clear()
            _ep_counts.clear()
            _ep_totals.clear()
            _current_iter[0] = iteration
            _current_iter[1] = max_iter
            _header_shown[0] = False

        def _on_ep_start(ep_name: str, operator_counts: dict) -> None:
            _current_ep[0] = ep_name
            _ep_counts[ep_name] = {}
            total = sum(operator_counts.values())
            _ep_totals[ep_name] = total
            # Show "Analyzing N nodes (iter X/Y)" on first EP of each iter
            if not _header_shown[0]:
                _header_shown[0] = True
                sl.detail(
                    f"[bold]Analyzing[/bold] [cyan]{total}[/cyan] nodes  "
                    f"[dim](iter {_current_iter[0]}/{_current_iter[1]})[/dim]"
                )
            _ep_bars[ep_name] = sl.ep_bar_add(ep_name, total=total)

        def _on_node_result(pattern_runtime: Any) -> None:
            ep_name = _current_ep[0]
            level = pattern_runtime.result.classification.value
            counts = _ep_counts.setdefault(ep_name, {})
            counts[level] = counts.get(level, 0) + 1
            s = counts.get("supported", 0)
            p = counts.get("partial", 0)
            u = counts.get("unsupported", 0)
            idx = _ep_bars.get(ep_name)
            if idx is not None:
                sl.ep_bar_update(
                    idx,
                    ep_name,
                    s,
                    p,
                    u,
                    total=_ep_totals.get(ep_name, 0),
                )

        def _on_patterns(autoconf_dict: dict) -> None:
            sl.detail("[bold]Patterns[/bold]")
            for key in autoconf_dict:
                name = key.replace("disable_", "").replace("_fusion", "").replace("_", " ").title()
                sl.detail(f"  [yellow]{name}[/yellow]  [dim]\u2192 {key}[/dim]")

        def _on_reoptimize(autoconf_dict: dict) -> None:
            sl.detail("[bold]Optimizing[/bold]  [dim](applying autoconf)[/dim]")
            sl.detail(f"  [dim]{autoconf_dict}[/dim]")

        t0 = time.monotonic()
        current_path, _, analyze_iters, _, analyze_details = run_optimize_analyze_loop(
            model_path=model_path,
            optimized_path=optimized_path,
            config=config,
            ep=ep,
            device=device,
            max_optim_iterations=max_iters,
            on_ep_start=_on_ep_start,
            on_node_result=_on_node_result,
            on_iteration_start=_on_iteration_start,
            on_patterns_discovered=_on_patterns,
            on_reoptimize=_on_reoptimize,
            use_external_data=True,
        )
        opt_elapsed = time.monotonic() - t0

        if analyze_iters > 0:
            converged = not analyze_details.get("autoconf_not_converged", False)
            conv_str = "converged" if converged else "NOT converged"
            # Show pattern result even when none found
            autoconf = analyze_details.get("autoconf", {})
            if not autoconf:
                sl.detail("[bold]Patterns[/bold]")
                sl.detail("  [dim]No optimization patterns found[/dim]")
            sl.detail(f"[dim]Autoconf {conv_str} after {analyze_iters} iteration(s)[/dim]")

        sl.set_done(opt_elapsed)
        sl.artifact(str(optimized_path), _safe_size(optimized_path))
        sl.blank()

    stage_timings.append(("Optimize", opt_elapsed))
    return current_path, opt_elapsed


def _run_quantize_stage(
    *,
    config: WinMLBuildConfig,
    current_path: Path,
    quantized_path: Path,
    stage_timings: list[tuple[str, float | None]],
) -> Path:
    """Run the quantize stage inside a StageLive context (if quant is configured).

    Handles QDQ skip detection, shows dataset/calibration/precision details,
    and appends timing to stage_timings.

    Args:
        config: Build configuration.
        current_path: Input model path.
        quantized_path: Output path for quantized model.
        stage_timings: List to append (stage_name, elapsed) tuple to.

    Returns:
        Updated current_path (quantized_path if quantization ran, else unchanged).
    """
    from ..onnx import is_quantized_onnx
    from ..quant import quantize_onnx
    from ..utils.console import StageLive

    if config.quant is None:
        return current_path

    if is_quantized_onnx(current_path):
        print_stage_skip(console, "quantize", "(QDQ nodes already present)")
        stage_timings.append(("Quantize", None))
        return current_path

    with StageLive("quantize", console) as sl:
        wt = config.quant.weight_type or "?"
        sl.set_status(f"Quantizing ({wt})...")
        # Calibration info before blocking call
        ds = config.quant.dataset_name or "default"
        sl.kv(
            "Dataset:",
            f"[cyan]{ds}[/cyan]  [dim]({config.quant.task or 'unknown'})[/dim]",
        )
        sl.kv(
            "Calibration:",
            f"[cyan]{config.quant.samples}[/cyan] samples"
            f"  [dim]({config.quant.calibration_method})[/dim]",
        )
        # Suppress tqdm/datasets progress bars during quantize
        # to keep Live display clean
        _datasets_available = False
        try:
            import datasets

            datasets.disable_progress_bars()
            _datasets_available = True
        except ImportError:
            pass  # datasets package not installed; progress bar suppression not needed

        t0 = time.monotonic()
        try:
            quant_result = quantize_onnx(
                model_path=current_path,
                output_path=quantized_path,
                config=config.quant,
                use_external_data=True,
            )
        finally:
            if _datasets_available:
                datasets.enable_progress_bars()
        if not quant_result.success:
            errors = ", ".join(quant_result.errors) if quant_result.errors else "Unknown"
            sl.set_error(errors)
            raise RuntimeError(f"Quantization failed: {errors}")
        current_path = quantized_path
        _quant_elapsed = time.monotonic() - t0
        sl.set_done(_quant_elapsed)
        sl.kv(
            "Precision:",
            f"[cyan]{config.quant.weight_type}/"
            f"{config.quant.activation_type}[/cyan]"
            f"  [dim](weight/activation)[/dim]",
        )
        sl.artifact(
            str(quantized_path),
            _safe_size(quantized_path),
        )
        sl.blank()
    stage_timings.append(("Quantize", _quant_elapsed))
    return current_path


def _run_compile_stage(
    *,
    config: WinMLBuildConfig,
    current_path: Path,
    compiled_path: Path,
    stage_timings: list[tuple[str, float | None]],
) -> Path:
    """Run the compile stage inside a StageLive context (if compile is configured).

    Shows graph summary after compilation and appends timing to stage_timings.

    Args:
        config: Build configuration.
        current_path: Input model path.
        compiled_path: Output path for compiled model.
        stage_timings: List to append (stage_name, elapsed) tuple to.

    Returns:
        Updated current_path (compiled_path if compilation ran, else unchanged).
    """
    from ..compiler import compile_onnx
    from ..onnx import copy_onnx_model
    from ..utils.console import StageLive, get_onnx_graph_summary

    if config.compile is None:
        return current_path

    with StageLive("compile", console) as sl:
        _cp = ""
        if hasattr(config.compile, "ep_config") and config.compile.ep_config:
            _cp = f" for {config.compile.ep_config.provider.upper()}"
        sl.set_status(f"Compiling{_cp}...")
        t0 = time.monotonic()
        compile_result = compile_onnx(
            model_path=current_path,
            output_path=compiled_path,
            config=config.compile,
        )
        if hasattr(compile_result, "success") and not compile_result.success:
            errors = ", ".join(compile_result.errors) if compile_result.errors else "Unknown"
            sl.set_error(errors)
            raise RuntimeError(f"Compilation failed: {errors}")
        if (
            compile_result.output_path
            and Path(compile_result.output_path).resolve() != compiled_path.resolve()
        ):
            copy_onnx_model(compile_result.output_path, compiled_path)
        if not compiled_path.exists():
            raise RuntimeError(f"Compile reported success but output not found: {compiled_path}")
        current_path = compiled_path
        _compile_elapsed = time.monotonic() - t0
        sl.set_done(_compile_elapsed)

        # Graph summary
        try:
            summary = get_onnx_graph_summary(compiled_path)
            op_parts = ", ".join(
                f"[cyan]{op}[/cyan] ({count})"
                for op, count in list(summary["op_counts"].items())[:8]
            )
            sl.detail(f"[bold]Graph:[/bold]  {op_parts}")
        except Exception:
            logger.debug("Could not load graph summary", exc_info=True)

        sl.artifact(
            str(compiled_path),
            _safe_size(compiled_path),
        )
    stage_timings.append(("Compile", _compile_elapsed))
    return current_path


# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================


def _build_hf_pipeline(
    *,
    config: WinMLBuildConfig,
    model_id: str | None,
    output_dir: Path,
    rebuild: bool,
    cache_key: str | None,
    ep: str | None,
    device: str | None,
    extra_kwargs: dict[str, Any],
) -> list[tuple[str, float | None]] | None:
    """HF build pipeline with cascading StageLive per stage.

    Returns list of (stage_name, elapsed_seconds | None) for summary,
    or None if build was reused.
    """
    from ..build.hf import _load_model
    from ..export import export_onnx
    from ..onnx import copy_onnx_model
    from ..utils.console import StageLive

    max_iters: int = extra_kwargs.pop("hack_max_optim_iterations", 3)
    model_label = model_id or "random-init"

    # ── Validate + setup ─────────────────────────────────────────
    try:
        config.validate()
    except ValueError as e:
        raise ValueError(f"Config validation failed: {e}") from e

    output_dir.mkdir(parents=True, exist_ok=True)

    def _name(base: str) -> str:
        return f"{cache_key}_{base}" if cache_key else base

    export_path = output_dir / _name("export.onnx")
    optimized_path = output_dir / _name("optimized.onnx")
    quantized_path = output_dir / _name("quantized.onnx")
    compiled_path = output_dir / _name("compiled.onnx")
    final_path = output_dir / _name("model.onnx")
    config_path = output_dir / _name("winml_build_config.json")

    # Reuse check
    if final_path.exists() and not rebuild:
        _print_reused(final_path)
        return None

    stage_timings: list[tuple[str, float | None]] = []

    # Clean old artifacts on rebuild
    if rebuild:
        pattern = f"{cache_key}_*.onnx" if cache_key else "*.onnx"
        for old in output_dir.glob(pattern):
            old.unlink()

    current_path = export_path

    # ── Export stage ──────────────────────────────────────────────
    with StageLive("export", console) as sl:
        sl.set_status("Exporting to ONNX...")

        # Load + export (blocking)
        pytorch_model = _load_model(config, model_id, trust_remote_code=False)
        t0 = time.monotonic()
        export_onnx(
            model=pytorch_model,
            output_path=export_path,
            export_config=config.export,
            model_id=model_label,
            task=config.loader.task,
            verbose=False,
            use_external_data=True,
        )
        _export_elapsed = time.monotonic() - t0
        sl.set_done(_export_elapsed)
        # Meta shown after export completes (avoids duplicate in Live frame)
        if config.loader.model_class:
            sl.kv("Model class:", f"[cyan]{config.loader.model_class}[/cyan]")
        if config.loader.task:
            sl.kv("Task:", f"[cyan]{config.loader.task}[/cyan]")
        _show_io(sl, config)
        sl.artifact(str(export_path), _safe_size(export_path))
        sl.blank()

    stage_timings.append(("Export", _export_elapsed))

    # ── Optimize stage ───────────────────────────────────────────
    current_path, _ = _run_optimize_stage(
        config=config,
        model_path=current_path,
        optimized_path=optimized_path,
        ep=ep,
        device=device,
        max_iters=max_iters,
        stage_timings=stage_timings,
        show_io_first=False,
    )

    # Persist config after autoconf
    config_path.write_text(json.dumps(config.to_dict(), indent=2))

    # ── Quantize stage ───────────────────────────────────────────
    current_path = _run_quantize_stage(
        config=config,
        current_path=current_path,
        quantized_path=quantized_path,
        stage_timings=stage_timings,
    )

    # ── Compile stage ────────────────────────────────────────────
    current_path = _run_compile_stage(
        config=config,
        current_path=current_path,
        compiled_path=compiled_path,
        stage_timings=stage_timings,
    )

    # ── Finalize ─────────────────────────────────────────────────
    if current_path != final_path:
        copy_onnx_model(current_path, final_path)

    return stage_timings


def _build_onnx_pipeline(
    *,
    config: WinMLBuildConfig,
    onnx_path: Path,
    output_dir: Path,
    rebuild: bool,
    ep: str | None,
    device: str | None,
    extra_kwargs: dict[str, Any],
) -> list[tuple[str, float | None]] | None:
    """ONNX build pipeline with cascading StageLive per stage.

    Returns list of (stage_name, elapsed_seconds | None) for summary,
    or None if build was reused.
    """
    from ..onnx import copy_onnx_model

    max_iters: int = extra_kwargs.pop("hack_max_optim_iterations", 3)

    # ── Validate + setup ─────────────────────────────────────────
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    try:
        config.validate()
    except ValueError as e:
        raise ValueError(f"Config validation failed: {e}") from e

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = onnx_path.stem
    optimized_path = output_dir / f"{stem}_optimized.onnx"
    quantized_path = output_dir / f"{stem}_quantized.onnx"
    compiled_path = output_dir / f"{stem}_compiled.onnx"
    final_path = output_dir / "model.onnx"
    config_path = output_dir / "winml_build_config.json"

    # Reuse check
    if final_path.exists() and not rebuild:
        _print_reused(final_path)
        return None

    stage_timings: list[tuple[str, float | None]] = []

    if rebuild:
        for old in output_dir.glob("*.onnx"):
            old.unlink()
        for old in output_dir.glob("*.onnx.data"):
            old.unlink()

    # Copy input ONNX to output dir
    current_path = output_dir / onnx_path.name
    if current_path.resolve() != onnx_path.resolve():
        copy_onnx_model(onnx_path, current_path)

    # ── Optimize stage (first stage for ONNX — show I/O here) ────
    current_path, _ = _run_optimize_stage(
        config=config,
        model_path=current_path,
        optimized_path=optimized_path,
        ep=ep,
        device=device,
        max_iters=max_iters,
        stage_timings=stage_timings,
        show_io_first=True,
    )

    config_path.write_text(json.dumps(config.to_dict(), indent=2))

    # ── Quantize stage ───────────────────────────────────────────
    current_path = _run_quantize_stage(
        config=config,
        current_path=current_path,
        quantized_path=quantized_path,
        stage_timings=stage_timings,
    )

    # ── Compile stage ────────────────────────────────────────────
    current_path = _run_compile_stage(
        config=config,
        current_path=current_path,
        compiled_path=compiled_path,
        stage_timings=stage_timings,
    )

    # ── Finalize ─────────────────────────────────────────────────
    if current_path != final_path:
        copy_onnx_model(current_path, final_path)

    return stage_timings
