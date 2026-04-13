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
    winml build -c config.json -m bert-base-uncased -o output/ --no-quant --no-compile
    winml build -c config.json -m microsoft/resnet-50 --random-init -o output/
    winml build -c config.json -m microsoft/resnet-50 -o output/ --rebuild -v
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console


if TYPE_CHECKING:
    from typing import Any

    from torch import nn

    from ..build import BuildResult
    from ..config import WinMLBuildConfig

logger = logging.getLogger(__name__)
console = Console(stderr=True)


# =============================================================================
# CLI HELPERS
# =============================================================================


def _load_config(
    config_file: str,
    *,
    no_quant: bool = False,
    no_compile: bool = False,
) -> WinMLBuildConfig | list[WinMLBuildConfig]:
    """Load WinMLBuildConfig from JSON file with CLI overrides.

    Supports both single config (JSON object) and module mode (JSON array).

    Args:
        config_file: Path to JSON config file.
        no_quant: If True, set config.quant = None (skip quantization).
        no_compile: If True, set config.compile = None (skip compilation).

    Returns:
        Single WinMLBuildConfig for normal mode, or list for module mode.

    Raises:
        click.UsageError: If config file is invalid.
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
        if no_compile:
            cfg.compile = None
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
    from ..loader.config import resolve_loader_config

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


@click.command()
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
    required=True,
    help="HuggingFace model ID or path to .onnx file.",
    # --model is mandatory because random-weight builds (omitting --model) are
    # unreliable: AutoConfig.for_model() returns architecture class defaults
    # which can differ from pretrained configs in ways that cause silent
    # runtime failures.  E.g. MPNet/Roberta-family models set
    # max_position_embeddings = usable_length + pad_token_id + 1 (514) in the
    # pretrained config, but the class default is only 512.  The smaller
    # embedding table causes "index out of range in self" during ONNX export
    # tracing -- a position-offset OOB that the OnnxConfig-level fix (PR #415)
    # cannot reach because HTPExporter uses pre-populated input_tensors, not
    # Optimum's input generation path.  Supporting random-init reliably would
    # require storing the full pretrained HF config (or at least the model ID)
    # in the build config so _load_model can call AutoConfig.from_pretrained()
    # instead of AutoConfig.for_model().  Until that plumbing exists, require
    # --model to guarantee correct model instantiation.
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
    "--random-init",
    is_flag=True,
    default=False,
    help="Skip weight download; use model config with random weights.",
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
    "--no-compile",
    is_flag=True,
    default=False,
    help="Skip compilation (overrides config)",
)
@click.option(
    "--no-optimize",
    is_flag=True,
    default=False,
    help="Skip optimization (for pre-quantized ONNX models)",
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
    "--max-optim-iterations",
    "max_optim_iterations",
    type=int,
    default=None,
    help="Maximum autoconf re-optimization rounds (default: 3). --no-analyze sets this to 0.",
)
@click.option(
    "--trust-remote-code",
    is_flag=True,
    default=False,
    help="Trust remote code for custom model architectures (e.g., Mu2).",
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
    random_init: bool,
    rebuild: bool,
    no_quant: bool,
    no_compile: bool,
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

        # Export + optimize only
        winml build -c config.json -m bert-base-uncased -o output/ --no-quant --no-compile

        # Random-weight build (no weight download)
        winml build -c config.json -m microsoft/resnet-50 --random-init -o output/

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
        from ..session.ep_registry import WinMLEPRegistry

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

            console.print()
            console.print("[bold]winml build[/bold] (module mode)")
            console.print(f"  Config:     {Path(config_file).name}")
            console.print(f"  Modules:    {len(configs)}")
            console.print(f"  Output:     {resolved_dir}")
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

            summary_path = resolved_dir / "module_summary.json"
            summary = {
                "model_id": model_id or "random-init",
                "module_class": configs[0].loader.model_class or "unknown",
                "instance_count": len(summary_instances),
                "instances": summary_instances,
            }
            summary_path.write_text(json.dumps(summary, indent=2))
            console.print(f"  Summary: {summary_path}")

            console.print()

        else:
            # ---- SINGLE MODEL MODE ----
            config = config_or_configs

            # Resolve output directory and cache_key
            cache_key: str | None = None
            if use_cache:
                from ..cache import get_cache_dir, get_cache_key, get_model_dir
                from ..loader.task import get_task_abbrev

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

            # Report build plan
            model_label = f"{model_id} (random-init)" if random_init else model_id
            console.print()
            console.print("[bold]winml build[/bold]")
            console.print(f"  Config:     {Path(config_file).name}")
            console.print(f"  Model:      {model_label}")
            console.print(f"  Output:     {resolved_dir}")
            console.print()

            # Call build API (late import to speed up CLI startup)
            from .config import _is_onnx_file

            if model_id and _is_onnx_file(model_id):
                from ..build import build_onnx_model

                result = build_onnx_model(
                    onnx_path=Path(model_id),
                    config=config,
                    output_dir=resolved_dir,
                    rebuild=rebuild,
                    ep=ep,
                    device=device,
                    **extra_kwargs,
                )
            else:
                from ..build import build_hf_model

                result = build_hf_model(
                    config=config,
                    output_dir=resolved_dir,
                    model_id=model_id,
                    rebuild=rebuild,
                    random_init=random_init,
                    cache_key=cache_key,
                    ep=ep,
                    device=device,
                    **extra_kwargs,
                )

            # Report results
            if result.reused:
                console.print(f"  Existing artifact: {result.final_onnx_path}")
                console.print("  Use --rebuild to force rebuild.")
            else:
                for stage in result.stages_completed:
                    t = result.stage_timings.get(stage, 0)
                    console.print(f"  {stage:<12} done  ({t:.1f}s)")
                for stage in result.stages_skipped:
                    console.print(f"  {stage:<12} skipped")
                console.print()
                console.print(f"  Build complete in {result.elapsed:.1f}s")
                console.print(f"  Final artifact: {result.final_onnx_path}")

            console.print()

    except click.UsageError:
        raise  # Let click handle its own errors
    except ValueError as e:
        raise click.UsageError(str(e)) from e
    except Exception as e:
        if verbose:
            logger.exception("Build failed")
        raise click.ClickException(f"Build failed: {e}") from e
