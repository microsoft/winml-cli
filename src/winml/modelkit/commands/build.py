# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Build command for WinML CLI.

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

import copy
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import click
from rich.logging import RichHandler

from ..utils import cli as cli_utils
from ..utils.console import (
    detect_model_source,
    get_console,
    print_error,
    print_final,
    print_setup,
    print_stages_header,
)
from ..utils.logging import configure_logging


if TYPE_CHECKING:
    from typing import Any

    from torch import nn

    from ..build import BuildResult
    from ..config import WinMLBuildConfig
    from ..utils.constants import EPName, EPNameOrAlias

logger = logging.getLogger(__name__)
console = get_console()


# =============================================================================
# CLI HELPERS
# =============================================================================


def _warn_partial_composite_build(completed: list[str], output_dir: Path) -> None:
    """Warn that a composite build failed mid-run, listing completed components.

    We deliberately do NOT delete anything: the targets may be pre-existing
    artifacts the user chose to ``--rebuild``, and a component can fail before
    writing anything, so auto-deleting could destroy artifacts this run never
    actually wrote. Instead we surface the completed sub-models and let the user
    decide whether to keep or remove the partial build.
    """
    if not completed:
        return
    console.print(
        "\n[yellow]Warning:[/yellow] composite build did not finish; "
        f"{len(completed)} sub-model(s) were built by this run:"
    )
    for name in completed:
        console.print(f"  • {name} ({output_dir / f'{name}_model.onnx'})")
    console.print(
        "[yellow]The build did not complete for every sub-model.[/yellow] "
        "Review these artifacts and remove them if you don't want to keep the "
        "partial build."
    )


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

    _, hf_config, resolved_class_typed, _resolution = resolve_loader_config(
        model_type=model_type,
        task=task,
    )
    # Annotated Any: resolver returns bare `type` but the class is a HF model
    # with extra methods (from_config) that bare `type` doesn't expose.
    resolved_class: Any = resolved_class_typed

    try:
        model = resolved_class(hf_config)
    except OSError as e:
        logger.debug("Direct construction failed (%s), using from_config()", e)
        model = resolved_class.from_config(hf_config)

    model.eval()
    return cast("nn.Module", model)


def _build_modules(
    configs: list[WinMLBuildConfig],
    output_dir: Path,
    *,
    rebuild: bool = False,
    ep: EPNameOrAlias | None = None,
    device: str | None = None,
    allow_unsupported_nodes: bool = False,
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
        allow_unsupported_nodes: If True, warn instead of failing the build when
            the analyzer reports unsupported nodes that persist.

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
            allow_unsupported_nodes=allow_unsupported_nodes,
        )
        results.append(result)

    return results


def _validate_task_supported_for_model(
    model_id: str,
    task: str,
    *,
    task_field_name: str = "task",
    trust_remote_code: bool = False,
    library_name: str = "transformers",
    hf_config: Any | None = None,
) -> Any:
    """Validate that a task is supported for a model's architecture.

    Private helper for ``winml build`` only. Loads HuggingFace config metadata
    and validates against ``TasksManager`` supported-task mapping.

    Why this lives here and not in ``loader/`` as public API:
        Only ``winml build`` accepts task and model from independent sources
        (config JSON's ``loader.task`` + ``--model``) and runs the full
        export+optimize+quantize+compile pipeline that benefits from a fast
        upfront fail. Other CLI entrypoints get equivalent coverage through
        their existing resolution paths:

        - ``winml config`` derives task from the model when both are present,
          so the mismatch can't be silently constructed.
        - ``winml export`` / ``winml perf`` surface incompatibilities through
          ``resolve_cfg`` -> ``ONNXConfigNotFoundError`` later in the call.

        Promoting this to public API would signal that any command should
        wire it in, which is not the current design. If a second caller
        appears, move this back to ``loader/`` and re-export it.

    Args:
        model_id: HuggingFace model ID or local path.
        task: Requested task name.
        task_field_name: Field label used in user-facing error messages.
        trust_remote_code: Whether to trust remote/custom code while loading config.
        library_name: Source library for TasksManager lookup.
        hf_config: Optional pre-loaded HF config. When supplied, the
            ``AutoConfig.from_pretrained`` round-trip is skipped. Used by
            ``_validate_loader_tasks_for_model`` to preflight multiple tasks
            against the same model without re-fetching.

    Returns:
        The loaded (or passed-through) HuggingFace config. Callers can reuse
        this to avoid a duplicate ``AutoConfig.from_pretrained`` later
        (see PR #719 -- same deduping pattern as ``resolve_loader_config``).

    Raises:
        ValueError: If the task is not supported for the model architecture.
    """
    from ..export.io import ensure_hf_models_registered
    from ..loader.task import TASK_SYNONYM_EXTENSIONS, get_supported_tasks, normalize_task

    if hf_config is None:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
    model_type = getattr(hf_config, "model_type", None)
    if not model_type:
        return hf_config

    # Ensure optimum.exporters.onnx.model_configs is imported before querying
    # the registry. TasksManager._SUPPORTED_MODEL_TYPE is populated lazily
    # when optimum's ONNX model_configs module is first imported (triggered by
    # any import of optimum.exporters.onnx). Without this, get_supported_tasks
    # returns [] for models like resnet that are registered there, not in the
    # winml custom registry.
    ensure_hf_models_registered()

    supported_tasks = get_supported_tasks(model_type, library_name=library_name)
    # If the upstream registry has no task list for this architecture,
    # defer to downstream loader resolution instead of hard-failing here.
    if not supported_tasks:
        return hf_config

    # [1] Verbatim canonical match — definitive accept. Comparing without
    #     normalization first means an arch that lists `image-feature-extraction`
    #     in its supported set accepts that name as-is, while a text-only arch
    #     that lists only `feature-extraction` does not silently accept it via
    #     Optimum's synonym collapse on this branch.
    if task in supported_tasks:
        return hf_config

    # [2] HF-pipeline-only task names that Optimum's TasksManager does not
    #     know but the rest of the CLI accepts (e.g. ``next-sentence-prediction``
    #     handled via HF_TASK_DEFAULTS, ``mask-generation`` preserved for SAM2).
    #     These are routed downstream by export/io.py::map_task_synonym, so
    #     rejecting here would break invocations that ``winml config`` and
    #     ``winml export`` accept.
    if task in TASK_SYNONYM_EXTENSIONS:
        return hf_config

    # [2.5] Composite pipeline tasks (summarization / translation /
    #     table-question-answering / …) fan out to an encoder/decoder pair via the
    #     composite registry rather than a single Optimum export, so they never
    #     appear in Optimum's supported_tasks. Accept when the task is a registered
    #     composite for this architecture. ensure_hf_models_registered() above has
    #     already populated the registry, so this is a cheap lookup.
    from ..loader import composite_pipeline_tasks

    if task in composite_pipeline_tasks(model_type.lower().replace("_", "-")):
        return hf_config

    # [3] Optimum synonym fallback — e.g. ``masked-lm`` -> ``fill-mask``.
    #     Accept, but warn so users converge on the canonical spelling.
    #
    #     Known limitation: Optimum collapses text/image variants of
    #     feature-extraction (``image-feature-extraction`` -> ``feature-extraction``)
    #     and routes ``sentence-similarity`` -> ``feature-extraction``. This
    #     branch therefore silently accepts cross-modality combinations such as
    #     ``--task image-feature-extraction`` against a text-only arch. Such
    #     mismatches must be caught downstream where the HF-pipeline-keyed
    #     registries see the un-collapsed ``loader.task`` value.
    normalized = normalize_task(task)
    normalized_supported = {normalize_task(t) for t in supported_tasks}
    if normalized in normalized_supported:
        if normalized != task:
            logger.warning(
                "%s=%r matches via Optimum synonym mapping; consider using the canonical name %r.",
                task_field_name,
                task,
                normalized,
            )
        return hf_config

    supported_list = ", ".join(supported_tasks)
    raise ValueError(
        f"{task_field_name}='{task}' is not supported for --model {model_id} "
        f"(architecture: {model_type}).\n"
        f"Supported tasks: {supported_list}."
    )


def _validate_loader_tasks_for_model(
    *,
    model_id: str | None,
    is_onnx: bool,
    configs: list[WinMLBuildConfig],
    trust_remote_code: bool,
) -> Any | None:
    """Validate config loader task(s) against --model architecture.

    This runs at command entry before setup/stage output so incompatible
    config/model combinations fail with an actionable one-line error.

    Loads ``AutoConfig`` at most once and reuses it across every per-task
    check, then returns it so the build pipeline can plumb it down to
    ``load_hf_model`` and avoid the second/third round-trip that PR #719
    deduped on the inspect path.

    See ``_validate_task_supported_for_model`` for the rationale on why this
    preflight is wired into ``winml build`` only.

    Returns:
        Pre-loaded ``PretrainedConfig`` (caller should pass this into
        ``_run_single_build`` so ``load_hf_model`` skips its own
        ``AutoConfig.from_pretrained`` call), or ``None`` when no model_id
        was provided / model_id is an ONNX file / no task to validate.
    """
    if model_id is None:
        return None

    if is_onnx:
        return None

    tasks = {
        cfg.loader.task for cfg in configs if cfg.loader is not None and cfg.loader.task is not None
    }
    if not tasks:
        return None

    hf_config: Any | None = None
    for task in sorted(tasks):
        hf_config = _validate_task_supported_for_model(
            model_id=model_id,
            task=task,
            task_field_name="config.loader.task",
            trust_remote_code=trust_remote_code,
            hf_config=hf_config,
        )
    return hf_config


# =============================================================================
# CLI COMMAND
# =============================================================================


@click.command("build")
@click.option(
    "-c",
    "--config",
    "config_file",
    type=click.Path(exists=True),
    required=False,
    default=None,
    help="WinMLBuildConfig JSON file. If omitted, config is auto-generated from -m.",
)
@cli_utils.model_option(
    required=False,
    help_text="HuggingFace model ID or path to .onnx file. Omit for random-weight build.",
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
    "--use-cache/--no-use-cache",
    default=False,
    show_default=True,
    help="Use WinML CLI global cache (~/.cache/winml/). Mutually exclusive with -o.",
)
@click.option(
    "--rebuild/--no-rebuild",
    default=False,
    show_default=True,
    help="Overwrite existing artifacts and rebuild",
)
@cli_utils.quant_option()
@cli_utils.compile_option(
    default=None,
    help_text="Override compilation. --compile forces enable (config must have a compile section). "
    "--no-compile forces skip. Default: inherit from config; when auto-generating "
    "config (no -c), compilation is off unless --compile is passed.",
)
@cli_utils.ep_option(
    required=False,
    optional_message="Falls back to compile config EP if not set.",
)
@cli_utils.device_option(
    required=False,
    default="auto",
    include_auto=True,
    optional_message="Default: auto-detect.",
)
@cli_utils.precision_option(
    optional_message="With -c, applied only when --device or --precision is passed.",
)
@cli_utils.analyze_option()
@cli_utils.optimize_option()
@cli_utils.max_optim_iterations_option()
@cli_utils.allow_unsupported_nodes_option()
@cli_utils.trust_remote_code_option(
    optional_message="Trust remote code for custom model architectures (e.g., Mu2)."
)
@click.option(
    "--submodel",
    type=str,
    default=None,
    help=(
        "Build a specific sub-model from a composite model "
        "(e.g., 'encoder', 'decoder'). Omit to build all sub-models automatically."
    ),
)
@cli_utils.verbosity_options()
@cli_utils.no_color_option()
@click.pass_context
def build(
    ctx: click.Context,
    config_file: str | None,
    model: str | None,
    output_dir: str | None,
    use_cache: bool,
    rebuild: bool,
    quant: bool,
    no_compile: bool | None,
    optimize: bool,
    ep: EPNameOrAlias | None,
    device: str,
    precision: str,
    analyze: bool,
    max_optim_iterations: int | None,
    allow_unsupported_nodes: bool,
    trust_remote_code: bool,
    submodel: str | None,
    verbose: int,
    quiet: bool,
) -> None:
    r"""Build a WinML-optimized ONNX model from a HuggingFace model or .onnx file.

    If -c is omitted, config is auto-generated from the model ID (-m required).
    Specify either --output-dir or --use-cache for artifact destination.

    If -m points to an existing .onnx file, the build skips export and runs
    optimize -> quantize -> compile directly (ONNX build path).

    \b
    Examples:
        # Auto-generate config (no -c needed)
        winml build -m microsoft/resnet-50 -o output/

        # Full pipeline with explicit config
        winml build -c config.json -m microsoft/resnet-50 -o output/

        # Device-aware auto precision (npu->w8a16, gpu/cpu->fp16)
        winml build -m microsoft/resnet-50 -o output/ --device npu --precision auto

        # Force fp16 (skips quantization)
        winml build -m bert-base-uncased -o output/ --device gpu --precision fp16

        # Build from pre-exported ONNX file
        winml build -c config.json -m model.onnx -o output/

        # Export + optimize only (config must have compile=null, or pass --no-compile to force skip)
        winml build -c config.json -m bert-base-uncased -o output/ --no-quant --no-compile

        # Use global cache
        winml build -m microsoft/resnet-50 --use-cache

        # Force rebuild
        winml build -c config.json -m microsoft/resnet-50 -o output/ --rebuild

        # Build with INT8 quantization
        winml build -m microsoft/resnet-50 -o output/ --precision int8

        # Build with mixed precision (INT8 weights, INT8 activations)
        winml build -m microsoft/resnet-50 -o output/ --precision w8a8
    """
    # Merge top-level -v/-q with subcommand-level flags so either position works.
    verbose, quiet = cli_utils.resolve_verbosity(ctx, verbose, quiet)
    configure_logging(verbosity=verbose, quiet=quiet)

    # Validate mutual exclusion
    if output_dir and use_cache:
        raise click.UsageError("--output-dir and --use-cache are mutually exclusive.")
    if not output_dir and not use_cache:
        raise click.UsageError("One of --output-dir or --use-cache is required.")

    # Validate precision value early for better error messages.
    if precision is not None:
        from ..config.precision import _is_valid_precision

        if not _is_valid_precision(precision.lower()):
            raise click.UsageError(
                f"Invalid precision '{precision}'. "
                "Expected: auto, fp32, fp16, int8, int16, or w{{x}}a{{y}} (e.g., w8a8, w8a16)."
            )

    # If ep unspecified, resolve the target device and pick the highest-priority
    # EP compatible with it. Avoids selecting an EP that does not match the host
    # hardware -- analyzing for the wrong EP leaves black nodes that block a
    # later build targeting the actual device (#663).
    #
    # resolve_check_device_ep() either returns a device with >=1 available EP
    # (auto-mode walks the priority list, falls back to cpu which is always
    # valid), or raises ValueError for an explicit device with no compatible EP.
    # So the following available_eps[0] is safe whenever it returns.
    if ep is None:
        from ..sysinfo import resolve_check_device_ep

        try:
            resolved_device, _, available_eps = resolve_check_device_ep(device=device, ep=ep)
        except ValueError as e:
            raise click.UsageError(str(e)) from e
        device = resolved_device
        ep = available_eps[0]
        logger.info("Auto-resolved device=%s, EP=%s", resolved_device, ep)

    try:
        # Hub-hosted ONNX (e.g. ``onnx-community/sam3-tracker-ONNX/onnx/...``)
        # is downloaded once and treated as a local .onnx file thereafter.
        if model is not None:
            model = cli_utils.normalize_model_arg(model)

        # Load or auto-generate config
        if config_file is not None:
            config_or_configs = _load_config(
                config_file,
                no_quant=not quant,
                no_compile=no_compile,
            )
        else:
            if not model:
                raise click.UsageError("-m/--model is required when -c is not provided.")
            from ..config import generate_build_config

            # When ``model`` resolves to an .onnx file (either a local path or
            # a Hub-hosted ONNX ref that was just downloaded by
            # ``normalize_model_arg``), route to the ONNX config generator instead
            # of treating the path as a HuggingFace repo id (which would try to
            # load the .onnx file as a JSON config and crash).
            if cli_utils.is_onnx_file_path(model):
                config_or_configs = generate_build_config(
                    onnx_path=model,
                    device=device,
                    precision=precision,
                    ep=ep,
                )
            else:
                config_or_configs = generate_build_config(
                    model,
                    trust_remote_code=trust_remote_code,
                    device=device,
                    precision=precision,
                    ep=ep,
                )
            if not quant:
                config_or_configs.quant = None
            # Auto-generated configs: compile disabled by default unless
            # --compile was explicitly passed (no_compile=False).
            if no_compile is None:
                no_compile = True
            if no_compile:
                config_or_configs.compile = None

        # If --device or --precision was explicitly provided, patch quant/compile
        # to honor the requested policy. fp16/fp32 clear quant; npu/int8 etc set it.
        if cli_utils.is_cli_provided(ctx, "device") or cli_utils.is_cli_provided(ctx, "precision"):
            from ..compiler.configs import WinMLCompileConfig

            def _patch_device(cfg: WinMLBuildConfig) -> None:
                from ..config import resolve_quant_compile_config

                resolved_quant, _ = resolve_quant_compile_config(
                    device=device, precision=precision, ep=ep
                )
                if cfg.skip_optimize or not quant or resolved_quant is None:
                    cfg.quant = None
                elif cfg.quant is None:
                    # Populate calibration identifiers from the loader/model
                    # so the resulting config passes HF-build validation.
                    if cfg.loader is not None and cfg.loader.task:
                        resolved_quant.task = cfg.loader.task
                    if model:
                        resolved_quant.model_id = model
                    cfg.quant = resolved_quant
                else:
                    # Only update precision fields; preserve task/model_id
                    # and other calibration settings from the existing config.
                    cfg.quant.weight_type = resolved_quant.weight_type
                    cfg.quant.activation_type = resolved_quant.activation_type
                    cfg.quant.mode = resolved_quant.mode
                    if resolved_quant.mode == "rtn":
                        cfg.quant.rtn_bits = resolved_quant.rtn_bits
                        cfg.quant.rtn_block_size = resolved_quant.rtn_block_size
                        cfg.quant.rtn_symmetric = resolved_quant.rtn_symmetric
                        cfg.quant.rtn_accuracy_level = resolved_quant.rtn_accuracy_level
                # Store the original precision string for stage display
                if precision:
                    cfg.precision = precision.lower()  # type: ignore[attr-defined]
                if cfg.compile is not None and cfg.compile.ep_config is not None:
                    provider = cfg.compile.ep_config.provider
                    patched = WinMLCompileConfig.for_provider(provider, device=device)
                    if patched is not None:
                        cfg.compile = patched

            if isinstance(config_or_configs, list):
                for _cfg in config_or_configs:
                    _patch_device(_cfg)
            else:
                _patch_device(config_or_configs)

        # Fail-fast schema validation: ensure the config is valid before
        # printing any banner or creating any output directories. This
        # surfaces malformed configs immediately and prevents partial
        # scratch state when the user passes the wrong file or a
        # hand-edited config (#P1 UX).
        _configs_to_validate: list[WinMLBuildConfig] = (
            config_or_configs if isinstance(config_or_configs, list) else [config_or_configs]
        )
        try:
            for _cfg in _configs_to_validate:
                _cfg.validate()
        except ValueError as e:
            raise click.UsageError(f"Config validation failed: {e}") from e

        model_input = cli_utils.classify_model_input(model) if model else None
        model_is_onnx = (
            model_input is not None and model_input.kind is cli_utils.ModelInputKind.ONNX_FILE
        )

        preloaded_hf_config = _validate_loader_tasks_for_model(
            model_id=model,
            is_onnx=model_is_onnx,
            configs=_configs_to_validate,
            trust_remote_code=trust_remote_code,
        )

        # Build extra kwargs for pipeline control. The optimize/analyze/max-optim
        # mapping is shared with perf and eval via build_pipeline_extra_kwargs.
        extra_kwargs: dict[str, Any] = cli_utils.build_pipeline_extra_kwargs(
            optimize=optimize,
            analyze=analyze,
            max_optim_iterations=max_optim_iterations,
        )
        if trust_remote_code:
            extra_kwargs["trust_remote_code"] = True
        # Always set (even when False) so downstream pipeline functions can rely
        # on the key being present, matching the module-mode path which passes
        # allow_unsupported_nodes explicitly regardless of its value.
        extra_kwargs["allow_unsupported_nodes"] = allow_unsupported_nodes

        if isinstance(config_or_configs, list):
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
                model=model or "random-init",
                config=Path(config_file).name if config_file else "(auto)",
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
                allow_unsupported_nodes=allow_unsupported_nodes,
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
                model_id=model or "random-init",
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
                    model or "random-init",
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
                    extra_kwargs,
                )
            else:
                # Guarded earlier (line ~381: `if not output_dir and not use_cache`).
                if output_dir is None:
                    raise click.UsageError("--output-dir is required when --use-cache is not set.")
                resolved_dir = Path(output_dir)

            # Detect composite pipeline (registry-driven, same pattern as
            # export command). A composite fans out into one build per
            # sub-component; a plain model builds to the single output dir.
            components = None
            if model and not model_is_onnx:
                try:
                    from ..loader.resolution import resolve_composite_components

                    # Only forward an explicit task from a config file. For an
                    # auto-generated config (no -c), loader.task is the
                    # auto-detected task (e.g. "text2text-generation" for
                    # seq2seq), which would take the resolver's explicit-task
                    # path and skip the seq2seq composite bridge. Pass task=None
                    # in that case so detection applies the bridge, while still
                    # forwarding model_type for explicit overrides.
                    task_hint = config.loader.task if (config_file and config.loader) else None
                    model_type_hint = config.loader.model_type if config.loader else None
                    components = resolve_composite_components(
                        model,
                        task=task_hint,
                        model_type=model_type_hint,
                        trust_remote_code=trust_remote_code,
                    )
                except click.ClickException:
                    raise
                except ValueError as e:
                    raise click.UsageError(str(e)) from e
                except RuntimeError:
                    raise
                except OSError as e:
                    logger.debug("Composite detection unavailable (config not resolvable): %s", e)
                except Exception as e:
                    raise click.ClickException(
                        f"Composite model detection failed unexpectedly: {e}"
                    ) from e

            # ── --submodel validation ──────────────────────────────────────
            if submodel is not None:
                if components is None:
                    raise click.BadParameter(
                        f"'{submodel}' was specified, but '{model}' "
                        f"is not a composite model (no sub-models detected).",
                        param_hint="--submodel",
                    )
                if submodel not in components:
                    raise click.BadParameter(
                        f"Unknown sub-model '{submodel}'. "
                        f"Available: {', '.join(components.keys())}",
                        param_hint="--submodel",
                    )
                components = {submodel: components[submodel]}

            if components:
                if use_cache:
                    raise click.UsageError(
                        "--use-cache is not supported for composite models. "
                        "Use --output-dir instead."
                    )
                console.print(
                    f"\n[dim]Composite model: {len(components)} sub-models "
                    f"({', '.join(components)})[/dim]"
                )

                completed: list[str] = []
                try:
                    for name, component_task in components.items():
                        console.print(
                            f"\n[bold blue]Sub-model:[/bold blue] {name} (task={component_task})"
                        )

                        from ..config import generate_build_config as gen_cfg

                        component_config = gen_cfg(
                            model,
                            task=component_task,
                            trust_remote_code=trust_remote_code,
                            device=device,
                            precision=precision,
                            ep=ep,
                        )
                        # Carry over quant/compile settings from the outer config
                        # (already patched by CLI overrides like --no-quant /
                        # --no-compile). Deep-copy to avoid sharing mutable state
                        # across sub-builds, and preserve the component-specific
                        # quant metadata (task, model_id, model_type) that
                        # generate_build_config populated for the sub-model.
                        if config.quant is None:
                            component_config.quant = None
                        else:
                            # Overlay the outer quant settings, but always keep
                            # the calibration identity (task / model_id /
                            # model_type) pointed at *this* sub-model. Prefer the
                            # component's own generated quant metadata; if the
                            # component config produced no quant, derive it from
                            # the sub-model's task, the model arg, and the
                            # component loader so calibration runs against the
                            # sub-model rather than the outer model.
                            component_loader = component_config.loader
                            if component_config.quant is not None:
                                saved_meta = (
                                    component_config.quant.task,
                                    component_config.quant.model_id,
                                    component_config.quant.model_type,
                                )
                            else:
                                saved_meta = (
                                    component_task,
                                    model,
                                    component_loader.model_type if component_loader else None,
                                )
                            component_config.quant = copy.deepcopy(config.quant)
                            (
                                component_config.quant.task,
                                component_config.quant.model_id,
                                component_config.quant.model_type,
                            ) = saved_meta

                        if config.compile is None:
                            component_config.compile = None
                        else:
                            component_config.compile = copy.deepcopy(config.compile)

                        try:
                            component_config.validate()
                        except ValueError as e:
                            raise click.UsageError(
                                f"Config validation failed for sub-model '{name}': {e}"
                            ) from e

                        _run_single_build(
                            config=component_config,
                            config_file=None,
                            model_id=model,
                            is_onnx=False,
                            resolved_dir=resolved_dir,
                            rebuild=rebuild,
                            cache_key=name,
                            ep=ep,
                            device=device,
                            extra_kwargs=dict(extra_kwargs),
                            preloaded_hf_config=preloaded_hf_config,
                        )
                        completed.append(name)
                except BaseException:
                    _warn_partial_composite_build(completed, resolved_dir)
                    raise
            else:
                _run_single_build(
                    config=config,
                    config_file=config_file,
                    model_id=model,
                    is_onnx=model_is_onnx,
                    resolved_dir=resolved_dir,
                    rebuild=rebuild,
                    cache_key=cache_key,
                    ep=ep,
                    device=device,
                    extra_kwargs=extra_kwargs,
                    preloaded_hf_config=preloaded_hf_config,
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
        err_lower = err_str.lower()
        hint = None
        if "disk space" in err_lower or "no space left" in err_lower:
            hint = (
                "Free up disk space (e.g. clear the HuggingFace cache or "
                "~/.cache/winml) and rebuild."
            )
        elif "Quantization failed" in err_str:
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
    config_file: str | None,
    model_id: str | None,
    is_onnx: bool,
    resolved_dir: Path,
    rebuild: bool,
    cache_key: str | None,
    ep: EPNameOrAlias | None,
    device: str | None,
    extra_kwargs: dict[str, Any],
    preloaded_hf_config: Any | None = None,
) -> None:
    """Run single-model build with Rich Live progress per stage."""
    _is_onnx = is_onnx
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
        config=Path(config_file).name if config_file else "(auto)",
        output=str(resolved_dir),
        source=source,
        auto=config.auto,
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
            assert model_id is not None  # _is_onnx implies this
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
                preloaded_hf_config=preloaded_hf_config,
            )

        elapsed = time.monotonic() - start_time
        final_name = f"{cache_key}_model.onnx" if cache_key else "model.onnx"
        final_path = resolved_dir / final_name
        if final_path.exists() and stage_timings:
            config_json = resolved_dir / (
                f"{cache_key}_winml_build_config.json" if cache_key else "winml_build_config.json"
            )
            print_final(
                console,
                elapsed,
                str(final_path),
                stage_timings=stage_timings,
                config=str(config_json) if config_json.exists() else None,
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
    for i, in_spec in enumerate(inputs):
        name = in_spec.name or "(unnamed)"
        shape = str(list(in_spec.shape)) if in_spec.shape else "dynamic"
        dtype = getattr(in_spec, "dtype", None) or "?"
        sl.io_input(name, shape, dtype, first=(i == 0))
    for i, out_spec in enumerate(outputs):
        name = out_spec.name or "(unnamed)"
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
    ep: EPNameOrAlias | None,
    device: str | None,
    max_iters: int,
    stage_timings: list[tuple[str, float | None]],
    show_io_first: bool = False,
    analyze_output_path: Path | None = None,
    allow_unsupported_nodes: bool = False,
    skip_optimize: bool = False,
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
        skip_optimize: When True, skip the ORT graph-optimization pass.
            Used for pre-quantized models (QDQ or QOperator format) whose
            integer ops have no kernel on the host EP.

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
        _current_ep: EPName | None = None
        _current_iter = [0, 0]  # [iteration, max_iter]
        _header_shown = [False]

        def _on_iteration_start(iteration: int, max_iter: int) -> None:
            _ep_bars.clear()
            _ep_counts.clear()
            _ep_totals.clear()
            _current_iter[0] = iteration
            _current_iter[1] = max_iter
            _header_shown[0] = False

        # Resolve "auto" to a concrete device once so that has_rule_data_for_ep
        # doesn't search for non-existent "*_AUTO_*.parquet" files. Use
        # resolve_check_device_ep so an explicit device+ep is validated
        # statically (no availability cross-check): a --no-compile build may
        # target a device absent on this machine (cross-compile), and this call
        # only needs a concrete device name for the rule-data lookup.
        from ..analyze.utils.ep_utils import has_rule_data_for_ep
        from ..sysinfo import resolve_check_device_ep

        _resolved_device, _, _ = resolve_check_device_ep(device=device or "auto", ep=ep)

        def _on_ep_start(ep_name: EPName, operator_counts: dict) -> None:
            nonlocal _current_ep
            _current_ep = ep_name
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
            # Skip bar for EPs with no rule data — all results would be 0/0/0
            if has_rule_data_for_ep(ep_name, _resolved_device or ""):
                _ep_bars[ep_name] = sl.ep_bar_add(ep_name, total=total)

        def _on_node_result(pattern_runtime: Any) -> None:
            ep_name = _current_ep
            if ep_name is None:
                return  # pre-init: _on_ep_start hasn't fired yet
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
            allow_unsupported_nodes=allow_unsupported_nodes,
            skip_optimize=skip_optimize,
            on_ep_start=_on_ep_start,
            on_node_result=_on_node_result,
            on_iteration_start=_on_iteration_start,
            on_patterns_discovered=_on_patterns,
            on_reoptimize=_on_reoptimize,
            use_external_data=True,
            analyze_output_path=analyze_output_path,
        )
        # Mark config as resolved so CI/CD reruns skip the analyzer.
        config.auto = False
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
    """Run the quantize stage (if quant is configured).

    Delegates quantization to ``quantize_onnx(config=...)``.
    The cmd layer only handles UI display and the QDQ skip check.

    Args:
        config: Build configuration.
        current_path: Input model path.
        quantized_path: Output path for quantized model.
        stage_timings: List to append (stage_name, elapsed) tuple to.

    Returns:
        Updated current_path (quantized_path if quantization ran, else unchanged).
    """
    from ..quant import quantize_onnx
    from ..utils.console import StageLive

    if config.skip_optimize:
        config.quant = None
        return current_path

    if config.quant is None:
        # ``generate_onnx_build_config`` and ``ensure_pre_quantized_stamped``
        # (in build/common.py) both clear ``config.quant`` for pre-quantized
        # inputs, so this single check covers both "user-explicit None" and
        # "auto-detected pre-quantized" cases.
        return current_path

    # Determine stage label from quant mode
    is_fp16_only = config.quant.mode == "fp16"
    stage_label = "fp16" if is_fp16_only else "quantize"
    stage_name = "FP16" if is_fp16_only else "Quantize"

    with StageLive(stage_label, console) as sl:
        # Show status based on what we're about to do
        if is_fp16_only:
            sl.set_status("Converting to FP16...")
        elif config.quant.mode == "rtn":
            sl.set_status(f"Quantizing (RTN {config.quant.rtn_bits}-bit)...")
        else:
            sl.set_status(f"Quantizing ({config.quant.weight_type})...")
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

        # Suppress tqdm/datasets progress bars for QDQ calibration
        _datasets_available = False
        if config.quant.mode in ("static", "dynamic"):
            try:
                import datasets

                datasets.disable_progress_bars()
                _datasets_available = True
            except ImportError:
                pass  # datasets package is optional; calibration falls back to random data

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
                import datasets

                datasets.enable_progress_bars()

        if not quant_result.success:
            errors = ", ".join(quant_result.errors) if quant_result.errors else "Unknown"
            sl.set_error(errors)
            raise RuntimeError(f"{stage_name} failed: {errors}")

        elapsed = time.monotonic() - t0
        sl.set_done(elapsed)

        # Show algorithm-specific result details
        if is_fp16_only:
            sl.detail("[dim]I/O types preserved as FP32[/dim]")
        elif config.quant.mode == "rtn":
            sl.kv(
                "Algorithm:",
                f"[cyan]RTN[/cyan]  [dim](weight-only {config.quant.rtn_bits}-bit)[/dim]",
            )
            sl.kv(
                "Config:",
                f"block_size={config.quant.rtn_block_size}, symmetric={config.quant.rtn_symmetric}",
            )
        else:
            sl.kv(
                "Precision:",
                f"[cyan]{config.quant.weight_type}/{config.quant.activation_type}[/cyan]"
                f"  [dim](weight/activation)[/dim]",
            )

        sl.artifact(str(quantized_path), _safe_size(quantized_path))
        sl.blank()

    stage_timings.append((stage_name, elapsed))
    return quantized_path


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
            ep = config.compile.ep_config.provider
            _cp = f" for {ep.upper()}" if ep else ""
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
    ep: EPNameOrAlias | None,
    device: str | None,
    extra_kwargs: dict[str, Any],
    preloaded_hf_config: Any | None = None,
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
    allow_unsupported_nodes: bool = extra_kwargs.pop("allow_unsupported_nodes", False)
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
    analyze_result_path = output_dir / _name("analyze_result.json")

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
        pytorch_model = _load_model(
            config,
            model_id,
            trust_remote_code=False,
            hf_config=preloaded_hf_config,
            model_type=config.loader.model_type,
        )
        t0 = time.monotonic()
        # config.export is None only for the ONNX build path; this is the HF path.
        assert config.export is not None, "HF build path requires config.export"
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
        analyze_output_path=analyze_result_path,
        allow_unsupported_nodes=allow_unsupported_nodes,
    )

    # Persist config after autoconf
    config_path.write_text(json.dumps(config.to_dict(), indent=2))

    # ── Quantize stage ───────────────────────────────────────────
    # A model-type-specific quant policy (e.g. the qwen3_transformer_only w8a16
    # finalizer) is resolved and applied inside ``quantize_onnx`` from
    # ``config.quant.model_type``; no per-call-site dispatch needed here. Carry
    # the resolved variant onto the quant config so configs that were hand-built
    # or loaded from JSON (skipping assemble_build_config) still trigger it.
    if config.quant is not None and config.quant.model_type is None:
        config.quant.model_type = config.loader.model_type

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
    ep: EPNameOrAlias | None,
    device: str | None,
    extra_kwargs: dict[str, Any],
) -> list[tuple[str, float | None]] | None:
    """ONNX build pipeline with cascading StageLive per stage.

    Returns list of (stage_name, elapsed_seconds | None) for summary,
    or None if build was reused.
    """
    from ..build.common import ensure_pre_quantized_stamped
    from ..onnx import copy_onnx_model

    max_iters: int = extra_kwargs.pop("hack_max_optim_iterations", 3)
    allow_unsupported_nodes: bool = extra_kwargs.pop("allow_unsupported_nodes", False)

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
    analyze_result_path = output_dir / "analyze_result.json"

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

    # Keep the CLI ONNX path aligned with the library build paths: if a user
    # supplies a pre-quantized model via ``-c config.json`` we must stamp the
    # config before any stage reads it, otherwise the optimize stage will still
    # run on integer ops and the quantize stage may try to re-quantize.
    ensure_pre_quantized_stamped(config, current_path)

    # Pre-quantized models (QDQ or QOperator format) cannot pass through
    # ORT-based graph optimization on hosts that lack kernels for ops like
    # ``ConvInteger``. The unified pipeline stamps ``config.skip_optimize``
    # exactly once in ``generate_onnx_build_config`` -- downstream stages
    # (here and inside ``build_onnx_model``) read the flag instead of
    # re-running ``is_quantized_onnx`` on the same file.
    is_pre_quantized = config.skip_optimize

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
        analyze_output_path=analyze_result_path,
        allow_unsupported_nodes=allow_unsupported_nodes,
        skip_optimize=is_pre_quantized,
    )

    config_path.write_text(json.dumps(config.to_dict(), indent=2))

    # ── Quantize stage ──────
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
