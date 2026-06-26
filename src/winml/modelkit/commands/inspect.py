# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Inspect input model's WinML CLI configuration.

Resolves loader, exporter, and WinML inference class for a given model,
showing what the build pipeline will use.

Usage:
    winml inspect -m microsoft/resnet-50
    winml inspect --model-type bert --task fill-mask
    winml inspect -m google-bert/bert-base-uncased --format json
    winml inspect --list-tasks
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..inspect.types import InspectResult

import click
from rich.console import Console

from ..utils import cli as cli_utils
from ..utils.logging import configure_logging


logger = logging.getLogger(__name__)
# `console` is stdout-bound — table/JSON output goes here.
# `_stderr_console` is for banners and spinners so they never contaminate
# stdout (important for `--format json` consumers parsing the output).
console = Console()
_stderr_console = Console(stderr=True, highlight=False)

# File extensions that unambiguously indicate a local file path.
# HF model IDs routinely contain dots in version numbers (Phi-3.5, Qwen2.5, …)
# so matching on any suffix would cause false-positives; restrict to known extensions.
_LOCAL_FILE_EXTS = frozenset({".onnx", ".pt", ".pth", ".safetensors", ".bin"})


def _validate_task(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
    """Click-time validation for --task against the hand-coded KNOWN_TASKS set.

    Imports only ..loader.task to keep validation cheap — going through optimum
    would cost ~10s on a warm cache and defeats fail-fast on bad input.
    """
    if value is None:
        return None
    from ..loader.task import KNOWN_TASKS

    if value in KNOWN_TASKS:
        return value
    examples = ", ".join(sorted(KNOWN_TASKS)[:5])
    raise click.UsageError(
        f"Invalid task '{value}'. Valid: {examples}, ... ({len(KNOWN_TASKS)} total). "
        f"See 'winml inspect --list-tasks' for the full list."
    )


def _looks_like_local_path(model_id: str) -> bool:
    """Return True when model_id is explicitly a local path.

    Conservative heuristic — only returns True for unambiguous local indicators:
    path separators, absolute paths, dot/tilde prefixes, or known model file extensions.
    """
    from pathlib import Path

    _p = Path(model_id).expanduser()
    return (
        _p.exists()
        or _p.is_absolute()
        or "\\" in model_id
        or model_id.startswith(("./", "../", "~/"))
        or _p.suffix.lower() in _LOCAL_FILE_EXTS
    )


@click.command("inspect")
@cli_utils.model_option(
    required=False,
    help_text="HuggingFace model ID (e.g., microsoft/resnet-50)",
)
@cli_utils.format_option(choices=["table", "json"], default="table")
@click.option(
    "-t",
    "--task",
    default=None,
    callback=_validate_task,
    help="Override auto-detected task (e.g., image-classification, feature-extraction)",
)
@click.option(
    "-H/-N",
    "--hierarchy/--no-hierarchy",
    default=False,
    show_default=True,
    help="Show HF module hierarchy (uses random weights, no weight download)",
)
@click.option(
    "--list-tasks",
    "list_tasks",
    is_flag=True,
    default=False,
    help="List all known tasks and exit",
)
@click.option(
    "--model-type",
    "model_type",
    default=None,
    help="Override model type (e.g., bert, resnet) — can be used without --model",
)
@click.option(
    "--model-class",
    "model_class",
    default=None,
    help="Override model class (e.g., BertForMaskedLM) — can be used without --model",
)
@cli_utils.verbosity_options()
@click.pass_context
def inspect(
    ctx: click.Context,
    model: str | None,
    output_format: cli_utils.OutputFormat,
    verbose: int,
    quiet: bool,
    task: str | None,
    hierarchy: bool,
    list_tasks: bool,
    model_type: str | None,
    model_class: str | None,
) -> None:
    r"""Inspect input model's WinML CLI configuration.

    Shows the loader, exporter, WinML inference class, I/O specs,
    and build resolution that the pipeline will use for the given model.

    Supports inspection without a model ID via --model-type or --model-class.

    \b
    Examples:
        # Basic inspection
        winml inspect -m microsoft/resnet-50

        # Inspect by model type only (no weight download)
        winml inspect --model-type bert --task fill-mask

        # Override model class
        winml inspect -m custom-model --model-class BertForCTC

        # JSON output
        winml inspect -m google-bert/bert-base-uncased --format json

        # List all known tasks
        winml inspect --list-tasks
    """
    # Handle --list-tasks (no model required).
    # Import the hand-coded KNOWN_TASKS directly from loader.task to keep this
    # branch fast — going through inspect.resolver pulls in ..models which
    # transitively imports transformers and costs ~10s on a warm cache.
    if list_tasks:
        from ..loader.task import KNOWN_TASKS

        for t in sorted(KNOWN_TASKS):
            click.echo(t)
        return

    # Validate: need at least one of model_id, model_type, model_class
    if model is None and model_type is None and model_class is None:
        raise click.UsageError(
            "At least one of -m/--model, --model-type, or --model-class is required. "
            "Use --list-tasks to see available tasks."
        )

    # Classify the input before hitting HF Hub: local paths must exist.
    # _looks_like_local_path uses a conservative allowlist to avoid misclassifying
    # HF IDs with version dots (Phi-3.5, Qwen2.5, …) as local paths.
    if model and _looks_like_local_path(model):
        from pathlib import Path

        _p = Path(model).expanduser()
        if _p.suffix == ".onnx" and _p.is_file():
            raise click.ClickException(
                "ONNX file inspection is not yet supported. "
                "Use 'winml config -m model.onnx' for ONNX build config."
            )
        if not _p.exists():
            raise click.ClickException(f"Local path '{model}' does not exist.")

    # Merge top-level -v/-q with subcommand-level flags so either position
    # works, once and up front. The banner decision below needs the merged
    # --quiet (so both `winml --quiet inspect …` and `winml inspect -q`
    # suppress it); configure_logging needs both. Single source of truth.
    verbose, quiet = cli_utils.resolve_verbosity(ctx, verbose, quiet)

    # Print a banner BEFORE the heavy import chain / network calls so users
    # see immediate feedback instead of ~14 s of silence and assume the
    # command hung (see #543). Banner + spinner go to stderr so `--format
    # json` consumers still get clean stdout. Suppressed in --quiet mode
    # and in JSON mode (Click 8.4 mixes stderr into CliRunner.result.output,
    # and JSON consumers expect clean stdout regardless).
    json_mode = output_format == "json"
    target = model or model_type or model_class
    if not quiet and not json_mode:
        _stderr_console.print(f"[dim]Inspecting [bold]{target}[/bold] …[/dim]")

    from ..inspect import InspectError, ModelNotFoundError, NetworkError
    from ..inspect.formatter import output_json, output_table

    configure_logging(verbosity=verbose, quiet=quiet)

    try:
        if quiet or json_mode:
            result = _inspect_model_v2(
                model_id=model,
                task_override=task,
                model_type_override=model_type,
                model_class_override=model_class,
                include_hierarchy=hierarchy,
            )
        else:
            with _stderr_console.status(
                f"[bold cyan]Resolving {target}…[/bold cyan]",
                spinner="dots",
            ):
                result = _inspect_model_v2(
                    model_id=model,
                    task_override=task,
                    model_type_override=model_type,
                    model_class_override=model_class,
                    include_hierarchy=hierarchy,
                )

        if output_format == "json":
            click.echo(output_json(result, verbose=bool(verbose)))
        else:
            output_table(console, result, verbose=bool(verbose))

    except ModelNotFoundError as e:
        raise click.ClickException(f"Model not found: {e}") from e

    except NetworkError as e:
        raise click.ClickException(f"Network error: {e}") from e

    except InspectError as e:
        raise click.ClickException(f"Inspection error: {e}") from e

    except (ValueError, RuntimeError, OSError) as e:
        logger.exception("Failed to inspect model")
        raise click.ClickException(f"Failed to inspect model: {e}") from e


def _inspect_model_v2(
    model_id: str | None = None,
    task_override: str | None = None,
    model_type_override: str | None = None,
    model_class_override: str | None = None,
    include_hierarchy: bool = False,
) -> InspectResult:
    """Inspect v2 core — calls shared loader/export modules directly.

    Args:
        model_id: HuggingFace model ID (optional when model_type_override set)
        task_override: Task to use instead of auto-detected task
        model_type_override: Model type override (e.g., "bert")
        model_class_override: Model class override (e.g., "BertForMaskedLM")
        include_hierarchy: Whether to extract module hierarchy

    Returns:
        InspectResult dataclass
    """
    import functools

    from transformers import AutoConfig

    from ..export import resolve_io_specs
    from ..inspect import (
        ExporterInfo,
        InspectError,
        InspectResult,
        LoaderInfo,
        ModelNotFoundError,
        NetworkError,
        SupportLevel,
        TensorInfo,
        build_tensor_infos_from_io_specs,
        compile_support_status,
        resolve_cache,
        resolve_composite_info,
        resolve_io_config,
        resolve_processor,
        resolve_winml,
    )
    from ..loader import HF_TASK_DEFAULTS, resolve_loader_config
    from ..models import (
        HF_MODEL_CLASS_MAPPING,
        MODEL_BUILD_CONFIGS,
    )

    # =========================================================================
    # STEP 1: Load parent hf_config once and feed it into resolve_loader_config
    #         to avoid a duplicate AutoConfig.from_pretrained round-trip.
    #         The parent (e.g., CLIPConfig) is preserved here because step 4
    #         inside resolve_loader_config may narrow it to a sub-config
    #         (e.g., CLIPTextConfig) for multimodal models.
    # =========================================================================
    parent_hf_config = None
    if model_id and not model_type_override:
        try:
            parent_hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        except Exception:
            pass  # resolve_loader_config will handle the error properly

    # =========================================================================
    # STEP 2: Shared loader resolution (same call as config command)
    # =========================================================================
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        loader_config, hf_config, _resolved_class, resolution = resolve_loader_config(
            model_id,
            task=task_override,
            model_type=model_type_override,
            model_class=model_class_override,
            hf_config=parent_hf_config,
        )
    except RepositoryNotFoundError as e:
        # Direct HF Hub 404 — keep full message (includes private-repo hint).
        raise ModelNotFoundError(str(e)) from e
    except ValueError as e:
        err_str = str(e).lower()
        if "not found" in err_str or "404" in err_str:
            raise ModelNotFoundError(str(e)) from e
        raise InspectError(str(e)) from e
    except OSError as e:
        # transformers wraps RepositoryNotFoundError as a plain OSError with a
        # recognizable message.  Detect that pattern so users see "Model not found"
        # (with the original hint text) rather than the misleading "Network error".
        err_msg = str(e)
        if "is not a valid model identifier" in err_msg or "is not a local folder" in err_msg:
            raise ModelNotFoundError(err_msg) from e
        raise NetworkError(err_msg) from e

    if parent_hf_config is None:
        parent_hf_config = hf_config

    model_type = loader_config.model_type
    task = loader_config.task
    if model_type is None:
        raise InspectError("Could not resolve model_type from loader config")
    if task is None:
        raise InspectError("Could not resolve task from loader config")
    architectures = getattr(parent_hf_config, "architectures", []) or []

    # =========================================================================
    # STEP 3: provenance comes straight from the resolver (no post-hoc recompute).
    # =========================================================================
    mt = model_type.lower().replace("_", "-")
    task_source = resolution.source.value

    # =========================================================================
    # STEP 4: Derive loader display info
    # =========================================================================
    if (mt, task) in HF_MODEL_CLASS_MAPPING:
        loader_source = "MODEL_CLASS_MAPPING"
        loader_level = SupportLevel.SUPPORTED
    elif task in HF_TASK_DEFAULTS:
        loader_source = "HF_TASK_DEFAULTS"
        loader_level = SupportLevel.DEFAULT
    else:
        loader_source = "TasksManager"
        loader_level = SupportLevel.DEFAULT

    loader_info = LoaderInfo(
        hf_model_class=loader_config.model_class or "Auto (TasksManager)",
        hf_model_class_source=loader_source,
        support_level=loader_level,
    )

    # =========================================================================
    # STEP 5: I/O tensor specs — registry first, then resolve_io_specs
    # =========================================================================
    input_tensors: list[TensorInfo] = []
    output_tensors: list[TensorInfo] = []
    onnx_config_class = None
    onnx_config_source = "none"
    exporter_level = SupportLevel.UNSUPPORTED
    opset_version = 17

    # Path 1: Check MODEL_BUILD_CONFIGS registry for predefined config
    registered = MODEL_BUILD_CONFIGS.get(mt)
    if registered and registered.export and registered.export.input_tensors is not None:
        export_cfg = registered.export
        input_tensors = [
            TensorInfo(name=s.name or "unknown", dtype=s.dtype, shape=s.shape)
            for s in (export_cfg.input_tensors or [])
        ]
        output_tensors = [
            TensorInfo(name=s.name or "unknown") for s in (export_cfg.output_tensors or [])
        ]
        onnx_config_class = f"{mt.upper()}IOConfig"
        onnx_config_source = "MODEL_BUILD_CONFIGS"
        exporter_level = SupportLevel.SUPPORTED
        opset_version = export_cfg.opset_version
    else:
        # Path 2: resolve_io_specs (shared with config command)
        try:
            import optimum.exporters.onnx.model_configs  # noqa: F401
            from optimum.exporters.tasks import TasksManager

            # TasksManager expects Optimum-canonical task names
            from ..loader import resolve_optimum_library, to_optimum_task

            onnx_config_cls = TasksManager.get_exporter_config_constructor(
                exporter="onnx",
                model_type=model_type,
                task=to_optimum_task(task),
                library_name=resolve_optimum_library(model_type),
            )
            if onnx_config_cls:
                config_name = (
                    onnx_config_cls.func.__name__
                    if isinstance(onnx_config_cls, functools.partial)
                    else onnx_config_cls.__name__
                )
                onnx_config_class = config_name
                onnx_config_source = "TasksManager"
                exporter_level = SupportLevel.DEFAULT

                if hf_config is not None:
                    try:
                        io_specs = resolve_io_specs(
                            model_type=model_type,
                            task=task,
                            hf_config=hf_config,
                            model_id=model_id,
                        )
                        input_tensors, output_tensors = build_tensor_infos_from_io_specs(io_specs)
                    except Exception as e:
                        logger.debug("resolve_io_specs failed for %s/%s: %s", model_type, task, e)
        except Exception as e:
            logger.debug("TasksManager lookup failed for %s/%s: %s", model_type, task, e)

    exporter_info = ExporterInfo(
        onnx_config_class=onnx_config_class,
        onnx_config_source=onnx_config_source,
        support_level=exporter_level,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        opset_version=opset_version,
    )

    # =========================================================================
    # STEP 6: WinML class (inspect-only lookup)
    # =========================================================================
    winml_info = resolve_winml(model_type, task)

    # =========================================================================
    # STEP 7: Module hierarchy (optional, requires model_id)
    # =========================================================================
    hierarchy_info = None
    if include_hierarchy and model_id:
        try:
            from ..inspect.hierarchy import extract_hierarchy

            hierarchy_info = extract_hierarchy(model_id)
        except Exception as e:
            logger.debug("Hierarchy extraction failed for %s: %s", model_id, e)

    # =========================================================================
    # STEP 8: Overall support status
    # =========================================================================
    overall_support, support_notes = compile_support_status(loader_info, exporter_info, winml_info)

    # =========================================================================
    # STEP 9: Build config (registry lookup only, no generation)
    # =========================================================================
    build_config = registered.to_dict() if registered else None

    # =========================================================================
    # STEP 10: Inspect-only enrichment (conditional on model_id)
    # =========================================================================
    cache_info = resolve_cache(model_id) if model_id else None
    processor_info = resolve_processor(model_id, model_type=model_type) if model_id else None
    io_config_info = resolve_io_config(
        parent_hf_config,
        model_id=model_id,
        model_type=model_type,
        task=task,
    )

    # Use the top-level model_type for the user-facing result.  For multimodal
    # models (CLIP, etc.) `loader_config.model_type` is the narrowed sub-config
    # type (e.g. "clip_text_model"), but users expect the top-level type ("clip").
    #
    # Precedence:
    #   1. model_type_override  — user explicitly passed --model-type
    #   2. parent_hf_config     — pre-narrowing config (only when model_id was
    #                             provided and AutoConfig succeeded in step 1)
    #   3. model_type           — narrowed loader_config.model_type (fallback)
    display_model_type: str = (
        model_type_override or getattr(parent_hf_config, "model_type", None) or model_type
    )

    # Composite pipeline structure. resolution.composite is set only on the auto-detect
    # path (resolve_task); an explicit --task / --model-class pins composite=None, so the
    # pipeline-led composite view is shown only for auto-detected composites — by design.
    composite_info = resolve_composite_info(display_model_type, resolution.composite)

    return InspectResult(
        model_id=model_id or display_model_type or model_class_override or "unknown",
        model_type=display_model_type,
        architectures=architectures,
        task=task,
        task_source=task_source,
        loader=loader_info,
        exporter=exporter_info,
        winml=winml_info,
        overall_support=overall_support,
        support_notes=support_notes,
        build_config=build_config,
        hierarchy=hierarchy_info,
        cache=cache_info,
        processor=processor_info,
        io_config=io_config_info,
        composite=composite_info,
    )
