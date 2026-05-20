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


logger = logging.getLogger(__name__)
console = Console()


def _looks_like_local_path(model_id: str) -> bool:
    """Return True when model_id is explicitly a local path."""
    from pathlib import Path

    _p = Path(model_id).expanduser()
    return _p.exists() or _p.is_absolute() or "\\" in model_id or model_id.startswith((".", "~"))


@click.command("inspect")
@click.option(
    "-m",
    "--model",
    "model_id",
    required=False,
    default=None,
    help="HuggingFace model ID (e.g., microsoft/resnet-50)",
)
@click.option(
    "-f",
    "--format",
    "output_format",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    help="Output format (default: table)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Show full configuration details",
)
@click.option(
    "-t",
    "--task",
    default=None,
    help="Override auto-detected task (e.g., image-classification, feature-extraction)",
)
@click.option(
    "-H",
    "--hierarchy",
    is_flag=True,
    default=False,
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
@click.pass_context
def inspect(
    ctx: click.Context,
    model_id: str | None,
    output_format: str,
    verbose: bool,
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
    # Handle --list-tasks (no model required)
    if list_tasks:
        from ..inspect.resolver import get_known_tasks

        for t in sorted(get_known_tasks()):
            click.echo(t)
        return

    # Validate: need at least one of model_id, model_type, model_class
    if model_id is None and model_type is None and model_class is None:
        raise click.UsageError(
            "At least one of -m/--model, --model-type, or --model-class is required. "
            "Use --list-tasks to see available tasks."
        )

    # Classify the input before hitting HF Hub: explicitly-local paths must exist.
    # Keep this conservative to avoid misclassifying valid HF IDs as local paths.
    if model_id:
        from pathlib import Path

        _p = Path(model_id).expanduser()
        _is_local = _looks_like_local_path(model_id)
        if _is_local:
            if _p.suffix == ".onnx" and _p.is_file():
                raise click.ClickException(
                    "ONNX file inspection is not yet supported. "
                    "Use 'winml config -m model.onnx' for ONNX build config."
                )
            if not _p.exists():
                raise click.ClickException(f"Local path '{model_id}' does not exist.")

    from ..inspect import InspectError, ModelNotFoundError, NetworkError
    from ..inspect.formatter import output_json, output_table

    # Inherit debug mode from parent context
    if ctx.obj and ctx.obj.get("debug"):
        verbose = True

    # Configure logging based on verbosity
    if verbose:
        logging.getLogger("winml.modelkit").setLevel(logging.DEBUG)

    try:
        result = _inspect_model_v2(
            model_id=model_id,
            task_override=task,
            model_type_override=model_type,
            model_class_override=model_class,
            include_hierarchy=hierarchy,
        )

        if output_format.lower() == "json":
            click.echo(output_json(result, verbose=verbose))
        else:
            output_table(console, result, verbose=verbose)

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
    # STEP 1: Preserve parent hf_config before resolve_loader_config narrows it
    #         for multimodal models (e.g., CLIPConfig → CLIPTextConfig)
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
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        loader_config, hf_config, _resolved_class = resolve_loader_config(
            model_id,
            task=task_override,
            model_type=model_type_override,
            model_class=model_class_override,
        )
    except RepositoryNotFoundError as e:
        raise ModelNotFoundError(f"Model '{model_id}' not found on Hugging Face Hub.") from e
    except ValueError as e:
        err_str = str(e).lower()
        if "not found" in err_str or "404" in err_str:
            raise ModelNotFoundError(str(e)) from e
        raise InspectError(str(e)) from e
    except OSError as e:
        raise NetworkError(str(e)) from e

    if parent_hf_config is None:
        parent_hf_config = hf_config

    model_type = loader_config.model_type
    task = loader_config.task
    architectures = getattr(parent_hf_config, "architectures", []) or []

    # =========================================================================
    # STEP 3: Derive task_source by checking registries post-hoc
    # =========================================================================
    mt = model_type.lower().replace("_", "-")
    task_source = "TasksManager"
    for m, t in HF_MODEL_CLASS_MAPPING:
        if m == mt and t == task:
            task_source = "HF_MODEL_CLASS_MAPPING"
            break

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
            for s in export_cfg.input_tensors
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

            onnx_config_cls = TasksManager.get_exporter_config_constructor(
                exporter="onnx",
                model_type=model_type,
                task=task,
                library_name="transformers",
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

    return InspectResult(
        model_id=model_id or model_type or model_class_override or "unknown",
        model_type=model_type,
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
    )
