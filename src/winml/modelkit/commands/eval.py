# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Accuracy evaluation CLI command."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console

from ..utils import cli as cli_utils
from ..utils.constants import ALL_EP_NAMES, SUPPORTED_DEVICES
from ..utils.eval_utils import EVAL_MODES, TASK_SCHEMAS, EvalMode, TaskSchema
from ..utils.logging import configure_logging


if TYPE_CHECKING:
    from ..eval import EvalResult, EvalRuntime, WinMLEvaluationConfig
    from ..utils.constants import EPNameOrAlias


logger = logging.getLogger(__name__)


@click.command("eval")
@cli_utils.model_option(
    required=False,
    multiple=True,
    help_text=(
        "Model to evaluate. Accepts a HuggingFace model ID, an ONNX file path "
        "(requires --model-id), or split-encoder role=path pairs (see --schema)."
    ),
)
@cli_utils.model_id_option(
    help_text="HuggingFace model ID when .onnx model file is provided in --model.",
)
@click.option(
    "--dataset",
    "dataset_path",
    type=str,
    default=None,
    help="HF dataset path (e.g. 'imagenet-1k', 'nyu-mll/glue'). "
    "If omitted, uses a default dataset for the task.",
)
@click.option(
    "--dataset-name",
    type=str,
    default=None,
    help="Dataset config name for multi-config datasets (e.g. 'mrpc').",
)
@click.option(
    "--dataset-revision",
    "revision",
    type=str,
    default=None,
    help="Git revision (branch, tag, or commit) to load. Useful for script-based "
    "datasets that have a parquet mirror at 'refs/convert/parquet'.",
)
@click.option(
    "--task",
    type=str,
    default=None,
    help="Task (e.g. 'image-classification'). Auto-detected from --model-id.",
)
@cli_utils.device_option(
    required=False,
    default="auto",
    include_auto=True,
)
@cli_utils.ep_option(required=False)
@cli_utils.precision_option(
    optional_message="Applied during model build. Ignored for pre-built ONNX inputs "
    "(precision is already baked in).",
)
@cli_utils.quant_option(
    optional_message="Applied during model build. Ignored for pre-built ONNX inputs."
)
@cli_utils.optimize_option(
    optional_message="Applied during model build. Ignored for pre-built ONNX inputs."
)
@cli_utils.analyze_option(
    optional_message="Applied during model build. Ignored for pre-built ONNX inputs."
)
@cli_utils.max_optim_iterations_option(optional_message="Ignored for pre-built ONNX inputs.")
@cli_utils.shape_config_option(
    param_name="shape_config_path",
    help_text=(
        "JSON with shape overrides for auto-generated HuggingFace export configs. "
        "Ignored for pre-built ONNX inputs."
    ),
)
@cli_utils.input_specs_option(
    help_text=(
        "JSON file with input specifications for HuggingFace export. "
        "Ignored for pre-built ONNX inputs."
    ),
)
@cli_utils.export_config_option(
    help_text=(
        "ONNX export configuration JSON for HuggingFace model builds "
        "(opset_version, do_constant_folding, etc.). Ignored for pre-built ONNX inputs."
    ),
)
@cli_utils.dynamic_axes_option(
    help_text=(
        "JSON dynamic axes mapping for HuggingFace ONNX export "
        '(e.g., {"input_ids": {"0": "batch", "1": "sequence"}}). '
        "Ignored for pre-built ONNX inputs."
    ),
)
@click.option(
    "--runtime",
    type=click.Choice(["winml-ort", "pytorch"]),
    default="winml-ort",
    show_default=True,
    help="Evaluation runtime. 'winml-ort' exports Hugging Face checkpoints to ONNX; "
    "'pytorch' evaluates the original checkpoint.",
)
@click.option(
    "--samples",
    type=int,
    default=100,
    show_default=True,
    help="Number of dataset samples.",
)
@click.option(
    "--audio-input-length",
    type=click.IntRange(min=1),
    default=None,
    help="Fixed waveform length for audio preprocessing.",
)
@click.option(
    "--split",
    type=str,
    default="validation",
    show_default=True,
    help="Dataset split.",
)
@click.option(
    "--shuffle/--no-shuffle",
    default=True,
    show_default=True,
    help="Shuffle dataset before sampling.",
)
@click.option(
    "--streaming/--no-streaming",
    default=False,
    show_default=True,
    help="Stream dataset instead of downloading fully.",
)
@click.option(
    "--column",
    multiple=True,
    help="Column mapping as key=value (e.g. --column input_column=image).",
)
@click.option(
    "--label-mapping",
    # Distinct Python variable name so ctx.params["label_mapping_path"] does
    # not collide with ``DatasetConfig.label_mapping`` (which is the *parsed*
    # ``dict[str, int] | None``, not a Path). ``collect_cli_overrides`` is
    # name-based, so without the rename the Path would be passed to the dict
    # field with the wrong type.
    "label_mapping_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Path to a JSON file with label mapping: {"label_name": id}.',
)
@cli_utils.output_option("Output JSON file path.")
@cli_utils.overwrite_option()
@click.option(
    "--dataset-script",
    type=str,
    default=None,
    help="Path to a Python script that builds the evaluation dataset.",
)
@cli_utils.trust_remote_code_option(
    optional_message="Used for native Hugging Face loading and required with --dataset-script."
)
@cli_utils.allow_unsupported_nodes_option()
@click.option(
    "--schema",
    "show_schema",
    is_flag=True,
    default=False,
    help="Print expected dataset schema for the given --task and exit.",
)
@click.option(
    "--mode",
    type=click.Choice(EVAL_MODES, case_sensitive=False),
    default="onnx",
    show_default=True,
    help=(
        "Evaluation mode. "
        "'onnx' (default): evaluate the ONNX candidate on the dataset. "
        "'compare': compare ONNX vs HF reference output tensors on identical "
        "random inputs and report tensor-similarity metrics per output tensor."
    ),
)
@click.option(
    "--input-data",
    "input_data",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help=(
        "Path to a .npz file of real input tensors to compare with instead of "
        "randomly generated ones (use with --mode compare). Keys must match the "
        "candidate model's input names; the leading axis of each array is the "
        "sample axis (N samples), and all inputs must share the same N."
    ),
)
@click.option(
    "--reference",
    "reference",
    type=str,
    default=None,
    help=(
        "Reference ONNX file to compare the candidate against (use with "
        "--mode compare). Compares two ONNX models on identical random inputs; "
        "--model-id / --task are not required in this mode."
    ),
)
@click.option(
    "--reference-device",
    type=click.Choice(
        ["auto", *(device.lower() for device in SUPPORTED_DEVICES)],
        case_sensitive=False,
    ),
    default="cpu",
    show_default=True,
    help="Device used to run the reference ONNX model.",
)
@click.option(
    "--reference-ep",
    type=click.Choice(ALL_EP_NAMES, case_sensitive=False),
    default=None,
    help="Execution provider used to run the reference ONNX model.",
)
@cli_utils.cache_options()
@cli_utils.skip_build_option()
@cli_utils.format_option()
@cli_utils.build_config_option()
@cli_utils.verbosity_options()
@cli_utils.no_color_option()
@click.pass_context
def eval(
    ctx: click.Context,
    model: tuple[str, ...],
    model_id: str | None,
    dataset_path: str,
    dataset_name: str | None,
    revision: str | None,
    task: str | None,
    device: str,
    precision: str,
    quant: bool,
    optimize: bool,
    analyze: bool,
    max_optim_iterations: int | None,
    shape_config_path: Path | None,
    input_specs: Path | None,
    export_config: Path | None,
    dynamic_axes: Path | None,
    runtime: EvalRuntime,
    ep: EPNameOrAlias | None,
    samples: int,
    audio_input_length: int | None,
    split: str,
    shuffle: bool,
    streaming: bool,
    column: tuple[str, ...],
    label_mapping_path: Path | None,
    output: Path | None,
    overwrite: bool,
    output_format: cli_utils.OutputFormat,
    verbose: int,
    quiet: bool,
    dataset_script: str | None,
    trust_remote_code: bool,
    allow_unsupported_nodes: bool,
    show_schema: bool,
    mode: EvalMode,
    input_data: str | None,
    reference: str | None,
    reference_device: str,
    reference_ep: EPNameOrAlias | None,
    config_file: Path | None,
    use_cache: bool,
    rebuild: bool,
    skip_build: bool,
) -> None:
    r"""Evaluate a model for a task.

    Examples:
        winml eval -m microsoft/resnet-50

        winml eval -m model.onnx --model-id microsoft/resnet-50

        winml eval --mode compare -m cand.onnx --model-id microsoft/resnet-50 --input-data data.npz

        winml eval --mode compare -m cand.onnx --reference baseline.onnx

    Run `winml eval --schema --task <task>` to see the dataset columns
    and options expected by each task.
    """
    # ── 0. --schema fast path: served from a local lightweight schema table
    #       so this branch does not import the heavy winml.modelkit.eval package.
    if show_schema:
        task_arg = task
        if task_arg is None:
            task_list = "\n  ".join(sorted(TASK_SCHEMAS))
            click.echo(
                "--schema requires --task <task>.\n\n"
                f"Supported tasks:\n  {task_list}\n\n"
                "Example: winml eval --schema --task image-classification"
            )
            return
        schema = TASK_SCHEMAS.get(task_arg)
        if schema is None:
            supported = ", ".join(sorted(TASK_SCHEMAS))
            raise click.UsageError(
                f"Task '{task_arg}' is not supported by `winml eval`. Supported tasks: {supported}."
            )
        _print_schema(task_arg, schema)
        return

    verbose, quiet = cli_utils.resolve_verbosity(ctx, verbose, quiet)
    configure_logging(verbosity=verbose, quiet=quiet)

    from ..eval import evaluate

    # ── 1. Build config: defaults ← config file ← CLI ──
    cfg, config_fields = _build_eval_config(ctx, config_file, column, label_mapping_path)

    if cfg.runtime not in ("winml-ort", "pytorch"):
        raise click.UsageError(
            f"Invalid eval runtime {cfg.runtime!r}; expected 'winml-ort' or 'pytorch'."
        )
    if cfg.runtime == "pytorch":
        _validate_pytorch_runtime_options(ctx, cfg, config_fields)

    if cfg.input_data is not None and cfg.mode != "compare":
        raise click.UsageError("--input-data is only valid with --mode compare.")

    if cfg.reference_path is not None and cfg.mode != "compare":
        raise click.UsageError("--reference is only valid with --mode compare.")

    reference_environment_requested = (
        cli_utils.is_cli_provided(ctx, "reference_device")
        or cli_utils.is_cli_provided(ctx, "reference_ep")
        or "reference_device" in config_fields
        or "reference_ep" in config_fields
    )
    if cfg.reference_path is None and reference_environment_requested:
        raise click.UsageError(
            "--reference-device and --reference-ep require --reference <onnx>."
        )

    # ── 2. Resolve in place ──
    _resolve_model(cfg, model, model_id, allow_missing_model_id=cfg.reference_path is not None)
    if cfg.runtime == "pytorch" and cfg.model_path is not None:
        raise click.UsageError(
            "--runtime pytorch requires a Hugging Face model ID or local Hugging Face "
            "checkpoint; ONNX files, composite role=path models, and GenAI bundles "
            "are not supported."
        )
    if cfg.runtime == "pytorch":
        from ..eval.evaluate import _validate_pytorch_runtime_config

        try:
            _validate_pytorch_runtime_config(cfg)
        except ValueError as error:
            raise click.UsageError(str(error)) from error
    _resolve_reference(cfg)
    _apply_export_overrides(cfg, shape_config_path, input_specs, export_config, dynamic_axes)
    _resolve_device(cfg)
    _resolve_genai_ep(ctx, cfg)
    _resolve_label_mapping(cfg)
    _run_dataset_script(cfg, cfg.trust_remote_code)

    # Refuse to clobber an existing report unless the user opted in — fail fast
    # before the (expensive) evaluation runs.
    cli_utils.guard_output(cfg.output_path, overwrite)

    if cfg.model_path is not None and cfg.precision != "auto":
        logger.warning(
            "--precision %s is ignored for pre-built ONNX inputs "
            "(precision is already baked into the model).",
            cfg.precision,
        )

    # --samples sizes the random/dataset sample count; with --input-data the
    # count comes from the archive's leading axis, so an explicit --samples is
    # ignored — warn instead of silently discarding it (mirrors perf's
    # --batch-size warning).
    if cfg.input_data is not None and cli_utils.is_cli_provided(ctx, "samples"):
        logger.warning(
            "--samples is ignored when --input-data is set; the sample count "
            "comes from the leading axis of the provided tensors."
        )

    json_mode = output_format == "json"

    # ── 3. Evaluate ──
    try:
        _warn_ignored_model_build_controls(ctx, cfg, config_fields)
        logger.debug("Effective eval config: %s", cfg.to_dict())
        result = evaluate(cfg)
        _write_and_display(result, cfg.output_path, json_mode=json_mode)
    except Exception as e:
        if verbose:
            logger.exception("Evaluation failed")
        raise click.ClickException(f"Evaluation failed: {e}") from e


def _build_eval_config(
    ctx: click.Context,
    config_file: Path | None,
    column: tuple[str, ...],
    label_mapping_path: Path | None,
) -> tuple[WinMLEvaluationConfig, set[str]]:
    """Build a WinMLEvaluationConfig with precedence: defaults ← config file ← CLI.

    Reads raw JSON for config-file values so only explicitly-present keys
    are applied (avoids overriding with dataclass defaults).
    Uses ``collect_cli_overrides`` for automatic CLI-to-field mapping.

    Returns the resolved config and the field names explicitly present in the
    config file.
    """
    from ..eval import DatasetConfig, WinMLEvaluationConfig
    from ..utils.config_utils import merge_config

    # Initialize config object from CLI ctx params. ``collect_cli_overrides``
    # filters to user-provided values and applies the cli_name → field_name
    # renames declared on the dataclass fields (e.g. output → output_path).
    # The --label-mapping Click option binds to ``label_mapping_path`` (see the
    # ``@click.option`` decorator) so it does NOT collide with the
    # ``DatasetConfig.label_mapping`` field name.
    eval_kwargs = cli_utils.collect_cli_overrides(ctx, WinMLEvaluationConfig)
    dataset_kwargs = cli_utils.collect_cli_overrides(ctx, DatasetConfig)
    cfg = WinMLEvaluationConfig(dataset=DatasetConfig(**dataset_kwargs), **eval_kwargs)
    config_fields: set[str] = set()

    # ── Config file layer (only explicitly-present keys) ──
    if config_file is not None:
        _, raw = cli_utils.load_build_config(config_file)

        # Loader task as lowest-priority fallback
        loader_section = raw.get("loader") or {}
        if "task" in loader_section:
            cfg.task = loader_section["task"]

        # Compile EP as fallback for --ep
        compile_section = raw.get("compile") or {}
        if "execution_provider" in compile_section:
            cfg.ep = compile_section["execution_provider"]

        # Eval section overrides loader/compile fallbacks
        eval_data = raw.get("eval")
        if eval_data:
            config_fields.update(eval_data)
            cfg = merge_config(cfg, eval_data)

    # ── CLI layer (highest priority, auto-mapped via metadata) ──
    overrides = cli_utils.collect_cli_overrides(ctx, type(cfg))
    ds_overrides = cli_utils.collect_cli_overrides(ctx, DatasetConfig)

    # --column is multiple=True; non-empty tuple means user provided it
    if column:
        columns_mapping: dict[str, str] = {}
        for c in column:
            if "=" not in c:
                raise click.BadParameter(
                    f"Invalid column format: '{c}'. Use key=value.",
                    param_hint="--column",
                )
            k, v = c.split("=", 1)
            columns_mapping[k] = v
        ds_overrides["columns_mapping"] = columns_mapping

    if label_mapping_path is not None:
        ds_overrides["label_mapping_file"] = str(label_mapping_path)

    if ds_overrides:
        overrides["dataset"] = ds_overrides

    if overrides:
        cfg = merge_config(cfg, overrides)

    return cfg, config_fields


@dataclass(frozen=True)
class _ModelBuildBypass:
    reason: str
    build_explanation: str = "no build runs"
    cache_explanation: str = "no build runs"


def _model_build_bypass(cfg: WinMLEvaluationConfig) -> _ModelBuildBypass | None:
    """Describe a selected loader that bypasses the model-build pipeline."""
    from ..eval.evaluate import _ModelLoaderKind, _select_model_loader

    loader = _select_model_loader(cfg)
    if loader is _ModelLoaderKind.PYTORCH:
        return _ModelBuildBypass("PyTorch runtime evaluation")
    if loader is _ModelLoaderKind.GENAI:
        return _ModelBuildBypass(
            reason="GenAI bundles",
            build_explanation="no model build pipeline runs",
            cache_explanation=(
                "model build cache controls do not govern the GenAI runtime _compiled/ cache"
            ),
        )
    if loader is _ModelLoaderKind.EVALUATOR_MANAGED:
        return _ModelBuildBypass("evaluator-managed composite inputs")
    if loader is _ModelLoaderKind.ONNX and cfg.skip_build:
        return _ModelBuildBypass("pre-built ONNX inputs")
    return None


def _warn_ignored_model_build_controls(
    ctx: click.Context,
    cfg: WinMLEvaluationConfig,
    config_fields: set[str],
) -> None:
    """Warn when explicit model-build controls cannot affect the selected loader."""
    _resolve_model_loader_task(cfg)
    bypass = _model_build_bypass(cfg)
    build_runs = bypass is None
    reason = bypass.reason if bypass is not None else None

    build_flags_warning = cli_utils.ignored_build_flags_warning(
        build_runs=build_runs,
        quant=cfg.quant,
        optimize=cfg.optimize,
        analyze=cfg.analyze,
        max_optim_iterations=cfg.max_optim_iterations,
        reason=reason,
        rebuild_hint=("--no-skip-build" if reason == "pre-built ONNX inputs" else None),
        explanation=bypass.build_explanation if bypass is not None else None,
    )
    if build_flags_warning:
        logger.warning(build_flags_warning)

    cache_flags_warning = cli_utils.ignored_cache_flags_warning(
        build_runs=build_runs,
        use_cache=cfg.use_cache,
        rebuild=cfg.rebuild,
        use_cache_was_set=cli_utils.is_cli_provided(ctx, "use_cache"),
        rebuild_was_set=cli_utils.is_cli_provided(ctx, "rebuild"),
        use_cache_source=("--config" if "use_cache" in config_fields else None),
        rebuild_source=("--config" if "rebuild" in config_fields else None),
        reason=reason,
        explanation=bypass.cache_explanation if bypass is not None else None,
    )
    if cache_flags_warning:
        logger.warning(cache_flags_warning)


def _resolve_model_loader_task(cfg: WinMLEvaluationConfig) -> None:
    """Resolve an omitted task when it can change the selected model loader."""
    if cfg.task is not None or cfg.reference_path is not None:
        return
    if not isinstance(cfg.model_path, dict) and not (
        isinstance(cfg.model_path, str) and Path(cfg.model_path).is_dir()
    ):
        return

    from ..eval.evaluate import _infer_task

    cfg.task = _infer_task(cfg)


_PYTORCH_RUNTIME_INCOMPATIBLE_OPTIONS: dict[str, str] = {
    "mode": "--mode",
    "input_data": "--input-data",
    "reference": "--reference",
    "ep": "--ep",
    "precision": "--precision",
    "quant": "--quant/--no-quant",
    "optimize": "--optimize/--no-optimize",
    "analyze": "--analyze/--no-analyze",
    "max_optim_iterations": "--max-optim-iterations",
    "shape_config_path": "--shape-config",
    "input_specs": "--input-specs",
    "export_config": "--export-config",
    "dynamic_axes": "--dynamic-axes",
    "allow_unsupported_nodes": "--allow-unsupported-nodes",
    "skip_build": "--skip-build/--no-skip-build",
    "use_cache": "--use-cache/--no-use-cache",
    "rebuild": "--rebuild/--no-rebuild",
}

_PYTORCH_RUNTIME_INCOMPATIBLE_CONFIG_FIELDS = {
    "mode",
    "input_data",
    "reference_path",
    "ep",
    "precision",
    "quant",
    "optimize",
    "analyze",
    "max_optim_iterations",
    "shape_config",
    "export_overrides",
    "allow_unsupported_nodes",
    "skip_build",
    "use_cache",
    "rebuild",
}


def _validate_pytorch_runtime_options(
    ctx: click.Context,
    cfg: WinMLEvaluationConfig,
    config_fields: set[str],
) -> None:
    """Reject options whose semantics require ONNX export or ONNX Runtime."""
    if cfg.device.lower() not in ("auto", "cpu", "gpu"):
        raise click.UsageError(
            f"--device {cfg.device} is not supported with --runtime pytorch; use auto, cpu, or gpu."
        )
    incompatible = [
        flag
        for param_name, flag in _PYTORCH_RUNTIME_INCOMPATIBLE_OPTIONS.items()
        if cli_utils.is_cli_provided(ctx, param_name)
    ]
    incompatible.extend(
        f"eval.{field}"
        for field in sorted(config_fields & _PYTORCH_RUNTIME_INCOMPATIBLE_CONFIG_FIELDS)
    )
    if incompatible:
        raise click.UsageError(
            "--runtime pytorch cannot be combined with incompatible options: "
            f"{', '.join(incompatible)}."
        )


def _resolve_model(
    cfg: WinMLEvaluationConfig,
    model: tuple[str, ...],
    model_id: str | None,
    *,
    allow_missing_model_id: bool = False,
) -> None:
    """Resolve ``-m`` / ``--model-id`` into ``cfg.model_path`` / ``cfg.model_id``."""
    if not model and model_id is None and (cfg.model_path is not None or cfg.model_id is not None):
        return
    model_path, resolved_id = _resolve_model_path(
        model=model, model_id=model_id, allow_missing_model_id=allow_missing_model_id
    )
    cfg.model_path = model_path
    cfg.model_id = resolved_id


def _resolve_reference(cfg: WinMLEvaluationConfig) -> None:
    """Validate and normalize ``cfg.reference_path`` for two-ONNX compare.

    Requires the candidate (``-m``) to be a single ONNX file (composite
    ``role=path`` candidates and build-from-id are not supported with
    ``--reference`` yet). Resolves Hub-hosted ONNX refs to local paths.
    """
    if cfg.reference_path is None:
        return

    if not isinstance(cfg.model_path, str):
        raise click.UsageError(
            "--reference requires the candidate (-m) to be a single ONNX file. "
            "Composite (role=path) candidates and build-from-id are not "
            "supported with --reference."
        )

    ref = cfg.reference_path
    if Path(ref).suffix.lower() != ".onnx":
        raise click.BadParameter(
            f"--reference must be an .onnx file, got: {ref}",
            param_hint="--reference",
        )
    try:
        ref = cli_utils.normalize_model_arg(ref) or ref
    except Exception as e:
        raise click.ClickException(
            f"Failed to resolve Hub-hosted reference ONNX path {ref!r}: {e}"
        ) from e
    if not Path(ref).exists():
        raise click.BadParameter(
            f"Reference ONNX file not found: {ref}",
            param_hint="--reference",
        )
    cfg.reference_path = ref


def _apply_export_overrides(
    cfg: WinMLEvaluationConfig,
    shape_config_path: Path | None,
    input_specs: Path | None,
    export_config: Path | None,
    dynamic_axes: Path | None,
) -> None:
    """Parse the HuggingFace export CLI overrides onto *cfg* (in place).

    ``--shape-config``/``--input-specs``/``--export-config``/``--dynamic-axes``
    only affect the HF-build path (``model_path is None``), where eval exports
    and builds the model. A pre-built ONNX input is consumed as-is (no export
    step), so any export/shape overrides are dropped with a warning — mirroring
    the ``--precision``/build-flag warnings and winml perf's ONNX path.
    Requires ``cfg.model_path`` to already be resolved (call after
    :func:`_resolve_model`).
    """
    export_flags = (
        ("--shape-config", shape_config_path),
        ("--input-specs", input_specs),
        ("--export-config", export_config),
        ("--dynamic-axes", dynamic_axes),
    )
    provided = [flag for flag, value in export_flags if value is not None]
    if not provided:
        return

    if cfg.model_path is not None:
        logger.warning(
            "%s ignored for pre-built ONNX inputs "
            "(no export runs; these apply only when building from a model ID).",
            ", ".join(provided),
        )
        return

    if shape_config_path is not None:
        cfg.shape_config = cli_utils.load_json_object(shape_config_path, "--shape-config")

    export_overrides = cli_utils.load_export_overrides(
        export_config=export_config,
        input_specs=input_specs,
        dynamic_axes=dynamic_axes,
    )
    if export_overrides:
        # Shallow-merge over any config-file export_overrides so CLI-provided
        # sub-keys win while config-file sub-keys the CLI didn't set survive
        # (config-file explicit > CLI default). load_export_overrides returns a
        # sparse dict, so a wholesale assignment would drop the untouched
        # config-file keys — mirrors build's merge_export_overrides intent.
        merged = dict(cfg.export_overrides or {})
        merged.update(export_overrides)
        cfg.export_overrides = merged


def _resolve_device(cfg: WinMLEvaluationConfig) -> None:
    """Resolve ``'auto'`` → concrete device string on *cfg* in place."""
    if cfg.runtime == "pytorch":
        from ..loader import resolve_native_device

        try:
            resolved = resolve_native_device(cfg.device)
        except ValueError as error:
            raise click.UsageError(str(error)) from error
        cfg._auto_device_selected = cfg.device.lower() == "auto"
        cfg.device = resolved.name
        return

    if cfg.device and cfg.device.lower() != "auto":
        return

    cfg._auto_device_selected = True
    from ..session import EPDeviceTarget, resolve_device

    console = Console(stderr=True)
    console.print("[bold]Detecting available devices...[/bold]")
    resolved_target = resolve_device(
        EPDeviceTarget(ep=cfg.ep or "auto", device=cfg.device or "auto")
    )
    cfg.device = resolved_target.device
    console.print(f"[dim]Using device:[/dim] {resolved_target.device}")


def _resolve_genai_ep(ctx: click.Context, cfg: WinMLEvaluationConfig) -> None:
    """Turn an explicit ``--device`` into an EP override for a genai bundle.

    A genai bundle mixes stages across EPs by design (its ``genai_config.json``
    encodes the per-stage routing), so :class:`GenaiSession` only re-routes when
    an EP override is present and otherwise leaves the bundle untouched. Passing
    the device straight through would make ``--device`` a no-op: the whole point
    of an explicit device is to force the pipeline onto it. Mirroring the
    ``ort-genai`` perf precedence, an explicitly supplied ``--device`` is
    resolved to a concrete EP (via the same device→EP path the ONNX runtime
    uses), while the default (``auto``) respects the bundle's own routing.

    A user-supplied ``--ep`` already forces the pipeline, so it wins untouched.
    """
    if cfg.ep is not None:
        return
    model_path = cfg.model_path
    if not isinstance(model_path, str):
        return
    bundle = Path(model_path).expanduser()
    if not (bundle.is_dir() and (bundle / "genai_config.json").is_file()):
        return
    if ctx.get_parameter_source("device") != click.core.ParameterSource.COMMANDLINE:
        return

    from ._perf_genai import resolve_genai_ep

    cfg.ep = resolve_genai_ep(cfg.device)


def _resolve_label_mapping(cfg: WinMLEvaluationConfig) -> None:
    """Load label-mapping JSON file (if any) into ``cfg.dataset.label_mapping``."""
    if cfg.dataset.label_mapping_file:
        with Path(cfg.dataset.label_mapping_file).open() as f:
            cfg.dataset.label_mapping = json.load(f)


def _run_dataset_script(cfg: WinMLEvaluationConfig, trust_remote_code: bool) -> None:
    """Run the dataset build script referenced by *cfg*, if any.

    The script is invoked with ``--output <dataset.path>`` so the built
    dataset lands at the path already configured in the config file.
    """
    if not cfg.dataset.build_script:
        return

    if not cfg.dataset.path:
        raise click.UsageError(
            "dataset.path is required when dataset.build_script is set. "
            "The path tells the script where to write the built dataset."
        )

    if not trust_remote_code:
        raise click.UsageError("--trust-remote-code is required to execute a dataset script.")

    import subprocess
    import sys

    script_path = Path(cfg.dataset.build_script)
    if not script_path.exists():
        raise click.BadParameter(f"Dataset script not found: {script_path}")

    cmd = [sys.executable, str(script_path), "--output", str(Path(cfg.dataset.path).expanduser())]

    Console(stderr=True).print(f"[bold]Building dataset via {script_path.name}...[/bold]")
    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    if result.returncode != 0:
        raise click.ClickException(
            f"Dataset script failed (exit {result.returncode}): "
            f"{result.stderr.strip()[-200:] or '(no stderr)'}"
        )


def _write_and_display(
    result: EvalResult, output_path: Path | None, *, json_mode: bool = False
) -> None:
    """Display evaluation results and optionally save to JSON."""
    if json_mode:
        click.echo(json.dumps(result.to_dict(), indent=2, default=_json_default))
    else:
        console = Console()
        display_eval_report(result, console)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(result.to_dict(), f, indent=2, default=_json_default)
        Console(stderr=True).print(f"[green]Results saved to:[/green] {output_path}")


def _resolve_model_path(
    *,
    model: tuple[str, ...],
    model_id: str | None,
    allow_missing_model_id: bool = False,
) -> tuple[str | dict[str, str] | None, str | None]:
    """Turn repeated -m values + --model-id into (model_path, model_id).

    When ``allow_missing_model_id`` is set (two-ONNX ``--mode compare``), a
    plain ``-m <file>.onnx`` is accepted without ``--model-id`` because the
    candidate uses the generic WinML model without HF config resolution.
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
        # Each role's path may be either a local .onnx file OR a Hub-hosted
        # ONNX ref (``org/repo/path/file.onnx``). ``normalize_model_arg``
        # resolves Hub refs to local cached paths so downstream code sees
        # only filesystem paths.
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
            try:
                path = cli_utils.normalize_model_arg(path) or path
            except Exception as e:
                raise click.ClickException(
                    f"Failed to resolve Hub-hosted ONNX path {path!r}: {e}"
                ) from e
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
    if Path(value).suffix.lower() == ".onnx":
        # Hub-hosted ONNX (e.g. ``onnx-community/sam3-tracker-ONNX/onnx/...``)
        # is downloaded once and treated as a local .onnx path thereafter.
        try:
            value = cli_utils.normalize_model_arg(value) or value
        except Exception as e:
            raise click.ClickException(
                f"Failed to resolve Hub-hosted ONNX path {value!r}: {e}"
            ) from e
        if not Path(value).exists():
            raise click.BadParameter(
                f"ONNX file not found: {value}",
                param_hint="-m/--model",
            )
        if model_id is None:
            if allow_missing_model_id:
                return value, None
            raise click.UsageError(
                "When using an ONNX file, --model-id is required "
                "for preprocessor and config resolution."
            )
        return value, model_id

    # An onnxruntime-genai bundle is a local *directory* (holding
    # ``genai_config.json``), not a Hub model id. Route it to model_path so the
    # genai loader reads the bundle from disk. Gate on the genai marker so a
    # plain local HF checkpoint directory still flows through the model_id path
    # (``from_pretrained``) as before.
    expanded = Path(value).expanduser()
    if expanded.is_dir() and (expanded / "genai_config.json").is_file():
        return str(expanded), model_id

    if model_id is not None and model_id != value:
        raise click.UsageError(
            "Cannot pass both `-m <hf_id>` and `--model-id`. "
            "Use `--model-id` only together with an ONNX file path in `-m`."
        )
    return None, model_id or value


def _json_default(obj: object) -> object:
    """Handle numpy types for JSON serialization."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def display_eval_report(result: EvalResult, console: Console) -> None:
    """Display evaluation results in formatted console output."""
    from rich.panel import Panel
    from rich.table import Table

    cfg = result.config
    ds = cfg.dataset
    metrics = result.metrics
    # For --input-data compare the effective sample count comes from the
    # archive (via EvalResult.num_samples), not the unused config default.
    samples = result.num_samples if result.num_samples is not None else ds.samples

    # Header — model_id when building from HF, otherwise the ONNX path(s). A
    # composite model_path is a {role: path} dict; join its paths so the title
    # stays a readable string instead of a raw dict repr.
    if cfg.model_id:
        eval_name = cfg.model_id
    elif isinstance(cfg.model_path, dict):
        eval_name = ", ".join(str(path) for path in cfg.model_path.values())
    else:
        eval_name = str(cfg.model_path)

    console.print()
    console.print(
        Panel.fit(
            f"[bold]Evaluation: {eval_name}[/bold]",
            border_style="blue",
        )
    )

    # Info section
    console.print()
    console.print(f"[dim]Task:[/dim]       {cfg.task}")
    console.print(f"[dim]Runtime:[/dim]    {cfg.runtime}")
    console.print(f"[dim]Device:[/dim]     {cfg.device}")
    if cfg.input_data:
        console.print(f"[dim]Input data:[/dim] {cfg.input_data}")
    elif ds.path:
        console.print(f"[dim]Dataset:[/dim]    {ds.path}")
    console.print(f"[dim]Samples:[/dim]    {samples}")
    if isinstance(cfg.model_path, dict):
        for role, path in cfg.model_path.items():
            console.print(f"[dim]ONNX ({role}):[/dim] {path}")
    elif cfg.model_path:
        console.print(f"[dim]ONNX:[/dim]       {cfg.model_path}")
    if cfg.reference_path:
        console.print(f"[dim]Reference:[/dim]  {cfg.reference_path}")
        console.print(f"[dim]Reference device:[/dim] {cfg.reference_device}")
        if cfg.reference_ep:
            console.print(f"[dim]Reference EP:[/dim] {cfg.reference_ep}")

    # Metrics table
    console.print()
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    for key, value in metrics.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        elif isinstance(value, dict):
            parts = []
            for k, v in value.items():
                parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
            table.add_row(key, "  ".join(parts))
        else:
            table.add_row(key, str(value))

    console.print(table)
    console.print()


def _print_schema(task: str, schema: TaskSchema) -> None:
    """Render the human-readable input schema for *task*."""
    width = 50
    title = f"Input schema for {task} models"
    click.echo(title)
    click.echo("=" * width)
    click.echo()

    click.echo("--column option schema")
    click.echo()
    click.echo("Evaluating needs a dataset with the following columns:")
    for item in schema.columns:
        click.echo(f"  {item.name}")
        click.echo(f"      {item.description} (default: {item.default})")

    if schema.params:
        click.echo()
        click.echo("Additional configuration parameters:")
        for p in schema.params:
            click.echo(f"  {p.name}")
            click.echo(f"      {p.description} (default: {p.default})")

    overrides = [c for c in (*schema.columns, *schema.params) if c.remap_hint]
    if overrides:
        click.echo()
        click.echo("Override any default with --column:")
        for c in overrides:
            click.echo(f"  --column {c.name}={c.remap_hint}")

    if schema.roles:
        click.echo()
        click.echo("-" * width)
        click.echo("-m option schema")
        click.echo()
        click.echo("Use one of the following model input forms:")
        click.echo("  1. use huggingface id: -m <hf-id>")
        model_args = " ".join(f"-m {r}=<{r}.onnx>" for r in schema.roles)
        click.echo(f"  2. use onnx file: {model_args} --model-id <hf-id>")
