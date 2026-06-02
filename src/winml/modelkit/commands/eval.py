# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Accuracy evaluation CLI command."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.console import Console

from ..utils import cli as cli_utils
from ..utils.eval_utils import TASK_SCHEMAS, TaskSchema


if TYPE_CHECKING:
    from ..eval import EvalResult, WinMLEvaluationConfig
    from ..utils.constants import EPNameOrAlias


logger = logging.getLogger(__name__)


@click.command("eval")
@click.option(
    "-m",
    "--model",
    type=str,
    multiple=True,
    default=(),
    help=(
        "Model to evaluate. Accepts a HuggingFace model ID, an ONNX file path "
        "(requires --model-id), or split-encoder role=path pairs (see --schema)."
    ),
)
@click.option(
    "--model-id",
    type=str,
    default=None,
    help="HuggingFace model ID when .onnx model file is provided in --model.",
)
@click.option(
    "--dataset",
    "dataset_path",
    type=str,
    default=None,
    help="HF dataset path (e.g. 'imagenet-1k', 'glue'). "
    "If omitted, uses a default dataset for the task.",
)
@click.option(
    "--dataset-name",
    type=str,
    default=None,
    help="Dataset config name for multi-config datasets (e.g. 'mrpc').",
)
@click.option(
    "--task",
    type=str,
    default=None,
    help="Task (e.g. 'image-classification'). Auto-detected from --model-id.",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "gpu", "npu"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Device to run on. 'auto' detects the best available device.",
)
@cli_utils.ep_option(required=False)
@click.option(
    "--precision",
    type=str,
    default="auto",
    show_default=True,
    help="Precision: auto, fp32, fp16, int8, int16, or w{x}a{y} (e.g., w8a16). "
    "Applied during model build (fp16/fp32 skip quantization). "
    "Ignored for pre-built ONNX inputs (precision is already baked in).",
)
@click.option(
    "--samples",
    type=int,
    default=100,
    show_default=True,
    help="Number of dataset samples.",
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
    "--streaming",
    is_flag=True,
    default=False,
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
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose output.",
)
@click.option(
    "--dataset-script",
    type=str,
    default=None,
    help="Path to a Python script that builds the evaluation dataset.",
)
@cli_utils.trust_remote_code_option(optional_message="Required when --dataset-script is used.")
@click.option(
    "--schema",
    "show_schema",
    is_flag=True,
    default=False,
    help="Print expected dataset schema for the given --task and exit.",
)
@cli_utils.build_config_option()
@click.pass_context
def eval(
    ctx: click.Context,
    model: tuple[str, ...],
    model_id: str | None,
    dataset_path: str,
    dataset_name: str | None,
    task: str | None,
    device: str,
    precision: str,
    ep: EPNameOrAlias | None,
    samples: int,
    split: str,
    shuffle: bool,
    streaming: bool,
    column: tuple[str, ...],
    label_mapping_path: Path | None,
    output: Path | None,
    verbose: bool,
    dataset_script: str | None,
    trust_remote_code: bool,
    show_schema: bool,
    config_file: Path | None,
) -> None:
    r"""Evaluate a model for a task.

    Examples:
        winml eval -m microsoft/resnet-50

        winml eval -m model.onnx --model-id microsoft/resnet-50

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

    if verbose or (ctx.obj and ctx.obj.get("debug")):
        logging.getLogger("winml.modelkit").setLevel(logging.DEBUG)

    from ..eval import evaluate

    # ── 1. Build config: defaults ← config file ← CLI ──
    cfg = _build_eval_config(ctx, config_file, column, label_mapping_path)

    # ── 2. Resolve in place ──
    _resolve_model(cfg, model, model_id)
    _resolve_device(cfg)
    _resolve_label_mapping(cfg)
    _run_dataset_script(cfg, trust_remote_code)

    if cfg.model_path is not None and cfg.precision != "auto":
        logger.warning(
            "--precision %s is ignored for pre-built ONNX inputs "
            "(precision is already baked into the model).",
            cfg.precision,
        )

    logger.debug("Effective eval config: %s", cfg.to_dict())

    # ── 3. Evaluate ──
    try:
        result = evaluate(cfg)
        _write_and_display(result, cfg.output_path)
    except Exception as e:
        if verbose:
            logger.exception("Evaluation failed")
        raise click.ClickException(f"Evaluation failed: {e}") from e


def _build_eval_config(
    ctx: click.Context,
    config_file: Path | None,
    column: tuple[str, ...],
    label_mapping_path: Path | None,
) -> WinMLEvaluationConfig:
    """Build a WinMLEvaluationConfig with precedence: defaults ← config file ← CLI.

    Reads raw JSON for config-file values so only explicitly-present keys
    are applied (avoids overriding with dataclass defaults).
    Uses ``collect_cli_overrides`` for automatic CLI-to-field mapping.
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

    return cfg


def _resolve_model(
    cfg: WinMLEvaluationConfig,
    model: tuple[str, ...],
    model_id: str | None,
) -> None:
    """Resolve ``-m`` / ``--model-id`` into ``cfg.model_path`` / ``cfg.model_id``."""
    model_path, resolved_id = _resolve_model_path(model=model, model_id=model_id)
    cfg.model_path = model_path
    cfg.model_id = resolved_id


def _resolve_device(cfg: WinMLEvaluationConfig) -> None:
    """Resolve ``'auto'`` → concrete device string on *cfg* in place."""
    if cfg.device and cfg.device.lower() != "auto":
        return

    from ..sysinfo import resolve_device

    console = Console()
    console.print("[bold]Detecting available devices...[/bold]")
    resolved, _ = resolve_device(cfg.device)
    cfg.device = resolved
    console.print(f"[dim]Using device:[/dim] {resolved}")


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

    Console().print(f"[bold]Building dataset via {script_path.name}...[/bold]")
    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise click.ClickException(
            f"Dataset script failed (exit {result.returncode}): "
            f"{result.stderr.strip()[-200:] or '(no stderr)'}"
        )


def _write_and_display(result: EvalResult, output_path: Path | None) -> None:
    """Display evaluation results and optionally save to JSON."""
    console = Console()
    display_eval_report(result, console)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(result.to_dict(), f, indent=2, default=_json_default)
        console.print(f"[green]Results saved to:[/green] {output_path}")


def _resolve_model_path(
    *,
    model: tuple[str, ...],
    model_id: str | None,
) -> tuple[str | dict[str, str] | None, str | None]:
    """Turn repeated -m values + --model-id into (model_path, model_id)."""
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
    if Path(value).suffix.lower() == ".onnx":
        if not Path(value).exists():
            raise click.BadParameter(
                f"ONNX file not found: {value}",
                param_hint="-m/--model",
            )
        if model_id is None:
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

    # Header
    console.print()
    console.print(
        Panel.fit(
            f"[bold]Evaluation: {cfg.model_id}[/bold]",
            border_style="blue",
        )
    )

    # Info section
    console.print()
    console.print(f"[dim]Task:[/dim]       {cfg.task}")
    console.print(f"[dim]Device:[/dim]     {cfg.device}")
    console.print(f"[dim]Dataset:[/dim]    {ds.path}")
    console.print(f"[dim]Samples:[/dim]    {ds.samples}")
    if cfg.model_path:
        console.print(f"[dim]ONNX:[/dim]       {cfg.model_path}")

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
