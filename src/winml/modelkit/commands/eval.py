# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Accuracy evaluation CLI command."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from ..utils import cli as cli_utils


logger = logging.getLogger(__name__)


@click.command("eval")
@click.option(
    "-m",
    "--model",
    type=str,
    multiple=True,
    default=(),
    help=(
        "Model to evaluate. Accepts three forms: "
        "(1) HuggingFace model ID, e.g. `-m <hf_model_id>`. "
        "(2) ONNX file path, e.g. `-m model.onnx` (requires --model-id). "
        "(3) Composite / split-encoder model as repeated role=path pairs, "
        "e.g. `-m image-encoder=vision.onnx -m text-encoder=text.onnx`."
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
@cli_utils.build_config_option
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
    ep: str | None,
    samples: int,
    split: str,
    shuffle: bool,
    streaming: bool,
    column: tuple[str, ...],
    label_mapping: Path | None,
    output: Path | None,
    verbose: bool,
    dataset_script: str | None,
    trust_remote_code: bool,
    show_schema: bool,
    config_file: Path | None,
) -> None:
    r"""Evaluate model accuracy on a dataset.

    If --dataset is not provided, a default dataset is used based on the task.

    \b
    Examples:
        # Use default dataset (auto-detected from task)
        winml eval -m microsoft/resnet-50
        winml eval -m model.onnx --model-id dslim/bert-base-NER

        # Specify dataset explicitly
        winml eval -m microsoft/resnet-50 --dataset imagenet-1k
        winml eval -m model.onnx --model-id microsoft/resnet-50 --dataset imagenet-1k

        # Multi-config dataset with column overrides
        winml eval -m model.onnx --model-id Intel/bert-base-uncased-mrpc \\
            --dataset glue --dataset-name mrpc \\
            --column input_column=sentence1
    """
    if verbose or (ctx.obj and ctx.obj.get("debug")):
        logging.getLogger("winml.modelkit").setLevel(logging.DEBUG)

    from ..eval import evaluate

    # ── 1. Build config: defaults ← config file ← CLI ──
    cfg = _build_eval_config(ctx, config_file, column, label_mapping)

    if show_schema:
        from ..eval.evaluate import get_evaluator_class

        if cfg.task is None:
            raise click.UsageError(
                "--schema requires --task. Example: winml eval --schema --task object-detection"
            )
        try:
            cls = get_evaluator_class(cfg.task)
        except ValueError as e:
            raise click.UsageError(str(e)) from e
        _print_schema(cfg.task, cls.schema_info())
        return

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
        logger.exception("Evaluation failed")
        raise click.ClickException(f"Evaluation failed: {e}") from e


def _build_eval_config(
    ctx: click.Context,
    config_file: Path | None,
    column: tuple[str, ...],
    label_mapping: Path | None,
) -> object:
    """Build a WinMLEvaluationConfig with precedence: defaults ← config file ← CLI.

    Reads raw JSON for config-file values so only explicitly-present keys
    are applied (avoids overriding with dataclass defaults).
    Uses ``collect_cli_overrides`` for automatic CLI-to-field mapping.
    """
    from ..datasets import DatasetConfig
    from ..eval import WinMLEvaluationConfig
    from ..utils.config_utils import merge_config

    # Initialize config object from CLI ctx params.
    p = ctx.params
    cfg = WinMLEvaluationConfig(
        task=p.get("task"),
        device=p.get("device"),
        precision=p.get("precision"),
        ep=p.get("ep"),
        output_path=p.get("output"),
        dataset=DatasetConfig(
            path=p.get("dataset_path"),
            name=p.get("dataset_name"),
            split=p.get("split"),
            samples=p.get("samples"),
            shuffle=p.get("shuffle"),
            streaming=p.get("streaming"),
            build_script=p.get("dataset_script"),
        ),
    )

    # ── Config file layer (only explicitly-present keys) ──
    if config_file is not None:
        build_cfg = cli_utils.load_build_config(config_file)

        # Loader task as lowest-priority fallback
        if build_cfg.loader and build_cfg.loader.task:
            cfg.task = build_cfg.loader.task

        # Compile EP as fallback for --ep
        if build_cfg.compile and build_cfg.compile.ep_config:
            cfg.ep = build_cfg.compile.ep_config.provider

        # Quant fields as fallback
        if build_cfg.quant:
            quant_overrides: dict = {}
            if build_cfg.quant.samples != 100:  # non-default
                quant_overrides.setdefault("dataset", {})["samples"] = build_cfg.quant.samples
            if build_cfg.quant.dataset_name:
                quant_overrides.setdefault("dataset", {})["name"] = build_cfg.quant.dataset_name
            if quant_overrides:
                cfg = merge_config(cfg, quant_overrides)

        # Eval section overrides quant/loader (read raw JSON for this)
        try:
            raw = json.loads(config_file.read_text())
        except (json.JSONDecodeError, OSError):
            raw = {}
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

    if label_mapping is not None:
        ds_overrides["label_mapping_file"] = str(label_mapping)

    if ds_overrides:
        overrides["dataset"] = ds_overrides

    if overrides:
        cfg = merge_config(cfg, overrides)

    return cfg


def _resolve_model(
    cfg: object,
    model: tuple[str, ...],
    model_id: str | None,
) -> None:
    """Resolve ``-m`` / ``--model-id`` into ``cfg.model_path`` / ``cfg.model_id``."""
    model_path, resolved_id = _resolve_model_path(model=model, model_id=model_id)
    cfg.model_path = model_path
    cfg.model_id = resolved_id


def _resolve_device(cfg: object) -> None:
    """Resolve ``'auto'`` → concrete device string on *cfg* in place."""
    from ..sysinfo import resolve_device

    resolved, _ = resolve_device(cfg.device)
    cfg.device = resolved


def _resolve_label_mapping(cfg: object) -> None:
    """Load label-mapping JSON file (if any) into ``cfg.dataset.label_mapping``."""
    if cfg.dataset.label_mapping_file:
        with Path(cfg.dataset.label_mapping_file).open() as f:
            cfg.dataset.label_mapping = json.load(f)


def _run_dataset_script(cfg: object, trust_remote_code: bool) -> None:
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

    logger.info("Building dataset via %s ...", script_path.name)
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


def _write_and_display(result: object, output_path: Path | None) -> None:
    """Display evaluation results and optionally save to JSON."""
    from rich.console import Console

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


def display_eval_report(result: object, console: object) -> None:
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


def _print_schema(task: str, schema: list, indent: int = 0) -> None:
    """Format and print structured schema info."""
    prefix = "  " * indent
    if indent == 0:
        click.echo(f"Dataset schema ({task}):\n")
        click.echo(f"{prefix}{'Column':<20} {'Type':<25} {'Override (--column)'}")
        click.echo(f"{prefix}{'-' * 20} {'-' * 25} {'-' * 25}")

    for col in schema:
        marker = "*" if col.required else " "
        override_str = f"--column {col.override}={col.name}" if col.override else ""
        click.echo(f"{prefix}{marker} {col.name:<18} {col.type:<25} {override_str}")
        if col.description:
            click.echo(f"{prefix}  {' ' * 18} {col.description}")
        for child in col.children:
            co = f"--column {child.override}={child.name}" if child.override else ""
            click.echo(f"{prefix}    .{child.name:<16} {child.type:<25} {co}")
            if child.description:
                click.echo(f"{prefix}    {' ' * 18} {child.description}")

    click.echo(f"\n{prefix}* = required")
