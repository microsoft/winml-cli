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
    default=None,
    help="Path to .onnx model file or HuggingFace model ID.",
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
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON file path.",
)
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
@click.option(
    "--trust-remote-code",
    is_flag=True,
    default=False,
    help="Allow execution of dataset scripts. Required when --dataset-script is used.",
)
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
    model: str | None,
    model_id: str | None,
    dataset_path: str,
    dataset_name: str | None,
    task: str | None,
    device: str,
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

    from ..eval import WinMLEvaluationConfig, evaluate

    # ── 1. Build config: CLI > config file eval > quant/loader > defaults ──
    cfg = WinMLEvaluationConfig()
    if config_file is not None:
        cfg = _merge_from_config_file(cfg, config_file)
    cfg = _merge_cli_overrides(cfg, ctx, column, label_mapping)

    if show_schema:
        from ..eval import WinMLEvaluator
        from ..eval.evaluate import _EVALUATOR_REGISTRY

        if cfg.task is None:
            raise click.UsageError(
                "--schema requires --task. Example: winml eval --schema --task object-detection"
            )
        cls = _EVALUATOR_REGISTRY.get(cfg.task, WinMLEvaluator)
        _print_schema(cfg.task, cls.schema_info())
        return

    # ── 2. Resolve in place ──
    _resolve_model(cfg, model)
    _resolve_device(cfg)
    _resolve_label_mapping(cfg)
    _run_dataset_script(cfg, trust_remote_code)

    logger.debug("Effective eval config: %s", cfg.to_dict())

    # ── 3. Evaluate ──
    try:
        result = evaluate(cfg)
        _write_and_display(result, cfg.output_path)
    except Exception as e:
        logger.exception("Evaluation failed")
        raise click.ClickException(f"Evaluation failed: {e}") from e


# ── CLI-to-config field mappings ──
# Adding a new option only requires one entry here; both config-file and CLI
# paths pick it up automatically via merge_config.
_CLI_TO_CONFIG: dict[str, str] = {
    "model_id": "model_id",
    "task": "task",
    "device": "device",
}

_CLI_TO_DATASET: dict[str, str] = {
    "dataset_path": "path",
    "dataset_name": "name",
    "split": "split",
    "samples": "samples",
    "shuffle": "shuffle",
    "streaming": "streaming",
    "dataset_script": "build_script",
}


def _merge_from_config_file(
    cfg: object,
    config_file: Path,
) -> object:
    """Merge eval-relevant values from a build config file into *cfg*.

    Reads the raw JSON so that only fields explicitly present in the file
    are applied (avoids overriding with dataclass defaults).

    Precedence (lowest → highest): quant/loader fields → eval section.
    """
    from ..utils.config_utils import merge_config

    raw = json.loads(config_file.read_text())

    # Loader task as lowest-priority fallback
    loader_data = raw.get("loader", {})
    if "task" in loader_data:
        cfg.task = loader_data["task"]

    # Quant fields as fallback (only explicitly-set values)
    quant_data = raw.get("quant", {})
    quant_overrides: dict = {}
    if "samples" in quant_data:
        quant_overrides.setdefault("dataset", {})["samples"] = quant_data["samples"]
    if "dataset_name" in quant_data:
        quant_overrides.setdefault("dataset", {})["name"] = quant_data["dataset_name"]
    if quant_overrides:
        cfg = merge_config(cfg, quant_overrides)

    # Eval section overrides quant/loader
    eval_data = raw.get("eval")
    if eval_data:
        cfg = merge_config(cfg, eval_data)

    return cfg


def _merge_cli_overrides(
    cfg: object,
    ctx: click.Context,
    column: tuple[str, ...],
    label_mapping: Path | None,
) -> object:
    """Overlay CLI-provided values onto *cfg* (highest priority).

    Only values the user explicitly passed on the command line are applied.
    Full precedence chain (lowest → highest):
        dataclass defaults → quant/loader → config file eval → CLI
    """
    from ..utils.config_utils import merge_config

    overrides: dict = {}
    ds_overrides: dict = {}

    for cli_name, cfg_name in _CLI_TO_CONFIG.items():
        if cli_utils.is_cli_provided(ctx, cli_name):
            overrides[cfg_name] = ctx.params[cli_name]

    for cli_name, cfg_name in _CLI_TO_DATASET.items():
        if cli_utils.is_cli_provided(ctx, cli_name):
            ds_overrides[cfg_name] = ctx.params[cli_name]

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

    if cli_utils.is_cli_provided(ctx, "output"):
        overrides["output_path"] = ctx.params["output"]

    if ds_overrides:
        overrides["dataset"] = ds_overrides

    if overrides:
        cfg = merge_config(cfg, overrides)

    return cfg


def _resolve_model(cfg: object, model_arg: str | None) -> None:
    """Resolve ``-m`` into ``model_id`` / ``model_path`` on *cfg* in place."""
    if model_arg is not None:
        p = Path(model_arg)
        if p.suffix.lower() == ".onnx":
            if not p.exists():
                raise click.BadParameter(
                    f"ONNX file not found: {model_arg}",
                    param_hint="-m/--model",
                )
            cfg.model_path = model_arg
            if cfg.model_id is None:
                raise click.UsageError(
                    "When using an ONNX file, --model-id is required "
                    "for preprocessor and config resolution."
                )
        else:
            cfg.model_id = cfg.model_id or model_arg

    if cfg.model_id is None and cfg.model_path is None:
        raise click.UsageError(
            "A model is required. Provide -m with a HuggingFace model ID or path to an .onnx file."
        )


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
        raise click.UsageError(
            "--trust-remote-code is required to execute a dataset script."
        )

    import subprocess
    import sys

    script_path = Path(cfg.dataset.build_script)
    if not script_path.exists():
        raise click.BadParameter(f"Dataset script not found: {script_path}")

    cmd = [sys.executable, str(script_path),
           "--output", str(Path(cfg.dataset.path).expanduser())]

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
