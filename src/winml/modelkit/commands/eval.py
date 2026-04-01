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
    type=click.Choice(["cpu", "gpu", "npu"], case_sensitive=False),
    default="cpu",
    show_default=True,
    help="Device to run on.",
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
    "--schema",
    "show_schema",
    is_flag=True,
    default=False,
    help="Print expected dataset schema for the given --task and exit.",
)
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
    show_schema: bool,
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

    if show_schema:
        from ..eval.base_evaluator import WinMLEvaluator
        from ..eval.evaluate import _EVALUATOR_REGISTRY

        if task is None:
            raise click.UsageError(
                "--schema requires --task. Example: winml eval --schema --task object-detection"
            )
        cls = _EVALUATOR_REGISTRY.get(task, WinMLEvaluator)
        _print_schema(task, cls.schema_info())
        return

    if model is None and model_id is None:
        raise click.UsageError(
            "A model is required. Provide -m with a HuggingFace model ID or path to an .onnx file."
        )

    # Detect: -m as HF model ID (not an ONNX file) → treat as model_id
    model_path = None
    if model is not None:
        p = Path(model)
        if p.suffix.lower() == ".onnx":
            if not p.exists():
                raise click.BadParameter(
                    f"ONNX file not found: {model}",
                    param_hint="-m/--model",
                )
            model_path = model
            if model_id is None:
                raise click.UsageError(
                    "When using an ONNX file, --model-id is required "
                    "for preprocessor and config resolution."
                )
        else:
            model_id = model_id or model

    # Parse column mappings from --column key=value pairs
    columns_mapping: dict[str, str] = {}
    for c in column:
        if "=" not in c:
            raise click.BadParameter(
                f"Invalid column format: '{c}'. Use key=value.",
                param_hint="--column",
            )
        k, v = c.split("=", 1)
        columns_mapping[k] = v

    # Parse label mapping from JSON file
    parsed_label_mapping = None
    if label_mapping:
        with Path(label_mapping).open() as f:
            parsed_label_mapping = json.load(f)

    from ..datasets.config import DatasetConfig
    from ..eval import WinMLEvaluationConfig, evaluate

    ds_config = DatasetConfig(
        path=dataset_path,
        name=dataset_name,
        split=split,
        samples=samples,
        shuffle=shuffle,
        columns_mapping=columns_mapping,
        label_mapping=parsed_label_mapping,
        streaming=streaming,
    )

    config = WinMLEvaluationConfig(
        model_path=model_path,
        model_id=model_id,
        task=task,
        device=device.lower(),
        dataset=ds_config,
        output_path=output,
    )

    try:
        result = evaluate(config)
        logger.info("Evaluation results: %s", result.to_dict())

        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w") as f:
                json.dump(result.to_dict(), f, indent=2, default=_json_default)
            logger.info("Results saved to: %s", output)

    except Exception as e:
        logger.exception("Evaluation failed")
        raise click.ClickException(f"Evaluation failed: {e}") from e


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
