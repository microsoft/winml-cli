"""Inspect command for ModelKit CLI.

Displays detailed information about a HuggingFace model's compatibility
with ModelKit, including loader, exporter, and WinML configurations.

Usage:
    wmk inspect -m openai/clip-vit-base-patch32
    wmk inspect -m google-bert/bert-base-uncased --format json
    wmk inspect -m facebook/detr-resnet-50 --verbose
    wmk inspect -m openai/clip-vit-base-patch32 --hierarchy
"""

from __future__ import annotations

import logging

import click
from rich.console import Console


logger = logging.getLogger(__name__)
console = Console()


@click.command()
@click.option(
    "-m",
    "--model",
    required=True,
    help="HuggingFace model ID (e.g., openai/clip-vit-base-patch32)",
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
@click.pass_context
def inspect(
    ctx: click.Context,
    model: str,
    output_format: str,
    verbose: bool,
    task: str | None,
    hierarchy: bool,
) -> None:
    r"""Inspect a HuggingFace model's ModelKit configuration.

    Shows the loader configuration, exporter configuration, and WinML
    inference class that will be used for the specified model.

    This command helps you understand:
    - Which HuggingFace model class will be used for loading
    - What ONNX export configuration will be applied
    - Which WinML inference class will handle the model
    - Overall support status in ModelKit

    \b
    Examples:
        # Basic inspection
        wmk inspect -m openai/clip-vit-base-patch32

        # JSON output for scripting
        wmk inspect -m google-bert/bert-base-uncased --format json

        # Show full build configuration
        wmk inspect -m facebook/detr-resnet-50 --verbose

        # Include HF module hierarchy (no weight download)
        wmk inspect -m openai/clip-vit-base-patch32 --hierarchy

        # Combined verbose + hierarchy
        wmk inspect -m google-bert/bert-base-uncased -v -H
    """
    # Import here to defer heavy transformers/torch imports
    from ..inspect import (
        InspectError,
        ModelNotFoundError,
        NetworkError,
        inspect_model,
    )
    from ..inspect.formatter import output_json, output_table

    # Inherit debug mode from parent
    if ctx.obj and ctx.obj.get("debug"):
        verbose = True

    # Configure logging based on verbosity
    if verbose:
        logging.getLogger("winml.modelkit").setLevel(logging.DEBUG)
    else:
        logging.getLogger("winml.modelkit").setLevel(logging.WARNING)
        logging.getLogger().setLevel(logging.WARNING)

    try:
        result = inspect_model(model, include_hierarchy=hierarchy, task_override=task)

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
        logger.exception("Failed to inspect model: %s", model)
        raise click.ClickException(f"Failed to inspect model: {e}") from e
