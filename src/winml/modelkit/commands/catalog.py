# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
r"""Catalog command for WinML CLI.

Lets users discover WinML CLI's curated built-in model catalog.  The catalog
is stored in ``modelkit/data/hub_models.json`` and lists specific, validated
HuggingFace model IDs with their task, architecture, and supported EPs.

Usage:
    winml catalog
    winml catalog --model-type bert
    winml catalog --task text-classification
    winml catalog --ep qnn
    winml catalog --device NPU
    winml catalog --output catalog.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable

import click
import rich.box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..utils import cli as cli_utils
from ..utils.constants import EPNameOrAlias, normalize_ep_name


logger = logging.getLogger(__name__)
console = Console(highlight=False)

# Color palette for type-based row styling (deterministic, no hardcoded arch names)
_TYPE_PALETTE = ["cyan", "green", "yellow", "magenta", "blue", "red"]

# ---------------------------------------------------------------------------
# Catalog loader
# ---------------------------------------------------------------------------


def _load_catalog() -> dict[str, Any]:
    """Load hub_models.json from package data.

    Resolves the path via ``__file__`` rather than ``importlib.resources.files``
    so that reading the catalog does not execute ``winml.modelkit.data``'s
    package init — that init pulls in heavy optional dependencies (onnx, numpy,
    torchvision via the dataset modules) that the catalog command never needs.

    Returns:
        Parsed catalog dict with ``version`` and ``models`` keys.
    """
    data_path = Path(__file__).resolve().parent.parent / "data" / "hub_models.json"
    return json.loads(data_path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _filter_models(
    models: list[dict[str, Any]],
    model_type: str | None,
    task: str | None,
) -> list[dict[str, Any]]:
    """Apply --model-type and --task filters.

    Args:
        models: Full model list from the catalog.
        model_type: Optional architecture filter (case-insensitive).
        task: Optional task filter (case-insensitive).

    Returns:
        Filtered model list.
    """
    result = models
    if model_type:
        result = [m for m in result if m["model_type"].lower() == model_type.lower()]
    if task:
        result = [m for m in result if m["task"].lower() == task.lower()]
    return result


def _filter_by_ep(
    models: list[dict[str, Any]],
    ep: EPNameOrAlias | None,
) -> list[dict[str, Any]]:
    """Filter models by --ep (execution provider).

    ``supported_eps`` is a ``{ep_name: [device, ...]}`` dict.

    Args:
        models: Model list to filter.
        ep: Raw EP value from CLI (alias or full name), or ``None`` to skip.

    Returns:
        Filtered model list.
    """
    if ep is None:
        return models
    ep_full = normalize_ep_name(ep)
    return [
        m
        for m in models
        if ep_full in {normalize_ep_name(k) for k in (m.get("supported_eps") or {})}
    ]


def _filter_by_device(
    models: list[dict[str, Any]],
    device: str | None,
) -> list[dict[str, Any]]:
    """Filter models by --device (CPU / GPU / NPU).

    ``supported_eps`` is a ``{ep_name: [device, ...]}`` dict.

    Args:
        models: Model list to filter.
        device: Device string from CLI (case-insensitive), or ``None`` to skip.

    Returns:
        Filtered model list.
    """
    if device is None:
        return models
    device_upper = device.upper()
    if device_upper not in {"CPU", "GPU", "NPU"}:
        return []
    return [
        m
        for m in models
        if any(device_upper in devs for devs in (m.get("supported_eps") or {}).values())
    ]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _fmt_model_id(model_id: str) -> Text:
    """Render model ID with org prefix and model name in uniform cyan bold.

    Args:
        model_id: HuggingFace model ID, e.g. ``openai/clip-vit-base-patch32``.

    Returns:
        Rich Text with the full model ID in cyan bold.
    """
    t = Text(overflow="crop", no_wrap=True)
    t.append(model_id, style="cyan bold")
    return t


def _type_color(model_type: str) -> str:
    """Return a deterministic palette color for *model_type*.

    Uses a character-sum hash so the same architecture always gets the same
    color without any hardcoded mapping.

    Args:
        model_type: Architecture string, e.g. ``"bert"`` or ``"vit"``.

    Returns:
        A Rich color name from :data:`_TYPE_PALETTE`.
    """
    idx = sum(ord(c) for c in model_type) % len(_TYPE_PALETTE)
    return _TYPE_PALETTE[idx]


def _make_ep_col_fn_for_ep(
    ep_full: str,
) -> tuple[str, Callable[[dict[str, Any]], str]]:
    """Build the column header and per-model cell function for *--ep* mode.

    Returns a column header of ``"Devices"`` and a function that returns the
    slash-separated devices for *ep_full* read from the model's
    ``supported_eps`` dict (``{ep_name: [device, ...]}``).

    Args:
        ep_full: Normalised full EP name (e.g. ``"QNNExecutionProvider"``).

    Returns:
        ``("Devices", cell_fn)`` tuple.
    """

    def cell_fn(m: dict[str, Any]) -> str:
        supported = m.get("supported_eps") or {}
        for k, devices in supported.items():
            if normalize_ep_name(k) == ep_full:
                return " / ".join(sorted(devices)) if devices else "\u2014"
        return "\u2014"

    return "Devices", cell_fn


def _make_ep_col_fn_for_device(
    device: str,
) -> tuple[str, Callable[[dict[str, Any]], str]]:
    """Build the column header and per-model cell function for *--device* mode.

    Returns a column header of ``"EPs"`` and a function that, for a given
    model, returns the slash-separated EP alias labels (uppercased) whose
    ``supported_eps`` entry includes *device*.

    Args:
        device: Device string (case-insensitive, e.g. ``"NPU"``).

    Returns:
        ``("EPs", cell_fn)`` tuple.
    """
    device_upper = device.upper()

    def cell_fn(m: dict[str, Any]) -> str:
        supported = m.get("supported_eps") or {}
        present = [k for k, devs in supported.items() if device_upper in devs]
        return " / ".join(k.upper() for k in present) if present else "\u2014"

    return "EPs", cell_fn


def _fmt_size(size_mb: float | None) -> str:
    """Format a model size in MB as a human-readable string.

    Args:
        size_mb: Model size in megabytes, or ``None`` if unknown.

    Returns:
        ``"<n>MB"`` for sub-gigabyte sizes, ``"<n.1f>GB"`` for gigabytes, or
        ``"—"`` when the value is absent.
    """
    if size_mb is None:
        return "\u2014"
    if size_mb >= 1024:
        return f"{size_mb / 1024:.1f}GB"
    return f"{size_mb:.0f}MB"


# ---------------------------------------------------------------------------
# List view
# ---------------------------------------------------------------------------


def _build_list_renderable(
    models: list[dict[str, Any]],
    ep_col_header: str | None = None,
    ep_col_fn: Callable[[dict[str, Any]], str] | None = None,
) -> Group:
    """Build the Rich renderable for the catalog list.

    A Panel wraps the data table (Model / Task / Size / Model Type, and
    optionally a fifth column when *ep_col_header* is given).

    Args:
        models: Filtered model list to display.
        ep_col_header: Header for the optional fifth column, or ``None`` to
            omit it.  Use ``"Devices"`` for *--ep* mode, ``"EPs"`` for
            *--device* mode.
        ep_col_fn: Called with each model dict to produce the fifth column's
            cell value.  Required when *ep_col_header* is not ``None``.

    Returns:
        A :class:`rich.console.Group` containing the Panel.
    """
    table = Table(
        box=rich.box.SQUARE,
        show_header=True,
        header_style="bold",
        padding=(0, 2),
        show_edge=False,
        expand=True,
    )
    table.add_column("Model", no_wrap=True, ratio=3, overflow="crop")
    table.add_column("Task", no_wrap=True, ratio=2, overflow="crop")
    table.add_column("Size", no_wrap=True, justify="right", width=5)
    table.add_column("Model Type", no_wrap=True, ratio=2, overflow="crop")
    if ep_col_header is not None:
        table.add_column(ep_col_header, no_wrap=True, ratio=2, overflow="crop")

    for m in models:
        color = _type_color(m["model_type"])
        row: list[Any] = [
            _fmt_model_id(m["model_id"]),
            m["task"],
            _fmt_size(m.get("size_mb")),
            Text(m["model_type"], style=color),
        ]
        if ep_col_header is not None and ep_col_fn is not None:
            row.append(ep_col_fn(m))
        table.add_row(*row)

    panel = Panel(
        table,
        title=f"[bold]WinML CLI Catalog[/bold]  [dim]|[/dim]  "
        f"[bold cyan]{len(models)}[/bold cyan] validated model(s)",
        border_style="blue",
        padding=(0, 1),
    )
    return Group(panel)


def _output_list(
    models: list[dict[str, Any]],
    ep_col_header: str | None = None,
    ep_col_fn: Callable[[dict[str, Any]], str] | None = None,
) -> None:
    """Render models as a compact list table.

    Args:
        models: Filtered model list to display.
        ep_col_header: Optional fifth column header (see
            :func:`_build_list_renderable`).
        ep_col_fn: Optional cell function for the fifth column.
    """
    if not models:
        console.print("[yellow]No models match the given filters.[/yellow]")
        return

    console.print(_build_list_renderable(models, ep_col_header=ep_col_header, ep_col_fn=ep_col_fn))
    console.print(
        "  [dim]Verified picks from Hugging Face. Try any model \u2014 run "
        "[/dim][bold cyan]winml inspect <model-id>[/bold cyan]"
        "[dim] to check compatibility first.[/dim]\n"
    )


# ---------------------------------------------------------------------------
# Detail view
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# JSON file output
# ---------------------------------------------------------------------------


def _save_json(data: Any, path: Path) -> None:
    """Write *data* as indented JSON to *path*, creating parent dirs as needed.

    Args:
        data: JSON-serialisable object.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    console.print(f"[green]Results saved to:[/green] {path}")


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--model-type",
    "-t",
    default=None,
    metavar="TYPE",
    help="Filter by model architecture (e.g. bert, roberta, vit).",
)
@click.option(
    "--task",
    "-k",
    default=None,
    metavar="TASK",
    help="Filter by HuggingFace task (e.g. text-classification, image-segmentation).",
)
@cli_utils.ep_option(
    required=False,
    optional_message="If not specified, shows all EPs",
)
@cli_utils.device_option(
    required=False,
    default=None,
    optional_message="If not specified, shows all devices",
)
@cli_utils.output_option("Save results to a JSON file.")
def catalog(
    model_type: str | None,
    task: str | None,
    ep: EPNameOrAlias | None,
    device: str | None,
    output: Path | None,
) -> None:
    r"""Browse WinML CLI's curated built-in model catalog.

    Lists HuggingFace models that have been validated end-to-end
    (export -> quantise -> run on device) with confirmed accuracy results.
    Use ``--output`` to save results to a JSON file.

    \b
    Examples:
        winml catalog
        winml catalog --model-type bert
        winml catalog --task text-classification
        winml catalog --ep qnn
        winml catalog --device NPU
        winml catalog --output results/catalog.json
    """
    try:
        data = _load_catalog()
    except Exception as e:
        raise click.ClickException(f"Failed to load model catalog: {e}") from e

    models = _filter_models(data["models"], model_type=model_type, task=task)
    models = _filter_by_ep(models, ep)
    models = _filter_by_device(models, device)

    # Show the extra column only when exactly one of --ep / --device is given.
    ep_col_header = None
    ep_col_fn = None
    if (ep is not None) ^ (device is not None):
        if ep is not None:
            ep_col_header, ep_col_fn = _make_ep_col_fn_for_ep(normalize_ep_name(ep) or "")
        else:
            ep_col_header, ep_col_fn = _make_ep_col_fn_for_device(device or "")
    _output_list(models, ep_col_header=ep_col_header, ep_col_fn=ep_col_fn)

    if output is not None:
        _save_json(models, output)
