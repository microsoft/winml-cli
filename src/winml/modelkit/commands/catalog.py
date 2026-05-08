# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
r"""Catalog command for ModelKit CLI.

Lets users discover ModelKit's curated built-in model catalog.  The catalog
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

import importlib.resources
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
from ..utils.constants import normalize_ep_name


logger = logging.getLogger(__name__)
console = Console(highlight=False)

# Color palette for type-based row styling (deterministic, no hardcoded arch names)
_TYPE_PALETTE = ["cyan", "green", "yellow", "magenta", "blue", "red"]

# ---------------------------------------------------------------------------
# Catalog loader
# ---------------------------------------------------------------------------


def _load_catalog() -> dict[str, Any]:
    """Load hub_models.json from package data.

    Returns:
        Parsed catalog dict with ``version`` and ``models`` keys.
    """
    pkg = importlib.resources.files("winml.modelkit.data")
    data = (pkg / "hub_models.json").read_text(encoding="utf-8")
    return json.loads(data)  # type: ignore[no-any-return]


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


# EPs always supported by every catalog model (no supported_eps entry needed)
_ALWAYS_ON_EPS: frozenset[str] = frozenset({"DmlExecutionProvider", "CPUExecutionProvider"})

# Maps optional EP full name → catalog supported_eps key
_OPTIONAL_EP_TO_CAT_KEY: dict[str, str] = {
    "QNNExecutionProvider": "QNN EP",
    "OpenVINOExecutionProvider": "OV EP",
    "VitisAIExecutionProvider": "VitisAI EP",
}

# Maps device (uppercase) → catalog supported_eps keys for EPs that target that device.
# An empty frozenset means the device is covered by an always-on EP → all models match.
_DEVICE_TO_CAT_KEYS: dict[str, frozenset[str]] = {
    "CPU": frozenset(),  # MLAS always-on → all models
    "GPU": frozenset(),  # DML always-on → all models
    "NPU": frozenset({"QNN EP", "OV EP", "VitisAI EP"}),
}


def _filter_by_ep(
    models: list[dict[str, Any]],
    ep: str | None,
) -> list[dict[str, Any]]:
    """Filter models by --ep (execution provider).

    Args:
        models: Model list to filter.
        ep: Raw EP value from CLI (alias or full name), or ``None`` to skip.

    Returns:
        Filtered model list.
    """
    if ep is None:
        return models
    ep_full = normalize_ep_name(ep)
    if ep_full in _ALWAYS_ON_EPS:
        return models  # DML/MLAS always supported by every catalog model
    cat_key = _OPTIONAL_EP_TO_CAT_KEY.get(ep_full or "")
    if cat_key is None:
        return []  # EP not represented in catalog
    return [m for m in models if cat_key in (m.get("supported_eps") or [])]


def _filter_by_device(
    models: list[dict[str, Any]],
    device: str | None,
) -> list[dict[str, Any]]:
    """Filter models by --device (CPU / GPU / NPU).

    Args:
        models: Model list to filter.
        device: Device string from CLI (case-insensitive), or ``None`` to skip.

    Returns:
        Filtered model list.
    """
    if device is None:
        return models
    cat_keys = _DEVICE_TO_CAT_KEYS.get(device.upper(), frozenset())
    if not cat_keys:
        return models  # always-on EP covers this device (CPU → MLAS, GPU → DML)
    return [m for m in models if any(k in (m.get("supported_eps") or []) for k in cat_keys)]


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


# Maps normalized full EP name → slash-separated target devices (for --ep column)
_EP_TO_DEVICES_STR: dict[str, str] = {
    "DmlExecutionProvider": "GPU",
    "CPUExecutionProvider": "CPU",
    "QNNExecutionProvider": "GPU / NPU",
    "OpenVINOExecutionProvider": "CPU / GPU / NPU",
    "VitisAIExecutionProvider": "NPU",
}

# Maps device (uppercase) → ordered (short_label, cat_key_or_None) pairs
# used to build per-model EP strings in --device column mode.
_DEVICE_EP_LABELS: dict[str, list[tuple[str, str | None]]] = {
    "CPU": [("MLAS", None), ("OV", "OV EP")],
    "GPU": [("OV", "OV EP"), ("QNN", "QNN EP")],
    "NPU": [("OV", "OV EP"), ("QNN", "QNN EP"), ("VitisAI", "VitisAI EP")],
}


def _make_ep_col_fn_for_ep(
    ep_full: str,
) -> tuple[str, Callable[[dict[str, Any]], str]]:
    """Build the column header and per-model cell function for *--ep* mode.

    Returns a column header of ``"Devices"`` and a function that, for any
    model, returns the slash-separated devices targeted by *ep_full*.

    Args:
        ep_full: Normalised full EP name (e.g. ``"QNNExecutionProvider"``).

    Returns:
        ``("Devices", cell_fn)`` tuple.
    """
    devices_str = _EP_TO_DEVICES_STR.get(ep_full, "")

    def cell_fn(m: dict[str, Any]) -> str:
        return devices_str

    return "Devices", cell_fn


def _make_ep_col_fn_for_device(
    device: str,
) -> tuple[str, Callable[[dict[str, Any]], str]]:
    """Build the column header and per-model cell function for *--device* mode.

    Returns a column header of ``"EPs"`` and a function that, for a given
    model, returns the slash-separated EP short names that support *device*.

    Args:
        device: Device string (case-insensitive, e.g. ``"NPU"``).

    Returns:
        ``("EPs", cell_fn)`` tuple.
    """
    labels = _DEVICE_EP_LABELS.get(device.upper(), [])

    def cell_fn(m: dict[str, Any]) -> str:
        eps_present = m.get("supported_eps") or []
        active = [lbl for lbl, key in labels if key is None or key in eps_present]
        return " / ".join(active) if active else "\u2014"

    return "EPs", cell_fn


def _fmt_size(num_params: int | None) -> str:
    """Format a parameter count (in millions) as a human-readable string.

    Args:
        num_params: Parameter count in millions, or ``None`` if unknown.

    Returns:
        ``"<n>M"`` for sub-billion counts, ``"<n.1f>B"`` for billions, or
        ``"—"`` when the value is absent.
    """
    if num_params is None:
        return "\u2014"
    if num_params >= 1000:
        return f"{num_params / 1000:.1f}B"
    return f"{num_params}M"


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
            _fmt_size(m.get("num_parameters")),
            Text(m["model_type"], style=color),
        ]
        if ep_col_header is not None and ep_col_fn is not None:
            row.append(ep_col_fn(m))
        table.add_row(*row)

    panel = Panel(
        table,
        title=f"[bold]ModelKit Catalog[/bold]  [dim]|[/dim]  "
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
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Save results to a JSON file.",
)
def catalog(
    model_type: str | None,
    task: str | None,
    ep: str | None,
    device: str | None,
    output: Path | None,
) -> None:
    r"""Browse ModelKit's curated built-in model catalog.

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
