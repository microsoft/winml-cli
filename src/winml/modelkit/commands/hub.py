# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
r"""Hub command for ModelKit CLI.

Lets users discover ModelKit's curated built-in model catalog.  The catalog
is stored in ``modelkit/data/hub_models.json`` and lists specific, validated
HuggingFace model IDs with their task, architecture, and benchmark results.

Accuracy fields explain *how much* quantization changed a model's metric:

* ``verdict`` -- "PASS" if drop is within tolerance, "AT_RISK" if it is
  borderline, "REGRESSION" if it exceeds the threshold.
* ``drop_pct`` -- relative change vs FP32 baseline expressed as a percentage.
  Negative means the quantized model scored lower.

Usage:
    winml hub
    winml hub --model-type bert
    winml hub --task text-classification
    winml hub --model ProsusAI/finbert
    winml hub --output catalog.json
"""

from __future__ import annotations

import importlib.resources
import json
import logging
from pathlib import Path
from typing import Any

import click
import rich.box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


logger = logging.getLogger(__name__)
console = Console(highlight=False)

# Verdict display config (ASCII-safe icons)
_VERDICT_ICON = {"PASS": "[+]", "AT_RISK": "[~]", "REGRESSION": "[!]"}
_VERDICT_STYLE = {"PASS": "bold green", "AT_RISK": "bold yellow", "REGRESSION": "bold red"}

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


def _find_model(models: list[dict[str, Any]], model_id: str) -> dict[str, Any] | None:
    """Look up a model by ID (exact first, then case-insensitive substring).

    Args:
        models: Full model list from the catalog.
        model_id: The model ID to search for.

    Returns:
        The matching model dict, or None if not found / ambiguous.
    """
    lower = model_id.lower()
    for m in models:
        if m["model_id"].lower() == lower:
            return m
    matches = [m for m in models if lower in m["model_id"].lower()]
    return matches[0] if len(matches) == 1 else None


# ---------------------------------------------------------------------------
# Rendering helpers (used by detail view and JSON formatting)
# ---------------------------------------------------------------------------


def _fmt_model_id(model_id: str) -> Text:
    """Render model ID with the org prefix dimmed.

    Args:
        model_id: HuggingFace model ID, e.g. ``openai/clip-vit-base-patch32``.

    Returns:
        Rich Text with org in dim and model name in cyan bold.
    """
    t = Text(overflow="crop", no_wrap=True)
    if "/" in model_id:
        org, name = model_id.split("/", 1)
        t.append(org + "/", style="dim")
        t.append(name, style="cyan bold")
    else:
        t.append(model_id, style="cyan bold")
    return t


def _overall_verdict(accuracy: dict[str, Any]) -> str:
    """Derive the display-level overall verdict from per-EP entries.

    Priority: REGRESSION > AT_RISK > PASS.

    Args:
        accuracy: Per-EP accuracy dict from the catalog entry.

    Returns:
        One of ``"PASS"``, ``"AT_RISK"``, or ``"REGRESSION"``.
    """
    verdicts = {info.get("verdict", "PASS") for info in accuracy.values()}
    if "REGRESSION" in verdicts:
        return "REGRESSION"
    if "AT_RISK" in verdicts:
        return "AT_RISK"
    return "PASS"


# ---------------------------------------------------------------------------
# List view
# ---------------------------------------------------------------------------


def _build_list_renderable(models: list[dict[str, Any]]) -> Group:
    """Build the Rich renderable for the catalog list.

    A Panel wraps the three-column data table (Model / Task / Type) and a
    hint line is printed below the Panel border.

    Args:
        models: Filtered model list to display.

    Returns:
        A :class:`rich.console.Group` with the Panel and the hint line.
    """
    table = Table(
        box=rich.box.SQUARE,
        show_header=True,
        header_style="bold",
        padding=(0, 2),
        show_edge=False,
        expand=True,
    )
    table.add_column("Model", no_wrap=True, max_width=38, overflow="crop")
    table.add_column("Task", no_wrap=True, overflow="crop")
    table.add_column("Model Type", no_wrap=True, style="magenta", overflow="crop")

    for m in models:
        table.add_row(
            _fmt_model_id(m["model_id"]),
            m["task"],
            m["model_type"],
        )

    panel = Panel(
        table,
        title=f"[bold]ModelKit Catalog[/bold]  [dim]|[/dim]  "
        f"[bold cyan]{len(models)}[/bold cyan] validated model(s)",
        border_style="blue",
        padding=(0, 1),
    )
    hint = Text(
        "Use  winml hub --model <id>  to see perf and accuracy details.",
        style="dim",
    )
    return Group(panel, hint)


def _output_list(models: list[dict[str, Any]]) -> None:
    """Render models as a compact 3-column list table.

    Args:
        models: Filtered model list to display.
    """
    if not models:
        console.print("[yellow]No models match the given filters.[/yellow]")
        return

    console.print(_build_list_renderable(models))


# ---------------------------------------------------------------------------
# Detail view
# ---------------------------------------------------------------------------


def _build_detail_renderable(m: dict[str, Any]) -> Group:
    """Build the detail view as three separate Panels.

    * Panel 1 -- General Information (Task, Type)
    * Panel 2 -- Latency (ms): one row per EP, columns Avg/P50/.../Max/QPS
    * Panel 3 -- Accuracy: one row per EP, columns Verdict/vs FP32

    When no benchmark data is available a single panel with a notice is shown.

    Args:
        m: A single model dict from the catalog.

    Returns:
        A :class:`rich.console.Group` of Panels suitable for ``console.print``.
    """
    _table_kwargs: dict[str, Any] = {
        "box": rich.box.SQUARE,
        "show_header": True,
        "header_style": "bold",
        "padding": (0, 1),
        "show_edge": False,
    }

    # -- Panel 1: General Information ----------------------------------------
    info = Table(box=None, show_header=False, padding=(0, 2), show_edge=False)
    info.add_column(style="dim", no_wrap=True)
    info.add_column(no_wrap=True)
    info.add_row("Task", m["task"])
    info.add_row("Type", f"[magenta]{m['model_type']}[/magenta]")

    panels: list[Any] = [
        Panel(
            info,
            title=_fmt_model_id(m["model_id"]),
            border_style="blue",
            padding=(0, 1),
        )
    ]

    perf = m.get("perf")
    accuracy = m.get("accuracy")

    # -- Panel 2: Latency (ms) -----------------------------------------------
    if perf:
        lat = Table(**_table_kwargs)
        lat.add_column("EP", no_wrap=True)
        for col in ["Avg", "P50", "P90", "P95", "P99", "Min", "Max"]:
            lat.add_column(col, justify="right", no_wrap=True)
        lat.add_column("QPS", justify="right", no_wrap=True)

        for ep, s in perf.items():
            lat.add_row(
                ep,
                f"[bold]{s['avg_ms']:.2f}[/bold]",
                f"{s.get('p50_ms', 0.0):.2f}",
                f"{s.get('p90_ms', 0.0):.2f}",
                f"{s.get('p95_ms', 0.0):.2f}",
                f"{s.get('p99_ms', 0.0):.2f}",
                f"{s.get('min_ms', 0.0):.2f}",
                f"{s.get('max_ms', 0.0):.2f}",
                f"{s.get('throughput_qps', 0.0):.0f}",
            )

        panels.append(
            Panel(lat, title="[bold]Latency (ms)[/bold]", border_style="blue", padding=(0, 1))
        )

    # -- Panel 3: Accuracy ---------------------------------------------------
    if accuracy:
        overall = _overall_verdict(accuracy)
        acc_panel_title = Text.assemble(
            ("Accuracy  ", "bold"),
            (
                f"{_VERDICT_ICON.get(overall, '')} {overall}",
                _VERDICT_STYLE.get(overall, ""),
            ),
        )

        acc = Table(**_table_kwargs, expand=True)
        acc.add_column("EP", no_wrap=True)
        acc.add_column("Verdict", no_wrap=True)
        acc.add_column("vs FP32", justify="right", no_wrap=True)

        for ep, ep_info in accuracy.items():
            verdict = ep_info.get("verdict", "PASS")
            drop = ep_info.get("drop_pct", 0.0)
            sign = "+" if drop > 0 else ""
            ep_icon = _VERDICT_ICON.get(verdict, "")
            ep_style = _VERDICT_STYLE.get(verdict, "")
            acc.add_row(
                ep,
                Text(f"{ep_icon} {verdict}", style=ep_style),
                Text(f"{sign}{drop:.2f}%", style=ep_style),
            )

        panels.append(Panel(acc, title=acc_panel_title, border_style="blue", padding=(0, 1)))

    if not perf and not accuracy:
        panels.append(
            Panel(
                Text("No benchmark data available.", style="dim"),
                title="[bold]Benchmark[/bold]",
                border_style="blue",
                padding=(0, 1),
            )
        )

    return Group(*panels)


def _print_detail(m: dict[str, Any]) -> None:
    """Render and print the detail Group of Panels for *m*.

    Args:
        m: A single model dict from the catalog.
    """
    console.print(_build_detail_renderable(m))
    console.print()


def _output_detail(models: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    """Find and render detailed perf/accuracy info for a single model.

    Args:
        models: Full model list from the catalog.
        model_id: The model ID string from --model.

    Returns:
        The matched model dict (for JSON saving).

    Raises:
        click.ClickException: When the model is not found or ambiguous.
    """
    m = _find_model(models, model_id)
    if m is None:
        lower = model_id.lower()
        candidates = [x["model_id"] for x in models if lower in x["model_id"].lower()]
        if candidates:
            msg = f"Ambiguous model ID '{model_id}'. Did you mean one of:\n"
            msg += "\n".join(f"  {c}" for c in candidates)
        else:
            msg = (
                f"Model '{model_id}' not found in the catalog. Run 'winml hub' to list all models."
            )
        raise click.ClickException(msg)

    _print_detail(m)
    return m


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
@click.option(
    "--model",
    "-m",
    default=None,
    metavar="MODEL_ID",
    help="Show perf and accuracy details for a specific model.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Save results to a JSON file.",
)
def hub(
    model_type: str | None,
    task: str | None,
    model: str | None,
    output: Path | None,
) -> None:
    r"""Browse ModelKit's curated built-in model catalog.

    Lists HuggingFace models that have been validated end-to-end
    (export -> quantise -> run on device) with confirmed accuracy results.
    Use ``--output`` to save results to a JSON file.

    \b
    Accuracy legend:
      [+] PASS       -- drop within tolerance
      [~] AT_RISK    -- borderline drop, use with care
      [!] REGRESSION -- accuracy degraded beyond threshold
      drop %         -- relative change vs FP32 baseline

    \b
    Use ``winml hub --model <model_id>`` for per-model perf and accuracy.
    Use ``winml inspect -m <model_id>`` for architecture details.

    \b
    Examples:
        winml hub
        winml hub --model-type bert
        winml hub --task text-classification
        winml hub --model ProsusAI/finbert
        winml hub --output results/catalog.json
    """
    try:
        catalog = _load_catalog()
    except Exception as e:
        raise click.ClickException(f"Failed to load model catalog: {e}") from e

    all_models = catalog["models"]

    if model:
        m = _output_detail(all_models, model)
        if output is not None:
            _save_json(m, output)
        return

    models = _filter_models(all_models, model_type=model_type, task=task)
    _output_list(models)

    if output is not None:
        _save_json(models, output)
