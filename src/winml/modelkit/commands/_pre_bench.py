# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Pre-benchmark identity block.

Renders a 3-sub-block intro before the benchmark loop: model identity,
surface (placeholder), and resolved device. Mirrors the mockup helper
in ``docs/design/perf/console_mockup.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table


if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.console import Console


def print_pre_bench_block(
    console: Console,
    *,
    model_id: str | None,
    task: str | None,
    opset: int | None,
    inputs: Sequence[tuple[str, str, tuple[int | str, ...]]] | None,
    outputs: Sequence[tuple[str, str, tuple[int | str, ...]]] | None,
    cached_onnx_path: str | None,
    onnx_file: str | None,
    device: str,
    ep: str,
) -> None:
    """Print the 3-sub-block pre-benchmark identity panel.

    For HF inputs (``model_id`` set), shows the full identity card.
    For raw ONNX-file inputs, shows just the file path.
    Surface sub-block is reserved for forward-looking use (no content yet).
    """
    # 1. Model identity
    if model_id:
        ident = Table.grid(padding=(0, 2))
        ident.add_column(justify="right", style="dim")
        ident.add_column()
        ident.add_row("Model:", model_id)
        if task:
            ident.add_row("Task:", task)
        if opset is not None:
            ident.add_row("Opset:", str(opset))
        if inputs:
            ident.add_row("Inputs:", _fmt_io(inputs))
        if outputs:
            ident.add_row("Outputs:", _fmt_io(outputs))
        if cached_onnx_path:
            ident.add_row("Cached ONNX:", cached_onnx_path)
        console.print(Panel(ident, title="Model", expand=True))
    elif onnx_file:
        ident = Table.grid(padding=(0, 2))
        ident.add_column(justify="right", style="dim")
        ident.add_column()
        ident.add_row("ONNX file:", onnx_file)
        console.print(Panel(ident, title="Model", expand=True))

    # 2. Surface (placeholder; forward-looking — no content emitted)

    # 3. Device
    dev = Table.grid(padding=(0, 2))
    dev.add_column(justify="right", style="dim")
    dev.add_column()
    dev.add_row("Device:", device)
    dev.add_row("EP:", ep)
    console.print(Panel(dev, title="Device", expand=True))


def _fmt_io(specs: Sequence[tuple[str, str, tuple[int | str, ...]]]) -> str:
    parts: list[str] = []
    for name, dtype, shape in specs:
        shape_str = "(" + ", ".join(str(d) for d in shape) + ")"
        parts.append(f"{name} ({dtype}, {shape_str})")
    return ", ".join(parts)
