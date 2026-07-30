# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Pre-benchmark identity block.

Renders two bordered panels before the benchmark loop: a **Model** panel
(identity + surface: task/opset/I/O) and a **Device** panel (resolved
device + EP provenance + DLL path). Option B semantics — no arrows, no
requested-vs-resolved leak — inside the classic two-Panel shell.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Group
from rich.panel import Panel
from rich.text import Text


if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.console import Console


# Every label column is padded to LABEL_WIDTH so the values on the right
# line up in a single column. Matches the width used in the mockup at
# ``docs/design/perf/console_mockup.py::build_pre_bench_block``.
_LABEL_WIDTH = 10


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
    hardware_name: str,
    ep: str,
    ep_source: str,
    ep_version: str | None,
    ep_dll_path: str,
) -> None:
    """Print the pre-benchmark identity block.

    Layout (Option B — see the mockup design doc for the target shape):

    - Identity: ``Model:`` (bold cyan; ``(HF)`` / ``(local)`` suffix), plus
      an ``ONNX:`` line when a cached artifact path is supplied.
    - Surface: ``Task:``, ``Opset:``, ``Inputs:``, ``Outputs:`` — each
      omitted when the source field is empty / ``None``.
    - Device: ``Device:`` (resolved short name + hardware name in dim
      parens), ``EP:`` (``<short>@<source>`` + optional ``v<version>``),
      ``EP DLL:`` (full plugin path; ``(bundled with ORT)`` when the EP
      is built into ORT and has no plugin DLL).

    Args:
        console: Rich console sink (usually ``Console(stderr=True)``).
        model_id: HF model identifier when the user passed one; ``None``
            selects the raw-``.onnx`` branch.
        task: Resolved task string (e.g. ``"image-classification"``).
        opset: ONNX opset for the surface block.
        inputs / outputs: I/O spec triples. Empty / ``None`` skips the row.
        cached_onnx_path: Path of the compiled ONNX cached on disk (HF
            path only). Rendered on a dedicated ``ONNX:`` line.
        onnx_file: Raw ``.onnx`` file path when the user bypassed HF.
        device: Resolved device short name (``"npu"`` / ``"gpu"`` /
            ``"cpu"``). Never the literal ``"auto"`` — callers are
            responsible for passing the resolved value.
        hardware_name: Human-readable hardware label (e.g. ``"Intel(R) AI
            Boost"``); rendered in dim parens after the device.
        ep: Short EP alias (e.g. ``"qnn"`` / ``"openvino"``). Never
            ``"auto"``.
        ep_source: Canonical origin tag (``"bundled"`` / ``"directory"`` /
            ``"pypi"`` / etc.).
        ep_version: Per-source EP version string, or ``None`` (omits the
            ``v<version>`` chunk when absent).
        ep_dll_path: Full path to the plugin DLL. Empty string signals a
            built-in EP and renders as ``(bundled with ORT)``.
    """
    # --- Model panel: identity + surface ---------------------------------
    model_lines: list[Text] = []
    if model_id:
        model_lines.append(
            _labeled_line(
                "Model:", f"[bold cyan]{model_id}[/bold cyan]  [dim](HF)[/dim]"
            )
        )
        if cached_onnx_path:
            model_lines.append(_labeled_line("ONNX:", f"[dim]{cached_onnx_path}[/dim]"))
    elif onnx_file:
        model_lines.append(
            _labeled_line(
                "Model:", f"[bold cyan]{onnx_file}[/bold cyan]  [dim](local)[/dim]"
            )
        )

    if task:
        model_lines.append(_labeled_line("Task:", f"[cyan]{task}[/cyan]"))
    if opset is not None:
        model_lines.append(_labeled_line("Opset:", f"[green]{opset}[/green]"))
    if inputs:
        model_lines.extend(_io_lines("Inputs:", inputs))
    if outputs:
        model_lines.extend(_io_lines("Outputs:", outputs))

    if model_lines:
        console.print(Panel(Group(*model_lines), title="Model", expand=True))

    # --- Device panel: resolved device + EP + DLL -------------------------
    hw_suffix = f"  [dim]({hardware_name})[/dim]" if hardware_name else ""
    ep_line = f"[cyan]{ep}[/cyan]@[cyan]{ep_source}[/cyan]"
    if ep_version:
        ep_line += f"  [green]v{ep_version}[/green]"
    dll_display = ep_dll_path if ep_dll_path else "(bundled with ORT)"

    device_lines: list[Text] = [
        _labeled_line("Device:", f"[cyan]{device}[/cyan]{hw_suffix}"),
        _labeled_line("EP:", ep_line),
        _labeled_line("EP DLL:", f"[dim]{dll_display}[/dim]"),
    ]
    console.print(Panel(Group(*device_lines), title="Device", expand=True))


def _labeled_line(label: str, value_markup: str) -> Text:
    """Render one ``<label><padding><value>`` row as Rich markup."""
    padded = f"{label:<{_LABEL_WIDTH}}"
    return Text.from_markup(f"{padded}{value_markup}")


def _io_lines(
    label: str,
    specs: Sequence[tuple[str, str, tuple[int | str, ...]]],
) -> list[Text]:
    """Emit one line per I/O spec, name column aligned across specs.

    First spec carries the ``label`` (e.g. ``"Inputs:"``); subsequent
    specs indent to keep the value column vertically aligned.
    """
    name_width = max((len(name) for name, _, _ in specs), default=0)
    out: list[Text] = []
    for i, (name, dtype, shape) in enumerate(specs):
        prefix = (
            f"{label:<{_LABEL_WIDTH}}" if i == 0 else " " * _LABEL_WIDTH
        )
        shape_str = "[" + ", ".join(str(d) for d in shape) + "]"
        dtype_suffix = f"   [dim]{dtype}[/dim]" if dtype else ""
        out.append(
            Text.from_markup(
                f"{prefix}[cyan]{name:<{name_width}}[/cyan]   {shape_str}{dtype_suffix}"
            )
        )
    return out
