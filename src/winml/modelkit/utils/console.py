# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Shared console output utilities for WinML CLI commands.

Provides consistent Rich-based formatting for:
- Config command: headers, I/O specs, resolution summary
- Build command: cascading StageLive, setup/stages sections, graph summary

All output goes to stderr via Console(stderr=True) so stdout stays clean
for machine-readable output (JSON configs, build manifests).
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.text import Text


F = TypeVar("F", bound=Callable[..., Any])


if TYPE_CHECKING:
    from ..export.config import WinMLExportConfig
    from .constants import EPName

logger = logging.getLogger(__name__)

HEAVY_SEP = "\u2550" * 60  # ═
LIGHT_SEP = "\u2500" * 60  # ─
MAX_BAR_WIDTH = 36

# Stage status icons
ICON_RUNNING = "\u23f3"  # ⏳
ICON_DONE = "\u2705"  # ✅
ICON_SKIP = "\u23f8\ufe0f "  # ⏸️
ICON_ERROR = "\u274c"  # ❌


def get_console() -> Console:
    """Return a Console that prints to stderr."""
    return Console(stderr=True)


# ══════════════════════════════════════════════════════════════════════════
# SHARED FORMATTING
# ══════════════════════════════════════════════════════════════════════════


def print_command_header(
    console: Console,
    title: str,
    subtitle: str | None = None,
) -> None:
    """Print a command header block (═══ separators)."""
    console.print()
    console.print(HEAVY_SEP)
    label = f"[bold]{title}[/bold]"
    if subtitle:
        label += f"  [dim]({subtitle})[/dim]"
    console.print(label)
    console.print(HEAVY_SEP)


def print_kv(
    console: Console,
    label: str,
    value: str,
    *,
    note: str | None = None,
    icon: str = "",
) -> None:
    """Print a key-value line with optional note."""
    line = f"   {icon} [bold]{label:<14}[/bold] [cyan]{value}[/cyan]"
    if note:
        line += f"  [dim]({note})[/dim]"
    console.print(line)


def print_success(console: Console, message: str) -> None:
    """Print a green success line with check icon."""
    console.print(f"   [green]{ICON_DONE} {message}[/green]")


def print_error(
    console: Console,
    message: str,
    hint: str | None = None,
) -> None:
    """Print a red error line with optional hint."""
    console.print(f"   [red]{ICON_ERROR} {message}[/red]")
    if hint:
        console.print(f"   [dim]\U0001f4a1 {hint}[/dim]")


# ══════════════════════════════════════════════════════════════════════════
# CONFIG COMMAND HELPERS
# ══════════════════════════════════════════════════════════════════════════


def print_io_specs_detail(
    console: Console,
    export_config: WinMLExportConfig,
) -> None:
    """Print resolved I/O specs — always full detail, aligned columns."""
    inputs = export_config.input_tensors or []
    outputs = export_config.output_tensors or []

    for i, t in enumerate(inputs):
        name = t.name or "(unnamed)"
        shape_str = str(list(t.shape)) if t.shape else "dynamic"
        dtype_str = getattr(t, "dtype", None) or "?"
        label = "Input:        " if i == 0 else "              "
        console.print(f"   {label}[cyan]{name:<18}[/cyan] {shape_str:<14} [dim]{dtype_str}[/dim]")
    for i, out_t in enumerate(outputs):
        name = out_t.name or "(unnamed)"
        # Fix #3: OutputTensorSpec only has name — show name only
        label = "Output:       " if i == 0 else "              "
        console.print(f"   {label}[cyan]{name}[/cyan]")


def print_io_specs_na(console: Console, reason: str = "") -> None:
    """Print I/O specs not-available line (e.g., ONNX mode)."""
    msg = reason or "inferred from ONNX graph at build time"
    console.print(f"   \U0001f4d0 [bold]I/O specs:[/bold]    [dim]N/A \u2014 {msg}[/dim]")


# ══════════════════════════════════════════════════════════════════════════
# BUILD COMMAND — SETUP / STAGES SECTIONS
# ══════════════════════════════════════════════════════════════════════════


def print_setup(
    console: Console,
    *,
    model: str,
    config: str,
    output: str,
    source: str = "HuggingFace",
    auto: bool = True,
) -> None:
    """Print the 🔧 Setup section header."""
    console.print()
    console.print(HEAVY_SEP)
    console.print(f"[bold]\U0001f527 Setup \u2014 {source}[/bold]")
    console.print(HEAVY_SEP)
    console.print(f"   \U0001f4e6 [bold]{'Model:':<10}[/bold] [cyan]{model}[/cyan]")
    config_suffix = "  [dim](autoconf off)[/dim]" if not auto else "  [dim](autoconf on)[/dim]"
    console.print(
        f"   \U0001f4c1 [bold]{'Config:':<10}[/bold] [cyan]{config}[/cyan]{config_suffix}"
    )
    console.print(f"   \U0001f4c2 [bold]{'Output:':<10}[/bold] [cyan]{output}[/cyan]")
    console.print()


def print_stages_header(console: Console) -> None:
    """Print the 🎯 Stages section header."""
    console.print(HEAVY_SEP)
    console.print("[bold]\U0001f3af Stages[/bold]")
    console.print(HEAVY_SEP)


def print_final(
    console: Console,
    elapsed: float,
    artifact: str,
    stage_timings: list[tuple[str, float | None]] | None = None,
    config: str | None = None,
) -> None:
    """Print the 📊 Summary section with stage timing breakdown.

    Args:
        stage_timings: list of (stage_name, elapsed_seconds | None for skipped)
        config: path to the saved build config JSON, printed after the artifact
    """
    console.print()
    console.print(HEAVY_SEP)
    console.print("[bold]\U0001f4ca Summary[/bold]")
    console.print(HEAVY_SEP)
    console.print(f"{ICON_DONE} [bold green]Build complete in {elapsed:.1f}s[/bold green]")
    if stage_timings:
        for name, t in stage_timings:
            if t is not None:
                console.print(f"   {name:<12} [green]{t:.1f}s[/green]")
            else:
                console.print(f"   {name:<12} [dim]skipped[/dim]")
    console.print(f"\U0001f4e6 Final artifact: [bold]{artifact}[/bold]")
    if config is not None:
        console.print(f"\U0001f4c4 Build config:   [bold]{config}[/bold]")
    console.print()


def print_stage_skip(
    console: Console,
    name: str,
    reason: str = "",
) -> None:
    """Print a skipped stage as static text (no Live needed)."""
    line = Text()
    line.append(f"{ICON_SKIP} ")
    line.append(name.capitalize(), style="dim")
    if reason:
        line.append(f"  {reason}", style="dim italic")
    console.print(line)
    console.print()


def detect_model_source(model_id: str | None) -> str:
    """Detect model source for Setup header."""
    if model_id is None:
        return "HuggingFace"
    p = Path(model_id)
    if p.suffix == ".onnx":
        return "ONNX"
    if p.is_dir():
        return "Local"
    return "HuggingFace"


def fmt_size(size_bytes: int | float) -> str:
    """Format file size from bytes to human-readable string."""
    mb = size_bytes / (1024 * 1024)
    if mb >= 1000:
        return f"{mb / 1000:.1f} GB"
    return f"{mb:.1f} MB"


def get_onnx_total_size(onnx_path: Path) -> int:
    """Get total ONNX model size including external data files.

    When ONNX models use external data storage, the main .onnx file
    is just metadata (~1-2MB) while weights live in separate .data files.
    This function sums all related files.
    """
    total = onnx_path.stat().st_size
    try:
        from onnx import external_data_helper as edh

        from ..onnx import load_onnx

        model = load_onnx(onnx_path, load_weights=False, validate=False)
        seen: set[str] = set()
        for init in model.graph.initializer:
            if edh.uses_external_data(init):
                ext_info = edh.ExternalDataInfo(init)
                if ext_info.location and ext_info.location not in seen:
                    seen.add(ext_info.location)
                    ext_path = onnx_path.parent / ext_info.location
                    if ext_path.exists():
                        total += ext_path.stat().st_size
    except Exception:
        logger.debug(
            "Could not read external data for %s; reporting main file size only",
            onnx_path,
            exc_info=True,
        )
    return total


# ══════════════════════════════════════════════════════════════════════════
# BUILD COMMAND — STAGE LIVE (cascading Live per stage)
# ══════════════════════════════════════════════════════════════════════════


class _SafeLive(Live):
    """A :class:`rich.live.Live` that survives Windows console hiccups.

    Some native dependencies (e.g. OpenVINO / ONNX Runtime EPs) can put
    the Windows console handle into a state where ANSI/control writes
    fail with ``OSError: [WinError 1]`` ("Incorrect function").  That
    error is raised from Rich's daemon refresh thread, which then dies
    and dumps an ugly traceback into the middle of build output even
    though the build itself succeeds.

    This subclass catches :class:`OSError` in :meth:`refresh` and, after
    the first failure, disables further refreshes for the lifetime of
    this Live instance.  ``start``/``stop``/``update`` are also wrapped
    so callers can use ``_SafeLive`` exactly like :class:`Live`.  The
    visible effect is that the final frame may not animate further, but
    no traceback escapes and the surrounding pipeline continues normally.
    """

    @staticmethod
    def _swallow_oserror(method: F) -> F:
        """Decorator: invoke ``method`` and swallow console ``OSError``."""

        @functools.wraps(method)
        def wrapper(self: _SafeLive, *args: Any, **kwargs: Any) -> Any:
            try:
                return method(self, *args, **kwargs)
            except OSError:
                logger.debug("Ignoring OSError from Live.%s", method.__name__, exc_info=True)
                return None

        return wrapper  # type: ignore[return-value]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._refresh_disabled: bool = False

    def refresh(self) -> None:
        if self._refresh_disabled:
            return
        try:
            super().refresh()
        except OSError:
            # Console handle is unusable (typically VT/handle state damaged
            # by a native library).  Disable further refreshes so the
            # daemon thread does not spam tracebacks.
            self._refresh_disabled = True
            logger.debug("Disabling Live refresh after OSError", exc_info=True)

    @_swallow_oserror
    def start(self, refresh: bool = False) -> None:
        super().start(refresh=refresh)

    @_swallow_oserror
    def stop(self) -> None:
        super().stop()

    @_swallow_oserror
    def update(self, renderable: RenderableType, *, refresh: bool = False) -> None:
        super().update(renderable, refresh=refresh)


class StageLive:
    """Live region for a single build stage.

    Each stage gets its own Rich Live context. When the stage completes,
    Live stops and the final frame persists as static text (transient=False).
    The next stage starts a new Live below.

    Usage::

        with StageLive("export", console) as sl:
            sl.kv("Task:", "fill-mask  [dim](auto-detected)[/dim]")
            sl.io_input("input_ids", "[1, 128]", "int64")
            # ... blocking work ...
            sl.set_done(12.3)
            sl.artifact("output/export.onnx", 438_200_000)
    """

    def __init__(self, name: str, console: Console) -> None:
        self._name = name
        self._console = console
        self._lines: list[RenderableType] = []
        self._live: _SafeLive | None = None
        self._status_idx: int = 0

    def __enter__(self) -> StageLive:
        self._lines = [self._make_running_line()]
        self._status_idx = 0
        self._live = _SafeLive(
            self._render(),
            console=self._console,
            refresh_per_second=15,
            transient=False,
        )
        self._live.start()
        return self

    def __exit__(self, *_: object) -> None:
        if self._live:
            self._live.update(self._render())
            self._live.stop()
            self._live = None

    def _render(self) -> Group:
        return Group(*self._lines)

    def _update(self) -> None:
        if self._live:
            self._live.update(self._render())

    # ── Status line management ────────────────────────────────────

    def _make_running_line(self, detail: str = "") -> Text:
        line = Text()
        line.append(f"{ICON_RUNNING} ")
        line.append(self._name.capitalize(), style="bold yellow")
        if detail:
            line.append(f"  {detail}", style="dim")
        return line

    def set_status(self, detail: str) -> None:
        """Update the running status text."""
        self._lines[self._status_idx] = self._make_running_line(detail)
        self._update()

    def set_done(self, elapsed: float) -> None:
        """Mark stage as done."""
        line = Text()
        line.append(f"{ICON_DONE} ")
        line.append(f"{self._name.capitalize():<48}", style="green")
        line.append(f"{elapsed:.1f}s", style="green")
        self._lines[self._status_idx] = line
        self._update()

    def set_error(self, error: str = "") -> None:
        """Mark stage as failed."""
        line = Text()
        line.append(f"{ICON_ERROR} ")
        line.append(self._name.capitalize(), style="bold red")
        if error:
            line.append(f"  {error}", style="red")
        self._lines[self._status_idx] = line
        self._update()

    # ── Detail lines (indented under stage) ───────────────────────

    def detail(self, markup: str) -> None:
        """Add a Rich markup detail line."""
        self._lines.append(Text.from_markup(f"   {markup}"))
        self._update()

    def kv(self, label: str, value: str) -> None:
        """Add a key-value detail line with aligned columns."""
        self._lines.append(Text.from_markup(f"   {label:<14}{value}"))
        self._update()

    def artifact(self, path: str, size_bytes: int | float) -> None:
        """Add artifact line (always last in stage)."""
        label = "\U0001f4e6 Artifact:"
        self._lines.append(
            Text.from_markup(f"   {label:<14}[dim]{path}[/dim]  ({fmt_size(size_bytes)})")
        )
        self._update()

    def blank(self) -> None:
        """Add a blank line."""
        self._lines.append(Text(""))
        self._update()

    # ── I/O lines (aligned columns) ──────────────────────────────

    def io_input(
        self,
        name: str,
        shape: str,
        dtype: str,
        *,
        first: bool = True,
    ) -> None:
        """Add an input tensor line."""
        label = "Input:        " if first else "              "
        self._lines.append(
            Text.from_markup(f"   {label}[cyan]{name:<18}[/cyan] {shape:<14} [dim]{dtype}[/dim]")
        )
        self._update()

    def io_output(
        self,
        name: str,
        shape: str,
        dtype: str,
        *,
        first: bool = True,
    ) -> None:
        """Add an output tensor line."""
        label = "Output:       " if first else "              "
        self._lines.append(
            Text.from_markup(f"   {label}[cyan]{name:<18}[/cyan] {shape:<14} [dim]{dtype}[/dim]")
        )
        self._update()

    # ── EP analyzer bar lines (for optimize stage) ────────────────

    def ep_bar_add(self, ep_name: EPName, total: int = 0) -> int:
        """Add a placeholder EP bar line, return index."""
        idx = len(self._lines)
        line = Text()
        line.append("   - ")
        line.append(f"{ep_name:<28}", style="dim")
        if total:
            line.append("\u2591" * MAX_BAR_WIDTH, style="dim")
        self._lines.append(line)
        self._update()
        return idx

    def ep_bar_update(
        self,
        idx: int,
        ep_name: EPName,
        s: int,
        p: int,
        u: int,
        total: int = 0,
    ) -> None:
        """Update an EP bar line by index with progress."""
        line = Text()
        line.append("   - ")
        line.append(f"{ep_name:<28}", style="cyan")
        line.append_text(_spu_text(s, p, u))
        line.append("  ")
        # Scale bar proportional to total (not analyzed count)
        analyzed = s + p + u
        anchor = max(total, analyzed, 1)
        line.append_text(_build_bar_scaled(s, p, u, anchor))
        remaining = total - analyzed if total else 0
        if remaining > 0:
            rem_w = max(
                1,
                round(remaining / anchor * MAX_BAR_WIDTH),
            )
            line.append("\u2591" * rem_w, style="dim")
        self._lines[idx] = line
        self._update()


# ══════════════════════════════════════════════════════════════════════════
# EP ANALYZER BAR HELPERS
# ══════════════════════════════════════════════════════════════════════════


def _build_bar(s: int, p: int, u: int) -> Text:
    """Build a compact stacked bar for S/P/U counts."""
    total = s + p + u
    if total == 0:
        return Text()
    return _build_bar_scaled(s, p, u, total)


def _build_bar_scaled(s: int, p: int, u: int, anchor: int) -> Text:
    """Build a stacked bar scaled to an anchor total."""
    if anchor == 0:
        return Text()
    bar = Text()
    s_w = max(1, round(s / anchor * MAX_BAR_WIDTH)) if s else 0
    p_w = max(1, round(p / anchor * MAX_BAR_WIDTH)) if p else 0
    u_w = max(1, round(u / anchor * MAX_BAR_WIDTH)) if u else 0
    # Clamp total to MAX_BAR_WIDTH
    used = s_w + p_w + u_w
    if used > MAX_BAR_WIDTH:
        overflow = used - MAX_BAR_WIDTH
        # Shrink from the largest segment
        if s_w >= p_w and s_w >= u_w:
            s_w = max(1, s_w - overflow)
        elif p_w >= u_w:
            p_w = max(1, p_w - overflow)
        else:
            u_w = max(1, u_w - overflow)
    bar.append("\u2588" * s_w, style="green")
    if p_w:
        bar.append("\u2588" * p_w, style="yellow")
    if u_w:
        bar.append("\u2588" * u_w, style="red")
    return bar


def _spu_text(s: int, p: int, u: int) -> Text:
    """Build 'S/P/U' colored count text."""
    t = Text()
    t.append(str(s), style="bold green")
    t.append("/", style="dim")
    t.append(str(p), style="bold yellow" if p > 0 else "dim")
    t.append("/", style="dim")
    t.append(str(u), style="bold red" if u > 0 else "dim")
    return t


# ══════════════════════════════════════════════════════════════════════════
# ONNX GRAPH SUMMARY (for compile stage)
# ══════════════════════════════════════════════════════════════════════════


def get_onnx_graph_summary(model_path: Path | str) -> dict[str, Any]:
    """Extract graph summary from ONNX model without loading weights.

    Returns dict with:
        op_counts: dict[str, int] — node count per op_type (excl QDQ)
        inputs: list[dict] — [{name, shape, dtype}, ...]
        outputs: list[dict] — [{name, shape, dtype}, ...]
        num_initializers: int
        total_nodes: int
    """
    from onnx import TensorProto

    from ..onnx import load_onnx

    _dtype_map = {
        TensorProto.FLOAT: "float32",
        TensorProto.FLOAT16: "float16",
        TensorProto.INT32: "int32",
        TensorProto.INT64: "int64",
        TensorProto.INT8: "int8",
        TensorProto.UINT8: "uint8",
        TensorProto.BOOL: "bool",
        TensorProto.STRING: "string",
    }

    model = load_onnx(model_path, load_weights=False, validate=False)
    graph = model.graph

    # Op counts (exclude QDQ nodes from display)
    qdq_ops = {"QuantizeLinear", "DequantizeLinear"}
    op_counts: dict[str, int] = {}
    for node in graph.node:
        if node.op_type not in qdq_ops:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    # Sort by count descending
    op_counts = dict(sorted(op_counts.items(), key=lambda x: x[1], reverse=True))

    # Inputs (exclude initializer names — they appear in graph.input too)
    init_names = {init.name for init in graph.initializer}

    def _parse_io(value_info: Any) -> dict:
        name = value_info.name
        tt = value_info.type.tensor_type
        dtype = _dtype_map.get(tt.elem_type, f"type({tt.elem_type})")
        dims = []
        if tt.HasField("shape"):
            for d in tt.shape.dim:
                if d.dim_param:
                    dims.append(d.dim_param)
                else:
                    dims.append(d.dim_value)
        return {"name": name, "shape": dims, "dtype": dtype}

    inputs = [_parse_io(inp) for inp in graph.input if inp.name not in init_names]
    outputs = [_parse_io(out) for out in graph.output]

    return {
        "op_counts": op_counts,
        "inputs": inputs,
        "outputs": outputs,
        "num_initializers": len(graph.initializer),
        "total_nodes": len(graph.node),
    }
