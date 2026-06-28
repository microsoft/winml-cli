# ruff: noqa
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Mock v5: Cascading Live — each stage is its own Live region.

Run:  uv run python temp/mock_build_output_v5.py [--2stage] [--3stage] [--onnx] [--qdq] [--reuse] [--error] [--all]

Design v5:
- Cascading Live: each stage gets its own Live area
- When stage completes, Live stops → final text printed as static
- Next stage starts a new Live
- Model source detection: HuggingFace / ONNX / Local
- Two sections: 🔧 Setup — {source} + 🎯 Stages
- Artifact always last in each stage
"""

from __future__ import annotations

import sys
import time

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.text import Text


console = Console(stderr=True)

HEAVY_SEP = "\u2550" * 60
LIGHT_SEP = "\u2500" * 60
MAX_BAR_WIDTH = 36

ICON_RUNNING = "\u23f3"
ICON_DONE = "\u2705"
ICON_ERROR = "\u274c"


def _fmt_size(mb: float) -> str:
    return f"{mb / 1000:.1f} GB" if mb >= 1000 else f"{mb:.1f} MB"


def _build_bar(s: int, p: int, u: int) -> Text:
    total = s + p + u
    if total == 0:
        return Text()
    bar = Text()
    s_w = max(1, round(s / total * MAX_BAR_WIDTH)) if s else 0
    p_w = max(1, round(p / total * MAX_BAR_WIDTH)) if p else 0
    u_w = max(1, round(u / total * MAX_BAR_WIDTH)) if u else 0
    used = s_w + p_w + u_w
    if used > MAX_BAR_WIDTH:
        s_w = max(1, s_w - (used - MAX_BAR_WIDTH))
    bar.append("\u2588" * s_w, style="green")
    if p_w:
        bar.append("\u2588" * p_w, style="yellow")
    if u_w:
        bar.append("\u2588" * u_w, style="red")
    return bar


def _spu_text(s: int, p: int, u: int) -> Text:
    t = Text()
    t.append(str(s), style="bold green")
    t.append("/", style="dim")
    t.append(str(p), style="bold yellow" if p > 0 else "dim")
    t.append("/", style="dim")
    t.append(str(u), style="bold red" if u > 0 else "dim")
    return t


# ══════════════════════════════════════════════════════════════════════════
# STAGE LIVE — a single stage's animated region
# ══════════════════════════════════════════════════════════════════════════


class StageLive:
    """Live region for a single build stage.

    Usage:
        with StageLive("export") as sl:
            sl.set_status("Exporting to ONNX...")
            ...
            sl.set_done(12.3)
            sl.detail("Task: fill-mask")
            sl.artifact("output/export.onnx", 438.2)
    After the `with` block exits, the final content is printed as static text.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._lines: list[RenderableType] = []
        self._live: Live | None = None
        self._status_line_idx: int = 0

    def __enter__(self) -> StageLive:
        # Start with a running status line
        self._lines = [self._make_running_line()]
        self._status_line_idx = 0
        self._live = Live(
            self._render(),
            console=console,
            refresh_per_second=15,
            transient=False,  # Keep final frame — avoids flicker between stages
        )
        self._live.start()
        return self

    def __exit__(self, *_: object) -> None:
        if self._live:
            # Final update to ensure last state is rendered
            self._live.update(self._render())
            self._live.stop()
            self._live = None
        # No re-print needed — Live's final frame stays on screen

    def _render(self) -> Group:
        return Group(*self._lines)

    def _update(self) -> None:
        if self._live:
            self._live.update(self._render())

    def _make_running_line(self, detail: str = "") -> Text:
        line = Text()
        line.append(f"{ICON_RUNNING} ")
        line.append(self._name.capitalize(), style="bold yellow")
        if detail:
            line.append(f"  {detail}", style="dim")
        return line

    def _make_done_line(self, elapsed: float) -> Text:
        line = Text()
        line.append(f"{ICON_DONE} ")
        line.append(f"{self._name.capitalize():<48}", style="green")
        line.append(f"{elapsed:.1f}s", style="green")
        return line

    def _make_error_line(self, error: str = "") -> Text:
        line = Text()
        line.append(f"{ICON_ERROR} ")
        line.append(self._name.capitalize(), style="bold red")
        if error:
            line.append(f"  {error}", style="red")
        return line

    # ── Public API ────────────────────────────────────────────────

    def set_status(self, detail: str) -> None:
        """Update the running status text."""
        self._lines[self._status_line_idx] = self._make_running_line(detail)
        self._update()

    def set_done(self, elapsed: float) -> None:
        """Mark stage as done — replaces the status line."""
        self._lines[self._status_line_idx] = self._make_done_line(elapsed)
        self._update()

    def set_error(self, error: str = "") -> None:
        """Mark stage as failed."""
        self._lines[self._status_line_idx] = self._make_error_line(error)
        self._update()

    def detail(self, markup: str) -> None:
        """Add a detail line (indented under stage)."""
        self._lines.append(Text.from_markup(f"   {markup}"))
        self._update()

    def kv(self, label: str, value: str) -> None:
        """Add a key-value detail line with aligned columns."""
        self._lines.append(Text.from_markup(f"   {label:<14}{value}"))
        self._update()

    def artifact(self, path: str, size_mb: float) -> None:
        """Add artifact line (always last)."""
        label = "\U0001f4e6 Artifact:"
        self._lines.append(
            Text.from_markup(f"   {label:<14}[dim]{path}[/dim]  ({_fmt_size(size_mb)})")
        )
        self._update()

    def blank(self) -> None:
        self._lines.append(Text(""))
        self._update()

    def ep_bar_add(self, ep_name: str) -> int:
        """Add a placeholder EP bar line, return its index."""
        idx = len(self._lines)
        self._lines.append(Text(f"   - {ep_name:<24}...", style="dim"))
        self._update()
        return idx

    def ep_bar_update(self, idx: int, ep_name: str, s: int, p: int, u: int) -> None:
        """Update an EP bar line by index."""
        line = Text()
        line.append("   - ")
        line.append(f"{ep_name:<24}", style="cyan")
        line.append_text(_spu_text(s, p, u))
        line.append("  ")
        line.append_text(_build_bar(s, p, u))
        self._lines[idx] = line
        self._update()

    def io_input(self, name: str, shape: str, dtype: str, first: bool = True) -> None:
        label = "Input:        " if first else "              "
        self._lines.append(
            Text.from_markup(f"   {label}[cyan]{name:<18}[/cyan] {shape:<14} [dim]{dtype}[/dim]")
        )
        self._update()

    def io_output(self, name: str, shape: str, dtype: str, first: bool = True) -> None:
        label = "Output:       " if first else "              "
        self._lines.append(
            Text.from_markup(f"   {label}[cyan]{name:<18}[/cyan] {shape:<14} [dim]{dtype}[/dim]")
        )
        self._update()


# ══════════════════════════════════════════════════════════════════════════
# STATIC STAGE (for skipped stages — no animation needed)
# ══════════════════════════════════════════════════════════════════════════


def print_stage_skip(name: str, reason: str = "") -> None:
    """Print a skipped stage as static text."""
    line = Text()
    line.append("\u23f8\ufe0f  ")
    line.append(name.capitalize(), style="dim")
    if reason:
        line.append(f"  {reason}", style="dim italic")
    console.print(line)
    console.print()


# ══════════════════════════════════════════════════════════════════════════
# HEADER / FOOTER
# ══════════════════════════════════════════════════════════════════════════


def print_setup(
    model: str,
    config: str,
    output: str,
    source: str = "HuggingFace",
) -> None:
    console.print()
    console.print(HEAVY_SEP)
    console.print(f"[bold]\U0001f527 Setup \u2014 {source}[/bold]")
    console.print(HEAVY_SEP)
    console.print(f"   \U0001f4e6 [bold]{'Model:':<10}[/bold] [cyan]{model}[/cyan]")
    console.print(f"   \U0001f4c1 [bold]{'Config:':<10}[/bold] [cyan]{config}[/cyan]")
    console.print(f"   \U0001f4c2 [bold]{'Output:':<10}[/bold] [cyan]{output}[/cyan]")
    console.print()


def print_stages_header() -> None:
    console.print(HEAVY_SEP)
    console.print("[bold]\U0001f3af Stages[/bold]")
    console.print(HEAVY_SEP)


def print_final(
    elapsed: float,
    artifact: str,
    stage_timings: list[tuple[str, float | None]] | None = None,
) -> None:
    """Print final summary section with stage timing breakdown.

    stage_timings: list of (stage_name, elapsed_seconds | None for skipped)
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
    console.print()


# ══════════════════════════════════════════════════════════════════════════
# ANIMATE HELPER
# ══════════════════════════════════════════════════════════════════════════


def animate_ep(sl: StageLive, ep_name: str, s: int, p: int) -> None:
    """Animate a single EP bar from 0 to final counts."""
    idx = sl.ep_bar_add(ep_name)
    steps = 15
    for i in range(1, steps + 1):
        frac = i / steps
        cur_s = min(int(s * frac), s)
        cur_p = min(int(p * frac), p)
        sl.ep_bar_update(idx, ep_name, cur_s, cur_p, 0)
        time.sleep(0.15)  # Realistic: each node check takes ~0.1-0.2s
    sl.ep_bar_update(idx, ep_name, s, p, 0)


# ══════════════════════════════════════════════════════════════════════════
# SCENARIOS
# ══════════════════════════════════════════════════════════════════════════


def demo_full_4stage() -> None:
    """Full: export → optimize → quantize → compile (HuggingFace)."""
    print_setup(
        model="bert-base-uncased  [dim](pretrained)[/dim]",
        config="config.json",
        output="output/",
        source="HuggingFace",
    )
    print_stages_header()

    # ── Export ────────────────────────────────────────────────────
    with StageLive("export") as sl:
        sl.set_status("Exporting to ONNX...")
        # Meta info known before export (from loader/config resolution)
        sl.kv("Model class:", "[cyan]BertForMaskedLM[/cyan]  [dim](auto-detected)[/dim]")
        sl.kv("Task:", "[cyan]fill-mask[/cyan]  [dim](auto-detected)[/dim]")
        sl.io_input("input_ids", "[1, 128]", "int64")
        sl.io_input("attention_mask", "[1, 128]", "int64", first=False)
        sl.io_input("token_type_ids", "[1, 128]", "int64", first=False)
        sl.io_output("logits", "[1, 30522]", "float32")
        time.sleep(10.0)  # Realistic: export takes ~10s
        sl.set_done(12.3)
        sl.artifact("output/export.onnx", 438.2)
        sl.blank()

    # ── Optimize ──────────────────────────────────────────────────
    with StageLive("optimize") as sl:
        sl.set_status("Optimizing ONNX graph...")
        time.sleep(2.0)  # Realistic: initial optimize ~2s

        # Autoconf iter 1
        sl.detail("[bold]Analyzing[/bold]  [dim](iter 1/3)[/dim]")
        animate_ep(sl, "QnnExecutionProvider", 325, 15)
        animate_ep(sl, "OpenVINOProvider", 340, 0)

        sl.detail("[bold]Patterns[/bold]")
        sl.detail("  [yellow]Gelu[/yellow]  [dim]\u2192 disable_gelu_fusion[/dim]")
        time.sleep(0.5)

        sl.detail("[bold]Optimizing[/bold]  [dim](applying autoconf)[/dim]")
        sl.detail("  [dim]{disable_gelu_fusion: true}[/dim]")
        time.sleep(2.0)  # Realistic: re-optimize ~2s

        # Autoconf iter 2
        sl.detail("[bold]Analyzing[/bold]  [dim](iter 2/3)[/dim]")
        animate_ep(sl, "QnnExecutionProvider", 340, 0)
        animate_ep(sl, "OpenVINOProvider", 340, 0)

        sl.detail("[dim]Autoconf converged after 2 iteration(s)[/dim]")
        sl.set_done(3.1)
        sl.artifact("output/optimized.onnx", 412.5)
        sl.blank()
        time.sleep(0.2)

    # ── Quantize ──────────────────────────────────────────────────
    with StageLive("quantize") as sl:
        sl.set_status("Quantizing (uint8)...")
        sl.kv("Dataset:", "[cyan]timm/imagenet-1k-wds[/cyan]  [dim](test)[/dim]")
        sl.kv("Calibration:", "[cyan]10[/cyan] samples  [dim](minmax)[/dim]")
        time.sleep(8.0)  # Realistic: quantize ~8s
        sl.set_done(8.7)
        sl.kv("Precision:", "[cyan]uint8/uint8[/cyan]  [dim](weight/activation)[/dim]")
        sl.artifact("output/quantized.onnx", 112.8)
        sl.blank()

    # ── Compile ───────────────────────────────────────────────────
    with StageLive("compile") as sl:
        sl.set_status("Compiling for QNN...")
        time.sleep(3.0)  # Realistic: compile ~3s
        sl.set_done(2.1)
        sl.detail(
            "[bold]Graph:[/bold]  [cyan]EPContext[/cyan] (1), "
            "[cyan]Conv[/cyan] (8), "
            "[cyan]MatMul[/cyan] (12), "
            "[cyan]Add[/cyan] (15), "
            "[cyan]Relu[/cyan] (8)"
        )
        sl.artifact("output/compiled.onnx", 112.8)
        time.sleep(0.1)

    print_final(
        26.2,
        "output/model.onnx",
        stage_timings=[
            ("Export", 12.3),
            ("Optimize", 3.1),
            ("Quantize", 8.7),
            ("Compile", 2.1),
        ],
    )


def demo_2stage() -> None:
    """Export + optimize only (HuggingFace)."""
    print_setup(
        model="bert-base-uncased  [dim](pretrained)[/dim]",
        config="config_portable.json",
        output="output/",
        source="HuggingFace",
    )
    print_stages_header()

    with StageLive("export") as sl:
        sl.set_status("Exporting to ONNX...")
        sl.kv("Model class:", "[cyan]BertForMaskedLM[/cyan]  [dim](auto-detected)[/dim]")
        sl.kv("Task:", "[cyan]fill-mask[/cyan]  [dim](auto-detected)[/dim]")
        sl.io_input("input_ids", "[1, 128]", "int64")
        sl.io_input("attention_mask", "[1, 128]", "int64", first=False)
        sl.io_input("token_type_ids", "[1, 128]", "int64", first=False)
        sl.io_output("logits", "[1, 30522]", "float32")
        time.sleep(5.0)
        sl.set_done(12.3)
        sl.artifact("output/export.onnx", 438.2)
        sl.blank()

    with StageLive("optimize") as sl:
        sl.set_status("Optimizing...")
        time.sleep(1.5)
        sl.detail("[bold]Analyzing[/bold]  [dim](iter 1/3)[/dim]")
        animate_ep(sl, "QnnExecutionProvider", 325, 15)
        sl.detail("[dim]Autoconf converged after 1 iteration(s)[/dim]")
        sl.set_done(3.1)
        sl.artifact("output/optimized.onnx", 412.5)

    print_final(
        15.4,
        "output/model.onnx",
        stage_timings=[
            ("Export", 12.3),
            ("Optimize", 3.1),
        ],
    )


def demo_3stage() -> None:
    """Export + optimize + quantize (HuggingFace, no compile)."""
    print_setup(
        model="microsoft/resnet-50  [dim](pretrained)[/dim]",
        config="config_npu_noc.json",
        output="output/",
        source="HuggingFace",
    )
    print_stages_header()

    with StageLive("export") as sl:
        sl.set_status("Exporting to ONNX...")
        sl.kv(
            "Model class:", "[cyan]ResNetForImageClassification[/cyan]  [dim](auto-detected)[/dim]"
        )
        sl.kv("Task:", "[cyan]image-classification[/cyan]  [dim](auto-detected)[/dim]")
        sl.io_input("pixel_values", "[1, 3, 224, 224]", "float32")
        sl.io_output("logits", "[1, 1000]", "float32")
        time.sleep(4.0)
        sl.set_done(5.1)
        sl.artifact("output/export.onnx", 97.3)
        sl.blank()

    with StageLive("optimize") as sl:
        sl.set_status("Optimizing...")
        time.sleep(1.5)
        sl.detail("[bold]Analyzing[/bold]  [dim](iter 1/3)[/dim]")
        animate_ep(sl, "QnnExecutionProvider", 127, 5)
        sl.detail("[dim]Autoconf converged after 1 iteration(s)[/dim]")
        sl.set_done(1.8)
        sl.artifact("output/optimized.onnx", 89.4)
        sl.blank()

    with StageLive("quantize") as sl:
        sl.set_status("Quantizing (uint8)...")
        sl.kv("Dataset:", "[cyan]timm/imagenet-1k-wds[/cyan]  [dim](test)[/dim]")
        sl.kv("Calibration:", "[cyan]10[/cyan] samples  [dim](minmax)[/dim]")
        time.sleep(4.0)
        sl.set_done(4.2)
        sl.kv("Precision:", "[cyan]uint8/uint8[/cyan]  [dim](weight/activation)[/dim]")
        sl.artifact("output/quantized.onnx", 25.1)

    print_final(
        11.1,
        "output/model.onnx",
        stage_timings=[
            ("Export", 5.1),
            ("Optimize", 1.8),
            ("Quantize", 4.2),
        ],
    )


def demo_onnx() -> None:
    """ONNX input — no export. I/O under optimize."""
    print_setup(
        model="model.onnx  [dim](438.2 MB)[/dim]",
        config="config.json",
        output="output/",
        source="ONNX",
    )
    print_stages_header()

    with StageLive("optimize") as sl:
        sl.set_status("Optimizing...")
        time.sleep(0.3)
        sl.detail("[bold]Analyzing[/bold]  [dim](iter 1/3)[/dim]")
        animate_ep(sl, "QnnExecutionProvider", 340, 0)
        sl.detail("[dim]Autoconf converged after 1 iteration(s)[/dim]")
        sl.io_input("pixel_values", "[1, 3, 224, 224]", "float32")
        sl.io_output("logits", "[1, 1000]", "float32")
        sl.set_done(3.1)
        sl.artifact("output/model_optimized.onnx", 412.5)
        sl.blank()

    with StageLive("quantize") as sl:
        sl.set_status("Quantizing (uint8)...")
        sl.kv("Dataset:", "[cyan]timm/imagenet-1k-wds[/cyan]  [dim](test)[/dim]")
        sl.kv("Calibration:", "[cyan]10[/cyan] samples  [dim](minmax)[/dim]")
        time.sleep(0.5)
        sl.set_done(8.7)
        sl.kv("Precision:", "[cyan]uint8/uint8[/cyan]  [dim](weight/activation)[/dim]")
        sl.artifact("output/model_quantized.onnx", 112.8)
        sl.blank()

    with StageLive("compile") as sl:
        sl.set_status("Compiling for QNN...")
        time.sleep(0.4)
        sl.set_done(2.1)
        sl.detail(
            "[bold]Graph:[/bold]  [cyan]EPContext[/cyan] (1), "
            "[cyan]Conv[/cyan] (8), [cyan]Relu[/cyan] (16), [cyan]Add[/cyan] (8)"
        )
        sl.artifact("output/model_compiled.onnx", 112.8)

    print_final(
        13.9,
        "output/model.onnx",
        stage_timings=[
            ("Optimize", 3.1),
            ("Quantize", 8.7),
            ("Compile", 2.1),
        ],
    )


def demo_qdq_skip() -> None:
    """Quantize in config but auto-skipped (QDQ detected)."""
    print_setup(
        model="prequantized-model  [dim](pretrained)[/dim]",
        config="config.json",
        output="output/",
        source="HuggingFace",
    )
    print_stages_header()

    with StageLive("export") as sl:
        sl.set_status("Exporting to ONNX...")
        sl.kv("Model class:", "[cyan]BertForMaskedLM[/cyan]")
        sl.kv("Task:", "[cyan]fill-mask[/cyan]")
        sl.io_input("input_ids", "[1, 128]", "int64")
        sl.io_output("logits", "[1, 30522]", "float32")
        time.sleep(5.0)
        sl.set_done(12.3)
        sl.artifact("output/export.onnx", 438.2)
        sl.blank()

    with StageLive("optimize") as sl:
        sl.set_status("Optimizing...")
        time.sleep(1.5)
        sl.detail("[bold]Analyzing[/bold]  [dim](iter 1/3)[/dim]")
        animate_ep(sl, "QnnExecutionProvider", 340, 0)
        sl.detail("[dim]Autoconf converged after 1 iteration(s)[/dim]")
        sl.set_done(3.1)
        sl.artifact("output/optimized.onnx", 412.5)
        sl.blank()

    # Skipped — static, no Live needed
    print_stage_skip("quantize", "(QDQ nodes already present)")

    with StageLive("compile") as sl:
        sl.set_status("Compiling for QNN...")
        time.sleep(2.0)
        sl.set_done(2.1)
        sl.detail("[bold]Graph:[/bold]  [cyan]EPContext[/cyan] (1), [cyan]Conv[/cyan] (8)")
        sl.artifact("output/compiled.onnx", 438.2)

    print_final(
        17.5,
        "output/model.onnx",
        stage_timings=[
            ("Export", 12.3),
            ("Optimize", 3.1),
            ("Quantize", None),
            ("Compile", 2.1),
        ],
    )


def demo_reuse() -> None:
    """Existing artifact found."""
    print_setup(
        model="bert-base-uncased",
        config="config.json",
        output="output/",
        source="HuggingFace",
    )
    print_stages_header()
    console.print()
    console.print(
        "   \u267b\ufe0f  [bold cyan]Existing artifact found:[/bold cyan] output/model.onnx"
    )
    console.print("   \U0001f4a1 [dim]Use --rebuild to force rebuild.[/dim]")
    console.print()


def demo_error() -> None:
    """Build failure during quantize."""
    print_setup(
        model="custom-model  [dim](pretrained)[/dim]",
        config="config.json",
        output="output/",
        source="HuggingFace",
    )
    print_stages_header()

    with StageLive("export") as sl:
        sl.set_status("Exporting to ONNX...")
        sl.kv("Model class:", "[cyan]CustomForSequenceClassification[/cyan]")
        sl.kv("Task:", "[cyan]text-classification[/cyan]")
        sl.io_input("input_ids", "[1, 128]", "int64")
        sl.io_output("logits", "[1, 2]", "float32")
        time.sleep(5.0)
        sl.set_done(12.3)
        sl.artifact("output/export.onnx", 438.2)
        sl.blank()

    with StageLive("optimize") as sl:
        sl.set_status("Optimizing...")
        time.sleep(1.5)
        sl.detail("[bold]Analyzing[/bold]  [dim](iter 1/3)[/dim]")
        animate_ep(sl, "QnnExecutionProvider", 300, 20)
        sl.detail("[dim]Autoconf converged after 1 iteration(s)[/dim]")
        sl.set_done(3.1)
        sl.artifact("output/optimized.onnx", 412.5)
        sl.blank()

    with StageLive("quantize") as sl:
        sl.set_status("Quantizing (int8)...")
        time.sleep(3.0)
        sl.set_error("Unsupported op 'CustomOp'")

    console.print()
    console.print(
        "   [bold red]\u274c Quantization failed:[/bold red]"
        " Unsupported op type 'CustomOp' for int8"
    )
    console.print("   \U0001f4a1 [dim]Try: --no-quant to skip quantization[/dim]")
    console.print("   \U0001f4a1 [dim]Try: wmk analyze -m model.onnx --ep qnn to investigate[/dim]")
    console.print()


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = set(sys.argv[1:])

    if "--help" in args or "-h" in args:
        console.print("[bold]Usage:[/bold] uv run python temp/mock_build_output_v5.py [OPTIONS]")
        console.print()
        console.print("  [dim](no flags)[/dim]    Full 4-stage (HuggingFace)")
        console.print("  --2stage       Export + Optimize only")
        console.print("  --3stage       Export + Optimize + Quantize")
        console.print("  --onnx         ONNX input")
        console.print("  --qdq          QDQ auto-skip")
        console.print("  --reuse        Existing artifact")
        console.print("  --error        Build failure")
        console.print("  --all          All scenarios")
        sys.exit(0)

    if "--all" in args:
        for label, fn in [
            ("Full 4-stage (HuggingFace)", demo_full_4stage),
            ("2-stage: export + optimize", demo_2stage),
            ("3-stage: no compile", demo_3stage),
            ("ONNX input", demo_onnx),
            ("QDQ auto-skip", demo_qdq_skip),
            ("Existing artifact", demo_reuse),
            ("Build failure", demo_error),
        ]:
            console.print()
            console.print(f"[bold yellow]{'=' * 60}[/bold yellow]")
            console.print(f"[bold yellow]\u25b6 Scenario: {label}[/bold yellow]")
            console.print(f"[bold yellow]{'=' * 60}[/bold yellow]")
            fn()
        sys.exit(0)

    if "--2stage" in args:
        demo_2stage()
    elif "--3stage" in args:
        demo_3stage()
    elif "--onnx" in args:
        demo_onnx()
    elif "--qdq" in args:
        demo_qdq_skip()
    elif "--reuse" in args:
        demo_reuse()
    elif "--error" in args:
        demo_error()
    else:
        demo_full_4stage()
