# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Live hardware monitor display for performance benchmarking.

Renders a live adapter/CPU utilization chart during benchmarking using
plotext for chart rendering and Rich Live for terminal refresh.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel

from ..session.monitor.hw_monitor import adapter_label
from ..utils.constants import ACCELERATOR_DEVICE_TYPES


# Moving window size for the x-axis (seconds)
_CHART_WINDOW_SECONDS = 15.0

# Display refresh rate (frames per second)
_REFRESH_FPS = 5


def _avg_now(
    samples: list[float] | None,
    fallback_now: float = 0.0,
) -> tuple[float, float]:
    """Return ``(avg, now)`` for a samples list.

    ``fallback_now`` is used for the ``now`` value when ``samples`` is empty
    or ``None`` (e.g. when a caller has a scalar current reading but no
    time-series to compute an average from — in that case ``avg`` mirrors
    the scalar so the display stays honest rather than reading 0.0).
    """
    if not samples:
        return (fallback_now, fallback_now)
    return (sum(samples) / len(samples), samples[-1])


class LiveMonitorDisplay:
    """Renders a live hardware utilization chart during benchmarking.

    Uses plotext for chart rendering and Rich Live for terminal refresh.
    """

    def __init__(
        self,
        total_iterations: int,
        warmup: int,
        model_id: str,
        device: str,
        chart_width: int = 120,
        chart_height: int = 15,
        poll_interval_ms: int = 100,
        device_kind: str | None = None,
    ) -> None:
        self._total = total_iterations
        self._warmup = warmup
        self._model_id = model_id
        self._device = device
        # `device_kind` is the value HWMonitor resolved at start() — pass it
        # in when you want the legend to reflect what's actually polled (e.g.
        # "auto" that resolved to GPU). Falls back to the requested string
        # when the caller doesn't know the resolved kind yet.
        if device_kind is None:
            requested = (device or "").lower()
            device_kind = requested if requested in ACCELERATOR_DEVICE_TYPES else None
        # When no adapter is polled (CPU-only / auto resolved to nothing),
        # hide the adapter line + status cell entirely instead of drawing
        # a flat zero series labelled "Adapter".
        self._show_adapter = device_kind is not None
        self._adapter_label = adapter_label(device_kind)
        self._chart_width = chart_width
        self._chart_height = chart_height
        self._poll_interval_s = poll_interval_ms / 1000.0
        self._live: Any = None
        # Track the last rendered panel for transient=False final display
        self._last_panel: Any = None

    def __enter__(self) -> LiveMonitorDisplay:
        from rich.live import Live

        self._live = Live(
            refresh_per_second=_REFRESH_FPS,
            console=Console(stderr=True),
            transient=False,  # Keep last frame visible in scrollback
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._live:
            self._live.__exit__(*exc)

    def update(
        self,
        iteration: int,
        latency_ms: float,
        util_samples: list[float],
        memory_local_mb: float = 0.0,
        memory_shared_mb: float = 0.0,
        cpu_pct: float = 0.0,
        ram_mb: float = 0.0,
        cpu_samples: list[float] | None = None,
        gpu_samples: list[float] | None = None,
        gpu_pct: float = 0.0,
    ) -> None:
        """Update the live display with current metrics."""
        if self._live is None:
            return

        try:
            chart_renderable = self._render_chart(util_samples, cpu_samples, gpu_samples)
            status_line = self._render_status(
                iteration,
                latency_ms,
                util_samples,
                memory_local_mb,
                memory_shared_mb,
                cpu_pct,
                ram_mb,
                gpu_pct=gpu_pct,
                cpu_samples=cpu_samples,
                gpu_samples=gpu_samples,
            )

            from rich.console import Group
            from rich.text import Text

            panel = Panel(
                Group(chart_renderable, Text.from_markup(status_line)),
                title=f"[bold]HW Monitor[/bold] - {self._model_id}",
                border_style="blue",
            )
            self._last_panel = panel
            self._live.update(panel)
        except Exception:
            pass  # Don't let display errors interrupt the benchmark

    def _render_chart(
        self,
        util_samples: list[float],
        cpu_samples: list[float] | None = None,
        gpu_samples: list[float] | None = None,
    ) -> Any:
        """Render utilization chart as a Rich renderable.

        Uses plotext with AnsiDecoder for flicker-free Rich Live integration.
        Plots the selected adapter (green), CPU (cyan), and aggregate GPU
        telemetry (yellow) with distinct colors.
        X-axis is a moving window of the last N seconds.
        Y-axis has fixed ticks: 0, 20, 40, 60, 80, 100.
        """
        adapter = self._adapter_label
        show_adapter = self._show_adapter
        try:
            import plotext as plt
        except ImportError:
            from rich.text import Text

            # CPU-only fallback: drop the adapter line entirely.
            if not show_adapter:
                if cpu_samples:
                    current = cpu_samples[-1]
                    bar_len = min(50, max(0, int(current / 2)))
                    bar = "#" * bar_len + "." * (50 - bar_len)
                    return Text(f"  CPU: [{bar}] {current:.1f}%")
                return Text("  CPU: [waiting for data...]")
            if util_samples:
                current = util_samples[-1]
                bar_len = min(50, max(0, int(current / 2)))
                bar = "#" * bar_len + "." * (50 - bar_len)
                return Text(f"  {adapter}: [{bar}] {current:.1f}%")
            return Text(f"  {adapter}: [waiting for data...]")

        plt.clf()
        plt.theme("clear")

        # Compute moving window: keep last N seconds of samples
        window_samples = int(_CHART_WINDOW_SECONDS / self._poll_interval_s)
        total_adapter = len(util_samples) if util_samples else 0

        # Plot the adapter line only when an adapter is actually being polled.
        if show_adapter:
            adapter_window = util_samples[-window_samples:] if util_samples else [0]
            window_start_idx = max(0, total_adapter - len(adapter_window))
            adapter_times = [
                (window_start_idx + i) * self._poll_interval_s for i in range(len(adapter_window))
            ]
            plt.plot(adapter_times, adapter_window, marker="braille", color="green")

        # Plot CPU in cyan (distinct from adapter)
        has_cpu = False
        total_cpu = len(cpu_samples) if cpu_samples else 0
        if cpu_samples:
            has_cpu = True
            cpu_window = cpu_samples[-window_samples:]
            cpu_start_idx = max(0, total_cpu - len(cpu_window))
            cpu_times = [
                (cpu_start_idx + i) * self._poll_interval_s for i in range(len(cpu_window))
            ]
            plt.plot(cpu_times, cpu_window, marker="braille", color="cyan")

        # Plot GPU in yellow (distinct from NPU green and CPU cyan)
        has_gpu = False
        if gpu_samples:
            has_gpu = True
            total_gpu = len(gpu_samples)
            gpu_window = gpu_samples[-window_samples:]
            gpu_start_idx = max(0, total_gpu - len(gpu_window))
            gpu_times = [
                (gpu_start_idx + i) * self._poll_interval_s for i in range(len(gpu_window))
            ]
            # plotext's palette exposes 'orange+' (ANSI bright yellow, code 11)
            # but has no 'yellow' key — `color="yellow"` silently falls through
            # to default (white). `orange+` matches Rich's `[bright_yellow]`
            # legend swatch below.
            plt.plot(gpu_times, gpu_window, marker="braille", color="orange+")

        # No plotext title -- we render our own Rich-colored title with legend
        plt.ylabel("Usage %")

        # Fixed y-axis: 0 to 100 with ticks at 0, 20, 40, 60, 80, 100
        plt.ylim(0, 100)
        plt.yticks([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])

        # X-axis: absolute elapsed time, sliding window. Use whichever series
        # we have to anchor the timeline so a CPU-only chart still scrolls.
        sample_count = total_adapter if show_adapter else total_cpu
        elapsed = sample_count * self._poll_interval_s
        x_min = max(0.0, elapsed - _CHART_WINDOW_SECONDS)
        x_max = max(elapsed, _CHART_WINDOW_SECONDS)
        plt.xlim(x_min, x_max)
        plt.xlabel("Time (s)")

        plt.plotsize(self._chart_width, self._chart_height)

        from rich.console import Group
        from rich.text import Text

        # Rich-colored title line with legend swatches
        legend_parts = []
        if show_adapter:
            legend_parts.append(f"[green]\u2588\u2588[/green] {adapter} %")
        if has_cpu:
            legend_parts.append("[cyan]\u2588\u2588[/cyan] CPU %")
        if has_gpu:
            legend_parts.append("[bright_yellow]\u2588\u2588[/bright_yellow] GPU %")
        title = Text.from_markup(f"  Utilization ({'  '.join(legend_parts)})")

        ansi_output = plt.build()
        chart_lines = [Text.from_ansi(line) for line in ansi_output.splitlines()]
        return Group(title, *chart_lines)

    def _render_status(
        self,
        iteration: int,
        latency_ms: float,
        util_samples: list[float],
        memory_local_mb: float = 0.0,
        memory_shared_mb: float = 0.0,
        cpu_pct: float = 0.0,
        ram_mb: float = 0.0,
        gpu_pct: float = 0.0,
        cpu_samples: list[float] | None = None,
        gpu_samples: list[float] | None = None,
    ) -> str:
        """Render 4-row status below the chart.

        Row 1: progress bar + phase counter + device label.
        Row 2: compute utilization (adapter / CPU / GPU) — unified ``now%/avg%``.
        Row 3: memory (Sys Mem + Device Mem local/shared).
        Row 4: inference latency + throughput.

        CPU and GPU accept a samples list to compute ``avg`` — the ``cpu_pct``
        and ``gpu_pct`` scalars remain as the ``now`` value (and as fallbacks
        for ``avg`` when no samples were supplied).
        """
        phase = "warmup" if iteration <= self._warmup else "benchmark"
        effective_iter = iteration - self._warmup if phase == "benchmark" else iteration
        total_bench = self._total - self._warmup

        pct = iteration / self._total if self._total > 0 else 0
        bar_len = int(pct * 20)
        bar = f"[{'=' * bar_len}{' ' * (20 - bar_len)}]"

        if phase == "warmup":
            progress = f"[yellow]Warmup: {iteration}/{self._warmup}[/yellow]"
        else:
            progress = f"[green]Iter: {effective_iter}/{total_bench}[/green]"

        throughput = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        adapter_avg, adapter_now = _avg_now(util_samples)
        cpu_avg, cpu_now = _avg_now(cpu_samples, fallback_now=cpu_pct)
        gpu_avg, gpu_now = _avg_now(gpu_samples, fallback_now=gpu_pct)

        # Row 1: Progress
        pct_cell = f"{bar} {pct:.0%}"
        row1 = f"  {pct_cell:<30}|  {progress}  |  Device: {self._device}"

        # Row 2: Compute (unified now/avg format across all three)
        adapter_cell = f"{self._adapter_label}: {adapter_now:.1f}%/{adapter_avg:.1f}%"
        cpu_cell = f"CPU: {cpu_now:.1f}%/{cpu_avg:.1f}%"
        gpu_cell = f"GPU: {gpu_now:.1f}%/{gpu_avg:.1f}%"
        row2 = f"  {adapter_cell:<20}| {cpu_cell:<20}| {gpu_cell:<20}"

        # Row 3: Memory
        ram_cell = f"Sys Mem: {ram_mb:.0f} MB"
        mem_cell = f"Device Mem: {memory_local_mb:.0f}/{memory_shared_mb:.0f} MB (local/shared)"
        row3 = f"  {ram_cell:<24}|  {mem_cell}"

        # Row 4: Inference
        lat_cell = f"Latency: {latency_ms:.2f} ms"
        thr_cell = f"Throughput: ~{throughput:.0f} smp/s"
        row4 = f"  {lat_cell:<24}|  {thr_cell}"

        return f"{row1}\n{row2}\n{row3}\n{row4}"

    def print_final_snapshot(
        self,
        util_samples: list[float],
        memory_mb: float,
        latency_ms: float,
        hw_dict: dict[str, Any],
        cpu_samples: list[float] | None = None,
    ) -> None:
        """No-op: Rich Live with transient=False keeps the last frame visible.

        The last rendered panel from update() persists in terminal scrollback
        automatically, so no separate snapshot is needed.
        """
