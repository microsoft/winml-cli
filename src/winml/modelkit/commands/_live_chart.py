# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Live hardware monitor display for performance benchmarking.

Renders a live NPU/CPU utilization chart during benchmarking using
plotext for chart rendering and Rich Live for terminal refresh.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.panel import Panel

from ..session.monitor.hw_monitor import adapter_label


# Moving window size for the x-axis (seconds)
_CHART_WINDOW_SECONDS = 10.0

# Display refresh rate (frames per second)
_REFRESH_FPS = 5


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
        chart_width: int = 80,
        chart_height: int = 15,
        poll_interval_ms: int = 100,
        device_kind: str | None = None,
        duration_sec: float | None = None,
    ) -> None:
        self._total = total_iterations
        self._warmup = warmup
        self._model_id = model_id
        self._device = device
        # When set, the benchmark phase runs on a wall-clock budget instead of a
        # fixed iteration count, so progress is reported as elapsed/total time.
        # ``_bench_start`` is stamped on the first benchmark-phase update().
        self._duration_sec = duration_sec
        self._bench_start: float | None = None
        # `device_kind` is the value HWMonitor resolved at start() — pass it
        # in when you want the legend to reflect what's actually polled (e.g.
        # "auto" that resolved to GPU). Falls back to the requested string
        # when the caller doesn't know the resolved kind yet.
        if device_kind is None:
            requested = (device or "").lower()
            device_kind = requested if requested in ("npu", "gpu") else None
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
    ) -> None:
        """Update the live display with current metrics."""
        if self._live is None:
            return

        # Stamp the start of the timed benchmark phase so duration-based progress
        # is measured from the first post-warmup iteration.
        if (
            self._duration_sec is not None
            and self._bench_start is None
            and iteration > self._warmup
        ):
            self._bench_start = time.perf_counter()

        try:
            chart_renderable = self._render_chart(util_samples, cpu_samples)
            status_line = self._render_status(
                iteration,
                latency_ms,
                util_samples,
                memory_local_mb,
                memory_shared_mb,
                cpu_pct,
                ram_mb,
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
        self, util_samples: list[float], cpu_samples: list[float] | None = None
    ) -> Any:
        """Render utilization chart as a Rich renderable.

        Uses plotext with AnsiDecoder for flicker-free Rich Live integration.
        Plots adapter (NPU/GPU, green) and CPU (cyan) with distinct colors.
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
        total_npu = len(util_samples) if util_samples else 0

        # Plot the adapter line only when an adapter is actually being polled.
        if show_adapter:
            npu_window = util_samples[-window_samples:] if util_samples else [0]
            window_start_idx = max(0, total_npu - len(npu_window))
            npu_times = [
                (window_start_idx + i) * self._poll_interval_s for i in range(len(npu_window))
            ]
            plt.plot(npu_times, npu_window, marker="braille", color="green")

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

        # No plotext title -- we render our own Rich-colored title with legend
        plt.ylabel("Usage %")

        # Fixed y-axis: 0 to 100 with ticks at 0, 20, 40, 60, 80, 100
        plt.ylim(0, 100)
        plt.yticks([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])

        # X-axis: absolute elapsed time, sliding window. Use whichever series
        # we have to anchor the timeline so a CPU-only chart still scrolls.
        sample_count = total_npu if show_adapter else total_cpu
        elapsed = sample_count * self._poll_interval_s
        x_min = max(0.0, elapsed - _CHART_WINDOW_SECONDS)
        x_max = max(elapsed, _CHART_WINDOW_SECONDS)
        plt.xlim(x_min, x_max)
        plt.xlabel("Time (s)")

        plt.plotsize(self._chart_width, self._chart_height)

        from rich.console import Group
        from rich.text import Text

        # Rich-colored title line with legend swatches.
        if show_adapter and has_cpu:
            title = Text.from_markup(
                f"  Utilization ([green]\u2588\u2588[/green] {adapter} %  "
                f"[cyan]\u2588\u2588[/cyan] CPU %)"
            )
        elif show_adapter:
            title = Text.from_markup(f"  Utilization ([green]\u2588\u2588[/green] {adapter} %)")
        else:
            title = Text.from_markup("  Utilization ([cyan]\u2588\u2588[/cyan] CPU %)")

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
    ) -> str:
        """Render 3-row status below the chart."""
        phase = "warmup" if iteration <= self._warmup else "benchmark"
        effective_iter = iteration - self._warmup if phase == "benchmark" else iteration
        total_bench = self._total - self._warmup

        current_util = util_samples[-1] if util_samples else 0.0
        mean_util = sum(util_samples) / len(util_samples) if util_samples else 0.0

        if self._duration_sec is not None and phase == "benchmark":
            # Duration mode: base progress on elapsed wall-clock time, since the
            # benchmark iteration count is not known ahead of time.
            elapsed = time.perf_counter() - self._bench_start if self._bench_start else 0.0
            pct = min(elapsed / self._duration_sec, 1.0) if self._duration_sec > 0 else 0.0
            shown = min(elapsed, self._duration_sec)
            progress = f"[green]Time: {shown:.1f}/{self._duration_sec:.0f}s[/green]"
        else:
            pct = iteration / self._total if self._total > 0 else 0
            if phase == "warmup":
                progress = f"[yellow]Warmup: {iteration}/{self._warmup}[/yellow]"
            else:
                progress = f"[green]Iter: {effective_iter}/{total_bench}[/green]"

        bar_len = int(pct * 20)
        bar = f"[{'=' * bar_len}{' ' * (20 - bar_len)}]"

        throughput = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        # Row 1: Progress
        pct_cell = f"{bar} {pct:.0%}"
        row1 = f"  {pct_cell:<30}|  {progress}  |  Device: {self._device}"

        # Row 2: Hardware (pad each cell to fixed width, spaces before divider).
        # CPU-only mode drops the adapter cell + device-memory cell since we
        # have no live values to populate them with.
        cpu_cell = f"CPU: {cpu_pct:.1f}%"
        ram_cell = f"RAM: {ram_mb:.0f} MB"
        if self._show_adapter:
            adapter_cell = f"{self._adapter_label}: {mean_util:.1f}% avg ({current_util:.1f}% now)"
            mem_cell = f"VRAM: {memory_local_mb:.0f}/{memory_shared_mb:.0f} MB (local/shared)"
            row2 = f"  {adapter_cell:<30}| {cpu_cell:<12}|  {ram_cell}  |  {mem_cell}"
        else:
            row2 = f"  {cpu_cell:<12}|  {ram_cell}"

        # Row 3: Inference (pad each cell to fixed width, spaces before divider)
        lat_cell = f"Latency: {latency_ms:.2f} ms"
        thr_cell = f"Throughput: ~{throughput:.0f} smp/s"
        row3 = f"  {lat_cell:<24}|  {thr_cell}"

        return f"{row1}\n{row2}\n{row3}"

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
