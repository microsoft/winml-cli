# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""HWLiveDisplay — self-contained live hardware monitor with terminal chart.

Wraps HWMonitor + plotext chart + Rich Live in a single context manager.
No iteration counter or latency tracking — just hardware utilization.

Usage::

    from winml.modelkit.session.monitor.live_display import HWLiveDisplay

    with HWLiveDisplay(title="resnet-50 eval"):
        # ... any long-running work ...
        results = evaluator.compute(...)

    # Terminal shows live NPU/CPU chart during the block,
    # final snapshot persists in scrollback after exit.
"""

from __future__ import annotations

import threading
from typing import Any

from .hw_monitor import HWMonitor, adapter_label


# Chart settings
_CHART_WINDOW_SECONDS = 10.0
_REFRESH_FPS = 5
_DEFAULT_CHART_WIDTH = 72
_DEFAULT_CHART_HEIGHT = 12


class HWLiveDisplay:
    """Self-updating live hardware utilization chart.

    Combines HWMonitor (background PDH polling) with a Rich Live display
    that auto-refreshes the chart in a separate thread. The caller just
    enters the context and does their work — no manual update() calls.
    """

    def __init__(
        self,
        title: str = "HW Monitor",
        poll_interval_ms: int = 200,
        chart_width: int = _DEFAULT_CHART_WIDTH,
        chart_height: int = _DEFAULT_CHART_HEIGHT,
        device: str = "auto",
    ) -> None:
        self._title = title
        self._poll_interval_s = poll_interval_ms / 1000.0
        self._chart_width = chart_width
        self._chart_height = chart_height
        self._hw = HWMonitor(poll_interval_ms=poll_interval_ms, device=device)
        self._live: Any = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> HWLiveDisplay:
        """Start HW monitoring and live display."""
        from rich.console import Console
        from rich.live import Live

        self._hw.__enter__()
        self._live = Live(
            refresh_per_second=_REFRESH_FPS,
            console=Console(stderr=True),
            transient=False,
        )
        self._live.__enter__()

        # Background thread drives display updates
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name="hw-live-display",
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        """Stop display thread, then stop HW monitor."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._live:
            self._live.__exit__(*exc)
        self._hw.__exit__(*exc)

    def _update_loop(self) -> None:
        """Background loop: read HWMonitor samples → render chart → update Live."""
        interval = 1.0 / _REFRESH_FPS
        while not self._stop_event.is_set():
            try:
                self._render_once()
            except Exception:
                pass  # Don't let display errors kill the thread
            self._stop_event.wait(interval)

    def _render_once(self) -> None:
        """Single render cycle."""
        if self._live is None:
            return

        from rich.panel import Panel

        adapter_samples = self._hw.utilization_samples
        cpu_samples = self._hw.cpu_samples

        chart = self._render_chart(adapter_samples, cpu_samples)
        status = self._render_status(adapter_samples, cpu_samples)

        from rich.console import Group
        from rich.text import Text

        panel = Panel(
            Group(chart, Text.from_markup(status)),
            title=f"[bold]{self._title}[/bold]",
            border_style="blue",
        )
        self._live.update(panel)

    @property
    def _adapter_label(self) -> str:
        """Live label that follows the resolved adapter (after start())."""
        return adapter_label(self._hw.device_kind)

    @property
    def _show_adapter(self) -> bool:
        """True iff HWMonitor resolved an actual NPU/GPU adapter to poll."""
        return self._hw.device_kind in ("npu", "gpu")

    def _render_chart(
        self,
        adapter_samples: list[float],
        cpu_samples: list[float],
    ) -> Any:
        """Render adapter (NPU/GPU) and CPU utilization chart via plotext."""
        adapter = self._adapter_label
        show_adapter = self._show_adapter
        try:
            import plotext as plt
        except ImportError:
            from rich.text import Text

            if not show_adapter:
                current = cpu_samples[-1] if cpu_samples else 0.0
                bar_len = min(50, max(0, int(current / 2)))
                bar = "#" * bar_len + "." * (50 - bar_len)
                return Text(f"  CPU: [{bar}] {current:.1f}%")
            current = adapter_samples[-1] if adapter_samples else 0.0
            bar_len = min(50, max(0, int(current / 2)))
            bar = "#" * bar_len + "." * (50 - bar_len)
            return Text(f"  {adapter}: [{bar}] {current:.1f}%")

        plt.clf()
        plt.theme("clear")

        window_samples = int(_CHART_WINDOW_SECONDS / self._poll_interval_s)

        # Adapter (green) \u2014 only when an adapter is actually polled.
        if show_adapter:
            adapter_window = adapter_samples[-window_samples:] if adapter_samples else [0]
            start_idx = max(0, len(adapter_samples) - len(adapter_window))
            adapter_times = [
                (start_idx + i) * self._poll_interval_s for i in range(len(adapter_window))
            ]
            plt.plot(adapter_times, adapter_window, marker="braille", color="green")

        # CPU (cyan)
        has_cpu = bool(cpu_samples)
        if has_cpu:
            cpu_window = cpu_samples[-window_samples:]
            cpu_start = max(0, len(cpu_samples) - len(cpu_window))
            cpu_times = [(cpu_start + i) * self._poll_interval_s for i in range(len(cpu_window))]
            plt.plot(cpu_times, cpu_window, marker="braille", color="cyan")

        plt.ylabel("Usage %")
        plt.ylim(0, 100)
        plt.yticks([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])

        # Anchor the timeline on whichever series we have.
        sample_count = len(adapter_samples) if show_adapter else len(cpu_samples)
        elapsed = sample_count * self._poll_interval_s
        x_min = max(0.0, elapsed - _CHART_WINDOW_SECONDS)
        x_max = max(elapsed, _CHART_WINDOW_SECONDS)
        plt.xlim(x_min, x_max)
        plt.xlabel("Time (s)")
        plt.plotsize(self._chart_width, self._chart_height)

        from rich.console import Group
        from rich.text import Text

        if show_adapter and has_cpu:
            legend = (
                f"  Utilization ([green]\u2588\u2588[/green] {adapter} %  "
                "[cyan]\u2588\u2588[/cyan] CPU %)"
            )
        elif show_adapter:
            legend = f"  Utilization ([green]\u2588\u2588[/green] {adapter} %)"
        else:
            legend = "  Utilization ([cyan]\u2588\u2588[/cyan] CPU %)"

        ansi_output = plt.build()
        chart_lines = [Text.from_ansi(line) for line in ansi_output.splitlines()]
        return Group(Text.from_markup(legend), *chart_lines)

    def _render_status(
        self,
        adapter_samples: list[float],
        cpu_samples: list[float],
    ) -> str:
        """Render hardware status line below the chart."""
        cpu_now = cpu_samples[-1] if cpu_samples else 0.0
        ram_mb = self._hw.ram_used_mb
        cpu_cell = f"CPU: {cpu_now:.1f}%"
        ram_cell = f"RAM: {ram_mb:.0f} MB"

        if not self._show_adapter:
            return f"  {cpu_cell:<12}| {ram_cell}"

        adapter_mean = sum(adapter_samples) / len(adapter_samples) if adapter_samples else 0.0
        adapter_now = adapter_samples[-1] if adapter_samples else 0.0
        mem_local = self._hw.peak_memory_local_mb
        mem_shared = self._hw.peak_memory_shared_mb

        adapter_cell = f"{self._adapter_label}: {adapter_mean:.1f}% avg ({adapter_now:.1f}% now)"
        mem_cell = f"VRAM: {mem_local:.0f}/{mem_shared:.0f} MB"

        return f"  {adapter_cell:<32}| {cpu_cell:<12}| {ram_cell:<16}| {mem_cell}"

    @property
    def hw(self) -> HWMonitor:
        """Access the underlying HWMonitor for metrics after exit."""
        return self._hw
