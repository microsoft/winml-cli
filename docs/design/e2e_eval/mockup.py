# ruff: noqa
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Mockup script for wmk eval CLI output.

Renders the proposed eval command UI using Rich, with simulated
progress and fake metrics. Run to see what the real command will look like.

Usage:
    uv run python temp/eval_mockup.py
    uv run python temp/eval_mockup.py --quiet
"""

from __future__ import annotations

import time

import click
from rich.console import Console
from rich.table import Table


console = Console(stderr=True)


def _render_header(
    model: str,
    device: str,
    task: str,
    dataset: str,
    split: str,
    samples: int,
) -> None:
    console.print()
    console.print("[bold]" + "=" * 80 + "[/bold]")
    console.print("[bold]EVALUATION[/bold]")
    console.print("[bold]" + "=" * 80 + "[/bold]")
    console.print(f"   Model:    [bold cyan]{model}[/bold cyan]")
    console.print(f"   Device:   [green]{device}[/green]")
    console.print(f"   Task:     {task}")
    console.print(f"   Dataset:  {dataset} ({split})")
    console.print(f"   Samples:  {samples:,}")
    console.print()


def _render_progress(total: int) -> None:
    """Simulate live progress bar."""
    from rich.live import Live
    from rich.text import Text

    console.print("-" * 80)

    with Live(console=console, refresh_per_second=15, transient=True) as live:
        for i in range(1, total + 1):
            pct = i / total
            bar_len = int(pct * 40)
            bar = f"[{'=' * bar_len}{' ' * (40 - bar_len)}]"

            # Simulated metrics
            latency = 8.32 + (0.5 if i % 7 == 0 else 0)
            throughput = 1000.0 / latency
            eta = (total - i) * latency / 1000.0

            line = Text.from_markup(
                f"  Evaluating...  {bar}  {i}/{total}  {pct:.0%}\n"
                f"  Latency: {latency:.2f} ms  |  ~{throughput:.0f} smp/s"
                f"  |  ETA: {eta:.1f}s"
            )
            live.update(line)
            time.sleep(0.01)

    console.print("-" * 80)
    console.print()


def _render_results(
    task: str,
    accuracy: float,
    f1: float | None,
    samples: int,
    total_time: float,
) -> None:
    console.print("[bold]" + "=" * 80 + "[/bold]")
    console.print("[bold]RESULTS[/bold]")
    console.print("[bold]" + "=" * 80 + "[/bold]")
    console.print()

    # Metrics
    console.print(f"  [bold]Accuracy:[/bold]    {accuracy:.2f}%")
    if f1 is not None:
        console.print(f"  [bold]F1:[/bold]          {f1:.2f}%")
    console.print()

    # Latency table
    console.print("  [bold]Latency (ms)[/bold]")
    table = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
    for col in ["Avg", "P50", "P90", "P95", "P99", "Std"]:
        table.add_column(col, justify="right")
    table.add_row("8.32", "7.95", "10.21", "11.44", "14.02", "1.87")
    console.print(table)
    console.print()

    # Throughput
    throughput = samples / total_time
    console.print(f"  [bold]Throughput:[/bold]  {throughput:.2f} samples/sec")
    console.print(f"  [bold]Total time:[/bold]  {total_time:.2f}s ({samples:,} samples)")
    console.print()


def _render_hardware() -> None:
    console.print("  [bold]Hardware[/bold]")
    console.print("  NPU: 67.3% avg, 89.1% peak  |  CPU: 12.4% avg")
    console.print("  Device Mem: 245/128 MB (local/shared)  |  Sys Mem: 8,432 MB")
    console.print()


@click.command()
@click.option("--quiet", "-q", is_flag=True, help="JSON-only output")
@click.option("--task", default="image-classification")
@click.option("--no-monitor", is_flag=True, help="Skip hardware section")
def main(quiet: bool, task: str, no_monitor: bool) -> None:
    """Render eval command mockup."""
    model = "microsoft/resnet-50"
    dataset = "timm/mini-imagenet"
    samples = 1000

    if quiet:
        import json

        print(
            json.dumps(
                {
                    "model": model,
                    "task": task,
                    "device": "npu",
                    "dataset": dataset,
                    "samples": samples,
                    "metrics": {"accuracy": 0.7842},
                    "latency_ms": {"mean": 8.32, "p50": 7.95, "p90": 10.21},
                    "throughput_sps": 120.19,
                }
            )
        )
        return

    f1 = 91.05 if task == "text-classification" else None
    accuracy = 91.28 if task == "text-classification" else 78.42

    _render_header(model, "npu", task, dataset, "test", samples)
    _render_progress(samples)
    _render_results(task, accuracy, f1, samples, total_time=8.32)
    if not no_monitor:
        _render_hardware()

    console.print(f"  Results saved to: temp/eval_{model.split('/')[-1]}_20260322.json")
    console.print()


if __name__ == "__main__":
    main()
