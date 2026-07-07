# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared generation-benchmark result and display helpers.

Both the WinML composite-model path (``perf.py``) and the
``onnxruntime-genai`` path (``_perf_genai.py``) report the same LLM-style
metrics — TTFT, prefill, decode throughput, TPOT, and total generation time.
This module provides the shared :class:`GenerationBenchmarkResult` dataclass
and :func:`display_generation_report` so neither module duplicates the
structure or display logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from rich.console import Console


@dataclass
class GenerationBenchmarkResult:
    """Aggregated results from a generation benchmark.

    Covers both WinML composite-model generation (``perf.py``) and
    onnxruntime-genai bundle generation (``_perf_genai.py``).  Fields that
    only apply to one path default to ``0.0`` / ``None`` when unused.
    """

    # --- Display / provenance info (populated by the caller) -----------------
    runtime: str = ""
    model_label: str = ""
    device: str = ""
    ep: str | None = None
    prompt: str = ""
    max_new_tokens: int = 0
    warmup: int = 0
    iterations: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # --- Generation shape ----------------------------------------------------
    prompt_tokens: int = 0
    generated_tokens: int = 0
    context_length: int | None = None

    # --- Time to first token (prefill + first decode), milliseconds ----------
    ttft_mean_ms: float = 0.0
    ttft_min_ms: float = 0.0
    ttft_max_ms: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p90_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    ttft_p99_ms: float = 0.0

    # --- Prefill / prompt-processing phase, milliseconds ---------------------
    prefill_mean_ms: float = 0.0

    # --- Decode phase --------------------------------------------------------
    decode_tokens_per_sec: float = 0.0
    avg_token_latency_ms: float = 0.0
    tpot_mean_ms: float = 0.0

    # --- Whole generation (prefill + all decode), milliseconds ----------------
    total_generation_mean_ms: float = 0.0

    # --- Per-iteration raw samples (warmup excluded) -------------------------
    raw_ttft_ms: list[float] = field(default_factory=list)
    raw_prefill_ms: list[float] = field(default_factory=list)
    raw_decode_tokens_per_sec: list[float] = field(default_factory=list)
    raw_tpot_ms: list[float] = field(default_factory=list)
    raw_total_ms: list[float] = field(default_factory=list)

    # Runtime-specific info merged into ``benchmark_info`` in ``to_dict()``.
    # E.g. genai adds ``compile``, ``apply_template``, ``bundle_dir``.
    extra_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        info: dict[str, Any] = {
            "runtime": self.runtime,
            "model": self.model_label,
            "device": self.device,
            "ep": self.ep,
            "prompt": self.prompt,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "context_length": self.context_length,
            "max_new_tokens": self.max_new_tokens,
            "iterations": self.iterations,
            "warmup": self.warmup,
            "timestamp": self.timestamp,
        }
        info.update(self.extra_info)
        return {
            "benchmark_info": info,
            "ttft_ms": {
                "mean": round(self.ttft_mean_ms, 3),
                "min": round(self.ttft_min_ms, 3),
                "max": round(self.ttft_max_ms, 3),
                "p50": round(self.ttft_p50_ms, 3),
                "p90": round(self.ttft_p90_ms, 3),
                "p95": round(self.ttft_p95_ms, 3),
                "p99": round(self.ttft_p99_ms, 3),
            },
            "prefill_ms": {"mean": round(self.prefill_mean_ms, 3)},
            "decode": {
                "tokens_per_sec": round(self.decode_tokens_per_sec, 2),
                "avg_token_latency_ms": round(self.avg_token_latency_ms, 3),
                "tpot_ms": round(self.tpot_mean_ms, 3),
            },
            "total_generation_ms": {
                "mean": round(self.total_generation_mean_ms, 3),
            },
            "raw": {
                "ttft_ms": [round(v, 3) for v in self.raw_ttft_ms],
                "prefill_ms": [round(v, 3) for v in self.raw_prefill_ms],
                "decode_tokens_per_sec": [round(v, 2) for v in self.raw_decode_tokens_per_sec],
                "tpot_ms": [round(v, 3) for v in self.raw_tpot_ms],
                "total_ms": [round(v, 3) for v in self.raw_total_ms],
            },
        }


def display_generation_report(
    result: GenerationBenchmarkResult,
    console: Console,
) -> None:
    """Render a generation benchmark report to the console.

    Works for both WinML composite and genai bundle results — the layout
    adapts based on which optional fields are populated.
    """
    from rich.table import Table

    console.print()
    console.print(f"[dim]Runtime:[/dim]   {result.runtime}")
    console.print(f"[dim]Model:[/dim]     {result.model_label}")
    if result.device:
        device_str = (
            result.device
            if not result.ep or result.device == result.ep
            else f"{result.device} ({result.ep})"
        )
        console.print(f"[dim]Device:[/dim]    {device_str}")
    console.print(
        f"[dim]Prompt:[/dim]    {result.prompt_tokens} tokens   "
        f"[dim]Generated:[/dim] {result.generated_tokens} tokens "
        f"(max_new_tokens={result.max_new_tokens})"
    )

    console.print()
    console.print("[bold]Time to first token (ms)[/bold]")
    table = Table(show_header=True, header_style="bold cyan")
    for col in ["Avg", "P50", "P90", "P95", "P99", "Min", "Max"]:
        table.add_column(col, justify="right")
    table.add_row(
        f"{result.ttft_mean_ms:.2f}",
        f"{result.ttft_p50_ms:.2f}",
        f"{result.ttft_p90_ms:.2f}",
        f"{result.ttft_p95_ms:.2f}",
        f"{result.ttft_p99_ms:.2f}",
        f"{result.ttft_min_ms:.2f}",
        f"{result.ttft_max_ms:.2f}",
    )
    console.print(table)

    console.print()
    if result.prefill_mean_ms > 0:
        console.print(
            f"[bold]Prefill:[/bold]   {result.prefill_mean_ms:.2f} ms avg (prompt processing)"
        )
    console.print(
        f"[bold]Decode:[/bold]    "
        f"{result.decode_tokens_per_sec:.2f} tokens/sec  |  "
        f"{result.tpot_mean_ms:.2f} ms/token (TPOT)"
    )
    console.print(
        f"[bold]Total:[/bold]     {result.total_generation_mean_ms:.2f} ms avg per generation"
    )
    if result.warmup > 0:
        console.print(
            f"  [dim]Excluded first {result.warmup} warmup generation(s) from statistics[/dim]"
        )
    console.print()
