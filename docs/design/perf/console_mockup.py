# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Console writer mockup — full perf --monitor --iterations 1000 output.

Demonstrates:
1. Three- or four-phase render: pre-bench header, animated live monitor,
   post-bench summary, and (when --op-tracing is set) an op-tracing report.
2. Three-silicon utilization chart (NPU + CPU + GPU) — current LiveMonitorDisplay
   plots NPU+CPU only; GPU is forward-looking by this design.
3. Two-tone progress bar (warmup dim + measured green + pending light) — current
   bar is single-tone; this is a small UX improvement the mockup proposes.

Run examples:

    # Original 3-phase animation (no op-tracing):
    uv run python docs/design/perf/console_mockup.py

    # 3-phase animation + Phase 4 operator tracing:
    uv run python docs/design/perf/console_mockup.py --op-tracing basic
    uv run python docs/design/perf/console_mockup.py --op-tracing detail
    uv run python docs/design/perf/console_mockup.py --op-tracing basic --top-k 10
    uv run python docs/design/perf/console_mockup.py --op-tracing basic --iterations 1

    # --top-k without --op-tracing is rejected with exit 2.

Data Contracts (shared with modelkit/commands/perf.py + LiveMonitorDisplay)
==========================================================================

Contract A — On-disk <model>_perf.json schema:

    {
        "benchmark_info": {
            "model_id": str, "task": str, "device": str, "precision": str,
            "iterations": int, "warmup": int, "batch_size": int,
            "timestamp": str,                    # ISO-8601 UTC
        },
        "model_info": {
            "input_names": list[str],
            "input_shapes": list[list[int]],
            "input_types": list[str],
            "output_names": list[str],
            "output_shapes": list[list[int]],
        },
        "latency_ms": {
            "mean": float, "min": float, "max": float,
            "p50": float, "p90": float, "p95": float, "p99": float,
            "std": float, "warmup_mean": float,
        },
        "throughput": {
            "samples_per_sec": float, "batches_per_sec": float,
        },
        "raw_samples_ms": list[float],            # length == iterations (post-warmup)
        "hw_monitor": {
            "monitor": str,
            "npu_pct_avg": float, "npu_pct_peak": float,
            "cpu_pct_avg": float,
            "gpu_pct_avg": float,                 # NEW (forward-looking)
            "ram_mb_avg": float,
            "device_mem_local_mb_peak": float,
            "device_mem_shared_mb_peak": float,
        },
    }

Contract B — In-memory HW sample (consumed by the chart):

    {
        "t": float,             # elapsed seconds since benchmark start
        "npu_pct": float,       # 0..100
        "cpu_pct": float,       # 0..100
        "gpu_pct": float,       # 0..100  (NEW)
        "ram_mb": float,
        "mem_local_mb": float,  # NPU device memory (local)
        "mem_shared_mb": float, # NPU device memory (shared)
    }

Contract C — In-memory progress state (consumed by the status block):

    {
        "iteration": int,       # 1..total_iterations (includes warmup)
        "total": int,           # total_iterations (warmup + measured)
        "warmup": int,          # warmup iteration count
        "latency_ms": float,    # most recent measurement
    }

Contract-to-derivation invariant
--------------------------------
The displayed `latency_ms` dict in Phase 3 is *derived from* RAW_SAMPLES_MS via
compute_latency_stats(). The displayed `hw_monitor` dict is *derived from* the
HW sample lists via compute_hw_aggregates(). Reruns are deterministic
(np.random.seed(42)). The table on screen is therefore provably the right
summary of the displayed raw data, not just plausibly.

Contract D — Op-tracing per-instance schema:

    {
        "node_name": str,                # full ONNX node path
        "op_type": str,                  # e.g. "Conv2d", "MatMul", "Add"
        "sample_durations_us": list[float],   # per-iteration durations
        "dram_read_bytes": int | None,
        "vtcm_hit_ratio": float | None,  # 0..1
        # derived:
        "avg_us":   sum(samples) / len(samples)
        "total_us": sum(samples)
        "p90_us":   inclusive 90th percentile of samples (== avg when n==1)
    }

The op-tracing report is gated on --op-tracing. Sort key is avg_us
descending, with node_name as a deterministic tiebreaker. The "Cum %"
column is a running sum of per-instance percent-of-total down the table.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import plotext as plt
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ── Module constants (dispatch tables; same role as static_analyzer's COLORS) ─

SILICON_COLORS = {"npu": "green", "cpu": "cyan", "gpu": "magenta"}

CHART_HEIGHT = 15
CHART_WIDTH = 120
# 15s window keeps per-sample density (chart 80->120 cols x1.5, so 10->15s).
WINDOW_SECONDS = 15.0
REFRESH_FPS = 5  # matches LiveMonitorDisplay
PROGRESS_WIDTH = 20  # progress-bar width in chars
POLL_INTERVAL_S = 0.1  # HW poll cadence (matches LiveMonitorDisplay)


# ── Op-tracing constants ────────────────────────────────────────────────────

OP_TRACING_TOP_K_DEFAULT = 5
OP_TRACING_NUM_SAMPLES = 100  # fake per-instance sample count
OP_TRACING_NODE_NAME_MAX_WIDTH = 80
OP_TRACING_SEED = 42  # deterministic jitter for reproducible visuals


# ── Fake data ─────────────────────────────────────────────────────────────────

np.random.seed(42)

# Run config (Contract A.benchmark_info / model_info inputs)
MODEL_ID = "facebook/convnext-base-224"
IS_HF_MODEL = True  # toggle for the demo: HF model_id (True) vs direct .onnx (False)
CACHED_ONNX_PATH = (
    r"C:\Users\zhengte\.cache\winml\artifacts"
    r"\facebook_convnext-base-224\imgcls_db24bf8910f169d6_compiled.onnx"
)
# Op-tracing artifact destinations (mockup paths; production resolves these
# from --output-dir / cwd in commands/perf.py).
OP_TRACE_JSON_PATH = "./facebook_convnext-base-224_op_trace.json"
OP_TRACE_CSV_PATH = "./profiling_output.csv"
TASK = "image-classification"
DEVICE = "auto"
DEVICE_RESOLVED = "npu"  # what `auto` resolves to in this scenario
EP_RESOLVED = "QNN"  # what the resolved device dispatches to
PRECISION = "auto"
ITERATIONS = 1000
WARMUP = 10
BATCH_SIZE = 1
OPSET = 17
PRODUCER = "pytorch v2.1.0"

IO_CONFIG = {
    "input_names": ["pixel_values"],
    "input_shapes": [[1, 3, 224, 224]],
    "input_types": ["float32"],
    "output_names": ["logits"],
    "output_shapes": [[1, 1000]],
}

# Latency: cold-cache compile cost on first warmup iter, settling fast
WARMUP_SAMPLES_MS: list[float] = [25.1, 8.2, 5.4, 4.5, 4.1, 3.95, 3.92, 3.93, 3.91, 3.91]

# Measured: tight distribution centred on 3.16 ms with std 0.09
RAW_SAMPLES_MS: list[float] = (3.16 + 0.09 * np.random.randn(ITERATIONS)).round(3).tolist()


def _hw_curve(steady_pct: float, ramp_start_pct: float, jitter: float, n: int = 30) -> list[float]:
    """Generate an HW utilization sample list: cold-start ramp, then jitter around steady.

    First 3 samples linearly ramp from ramp_start_pct to steady_pct (cold-cache
    warm-up). Remaining samples oscillate around steady_pct with Gaussian jitter.
    """
    ramp = np.linspace(ramp_start_pct, steady_pct, 3)
    rest = steady_pct + jitter * np.random.randn(n - 3)
    return np.clip(np.concatenate([ramp, rest]), 0.0, 100.0).round(1).tolist()


# 30 samples per silicon = 100ms poll * 30 = 3.0 sec total run time
NPU_SAMPLES_FULL: list[float] = _hw_curve(steady_pct=73.0, ramp_start_pct=8.0, jitter=8.0)
CPU_SAMPLES_FULL: list[float] = _hw_curve(steady_pct=31.0, ramp_start_pct=12.0, jitter=4.0)
GPU_SAMPLES_FULL: list[float] = _hw_curve(steady_pct=8.0, ramp_start_pct=2.0, jitter=3.0)

MEM_LOCAL_MB_FULL: list[float] = [95.0] * 30
MEM_SHARED_MB_FULL: list[float] = [119.0] * 30
RAM_MB_FULL: list[float] = [round(52490.0 + 30.0 * np.random.randn()) for _ in range(30)]


# ── Op-tracing fake data ────────────────────────────────────────────────────


@dataclass
class FakeOp:
    """A single op instance with per-sample timing and hardware metrics.

    ``sample_durations_us`` is the source of truth.  ``avg_us``, ``total_us``,
    and ``p90_us`` are computed from it on demand.
    """

    node_name: str
    op_type: str
    sample_durations_us: list[float] = field(default_factory=list)
    dram_read_bytes: int | None = None
    vtcm_hit_ratio: float | None = None

    @property
    def sample_count(self) -> int:
        """Number of recorded sample durations."""
        return len(self.sample_durations_us)

    @property
    def avg_us(self) -> float:
        """Arithmetic mean of sample durations (microseconds)."""
        return statistics.fmean(self.sample_durations_us)

    @property
    def total_us(self) -> float:
        """Sum of all sample durations (microseconds)."""
        return float(sum(self.sample_durations_us))

    @property
    def p90_us(self) -> float:
        """90th percentile of sample durations (microseconds)."""
        # Inclusive percentile; degenerate to the single value when n == 1.
        if self.sample_count == 1:
            return self.sample_durations_us[0]
        deciles = statistics.quantiles(self.sample_durations_us, n=10, method="inclusive")
        return deciles[8]  # 9 cut points -> [8] is the 90th percentile


# Backbone: realistic ResNet-50-ish op mix (matches optrace_resnet50.csv shape).
# Each tuple: (node_name, op_type, base_avg_us, dram_read_bytes, vtcm_hit_ratio)
# fmt: off
_OP_TEMPLATES: list[tuple[str, str, float, int, float]] = [
    ("/resnet/encoder/stage.3/block.0/Conv_token_2",     "Conv2d",    1247.0, 18_432, 0.942),
    ("/resnet/encoder/stage.2/block.0/Conv_token_2",     "Conv2d",     982.0, 12_800, 0.917),
    ("/resnet/encoder/stage.1/attention/MatMul_0",       "MatMul",     873.0,  8_192, 0.971),
    ("/resnet/encoder/stage.0/block.0/Conv_token_2",     "Conv2d",     641.0,  6_344, 0.894),
    ("/resnet/encoder/stage.3/block.2/Add_2",            "Add",        420.0,  4_096, 0.990),
    ("/resnet/encoder/stage.2/block.1/Add_1",            "Add",        351.0,  3_584, 0.985),
    ("/resnet/encoder/stage.3/block.0/LayerNorm_0",      "LayerNorm",  298.0,  2_304, 0.998),
    ("/resnet/encoder/stage.0/block.0/Add_0",            "Add",        284.0,  2_048, 0.991),
    ("/resnet/encoder/stage.1/block.0/Conv_token_2",     "Conv2d",     256.0,  3_072, 0.880),
    ("/resnet/encoder/stage.2/attention/MatMul_0",       "MatMul",     231.0,  4_352, 0.964),
    ("/resnet/encoder/stage.0/block.1/Conv_token_1",     "Conv2d",     188.0,  2_560, 0.873),
    ("/resnet/encoder/stage.1/block.1/Conv_token_1",     "Conv2d",     162.0,  1_792, 0.901),
    ("/resnet/encoder/stage.3/block.1/Mul_0",            "Mul",        134.0,  1_024, 0.996),
    ("/resnet/embedder/embedder/Conv_token_0",           "Conv2d",     128.0,  4_608, 0.812),
    ("/resnet/encoder/stage.2/block.0/Softmax_0",        "Softmax",    119.0,    768, 0.983),
    ("/resnet/encoder/stage.1/block.0/Reshape_0",        "Reshape",     94.0,    512, 1.000),
    ("/resnet/encoder/stage.0/block.0/Relu_0",           "Relu",        82.0,      0, 1.000),
    ("/resnet/encoder/stage.3/Transpose_0",              "Transpose",   71.0,    256, 1.000),
    ("/resnet/encoder/stage.2/block.1/Gemm_0",           "Gemm",        63.0,  1_024, 0.945),
    ("/resnet/encoder/stage.1/block.1/Sigmoid_0",        "Sigmoid",     54.0,    128, 1.000),
]
# fmt: on


def generate_fake_ops(num_samples: int, seed: int = OP_TRACING_SEED) -> list[FakeOp]:
    """Build per-sample durations around each template's base avg.

    Uses lognormal multiplicative jitter so p90 sits noticeably above avg,
    matching NPU jitter in practice (occasional spikes from cache misses
    or thermal throttling).
    """
    rng = random.Random(seed)
    ops: list[FakeOp] = []

    for node_name, op_type, base_us, dram_r, vtcm in _OP_TEMPLATES:
        samples: list[float] = []
        for _ in range(num_samples):
            jitter = rng.lognormvariate(mu=0.0, sigma=0.08)
            # 5% chance of a moderate spike (1.15x to 1.45x).
            if rng.random() < 0.05:
                jitter *= rng.uniform(1.15, 1.45)
            samples.append(base_us * jitter)

        ops.append(
            FakeOp(
                node_name=node_name,
                op_type=op_type,
                sample_durations_us=samples,
                dram_read_bytes=dram_r,
                vtcm_hit_ratio=vtcm,
            )
        )

    return ops


# ── Op-tracing formatters (mirror modelkit/session/monitor/report.py) ───────


def _format_number(n: float | int | None) -> str:
    """Format a number with comma separators; one decimal for floats."""
    if n is None:
        return "-"
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{n:,}"


def _format_bytes(n: int | float | None) -> str:
    """Format a byte count to a human-readable string."""
    if n is None or n == 0:
        return "0"
    value: float = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024:
            if unit == "B" and value == int(value):
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _truncate_node_name(name: str, max_width: int = OP_TRACING_NODE_NAME_MAX_WIDTH) -> str:
    """Leading-ellipsis truncation; preserves the trailing op suffix.

    Keeps the right-hand portion of the node path visible (e.g. the actual
    operator name like ``Conv_token_2``) at the cost of dropping the leading
    namespace prefix. Better fit for top-K reports where the operator
    identity matters more than the module hierarchy.
    """
    if max_width <= 0:
        return ""
    if len(name) <= max_width:
        return name
    if max_width == 1:
        return "…"
    return "…" + name[-(max_width - 1) :]


def _build_op_tracing_summary_line(top_ops: list[FakeOp], all_ops: list[FakeOp], k: int) -> str:
    """Render the 'Top K instances account for X%' summary string.

    Switches phrasing to 'All N instances account for 100.0%' when ``k``
    covers the full operator set.
    """
    sum_top = sum(op.total_us for op in top_ops)
    sum_all = sum(op.total_us for op in all_ops)
    pct = (sum_top / sum_all * 100.0) if sum_all > 0 else 0.0

    if len(top_ops) >= len(all_ops):
        prefix = f"All {len(all_ops)} instances"
    else:
        prefix = f"Top {k} instances"

    return (
        f"{prefix} account for [bold]{pct:.1f}%[/bold] of execute time "
        f"([bold]{_format_number(sum_top)}[/bold] / "
        f"{_format_number(sum_all)} μs accumulated)"
    )


def render_op_tracing(
    console: Console,
    ops: list[FakeOp],
    *,
    level: str,
    top_k: int,
    num_samples: int,
    json_path: str = "",
    csv_path: str = "",
) -> None:
    """Render the operator tracing section (basic or detail mode).

    Layout:
      1. Section rule: '── Operator Tracing (basic|detail, N samples) ──'
      2. Rich Table titled 'Top K Operator Instances by Avg Duration  (timings in μs)'
         - Basic columns:  Node | Type | p90 | % Tot
         - Detail columns: # | Node | Type | Avg | Total | % Tot | Cum % | p90 | DRAM(R) | VTCM Hit
      3. Summary line: 'Top K instances account for X% of execute time (...)'
      4. Mode-specific footer:
         - Basic:  HVX threads, Accel execute, Samples
         - Detail: HVX threads, Inference, Execute, Utilization (line 1)
                   DRAM read total, VTCM read total, Peak VTCM alloc (line 2)
      5. If num_samples == 1, an italic note explaining degenerate p90.
    """
    console.print()
    console.rule(f"[bold]Op-Tracing ({level}, {num_samples} samples)[/bold]")

    if not ops:
        console.print(
            "[yellow]Warning:[/yellow] No operator data available; "
            "profiling artifacts may be missing."
        )
        return

    # Sort by avg desc; tiebreak on node_name for determinism.
    ops_sorted = sorted(ops, key=lambda o: (-o.avg_us, o.node_name))
    k = min(top_k, len(ops_sorted))
    top_ops = ops_sorted[:k]
    sum_all_total = sum(o.total_us for o in ops_sorted)

    # Both modes auto-fit total width from column sums. Basic mode achieves
    # its target 120-cell envelope by forcing Node to 80 + fixed metrics
    # (12+9+6) + borders (~13). Detail mode lets Node auto-size (32-80) and
    # natural-fits all 10 columns.
    table = Table(
        title=f"Top {k} Operator Instances by Avg Duration  (timings in μs)",
        show_lines=False,
    )
    if level == "basic":
        # Slim 4-column scan view: Node + Type qualifier + the two metrics
        # that matter. Drops #, Avg, Total, Cum % which add noise without
        # changing the user's decision at a glance.
        # Node is fixed at 80 cells (truncate-left if name exceeds);
        # Type/p90/% Tot are fixed at 12/9/6. Total table = 80+12+9+6 + 13
        # borders/padding = 120 cells, matching the live-monitor chart.
        table.add_column(
            "Node",
            min_width=OP_TRACING_NODE_NAME_MAX_WIDTH,
            max_width=OP_TRACING_NODE_NAME_MAX_WIDTH,
            no_wrap=True,
            overflow="ellipsis",
        )
        table.add_column("Type", width=12, no_wrap=True)
        table.add_column("p90", justify="right", width=9)
        table.add_column("% Tot", justify="right", width=6)
    else:  # detail — full investigation view
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column(
            "Node",
            min_width=32,
            max_width=OP_TRACING_NODE_NAME_MAX_WIDTH,
            no_wrap=True,
            overflow="ellipsis",
        )
        table.add_column("Type", min_width=9, no_wrap=True)
        table.add_column("Avg", justify="right", width=9)
        table.add_column("Total", justify="right", width=10)
        table.add_column("% Tot", justify="right", width=6)
        table.add_column("Cum %", justify="right", width=6)
        table.add_column("p90", justify="right", width=9)
        table.add_column("DRAM(R)", justify="right", width=8)
        table.add_column("VTCM Hit", justify="right", width=8)

    cum_total_us = 0.0
    for i, op in enumerate(top_ops, 1):
        cum_total_us += op.total_us
        pct_total = (op.total_us / sum_all_total * 100.0) if sum_all_total > 0 else 0.0
        cum_pct = (cum_total_us / sum_all_total * 100.0) if sum_all_total > 0 else 0.0

        if level == "basic":
            row = [
                _truncate_node_name(op.node_name),
                op.op_type,
                _format_number(op.p90_us),
                f"{pct_total:.1f}%",
            ]
        else:  # detail
            vtcm_str = f"{op.vtcm_hit_ratio * 100:.1f}%" if op.vtcm_hit_ratio is not None else "-"
            row = [
                str(i),
                _truncate_node_name(op.node_name),
                op.op_type,
                _format_number(op.avg_us),
                _format_number(op.total_us),
                f"{pct_total:.1f}%",
                f"{cum_pct:.1f}%",
                _format_number(op.p90_us),
                _format_bytes(op.dram_read_bytes),
                vtcm_str,
            ]

        table.add_row(*row)

    console.print(table)
    console.print(_build_op_tracing_summary_line(top_ops, ops_sorted, k))

    # Mode-specific footer
    if level == "basic":
        accel_us = int(sum_all_total)
        console.print(
            f"[dim]HVX threads: 4   Accel execute: {_format_number(accel_us)} μs"
            f"   Samples: {num_samples}[/dim]"
        )
    else:  # detail
        total_dram_r = sum((op.dram_read_bytes or 0) for op in ops_sorted)
        # VTCM read ~= 8x DRAM read approximates typical NPU residency win.
        total_vtcm_r = total_dram_r * 8
        peak_vtcm = 1_843_200  # 1.8 MB
        execute_us = int(sum_all_total)
        inference_us = int(execute_us * 1.009)  # ~1% session overhead
        utilization = 91.3
        console.print(
            f"[dim]HVX threads: 4   Inference: {_format_number(inference_us)} μs"
            f"   Execute: {_format_number(execute_us)} μs"
            f"   Utilization: {utilization}%[/dim]"
        )
        console.print(
            f"[dim]DRAM read total: {_format_bytes(total_dram_r)}"
            f"   VTCM read total: {_format_bytes(total_vtcm_r)}"
            f"   Peak VTCM alloc: {_format_bytes(peak_vtcm)}[/dim]"
        )

    if num_samples == 1:
        console.print()
        console.print(
            "[dim]Note: p90 reflects single-sample data; "
            "increase --iterations for meaningful percentiles.[/dim]"
        )

    # Saved-to footer — tells the user where the op-trace artifacts went,
    # mirroring the Phase 3 perf.json save footer pattern.
    if json_path or csv_path:
        console.print()
        if json_path:
            console.print(Text.from_markup(f"  📁 Op-trace JSON: [cyan]{json_path}[/cyan]"))
        if csv_path:
            console.print(Text.from_markup(f"  📁 Profiling CSV: [cyan]{csv_path}[/cyan]"))


# ── Stat derivation helpers (the contract-binding layer) ──────────────────────


def compute_latency_stats(
    raw_samples_ms: list[float], warmup_samples_ms: list[float]
) -> dict[str, float]:
    """Derive Contract A.latency_ms from raw samples.

    All percentiles use np.percentile's linear interpolation (default).
    """
    arr = np.asarray(raw_samples_ms)
    return {
        "mean": float(round(arr.mean(), 3)),
        "min": float(round(arr.min(), 3)),
        "max": float(round(arr.max(), 3)),
        "p50": float(round(np.percentile(arr, 50), 3)),
        "p90": float(round(np.percentile(arr, 90), 3)),
        "p95": float(round(np.percentile(arr, 95), 3)),
        "p99": float(round(np.percentile(arr, 99), 3)),
        "std": float(round(arr.std(ddof=1), 3)),
        "warmup_mean": float(round(np.mean(warmup_samples_ms), 3)),
    }


def compute_hw_aggregates(
    npu_samples: list[float],
    cpu_samples: list[float],
    gpu_samples: list[float],
    ram_mb: list[float],
    mem_local_mb: list[float],
    mem_shared_mb: list[float],
) -> dict[str, float]:
    """Derive Contract A.hw_monitor block from per-silicon sample lists."""
    return {
        "npu_pct_avg": float(round(np.mean(npu_samples), 1)),
        "npu_pct_peak": float(round(np.max(npu_samples), 1)),
        "cpu_pct_avg": float(round(np.mean(cpu_samples), 1)),
        "gpu_pct_avg": float(round(np.mean(gpu_samples), 1)),
        "ram_mb_avg": float(round(np.mean(ram_mb), 0)),
        "device_mem_local_mb_peak": float(round(np.max(mem_local_mb), 0)),
        "device_mem_shared_mb_peak": float(round(np.max(mem_shared_mb), 0)),
    }


# ── Phase 1 helper ────────────────────────────────────────────────────────────


def build_pre_bench_block(
    model_id: str,
    opset: int,
    task: str,
    device: str,
    device_resolved: str,
    io_config: dict,
    *,
    is_hf: bool,
    onnx_path: str,
    ep_resolved: str = "",
) -> Group:
    """Render the pre-benchmark header: model info, then device info.

    Order:
      1. Model section: id/path, task, opset+producer, ONNX path, inputs/outputs
      2. Device section: requested → resolved (with EP if known)
    """
    # ── Model identity (block 1) ─────────────────────────────────────────────
    source_label = "HF" if is_hf else "local"
    lines: list[Text] = [
        Text.from_markup(
            f"Model:    [bold cyan]{model_id}[/bold cyan]  [dim]({source_label})[/dim]"
        ),
        Text.from_markup(f"ONNX:     [dim]{onnx_path}[/dim]"),
        Text(""),
    ]

    # ── Model surface (block 2: task / opset / I/O) ──────────────────────────
    lines.append(Text.from_markup(f"Task:     [cyan]{task}[/cyan]"))
    lines.append(Text.from_markup(f"Opset:    [green]{opset}[/green]"))

    # Inputs / Outputs aligned in two columns
    name_width = max(
        max((len(n) for n in io_config["input_names"]), default=0),
        max((len(n) for n in io_config["output_names"]), default=0),
    )
    shape_width = max(
        max((len(str(s)) for s in io_config["input_shapes"]), default=0),
        max((len(str(s)) for s in io_config["output_shapes"]), default=0),
    )

    output_types = io_config.get("output_types") or [""] * len(io_config["output_names"])
    in_names = io_config["input_names"]
    in_shapes = io_config["input_shapes"]
    in_types = io_config["input_types"]
    out_names = io_config["output_names"]
    out_shapes = io_config["output_shapes"]
    # Labels padded to 10 chars to align with "Model:    " / "Task:     " column.
    sections = (
        ("Inputs:   ", in_names, in_shapes, in_types),
        ("Outputs:  ", out_names, out_shapes, output_types),
    )
    for label, names, shapes, types in sections:
        for i, (nm, sh, ty) in enumerate(zip(names, shapes, types, strict=False)):
            prefix = label if i == 0 else " " * len(label)
            ty_str = f"   [dim]{ty}[/dim]" if ty else ""
            lines.append(
                Text.from_markup(
                    f"{prefix}[cyan]{nm:<{name_width}}[/cyan]   {sh!s:<{shape_width}}{ty_str}"
                )
            )

    # ── Device section ───────────────────────────────────────────────────────
    if device.lower() == device_resolved.lower():
        device_label = f"[cyan]{device_resolved}[/cyan]"
    else:
        device_label = f"[green]{device}[/green] → [cyan]{device_resolved}[/cyan]"
    ep_suffix = f"  [dim]({ep_resolved})[/dim]" if ep_resolved else ""
    lines.append(Text(""))
    lines.append(Text.from_markup(f"Device:   {device_label}{ep_suffix}"))

    return Group(*lines)


# ── Phase 2 helpers ───────────────────────────────────────────────────────────


def build_chart(
    npu_samples: list[float],
    cpu_samples: list[float],
    gpu_samples: list[float],
    *,
    t_now: float,
    window_s: float = WINDOW_SECONDS,
    poll_interval_s: float = POLL_INTERVAL_S,
) -> Group:
    """Render NPU/CPU/GPU overlaid line plot via plotext.

    Y-axis fixed 0..100 with ticks at 0/20/40/60/80/100. X-axis slides:
    shows the last `window_s` seconds of samples.
    """
    plt.clf()
    plt.theme("clear")

    window_count = int(window_s / poll_interval_s)

    def _plot_line(samples: list[float], color: str) -> None:
        if not samples:
            return
        window = samples[-window_count:]
        start_idx = max(0, len(samples) - len(window))
        times = [(start_idx + i) * poll_interval_s for i in range(len(window))]
        plt.plot(times, window, marker="braille", color=color)

    _plot_line(npu_samples, SILICON_COLORS["npu"])
    _plot_line(cpu_samples, SILICON_COLORS["cpu"])
    _plot_line(gpu_samples, SILICON_COLORS["gpu"])

    plt.ylabel("Usage %")
    plt.ylim(0, 100)
    plt.yticks([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])

    x_min = max(0.0, t_now - window_s)
    x_max = max(t_now, window_s)
    plt.xlim(x_min, x_max)
    plt.xlabel("Time (s)")

    plt.plotsize(CHART_WIDTH, CHART_HEIGHT)

    title = Text.from_markup(
        f"  Utilization ("
        f"[{SILICON_COLORS['npu']}]██[/{SILICON_COLORS['npu']}] NPU %  "
        f"[{SILICON_COLORS['cpu']}]██[/{SILICON_COLORS['cpu']}] CPU %  "
        f"[{SILICON_COLORS['gpu']}]██[/{SILICON_COLORS['gpu']}] GPU %)"
    )

    ansi_output = plt.build()
    chart_lines = [Text.from_ansi(line) for line in ansi_output.splitlines()]
    return Group(title, *chart_lines)


def build_progress_bar(
    iteration: int,
    total: int,
    warmup: int,
    *,
    width: int = PROGRESS_WIDTH,
) -> str:
    """Two-tone progress bar markup string.

    Layout: ``[<warmup-section><measured-section>]`` where:
    - warmup section is sized proportional to `warmup / total` and styled dim
    - measured section fills the remaining width and styled green
    - unfilled portions in either section use light shade `░`
    """
    warmup_width = max(1, round(warmup / total * width)) if warmup > 0 and total > 0 else 0
    measured_width = width - warmup_width

    if iteration <= warmup:
        warmup_filled = (
            min(warmup_width, round(iteration / warmup * warmup_width)) if warmup > 0 else 0
        )
        measured_filled = 0
    else:
        warmup_filled = warmup_width
        measured_done = iteration - warmup
        measured_total = total - warmup
        measured_filled = (
            min(measured_width, round(measured_done / measured_total * measured_width))
            if measured_total > 0
            else 0
        )

    return (
        "["
        f"[dim]{'█' * warmup_filled}{'░' * (warmup_width - warmup_filled)}[/dim]"
        f"[green]{'█' * measured_filled}[/green]"
        f"[dim]{'░' * (measured_width - measured_filled)}[/dim]"
        "]"
    )


def build_status_lines(
    progress: dict,
    hw_now: dict,
    latency_ms: float,
    *,
    device_label: str,
) -> Text:
    """3-row status block joined with newlines (returns one Text for alignment)."""
    iteration = progress["iteration"]
    total = progress["total"]
    warmup = progress["warmup"]
    pct = iteration / total if total > 0 else 0.0

    bar = build_progress_bar(iteration, total, warmup)

    if iteration <= warmup:
        progress_label = f"[yellow]Warmup: {iteration}/{warmup}[/yellow]"
    else:
        effective_iter = iteration - warmup
        total_bench = total - warmup
        progress_label = f"[green]Iter: {effective_iter}/{total_bench}[/green]"

    throughput = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    pct_cell = f"{bar} {pct:.0%}"
    row1 = f"  {pct_cell:<40}|  {progress_label}  |  Device: [cyan]{device_label}[/cyan]"
    mem_local = hw_now["mem_local_mb"]
    mem_shared = hw_now["mem_shared_mb"]
    row2 = (
        f"  NPU: {hw_now['npu_pct_avg']:.1f}% avg ({hw_now['npu_pct_now']:.1f}% now)"
        f"   |  CPU: {hw_now['cpu_pct_now']:.1f}%"
        f"   |  GPU: {hw_now['gpu_pct_now']:.1f}%"
        f"   |  Sys Mem: {hw_now['ram_mb']:.0f} MB"
        f"   |  Device Mem: {mem_local:.0f}/{mem_shared:.0f} MB (local/shared)"
    )
    row3 = f"  Latency: {latency_ms:.2f} ms           |  Throughput: ~{throughput:.0f} smp/s"

    return Text.from_markup(f"{row1}\n{row2}\n{row3}")


def build_live_panel(chart: Group, status_lines: Text, model_id: str) -> Panel:
    """Wrap chart + status block in a blue-bordered Rich Panel."""
    return Panel(
        Group(chart, Text(""), status_lines),
        title=f"[bold]HW Monitor[/bold] - {model_id}",
        border_style="blue",
    )


# ── Phase 3 helpers ───────────────────────────────────────────────────────────


def build_latency_table(latency_ms: dict[str, float]) -> Table:
    """8-column latency stats table (Avg/P50/P90/P95/P99/Min/Max/Std)."""
    table = Table(show_header=True, header_style="bold", expand=False)
    for col in ("Avg", "P50", "P90", "P95", "P99", "Min", "Max", "Std"):
        table.add_column(col, justify="right")
    table.add_row(
        f"{latency_ms['mean']:.2f}",
        f"{latency_ms['p50']:.2f}",
        f"{latency_ms['p90']:.2f}",
        f"{latency_ms['p95']:.2f}",
        f"{latency_ms['p99']:.2f}",
        f"{latency_ms['min']:.2f}",
        f"{latency_ms['max']:.2f}",
        f"{latency_ms['std']:.2f}",
    )
    return table


def build_throughput_line(samples_per_sec: float) -> Text:
    """Render the post-bench throughput line."""
    return Text.from_markup(
        f"Throughput: [bold green]{samples_per_sec:.2f}[/bold green] samples/sec"
    )


def build_hw_summary_block(hw: dict[str, float]) -> Group:
    """Static post-bench HW summary."""
    mem_local = hw["device_mem_local_mb_peak"]
    mem_shared = hw["device_mem_shared_mb_peak"]
    return Group(
        Text.from_markup("[bold]Hardware (during benchmark)[/bold]"),
        Text.from_markup(
            f"  [{SILICON_COLORS['npu']}]NPU[/{SILICON_COLORS['npu']}]: "
            f"{hw['npu_pct_avg']:.1f}% avg, {hw['npu_pct_peak']:.1f}% peak"
            f"   |  [{SILICON_COLORS['cpu']}]CPU[/{SILICON_COLORS['cpu']}]: "
            f"{hw['cpu_pct_avg']:.1f}% avg"
            f"   |  [{SILICON_COLORS['gpu']}]GPU[/{SILICON_COLORS['gpu']}]: "
            f"{hw['gpu_pct_avg']:.1f}% avg"
        ),
        Text.from_markup(
            f"  Sys Mem: {hw['ram_mb_avg']:,.0f} MB  "
            f"|  Device Mem: {mem_local:.0f}/{mem_shared:.0f} MB (local/shared)"
        ),
    )


def build_save_footer(json_path: str) -> Text:
    """Render the post-bench 'Results saved to:' footer."""
    return Text.from_markup(f"  📁 Results saved to: [cyan]{json_path}[/cyan]")


# ── Demo orchestrator ─────────────────────────────────────────────────────────


def _hw_now_at_tick(samples_taken: int) -> dict[str, float]:
    """Snapshot of 'current' HW state at a given sample-count cursor.

    `samples_taken` indexes into the FULL pre-generated sample lists. avg/now
    fields are computed on the slice [:samples_taken].
    """
    npu_seen = NPU_SAMPLES_FULL[:samples_taken] or [0.0]
    cpu_seen = CPU_SAMPLES_FULL[:samples_taken] or [0.0]
    gpu_seen = GPU_SAMPLES_FULL[:samples_taken] or [0.0]
    return {
        "npu_pct_now": npu_seen[-1],
        "npu_pct_avg": float(np.mean(npu_seen)),
        "cpu_pct_now": cpu_seen[-1],
        "gpu_pct_now": gpu_seen[-1],
        "ram_mb": RAM_MB_FULL[min(samples_taken - 1, len(RAM_MB_FULL) - 1)],
        "mem_local_mb": MEM_LOCAL_MB_FULL[0],
        "mem_shared_mb": MEM_SHARED_MB_FULL[0],
    }


def _latency_at_iteration(iteration: int) -> float:
    """Pick the latency sample for the current iteration index (1-based)."""
    if iteration <= WARMUP:
        return WARMUP_SAMPLES_MS[max(0, iteration - 1)]
    measured_idx = iteration - WARMUP - 1
    measured_idx = max(0, min(measured_idx, len(RAW_SAMPLES_MS) - 1))
    return RAW_SAMPLES_MS[measured_idx]


def demo(
    *,
    op_tracing: str | None = None,
    top_k: int = OP_TRACING_TOP_K_DEFAULT,
    op_tracing_iterations: int = OP_TRACING_NUM_SAMPLES,
) -> None:
    """Render the full --monitor --iterations 1000 output sequence.

    When ``op_tracing`` is ``"basic"`` or ``"detail"``, append a Phase 4
    operator-tracing section after the post-bench summary.
    """
    console = Console()

    # ── Phase 1 — Pre-bench header ───────────────────────────────────────────
    console.print()
    console.print(
        build_pre_bench_block(
            model_id=MODEL_ID,
            opset=OPSET,
            task=TASK,
            device=DEVICE,
            device_resolved=DEVICE_RESOLVED,
            io_config=IO_CONFIG,
            is_hf=IS_HF_MODEL,
            onnx_path=CACHED_ONNX_PATH,
            ep_resolved=EP_RESOLVED,
        )
    )
    console.print()

    # ── Phase 2 — Live monitor (3.0 sec animation) ───────────────────────────
    total_iters = ITERATIONS + WARMUP  # 1010
    n_ticks = 15  # 5 fps x 3 sec
    iters_per_tick = total_iters / n_ticks  # ~67.3
    samples_per_tick = len(NPU_SAMPLES_FULL) // n_ticks  # 2

    initial_progress = {"iteration": 0, "total": total_iters, "warmup": WARMUP}
    initial_hw = _hw_now_at_tick(1)
    initial_status = build_status_lines(
        initial_progress,
        initial_hw,
        latency_ms=WARMUP_SAMPLES_MS[0],
        device_label=DEVICE,
    )
    initial_chart = build_chart([], [], [], t_now=0.0)

    with Live(
        build_live_panel(initial_chart, initial_status, MODEL_ID),
        console=console,
        refresh_per_second=REFRESH_FPS,
        transient=False,  # last frame stays in scrollback
    ) as live:
        for tick in range(1, n_ticks + 1):
            samples_so_far = min(tick * samples_per_tick, len(NPU_SAMPLES_FULL))
            current_iter = min(int(tick * iters_per_tick), total_iters)
            t_now = samples_so_far * POLL_INTERVAL_S

            npu = NPU_SAMPLES_FULL[:samples_so_far]
            cpu = CPU_SAMPLES_FULL[:samples_so_far]
            gpu = GPU_SAMPLES_FULL[:samples_so_far]

            chart = build_chart(npu, cpu, gpu, t_now=t_now)
            progress = {
                "iteration": current_iter,
                "total": total_iters,
                "warmup": WARMUP,
            }
            hw_now = _hw_now_at_tick(samples_so_far)
            status = build_status_lines(
                progress,
                hw_now,
                latency_ms=_latency_at_iteration(current_iter),
                device_label=DEVICE,
            )
            live.update(build_live_panel(chart, status, MODEL_ID))
            time.sleep(0.2)

    console.print()

    # ── Phase 3 — Post-bench static summary ──────────────────────────────────
    latency_stats = compute_latency_stats(RAW_SAMPLES_MS, WARMUP_SAMPLES_MS)
    hw_aggregates = compute_hw_aggregates(
        NPU_SAMPLES_FULL,
        CPU_SAMPLES_FULL,
        GPU_SAMPLES_FULL,
        RAM_MB_FULL,
        MEM_LOCAL_MB_FULL,
        MEM_SHARED_MB_FULL,
    )

    console.print(Text.from_markup("[bold]Latency (ms)[/bold]"))
    console.print(build_latency_table(latency_stats))
    console.print(
        Text.from_markup(
            f"  [dim]Warmup: {latency_stats['warmup_mean']:.2f} ms avg "
            f"(first {WARMUP} iterations)[/dim]"
        )
    )
    console.print()

    samples_per_sec = 1000.0 / latency_stats["mean"]
    console.print(build_throughput_line(samples_per_sec))
    console.print()

    console.print(build_hw_summary_block(hw_aggregates))
    console.print()

    console.print(build_save_footer(f"./{MODEL_ID.replace('/', '_')}_perf.json"))
    console.print()

    # ── Phase 4 — Op-tracing (only when requested) ───────────────────────────
    if op_tracing is not None:
        ops = generate_fake_ops(num_samples=op_tracing_iterations)
        render_op_tracing(
            console,
            ops,
            level=op_tracing,
            top_k=top_k,
            num_samples=op_tracing_iterations,
            json_path=OP_TRACE_JSON_PATH,
            csv_path=OP_TRACE_CSV_PATH,
        )
        console.print()


# ── CLI entry point ─────────────────────────────────────────────────────────


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="console_mockup.py",
        description=(
            "Render a mockup of `wmk perf` output. "
            "Pass --op-tracing to append the operator tracing section."
        ),
    )
    parser.add_argument(
        "--op-tracing",
        choices=["basic", "detail"],
        default=None,
        help="Enable operator tracing section (basic or detail mode).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=OP_TRACING_TOP_K_DEFAULT,
        help=f"Number of top operator instances to show (default: {OP_TRACING_TOP_K_DEFAULT}).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=OP_TRACING_NUM_SAMPLES,
        help=(
            "Number of fake inference samples per operator instance. "
            f"Default: {OP_TRACING_NUM_SAMPLES}. "
            "When --op-tracing is set without an explicit --iterations, "
            "collapses to 1 (a single inference produces a usable per-op trace; "
            "more iterations just inflate the rendered table)."
        ),
    )
    return parser.parse_args(argv)


def _user_passed_top_k(argv: list[str]) -> bool:
    """Detect whether the user explicitly typed --top-k on the CLI.

    Argparse's default value is indistinguishable from an explicit
    ``--top-k 5``, so we inspect ``sys.argv`` directly.
    """
    return any(a == "--top-k" or a.startswith("--top-k=") for a in argv)


def _user_passed_iterations(argv: list[str]) -> bool:
    """Detect whether the user explicitly typed --iterations on the CLI.

    Same rationale as :func:`_user_passed_top_k`: enables a smart default
    where ``--op-tracing`` without ``--iterations`` collapses to 1.
    """
    return any(a == "--iterations" or a.startswith("--iterations=") for a in argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Parses args, validates them, and dispatches to ``demo``."""
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = _parse_args(raw_argv)

    console = Console()

    # Hard-error: --top-k without --op-tracing is meaningless.
    if args.op_tracing is None and _user_passed_top_k(raw_argv):
        console.print(
            "[red]Error:[/red] --top-k requires --op-tracing to be set.",
            highlight=False,
        )
        return 2

    if args.top_k < 1:
        console.print("[red]Error:[/red] --top-k must be >= 1.", highlight=False)
        return 2

    # Smart default: --op-tracing produces a usable per-op trace from a single
    # inference; the default 100 just inflates the rendered table without
    # adding profiling value (operators are averaged across iterations). When
    # --op-tracing is set AND --iterations was not explicitly passed, collapse
    # to 1.
    if args.op_tracing is not None and not _user_passed_iterations(raw_argv):
        args.iterations = 1

    if args.iterations < 1:
        console.print("[red]Error:[/red] --iterations must be >= 1.", highlight=False)
        return 2

    demo(
        op_tracing=args.op_tracing,
        top_k=args.top_k,
        op_tracing_iterations=args.iterations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
