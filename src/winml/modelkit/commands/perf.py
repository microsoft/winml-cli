# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Performance benchmarking command.

Benchmarks model inference performance using WinMLAutoModel and WinMLSession.

Usage:
    wmk perf -m microsoft/resnet-50
    wmk perf -m microsoft/resnet-50 --device npu --iterations 100
    wmk perf -m bert-base-uncased --task text-classification
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .live_chart import LiveMonitorDisplay


if TYPE_CHECKING:
    from ..models.winml.base import WinMLPreTrainedModel
    from ..session.stats import PerfStats

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Hardware monitor polling interval (milliseconds)
_HW_POLL_INTERVAL_MS = 200

# =============================================================================
# Constants for Data Generation
# =============================================================================

# Default values for dynamic dimensions (by position)
DYNAMIC_DIM_DEFAULTS = {
    0: 1,  # Batch dimension
    1: 128,  # Sequence length or channels
    2: 224,  # Height
    3: 224,  # Width
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    model_id: str
    task: str | None = None
    device: str = "auto"
    precision: str = "auto"
    iterations: int = 100
    warmup: int = 10
    batch_size: int = 1
    output_path: Path | None = None
    no_quantize: bool = False
    rebuild: bool = False
    ignore_cache: bool = False
    monitor: bool = False
    ep: str | None = None
    shape_config: dict | None = None


@dataclass
class BenchmarkResult:
    """Results from benchmark execution."""

    # Benchmark config
    config: BenchmarkConfig
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Model info
    input_names: list[str] = field(default_factory=list)
    input_shapes: list[list[int]] = field(default_factory=list)
    input_types: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    output_shapes: list[list[int]] = field(default_factory=list)

    # Latency stats (milliseconds)
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    std_ms: float = 0.0

    # Warmup latency
    warmup_mean_ms: float = 0.0

    # Raw samples (for advanced analysis)
    raw_samples_ms: list[float] = field(default_factory=list)

    # Throughput
    samples_per_sec: float = 0.0
    batches_per_sec: float = 0.0

    # Actual values used (after auto-detection)
    actual_device: str = ""
    actual_task: str = ""

    # Hardware monitor metrics (from HWMonitor.to_dict())
    hw_monitor: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "benchmark_info": {
                "model_id": self.config.model_id,
                "task": self.actual_task,
                "device": self.actual_device,
                "precision": self.config.precision,
                "iterations": self.config.iterations,
                "warmup": self.config.warmup,
                "batch_size": self.config.batch_size,
                "timestamp": self.timestamp,
            },
            "model_info": {
                "input_names": self.input_names,
                "input_shapes": self.input_shapes,
                "input_types": self.input_types,
                "output_names": self.output_names,
                "output_shapes": self.output_shapes,
            },
            "latency_ms": {
                "mean": round(self.mean_ms, 3),
                "min": round(self.min_ms, 3),
                "max": round(self.max_ms, 3),
                "p50": round(self.p50_ms, 3),
                "p90": round(self.p90_ms, 3),
                "p95": round(self.p95_ms, 3),
                "p99": round(self.p99_ms, 3),
                "std": round(self.std_ms, 3),
                "warmup_mean": round(self.warmup_mean_ms, 3),
            },
            "throughput": {
                "samples_per_sec": round(self.samples_per_sec, 2),
                "batches_per_sec": round(self.batches_per_sec, 2),
            },
            "raw_samples_ms": [round(s, 3) for s in self.raw_samples_ms],
        }
        if self.hw_monitor:
            result["hw_monitor"] = self.hw_monitor
        return result


# =============================================================================
# Data Generation
# =============================================================================


def generate_random_inputs(
    io_config: dict[str, Any],
    batch_size: int = 1,
) -> dict[str, np.ndarray]:
    """Generate random inputs based on model io_config.

    Uses modelkit.core.model_input_generator for spec-driven generation,
    then converts torch tensors to numpy for ONNX Runtime.

    Args:
        io_config: Model I/O configuration from WinMLSession.io_config.
            Expected keys: ``input_names``, ``input_shapes``, ``input_types``.
            Optional key: ``input_value_ranges`` -- a dict mapping input names
            to ``[low, high)`` integer ranges sourced from the build config.
        batch_size: Override batch dimension

    Returns:
        Dictionary of input_name -> numpy array
    """
    from ..core.model_input_generator import generate_dummy_inputs_from_specs

    specs: dict[str, dict[str, Any]] = {}
    for name, shape, dtype_str in zip(
        io_config["input_names"],
        io_config["input_shapes"],
        io_config["input_types"],
        strict=True,
    ):
        resolved_shape = _resolve_shape(
            shape=shape,
            input_name=name,
            batch_size=batch_size,
        )

        np_dtype = np.dtype(dtype_str)
        if np.issubdtype(np_dtype, np.integer) or np_dtype == np.bool_:
            gen_dtype = "int"
        else:
            gen_dtype = "float"

        specs[name] = {
            "dtype": gen_dtype,
            "shape": list(resolved_shape),
        }

    torch_inputs = generate_dummy_inputs_from_specs(specs)
    return {name: tensor.numpy() for name, tensor in torch_inputs.items()}


def _resolve_shape(
    shape: list | tuple | None,
    input_name: str,
    batch_size: int,
) -> tuple[int, ...]:
    """Resolve dynamic dimensions in shape."""
    if shape is None:
        logger.warning("Shape unknown for '%s', using (batch_size,)", input_name)
        return (batch_size,)

    resolved = []
    for i, dim in enumerate(shape):
        if dim is None or dim == -1 or (isinstance(dim, str)):
            # Dynamic dimension - resolve
            if i == 0:
                # First dimension is almost always batch
                resolved.append(batch_size)
            else:
                # Use position-based default
                default = DYNAMIC_DIM_DEFAULTS.get(i, 128)
                resolved.append(default)
                logger.debug(
                    "Resolved dynamic dim %d for '%s' to %d",
                    i,
                    input_name,
                    default,
                )
        else:
            resolved.append(int(dim))

    return tuple(resolved)


# =============================================================================
# Benchmark Engine
# =============================================================================


class PerfBenchmark:
    """Performance benchmarking engine.

    Orchestrates model loading, data generation, benchmark execution,
    and result collection.

    Example:
        >>> config = BenchmarkConfig(model_id="microsoft/resnet-50", device="npu")
        >>> benchmark = PerfBenchmark(config)
        >>> result = benchmark.run()
        >>> print(f"Mean latency: {result.mean_ms:.2f} ms")  # ~2ms on NPU/QNN EP
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize benchmark with configuration."""
        self.config = config
        self._model: WinMLPreTrainedModel | None = None
        self._inputs: dict[str, np.ndarray] | None = None

    def run(self) -> BenchmarkResult:
        """Execute full benchmark pipeline.

        Returns:
            BenchmarkResult with timing statistics
        """
        # [1] Load model
        logger.info("Loading model: %s", self.config.model_id)
        self._load_model()

        # [2] Generate inputs
        logger.info("Generating benchmark inputs")
        self._generate_inputs()

        # [3] Run benchmark
        logger.info(
            "Running benchmark: %d iterations + %d warmup",
            self.config.iterations,
            self.config.warmup,
        )
        stats = self._run_benchmark()

        # [4] Collect results
        logger.info("Collecting results")
        return self._collect_results(stats)

    def _load_model(self) -> None:
        """Load model via WinMLAutoModel (handles both HF and ONNX)."""
        from ..config import WinMLBuildConfig
        from ..models import WinMLAutoModel
        from ..sysinfo import resolve_device

        model_id = self.config.model_id
        model_path = Path(model_id)
        is_onnx = model_path.suffix.lower() == ".onnx" and model_path.exists()

        # Resolve device once — "auto" becomes concrete (e.g., "npu")
        resolved_device, _ = resolve_device(device=self.config.device)

        # Only override config when user explicitly passes --no-quantize
        override = None
        if self.config.no_quantize:
            override = WinMLBuildConfig(quant=None)

        # Cache control: --ignore-cache → temp dir, --rebuild → overwrite cache
        use_cache = not self.config.ignore_cache
        force_rebuild = self.config.rebuild or self.config.ignore_cache

        common_kwargs = {
            "task": self.config.task,
            "config": override,
            "device": resolved_device,
            "precision": self.config.precision,
            "ep": self.config.ep,
            "use_cache": use_cache,
            "force_rebuild": force_rebuild,
            "shape_config": self.config.shape_config,
        }

        if is_onnx:
            self._model = WinMLAutoModel.from_onnx(
                onnx_path=model_path,
                **common_kwargs,
            )
        else:
            self._model = WinMLAutoModel.from_pretrained(
                model_id,
                **common_kwargs,
            )

    def _generate_inputs(self) -> None:
        """Generate random inputs based on model io_config."""
        io_config = self._model.io_config
        self._inputs = generate_random_inputs(
            io_config=io_config,
            batch_size=self.config.batch_size,
        )

    def _run_benchmark(self) -> PerfStats:
        """Execute benchmark iterations with timing."""
        if self.config.monitor:
            return self._run_benchmark_monitored()
        return self._run_benchmark_simple()

    def _run_benchmark_simple(self) -> PerfStats:
        """Execute benchmark without live monitoring."""
        session = self._model._session
        total_iterations = self.config.warmup + self.config.iterations

        with session.perf(warmup=self.config.warmup) as stats:
            for i in range(total_iterations):
                session.run(self._inputs)

                # Progress logging (every 10%)
                if (i + 1) % max(1, total_iterations // 10) == 0:
                    logger.debug("Progress: %d/%d", i + 1, total_iterations)

        return stats

    def _run_benchmark_monitored(self) -> PerfStats:
        """Execute benchmark with live hardware monitoring.

        Always runs HWMonitor for system-wide metrics (CPU, RAM, NPU/GPU).
        Optionally runs an EP-specific monitor (e.g., VitisAIMonitor)
        alongside for vendor proof-of-execution. Uses NullEPMonitor when
        no vendor monitor is available, eliminating null checks.
        """
        from ..session.monitor.ep_monitor import NullEPMonitor
        from ..session.monitor.hw_monitor import HWMonitor
        from ..session.monitor.vitisai_monitor import VitisAIMonitor

        session = self._model._session
        total_iterations = self.config.warmup + self.config.iterations

        if not HWMonitor.is_available():
            Console(stderr=True).print(
                "[yellow]Warning:[/yellow] HWMonitor unavailable on this system. "
                "Running without hardware monitoring."
            )
            return self._run_benchmark_simple()

        hw_monitor = HWMonitor(poll_interval_ms=_HW_POLL_INTERVAL_MS)

        # EP-specific proof-of-execution monitor.
        # When QNN/OpenVINO monitors become real, add entries here.
        _ep_monitors = {"vitisai": VitisAIMonitor}
        ep = self.config.ep
        monitor_cls = _ep_monitors.get(ep)
        if monitor_cls and monitor_cls.is_available():
            ep_monitor = monitor_cls()
        else:
            ep_monitor = NullEPMonitor()

        with (
            session.perf(warmup=self.config.warmup) as stats,
            hw_monitor as hw,
            ep_monitor as ep_mon,
        ):
            display = LiveMonitorDisplay(
                total_iterations=total_iterations,
                warmup=self.config.warmup,
                model_id=self.config.model_id,
                device=self.config.device,
            )
            with display:
                for i in range(total_iterations):
                    session.run(self._inputs)

                    latest_latency = stats.all_samples_ms[-1] if stats.all_samples_ms else 0
                    display.update(
                        iteration=i + 1,
                        latency_ms=latest_latency,
                        util_samples=hw.utilization_samples,
                        memory_local_mb=hw.peak_memory_local_mb,
                        memory_shared_mb=hw.peak_memory_shared_mb,
                        cpu_pct=hw.mean_cpu_pct,
                        ram_mb=hw.ram_used_mb,
                        cpu_samples=hw.cpu_samples,
                    )

            # Print final monitor snapshot
            display.print_final_snapshot(
                util_samples=hw.utilization_samples,
                memory_mb=hw.peak_memory_mb,
                latency_ms=stats.mean_ms,
                hw_dict=hw.to_dict(),
                cpu_samples=hw.cpu_samples,
            )

            # Store hardware metrics
            self._hw_metrics = hw.to_dict()
            ep_dict = ep_mon.to_dict()
            if ep_dict:  # NullEPMonitor returns {}, real monitors return data
                self._hw_metrics["ep_proof"] = ep_dict

        return stats

    def _collect_results(self, stats: PerfStats) -> BenchmarkResult:
        """Collect benchmark results from PerfStats."""
        io_config = self._model.io_config

        # Calculate throughput
        mean_latency_sec = stats.mean_ms / 1000.0
        samples_per_sec = self.config.batch_size / mean_latency_sec if mean_latency_sec > 0 else 0
        batches_per_sec = 1.0 / mean_latency_sec if mean_latency_sec > 0 else 0

        # Calculate standard deviation
        samples = stats.samples_ms
        std_ms = float(np.std(samples)) if samples else 0.0

        # Calculate warmup mean latency
        warmup_samples = stats.all_samples_ms[: self.config.warmup]
        warmup_mean_ms = float(np.mean(warmup_samples)) if warmup_samples else 0.0

        return BenchmarkResult(
            config=self.config,
            # Model info
            input_names=io_config["input_names"],
            input_shapes=[list(s) if s else [] for s in io_config["input_shapes"]],
            input_types=[str(t) for t in io_config["input_types"]],
            output_names=io_config["output_names"],
            output_shapes=[list(s) if s else [] for s in io_config["output_shapes"]],
            # Latency stats
            mean_ms=stats.mean_ms,
            min_ms=stats.min_ms,
            max_ms=stats.max_ms,
            p50_ms=stats.p50_ms,
            p90_ms=stats.p90_ms,
            p95_ms=stats.p95_ms,
            p99_ms=stats.p99_ms,
            std_ms=std_ms,
            warmup_mean_ms=warmup_mean_ms,
            raw_samples_ms=stats.samples_ms,
            # Throughput
            samples_per_sec=samples_per_sec,
            batches_per_sec=batches_per_sec,
            # Actual values
            actual_device=self._model._session.device,
            actual_task=self.config.task or "auto-detected",
            # Hardware monitor metrics (only present when --monitor is used)
            hw_monitor=getattr(self, "_hw_metrics", None),
        )


# =============================================================================
# Per-Module Perf
# =============================================================================


def _perf_modules(
    *,
    hf_model: str,
    module_class: str,
    task: str | None,
    iterations: int,
    warmup: int,
    batch_size: int,
    no_quantize: bool,
    output: Path | None,
    verbose: bool,
    console: Console,
    monitor: bool = False,
) -> None:
    """Run per-module build and benchmark for matching submodules.

    Generates module configs via generate_build_config(module=...), builds
    each submodule ONNX, then benchmarks via WinMLSession. Results are
    displayed in a summary table and saved to JSON.

    Args:
        hf_model: HuggingFace model ID.
        module_class: Module class name to match (e.g., "BertAttention").
        task: Explicit task override, or None for auto-detection.
        iterations: Number of benchmark iterations.
        warmup: Number of warmup iterations.
        batch_size: Batch size for input generation.
        no_quantize: If True, skip quantization and compilation.
        output: Output JSON path, or None for auto-generated path.
        verbose: If True, log exceptions at DEBUG level.
        console: Rich console for output.
        monitor: If True, wrap each per-module benchmark with HWMonitor.
    """
    import json as json_mod
    import tempfile

    from ..build import build_hf_model
    from ..config import generate_hf_build_config
    from .build import _instantiate_parent_model

    console.print(f"[dim]Generating module configs for {module_class}...[/dim]")

    try:
        module_configs = generate_hf_build_config(
            model_id=hf_model,
            task=task,
            module=module_class,
        )
    except Exception as e:
        console.print(f"[red]Error generating module configs: {e}[/red]")
        if verbose:
            logger.exception("Module config generation failed")
        sys.exit(3)

    if not module_configs:
        console.print(f"[yellow]No modules matching '{module_class}' found[/yellow]")
        sys.exit(0)

    console.print(f"[dim]Found {len(module_configs)} {module_class} instances[/dim]")

    # Instantiate parent with init weights (no pretrained download)
    model_type = module_configs[0].loader.model_type
    if not model_type:
        console.print("[red]Error: module configs missing model_type[/red]")
        sys.exit(3)
    parent_model = _instantiate_parent_model(model_type, task=task)

    all_results: list[dict[str, Any]] = []
    for i, cfg in enumerate(module_configs):
        module_path = cfg.loader.module_path
        if not module_path:
            console.print(f"[red]  Config #{i} missing loader.module_path[/red]")
            all_results.append(
                {
                    "module_path": "unknown",
                    "mean_ms": -1,
                    "p90_ms": -1,
                    "min_ms": -1,
                    "max_ms": -1,
                    "error": "Missing module_path",
                }
            )
            continue
        label = f"{module_class}[{module_path}]"
        console.print(f"[dim]  [{i + 1}/{len(module_configs)}] {label}[/dim]")

        submodule = parent_model.get_submodule(module_path)

        # Skip quant/compile for faster iteration when requested
        if no_quantize:
            cfg.quant = None
            cfg.compile = None

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                build_result = build_hf_model(
                    config=cfg,
                    output_dir=Path(tmpdir),
                    pytorch_model=submodule,
                )

                # Benchmark using WinMLSession
                from ..session import WinMLSession

                session = WinMLSession(str(build_result.final_onnx_path))
                io_cfg = session.io_config
                inputs = generate_random_inputs(io_cfg, batch_size=batch_size)

                total_iters = warmup + iterations
                hw_ctx = None
                hw_metrics = None

                if monitor:
                    from ..session.monitor.hw_monitor import HWMonitor

                    if HWMonitor.is_available():
                        hw_ctx = HWMonitor(poll_interval_ms=_HW_POLL_INTERVAL_MS)

                if hw_ctx:
                    with session.perf(warmup=warmup) as stats, hw_ctx as hw:
                        for _ in range(total_iters):
                            session.run(inputs)
                        hw_metrics = hw.to_dict()
                else:
                    with session.perf(warmup=warmup) as stats:
                        for _ in range(total_iters):
                            session.run(inputs)

                mod_stats = stats
                result_entry: dict[str, Any] = {
                    "module_path": module_path,
                    "mean_ms": round(mod_stats.mean_ms, 3),
                    "p50_ms": round(mod_stats.p50_ms, 3),
                    "p90_ms": round(mod_stats.p90_ms, 3),
                    "p95_ms": round(mod_stats.p95_ms, 3),
                    "p99_ms": round(mod_stats.p99_ms, 3),
                    "min_ms": round(mod_stats.min_ms, 3),
                    "max_ms": round(mod_stats.max_ms, 3),
                    "std_ms": round(
                        float(np.std(mod_stats.samples_ms)) if mod_stats.samples_ms else 0.0,
                        3,
                    ),
                    "throughput_sps": (
                        round(1000.0 / mod_stats.mean_ms, 2) if mod_stats.mean_ms > 0 else 0.0
                    ),
                }
                if hw_metrics:
                    result_entry["hw_monitor"] = hw_metrics
                all_results.append(result_entry)
            except Exception as e:
                console.print(f"[red]  {label}: FAILED ({e})[/red]")
                all_results.append(
                    {
                        "module_path": module_path,
                        "mean_ms": -1,
                        "p90_ms": -1,
                        "min_ms": -1,
                        "max_ms": -1,
                        "error": str(e),
                    }
                )

    # Display results table
    table = Table(title=f"Per-Module Perf: {module_class}", show_header=True)
    table.add_column("Module Path", style="cyan")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("P90 (ms)", justify="right")
    table.add_column("Min (ms)", justify="right")
    table.add_column("Max (ms)", justify="right")

    for r in all_results:
        if r.get("error"):
            table.add_row(r["module_path"], "[red]FAIL[/red]", "", "", "")
        else:
            table.add_row(
                r["module_path"],
                f"{r['mean_ms']:.2f}",
                f"{r['p90_ms']:.2f}",
                f"{r['min_ms']:.2f}",
                f"{r['max_ms']:.2f}",
            )

    console.print()
    console.print(table)
    console.print()

    # Write JSON report
    if output is None:
        slug = hf_model.replace("/", "_").replace("\\", "_")
        output = Path(f"{slug}_{module_class}_perf.json")

    module_report = {
        "model_id": hf_model,
        "module_class": module_class,
        "instance_count": len(all_results),
        "iterations": iterations,
        "warmup": warmup,
        "instances": all_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json_mod.dump(module_report, f, indent=2)
    console.print(f"[green]Results saved to:[/green] {output}")


# =============================================================================
# Report Generation
# =============================================================================


def display_console_report(result: BenchmarkResult, console: Console) -> None:
    """Display benchmark results in formatted console output."""
    # Header
    console.print()
    console.print(
        Panel.fit(
            f"[bold]Benchmark: {result.config.model_id}[/bold]",
            border_style="blue",
        )
    )

    # Info section
    console.print()
    console.print(f"[dim]Device:[/dim]      {result.actual_device}")
    console.print(f"[dim]Precision:[/dim]   {result.config.precision}")
    console.print(f"[dim]Task:[/dim]        {result.actual_task}")
    console.print(
        f"[dim]Iterations:[/dim]  {result.config.iterations} (+ {result.config.warmup} warmup)"
    )
    console.print(f"[dim]Batch Size:[/dim]  {result.config.batch_size}")

    # Latency table
    console.print()
    console.print("[bold]Latency (ms)[/bold]")

    table = Table(show_header=True, header_style="bold cyan")
    for col in ["Avg", "P50", "P90", "P95", "P99", "Min", "Max", "Std"]:
        table.add_column(col, justify="right")

    table.add_row(
        f"{result.mean_ms:.2f}",
        f"{result.p50_ms:.2f}",
        f"{result.p90_ms:.2f}",
        f"{result.p95_ms:.2f}",
        f"{result.p99_ms:.2f}",
        f"{result.min_ms:.2f}",
        f"{result.max_ms:.2f}",
        f"{result.std_ms:.2f}",
    )

    console.print(table)

    if result.warmup_mean_ms > 0:
        console.print(
            f"  [dim]Warmup: {result.warmup_mean_ms:.2f} ms avg "
            f"(first {result.config.warmup} iterations)[/dim]"
        )

    # Throughput
    console.print()
    console.print(f"[bold]Throughput:[/bold] {result.samples_per_sec:.2f} samples/sec")

    # Hardware section (only when monitoring was active)
    if result.hw_monitor:
        console.print()
        console.print("[bold]Hardware (during benchmark)[/bold]")
        npu = result.hw_monitor.get("npu", {})
        cpu = result.hw_monitor.get("cpu", {})
        ram = result.hw_monitor.get("ram", {})
        dev_mem = result.hw_monitor.get("device_memory", {})
        console.print(
            f"  NPU: {npu.get('mean_pct', 0):.1f}% avg, "
            f"{npu.get('peak_pct', 0):.1f}% peak  |  "
            f"CPU: {cpu.get('mean_pct', 0):.1f}% avg"
        )
        console.print(
            f"  Sys Mem: {ram.get('used_mb', 0):.0f} MB  |  "
            f"Device Mem: {dev_mem.get('local_peak_mb', 0):.0f}/"
            f"{dev_mem.get('shared_peak_mb', 0):.0f} MB (local/shared)"
        )

    console.print()


def write_json_report(result: BenchmarkResult, output_path: Path) -> None:
    """Write benchmark results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)


def generate_output_path(model_id: str) -> Path:
    """Generate default output path from model ID.

    For ONNX files: uses the file stem (e.g., "model.onnx" -> "model_perf.json").
    For HF model IDs: slugifies org/name
    (e.g., "microsoft/resnet-50" -> "microsoft_resnet-50_perf.json").
    """
    p = Path(model_id)
    if p.suffix.lower() == ".onnx":
        return Path(f"{p.stem}_perf.json")
    slug = model_id.replace("/", "_").replace("\\", "_")
    return Path(f"{slug}_perf.json")


# =============================================================================
# CLI Command
# =============================================================================


@click.command("perf")
@click.option(
    "-m",
    "--model",
    "model_id",
    type=str,
    default=None,
    help="Model identifier: HuggingFace model ID or local .onnx file.",
)
@click.option(
    "--hf-model",
    "hf_model_deprecated",
    type=str,
    default=None,
    hidden=True,
    help="[Deprecated] Use -m/--model instead.",
)
@click.option(
    "--task",
    type=str,
    default=None,
    help="Explicit task (e.g., 'image-classification'). Auto-detected if not specified.",
)
@click.option(
    "--iterations",
    type=int,
    default=100,
    show_default=True,
    help="Number of benchmark iterations",
)
@click.option(
    "--warmup",
    type=int,
    default=10,
    show_default=True,
    help="Number of warmup iterations (excluded from statistics)",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "gpu", "npu"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Device to run benchmark on",
)
@click.option(
    "--precision",
    type=str,
    default="auto",
    show_default=True,
    help="Precision mode: auto, fp32, fp16, int8, int16, or w{x}a{y} (e.g., w8a16).",
)
@click.option(
    "--ep",
    "ep",
    type=str,
    default=None,
    help="Force specific execution provider "
    "(qnn, dml, migraphx, tensorrt, vitisai, openvino, cpu). "
    "Overrides device-to-provider mapping.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output JSON file path. Defaults to '{model_slug}_perf.json'",
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    show_default=True,
    help="Batch size for input generation",
)
@click.option(
    "--shape-config",
    "shape_config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='JSON file with shape overrides (e.g., {"height": 480, "width": 480}).',
)
@click.option(
    "--no-quantize",
    is_flag=True,
    default=False,
    help="Skip quantization during model build",
)
@click.option(
    "--rebuild",
    is_flag=True,
    default=False,
    help="Force rebuild even if cached artifacts exist",
)
@click.option(
    "--ignore-cache",
    is_flag=True,
    default=False,
    help="Build from scratch in a temp folder (discard after benchmarking)",
)
@click.option(
    "--module",
    "module_class",
    default=None,
    type=str,
    help="HF module class name for per-module benchmarking (e.g., 'BertAttention'). "
    "Builds and benchmarks each instance separately.",
)
@click.option(
    "--monitor",
    is_flag=True,
    default=False,
    help="Show live NPU utilization chart during benchmark",
)
@click.option(
    "--op-tracing",
    "op_tracing",
    type=click.Choice(["basic", "detail"], case_sensitive=False),
    default=None,
    help="Enable operator-level profiling (requires onnxruntime-qnn)",
)
@click.option(
    "--compare-devices",
    type=str,
    default=None,
    help="Compare benchmark across devices (e.g., 'cpu,npu'). Not yet implemented.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose output",
)
@click.pass_context
def perf(
    ctx: click.Context,
    model_id: str | None,
    hf_model_deprecated: str | None,
    task: str | None,
    iterations: int,
    warmup: int,
    device: str,
    precision: str,
    ep: str | None,
    output: Path | None,
    batch_size: int,
    shape_config_path: Path | None,
    no_quantize: bool,
    rebuild: bool,
    ignore_cache: bool,
    module_class: str | None,
    monitor: bool,
    op_tracing: str | None,
    compare_devices: str | None,
    verbose: bool,
) -> None:
    r"""Benchmark model inference performance.

    Measures latency and throughput using random input data generated
    from the model's I/O configuration.

    Accepts both HuggingFace model IDs and local .onnx files.
    Both paths go through PerfBenchmark with WinMLAutoModel.

    \b
    Examples:
        # Basic benchmark (HuggingFace model)
        wmk perf -m microsoft/resnet-50

        # Benchmark a pre-exported ONNX file directly
        wmk perf -m model.onnx --device cpu

        # With custom iterations on NPU
        wmk perf -m microsoft/resnet-50 --iterations 500 --device npu

        # Text model with explicit task
        wmk perf -m bert-base-uncased --task text-classification

        # Per-module benchmarking
        wmk perf -m bert-base-uncased --module BertAttention

        # Operator-level profiling (QNN NPU)
        wmk perf -m model.onnx --op-tracing basic
    """
    # Resolve deprecated --hf-model alias
    if hf_model_deprecated and model_id:
        raise click.UsageError(
            "Cannot use both -m/--model and --hf-model. Use -m/--model (--hf-model is deprecated)."
        )
    if hf_model_deprecated:
        import warnings

        warnings.warn(
            "--hf-model is deprecated. Use -m/--model instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        model_id = hf_model_deprecated

    if not model_id:
        raise click.UsageError("A model is required via -m/--model.")

    hf_model = model_id

    # Setup logging
    if verbose or (ctx.obj and ctx.obj.get("debug")):
        logging.getLogger("winml.modelkit").setLevel(logging.DEBUG)

    console = Console()

    if compare_devices:
        console.print(
            "[yellow]--compare-devices is not yet implemented. "
            "Run benchmarks separately and compare JSON outputs.[/yellow]"
        )
        return

    # =========================================================================
    # MODULE MODE: per-module build + benchmark
    # =========================================================================
    if module_class:
        if shape_config_path:
            console.print(
                "[yellow]Warning:[/yellow] --shape-config is not supported "
                "in --module mode and will be ignored."
            )
        _perf_modules(
            hf_model=hf_model,
            module_class=module_class,
            task=task,
            iterations=iterations,
            warmup=warmup,
            batch_size=batch_size,
            no_quantize=no_quantize,
            output=output,
            verbose=verbose,
            console=console,
            monitor=monitor,
        )
        return

    # =========================================================================
    # SINGLE MODEL MODE: existing benchmark flow
    # =========================================================================

    # Load shape overrides from JSON if provided
    shape_config = None
    if shape_config_path:
        try:
            with shape_config_path.open() as f:
                shape_config = json.load(f)
            if not isinstance(shape_config, dict):
                raise click.ClickException(
                    f"--shape-config must contain a JSON object, got {type(shape_config).__name__}"
                )
        except json.JSONDecodeError as e:
            raise click.ClickException(
                f"Invalid JSON in --shape-config: {shape_config_path}: {e}"
            ) from e
        console.print(f"[dim]Shape overrides: {shape_config}[/dim]")

    # Resolve output path
    if output is None:
        output = generate_output_path(hf_model)

    # Create config
    config = BenchmarkConfig(
        model_id=hf_model,
        task=task,
        device=device.lower(),
        precision=precision.lower(),
        iterations=iterations,
        warmup=warmup,
        batch_size=batch_size,
        output_path=output,
        no_quantize=no_quantize,
        rebuild=rebuild,
        ignore_cache=ignore_cache,
        monitor=monitor,
        ep=ep.lower() if ep else None,
        shape_config=shape_config,
    )

    try:
        # Both ONNX and HF go through PerfBenchmark (unified pipeline)
        model_path = Path(hf_model)
        is_onnx = model_path.suffix.lower() == ".onnx"
        if is_onnx:
            if shape_config:
                console.print(
                    "[yellow]Warning:[/yellow] --shape-config is ignored for "
                    "pre-exported ONNX files (shapes are baked into the model)."
                )
                config.shape_config = None
            if not model_path.exists():
                raise FileNotFoundError(f"ONNX file not found: {model_path}")
            console.print(f"[dim]Building + benchmarking ONNX:[/dim] {model_path}")
        else:
            if precision != "auto":
                console.print(f"[dim]Precision: {precision} (applied during model build)[/dim]")
            console.print(f"[dim]Loading model:[/dim] {hf_model}")

        benchmark = PerfBenchmark(config)
        result = benchmark.run()

        # Display console report
        display_console_report(result, console)

        # Write JSON report
        write_json_report(result, output)
        console.print(f"[green]Results saved to:[/green] {output}")

        # =================================================================
        # Op-tracing (additive to existing benchmark)
        # =================================================================
        if op_tracing:
            from ..optracing import is_qnn_profiling_available

            if not is_qnn_profiling_available():
                console.print("[red]Error:[/red] Op-tracing requires onnxruntime-qnn")
                console.print("Install with: [bold]pip install onnxruntime-qnn[/bold]")
                raise SystemExit(1)

            from ..optracing.registry import get_tracer
            from ..optracing.report import (
                display_op_trace_report,
                write_op_trace_json,
            )

            # Determine the ONNX model path from the benchmark flow.
            # For HF models the ONNX is built internally by PerfBenchmark.
            try:
                onnx_for_trace = model_path if is_onnx else benchmark._model._onnx_path
            except AttributeError:
                console.print(
                    "[red]Error:[/red] Could not determine ONNX model path for op-tracing"
                )
                raise SystemExit(1) from None

            output_dir = output.parent if output else Path()

            # Look up tracer via registry (EP-agnostic).
            tracer_cls = get_tracer("QNNExecutionProvider", op_tracing)
            if tracer_cls is None:
                console.print(
                    f"[red]Error:[/red] No tracer registered for QNN EP at level '{op_tracing}'"
                )
                raise SystemExit(1)

            profiler = tracer_cls(
                onnx_for_trace,
                output_dir=output_dir,
                level=op_tracing,
            )
            trace_result = profiler.run(
                iterations=min(iterations, 10),
                warmup=min(warmup, 3),
            )

            # Display and save
            display_op_trace_report(trace_result, console)

            model_slug = hf_model.replace("/", "_").replace("\\", "_")
            if is_onnx:
                model_slug = model_path.stem
            trace_output = output_dir / f"{model_slug}_op_trace.json"
            write_op_trace_json(trace_result, trace_output)
            console.print(f"[green]Op-trace saved to:[/green] {trace_output}")

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] Model not found: {e}")
        sys.exit(3)

    except Exception as e:
        console.print(f"[red]Error:[/red] Benchmark failed: {e}")
        if verbose:
            logger.exception("Benchmark failed")
        sys.exit(4)
