# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Performance benchmarking command.

Benchmarks model inference performance using WinMLAutoModel and WinMLSession.

Usage:
    winml perf -m microsoft/resnet-50
    winml perf -m microsoft/resnet-50 --device npu --iterations 100
    winml perf -m bert-base-uncased --task text-classification
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import numpy as np
from rich.console import Console
from rich.table import Table

from ..session import VALID_DEVICES
from ._live_chart import LiveMonitorDisplay
from ._pre_bench import print_pre_bench_block


if TYPE_CHECKING:
    from ..models.winml.base import WinMLPreTrainedModel
    from ..session import EPDeviceTarget
    from ..session.monitor.ep_monitor import WinMLEPMonitor
    from ..session.stats import PerfStats

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Hardware monitor polling interval (milliseconds)
_HW_POLL_INTERVAL_MS = 200


# =============================================================================
# Monitor JSON Dispatch (v2.4 — typed accessor + transitional to_dict)
# =============================================================================


def _monitor_to_json_dict(monitor: WinMLEPMonitor) -> dict[str, Any]:
    """Extract JSON-serializable monitor data via typed accessors (v2.4).

    Op-tracing monitors expose data via ``monitor.result`` (returns
    :class:`OpTraceResult`).  Proof-of-execution monitors (VitisAI,
    OpenVINO) still expose theirs via ``to_dict()`` transitionally — to be
    replaced by a typed ``proof`` accessor in a follow-up PR (see PRD
    OQ-6).  ``NullEPMonitor`` returns an empty dict (no monitor data to
    surface).

    Order matters:

    1. ``monitor.result`` first — catches QNNMonitor (and any future
       op-tracing monitor); ``result.to_dict()`` returns the
       :class:`OpTraceResult` nested schema.
    2. ``hasattr(monitor, "to_dict")`` — VitisAI / OpenVINO transitional
       path until they expose a typed ``proof`` accessor.
    3. ``{}`` — :class:`NullEPMonitor` (no data to surface).

    Error containment (Bundle B): a regression in any monitor's serializer
    must not crash ``wmk perf`` mid-output after the benchmark already ran.
    Exceptions raised by either ``result.to_dict()`` or ``monitor.to_dict()``
    are logged at WARNING (so the regression is diagnosable) and surfaced
    as a sentinel ``{"error": "monitor_serialization_failed: ..."}`` dict
    so the JSON report still serialises successfully.
    """
    try:
        result = monitor.result
        if result is not None:
            return result.to_dict()
        if hasattr(monitor, "to_dict"):
            return monitor.to_dict()
    except Exception as exc:
        logger.warning(
            "Monitor JSON serialization failed for %s: %s",
            type(monitor).__name__,
            exc,
        )
        return {"error": f"monitor_serialization_failed: {exc}"}
    return {}


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
# EP Monitor Dispatch
# =============================================================================


def _resolve_ep_monitor(
    ep: str | None,
    op_tracing: str | None,
    output_dir: Path,
    device: str | None = None,
) -> Any:
    """Pick the WinMLEPMonitor for the requested EP and optional op-tracing level.

    Explicit dispatch — no registry, no plugin loading. Raises RuntimeError
    when op-tracing is requested against an EP that has no op-tracing monitor.

    EP names are matched case-insensitively (``QNN``, ``Qnn``, ``qnn`` all
    behave identically). When ``op_tracing`` is set and ``ep`` is empty,
    ``device`` is consulted to auto-infer the EP (e.g. ``device="npu"``
    selects QNN when QNNMonitor reports availability). This keeps the
    headline ``wmk perf --device npu --op-tracing basic`` invocation working
    without requiring an explicit ``--ep qnn``.

    Args:
        ep: Short EP name from CLI (e.g. "qnn", "vitisai", "cpu", None/empty).
        op_tracing: "basic" | "detail" | None (from --op-tracing flag).
        output_dir: Directory for monitor artifacts (CSV, schematic, etc.).
        device: Device hint from CLI (``"npu"``, ``"cpu"``, etc.). Used only
            to auto-infer EP when ``op_tracing`` is set and ``ep`` is empty.

    Returns:
        An WinMLEPMonitor subclass instance. NullEPMonitor when no monitor applies.

    Raises:
        RuntimeError: If op_tracing is truthy but the EP has no op-tracing
            monitor available on this system.
    """
    from ..session.monitor.ep_monitor import NullEPMonitor

    ep_norm = (ep or "").lower()
    device_norm = (device or "").lower()

    if op_tracing:
        from ..session.monitor.qnn_monitor import QNNMonitor

        # Auto-infer EP when not explicitly set. --op-tracing is itself a
        # strong intent signal for QNN-only profiling, so:
        #   --device npu  -> QNN (SC-1 invocation)
        #   --device auto -> QNN when available (default CLI invocation)
        #   --device ""   -> QNN when available (programmatic callers)
        # Explicit --device cpu / --device gpu still falls through to the
        # hard-fail branch below — those EPs have no op-tracing monitor.
        if not ep_norm and device_norm in ("npu", "auto", "") and QNNMonitor.is_available():
            ep_norm = "qnn"

        if ep_norm == "qnn":
            if not QNNMonitor.is_available():
                raise RuntimeError(
                    "Op-tracing requires QNN EP, but QNN is not available on this system. "
                    "Install onnxruntime-qnn or onnxruntime-windowsml with QNN runtime, "
                    "or run `wmk perf` without --op-tracing."
                )
            return QNNMonitor(level=op_tracing, output_dir=output_dir)

        raise RuntimeError(
            f"Op-tracing not available for EP {ep!r} on device {device!r}. "
            "Op-tracing currently requires QNN. Ensure QNN is available "
            "(install onnxruntime-qnn or onnxruntime-windowsml with QNN runtime)."
        )

    # Proof-of-execution monitors (no op-tracing)
    from ..session.monitor.vitisai_monitor import VitisAIMonitor

    if ep_norm == "vitisai" and VitisAIMonitor.is_available():
        return VitisAIMonitor()
    return NullEPMonitor()


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
    op_tracing: str | None = None


@dataclass
class BenchmarkResult:
    """Results from benchmark execution."""

    # Benchmark config
    config: BenchmarkConfig
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

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

    Uses modelkit.core.model_input_generator for spec-driven generation.
    Returns numpy arrays directly (no torch dependency).

    Args:
        io_config: Model I/O configuration from WinMLSession.io_config.
            Expected keys: ``input_names``, ``input_shapes``, ``input_types``.
            Optional key: ``input_value_ranges`` -- a dict mapping input names
            to ``[low, high)`` integer ranges sourced from the build config.
        batch_size: Override batch dimension

    Returns:
        Dictionary of input_name -> numpy array
    """
    from ..core import generate_dummy_inputs_from_specs

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

    return generate_dummy_inputs_from_specs(specs)


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

        # Compile session early so model.device is resolved for display
        self._model._session.compile()

        # Pre-benchmark identity block (model + device sub-blocks).
        # opset is not currently extracted on this path; pass None.
        io_cfg = self._model.io_config
        print_pre_bench_block(
            Console(stderr=True),
            model_id=self.config.model_id,
            task=self._model.task or self.config.task,
            opset=None,
            inputs=_io_specs_from_config(io_cfg, prefix="input"),
            outputs=_io_specs_from_config(io_cfg, prefix="output"),
            cached_onnx_path=str(self._model._onnx_path)
            if getattr(self._model, "_onnx_path", None)
            else None,
            onnx_file=None,
            device=str(self._model.device),
            ep=str(self.config.ep) if self.config.ep else "auto",
        )

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
        from ..session import resolve_device

        model_id = self.config.model_id
        model_path = Path(model_id)
        is_onnx = model_path.suffix.lower() == ".onnx" and model_path.exists()

        # Resolve (ep, device) to EPDeviceTarget at the CLI boundary.
        # resolve_device deduces missing sides and normalizes "auto".
        ep_device = resolve_device(ep=self.config.ep or None, device=self.config.device)

        # Only override config when user explicitly passes --no-quantize
        override = None
        if self.config.no_quantize:
            override = WinMLBuildConfig(quant=None)

        # Cache control: --ignore-cache -> temp dir, --rebuild -> overwrite cache
        use_cache = not self.config.ignore_cache
        force_rebuild = self.config.rebuild or self.config.ignore_cache

        common_kwargs = {
            "task": self.config.task,
            "config": override,
            "ep_device": ep_device,
            "precision": self.config.precision,
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
        """Execute benchmark iterations with timing.

        Dispatches to the monitored path whenever ``--monitor`` was passed OR
        ``--op-tracing`` was requested. Op-tracing requires the EP monitor to
        wrap ``session.perf()``, so the simple no-monitor path cannot fulfill
        it; routing both flags through the same code path guarantees parity.
        """
        if self.config.monitor or self.config.op_tracing:
            return self._run_benchmark_monitored()
        return self._run_benchmark_simple()

    def _run_benchmark_simple(self) -> PerfStats:
        """Execute benchmark without live monitoring."""
        session = self._model._session
        total_iterations = self.config.warmup + self.config.iterations

        with session.perf(warmup=self.config.warmup) as ctx:
            _run_simple_loop(session, self._inputs, total_iterations)

        # Expose ctx for post-benchmark reporting (parity with monitored path).
        self._perf_ctx = ctx
        return ctx.stats

    def _run_benchmark_monitored(self) -> PerfStats:
        """Execute benchmark with live hardware monitoring and/or op-tracing.

        Resolves the EP-specific monitor (e.g., QNNMonitor, VitisAIMonitor)
        via :func:`_resolve_ep_monitor` (NullEPMonitor when nothing applies).
        The EP monitor is integrated into ``session.perf()`` so op-tracing
        observes the user's actual benchmark iterations.

        HWMonitor (system-wide CPU/RAM/NPU metrics) is engaged when available
        AND either ``--monitor`` was set or HW data is otherwise needed. When
        HWMonitor is unavailable but op-tracing is still requested, the run
        proceeds with the EP monitor only — op-tracing is the headline goal
        and must not be blocked by missing HW telemetry.
        """
        from ..session.monitor.hw_monitor import HWMonitor

        session = self._model._session
        total_iterations = self.config.warmup + self.config.iterations

        output_dir = self.config.output_path.parent if self.config.output_path else Path.cwd()
        try:
            ep_monitor = _resolve_ep_monitor(
                ep=self.config.ep,
                op_tracing=self.config.op_tracing,
                output_dir=output_dir,
                device=self.config.device,
            )
        except RuntimeError as e:
            Console(stderr=True).print(f"[red]Error:[/red] {e}")
            raise SystemExit(1) from None

        # HWMonitor is best-effort: required only for the live-chart UI on
        # --monitor. When it's unavailable but op-tracing is requested, run
        # without HW telemetry rather than degrading op-tracing to a no-op.
        hw_available = HWMonitor.is_available()
        if self.config.monitor and not hw_available:
            Console(stderr=True).print(
                "[yellow]Warning:[/yellow] HWMonitor unavailable on this system. "
                "Running without hardware monitoring."
            )

        if hw_available:
            hw_monitor = HWMonitor(poll_interval_ms=_HW_POLL_INTERVAL_MS)
            with (
                session.perf(warmup=self.config.warmup, monitor=ep_monitor) as ctx,
                hw_monitor as hw,
            ):
                _run_monitored_loop(
                    session,
                    self._inputs,
                    ctx.stats,
                    hw,
                    total_iterations=total_iterations,
                    warmup=self.config.warmup,
                    model_id=self.config.model_id,
                    device=self.config.device,
                )
                self._hw_metrics = hw.to_dict()

            # EP proof / op-trace data — dispatch via typed accessor when
            # available, falling through to to_dict() for transitional
            # proof-of-execution monitors. See _monitor_to_json_dict.
            ep_dict = _monitor_to_json_dict(ctx.monitor)
            if ep_dict:  # NullEPMonitor returns {}, real monitors return data
                self._hw_metrics["ep_proof"] = ep_dict
        else:
            # HW unavailable: run with EP monitor only (op-tracing path).
            with session.perf(warmup=self.config.warmup, monitor=ep_monitor) as ctx:
                _run_simple_loop(session, self._inputs, total_iterations)
            ep_dict = _monitor_to_json_dict(ctx.monitor)
            if ep_dict:
                self._hw_metrics = {"ep_proof": ep_dict}

        # Store the op-trace context for post-benchmark reporting
        self._perf_ctx = ctx
        return ctx.stats

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
            # Actual values (resolved after build + compile)
            actual_device=self._model.device,
            actual_task=self._model.task or self.config.task or "auto-detected",
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
                from ..session import WinMLSession, resolve_device

                # CPU sniff — uses live resolve_device; future opt: cache
                session = WinMLSession(
                    str(build_result.final_onnx_path),
                    ep_device=resolve_device("cpu", "cpu"),
                )
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
                    with session.perf(warmup=warmup) as ctx, hw_ctx as hw:
                        for _ in range(total_iters):
                            session.run(inputs)
                        hw_metrics = hw.to_dict()
                    mod_stats = ctx.stats
                else:
                    with session.perf(warmup=warmup) as ctx:
                        for _ in range(total_iters):
                            session.run(inputs)
                    mod_stats = ctx.stats
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
    # Info section — show "requested (resolved)" when they differ
    console.print()

    req_device = result.config.device
    act_device = result.actual_device
    device_str = f"{req_device} ({act_device})" if req_device != act_device else act_device
    console.print(f"[dim]Device:[/dim]      {device_str}")

    # TODO: show resolved precision once WinMLPreTrainedModel.precision
    # is implemented (derive from _build_config.quant.weight_type)

    act_task = result.actual_task
    if act_task.startswith("n/a"):
        task_str = act_task
    else:
        req_task = result.config.task or "auto"
        task_str = f"{req_task} ({act_task})" if req_task != act_task else act_task
    console.print(f"[dim]Task:[/dim]        {task_str}")

    # I/O tensor info is rendered before the benchmark by print_pre_bench_block.

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
# Shared benchmark helpers
# =============================================================================


def _io_specs_from_config(
    io_config: dict, *, prefix: str
) -> list[tuple[str, str, tuple[int | str, ...]]] | None:
    """Project io_config into (name, dtype, shape) triples for the pre-bench panel.

    ``prefix`` is ``"input"`` or ``"output"``; selects the matching name/
    shape/type lists. Returns ``None`` when names are missing so the
    pre-bench helper can omit the row entirely.

    Dynamic dims (``None`` in shape) render as the string sentinel ``"?"``
    rather than collapsing to integer ``0`` (which readers misinterpret
    as a fixed batch=0).
    """
    names = io_config.get(f"{prefix}_names") or []
    if not names:
        return None
    shapes = io_config.get(f"{prefix}_shapes") or []
    types = io_config.get(f"{prefix}_types") or []
    specs: list[tuple[str, str, tuple[int | str, ...]]] = []
    for i, name in enumerate(names):
        shape = shapes[i] if i < len(shapes) else ()
        dtype = str(types[i]) if i < len(types) else ""
        shape_tuple: tuple[int | str, ...] = (
            tuple(int(d) if d is not None else "?" for d in shape) if shape else ()
        )
        specs.append((str(name), dtype, shape_tuple))
    return specs


def _print_save_to_footer(
    console: Console,
    *,
    trace_json: str | None,
    profiling_csv: str | None,
) -> None:
    """Print save-to footer lines after the op-trace report.

    Each line is rendered only when its path is supplied; if both are
    ``None`` the helper emits nothing. The ``[dim]...[/dim]`` markup
    softens the label so the path itself is the visual anchor.
    """
    if trace_json:
        console.print(f"[dim]Op-trace JSON:[/dim] {trace_json}")
    if profiling_csv:
        console.print(f"[dim]Profiling CSV:[/dim] {profiling_csv}")


def _run_monitored_loop(
    session: Any,
    inputs: dict[str, Any],
    stats: PerfStats,
    hw: Any,
    *,
    total_iterations: int,
    warmup: int,
    model_id: str,
    device: str,
) -> None:
    """Run the benchmark iteration loop with live hardware monitoring.

    Shared by both HF-path (PerfBenchmark) and ONNX-path (_run_onnx_benchmark).
    """
    display = LiveMonitorDisplay(
        total_iterations=total_iterations,
        warmup=warmup,
        model_id=model_id,
        device=device,
    )
    with display:
        for i in range(total_iterations):
            session.run(inputs)

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


def _run_simple_loop(
    session: Any,
    inputs: dict[str, Any],
    total_iterations: int,
) -> None:
    """Run the benchmark iteration loop with periodic debug logging.

    Shared by both HF-path (PerfBenchmark) and ONNX-path (_run_onnx_benchmark).
    """
    for i in range(total_iterations):
        session.run(inputs)

        if (i + 1) % max(1, total_iterations // 10) == 0:
            logger.debug("Progress: %d/%d", i + 1, total_iterations)


# =============================================================================
# ONNX Direct Benchmark
# =============================================================================


def _run_onnx_benchmark(
    onnx_path: Path,
    *,
    ep_device: EPDeviceTarget,
    iterations: int,
    warmup: int,
    batch_size: int,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Benchmark an ONNX file directly via WinMLSession (no HF build).

    Creates a WinMLSession, reads io_config for input shapes,
    generates random inputs, and runs the standard benchmark loop.
    """
    from ..session import WinMLSession

    session = WinMLSession(onnx_path=onnx_path, ep_device=ep_device)

    # Generate random inputs from session's I/O config
    io_cfg = session.io_config
    inputs = generate_random_inputs(io_config=io_cfg, batch_size=batch_size)

    # Compile session early so session.device is resolved for display
    session.compile()

    # Pre-benchmark identity block (raw ONNX path + device sub-blocks).
    print_pre_bench_block(
        Console(stderr=True),
        model_id=None,
        task=None,
        opset=None,
        inputs=None,
        outputs=None,
        cached_onnx_path=None,
        onnx_file=str(onnx_path),
        device=str(session.device),
        ep=str(config.ep) if config.ep else "auto",
    )

    # Run benchmark
    total_iterations = warmup + iterations
    hw_metrics = None
    hw_ctx = None

    # Determine if hardware monitoring is available
    if config.monitor:
        from ..session.monitor.hw_monitor import HWMonitor

        if HWMonitor.is_available():
            hw_ctx = HWMonitor(poll_interval_ms=_HW_POLL_INTERVAL_MS)
        else:
            Console(stderr=True).print(
                "[yellow]Warning:[/yellow] HWMonitor unavailable. "
                "Running ONNX benchmark without monitoring."
            )

    if hw_ctx:
        with session.perf(warmup=warmup) as ctx, hw_ctx as hw:
            _run_monitored_loop(
                session,
                inputs,
                ctx.stats,
                hw,
                total_iterations=total_iterations,
                warmup=warmup,
                model_id=str(onnx_path.name),
                device=ep_device.device,
            )
            hw_metrics = hw.to_dict()
        stats = ctx.stats
    else:
        with session.perf(warmup=warmup) as ctx:
            _run_simple_loop(session, inputs, total_iterations)
        stats = ctx.stats

    # Collect results
    mean_latency_sec = stats.mean_ms / 1000.0
    samples_per_sec = batch_size / mean_latency_sec if mean_latency_sec > 0 else 0
    batches_per_sec = 1.0 / mean_latency_sec if mean_latency_sec > 0 else 0
    samples = stats.samples_ms
    std_ms = float(np.std(samples)) if samples else 0.0

    return BenchmarkResult(
        config=config,
        input_names=io_cfg["input_names"],
        input_shapes=[list(s) if s else [] for s in io_cfg["input_shapes"]],
        input_types=[str(t) for t in io_cfg["input_types"]],
        output_names=io_cfg["output_names"],
        output_shapes=[list(s) if s else [] for s in io_cfg["output_shapes"]],
        mean_ms=stats.mean_ms,
        min_ms=stats.min_ms,
        max_ms=stats.max_ms,
        p50_ms=stats.p50_ms,
        p90_ms=stats.p90_ms,
        p95_ms=stats.p95_ms,
        p99_ms=stats.p99_ms,
        std_ms=std_ms,
        raw_samples_ms=stats.samples_ms,
        samples_per_sec=samples_per_sec,
        batches_per_sec=batches_per_sec,
        actual_device=session.device,
        actual_task="n/a (direct ONNX)",
        hw_monitor=hw_metrics,
    )


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
    help=(
        "Number of benchmark iterations. "
        "When --op-tracing is set without an explicit --iterations, "
        "defaults to 1 (a single inference produces a usable per-op trace; "
        "more iterations just inflate the CSV)."
    ),
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
    type=click.Choice(["auto", *sorted(VALID_DEVICES)], case_sensitive=False),
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
    "(qnn, dml, migraphx, nv_tensorrt_rtx, vitisai, openvino, cpu). "
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
    help="Enable operator-level profiling (requires onnxruntime-qnn). "
    "Currently supported only for HuggingFace model IDs and built model "
    "directories — not for direct .onnx file inputs.",
)
@click.option(
    "--top-k",
    "top_k",
    type=int,
    default=None,
    help="Number of top operator instances to show in the op-tracing table "
    "(default: 5, per mockup spec OP_TRACING_TOP_K_DEFAULT). "
    "Requires --op-tracing.",
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
    top_k: int | None,
    compare_devices: str | None,
    verbose: bool,
) -> None:
    r"""Benchmark model inference performance.

    Measures latency and throughput using random input data generated
    from the model's I/O configuration.

    Accepts both HuggingFace model IDs and local .onnx files.
    HF models go through PerfBenchmark; .onnx files use _run_onnx_benchmark.

    \b
    Examples:
        # Basic benchmark (HuggingFace model)
        winml perf -m microsoft/resnet-50

        # Benchmark a pre-exported ONNX file directly
        winml perf -m model.onnx --device cpu

        # With custom iterations on NPU
        winml perf -m microsoft/resnet-50 --iterations 500 --device npu

        # Text model with explicit task
        winml perf -m bert-base-uncased --task text-classification

        # Per-module benchmarking
        winml perf -m bert-base-uncased --module BertAttention

        # Operator-level profiling (QNN NPU)
        winml perf -m model.onnx --op-tracing basic
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

    # AC 11 (mockup spec): --top-k requires --op-tracing. Outside the
    # op-tracing section the flag is meaningless, so reject it explicitly
    # rather than silently ignoring a user's intent.
    if top_k is not None and op_tracing is None:
        raise click.UsageError("--top-k requires --op-tracing to be set.")
    if top_k is not None and top_k < 1:
        raise click.UsageError("--top-k must be >= 1.")

    # Smart default: --op-tracing produces a usable per-op trace from a single
    # inference; the default 100 iterations just inflates the profiling CSV
    # without adding profiling value (operators are averaged across iterations).
    # When the user did not explicitly pass --iterations alongside --op-tracing,
    # collapse to 1.
    if op_tracing and ctx.get_parameter_source("iterations") == click.core.ParameterSource.DEFAULT:
        iterations = 1

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
        op_tracing=op_tracing,
    )

    model_path = Path(hf_model)
    is_onnx = model_path.suffix.lower() == ".onnx"

    # NFR-2: --op-tracing on a direct .onnx input is not yet supported.
    # _run_onnx_benchmark does not thread the EP monitor through session.perf
    # yet — fail fast and clearly rather than running the benchmark and
    # silently producing no profiling data.
    if op_tracing and is_onnx:
        raise click.UsageError(
            "--op-tracing is not yet supported for direct ONNX file inputs. "
            "Use a HuggingFace model ID or a built model directory."
        )

    try:
        if is_onnx:
            # ONNX direct path -- skip HF build, benchmark via WinMLSession
            if shape_config:
                console.print(
                    "[yellow]Warning:[/yellow] --shape-config is ignored for "
                    "pre-exported ONNX files (shapes are baked into the model)."
                )
                config.shape_config = None
            if not model_path.exists():
                raise FileNotFoundError(f"ONNX file not found: {model_path}")
            console.print(f"[dim]Benchmarking ONNX:[/dim] {model_path}")

            from ..session import resolve_device

            # Resolve to a EPDeviceTarget: resolve_device handles "auto" and
            # deduces missing ep or device.
            ep_device = resolve_device(ep=config.ep or None, device=config.device)

            result = _run_onnx_benchmark(
                model_path,
                ep_device=ep_device,
                iterations=iterations,
                warmup=warmup,
                batch_size=batch_size,
                config=config,
            )
        else:
            # HF model path -- full build + benchmark via PerfBenchmark
            if precision != "auto":
                console.print(f"[dim]Precision: {precision} (applied during model build)[/dim]")
            console.print(f"[dim]Loading model:[/dim] {hf_model}")

            benchmark = PerfBenchmark(config)
            result = benchmark.run()

        # Display console report
        display_console_report(result, console)

        # =================================================================
        # Op-tracing post-benchmark report
        # Op-tracing is integrated into session.perf(monitor=...) via
        # _resolve_ep_monitor; the monitor observes the actual benchmark
        # iterations rather than a separate synthetic profiling pass.
        #
        # NFR-2: when op_tracing was requested, missing/failed profiling data
        # is an ERROR (exit 4), NOT a soft warning. The only degraded-success
        # status is "basic_fallback" (yellow notice, exit 0).
        #
        # A3: the JSON report write is intentionally deferred to AFTER this
        # status check so that a failed op-trace (exit 4) does NOT leave a
        # misleading JSON artifact on disk for CI consumers.
        # =================================================================
        if op_tracing:
            from ..session.monitor.report import display_op_trace_report, write_op_trace_json

            # ONNX direct path is rejected upstream with click.UsageError;
            # only the HF / PerfBenchmark path reaches here with op_tracing.
            perf_ctx = getattr(benchmark, "_perf_ctx", None)
            trace_result = perf_ctx.monitor.result if perf_ctx is not None else None

            if trace_result is None:
                console.print(
                    "[red]Error:[/red] Op-tracing requested but no profiling data was "
                    "produced. Check that the EP is correctly installed and the model "
                    "compiled successfully."
                )
                sys.exit(4)
            if trace_result.status == "no_data":
                detail = trace_result.error or "no CSV written"
                console.print(
                    f"[red]Error:[/red] Op-tracing produced no profiling data "
                    f"({detail}). The EP may have silently fallen back to CPU."
                )
                sys.exit(4)
            if trace_result.status == "parse_failed":
                console.print(
                    f"[red]Error:[/red] Op-tracing artifact parse failed: {trace_result.error}"
                )
                sys.exit(4)
            if trace_result.status == "basic_fallback":
                console.print(
                    "[yellow]Notice:[/yellow] Detail mode degraded to basic CSV "
                    "(QHAS unavailable; set QNN_SDK_ROOT to enable)."
                )

            # Op-trace status is valid (ok or basic_fallback) — safe to write
            # the benchmark JSON now.  Writing after the guard means a failed
            # op-trace (exit 4 above) leaves no JSON artifact on disk.
            write_json_report(result, output)
            console.print(f"[green]Results saved to:[/green] {output}")

            if top_k is not None:
                display_op_trace_report(trace_result, console, top_n=top_k)
            else:
                display_op_trace_report(trace_result, console)
            model_slug = hf_model.replace("/", "_").replace("\\", "_")
            output_dir = output.parent if output else Path.cwd()
            trace_output = output_dir / f"{model_slug}_op_trace.json"
            write_op_trace_json(trace_result, trace_output)
            profiling_csv = trace_result.artifacts.get("csv")
            _print_save_to_footer(
                console,
                trace_json=str(trace_output),
                profiling_csv=profiling_csv,
            )
        else:
            # No op-tracing: write JSON immediately after the console report.
            write_json_report(result, output)
            console.print(f"[green]Results saved to:[/green] {output}")

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] Model not found: {e}")
        sys.exit(3)

    except Exception as e:
        console.print(f"[red]Error:[/red] Benchmark failed: {e}")
        if verbose:
            logger.exception("Benchmark failed")
        sys.exit(4)
