# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Performance benchmarking command.

Benchmarks model inference performance using WinMLAutoModel and WinMLSession.

Usage:
    winml perf -m microsoft/resnet-50
    winml perf -m microsoft/resnet-50 --device npu --iterations 100
    winml perf -m microsoft/resnet-50 --module ResNetConvLayer
    winml perf -m bert-base-uncased --task text-classification
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import click
import numpy as np
from rich.console import Console
from rich.table import Table

from ..utils import cli as cli_utils
from ..utils.constants import EPName, EPNameOrAlias
from ..utils.logging import configure_logging
from ._live_chart import LiveMonitorDisplay


if TYPE_CHECKING:
    from ..models.winml.base import WinMLPreTrainedModel
    from ..models.winml.composite_model import WinMLCompositeModel
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
    no_compile: bool = True
    rebuild: bool = False
    ignore_cache: bool = False
    skip_build: bool = True
    allow_unsupported_nodes: bool = False
    monitor: bool = False
    memory: bool = True
    ep: EPNameOrAlias | None = None
    ep_options: dict[str, str] | None = None
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

    # Resolved model precision from io_config (None if the model does not
    # expose one). Distinct from the requested config.precision policy.
    model_precision: str | None = None

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

    # Batch dimension the session actually ran. Equals config.batch_size when
    # the model's leading input dim is dynamic; falls back to the model's
    # static batch (often 1) otherwise. samples_per_sec is scaled by this, not
    # by the requested config.batch_size.
    effective_batch_size: int = 1

    # Actual values used (after auto-detection)
    actual_device: str = ""
    actual_task: str = ""
    actual_ep: EPName | None = None

    # ONNX model ORT actually loaded (may be an EPContext model, differing
    # from the input model_id when compiled or a cached one is reused)
    running_model_path: str = ""

    # Hardware monitor metrics (from HWMonitor.to_dict())
    hw_monitor: dict[str, Any] | None = None

    # Memory profile dict (rss deltas from memory_tracker)
    memory_profile: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "benchmark_info": {
                "model_id": self.config.model_id,
                "running_model_path": self.running_model_path,
                "task": self.actual_task,
                "device": self.actual_device,
                "ep": self.actual_ep,
                "ep_options": self.config.ep_options,
                "precision": self.config.precision,
                "iterations": self.config.iterations,
                "warmup": self.config.warmup,
                "batch_size": self.config.batch_size,
                "effective_batch_size": self.effective_batch_size,
                "timestamp": self.timestamp,
            },
            "model_info": {
                "input_names": self.input_names,
                "input_shapes": self.input_shapes,
                "input_types": self.input_types,
                "output_names": self.output_names,
                "output_shapes": self.output_shapes,
                "precision": self.model_precision,
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
        if self.memory_profile:
            result["memory"] = self.memory_profile
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


def effective_batch_size(
    inputs: dict[str, np.ndarray],
    input_names: list[str],
    requested: int,
) -> int:
    """The batch dimension actually present in the generated inputs.

    The requested ``--batch-size`` only lands on inputs whose leading
    dimension is dynamic; a model with a statically-fixed batch dim ignores
    it (see :func:`_resolve_shape`). Throughput (samples/sec) must be scaled
    by what the session actually ran, not by what was asked, or a static-batch
    model reports ``requested / latency`` while only processing one batch per
    call -- inflating samples/sec by ``requested``.

    Reads the leading dim back from the first batched (rank >= 1) input,
    matching the "first dim is batch" convention used throughout this module.
    Falls back to ``requested`` when no batched input exists (e.g. all-scalar
    inputs), which preserves the prior behavior for that edge case.
    """
    for name in input_names:
        arr = inputs.get(name)
        if arr is not None and arr.ndim >= 1:
            return int(arr.shape[0])
    return requested


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
        self._model: WinMLPreTrainedModel | WinMLCompositeModel | None = None
        self._inputs: dict[str, np.ndarray] | None = None
        self._effective_batch: int = config.batch_size
        self._memory: dict[str, float] | None = None

    @property
    def _is_composite(self) -> bool:
        """Composite models orchestrate multiple sub-sessions (e.g. CLIP/SigLIP).

        Uses a concrete ``isinstance(..., WinMLCompositeModel)`` check rather
        than duck-typing on ``sub_models`` so a future single-session model
        carrying an unrelated ``sub_models`` attribute can't be misrouted. The
        import is function-local because ``composite_model`` pulls in torch: a
        module-level runtime import would blow the ``winml perf --help`` import
        budget (see tests/cli/test_import_time.py). A function-local import
        runs only when this property is read — i.e. after a model is loaded, by
        which point torch is already imported — and never at module load.
        """
        from ..models.winml.composite_model import WinMLCompositeModel

        return isinstance(self._model, WinMLCompositeModel)

    @property
    def _sub_models(self) -> dict[str, WinMLPreTrainedModel]:
        """Sub-models of a composite model (only valid when ``_is_composite``)."""
        from ..models.winml.composite_model import WinMLCompositeModel

        assert isinstance(self._model, WinMLCompositeModel)
        return self._model.sub_models

    @property
    def _single(self) -> WinMLPreTrainedModel:
        """The model under benchmark, narrowed to a single-session model.

        Only valid for non-composite models: composites dispatch to
        ``_run_sub_models``, which benchmarks each sub-model through a child
        ``PerfBenchmark`` whose ``_model`` is itself single-session. Exposes
        ``io_config`` / ``device`` / ``ep_name`` / ``task`` directly (the
        session caches ``io_config``), so callers read ``self._single.*``
        rather than going through per-attribute wrappers.
        """
        assert self._model is not None
        return cast("WinMLPreTrainedModel", self._model)

    def run(self) -> BenchmarkResult | dict[str, BenchmarkResult]:
        """Execute full benchmark pipeline.

        Returns:
            A single ``BenchmarkResult`` for single-session models, or a
            ``{sub_model_name: BenchmarkResult}`` mapping for composite models
            (e.g. CLIP/SigLIP dual-encoders). Composite models have no single
            ORT session, so each sub-model is benchmarked individually rather
            than timing the aggregate ``forward()`` pass.
        """
        # [1] Load model (build pipeline: optimize, cache, etc.)
        logger.info("Loading model: %s", self.config.model_id)
        self._load_model()
        assert self._model is not None

        if self._is_composite:
            return self._run_sub_models()
        return self._run_single()

    def _run_sub_models(self) -> dict[str, BenchmarkResult]:
        """Benchmark each sub-model of a composite individually.

        Each sub-model is itself a single-session ``WinMLAutoModel``, so it is
        benchmarked through the standard single-model pipeline by spawning a
        child ``PerfBenchmark`` with the already-loaded sub-model. Results are
        keyed by sub-model name for per-component reporting.
        """
        results: dict[str, BenchmarkResult] = {}
        for name, sub in self._sub_models.items():
            logger.info("Benchmarking sub-model '%s'", name)
            Console(stderr=True).print(f"\n[bold]Sub-model:[/bold] {name}")
            child = PerfBenchmark(self.config)
            child._model = sub
            try:
                results[name] = child._run_single()
            except Exception as exc:
                logger.error("Benchmarking sub-model '%s' failed", name)
                raise RuntimeError(f"Sub-model '{name}' failed: {exc}") from exc
        return results

    def _run_single(self) -> BenchmarkResult:
        """Benchmark the loaded single-session model.

        Returns:
            BenchmarkResult with timing statistics
        """
        import gc

        assert self._model is not None

        # Initialize memory tracking variables
        adapter_luid: str | None = None
        rss_baseline = rss_after_compile = 0.0
        vram_local_baseline = vram_shared_baseline = 0.0
        vram_local_compile = vram_shared_compile = 0.0

        # Memory: baseline right before compile() — excludes all Python lib
        # imports, EP DLLs, and build pipeline overhead. Measures only ORT
        # session compilation (model weights loaded into memory).
        if self.config.memory:
            from ..session.monitor.memory_tracker import get_rss_mb, get_vram_mb

            adapter_luid = self._resolve_adapter_luid()
            gc.collect()
            rss_baseline = get_rss_mb()
            vram_local_baseline, vram_shared_baseline = get_vram_mb(adapter_luid)

        # [2] Generate inputs
        logger.info("Generating benchmark inputs")
        self._generate_inputs()

        # Compile session early so model.device is resolved for display
        self._single._session.compile()

        if self.config.memory:
            gc.collect()
            rss_after_compile = get_rss_mb()
            vram_local_compile, vram_shared_compile = get_vram_mb(adapter_luid)

        # Print model info before benchmark starts
        _print_model_info(
            self._single.io_config,
            task=self._single.task or self.config.task,
            req_device=self.config.device,
            act_device=self._single.device,
            ep_name=self._single.ep_name,
        )

        # [3] Run benchmark
        logger.info(
            "Running benchmark: %d iterations + %d warmup",
            self.config.iterations,
            self.config.warmup,
        )
        stats = self._run_benchmark()

        if self.config.memory:
            rss_after_inference = get_rss_mb()
            vram_local_infer, vram_shared_infer = get_vram_mb(adapter_luid)
            self._memory = {
                "rss_baseline_mb": round(rss_baseline, 2),
                "rss_after_compile_mb": round(rss_after_compile, 2),
                "rss_after_inference_mb": round(rss_after_inference, 2),
                "rss_model_load_delta_mb": round(rss_after_compile - rss_baseline, 2),
                "rss_inference_delta_mb": round(rss_after_inference - rss_after_compile, 2),
                "rss_total_delta_mb": round(rss_after_inference - rss_baseline, 2),
                "vram_local_after_inference_mb": round(vram_local_infer, 2),
                "vram_shared_after_inference_mb": round(vram_shared_infer, 2),
                "vram_local_model_load_delta_mb": round(
                    vram_local_compile - vram_local_baseline, 2
                ),
                "vram_local_inference_delta_mb": round(vram_local_infer - vram_local_compile, 2),
                "vram_local_total_delta_mb": round(vram_local_infer - vram_local_baseline, 2),
                "vram_shared_model_load_delta_mb": round(
                    vram_shared_compile - vram_shared_baseline, 2
                ),
                "vram_shared_inference_delta_mb": round(vram_shared_infer - vram_shared_compile, 2),
                "vram_shared_total_delta_mb": round(vram_shared_infer - vram_shared_baseline, 2),
            }

        # [4] Collect results
        logger.info("Collecting results")
        return self._collect_results(stats)

    def _load_model(self) -> None:
        """Load model via WinMLAutoModel.

        Both HF model IDs and pre-exported .onnx files flow through this
        single path so latency numbers stay comparable: HF runs export →
        optimize → [quantize] → [compile], and ONNX runs the same pipeline
        minus export.
        """
        from ..config import WinMLBuildConfig
        from ..models import WinMLAutoModel

        model_id = self.config.model_id
        model_path = Path(model_id)
        is_onnx = model_path.suffix.lower() == ".onnx"
        if is_onnx and not model_path.exists():
            # Surface a clear error for programmatic callers. The CLI guards
            # this earlier, but without this check from_pretrained would fall
            # through to HF loading and produce a confusing "not a valid JSON
            # file" error from AutoConfig.
            raise FileNotFoundError(f"ONNX file not found: {model_path}")

        # Only override config when user explicitly passes --no-quantize
        override = None
        if self.config.no_quantize:
            override = WinMLBuildConfig(quant=None)

        # Cache control: --ignore-cache -> temp dir, --rebuild -> overwrite cache
        use_cache = not self.config.ignore_cache
        force_rebuild = self.config.rebuild or self.config.ignore_cache

        common_kwargs: dict[str, Any] = {
            "task": self.config.task,
            "config": override,
            "device": self.config.device,
            "precision": self.config.precision,
            "ep": self.config.ep,
            "provider_options": self.config.ep_options,
            "use_cache": use_cache,
            "force_rebuild": force_rebuild,
            "shape_config": self.config.shape_config,
            "allow_unsupported_nodes": self.config.allow_unsupported_nodes,
            "no_compile": self.config.no_compile,
        }

        if is_onnx:
            self._model = WinMLAutoModel.from_onnx(
                onnx_path=model_path,
                skip_build=self.config.skip_build,
                **common_kwargs,
            )
        else:
            self._model = WinMLAutoModel.from_pretrained(
                model_id,
                **common_kwargs,
            )

    def _generate_inputs(self) -> None:
        """Generate random inputs based on model io_config."""
        io_config = self._single.io_config
        self._inputs = generate_random_inputs(
            io_config=io_config,
            batch_size=self.config.batch_size,
        )
        self._effective_batch = effective_batch_size(
            self._inputs,
            io_config["input_names"],
            self.config.batch_size,
        )
        if self.config.batch_size != 1 and self._effective_batch != self.config.batch_size:
            logger.warning(
                "Requested --batch-size %d could not be applied: the model's "
                "leading input dimension is statically %d. Throughput is scaled "
                "by the actual batch (%d), not the requested value.",
                self.config.batch_size,
                self._effective_batch,
                self._effective_batch,
            )

    def _resolve_adapter_luid(self) -> str | None:
        """Resolve adapter LUID for VRAM queries."""
        import sys

        if sys.platform != "win32":
            return None

        assert self._model is not None
        device = self._single.device or self.config.device
        if device == "cpu":
            return None

        try:
            from ..sysinfo.pdh_adapters import resolve_adapter_luid

            ep_name = self._single.ep_name
            for kind in [device] if device in ("npu", "gpu") else ["npu", "gpu"]:
                luid = resolve_adapter_luid(kind, ep_name=ep_name)
                if luid:
                    return luid
            return None
        except Exception:
            logger.debug("Could not resolve adapter LUID", exc_info=True)
            return None

    def _run_benchmark(self) -> PerfStats:
        """Execute benchmark iterations with timing."""
        if self.config.monitor:
            return self._run_benchmark_monitored()
        return self._run_benchmark_simple()

    def _run_benchmark_simple(self) -> PerfStats:
        """Execute benchmark without live monitoring."""
        assert self._inputs is not None
        total_iterations = self.config.warmup + self.config.iterations

        session = self._single._session
        with session.perf(warmup=self.config.warmup) as stats:
            _run_simple_loop(session, self._inputs, total_iterations)

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

        assert self._inputs is not None
        total_iterations = self.config.warmup + self.config.iterations

        if not HWMonitor.is_available():
            Console(stderr=True).print(
                "[yellow]Warning:[/yellow] HWMonitor unavailable on this system. "
                "Running without hardware monitoring."
            )
            return self._run_benchmark_simple()

        # Track the device actually being benchmarked so the monitor polls
        # GPU when --device gpu is specified, NPU when --device npu, etc.
        # ep_name lets the monitor resolve the exact LUID via ORT's autoEP
        # metadata so we follow the adapter the session actually binds to.
        ep_name = self._single.ep_name
        monitor_device = self._single.device or self.config.device or "auto"
        hw_monitor = HWMonitor(
            poll_interval_ms=_HW_POLL_INTERVAL_MS,
            device=monitor_device,
            ep_name=ep_name,
        )

        # EP-specific proof-of-execution monitor.
        # When QNN/OpenVINO monitors become real, add entries here.
        _ep_monitors: dict[EPName, Any] = {"VitisAIExecutionProvider": VitisAIMonitor}
        monitor_cls = _ep_monitors.get(ep_name) if ep_name else None
        ep_monitor: Any
        if monitor_cls and monitor_cls.is_available():
            ep_monitor = monitor_cls()
        else:
            ep_monitor = NullEPMonitor()

        session = self._single._session
        with (
            session.perf(warmup=self.config.warmup) as stats,
            hw_monitor as hw,
            ep_monitor as ep_mon,
        ):
            _run_monitored_loop(
                session,
                self._inputs,
                stats,
                hw,
                total_iterations=total_iterations,
                warmup=self.config.warmup,
                model_id=self.config.model_id,
                device=monitor_device,
            )

            # Store hardware metrics
            self._hw_metrics = hw.to_dict()
            ep_dict = ep_mon.to_dict()
            if ep_dict:  # NullEPMonitor returns {}, real monitors return data
                self._hw_metrics["ep_proof"] = ep_dict

        return stats

    def _collect_results(self, stats: PerfStats) -> BenchmarkResult:
        """Collect benchmark results from PerfStats."""
        io_config = self._single.io_config

        # Calculate throughput. Scale by the batch the session actually ran
        # (self._effective_batch), not the requested config.batch_size, which a
        # static-batch model silently ignores during input generation.
        mean_latency_sec = stats.mean_ms / 1000.0
        samples_per_sec = self._effective_batch / mean_latency_sec if mean_latency_sec > 0 else 0
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
            model_precision=io_config.get("precision"),
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
            effective_batch_size=self._effective_batch,
            # Actual values (resolved after build + compile)
            actual_device=self._single.device,
            actual_task=self._single.task or self.config.task or "auto-detected",
            actual_ep=self._single.ep_name,
            running_model_path=str(self._single.running_model_path),
            # Hardware monitor metrics (only present when --monitor is used)
            hw_monitor=getattr(self, "_hw_metrics", None),
            # Memory profile (only present when --memory is used)
            memory_profile=self._memory,
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
    no_compile: bool,
    output: Path | None,
    verbose: bool,
    console: Console,
    monitor: bool = False,
    device: str = "auto",
    ep: EPNameOrAlias | None = None,
    ep_options: dict[str, str] | None = None,
    precision: str = "auto",
    allow_unsupported_nodes: bool = False,
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
        no_quantize: If True, skip quantization during the per-module build.
        no_compile: If True, skip the build's compile stage for each module.
        output: Output JSON path, or None for auto-generated path.
        verbose: If True, log exceptions at DEBUG level.
        console: Rich console for output.
        monitor: If True, wrap each per-module benchmark with HWMonitor.
        device: Target device policy ("auto", "cpu", "gpu", "npu").
        ep: Explicit execution provider (e.g., "qnn", "dml"). Overrides
            device-to-provider mapping when set.
        ep_options: Runtime EP provider options (e.g. QNN
            ``htp_performance_mode``) forwarded to each per-module session.
        precision: Precision mode passed through to the build stage.
        allow_unsupported_nodes: If True, warn instead of failing the build when
            the analyzer reports unsupported nodes that persist.
    """
    import difflib
    import json as json_mod
    import tempfile

    from ..build import build_hf_model
    from ..config import SubmoduleClassNotFoundError, generate_hf_build_config
    from ..sysinfo import resolve_device
    from .build import _instantiate_parent_model

    resolved_device, _ = resolve_device(device=device, ep=ep)

    console.print(f"[dim]Generating module configs for {module_class}...[/dim]")

    try:
        module_configs = generate_hf_build_config(
            model_id=hf_model,
            task=task,
            module=module_class,
            device=resolved_device,
            precision=precision,
            ep=ep,
        )
    except SubmoduleClassNotFoundError as e:
        # User-error: --module pattern didn't match. List what's available so
        # the user can correct the typo without re-discovering classes manually.
        msg = [f"No modules matching '{e.class_name}' found."]
        suggestions = difflib.get_close_matches(e.class_name, e.available_classes, n=5)
        if suggestions:
            msg.append(f"Did you mean: {', '.join(suggestions)}?")
        if e.available_classes:
            msg.append("Available module class names in this model:")
            msg.append("  " + "\n  ".join(e.available_classes))
        raise click.UsageError("\n".join(msg)) from e
    except Exception as e:
        if verbose:
            logger.exception("Module config generation failed")
        raise click.ClickException(f"Error generating module configs: {e}") from e

    if not module_configs:
        # Defense-in-depth: _find_submodules_by_class now raises on no match,
        # but keep this branch for builders that might bypass it.
        raise click.UsageError(f"No modules matching '{module_class}' found")

    console.print(f"[dim]Found {len(module_configs)} {module_class} instances[/dim]")

    # Instantiate parent with init weights (no pretrained download).
    # Submodule configs intentionally drop `loader.task`, so re-resolve the
    # parent task from the model_id — the same path `generate_hf_build_config`
    # used to compute module_path. Without this, models whose `architectures`
    # field maps to a different task than `get_supported_tasks(model_type)[0]`
    # instantiate the wrong parent class and `get_submodule()` raises
    # AttributeError.
    model_type = module_configs[0].loader.model_type
    if not model_type:
        raise click.ClickException("module configs missing model_type")

    from ..loader import resolve_loader_config

    parent_loader_cfg, _, _, _resolution = resolve_loader_config(model_id=hf_model, task=task)
    parent_model = _instantiate_parent_model(model_type, task=parent_loader_cfg.task)

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

        # Skip quant/compile for faster iteration when requested. Quantization
        # and compilation are independent toggles (mirrors the single-model path).
        if no_quantize:
            cfg.quant = None
        if no_compile:
            cfg.compile = None

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            try:
                build_result = build_hf_model(
                    config=cfg,
                    output_dir=Path(tmpdir),
                    pytorch_model=submodule,
                    ep=ep,
                    device=resolved_device,
                    allow_unsupported_nodes=allow_unsupported_nodes,
                )

                # Benchmark using WinMLSession
                from ..session import WinMLSession

                session = WinMLSession(
                    str(build_result.final_onnx_path),
                    device=resolved_device,
                    ep=ep,
                    provider_options=ep_options,
                )
                io_cfg = session.io_config
                inputs = generate_random_inputs(io_cfg, batch_size=batch_size)

                # Compile session early so session.device is resolved for display
                session.compile()

                total_iters = warmup + iterations
                hw_ctx = None
                hw_metrics = None

                if monitor:
                    from ..session.monitor.hw_monitor import HWMonitor

                    if HWMonitor.is_available():
                        hw_ctx = HWMonitor(
                            poll_interval_ms=_HW_POLL_INTERVAL_MS,
                            device=resolved_device,
                            ep_name=session.ep_name,
                        )

                if hw_ctx:
                    # Drive the same live chart single-model mode uses so
                    # --monitor renders a per-module HW utilization chart
                    # instead of silently dumping metrics to JSON (issue #654).
                    with session.perf(warmup=warmup) as stats, hw_ctx as hw:
                        _run_monitored_loop(
                            session,
                            inputs,
                            stats,
                            hw,
                            total_iterations=total_iters,
                            warmup=warmup,
                            model_id=label,
                            device=resolved_device,
                        )
                        # Collect inside the `with` block: hw_ctx.__exit__
                        # stops the monitor, so to_dict() must read while it's
                        # still live (mirrors the single-model path).
                        hw_metrics = hw.to_dict()
                else:
                    with session.perf(warmup=warmup) as stats:
                        for _ in range(total_iters):
                            session.run(inputs)

                mod_stats = stats
                result_entry: dict[str, Any] = {
                    "module_path": module_path,
                    "running_model_path": str(session.running_model_path),
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
        output = generate_output_path(hf_model, module_class=module_class)

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


def _device_string(req_device: str, act_device: str, ep_name: EPName | None) -> str:
    device_str = f"{req_device} ({act_device})" if req_device != act_device else act_device
    if ep_name:
        device_str = f"{device_str} / {ep_name}"
    return device_str


def display_console_report(result: BenchmarkResult, console: Console) -> None:
    """Display benchmark results in formatted console output."""
    # Info section — show "requested (resolved)" when they differ
    console.print()

    req_device = result.config.device
    act_device = result.actual_device
    device_str = _device_string(req_device, act_device, result.actual_ep)
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

    # I/O tensor info is printed before the benchmark via _print_model_info()

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
    throughput_line = f"[bold]Throughput:[/bold] {result.samples_per_sec:.2f} samples/sec"
    if result.effective_batch_size != 1:
        throughput_line += f" [dim](batch {result.effective_batch_size})[/dim]"
    console.print(throughput_line)
    # Flag when the requested batch couldn't be honored so a static-batch model
    # doesn't look like it silently ran the requested batch.
    if result.config.batch_size != result.effective_batch_size:
        console.print(
            f"  [yellow]Note:[/yellow] requested batch {result.config.batch_size} "
            f"could not be applied (model has a static batch of "
            f"{result.effective_batch_size})."
        )

    # Hardware section (only when monitoring was active)
    if result.hw_monitor:
        console.print()
        console.print("[bold]Hardware (during benchmark)[/bold]")
        cpu = result.hw_monitor.get("cpu", {})
        ram = result.hw_monitor.get("ram", {})
        # to_dict() emits both "npu" (always) and "gpu" (when monitoring GPU).
        # device_kind is None when CPU/RAM-only — drop the adapter line entirely
        # rather than printing zeroed "NPU: 0.0% avg".
        device_kind = result.hw_monitor.get("device_kind")
        if device_kind in ("npu", "gpu"):
            adapter = result.hw_monitor.get(device_kind, {})
            console.print(
                f"  {device_kind.upper()}: {adapter.get('mean_pct', 0):.1f}% avg, "
                f"{adapter.get('peak_pct', 0):.1f}% peak  |  "
                f"CPU: {cpu.get('mean_pct', 0):.1f}% avg  |  "
                f"RAM: {ram.get('used_mb', 0):.0f} MB"
            )
        else:
            console.print(
                f"  CPU: {cpu.get('mean_pct', 0):.1f}% avg  |  RAM: {ram.get('used_mb', 0):.0f} MB"
            )

    # Memory section (only when --memory is enabled)
    if result.memory_profile:
        mem = result.memory_profile
        console.print()
        console.print("[bold]Memory:[/bold]")
        console.print(
            f"  RAM:  {mem['rss_after_inference_mb']:.1f} MB -> "
            f"model load: {mem['rss_model_load_delta_mb']:+.1f} MB  |  "
            f"inference: {mem['rss_inference_delta_mb']:+.1f} MB  |  "
            f"total: {mem['rss_total_delta_mb']:+.1f} MB"
        )
        vram_local = mem.get("vram_local_after_inference_mb", 0)
        vram_shared = mem.get("vram_shared_after_inference_mb", 0)
        if vram_local > 0 or vram_shared > 0:
            console.print(
                f"  VRAM: {vram_local:.1f}/{vram_shared:.1f} MB (local/shared) -> "
                f"model load: {mem['vram_local_model_load_delta_mb']:+.1f}/"
                f"{mem['vram_shared_model_load_delta_mb']:+.1f} MB  |  "
                f"inference: {mem['vram_local_inference_delta_mb']:+.1f}/"
                f"{mem['vram_shared_inference_delta_mb']:+.1f} MB  |  "
                f"total: {mem['vram_local_total_delta_mb']:+.1f}/"
                f"{mem['vram_shared_total_delta_mb']:+.1f} MB"
            )

    console.print()


def write_json_report(result: BenchmarkResult, output_path: Path) -> None:
    """Write benchmark results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)


def _composite_report_dict(
    results: dict[str, BenchmarkResult],
    *,
    model_id: str,
    task: str | None,
) -> dict[str, Any]:
    """Build the combined JSON report for a composite model's sub-models."""
    return {
        "model_id": model_id,
        "task": task,
        "component_count": len(results),
        "components": {name: result.to_dict() for name, result in results.items()},
    }


def report_composite_results(
    results: dict[str, BenchmarkResult],
    *,
    console: Console,
    json_mode: bool,
    output_path: Path,
    model_id: str,
    task: str | None,
) -> None:
    """Display and persist per-sub-model results for a composite model.

    Composite models (e.g. CLIP/SigLIP dual-encoders) have no single ORT
    session; each sub-model is benchmarked individually (like ``--module``)
    and reported as its own summary row rather than timing the aggregate
    ``forward()`` pass. The combined JSON nests each sub-model's full
    ``BenchmarkResult.to_dict()`` under ``components``.
    """
    combined = _composite_report_dict(results, model_id=model_id, task=task)

    if json_mode:
        click.echo(json.dumps(combined, indent=2))
    else:
        table = Table(title="Per-Sub-Model Perf", show_header=True)
        table.add_column("Sub-Model", style="cyan")
        table.add_column("Task")
        table.add_column("Device")
        table.add_column("Mean (ms)", justify="right")
        table.add_column("P90 (ms)", justify="right")
        table.add_column("Min (ms)", justify="right")
        table.add_column("Max (ms)", justify="right")
        for name, result in results.items():
            device_str = _device_string(
                result.config.device, result.actual_device, result.actual_ep
            )
            table.add_row(
                name,
                result.actual_task,
                device_str,
                f"{result.mean_ms:.2f}",
                f"{result.p90_ms:.2f}",
                f"{result.min_ms:.2f}",
                f"{result.max_ms:.2f}",
            )
        console.print()
        console.print(table)
        console.print()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)


def generate_output_path(model_id: str, *, module_class: str | None = None) -> Path:
    r"""Generate default output path under the user's cache directory.

    Returns ``~/.cache/winml/perf/<slug>[/<module_class>]/<timestamp>.json``
    so repeated runs accumulate under a stable per-model directory without
    polluting CWD (see #551). The timestamp is generated at call time using
    local time, format ``YYYYMMDD-HHMMSS``.

    For ONNX inputs, the file stem is used as the slug
    (e.g., ``model.onnx`` -> ``model``). For HF model IDs, ``/`` and ``\``
    are replaced with ``_`` (e.g., ``microsoft/resnet-50`` ->
    ``microsoft_resnet-50``).
    """
    p = Path(model_id)
    slug = p.stem if p.suffix.lower() == ".onnx" else model_id.replace("/", "_").replace("\\", "_")

    out_dir = Path.home() / ".cache" / "winml" / "perf" / slug
    if module_class:
        out_dir = out_dir / module_class

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return out_dir / f"{timestamp}.json"


# =============================================================================
# Shared benchmark helpers
# =============================================================================


def _print_model_info(
    io_config: dict,
    *,
    task: str | None = None,
    req_device: str = "auto",
    act_device: str = "auto",
    ep_name: EPName | None = None,
) -> None:
    """Print model I/O metadata before the benchmark starts."""
    console = Console(stderr=True)
    console.print()
    device_line = _device_string(req_device, act_device, ep_name)
    console.print(f"[dim]Device:[/dim]      {device_line}")
    if task:
        console.print(f"[dim]Task:[/dim]        {task}")

    precision = io_config.get("precision")
    if precision:
        console.print(f"[dim]Model Precision:[/dim]   {precision}")

    names = io_config.get("input_names", [])
    shapes = io_config.get("input_shapes", [])
    types = io_config.get("input_types", [])
    if names:
        label = "[dim]Inputs:[/dim]      "
        pad = "             "
        for i, name in enumerate(names):
            shape = shapes[i] if i < len(shapes) else []
            dtype = str(types[i]) if i < len(types) else ""
            shape_str = f"{shape!s}"
            line = f"{name:<20s} {shape_str:<22s} {dtype}"
            console.print(f"{label if i == 0 else pad}{line}")

    out_names = io_config.get("output_names", [])
    out_shapes = io_config.get("output_shapes", [])
    if out_names:
        label = "[dim]Outputs:[/dim]     "
        pad = "             "
        for i, name in enumerate(out_names):
            shape = out_shapes[i] if i < len(out_shapes) else []
            console.print(f"{label if i == 0 else pad}{name:<20s} {shape!s}")

    console.print()


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
    """Run the benchmark iteration loop with live hardware monitoring."""
    display = LiveMonitorDisplay(
        total_iterations=total_iterations,
        warmup=warmup,
        model_id=model_id,
        device=device,
        device_kind=getattr(hw, "device_kind", None),
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
    """Run the benchmark iteration loop with periodic debug logging."""
    for i in range(total_iterations):
        session.run(inputs)

        if (i + 1) % max(1, total_iterations // 10) == 0:
            logger.debug("Progress: %d/%d", i + 1, total_iterations)


# =============================================================================
# CLI Command
# =============================================================================


@click.command("perf")
@cli_utils.model_option(required=False)
@click.option(
    "--task",
    type=str,
    default=None,
    help="Explicit task (e.g., 'image-classification'). Auto-detected if not specified.",
)
@click.option(
    "--iterations",
    type=click.IntRange(min=1),
    default=100,
    show_default=True,
    help="Number of benchmark iterations (must be > 0)",
)
@click.option(
    "--warmup",
    type=click.IntRange(min=0),
    default=10,
    show_default=True,
    help="Number of warmup iterations (excluded from statistics; must be >= 0)",
)
@cli_utils.device_option(required=False, default="auto", include_auto=True)
@cli_utils.precision_option()
@cli_utils.ep_option(
    required=False,
    optional_message="Overrides device-to-provider mapping.",
)
@cli_utils.ep_options_option(
    optional_message="Applied to both HuggingFace model IDs and ONNX file inputs.",
)
@cli_utils.output_option(
    "Output JSON file path. Defaults to "
    "'~/.cache/winml/perf/<model_slug>[/<module_class>]/<timestamp>.json'."
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
    "--quantize/--no-quantize",
    "quantize",
    default=True,
    show_default=True,
    help="Include quantization during model build (use --no-quantize to skip)",
)
@click.option(
    "--rebuild/--no-rebuild",
    default=False,
    show_default=True,
    help="Force rebuild even if cached artifacts exist",
)
@click.option(
    "--ignore-cache/--no-ignore-cache",
    default=False,
    show_default=True,
    help="Build from scratch in a temp folder (discard after benchmarking)",
)
@cli_utils.skip_build_option()
@cli_utils.compile_option(
    default=True,
    help_text="Compile the model during build. Default: off (skip compilation); "
    "use --compile to enable.",
)
@cli_utils.allow_unsupported_nodes_option()
@click.option(
    "--module",
    "module_class",
    default=None,
    type=str,
    help="PyTorch module class name (NOT a dotted module path) for per-module "
    "benchmarking. Every instance of the class in the model is built and "
    "benchmarked separately. Example: '--module BertAttention' (correct), "
    "not '--module encoder.layer.0.attention' (a path, will not match).",
)
@click.option(
    "--monitor/--no-monitor",
    default=False,
    show_default=True,
    help="Show live hardware utilization chart for the benchmarked device (NPU, GPU, or CPU)",
)
@click.option(
    "--memory/--no-memory",
    default=True,
    show_default=True,
    help="Measure process and device memory at each benchmark phase",
)
@click.option(
    "--op-tracing",
    "op_tracing",
    type=click.Choice(["basic", "detail"], case_sensitive=False),
    default=None,
    help="Enable operator-level profiling (requires onnxruntime-qnn)",
    hidden=True,  # Not ready, so hide from --help for now
)
@cli_utils.format_option()
@cli_utils.build_config_option()
@cli_utils.verbosity_options()
@click.pass_context
def perf(
    ctx: click.Context,
    model: str | None,
    task: str | None,
    iterations: int,
    warmup: int,
    device: str,
    precision: str,
    ep: EPNameOrAlias | None,
    ep_options: tuple[str, ...],
    output: Path | None,
    batch_size: int,
    shape_config_path: Path | None,
    quantize: bool,
    rebuild: bool,
    ignore_cache: bool,
    skip_build: bool,
    no_compile: bool,
    allow_unsupported_nodes: bool,
    module_class: str | None,
    monitor: bool,
    memory: bool,
    op_tracing: str | None,
    output_format: cli_utils.OutputFormat,
    verbose: int,
    quiet: bool,
    config_file: Path | None,
) -> None:
    r"""Benchmark model inference performance.

    Measures latency and throughput using random input data generated
    from the model's I/O configuration.

    Accepts both HuggingFace model IDs and local .onnx files. Both flow
    through the same PerfBenchmark pipeline (optimize → [quantize] → [compile]
    minus export for ONNX inputs), so latency numbers are directly comparable
    between the two inputs.

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

        # Pass runtime EP provider options (repeatable)
        winml perf -m model.onnx --device npu --ep-options htp_performance_mode=burst

        # Per-module benchmarking
        winml perf -m bert-base-uncased --module BertAttention

        # Operator-level profiling (QNN NPU)
        winml perf -m model.onnx --op-tracing basic
    """
    if not model:
        raise click.UsageError("A model is required via -m/--model.")

    hf_model = model

    # Apply build config defaults (CLI explicit options take precedence).
    # Read raw JSON so missing keys are distinguishable from dataclass defaults.
    if config_file is not None:
        _, raw_cfg = cli_utils.load_build_config(config_file)
        lc = raw_cfg.get("loader") or {}
        cc = raw_cfg.get("compile") or {}
        if not cli_utils.is_cli_provided(ctx, "task") and "task" in lc:
            task = lc["task"]
        if not cli_utils.is_cli_provided(ctx, "ep") and "execution_provider" in cc:
            ep = cc["execution_provider"]

    # Merge top-level -v/-q with subcommand-level flags so either position works.
    verbose, quiet = cli_utils.resolve_verbosity(ctx, verbose, quiet)
    configure_logging(verbosity=verbose, quiet=quiet)

    # Runtime EP provider options (e.g. QNN htp_performance_mode) forwarded to
    # the inference session for both HF model IDs and ONNX file inputs.
    ep_provider_options = cli_utils.parse_ep_options(ep_options)

    json_mode = output_format == "json"
    console = Console(stderr=True) if json_mode else Console()

    # =========================================================================
    # MODULE MODE: per-module build + benchmark
    # =========================================================================
    if module_class:
        # Reject --module on a pre-exported ONNX path. Submodule discovery
        # walks a live nn.Module graph (torchinfo), which an ONNX file does
        # not carry. Without this guard, the user sees a misleading
        # "not a valid JSON file" error from AutoConfig.from_pretrained
        # trying to load the .onnx as an HF config dir (issue #553).
        if Path(hf_model).suffix.lower() == ".onnx":
            raise click.UsageError(
                f"--module is not supported for ONNX files. "
                f"Submodule benchmarking requires a HuggingFace model ID "
                f"(e.g., 'microsoft/resnet-50'), not '{hf_model}'."
            )
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
            no_quantize=not quantize,
            no_compile=no_compile,
            output=output,
            verbose=bool(verbose),
            console=console,
            monitor=monitor,
            device=device.lower(),
            ep=ep,
            ep_options=ep_provider_options,
            precision=precision.lower(),
            allow_unsupported_nodes=allow_unsupported_nodes,
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
        no_quantize=not quantize,
        rebuild=rebuild,
        ignore_cache=ignore_cache,
        skip_build=skip_build,
        no_compile=no_compile,
        allow_unsupported_nodes=allow_unsupported_nodes,
        monitor=monitor,
        memory=memory,
        ep=ep,
        ep_options=ep_provider_options,
        shape_config=shape_config,
    )

    try:
        model_path = Path(hf_model)
        is_onnx = model_path.suffix.lower() == ".onnx"

        if is_onnx:
            # Validate file existence up front; otherwise WinMLAutoModel would
            # fall through to HF loading and surface a confusing
            # "not a valid JSON file" error from AutoConfig.
            if not model_path.exists():
                raise FileNotFoundError(f"ONNX file not found: {model_path}")
            if shape_config:
                console.print(
                    "[yellow]Warning:[/yellow] --shape-config is ignored for "
                    "pre-exported ONNX files (shapes are baked into the model)."
                )
                config.shape_config = None
            console.print(f"[dim]Benchmarking ONNX:[/dim] {model_path}")
        else:
            if precision != "auto":
                console.print(f"[dim]Precision: {precision} (applied during model build)[/dim]")
            console.print(f"[dim]Loading model:[/dim] {hf_model}")

        benchmark = PerfBenchmark(config)
        result = benchmark.run()

        # Composite models (e.g. CLIP/SigLIP dual-encoders) have no single ORT
        # session; each sub-model is benchmarked individually and reported as
        # its own row (like --module), not as one aggregate forward() timing.
        if isinstance(result, dict):
            if op_tracing:
                console.print(
                    "[yellow]Warning:[/yellow] --op-tracing is not supported for "
                    "composite models and will be skipped."
                )
            report_composite_results(
                result,
                console=console,
                json_mode=json_mode,
                output_path=output,
                model_id=hf_model,
                task=task,
            )
            console.print(f"[green]Results saved to:[/green] {output}")
            return

        # Display results
        if json_mode:
            click.echo(json.dumps(result.to_dict(), indent=2))
        else:
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

            from ..optracing import (
                display_op_trace_report,
                get_tracer,
                write_op_trace_json,
            )

            # Determine the ONNX model path from the benchmark flow.
            # For HF models the ONNX is built internally by PerfBenchmark.
            try:
                onnx_for_trace = (
                    model_path if is_onnx else getattr(benchmark._model, "_onnx_path", None)
                )
                if onnx_for_trace is None:
                    raise AttributeError("benchmark._model not initialized")
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
        # User-error: bad model path. UsageError so the exit code (2) matches
        # the convention used by Click for argument problems.
        raise click.UsageError(f"Model not found: {e}") from e

    except Exception as e:
        if verbose:
            logger.exception("Benchmark failed")
        raise click.ClickException(f"Benchmark failed: {e}") from e
