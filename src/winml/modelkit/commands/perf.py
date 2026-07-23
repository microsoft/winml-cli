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
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

import click
import numpy as np
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ..utils import cli as cli_utils
from ..utils.constants import EPName, EPNameOrAlias
from ..utils.logging import configure_logging
from ..utils.model_input import ModelInputKind, classify_model_input
from ._live_chart import LiveMonitorDisplay


if TYPE_CHECKING:
    import contextlib

    from ..models.winml.base import WinMLPreTrainedModel
    from ..models.winml.composite_model import WinMLCompositeModel
    from ..session.stats import PerfStats

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Hardware monitor polling interval (milliseconds)
_HW_POLL_INTERVAL_MS = 200

# Inference runtimes selectable via ``--runtime`` (closed set; mirrors the
# ``--compiler`` / ``COMPILER_NAMES`` convention in utils.constants):
#   "winml"       -> single-shot ONNX inference (default)
#   "winml-genai" -> onnxruntime-genai decoder-pipeline generation
RuntimeName = Literal["winml", "winml-genai"]
RUNTIME_NAMES: tuple[RuntimeName, ...] = get_args(RuntimeName)

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
    submodel: str | None = None
    device: str = "auto"
    precision: str = "auto"
    iterations: int = 100
    warmup: int = 10
    batch_size: int = 1
    output_path: Path | None = None
    no_quantize: bool = False
    no_optimize: bool = False
    no_analyze: bool = False
    max_optim_iterations: int | None = None
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
    export_overrides: dict[str, Any] | None = None
    # Path to a .npz file of real input tensors. When set, benchmarking uses
    # these instead of randomly generated inputs (single-model path only).
    input_data: Path | None = None


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
    shape_config: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Generate random inputs based on model io_config.

    Uses modelkit.core.model_input_generator for spec-driven generation.
    Returns numpy arrays directly (no torch dependency).

    Args:
        io_config: Model I/O configuration from WinMLSession.io_config.
            Expected keys: ``input_names``, ``input_shapes``, ``input_types``.
            Optional keys: ``input_value_ranges`` -- a dict mapping input
            names to ``[low, high)`` integer ranges sourced from the build
            config; ``input_symbolic_shapes`` -- a list of shapes whose
            dynamic dims hold the declared symbolic dim_param name.
        batch_size: Override batch dimension
        shape_config: Optional overrides for dynamic dimensions. Two forms
            are supported and may be mixed:

            * Per-input full-shape override:
              ``{"input_points": [1, 1, 1, 2], ...}`` -- the value is used
              as the resolved shape verbatim.
            * Symbolic dim override:
              ``{"num_points_per_image": 1, "num_boxes_per_image": 1}`` --
              applied to any dim whose ``dim_param`` matches the key.

            Symbolic overrides take precedence over positional defaults.

    Returns:
        Dictionary of input_name -> numpy array
    """
    from ..core import generate_dummy_inputs_from_specs

    symbolic_shapes = io_config.get("input_symbolic_shapes") or [
        [None] * len(s) for s in io_config["input_shapes"]
    ]
    overrides = shape_config or {}

    specs: dict[str, dict[str, Any]] = {}
    for name, shape, symbolic, dtype_str in zip(
        io_config["input_names"],
        io_config["input_shapes"],
        symbolic_shapes,
        io_config["input_types"],
        strict=True,
    ):
        resolved_shape = _resolve_shape(
            shape=shape,
            symbolic_shape=symbolic,
            input_name=name,
            batch_size=batch_size,
            shape_config=overrides,
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
    symbolic_shape: list | tuple | None = None,
    shape_config: dict[str, Any] | None = None,
) -> tuple[int, ...]:
    """Resolve dynamic dimensions in shape.

    Resolution priority for each dim:
      1. ``shape_config[input_name]`` -- full per-input shape override.
      2. ``shape_config[dim_param]`` -- symbolic dim override (when the
         ONNX graph exposed a ``dim_param`` for this dim).
      3. ``batch_size`` for the first dim.
      4. ``DYNAMIC_DIM_DEFAULTS`` positional fallback (defaults to 128).
    """
    overrides = shape_config or {}

    # Form 1: full per-input shape override.
    if input_name in overrides and isinstance(overrides[input_name], (list, tuple)):
        return tuple(int(d) for d in overrides[input_name])

    if shape is None:
        logger.warning("Shape unknown for '%s', using (batch_size,)", input_name)
        return (batch_size,)

    sym = list(symbolic_shape) if symbolic_shape is not None else [None] * len(shape)
    resolved = []
    for i, dim in enumerate(shape):
        if dim is None or dim == -1 or isinstance(dim, str):
            # Dynamic dimension - resolve
            sym_name = sym[i] if i < len(sym) else None
            if isinstance(sym_name, str) and sym_name in overrides:
                value = overrides[sym_name]
                if isinstance(value, (list, tuple, dict)):
                    raise click.ClickException(
                        f"--shape-config symbolic dimension '{sym_name}' must be "
                        f"a scalar integer, got {type(value).__name__}. Use the "
                        f"input name '{input_name}' for full-shape overrides."
                    )
                if isinstance(value, bool):
                    raise click.ClickException(
                        f"--shape-config symbolic dimension '{sym_name}' must be "
                        f"a scalar integer, got {value!r}."
                    )
                try:
                    coerced_value = int(value)
                except (OverflowError, TypeError, ValueError) as e:
                    raise click.ClickException(
                        f"--shape-config symbolic dimension '{sym_name}' must be "
                        f"a scalar integer, got {value!r}."
                    ) from e
                try:
                    is_integral = isinstance(value, str) or value == coerced_value
                except (TypeError, ValueError):
                    is_integral = False
                if not is_integral:
                    raise click.ClickException(
                        f"--shape-config symbolic dimension '{sym_name}' must be "
                        f"a scalar integer, got {value!r}."
                    )
                resolved.append(coerced_value)
            elif i == 0:
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


def load_input_data(
    path: Path,
    io_config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Load benchmark inputs from a ``.npz`` file, validated against the model.

    Thin wrapper over the shared
    :func:`winml.modelkit.datasets.input_data.load_input_data`, which is also
    used by ``winml eval --mode compare --input-data``. Imported lazily so
    ``winml perf`` startup does not pull in the datasets package unless
    ``--input-data`` is actually used.
    """
    from ..datasets.input_data import load_input_data as _load_input_data

    return _load_input_data(path, io_config)


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

    Single-input assumption: only the first batched input is inspected. For
    multimodal or encoder-decoder models whose batched inputs disagree on the
    leading dim (e.g. an image batch of 4 alongside a differently batched
    tensor), the reported value reflects only the first batched input.
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
        # Concrete device + EP resolved from the config's request, populated by
        # _resolve_device_ep() on the first call (before the build). The config
        # keeps the raw request (e.g. "auto"); these hold what actually drives
        # the build and inference.
        self._resolved_device: str | None = None
        self._resolved_ep: EPNameOrAlias | None = None

    def _resolve_device_ep(self) -> None:
        """Resolve the concrete target device + EP, failing fast on a bad combo.

        Idempotent: resolves once, then returns cached values. Called at the
        start of model loading so an unavailable/invalid device+EP raises here —
        before the export/optimize/quantize/compile pipeline runs — instead of
        only surfacing at session.compile(). Deriving a concrete EP also lets the
        build's static analyzer target one EP instead of aggregating across all
        of them (WinMLAutoModel itself stays permissive: ep=None is a valid
        library mode).

        Raises:
            ValueError: If the requested device/EP combination is unavailable
                or invalid (propagated from ``resolve_device``).
        """
        if self._resolved_device is not None:
            return

        from ..sysinfo import resolve_device, resolve_eps

        # resolve_device() availability-checks even when --ep is explicit, so a
        # named-but-absent EP is caught here too.
        resolved_device, _ = resolve_device(device=self.config.device, ep=self.config.ep)
        if self.config.ep is not None:
            # Keep the user's EP (alias or canonical) verbatim — downstream
            # stages normalize it. Only derive one when the user gave none.
            resolved_ep: EPNameOrAlias | None = self.config.ep
        else:
            device_eps = resolve_eps(resolved_device)
            resolved_ep = device_eps[0] if device_eps else None

        self._resolved_device = resolved_device
        self._resolved_ep = resolved_ep

    @property
    def resolved_device(self) -> str | None:
        """Concrete device driving the build/inference (``None`` until resolved)."""
        return self._resolved_device

    @property
    def resolved_ep(self) -> EPNameOrAlias | None:
        """Concrete EP driving the build/inference (``None`` until resolved)."""
        return self._resolved_ep

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
            # Composite-ness is only known after _load_model, so this guard
            # can't live with the up-front --module / --runtime checks. Without
            # it, each sub-model's child benchmark calls load_input_data with a
            # single .npz that can't match two different encoders' input names,
            # surfacing as a re-wrapped "Sub-model '…' failed" RuntimeError
            # instead of a clean up-front error.
            if self.config.input_data is not None:
                raise click.UsageError(
                    "--input-data is not supported for composite (dual-encoder) "
                    "models; each sub-model has its own inputs that a single "
                    ".npz cannot address."
                )
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
            actual_shapes=(
                {name: tuple(arr.shape) for name, arr in self._inputs.items()}
                if self._inputs
                else None
            ),
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

        # Resolve the concrete device + EP first so a bad combo fails fast,
        # before from_pretrained/from_onnx kick off the build pipeline.
        self._resolve_device_ep()

        model_id = self.config.model_id
        model_path = Path(model_id)
        is_onnx = model_path.suffix.lower() == ".onnx"
        if is_onnx and not model_path.exists():
            # Surface a clear error for programmatic callers. The CLI guards
            # this earlier, but without this check from_pretrained would fall
            # through to HF loading and produce a confusing "not a valid JSON
            # file" error from AutoConfig.
            raise FileNotFoundError(f"ONNX file not found: {model_path}")

        # Composite auto-detection. A bare seq2seq model such as T5 auto-detects
        # to a granular single-model task (text2text-generation) and would
        # benchmark only the decoder.
        #
        # Why this differs from build/export: those commands *fan out* into one
        # independent build per sub-model, so they use resolve_composite_components
        # -> a {name: sub_model_task} map (encoder=feature-extraction,
        # decoder=text2text-generation) and never construct a composite object.
        # perf instead loads ONE live WinMLCompositeModel and benchmarks its
        # sub-models, which requires a registered composite *pipeline* task
        # (translation/summarization) to route WinMLAutoModel.from_pretrained --
        # a different namespace from the sub-model tasks. resolve_composite_load_task
        # bridges detection to that loadable pipeline task. Explicit --task and
        # ONNX inputs keep their resolved task untouched.
        resolved_task = self.config.task
        if not is_onnx and resolved_task is None:
            from ..loader.resolution import resolve_composite_load_task

            try:
                resolved_task = resolve_composite_load_task(model_id)
            except OSError as e:
                # Config not resolvable (e.g. invalid or unreachable model id).
                # Fall back to single-model loading; from_pretrained re-attempts
                # the config load and surfaces a clear error if the id is truly
                # bad. Mirrors build's composite-detection guard.
                logger.debug("Composite detection unavailable (config not resolvable): %s", e)

        # Only override config for explicitly requested build/export changes.
        override: WinMLBuildConfig | dict[str, Any] | None = None
        if self.config.export_overrides:
            if is_onnx:
                raise ValueError(
                    "Export overrides are only supported for HuggingFace model inputs."
                )
            override_dict: dict[str, Any] = {"export": self.config.export_overrides}
            if self.config.no_quantize:
                override_dict["quant"] = None
            override = override_dict
        elif self.config.no_quantize:
            override = WinMLBuildConfig(quant=None)

        # Cache control: --ignore-cache -> temp dir, --rebuild -> overwrite cache
        use_cache = not self.config.ignore_cache
        force_rebuild = self.config.rebuild or self.config.ignore_cache

        common_kwargs: dict[str, Any] = {
            "task": resolved_task,
            "config": override,
            "device": self._resolved_device or self.config.device,
            "precision": self.config.precision,
            "ep": self._resolved_ep,
            "provider_options": self.config.ep_options,
            "use_cache": use_cache,
            "force_rebuild": force_rebuild,
            "shape_config": self.config.shape_config,
            "allow_unsupported_nodes": self.config.allow_unsupported_nodes,
            "no_compile": self.config.no_compile,
            # optimize/analyze/max-optim toggles, forwarded by WinMLAutoModel to
            # build_hf_model / build_onnx_model. Shared mapping with build/eval.
            **cli_utils.build_pipeline_extra_kwargs(
                optimize=not self.config.no_optimize,
                analyze=not self.config.no_analyze,
                max_optim_iterations=self.config.max_optim_iterations,
            ),
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
        """Generate random inputs, or load real inputs from a .npz file."""
        io_config = self._single.io_config
        if self.config.input_data is not None:
            self._inputs = load_input_data(self.config.input_data, io_config)
            self._effective_batch = effective_batch_size(
                self._inputs,
                io_config["input_names"],
                self.config.batch_size,
            )
            logger.info(
                "Loaded benchmark inputs from %s (effective batch %d)",
                self.config.input_data,
                self._effective_batch,
            )
            return

        self._inputs = generate_random_inputs(
            io_config=io_config,
            batch_size=self.config.batch_size,
            shape_config=self.config.shape_config,
        )
        self._effective_batch = effective_batch_size(
            self._inputs,
            io_config["input_names"],
            self.config.batch_size,
        )
        if self._effective_batch != self.config.batch_size:
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
        device = self._single.device or self._resolved_device or self.config.device
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
    no_optimize: bool,
    no_analyze: bool,
    max_optim_iterations: int | None,
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
    rebuild: bool = False,
    ignore_cache: bool = False,
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
        no_optimize: If True, skip graph optimization during the per-module build.
        no_analyze: If True, skip the analyzer loop (forces max_optim_iterations=0).
        max_optim_iterations: Max autoconf re-optimization rounds (None = default 3).
        no_compile: If True, skip the build's compile stage for each module.
        output: Output JSON path, or None for auto-generated path.
        verbose: If True, log exceptions at DEBUG level.
        console: Rich console for output.
        monitor: If True, wrap each per-module benchmark with HWMonitor.
        device: Target device policy ("auto", "cpu", "gpu", "npu").
        ep: Explicit execution provider (e.g., "qnn", "dml"). Overrides
            device-to-provider mapping when set. When ``None``, a concrete EP is
            derived from the resolved device so the analyzer targets one EP.
        ep_options: Runtime EP provider options (e.g. QNN
            ``htp_performance_mode``) forwarded to each per-module session.
        precision: Precision mode passed through to the build stage.
        allow_unsupported_nodes: If True, warn instead of failing the build when
            the analyzer reports unsupported nodes that persist.
        rebuild: If True, overwrite cached per-module artifacts and re-run the
            build (mirrors the single-model ``--rebuild``).
        ignore_cache: If True, build each module in a throwaway temp dir and
            always rebuild, discarding artifacts afterward (mirrors the
            single-model ``--ignore-cache``).
    """
    import contextlib
    import difflib
    import json as json_mod
    import tempfile

    from ..build import build_hf_model
    from ..cache import get_cache_dir, get_cache_key, get_model_dir
    from ..config import SubmoduleClassNotFoundError, generate_hf_build_config
    from ..loader.task import get_task_abbrev
    from ..sysinfo import resolve_device, resolve_eps
    from .build import _instantiate_parent_model

    resolved_device, _ = resolve_device(device=device, ep=ep)
    # Derive a concrete EP when none was given so each per-module build's static
    # analyzer targets one EP instead of ep=None (which aggregates across all
    # EPs and warns; see #931). An explicit EP is kept verbatim — downstream
    # stages normalize it.
    if ep is None:
        device_eps = resolve_eps(resolved_device)
        ep = device_eps[0] if device_eps else None

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

    # Cache control mirrors auto.py / the single-model path:
    #   --ignore-cache -> build each module in a throwaway temp dir, always
    #                     rebuild, discard afterward
    #   --rebuild      -> reuse the persistent model dir but overwrite artifacts
    # Each module's cache_key folds in loader.module_path (and its I/O shapes),
    # so sibling instances of the same class get distinct keys and coexist in
    # the shared model dir without colliding.
    use_cache = not ignore_cache
    force_rebuild = rebuild or ignore_cache
    task_abbrev = get_task_abbrev(parent_loader_cfg.task) if parent_loader_cfg.task else "module"
    cache_model_dir = get_model_dir(hf_model, cache_dir=get_cache_dir()) if use_cache else None

    # Optimize/analyze toggles aren't part of ``cfg``; fold them into the cache
    # key so e.g. a later ``--no-optimize`` run doesn't silently reuse a cached
    # optimized artifact (mirrors the single-model path in auto.py).
    build_control_kwargs = cli_utils.build_pipeline_extra_kwargs(
        optimize=not no_optimize,
        analyze=not no_analyze,
        max_optim_iterations=max_optim_iterations,
    )

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

        # Compute the cache key AFTER the quant/compile mutations above so it
        # reflects what is actually built. Build controls (optimize/analyze
        # toggles) are folded in too since they aren't part of ``cfg``.
        cache_key = get_cache_key(task_abbrev, cfg.generate_cache_key(), build_control_kwargs)

        # Persistent model dir (reused across runs) when caching, else a
        # throwaway temp dir that is removed when the with-block exits.
        build_dir_ctx: Any = (
            contextlib.nullcontext(cache_model_dir)
            if use_cache
            else tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        )
        with build_dir_ctx as build_dir_raw:
            build_dir = Path(build_dir_raw)
            try:
                build_result = build_hf_model(
                    config=cfg,
                    output_dir=build_dir,
                    pytorch_model=submodule,
                    rebuild=force_rebuild,
                    cache_key=cache_key,
                    ep=ep,
                    device=resolved_device,
                    allow_unsupported_nodes=allow_unsupported_nodes,
                    **build_control_kwargs,
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


def generate_output_path(
    model_id: str, *, module_class: str | None = None, submodel: str | None = None
) -> Path:
    r"""Generate default output path under the user's cache directory.

    Returns ``~/.cache/winml/perf/<slug>[/<module_class>][/<submodel>]/<timestamp>.json``
    so repeated runs accumulate under a stable per-model directory without
    polluting CWD (see #551). The timestamp is generated at call time using
    local time, format ``YYYYMMDD-HHMMSS``.

    For ONNX inputs, the file stem is used as the slug
    (e.g., ``model.onnx`` -> ``model``). For HF model IDs, ``/`` and ``\``
    are replaced with ``_`` (e.g., ``microsoft/resnet-50`` ->
    ``microsoft_resnet-50``). A ``submodel`` (composite sub-component) is nested
    under its own directory so per-sub-model reports don't collide.
    """
    p = Path(model_id)
    slug = p.stem if p.suffix.lower() == ".onnx" else model_id.replace("/", "_").replace("\\", "_")

    out_dir = Path.home() / ".cache" / "winml" / "perf" / slug
    if module_class:
        out_dir = out_dir / module_class
    if submodel:
        out_dir = out_dir / submodel

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return out_dir / f"{timestamp}.json"


# =============================================================================
# Shared benchmark helpers
# =============================================================================


def _format_input_shape(shape: list, actual: tuple | None) -> str:
    """Render a declared input shape, marking dynamic dims as ``dynamic``.

    A dynamic dimension (declared as ``None``) is shown as ``dynamic(<n>)``
    where ``<n>`` is the concrete size the generated input data actually used
    for that axis, so the real batch/sequence sizes stay visible alongside the
    fact that the model left them free.
    """
    dims: list[str] = []
    for i, dim in enumerate(shape):
        if dim is None:
            if actual is not None and i < len(actual):
                dims.append(f"dynamic({actual[i]})")
            else:
                dims.append("dynamic")
        else:
            dims.append(str(dim))
    return f"[{', '.join(dims)}]"


def _print_model_info(
    io_config: dict,
    *,
    task: str | None = None,
    req_device: str = "auto",
    act_device: str = "auto",
    ep_name: EPName | None = None,
    actual_shapes: dict[str, tuple] | None = None,
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
            actual = actual_shapes.get(name) if actual_shapes else None
            shape_str = _format_input_shape(shape, actual)
            # ``shape_str`` can start with a lowercase ``dynamic(...)`` which Rich
            # would otherwise parse as a markup tag and swallow -- escape it.
            line = f"{name:<20s} {escape(shape_str):<22s} {dtype}"
            console.print(f"{label if i == 0 else pad}{line}")

    out_names = io_config.get("output_names", [])
    out_shapes = io_config.get("output_shapes", [])
    if out_names:
        label = "[dim]Outputs:[/dim]     "
        pad = "             "
        for i, name in enumerate(out_names):
            shape = out_shapes[i] if i < len(out_shapes) else []
            shape_str = escape(_format_input_shape(shape, None))
            console.print(f"{label if i == 0 else pad}{name:<20s} {shape_str}")

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


# perf() param names for WinML-only options that a prebuilt genai bundle
# ignores. Mapped to the user-facing flag for the warning message.
# NB: ``--ep`` is intentionally absent — it is honored for winml-genai as an EP
# override (explicit --ep > concrete --device > respect config).
_GENAI_IGNORED_FLAGS: dict[str, str] = {
    "task": "--task",
    "precision": "--precision",
    "ep_options": "--ep-options",
    "shape_config_path": "--shape-config",
    "input_specs": "--input-specs",
    "export_config": "--export-config",
    "dynamic_axes": "--dynamic-axes",
    "quant": "--quant/--no-quantize",
    "optimize": "--optimize/--no-optimize",
    "analyze": "--analyze/--no-analyze",
    "max_optim_iterations": "--max-optim-iterations",
    "rebuild": "--rebuild",
    "ignore_cache": "--ignore-cache",
    "skip_build": "--skip-build",
    "allow_unsupported_nodes": "--allow-unsupported-nodes",
    "monitor": "--monitor",
    "memory": "--memory",
    "op_tracing": "--op-tracing",
    "batch_size": "--batch-size",
}

# Subsets of the above that the model-id auto-build path honors, so they are
# excluded from the ignored-flags warning when a bundle is auto-built. A prebuilt
# bundle still ignores them all.
#
# * Cache-behavior flags force the build path (the reuse fast-path is never taken
#   when they are set), so they are honored whenever an auto-build runs.
# * Artifact-shaping flags only take effect when a build actually runs; a cache
#   hit reuses a bundle keyed by the model id alone and silently drops them, so
#   they are still reported as ignored in that case.
_GENAI_BUILD_CONTROL_FLAGS: frozenset[str] = frozenset({"rebuild", "ignore_cache"})
_GENAI_BUILD_INPUT_FLAGS: frozenset[str] = frozenset({"task", "precision"})


def _warn_ignored_genai_flags(
    ctx: click.Context, console: Console, *, autobuilt: bool = False, built_fresh: bool = False
) -> None:
    """Warn about WinML-only flags the user passed that genai ignores.

    When the bundle was auto-built from a model id (``autobuilt``), the
    cache-behavior flags (rebuild/ignore-cache) are always honored by the build.
    The artifact-shaping flags (task/precision) are honored only when a fresh
    build actually ran (``built_fresh``); on a cache hit the model-id-keyed bundle
    is reused as-is, so those flags are reported as ignored. A prebuilt bundle
    ignores them all.
    """
    honored: set[str] = set()
    if autobuilt:
        honored |= _GENAI_BUILD_CONTROL_FLAGS
        if built_fresh:
            honored |= _GENAI_BUILD_INPUT_FLAGS
    ignored = [
        flag
        for param, flag in _GENAI_IGNORED_FLAGS.items()
        if cli_utils.is_cli_provided(ctx, param) and param not in honored
    ]
    if ignored:
        console.print(
            "[yellow]Warning:[/yellow] the following options are ignored with "
            f"--runtime winml-genai: {', '.join(sorted(ignored))}"
        )


def _autobuild_genai_bundle(
    ctx: click.Context, *, model: str, console: Console, stack: contextlib.ExitStack
) -> tuple[Path, bool]:
    """Build (or reuse) a genai bundle for a HuggingFace model id.

    Mirrors the winml runtime's on-the-fly build: when ``-m`` is a model id
    rather than a prebuilt bundle directory, emit a genai bundle and benchmark
    it. Cache handling matches the single-model path:

    * a plain run reuses a previously built bundle keyed by the model id;
    * ``--rebuild`` overwrites that cached bundle in place;
    * ``--ignore-cache`` builds fresh in a throwaway temp dir -- both the
      assembled bundle and its component build cache -- and leaves the managed
      cache untouched. The temp dir is entered on *stack* so it outlives the
      benchmark and is removed afterwards.

    genai bundles target the NPU HTP via QNN, so the build pins ``ep=qnn`` /
    ``device=npu`` regardless of the benchmark's ``--device`` (which still
    selects the runtime EP).

    The imports are function-local so ``winml perf --help`` does not pay their
    cost and so a bundle-directory run never imports the build stack.

    Returns a ``(bundle_dir, built_fresh)`` pair. ``built_fresh`` is ``True`` when
    a build actually ran and ``False`` when the managed-cache fast-path reused an
    existing bundle (in which case task/precision were not applied to it).
    """
    import tempfile

    from ..cache import get_cache_dir, get_model_dir
    from ..loader import resolve_loader_config
    from ..models.winml import build_genai_bundle, resolve_genai_bundle

    p = ctx.params

    if p.get("ignore_cache"):
        # Mirror the winml runtime's use_cache=False path: build everything
        # fresh in a throwaway temp dir and neither read from nor write to the
        # managed cache. The assembled bundle and the component build cache both
        # live under the temp root; the ExitStack removes it after benchmarking.
        tmp_root = Path(
            stack.enter_context(
                tempfile.TemporaryDirectory(prefix="winml-genai-perf-", ignore_cleanup_errors=True)
            )
        )
        bundle_dir = tmp_root / "genai-bundle"
        build_cache_dir: Path = tmp_root / "cache"
        force_rebuild = True
    else:
        cache_dir = get_cache_dir()
        bundle_dir = get_model_dir(model, cache_dir=cache_dir) / "genai-bundle"
        build_cache_dir = cache_dir
        # --rebuild overwrites the cached bundle; a plain run reuses it. Checked
        # before any model resolution so a cache hit never touches the network.
        force_rebuild = bool(p.get("rebuild"))
        if (bundle_dir / "genai_config.json").exists() and not force_rebuild:
            console.print(f"[dim]Reusing cached genai bundle:[/dim] {bundle_dir}")
            return bundle_dir, False

    # Cache miss (or forced rebuild): resolve the model family so its
    # genai-bundle recipe can drive the build.
    try:
        loader_cfg, hf_config, *_rest = resolve_loader_config(model_id=model, task=p.get("task"))
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.UsageError(
            f"Could not resolve '{model}' for a genai bundle build ({exc}). Pass a "
            "prebuilt genai bundle *directory* produced by a winml-cli export instead."
        ) from exc

    model_type = getattr(hf_config, "model_type", None) or loader_cfg.model_type
    recipe = resolve_genai_bundle(model_type)
    if recipe is None:
        raise click.UsageError(
            f"--runtime winml-genai cannot auto-build '{model}': no genai bundle recipe "
            f"is registered for model type '{model_type or 'unknown'}'. Pass a prebuilt "
            "genai bundle *directory* (e.g. from "
            f"'winml build -m {model} -o <dir> --device npu --ep qnn')."
        )

    precision = p["precision"] if cli_utils.is_cli_provided(ctx, "precision") else None
    bundle_dir.mkdir(parents=True, exist_ok=True)
    console.print(
        f"[dim]Building genai bundle for[/dim] {model} "
        f"[dim](model_type={model_type}) ->[/dim] {bundle_dir}"
    )
    build_genai_bundle(
        model,
        bundle_dir,
        recipe,
        ep="qnn",
        device="npu",
        precision=precision,
        force_rebuild=force_rebuild,
        cache_dir=build_cache_dir,
        emit=lambda msg: console.print(msg, markup=False),
    )
    return bundle_dir, True


def _run_genai_runtime(ctx: click.Context, *, console: Console, json_mode: bool) -> None:
    """Validate folder input and dispatch to the winml-genai benchmark path.

    The genai imports are function-local so ``winml perf --help`` does not pay
    their import cost (see tests/cli/test_import_time.py).
    """
    import contextlib

    from ._perf_genai import (
        GenaiPerfConfig,
        genai_output_path,
        resolve_genai_ep,
        run_genai_perf,
    )

    p = ctx.params
    model: str = p["model"]

    # --module walks a live nn.Module graph; meaningless for a prebuilt bundle.
    if p.get("module_class"):
        raise click.UsageError("--module is not supported with --runtime winml-genai.")

    # --submodel narrows a composite into a single sub-component benchmarked as a
    # standalone session; a genai bundle is already the full composite generation
    # pipeline, so selecting one sub-component is meaningless. Reject rather than
    # silently ignore (this return runs before the winml-path --submodel handling).
    if p.get("submodel"):
        raise click.UsageError("--submodel is not supported with --runtime winml-genai.")

    # The ExitStack keeps an --ignore-cache auto-build's throwaway temp dir alive
    # across the benchmark below, then removes it on exit. A bundle dir or a
    # cached auto-build registers nothing, so it is a no-op.
    with contextlib.ExitStack() as stack:
        # Resolve the bundle. A local directory is used as-is (it must be a real
        # genai bundle); an ``.onnx`` file is rejected; anything else is treated
        # as a HuggingFace model id and a genai bundle is auto-built on demand --
        # the same way the winml runtime auto-builds an ONNX from a model id.
        bundle_dir = Path(model)
        autobuilt_from: str | None = None
        built_fresh = False
        if bundle_dir.is_dir():
            if not (bundle_dir / "genai_config.json").exists():
                raise click.UsageError(
                    f"No genai_config.json found in '{model}'. Point --model at a bundle "
                    "folder produced by a winml-cli export."
                )
        elif bundle_dir.suffix.lower() == ".onnx":
            raise click.UsageError(
                f"--runtime winml-genai requires a genai bundle *directory*, got '{model}'."
            )
        else:
            bundle_dir, built_fresh = _autobuild_genai_bundle(
                ctx, model=model, console=console, stack=stack
            )
            autobuilt_from = model

        _warn_ignored_genai_flags(
            ctx, console, autobuilt=autobuilt_from is not None, built_fresh=built_fresh
        )

        # A full generation is far costlier than one session.run(): default to
        # fewer iterations/warmup unless the user set them explicitly.
        iterations = p["iterations"] if cli_utils.is_cli_provided(ctx, "iterations") else 10
        warmup = p["warmup"] if cli_utils.is_cli_provided(ctx, "warmup") else 2

        # genai defaults to "config" (respect the bundle's own per-stage routing).
        # The shared --device default is "auto" (the ONNX default), so an omitted
        # flag is treated as "config"; an explicit --device is a deliberate override.
        device = p["device"].lower() if cli_utils.is_cli_provided(ctx, "device") else "config"
        if p.get("output"):
            output = p["output"]
        elif autobuilt_from is not None:
            # The auto-built bundle lives in a cache dir whose name is not a
            # meaningful slug, so derive the report path from the model id instead.
            output = generate_output_path(autobuilt_from)
        else:
            output = genai_output_path(bundle_dir)
        cli_utils.guard_output(output, p["overwrite"])

        # EP override precedence: an explicit ``--ep`` wins over the ``--device``
        # resolution, which in turn wins over the default ("config" = respect the
        # bundle's genai_config.json routing).  GenaiSession validates the value.
        ep: EPNameOrAlias | None = (
            p["ep"] if cli_utils.is_cli_provided(ctx, "ep") else resolve_genai_ep(device)
        )

        config = GenaiPerfConfig(
            bundle_dir=bundle_dir,
            ep=ep,
            device=device,
            prompt=p["prompt"],
            apply_template=p["apply_template"],
            max_new_tokens=p["max_new_tokens"],
            iterations=iterations,
            warmup=warmup,
            compile=not p["no_compile"],
            compile_timeout=p["compile_timeout"],
            output_path=output,
        )
        run_genai_perf(config, console=console, json_mode=json_mode)


def _resolve_composite_components_for_perf(model: str, task: str | None) -> dict[str, str] | None:
    """Detect a composite model's sub-components (name -> component task), else None.

    Mirrors the registry-driven detection in ``winml export`` / ``winml build``
    so ``--submodel`` resolves the same components (and the same seq2seq bridge
    when ``--task`` is omitted). Only the "not a resolvable HF config" case
    (``OSError``) is suppressed (fall through to "not composite"); intentional
    loud guards (empty registry, model-task incompatibility) and any unexpected
    failure are surfaced rather than masked.
    """
    from ..loader.resolution import resolve_composite_components

    try:
        return resolve_composite_components(model, task=task)
    except click.ClickException:
        raise
    except ValueError as e:
        raise click.UsageError(str(e)) from e
    except RuntimeError:
        raise
    except OSError as e:
        logger.debug("Composite detection unavailable (config not resolvable): %s", e)
        return None
    except Exception as e:
        raise click.ClickException(f"Composite model detection failed unexpectedly: {e}") from e


@click.command("perf")
@cli_utils.model_option(required=False)
@click.option(
    "--runtime",
    type=click.Choice(list(RUNTIME_NAMES)),
    default="winml",
    show_default=True,
    help="Inference runtime. 'winml' benchmarks single-shot ONNX inference; "
    "'winml-genai' benchmarks an onnxruntime-genai bundle folder "
    "(LLM generation: TTFT + decode tokens/sec).",
)
@click.option(
    "--prompt",
    type=str,
    default="Explain the theory of relativity in simple terms.",
    show_default=True,
    help="[winml-genai] Prompt text to generate from. By default it is wrapped in "
    "the bundle's chat template; pass --no-apply-template to benchmark it verbatim.",
)
@click.option(
    "--apply-template/--no-apply-template",
    default=True,
    show_default=True,
    help="[winml-genai] Wrap --prompt in the bundle's chat template before timing. "
    "Use --no-apply-template to benchmark a prompt that is already formatted.",
)
@click.option(
    "--max-new-tokens",
    type=click.IntRange(min=1),
    default=128,
    show_default=True,
    help="[winml-genai] Number of new tokens to generate per iteration.",
)
@click.option(
    "--compile-timeout",
    type=int,
    default=300,
    show_default=True,
    help="[winml-genai] Max seconds to compile each EPContext stage before falling back "
    "to the original ONNX (requires --compile).",
)
@click.option(
    "--task",
    type=str,
    default=None,
    help="Explicit task (e.g., 'image-classification'). Auto-detected if not specified.",
)
@click.option(
    "--submodel",
    type=str,
    default=None,
    help=(
        "Benchmark a specific sub-model of a composite model "
        "(e.g., 'text_model', 'vision_model'). Omit to benchmark all sub-models."
    ),
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
@cli_utils.device_option(
    required=False,
    default="auto",
    include_auto=True,
    include_config=True,
    optional_message="'config' (winml-genai only) respects the bundle's genai_config.json routing.",
)
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
@cli_utils.overwrite_option()
@click.option(
    "--batch-size",
    type=int,
    default=1,
    show_default=True,
    help="Batch size for input generation",
)
@click.option(
    "--input-data",
    "input_data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a .npz file of real input tensors to benchmark with instead "
    "of randomly generated inputs. Keys must match the model's input names and "
    "dtypes exactly. Not supported with --module or --runtime winml-genai.",
)
@cli_utils.shape_config_option(param_name="shape_config_path")
@cli_utils.input_specs_option()
@cli_utils.export_config_option(
    help_text="ONNX export configuration JSON for HuggingFace model builds.",
)
@cli_utils.dynamic_axes_option(
    help_text=(
        "JSON dynamic axes mapping for HuggingFace ONNX export "
        '(e.g., {"input_ids": {"0": "batch", "1": "sequence"}}).'
    )
)
@cli_utils.quant_option(optional_message="Applied during model build.")
@cli_utils.optimize_option(optional_message="Applied during model build.")
@cli_utils.analyze_option(optional_message="Applied during model build.")
@cli_utils.max_optim_iterations_option()
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
@cli_utils.no_color_option()
@click.pass_context
def perf(
    ctx: click.Context,
    model: str | None,
    runtime: RuntimeName,
    prompt: str,
    apply_template: bool,
    max_new_tokens: int,
    compile_timeout: int,
    task: str | None,
    submodel: str | None,
    iterations: int,
    warmup: int,
    device: str,
    precision: str,
    ep: EPNameOrAlias | None,
    ep_options: tuple[str, ...],
    output: Path | None,
    overwrite: bool,
    batch_size: int,
    input_data: Path | None,
    shape_config_path: Path | None,
    input_specs: Path | None,
    export_config: Path | None,
    dynamic_axes: Path | None,
    quant: bool,
    optimize: bool,
    analyze: bool,
    max_optim_iterations: int | None,
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

        # Benchmark with real inputs from a .npz file
        winml perf -m model.onnx --input-data inputs.npz

        # Operator-level profiling (QNN NPU)
        winml perf -m model.onnx --op-tracing basic
    """
    if not model:
        raise click.UsageError("A model is required via -m/--model.")

    # Hub-hosted ONNX (e.g. ``onnx-community/sam3-tracker-ONNX/onnx/...``)
    # is downloaded once and treated as a local .onnx path thereafter.
    # Must run BEFORE the ``Path(hf_model).suffix == ".onnx"`` check below
    # so a Hub ref is not mistaken for a missing local file.
    # ``normalize_model_arg`` returns ``str | None`` per its signature;
    # the ``or model`` keeps the narrowed ``str`` type for downstream use.
    try:
        hf_model: str = cli_utils.normalize_model_arg(model) or model
    except Exception as e:
        raise click.ClickException(f"Failed to resolve Hub-hosted ONNX path {model!r}: {e}") from e
    model = hf_model

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
    # GENAI RUNTIME: benchmark an onnxruntime-genai bundle folder
    # =========================================================================
    if runtime == "winml-genai":
        if input_data is not None:
            raise click.UsageError(
                "--input-data is not supported with --runtime winml-genai; "
                "genai benchmarking is driven by --prompt."
            )
        _run_genai_runtime(ctx, console=console, json_mode=json_mode)
        return

    # ``--device config`` is a winml-genai-only sentinel (respect the bundle's
    # genai_config.json routing).  It is meaningless for the single-shot WinML
    # path, so reject it explicitly rather than letting resolve_device raise a
    # generic "unknown device" error.
    if device.lower() == "config":
        raise click.UsageError(
            "--device config is only valid with --runtime winml-genai "
            "(it means 'respect the bundle's genai_config.json routing')."
        )

    # Classify the -m value once so module mode and the single-model path share
    # one source of truth. Rejects an invalid id up front; a path-shaped .onnx
    # that doesn't exist is caught below with a friendly "not found" message
    # (the pure classifier stays existence-agnostic).
    model_input = classify_model_input(hf_model)
    if model_input.kind is ModelInputKind.INVALID:
        raise click.UsageError(model_input.error or f"Invalid model input: {hf_model}")
    is_onnx = model_input.kind is ModelInputKind.ONNX_FILE
    if is_onnx and model_input.local_path and not Path(model_input.local_path).exists():
        raise click.UsageError(f"ONNX file not found: {hf_model}")

    # =========================================================================
    # --submodel: narrow a composite model to one sub-component, benchmarked as
    # a standalone single-session model. The composite is detected the same way
    # `winml export` / `winml build` / `winml inspect` do (registry-driven, via
    # the seq2seq bridge), so it works even when --task is omitted. The selected
    # component is then loaded through the normal single-model path using its own
    # component task — exactly how the composite builds that sub-model — which
    # sidesteps the config-ambiguous pipeline task (e.g. t5 translation vs
    # summarization) and avoids building the other sub-models just to discard.
    # =========================================================================
    if submodel is not None:
        if is_onnx:
            raise click.BadParameter(
                "--submodel is not supported for ONNX files; a .onnx file is "
                "already a single model.",
                param_hint="--submodel",
            )
        if module_class:
            raise click.BadParameter(
                "--submodel cannot be combined with --module.",
                param_hint="--submodel",
            )
        components = _resolve_composite_components_for_perf(hf_model, task)
        if components is None:
            raise click.BadParameter(
                f"'{submodel}' was specified, but '{hf_model}' is not a "
                f"composite model (no sub-models detected).",
                param_hint="--submodel",
            )
        if submodel not in components:
            raise click.BadParameter(
                f"Unknown sub-model '{submodel}'. Available: {', '.join(components)}",
                param_hint="--submodel",
            )
        # Load only this component, using its own task, via the single-model path.
        task = components[submodel]
        console.print(
            f"[dim]Composite sub-model:[/dim] {submodel} (task={task}) [dim]from[/dim] {hf_model}"
        )

    # =========================================================================
    # MODULE MODE: per-module build + benchmark
    # =========================================================================
    if module_class:
        # Reject --module on a pre-exported ONNX path. Submodule discovery
        # walks a live nn.Module graph (torchinfo), which an ONNX file does
        # not carry. Without this guard, the user sees a misleading
        # "not a valid JSON file" error from AutoConfig.from_pretrained
        # trying to load the .onnx as an HF config dir (issue #553).
        if is_onnx:
            raise click.UsageError(
                f"--module is not supported for ONNX files. "
                f"Submodule benchmarking requires a HuggingFace model ID "
                f"(e.g., 'microsoft/resnet-50'), not '{hf_model}'."
            )
        if input_data is not None:
            raise click.UsageError(
                "--input-data is not supported in --module mode. Each submodule "
                "has its own internal inputs, which a single .npz cannot address."
            )
        if shape_config_path:
            console.print(
                "[yellow]Warning:[/yellow] --shape-config is not supported "
                "in --module mode and will be ignored."
            )
        ignored_export_flags = [
            flag
            for flag, value in (
                ("--input-specs", input_specs),
                ("--export-config", export_config),
                ("--dynamic-axes", dynamic_axes),
            )
            if value is not None
        ]
        if ignored_export_flags:
            console.print(
                "[yellow]Warning:[/yellow] "
                f"{', '.join(ignored_export_flags)} are not supported in --module mode "
                "and will be ignored."
            )
        # _perf_modules resolves the device + derives a concrete EP internally
        # (it will fold into PerfBenchmark — see #939).
        _perf_modules(
            hf_model=hf_model,
            module_class=module_class,
            task=task,
            iterations=iterations,
            warmup=warmup,
            batch_size=batch_size,
            no_quantize=not quant,
            no_optimize=not optimize,
            no_analyze=not analyze,
            max_optim_iterations=max_optim_iterations,
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
            rebuild=rebuild,
            ignore_cache=ignore_cache,
        )
        return

    # =========================================================================
    # SINGLE MODEL MODE: existing benchmark flow
    # =========================================================================

    # Load shape overrides from JSON if provided.
    #
    # Real input tensors define their own shapes, so --shape-config doesn't
    # apply when --input-data is set. Skip parsing/printing the overrides in
    # that case, otherwise we'd announce "Shape overrides: {…}" and then
    # immediately warn they're ignored.
    shape_config = None
    if input_data is not None:
        if shape_config_path:
            console.print(
                "[yellow]Warning:[/yellow] --shape-config is ignored when "
                "--input-data is set; shapes come from the provided tensors."
            )
    elif shape_config_path:
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

    # --batch-size likewise doesn't apply to real input tensors; warn instead
    # of silently ignoring so the user isn't surprised by the reported batch.
    if input_data is not None and cli_utils.is_cli_provided(ctx, "batch_size"):
        console.print(
            "[yellow]Warning:[/yellow] --batch-size is ignored when "
            "--input-data is set; the batch comes from the provided tensors."
        )

    export_overrides = None
    export_flag_values = (input_specs, export_config, dynamic_axes)
    if any(value is not None for value in export_flag_values):
        if is_onnx:
            ignored = [
                flag
                for flag, value in (
                    ("--input-specs", input_specs),
                    ("--export-config", export_config),
                    ("--dynamic-axes", dynamic_axes),
                )
                if value is not None
            ]
            console.print(
                "[yellow]Warning:[/yellow] "
                f"{', '.join(ignored)} are ignored for pre-exported ONNX inputs."
            )
        else:
            export_overrides = cli_utils.load_export_overrides(
                export_config=export_config,
                input_specs=input_specs,
                dynamic_axes=dynamic_axes,
            )
            if input_specs:
                console.print(f"[dim]Input specs:[/dim] {input_specs}")
            if export_config:
                console.print(f"[dim]Export config:[/dim] {export_config}")
            if dynamic_axes:
                console.print(f"[dim]Dynamic axes:[/dim] {dynamic_axes}")

    # Resolve output path
    if output is None:
        output = generate_output_path(hf_model, submodel=submodel)

    # Refuse to clobber an existing report unless the user opted in.
    cli_utils.guard_output(output, overwrite)

    # Create config. The raw device/EP request is passed through unchanged;
    # PerfBenchmark resolves the concrete device + EP internally (failing fast
    # before the build), so the CLI does not pre-resolve here.
    config = BenchmarkConfig(
        model_id=hf_model,
        task=task,
        submodel=submodel,
        device=device.lower(),
        precision=precision.lower(),
        iterations=iterations,
        warmup=warmup,
        batch_size=batch_size,
        output_path=output,
        no_quantize=not quant,
        no_optimize=not optimize,
        no_analyze=not analyze,
        max_optim_iterations=max_optim_iterations,
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
        export_overrides=export_overrides,
        input_data=input_data,
    )

    try:
        model_path = Path(hf_model)

        if is_onnx:
            # Existence already validated by classify_model_input above.
            # Build-pipeline flags are forwarded to from_onnx but no-op when the
            # build is skipped (the default). Warn so the silent no-op is visible
            # — shared detection with eval via utils/cli.py.
            build_flags_warning = cli_utils.ignored_build_flags_warning(
                skip_build_onnx=skip_build,
                quant=quant,
                optimize=optimize,
                analyze=analyze,
                max_optim_iterations=max_optim_iterations,
            )
            if build_flags_warning:
                console.print(f"[yellow]Warning:[/yellow] {build_flags_warning}")
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
            from ..optracing import is_profiling_available

            if not is_profiling_available(
                benchmark.resolved_ep, benchmark.resolved_device, op_tracing
            ):
                raise click.ClickException(
                    "Op-tracing is only supported for the QNN EP "
                    "on NPU at the 'basic' level "
                    f"(resolved EP={benchmark.resolved_ep}, "
                    f"device={benchmark.resolved_device}, level={op_tracing})."
                )

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
                raise click.ClickException(
                    "Could not determine ONNX model path for op-tracing"
                ) from None

            output_dir = output.parent if output else Path()

            # Look up tracer via registry (EP-agnostic).
            tracer_cls = get_tracer("QNNExecutionProvider", op_tracing)
            if tracer_cls is None:
                raise click.ClickException(
                    f"No tracer registered for QNN EP at level '{op_tracing}'"
                )

            # When --input-data was supplied, trace on the same real tensors the
            # benchmark used (benchmark._inputs), so the op trace isn't measured
            # on random data. The profiler falls back to random inputs if these
            # don't match the traced session's inputs.
            trace_inputs = getattr(benchmark, "_inputs", None) if input_data else None

            profiler = tracer_cls(
                onnx_for_trace,
                output_dir=output_dir,
                level=op_tracing,
                input_data=trace_inputs,
            )
            trace_result = profiler.run(
                iterations=min(iterations, 10),
                warmup=min(warmup, 3),
            )

            # Display and save
            display_op_trace_report(trace_result, console)

            # Mirror the benchmark report path so the two files sit side by side:
            # a/b.json -> a/b_op_trace.json.
            trace_output = output.with_name(f"{output.stem}_op_trace{output.suffix}")
            write_op_trace_json(trace_result, trace_output)
            console.print(f"[green]Op-trace saved to:[/green] {trace_output}")

    except FileNotFoundError as e:
        # User-error: bad model path. UsageError so the exit code (2) matches
        # the convention used by Click for argument problems.
        raise click.UsageError(f"Model not found: {e}") from e

    except click.ClickException:
        # Click exceptions are already intentional control flow; re-raise so
        # the catch-all below doesn't relabel them as "Benchmark failed".
        raise
    except Exception as e:
        if verbose:
            logger.exception("Benchmark failed")
        raise click.ClickException(f"Benchmark failed: {e}") from e
