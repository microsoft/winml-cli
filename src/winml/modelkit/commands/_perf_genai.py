# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""GenAI generation benchmarking for ``winml perf --runtime winml-genai``.

Benchmarks a prebuilt ``onnxruntime-genai`` bundle folder through
:class:`GenaiSession`.  Unlike the single-shot WinML path (which times each
``session.run()`` call), decoder pipelines split into a **prefill** phase
(prompt -> first token) and a **decode** phase (subsequent tokens), so this
module reports LLM-style metrics: startup/cold-start spans, time-to-first-token (TTFT),
prefill throughput, decode throughput (tokens/sec), time-per-output-token
(TPOT), warm-start latency, and total generation time.

Timing is captured inside :meth:`GenaiSession.generate_timed` at the
onnxruntime-genai call boundaries (``append_tokens`` = prefill, each
``generate_next_token`` = one decode step), mirroring onnxruntime-genai's
official ``benchmark_e2e.py``.  onnxruntime-genai exposes no native
perf-metrics API, so these are external wall-clock spans taken around the
library calls.

The ``perf`` command validates the folder input and delegates here via
:func:`run_genai_perf`; ``perf.py`` itself stays single-shot-focused.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import click

from ..session import (
    GenaiLoadError,
    GenaiNotInstalledError,
    GenaiSession,
    GenaiSessionError,
    GenerationConfig,
    short_ep_name,
)
from ..utils.constants import (
    ACCELERATOR_DEVICE_TYPES,
    EP_SUPPORTED_DEVICES,
    EPNameOrAlias,
    normalize_ep_name,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.console import Console

    from ..utils.constants import EPName, RuntimeName

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

RUNTIME_TYPE: RuntimeName = "winml-genai"
_HW_POLL_INTERVAL_MS = 200

# Built-in benchmark prompt.  Mirrored by the ``--prompt`` CLI default and the
# ``GenaiPerfConfig.prompt`` field default (a test asserts the two stay in sync).
_DEFAULT_PROMPT = "Explain the theory of relativity in simple terms."

# Sentinel ``--device`` value meaning "respect the bundle's genai_config.json
# routing" (no EP override).  It is the winml-genai default: a genai bundle is
# mixed by design (e.g. ctx/iter on the NPU, embeddings/lm_head on CPU) and its
# config already encodes that per-stage routing, so the common case leaves it
# untouched.  A concrete ``--device`` (or ``--ep``) is an explicit override that
# forces the *whole* decoder pipeline onto one EP.
GENAI_CONFIG_DEVICE = "config"


def resolve_genai_ep(device: str) -> EPNameOrAlias | None:
    """Resolve a ``--device`` value to a :class:`GenaiSession` EP override.

    ``config`` -> ``None`` (respect ``genai_config.json`` as-is).  Any concrete
    device (``auto``/``npu``/``gpu``/``cpu``) goes through the same
    :func:`resolve_device` / :func:`resolve_eps` path the WinML ONNX runtime
    uses, so it forces the whole pipeline onto the *best EP actually available
    for that device on this machine* (e.g. an NPU that is VitisAI/OpenVINO
    rather than QNN) instead of a static short-name guess.  Returns the EP's
    canonical short alias, or ``None`` when the device resolves to no EP.

    Raises:
        ValueError: propagated from :func:`resolve_device` when the requested
            device has no compatible EP available -- fail fast, like the ONNX
            path, rather than silently falling back to CPU.
    """
    if device == GENAI_CONFIG_DEVICE:
        return None

    # Function-local import mirrors the ONNX path (perf.py) and avoids a
    # module-level cycle.
    from ..session import EPDeviceTarget, available_eps_for_device, resolve_device

    resolved_device = resolve_device(EPDeviceTarget(ep="auto", device=device)).device
    eps = available_eps_for_device(resolved_device)
    if not eps:
        return None

    # Prefer EPs whose *primary* device (first entry in EP_SUPPORTED_DEVICES)
    # matches the resolved device.  Multi-device EPs like OpenVINO advertise
    # cpu/gpu support but their primary target is npu — when the user says
    # ``--device cpu`` or ``--device gpu`` they expect the native EP for that
    # device, not a cross-device accelerator that also happens to support it.
    native = [ep for ep in eps if EP_SUPPORTED_DEVICES[cast("EPName", ep)][0] == resolved_device]
    best = native[0] if native else eps[0]
    # short_ep_name returns a plain ``str``; the value is a canonical EP short
    # alias (a member of EPAlias) that GenaiSession accepts as an override.
    return cast("EPNameOrAlias", short_ep_name(best))


def genai_output_path(bundle_dir: str | Path) -> Path:
    """Default JSON report path for a genai bundle.

    Delegates to :func:`perf.generate_output_path` so both perf runtimes share
    one report-path convention (``~/.cache/winml/perf/<slug>/<ts>.json``).  The
    import is function-local to avoid a module-level cycle with ``perf`` (which
    imports this module lazily inside its command body).
    """
    from .perf import generate_output_path

    return generate_output_path(Path(bundle_dir).name or "genai")


# =============================================================================
# Statistics helpers
# =============================================================================


def _mean(xs: list[float]) -> float:
    """Arithmetic mean, or ``0.0`` for an empty sequence."""
    return sum(xs) / len(xs) if xs else 0.0


def _percentile(sorted_xs: list[float], p: float) -> float:
    """Nearest-rank ``p``-th percentile (0-100) of an already-sorted list.

    Matches :meth:`winml.modelkit.session.stats.PerfStats.percentile` so the
    two perf paths report percentiles the same way.
    """
    if not sorted_xs:
        return 0.0
    idx = int(len(sorted_xs) * p / 100)
    idx = min(idx, len(sorted_xs) - 1)
    return sorted_xs[idx]


def _stats(values: list[float]) -> dict[str, float]:
    """Return the percentile summary shape used by perf JSON blocks."""
    sorted_values = sorted(values)
    mean = _mean(values)
    variance = _mean([(value - mean) ** 2 for value in values]) if values else 0.0
    return {
        "mean": mean,
        "std": variance**0.5,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "p50": _percentile(sorted_values, 50),
        "p90": _percentile(sorted_values, 90),
        "p95": _percentile(sorted_values, 95),
        "p99": _percentile(sorted_values, 99),
    }


def _round_stats(values: dict[str, float]) -> dict[str, float]:
    """Round a stats block for JSON output."""
    return {key: round(value, 3) for key, value in values.items()}


def _get_rss_mb() -> float:
    """Return current process RSS in MB."""
    from ..session.monitor.memory_tracker import get_rss_mb

    return get_rss_mb()


def _get_vram_mb(adapter_luid: str | None) -> tuple[float, float]:
    """Return current process device-memory usage as local/shared MB."""
    from ..session.monitor.memory_tracker import get_vram_mb

    return get_vram_mb(adapter_luid)


def _resolve_adapter_luid(device: str, ep: EPNameOrAlias | None) -> str | None:
    """Resolve the adapter LUID used for best-effort process VRAM sampling."""
    if device not in ACCELERATOR_DEVICE_TYPES:
        return None

    ep_name = normalize_ep_name(ep) if ep is not None else None
    try:
        from ..sysinfo.pdh_adapters import resolve_adapter_luid

        return resolve_adapter_luid(device, ep_name=ep_name)
    except Exception:
        logger.debug("Could not resolve adapter LUID for genai memory tracking", exc_info=True)
    return None


@dataclass
class _MemorySnapshot:
    """Point-in-time process RAM and device-memory usage."""

    rss_mb: float = 0.0
    vram_local_mb: float = 0.0
    vram_shared_mb: float = 0.0


class _GenaiMemoryTracker:
    """Best-effort process memory tracking for the genai benchmark phases."""

    def __init__(self, *, adapter_luid: str | None) -> None:
        self._adapter_luid = adapter_luid
        self._baseline = _MemorySnapshot()
        self._after_load = _MemorySnapshot()
        self._after_benchmark = _MemorySnapshot()

    def _snapshot(self) -> _MemorySnapshot:
        gc.collect()
        rss = _get_rss_mb()
        local, shared = _get_vram_mb(self._adapter_luid) if self._adapter_luid else (0.0, 0.0)
        return _MemorySnapshot(rss_mb=rss, vram_local_mb=local, vram_shared_mb=shared)

    def record_baseline(self) -> None:
        self._baseline = self._snapshot()

    def record_after_load(self) -> None:
        self._after_load = self._snapshot()

    def record_after_benchmark(self) -> None:
        self._after_benchmark = self._snapshot()

    @staticmethod
    def _delta(after: float, before: float) -> float:
        return round(after - before, 2)

    def to_dict(self) -> dict[str, float]:
        baseline = self._baseline
        after_load = self._after_load
        after_benchmark = self._after_benchmark
        result = {
            "rss_baseline_mb": round(baseline.rss_mb, 2),
            "rss_after_compile_mb": round(after_load.rss_mb, 2),
            "rss_after_inference_mb": round(after_benchmark.rss_mb, 2),
            "rss_checkpoint_peak_mb": round(
                max(baseline.rss_mb, after_load.rss_mb, after_benchmark.rss_mb), 2
            ),
            "rss_model_load_delta_mb": self._delta(after_load.rss_mb, baseline.rss_mb),
            "rss_inference_delta_mb": self._delta(after_benchmark.rss_mb, after_load.rss_mb),
            "rss_total_delta_mb": self._delta(after_benchmark.rss_mb, baseline.rss_mb),
        }
        if self._adapter_luid is None:
            return result
        result.update(
            {
                "vram_local_baseline_mb": round(baseline.vram_local_mb, 2),
                "vram_shared_baseline_mb": round(baseline.vram_shared_mb, 2),
                "vram_local_after_compile_mb": round(after_load.vram_local_mb, 2),
                "vram_shared_after_compile_mb": round(after_load.vram_shared_mb, 2),
                "vram_local_after_inference_mb": round(after_benchmark.vram_local_mb, 2),
                "vram_shared_after_inference_mb": round(after_benchmark.vram_shared_mb, 2),
                "vram_local_checkpoint_peak_mb": round(
                    max(
                        baseline.vram_local_mb,
                        after_load.vram_local_mb,
                        after_benchmark.vram_local_mb,
                    ),
                    2,
                ),
                "vram_shared_checkpoint_peak_mb": round(
                    max(
                        baseline.vram_shared_mb,
                        after_load.vram_shared_mb,
                        after_benchmark.vram_shared_mb,
                    ),
                    2,
                ),
                "vram_local_model_load_delta_mb": self._delta(
                    after_load.vram_local_mb, baseline.vram_local_mb
                ),
                "vram_shared_model_load_delta_mb": self._delta(
                    after_load.vram_shared_mb, baseline.vram_shared_mb
                ),
                "vram_local_inference_delta_mb": self._delta(
                    after_benchmark.vram_local_mb, after_load.vram_local_mb
                ),
                "vram_shared_inference_delta_mb": self._delta(
                    after_benchmark.vram_shared_mb, after_load.vram_shared_mb
                ),
                "vram_local_total_delta_mb": self._delta(
                    after_benchmark.vram_local_mb, baseline.vram_local_mb
                ),
                "vram_shared_total_delta_mb": self._delta(
                    after_benchmark.vram_shared_mb, baseline.vram_shared_mb
                ),
            }
        )
        return result


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class GenaiPerfConfig:
    """Resolved request for a genai generation benchmark."""

    bundle_dir: Path
    model_id: str | None = None
    ep: EPNameOrAlias | None = None
    device: str = "auto"
    prompt: str = _DEFAULT_PROMPT
    apply_template: bool = True
    max_new_tokens: int = 128
    iterations: int = 10
    warmup: int = 2
    compile: bool = False
    compile_timeout: int = 300
    monitor: bool = False
    context_length: int | None = None
    memory: bool = False
    output_path: Path | None = None


@dataclass
class _RequestSample:
    """Canonical timing captured for one generation request."""

    kind: str
    index: int
    prompt_tokens: int
    generated_tokens: int
    template_duration_ms: float
    tokenization_duration_ms: float
    generator_create_duration_ms: float
    prefill_duration_ms: float
    first_token_duration_ms: float
    decode_token_durations_ms: list[float]
    sequence_fetch_duration_ms: float
    detokenization_duration_ms: float

    @property
    def model_ttft_duration_ms(self) -> float:
        """Model-only TTFT: prefill + first generated token."""
        return self.prefill_duration_ms + self.first_token_duration_ms

    @property
    def request_ttft_duration_ms(self) -> float:
        """Request-level TTFT from prompt preparation through first token."""
        return (
            self.template_duration_ms
            + self.tokenization_duration_ms
            + self.generator_create_duration_ms
            + self.model_ttft_duration_ms
        )

    @property
    def response_eval_duration_ms(self) -> float:
        """Time spent generating response tokens, including the first token."""
        return self.first_token_duration_ms + sum(self.decode_token_durations_ms)

    @property
    def model_compute_duration_ms(self) -> float:
        """Model compute: prefill + first token + steady-state decode."""
        return self.prefill_duration_ms + self.response_eval_duration_ms

    @property
    def request_duration_ms(self) -> float:
        """Full warm request duration from prompt prep to response text ready."""
        return (
            self.template_duration_ms
            + self.tokenization_duration_ms
            + self.generator_create_duration_ms
            + self.model_compute_duration_ms
            + self.sequence_fetch_duration_ms
            + self.detokenization_duration_ms
        )

    @property
    def prefill_tokens_per_second(self) -> float:
        """Prompt-processing throughput."""
        seconds = self.prefill_duration_ms / 1000.0
        return self.prompt_tokens / seconds if seconds > 0 else 0.0

    @property
    def steady_state_decode_tokens_per_second(self) -> float:
        """Decode throughput excluding the first generated token."""
        total_ms = sum(self.decode_token_durations_ms)
        return len(self.decode_token_durations_ms) / (total_ms / 1000.0) if total_ms > 0 else 0.0

    @property
    def response_eval_tokens_per_second(self) -> float:
        """Response-token throughput including the first generated token."""
        seconds = self.response_eval_duration_ms / 1000.0
        return self.generated_tokens / seconds if seconds > 0 else 0.0

    @property
    def steady_state_tpot_ms(self) -> float:
        """Mean per-token latency for tokens after the first."""
        return _mean(self.decode_token_durations_ms)

    def to_dict(self) -> dict[str, Any]:
        """Convert to the canonical JSON request sample shape."""
        return {
            "kind": self.kind,
            "index": self.index,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "template_duration_ms": round(self.template_duration_ms, 3),
            "tokenization_duration_ms": round(self.tokenization_duration_ms, 3),
            "generator_create_duration_ms": round(self.generator_create_duration_ms, 3),
            "prefill_duration_ms": round(self.prefill_duration_ms, 3),
            "first_token_duration_ms": round(self.first_token_duration_ms, 3),
            "decode_token_durations_ms": [
                round(value, 3) for value in self.decode_token_durations_ms
            ],
            "sequence_fetch_duration_ms": round(self.sequence_fetch_duration_ms, 3),
            "detokenization_duration_ms": round(self.detokenization_duration_ms, 3),
            "request_ttft_duration_ms": round(self.request_ttft_duration_ms, 3),
            "model_ttft_duration_ms": round(self.model_ttft_duration_ms, 3),
            "response_eval_duration_ms": round(self.response_eval_duration_ms, 3),
            "model_compute_duration_ms": round(self.model_compute_duration_ms, 3),
            "request_duration_ms": round(self.request_duration_ms, 3),
            "prefill_tokens_per_second": round(self.prefill_tokens_per_second, 2),
            "steady_state_decode_tokens_per_second": round(
                self.steady_state_decode_tokens_per_second, 2
            ),
            "response_eval_tokens_per_second": round(self.response_eval_tokens_per_second, 2),
            "steady_state_tpot_ms": round(self.steady_state_tpot_ms, 3),
        }


@dataclass
class GenaiBenchmarkResult:
    """Canonical results from a genai generation benchmark."""

    config: GenaiPerfConfig
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    prompt_tokens: int = 0
    generated_tokens: int = 0
    context_length: int | None = None
    effective_ep: str | None = None
    effective_device: str | None = None
    load: dict[str, float | str | None] = field(default_factory=dict)
    requests: list[_RequestSample] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    memory_profile: dict[str, float] | None = None
    hw_monitor: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        result: dict[str, Any] = {
            "schema_version": 2,
            "benchmark_info": {
                "runtime": RUNTIME_TYPE,
                "model_id": self.config.model_id or str(self.config.bundle_dir),
                "running_model_path": str(self.config.bundle_dir),
                "bundle_dir": str(self.config.bundle_dir),
                "ep": self.effective_ep or "config",
                "device": self.config.device,
                "effective_device": self.effective_device,
                "compile": self.config.compile,
                "compile_timeout": self.config.compile_timeout,
                "monitor": self.config.monitor,
                "iterations": self.config.iterations,
                "warmup": self.config.warmup,
                "max_new_tokens": self.config.max_new_tokens,
                "apply_template": self.config.apply_template,
                "prompt": self.config.prompt,
                "prompt_tokens": self.prompt_tokens,
                "generated_tokens": self.generated_tokens,
                "context_length": self.context_length,
                "timestamp": self.timestamp,
            },
            "load": {
                key: round(value, 3) if isinstance(value, float) else value
                for key, value in self.load.items()
            },
            "requests": [sample.to_dict() for sample in self.requests],
            "aggregate": self._round_aggregate(),
        }
        if self.memory_profile:
            result["memory"] = self.memory_profile
        if self.hw_monitor:
            result["hw_monitor"] = self.hw_monitor
        return result

    def _round_aggregate(self) -> dict[str, Any]:
        """Round aggregate statistics while preserving counters and flags."""
        rounded: dict[str, Any] = {}
        for key, value in self.aggregate.items():
            if isinstance(value, dict):
                rounded[key] = _round_stats(value)
            elif isinstance(value, float):
                rounded[key] = round(value, 3)
            else:
                rounded[key] = value
        return rounded


# =============================================================================
# Benchmark engine
# =============================================================================


class GenaiPerfBenchmark:
    """Runs warmup + timed generations and aggregates LLM metrics.

    Args:
        config: The resolved benchmark request.
        session: Pre-built session (dependency injection for tests).  When
            ``None`` a :class:`GenaiSession` is constructed from ``config``.

    Note:
        The session is loaded explicitly so model-load and prompt-preparation
        spans can be reported separately from warm-start generation. Each timed
        generation is driven by :meth:`GenaiSession.generate_timed`, which
        captures wall-clock spans at the onnxruntime-genai call boundaries
        (``append_tokens`` = prefill, each ``generate_next_token`` = one decode
        step), so TTFT and TPOT reflect model compute rather than
        generator-construction or detokenization overhead.
    """

    def __init__(
        self,
        config: GenaiPerfConfig,
        *,
        session: GenaiSession | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._config = config
        self._session = session
        self._clock = clock
        self._generation_count = 0

    def _build_session(self) -> GenaiSession:
        return GenaiSession(
            self._config.bundle_dir,
            self._config.ep,
            device=self._session_device(),
            context_length=self._config.context_length,
            compile=self._config.compile,
            compile_timeout=self._config.compile_timeout,
        )

    def _session_device(self) -> str | None:
        """Concrete device (npu/gpu/cpu) the forced EP should target, or None.

        Only meaningful when an EP override is active: it lets the session
        synthesize ``device_type`` for device-parameterized EPs (OpenVINO /
        VitisAI) when a re-routed stage has no reusable options.  A concrete
        ``--device`` is used verbatim; the sentinels ``config``/``auto`` (e.g.
        ``--ep`` given alone) fall back to the EP's primary supported device.
        """
        ep = self._config.ep
        if ep is None:
            return None
        device = (self._config.device or "").lower()
        if device and device not in ("config", "auto"):
            return device
        canonical = normalize_ep_name(ep)
        devices = EP_SUPPORTED_DEVICES.get(canonical)
        return devices[0] if devices else None

    def _prompt_text(self, session: GenaiSession) -> str:
        """Return the prompt to benchmark, chat-templated when enabled.

        With ``apply_template`` set (the default) the configured prompt is
        wrapped in the bundle's own chat template (via
        :meth:`GenaiSession.apply_chat_template`) so the measured prefill
        matches how the model is actually prompted; bundles that ship no chat
        template benchmark the raw prompt unchanged.

        With ``apply_template`` disabled the prompt is benchmarked verbatim, so
        a caller can supply a prompt they have already wrapped in a template
        (or a raw completion prompt) and time exactly those tokens.
        """
        if not self._config.apply_template:
            logger.info("genai perf: apply_template disabled; benchmarking the prompt verbatim")
            return self._config.prompt
        try:
            templated = session.apply_chat_template(self._config.prompt)
        except GenaiSessionError as exc:
            logger.info("genai perf: no chat template applied (%s); benchmarking raw prompt", exc)
            return self._config.prompt
        logger.info("genai perf: applied the bundle's chat template to the prompt")
        return templated

    def run(self) -> GenaiBenchmarkResult:
        """Execute the benchmark and return aggregated metrics."""
        return self._run_unmonitored()

    def _build_hw_monitor(self) -> Any | None:
        """Return an HWMonitor for the proven effective genai route, if available."""
        try:
            from ..session.monitor.hw_monitor import HWMonitor
        except Exception:
            logger.debug("HWMonitor import failed for genai benchmark", exc_info=True)
            return None

        if not HWMonitor.is_available():
            logger.warning("HWMonitor unavailable; running genai benchmark without monitoring")
            return None

        monitor_device, ep_name = self._effective_monitor_route()
        return HWMonitor(
            poll_interval_ms=_HW_POLL_INTERVAL_MS,
            device=monitor_device,
            ep_name=ep_name,
        )

    def _run_unmonitored(self) -> GenaiBenchmarkResult:
        """Execute the benchmark with optional monitoring and memory tracking."""
        if self._session is None:
            self._session = self._build_session()
        session = self._session
        session.resolve_effective_route()

        hw_monitor = self._build_hw_monitor() if self._config.monitor else None
        if hw_monitor is not None:
            with hw_monitor as hw:
                result = self._run_benchmark_body(session)
            result.hw_monitor = hw.to_dict()
            return result
        return self._run_benchmark_body(session)

    def _run_benchmark_body(self, session: GenaiSession) -> GenaiBenchmarkResult:
        """Run load + generations after the effective route has been resolved."""
        memory_tracker: _GenaiMemoryTracker | None = None
        if self._config.memory:
            memory_tracker = _GenaiMemoryTracker(adapter_luid=self._effective_adapter_luid())
            memory_tracker.record_baseline()

        session_load_start = self._clock()
        session.load()
        session_load_end = self._clock()
        session_load_ms = (session_load_end - session_load_start) * 1000.0
        if memory_tracker is not None:
            memory_tracker.record_after_load()

        gen_config = GenerationConfig(
            max_new_tokens=self._config.max_new_tokens,
            do_sample=False,
        )

        total_runs = self._config.warmup + self._config.iterations
        logger.info(
            "genai perf: %d warmup + %d timed generations (max_new_tokens=%d)",
            self._config.warmup,
            self._config.iterations,
            self._config.max_new_tokens,
        )
        samples = [
            self._time_one_generation(
                session,
                gen_config,
                kind="warmup" if i < self._config.warmup else "timed",
                index=i,
            )
            for i in range(total_runs)
        ]

        result = self._aggregate(samples, session_load_ms=session_load_ms)

        if memory_tracker is not None:
            memory_tracker.record_after_benchmark()
            result.memory_profile = memory_tracker.to_dict()

        return result

    def _effective_monitor_route(self) -> tuple[str, EPName | None]:
        """Return a monitor route that is proven by the effective bundle config."""
        session = self._session
        effective_device = getattr(session, "effective_device", None)
        effective_ep = getattr(session, "effective_hardware_ep", None)
        if effective_device in ACCELERATOR_DEVICE_TYPES and effective_ep is not None:
            return effective_device, effective_ep
        return "cpu", None

    def _effective_adapter_luid(self) -> str | None:
        """Return the proven adapter LUID for VRAM tracking, or None to omit VRAM."""
        session = self._session
        effective_device = getattr(session, "effective_device", None)
        effective_ep = getattr(session, "effective_hardware_ep", None)
        if effective_device not in ACCELERATOR_DEVICE_TYPES or effective_ep is None:
            return None
        return _resolve_adapter_luid(effective_device, effective_ep)

    def _time_one_generation(
        self,
        session: GenaiSession,
        gen_config: GenerationConfig,
        *,
        kind: str,
        index: int,
    ) -> _RequestSample:
        """Run one generation and convert spans to a canonical request sample.

        Prompt templating and tokenization are timed per request so the core
        report can expose both request-level and model-compute metrics.
        """
        template_start = self._clock()
        prompt_text = self._prompt_text(session)
        template_end = self._clock()
        prompt_token_ids = session.encode(prompt_text)
        tokenization_end = self._clock()

        timing = session.generate_timed(prompt_token_ids, gen_config, clock=self._clock)
        self._generation_count += 1
        if self._generation_count == 1:
            logger.info("Model response (iteration 1): %s", timing.response_text)
        else:
            logger.debug(
                "Model response (iteration %d): %s",
                self._generation_count,
                timing.response_text,
            )
        return _RequestSample(
            kind=kind,
            index=index,
            prompt_tokens=timing.input_tokens,
            generated_tokens=timing.generated_tokens,
            template_duration_ms=(template_end - template_start) * 1000.0,
            tokenization_duration_ms=(tokenization_end - template_end) * 1000.0,
            generator_create_duration_ms=timing.generator_create_s * 1000.0,
            prefill_duration_ms=timing.prefill_s * 1000.0,
            first_token_duration_ms=timing.first_token_s * 1000.0,
            decode_token_durations_ms=[value * 1000.0 for value in timing.decode_s],
            sequence_fetch_duration_ms=timing.sequence_fetch_s * 1000.0,
            detokenization_duration_ms=timing.detokenization_s * 1000.0,
        )

    def _aggregate(
        self, samples: list[_RequestSample], *, session_load_ms: float
    ) -> GenaiBenchmarkResult:
        """Aggregate canonical samples (warmup requests excluded from stats)."""
        timed = [sample for sample in samples if sample.kind == "timed"] or samples
        first = samples[0] if samples else None
        load = self._load_metrics(session_load_ms)
        aggregate: dict[str, Any] = {
            "warmup_excluded": True,
            "warmup_request_count": len([sample for sample in samples if sample.kind == "warmup"]),
            "timed_request_count": len(timed),
            "cold_start_ttft_duration_ms": (
                session_load_ms + first.request_ttft_duration_ms if first else 0.0
            ),
            "cold_start_total_duration_ms": (
                session_load_ms + first.request_duration_ms if first else 0.0
            ),
            "request_duration_ms": _stats([s.request_duration_ms for s in timed]),
            "model_compute_duration_ms": _stats([s.model_compute_duration_ms for s in timed]),
            "model_ttft_duration_ms": _stats([s.model_ttft_duration_ms for s in timed]),
            "request_ttft_duration_ms": _stats([s.request_ttft_duration_ms for s in timed]),
            "prefill_duration_ms": _stats([s.prefill_duration_ms for s in timed]),
            "response_eval_duration_ms": _stats([s.response_eval_duration_ms for s in timed]),
            "steady_state_tpot_ms": _stats([s.steady_state_tpot_ms for s in timed]),
            "prefill_tokens_per_second": _stats([s.prefill_tokens_per_second for s in timed]),
            "steady_state_decode_tokens_per_second": _stats(
                [s.steady_state_decode_tokens_per_second for s in timed]
            ),
            "response_eval_tokens_per_second": _stats(
                [s.response_eval_tokens_per_second for s in timed]
            ),
        }

        return GenaiBenchmarkResult(
            config=self._config,
            effective_ep=getattr(self._session, "effective_ep", None),
            effective_device=getattr(self._session, "effective_device", None),
            prompt_tokens=timed[0].prompt_tokens if timed else 0,
            generated_tokens=timed[0].generated_tokens if timed else 0,
            context_length=self._session.context_length if self._session else None,
            load=load,
            requests=samples,
            aggregate=aggregate,
        )

    def _load_metrics(self, session_load_ms: float) -> dict[str, float | str | None]:
        """Return canonical load metrics with a clear weight-upload estimate."""
        load_timings = getattr(self._session, "load_timings_ms", {}) if self._session else {}
        metrics: dict[str, float | str | None] = {
            "session_load_duration_ms": session_load_ms,
            "ep_registration_duration_ms": 0.0,
            "bundle_prepare_duration_ms": 0.0,
            "native_load_duration_ms": 0.0,
            "config_create_duration_ms": 0.0,
            "model_create_duration_ms": 0.0,
            "tokenizer_create_duration_ms": 0.0,
            "weight_upload_duration_ms": None,
            "weight_upload_estimate_duration_ms": None,
            "weight_upload_estimate_source": "unavailable",
        }
        for key in (
            "ep_registration_duration_ms",
            "bundle_prepare_duration_ms",
            "native_load_duration_ms",
            "config_create_duration_ms",
            "model_create_duration_ms",
            "tokenizer_create_duration_ms",
        ):
            if key in load_timings:
                metrics[key] = float(load_timings[key])
        if "session_load_duration_ms" in load_timings:
            metrics["session_load_duration_ms"] = session_load_ms
        model_create_ms = float(metrics["model_create_duration_ms"] or 0.0)
        if model_create_ms > 0:
            metrics["weight_upload_estimate_duration_ms"] = model_create_ms
            metrics["weight_upload_estimate_source"] = "model_create_duration"
        return metrics


# =============================================================================
# Reporting
# =============================================================================


def display_genai_report(result: GenaiBenchmarkResult, console: Console) -> None:
    """Render a genai benchmark report to the console."""
    from rich.table import Table

    cfg = result.config
    console.print()
    console.print(f"[dim]Runtime:[/dim]   {RUNTIME_TYPE}")
    ep_label = result.effective_ep or "config"
    device_str = cfg.device if cfg.device == ep_label else f"{cfg.device} ({ep_label})"
    console.print(f"[dim]Device:[/dim]    {device_str}")
    console.print(f"[dim]Bundle:[/dim]    {cfg.bundle_dir}")
    console.print(
        f"[dim]Prompt:[/dim]    {result.prompt_tokens} tokens   "
        f"[dim]Generated:[/dim] {result.generated_tokens} tokens "
        f"(max_new_tokens={cfg.max_new_tokens})"
    )

    load = result.load
    aggregate = result.aggregate
    if load or aggregate.get("cold_start_ttft_duration_ms"):
        console.print()
        console.print("[bold]Startup[/bold]")
        console.print(
            f"  Session load: {float(load.get('session_load_duration_ms') or 0.0):.2f} ms  |  "
            f"Native load: {float(load.get('native_load_duration_ms') or 0.0):.2f} ms"
        )
        console.print(
            f"  Cold TTFT: {float(aggregate.get('cold_start_ttft_duration_ms') or 0.0):.2f} ms  |  "
            f"Cold total: {float(aggregate.get('cold_start_total_duration_ms') or 0.0):.2f} ms"
        )
        weight_upload_estimate = float(load.get("weight_upload_estimate_duration_ms") or 0.0)
        if weight_upload_estimate:
            console.print(
                f"  Weight upload estimate: {weight_upload_estimate:.2f} ms "
                f"[dim]({load.get('weight_upload_estimate_source')})[/dim]"
            )

    console.print()
    console.print("[bold]Time to first token (ms)[/bold]")
    model_ttft = aggregate.get("model_ttft_duration_ms", {})
    table = Table(show_header=True, header_style="bold cyan")
    for col in ["Avg", "P50", "P90", "P95", "P99", "Min", "Max"]:
        table.add_column(col, justify="right")
    table.add_row(
        f"{model_ttft.get('mean', 0.0):.2f}",
        f"{model_ttft.get('p50', 0.0):.2f}",
        f"{model_ttft.get('p90', 0.0):.2f}",
        f"{model_ttft.get('p95', 0.0):.2f}",
        f"{model_ttft.get('p99', 0.0):.2f}",
        f"{model_ttft.get('min', 0.0):.2f}",
        f"{model_ttft.get('max', 0.0):.2f}",
    )
    console.print(table)

    prefill_duration = aggregate.get("prefill_duration_ms", {})
    prefill_tps = aggregate.get("prefill_tokens_per_second", {})
    decode_tps = aggregate.get("steady_state_decode_tokens_per_second", {})
    tpot = aggregate.get("steady_state_tpot_ms", {})
    request_duration = aggregate.get("request_duration_ms", {})
    console.print()
    console.print(
        f"[bold]Prefill:[/bold]   {prefill_duration.get('mean', 0.0):.2f} ms avg  |  "
        f"{prefill_tps.get('mean', 0.0):.2f} tokens/sec"
    )
    console.print(
        f"[bold]Decode:[/bold]    {decode_tps.get('mean', 0.0):.2f} tokens/sec  |  "
        f"{tpot.get('mean', 0.0):.2f} ms/token (TPOT)"
    )
    console.print(
        f"[bold]Warm start:[/bold] {request_duration.get('mean', 0.0):.2f} ms avg per generation"
    )
    console.print(f"[bold]Latency:[/bold]    {request_duration.get('mean', 0.0):.2f} ms avg")
    if cfg.warmup > 0:
        console.print(
            f"  [dim]Excluded first {cfg.warmup} warmup generation(s) from statistics[/dim]"
        )

    if result.hw_monitor:
        console.print()
        console.print("[bold]Hardware (during genai benchmark)[/bold]")
        cpu = result.hw_monitor.get("cpu", {})
        ram = result.hw_monitor.get("ram", {})
        device_kind = result.hw_monitor.get("device_kind")
        if device_kind in ACCELERATOR_DEVICE_TYPES:
            adapter = result.hw_monitor.get("adapter") or result.hw_monitor.get(device_kind, {})
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
        vram_local = mem.get("vram_local_after_inference_mb", 0.0)
        vram_shared = mem.get("vram_shared_after_inference_mb", 0.0)
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


def write_genai_report(result: GenaiBenchmarkResult, output_path: str | Path) -> None:
    """Write the genai benchmark result to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)


# =============================================================================
# Entry point
# =============================================================================


def run_genai_perf(
    config: GenaiPerfConfig,
    *,
    console: Console,
    json_mode: bool,
) -> GenaiBenchmarkResult:
    """Run a genai benchmark, print the report, and persist JSON.

    Translates GenaiSession failures into ``click`` errors so the CLI exits
    cleanly instead of dumping a traceback.
    """
    benchmark = GenaiPerfBenchmark(config)
    try:
        result = benchmark.run()
    except GenaiNotInstalledError as exc:
        raise click.ClickException(
            f"{exc} Install it with: pip install onnxruntime-genai-winml"
        ) from exc
    except (GenaiLoadError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc

    if json_mode:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        display_genai_report(result, console)

    output_path = config.output_path or genai_output_path(config.bundle_dir)
    write_genai_report(result, output_path)
    console.print(f"[green]Results saved to:[/green] {output_path}")
    return result
