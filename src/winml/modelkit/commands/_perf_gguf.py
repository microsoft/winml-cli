# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""GGUF generation benchmarking for ``winml perf`` (experimental).

``winml perf -m model.gguf`` benchmarks a GGUF decoder on the Windows ML
Runtime llama.cpp backend through :class:`~winml.modelkit.session.WinMLLlamaSession`.
Like ``--runtime ort-genai``, decoder generation splits into a **prefill** phase
(prompt -> first token) and a **decode** phase (later tokens), so the report
reuses the GenAI phase-based schema (``load`` / ``requests`` / ``aggregate`` /
``memory``) with ``benchmark_info.runtime == "winml-runtime"`` and
``benchmark_info.backend == "llama.cpp"``. GGUF and GenAI runs of the same model
therefore compare field by field.

llama.cpp support ships only in preview Windows ML packages that winml-cli does
not depend on (``windowsml`` with ``windowsml.runtime`` plus
``windowsml-llama-core``). Missing packages surface as actionable errors; see
``docs/samples/gguf-llama-perf.md`` for the environment.

The ``perf`` command validates the ``.gguf`` input and delegates here via
:func:`run_gguf_perf`; this module is imported only for GGUF inputs.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from rich.markup import escape

from ..session import LlamaSessionError, WinMLLlamaSession, llama_package_versions
from ._perf_genai import (
    _DEFAULT_PROMPT,
    _aggregate_request_samples,
    _GenaiMemoryTracker,
    _RequestSample,
    _round_stats,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.console import Console

    from ..utils.constants import RuntimeName

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

RUNTIME_TYPE: RuntimeName = "winml-runtime"
BACKEND_NAME = "llama.cpp"
MODEL_FORMAT = "gguf"
GGUF_SUFFIX = ".gguf"


def is_gguf_model(model: str | Path) -> bool:
    """Return whether a ``-m/--model`` value names a GGUF file (by extension)."""
    return Path(model).suffix.lower() == GGUF_SUFFIX


def gguf_output_path(model_path: str | Path) -> Path:
    """Default JSON report path for a GGUF model.

    Delegates to :func:`perf.generate_output_path` so every perf runtime shares
    one report-path convention (``~/.cache/winml/perf/<stem>/<ts>.json``). The
    import is function-local to avoid a module-level cycle with ``perf``.
    """
    from .perf import generate_output_path

    return generate_output_path(str(model_path))


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class GGUFPerfConfig:
    """Resolved request for a GGUF generation benchmark.

    Attributes:
        model_path: The ``.gguf`` file (first shard of a sharded model).
        device: Device the llama.cpp backend targets (``cpu`` or ``gpu``).
        requested_device: The ``--device`` value as passed (``auto`` maps to ``cpu``).
        prompt: Prompt text generated from on every request.
        apply_template: Render the prompt with the GGUF chat template.
        max_new_tokens: Upper bound on generated tokens per request.
        iterations: Timed requests.
        warmup: Warmup requests excluded from statistics.
        memory: Record process RAM at the load and inference checkpoints.
        output_path: JSON report path (default: :func:`gguf_output_path`).
    """

    model_path: Path
    device: str = "cpu"
    requested_device: str = "auto"
    prompt: str = _DEFAULT_PROMPT
    apply_template: bool = True
    max_new_tokens: int = 128
    iterations: int = 10
    warmup: int = 2
    memory: bool = False
    output_path: Path | None = None


@dataclass
class GGUFBenchmarkResult:
    """Canonical results from a GGUF generation benchmark."""

    config: GGUFPerfConfig
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    prompt_tokens: int = 0
    generated_tokens: int = 0
    response_text: str = ""
    stop_reason: str | None = None
    context_length: int | None = None
    effective_device: str | None = None
    chat_template_applied: bool | None = None
    packages: dict[str, str | None] = field(default_factory=dict)
    llama_backends: list[str] = field(default_factory=list)
    load: dict[str, float | str | None] = field(default_factory=dict)
    requests: list[_RequestSample] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    memory_profile: dict[str, float | None] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary (GenAI-compatible schema)."""
        cfg = self.config
        result: dict[str, Any] = {
            "schema_version": 2,
            "benchmark_info": {
                "runtime": RUNTIME_TYPE,
                "backend": BACKEND_NAME,
                "model_format": MODEL_FORMAT,
                "experimental": True,
                "model_id": str(cfg.model_path),
                "running_model_path": str(cfg.model_path),
                "ep": None,
                "device": cfg.requested_device,
                "effective_device": self.effective_device,
                "iterations": cfg.iterations,
                "warmup": cfg.warmup,
                "max_new_tokens": cfg.max_new_tokens,
                "apply_template": cfg.apply_template,
                "chat_template_applied": self.chat_template_applied,
                "prompt": cfg.prompt,
                "prompt_tokens": self.prompt_tokens,
                "generated_tokens": self.generated_tokens,
                "stop_reason": self.stop_reason,
                "context_length": self.context_length,
                "packages": dict(self.packages),
                "llama_backends": list(self.llama_backends),
                "timestamp": self.timestamp,
            },
            "load": {
                key: round(value, 3) if isinstance(value, float) else value
                for key, value in self.load.items()
            },
            "requests": [sample.to_dict() for sample in self.requests],
            "aggregate": self._round_aggregate(),
            "response_text": self.response_text,
        }
        if self.memory_profile:
            result["memory"] = self.memory_profile
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


class GGUFPerfBenchmark:
    """Runs warmup + timed GGUF generations and aggregates LLM metrics.

    The tokenizer is opened before the model so the context capacity can be
    sized to the encoded prompt plus ``max_new_tokens``; without that hint
    llama.cpp may reserve a KV cache for the model's full training context.

    Args:
        config: The resolved benchmark request.
        session: Pre-built session (dependency injection for tests). When
            ``None`` a :class:`WinMLLlamaSession` is created from ``config`` and
            closed after the run.
        clock: Monotonic clock in seconds (injectable for tests).
    """

    def __init__(
        self,
        config: GGUFPerfConfig,
        *,
        session: WinMLLlamaSession | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._config = config
        self._session = session
        self._owns_session = session is None
        self._clock = clock
        self._stop_reasons: list[tuple[str, str]] = []

    def run(self) -> GGUFBenchmarkResult:
        """Execute the benchmark and return aggregated metrics."""
        if self._session is None:
            self._session = WinMLLlamaSession(
                self._config.model_path,
                device=self._config.device,
            )
        try:
            return self._run_benchmark_body(self._session)
        finally:
            if self._owns_session:
                self._session.close()

    def _run_benchmark_body(self, session: WinMLLlamaSession) -> GGUFBenchmarkResult:
        config = self._config
        memory_tracker: _GenaiMemoryTracker | None = None
        if config.memory:
            # llama.cpp does not report which adapter a GPU backend selected, so
            # only process RAM is tracked; VRAM fields are omitted, not guessed.
            memory_tracker = _GenaiMemoryTracker(adapter_luid=None)
            memory_tracker.record_baseline()

        load_start = self._clock()
        session.open_tokenizer()
        tokenizer_ready = self._clock()
        # Size the context to this benchmark (excluded from the load span).
        prompt_ids = session.encode_prompt(config.prompt, apply_template=config.apply_template)
        model_start = self._clock()
        session.load(context_length=len(prompt_ids) + config.max_new_tokens)
        load_end = self._clock()
        session_load_ms = ((tokenizer_ready - load_start) + (load_end - model_start)) * 1000.0
        if memory_tracker is not None:
            memory_tracker.record_after_load()

        total_runs = config.warmup + config.iterations
        logger.info(
            "GGUF perf: %d warmup + %d timed generations (max_new_tokens=%d)",
            config.warmup,
            config.iterations,
            config.max_new_tokens,
        )
        samples = [
            self._time_one_generation(
                session,
                kind="warmup" if index < config.warmup else "timed",
                index=index,
            )
            for index in range(total_runs)
        ]
        result = self._aggregate(samples, session_load_ms=session_load_ms)

        if memory_tracker is not None:
            memory_tracker.record_after_benchmark()
            result.memory_profile = memory_tracker.to_dict()
        return result

    def _time_one_generation(
        self,
        session: WinMLLlamaSession,
        *,
        kind: str,
        index: int,
    ) -> _RequestSample:
        """Run one generation and convert its spans to a canonical request sample.

        The Runtime renders the chat template and tokenizes in one call, so that
        span is reported as ``tokenization_duration_ms``; ``template_duration_ms``
        is ``0``. Resetting the sequence state for a new request is reported as
        ``generator_create_duration_ms``.
        """
        config = self._config
        tokenize_start = self._clock()
        prompt_ids = session.encode_prompt(config.prompt, apply_template=config.apply_template)
        tokenize_end = self._clock()
        timing = session.generate_timed(
            prompt_ids,
            max_new_tokens=config.max_new_tokens,
            clock=self._clock,
        )
        self._stop_reasons.append((kind, timing.stop_reason))
        if index == 0:
            logger.info("Model response (iteration 1): %s", timing.response_text)
        else:
            logger.debug("Model response (iteration %d): %s", index + 1, timing.response_text)
        return _RequestSample(
            kind=kind,
            index=index,
            prompt_tokens=timing.input_tokens,
            generated_tokens=timing.generated_tokens,
            template_duration_ms=0.0,
            tokenization_duration_ms=(tokenize_end - tokenize_start) * 1000.0,
            generator_create_duration_ms=timing.reset_s * 1000.0,
            prefill_duration_ms=timing.prefill_s * 1000.0,
            first_token_duration_ms=timing.first_token_s * 1000.0,
            decode_token_durations_ms=[value * 1000.0 for value in timing.decode_s],
            sequence_fetch_duration_ms=0.0,
            detokenization_duration_ms=timing.detokenization_s * 1000.0,
            response_text=timing.response_text,
        )

    def _aggregate(
        self, samples: list[_RequestSample], *, session_load_ms: float
    ) -> GGUFBenchmarkResult:
        """Aggregate canonical samples (warmup requests excluded from stats)."""
        timed, aggregate = _aggregate_request_samples(samples, session_load_ms=session_load_ms)
        timed_kinds = {sample.kind for sample in timed}
        timed_stops = [reason for kind, reason in self._stop_reasons if kind in timed_kinds]
        aggregate["stop_reason_counts"] = dict(sorted(Counter(timed_stops).items()))
        first_timed = timed[0] if timed else None
        session = self._session
        payload = getattr(session, "payload", None)
        return GGUFBenchmarkResult(
            config=self._config,
            prompt_tokens=first_timed.prompt_tokens if first_timed else 0,
            generated_tokens=first_timed.generated_tokens if first_timed else 0,
            response_text=first_timed.response_text if first_timed else "",
            stop_reason=timed_stops[0] if timed_stops else None,
            context_length=getattr(session, "context_length", None),
            effective_device=getattr(session, "effective_device", None),
            chat_template_applied=getattr(session, "chat_template_applied", None),
            packages=llama_package_versions(),
            llama_backends=list(payload.backends) if payload is not None else [],
            load=self._load_metrics(session_load_ms),
            requests=samples,
            aggregate=aggregate,
        )

    def _load_metrics(self, session_load_ms: float) -> dict[str, float | str | None]:
        """Return startup spans; exact weight-upload telemetry is not exposed."""
        timings = getattr(self._session, "load_timings_ms", None) or {}
        model_create_ms = float(timings.get("model_create_duration_ms", 0.0))
        pipeline_build_ms = float(timings.get("pipeline_build_duration_ms", 0.0))
        return {
            "session_load_duration_ms": session_load_ms,
            "runtime_create_duration_ms": float(timings.get("runtime_create_duration_ms", 0.0)),
            "tokenizer_create_duration_ms": float(timings.get("tokenizer_create_duration_ms", 0.0)),
            "model_create_duration_ms": model_create_ms,
            "pipeline_build_duration_ms": pipeline_build_ms,
            "native_load_duration_ms": model_create_ms + pipeline_build_ms,
            "weight_upload_duration_ms": None,
            "weight_upload_estimate_duration_ms": None,
            "weight_upload_estimate_source": "unavailable",
        }


# =============================================================================
# Reporting
# =============================================================================


def display_gguf_report(result: GGUFBenchmarkResult, console: Console) -> None:
    """Render a GGUF benchmark report to the console."""
    from rich.table import Table

    cfg = result.config
    device = result.effective_device or cfg.device
    device_str = device if cfg.requested_device == device else f"{cfg.requested_device} -> {device}"
    console.print()
    console.print(f"[dim]Runtime:[/dim]   {RUNTIME_TYPE} ({BACKEND_NAME}, experimental)")
    console.print(f"[dim]Device:[/dim]    {device_str}")
    console.print(f"[dim]Model:[/dim]     {escape(str(cfg.model_path))}")
    packages = ", ".join(
        f"{name} {version}" for name, version in result.packages.items() if version
    )
    if packages:
        console.print(f"[dim]Packages:[/dim]  {escape(packages)}")
    template_note = ""
    if cfg.apply_template and result.chat_template_applied is False:
        template_note = "  [dim](no chat template; raw prompt)[/dim]"
    console.print(
        f"[dim]Prompt:[/dim]    {result.prompt_tokens} tokens   "
        f"[dim]Generated:[/dim] {result.generated_tokens} tokens "
        f"(max_new_tokens={cfg.max_new_tokens}, stop={result.stop_reason}){template_note}"
    )
    if result.response_text:
        console.print()
        console.print("[bold]Response[/bold]")
        console.print(result.response_text, markup=False)

    load = result.load
    aggregate = result.aggregate
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

    console.print()
    console.print("[bold]Time to first token (ms)[/bold]")
    model_ttft = aggregate.get("model_ttft_duration_ms", {})
    table = Table(show_header=True, header_style="bold cyan")
    for column in ["Avg", "P50", "P90", "P95", "P99", "Min", "Max"]:
        table.add_column(column, justify="right")
    table.add_row(
        *(
            f"{model_ttft.get(key, 0.0):.2f}"
            for key in ("mean", "p50", "p90", "p95", "p99", "min", "max")
        )
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
    if cfg.warmup > 0:
        console.print(
            f"  [dim]Excluded first {cfg.warmup} warmup generation(s) from statistics[/dim]"
        )
    if result.generated_tokens == 0:
        console.print(
            "[yellow]Warning:[/yellow] the model stopped before generating a token, so "
            "decode metrics are empty. Try another --prompt or --no-apply-template."
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
    console.print()


def write_gguf_report(result: GGUFBenchmarkResult, output_path: str | Path) -> None:
    """Write the GGUF benchmark result to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)


# =============================================================================
# Entry point
# =============================================================================


def run_gguf_perf(
    config: GGUFPerfConfig,
    *,
    console: Console,
    json_mode: bool,
) -> GGUFBenchmarkResult:
    """Run a GGUF benchmark, print the report, and persist JSON.

    Translates session failures into ``click`` errors so the CLI exits cleanly
    instead of dumping a traceback.
    """
    benchmark = GGUFPerfBenchmark(config)
    try:
        result = benchmark.run()
    except (LlamaSessionError, FileNotFoundError) as exc:  # incl. not-installed and load errors
        raise click.ClickException(str(exc)) from exc

    if json_mode:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        display_gguf_report(result, console)

    output_path = config.output_path or gguf_output_path(config.model_path)
    write_gguf_report(result, output_path)
    console.print(f"[green]Results saved to:[/green] {output_path}")
    return result
