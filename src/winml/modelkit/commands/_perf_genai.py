# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""GenAI generation benchmarking for ``winml perf --runtime winml-genai``.

Benchmarks a prebuilt ``onnxruntime-genai`` bundle folder through
:class:`GenaiSession`.  Unlike the single-shot WinML path (which times each
``session.run()`` call), decoder pipelines split into a **prefill** phase
(prompt -> first token) and a **decode** phase (subsequent tokens), so this
module reports LLM-style metrics: time-to-first-token (TTFT), prefill latency,
decode throughput (tokens/sec), time-per-output-token (TPOT), and total
generation time.

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

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from ..session import (
    GenaiLoadError,
    GenaiNotInstalledError,
    GenaiSession,
    GenaiSessionError,
    GenerationConfig,
)
from ..utils.constants import EPNameOrAlias


if TYPE_CHECKING:
    from rich.console import Console

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

RUNTIME_TYPE = "winml-genai"

# Built-in benchmark prompt.  Mirrored by the ``--prompt`` CLI default and the
# ``GenaiPerfConfig.prompt`` field default (a test asserts the two stay in sync).
_DEFAULT_PROMPT = "Explain the theory of relativity in simple terms."

# --device -> GenaiSession EP override.  The default ("auto") resolves to
# ``None`` = *respect the bundle config*: genai bundles route each stage via its
# own session_options in genai_config.json (e.g. ctx/iter on the NPU/QNN,
# embeddings/lm_head on CPU), and ``None`` honors that per-stage routing while
# still registering the WinML EPs those stages need.  A concrete device forces
# the whole decoder pipeline onto one EP override: "cpu" strips the hardware
# providers (CPU fallback, no WinML EP registration), "qnn"/"dml" route every
# stage to that accelerator.  An unknown device also falls back to ``None``.
_DEVICE_TO_GENAI_EP: dict[str, EPNameOrAlias | None] = {
    "cpu": "cpu",
    "npu": "qnn",
    "gpu": "dml",
    "auto": None,
}


def device_to_genai_ep(device: str) -> EPNameOrAlias | None:
    """Map a ``--device`` value to a :class:`GenaiSession` EP override.

    Returns ``None`` for ``auto`` (and any unrecognized device), meaning the
    bundle's ``genai_config.json`` routing is respected; otherwise a concrete
    EP short name that forces the whole decoder pipeline onto that provider.
    """
    return _DEVICE_TO_GENAI_EP.get(device.lower())


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


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class GenaiPerfConfig:
    """Resolved request for a genai generation benchmark."""

    bundle_dir: Path
    ep: EPNameOrAlias | None = None
    device: str = "auto"
    prompt: str = _DEFAULT_PROMPT
    max_new_tokens: int = 128
    iterations: int = 10
    warmup: int = 2
    compile: bool = False
    compile_timeout: int = 300
    context_length: int | None = None
    output_path: Path | None = None


@dataclass
class _RunSample:
    """Timing captured for a single full generation."""

    ttft_ms: float
    prefill_ms: float
    total_ms: float
    decode_tokens_per_sec: float
    tpot_ms: float
    n_tokens: int


@dataclass
class GenaiBenchmarkResult:
    """Aggregated results from a genai generation benchmark."""

    config: GenaiPerfConfig
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Generation shape
    prompt_tokens: int = 0
    generated_tokens: int = 0
    context_length: int | None = None

    # Time to first token (prefill + first decode), milliseconds
    ttft_mean_ms: float = 0.0
    ttft_min_ms: float = 0.0
    ttft_max_ms: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p90_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    ttft_p99_ms: float = 0.0

    # Prefill / prompt-processing phase (og append_tokens), milliseconds
    prefill_mean_ms: float = 0.0

    # Decode phase
    decode_tokens_per_sec: float = 0.0
    avg_token_latency_ms: float = 0.0
    # Time per output token — steady-state decode (og generate_next_token), ms
    tpot_mean_ms: float = 0.0

    # Whole generation (prefill + all decode), milliseconds
    total_generation_mean_ms: float = 0.0

    # Per-iteration samples (warmup excluded)
    raw_ttft_ms: list[float] = field(default_factory=list)
    raw_prefill_ms: list[float] = field(default_factory=list)
    raw_decode_tokens_per_sec: list[float] = field(default_factory=list)
    raw_tpot_ms: list[float] = field(default_factory=list)
    raw_total_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "benchmark_info": {
                "runtime": RUNTIME_TYPE,
                "bundle_dir": str(self.config.bundle_dir),
                "ep": self.config.ep or "config",
                "device": self.config.device,
                "compile": self.config.compile,
                "compile_timeout": self.config.compile_timeout,
                "iterations": self.config.iterations,
                "warmup": self.config.warmup,
                "max_new_tokens": self.config.max_new_tokens,
                "prompt_tokens": self.prompt_tokens,
                "generated_tokens": self.generated_tokens,
                "context_length": self.context_length,
                "timestamp": self.timestamp,
            },
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
            "total_generation_ms": {"mean": round(self.total_generation_mean_ms, 3)},
            "raw": {
                "ttft_ms": [round(v, 3) for v in self.raw_ttft_ms],
                "prefill_ms": [round(v, 3) for v in self.raw_prefill_ms],
                "decode_tokens_per_sec": [round(v, 2) for v in self.raw_decode_tokens_per_sec],
                "tpot_ms": [round(v, 3) for v in self.raw_tpot_ms],
                "total_ms": [round(v, 3) for v in self.raw_total_ms],
            },
        }


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
        The prompt is pre-encoded once (via :meth:`GenaiSession.encode`, which
        also loads the model) so model-load and tokenization costs are
        excluded from the timed generations.  Each timed generation is driven
        by :meth:`GenaiSession.generate_timed`, which captures wall-clock spans
        at the onnxruntime-genai call boundaries (``append_tokens`` = prefill,
        each ``generate_next_token`` = one decode step), so TTFT and TPOT
        reflect model compute rather than generator-construction or
        detokenization overhead.
    """

    def __init__(
        self,
        config: GenaiPerfConfig,
        *,
        session: GenaiSession | None = None,
    ) -> None:
        self._config = config
        self._session = session
        self._prompt_token_ids: list[int] = []

    def _build_session(self) -> GenaiSession:
        return GenaiSession(
            self._config.bundle_dir,
            self._config.ep,
            context_length=self._config.context_length,
            compile=self._config.compile,
            compile_timeout=self._config.compile_timeout,
        )

    def _prompt_text(self, session: GenaiSession) -> str:
        """Return the prompt to benchmark, chat-templated when possible.

        Wraps the configured prompt in the bundle's own chat template (via
        :meth:`GenaiSession.apply_chat_template`) so the measured prefill
        matches how the model is actually prompted.  Bundles that ship no chat
        template benchmark the raw prompt unchanged.
        """
        try:
            templated = session.apply_chat_template(self._config.prompt)
        except GenaiSessionError as exc:
            logger.info("genai perf: no chat template applied (%s); benchmarking raw prompt", exc)
            return self._config.prompt
        logger.info("genai perf: applied the bundle's chat template to the prompt")
        return templated

    def run(self) -> GenaiBenchmarkResult:
        """Execute the benchmark and return aggregated metrics."""
        if self._session is None:
            self._session = self._build_session()
        session = self._session

        # Loads the model and tokenizer, then encodes the prompt once.  Both
        # are outside the timed loop so they don't inflate TTFT.  The prompt is
        # wrapped in the bundle's own chat template so the measured prefill
        # reflects realistic chat usage (falls back to the raw prompt when the
        # bundle ships no template).
        self._prompt_token_ids = session.encode(self._prompt_text(session))

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
        samples = [self._time_one_generation(session, gen_config) for _ in range(total_runs)]
        return self._aggregate(samples)

    def _time_one_generation(
        self,
        session: GenaiSession,
        gen_config: GenerationConfig,
    ) -> _RunSample:
        """Run one generation and convert its og-boundary timing to a sample.

        Timing is captured inside :meth:`GenaiSession.generate_timed` at the
        onnxruntime-genai call boundaries (``append_tokens`` = prefill, each
        ``generate_next_token`` = one decode step), so TTFT and TPOT reflect
        model compute rather than generator-construction / detokenization
        overhead.
        """
        timing = session.generate_timed(self._prompt_token_ids, gen_config)
        return _RunSample(
            ttft_ms=timing.ttft_s * 1000.0,
            prefill_ms=timing.prefill_s * 1000.0,
            total_ms=timing.total_s * 1000.0,
            decode_tokens_per_sec=timing.decode_tokens_per_sec,
            tpot_ms=timing.tpot_s * 1000.0,
            n_tokens=timing.generated_tokens,
        )

    def _aggregate(self, samples: list[_RunSample]) -> GenaiBenchmarkResult:
        """Aggregate timed samples (first ``warmup`` runs excluded)."""
        timed = samples[self._config.warmup :] or samples
        ttfts = [s.ttft_ms for s in timed]
        prefills = [s.prefill_ms for s in timed]
        totals = [s.total_ms for s in timed]
        decode_tps = [s.decode_tokens_per_sec for s in timed]
        tpots = [s.tpot_ms for s in timed]
        token_latencies = [s.total_ms / s.n_tokens for s in timed if s.n_tokens]
        sorted_ttfts = sorted(ttfts)

        return GenaiBenchmarkResult(
            config=self._config,
            prompt_tokens=len(self._prompt_token_ids),
            generated_tokens=timed[0].n_tokens if timed else 0,
            context_length=self._session.context_length if self._session else None,
            ttft_mean_ms=_mean(ttfts),
            ttft_min_ms=min(ttfts) if ttfts else 0.0,
            ttft_max_ms=max(ttfts) if ttfts else 0.0,
            ttft_p50_ms=_percentile(sorted_ttfts, 50),
            ttft_p90_ms=_percentile(sorted_ttfts, 90),
            ttft_p95_ms=_percentile(sorted_ttfts, 95),
            ttft_p99_ms=_percentile(sorted_ttfts, 99),
            prefill_mean_ms=_mean(prefills),
            decode_tokens_per_sec=_mean(decode_tps),
            avg_token_latency_ms=_mean(token_latencies),
            tpot_mean_ms=_mean(tpots),
            total_generation_mean_ms=_mean(totals),
            raw_ttft_ms=ttfts,
            raw_prefill_ms=prefills,
            raw_decode_tokens_per_sec=decode_tps,
            raw_tpot_ms=tpots,
            raw_total_ms=totals,
        )


# =============================================================================
# Reporting
# =============================================================================


def display_genai_report(result: GenaiBenchmarkResult, console: Console) -> None:
    """Render a genai benchmark report to the console."""
    from rich.table import Table

    cfg = result.config
    console.print()
    console.print(f"[dim]Runtime:[/dim]   {RUNTIME_TYPE}")
    ep_label = cfg.ep or "config"
    device_str = cfg.device if cfg.device == ep_label else f"{cfg.device} ({ep_label})"
    console.print(f"[dim]Device:[/dim]    {device_str}")
    console.print(f"[dim]Bundle:[/dim]    {cfg.bundle_dir}")
    console.print(
        f"[dim]Prompt:[/dim]    {result.prompt_tokens} tokens   "
        f"[dim]Generated:[/dim] {result.generated_tokens} tokens "
        f"(max_new_tokens={cfg.max_new_tokens})"
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
    console.print(
        f"[bold]Prefill:[/bold]   {result.prefill_mean_ms:.2f} ms avg (prompt processing)"
    )
    console.print(
        f"[bold]Decode:[/bold]    {result.decode_tokens_per_sec:.2f} tokens/sec  |  "
        f"{result.tpot_mean_ms:.2f} ms/token (TPOT)"
    )
    console.print(
        f"[bold]Total:[/bold]     {result.total_generation_mean_ms:.2f} ms avg per generation"
    )
    if cfg.warmup > 0:
        console.print(
            f"  [dim]Excluded first {cfg.warmup} warmup generation(s) from statistics[/dim]"
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
