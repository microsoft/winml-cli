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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click

from ..session import (
    GenaiLoadError,
    GenaiNotInstalledError,
    GenaiSession,
    GenaiSessionError,
    GenerationConfig,
)
from ._perf_generation import (
    GenerationBenchmarkResult,
    display_generation_report,
)


# Backward-compatible aliases — existing code and tests import these names.
GenaiBenchmarkResult = GenerationBenchmarkResult
display_genai_report = display_generation_report


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

# --device -> GenaiSession EP short name.  The default ("auto") resolves to
# "mixed": genai bundles route each stage via its own session_options in
# genai_config.json (e.g. ctx/iter on the NPU/QNN, embeddings/lm_head on CPU).
# "mixed" registers the WinML EPs those stages need while honoring that
# per-stage routing, so it is the correct default for real decoder bundles.
# "cpu" skips WinML EP registration (CPU-only bundles); "qnn"/"dml" register
# the WinML EPs like "mixed" but name the intended accelerator explicitly.
_DEVICE_TO_GENAI_EP: dict[str, str] = {
    "cpu": "cpu",
    "npu": "qnn",
    "gpu": "dml",
    "auto": "mixed",
}


def device_to_genai_ep(device: str) -> str:
    """Map a ``--device`` value to a :class:`GenaiSession` EP short name."""
    return _DEVICE_TO_GENAI_EP.get(device.lower(), "mixed")


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
    ep: str = "mixed"
    device: str = "auto"
    prompt: str = _DEFAULT_PROMPT
    apply_template: bool = True
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
        self._generation_count = 0

    def _build_session(self) -> GenaiSession:
        return GenaiSession(
            self._config.bundle_dir,
            self._config.ep,
            context_length=self._config.context_length,
            compile=self._config.compile,
            compile_timeout=self._config.compile_timeout,
        )

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
        if self._session is None:
            self._session = self._build_session()
        session = self._session

        # Loads the model and tokenizer, then encodes the prompt once.  Both
        # are outside the timed loop so they don't inflate TTFT.  Unless
        # apply_template is disabled, the prompt is wrapped in the bundle's own
        # chat template so the measured prefill reflects realistic chat usage
        # (falls back to the raw prompt when the bundle ships no template).
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
        self._generation_count += 1
        if self._generation_count == 1:
            logger.info("Model response (iteration 1): %s", timing.response_text)
        else:
            logger.debug(
                "Model response (iteration %d): %s",
                self._generation_count,
                timing.response_text,
            )
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

        cfg = self._config
        return GenaiBenchmarkResult(
            runtime=RUNTIME_TYPE,
            model_label=str(cfg.bundle_dir),
            device=cfg.device,
            ep=cfg.ep,
            prompt=cfg.prompt,
            max_new_tokens=cfg.max_new_tokens,
            warmup=cfg.warmup,
            iterations=cfg.iterations,
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
            extra_info={
                "bundle_dir": str(cfg.bundle_dir),
                "compile": cfg.compile,
                "compile_timeout": cfg.compile_timeout,
                "apply_template": cfg.apply_template,
            },
        )


# =============================================================================
# Reporting
# =============================================================================


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
