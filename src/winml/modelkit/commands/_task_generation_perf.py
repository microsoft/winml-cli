# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Schema 3 benchmark engine for Windows ML Task text generation (experimental).

``winml perf`` benchmarks text generation through a native ``windowsml.tasks``
session, whichever engine runs the model. An engine binding (llama.cpp in
:mod:`._perf_gguf`, CGC in :mod:`._perf_cgc`) supplies its session, memory
tracker, result and load metrics through the hooks of :class:`TaskPerfBenchmark`. This
module owns the request loop, aggregation, the schema 3 report, its console
view and JSON writer, the shared :func:`run_task_perf` entry point, and the GGUF
shard and token-ID file helpers.

Schema version 3 measures Task TTFT and successive stream-update arrivals;
unavailable model-only phases remain null in the ``load`` / ``requests`` /
``aggregate`` / ``memory`` layout.

At runtime the module imports only ``click``, ``rich``, the generation metrics
helpers and the standard library, so loading it never loads Windows ML or a
session backend.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar

import click
from rich.markup import escape

from ._generation_metrics import (
    _GenerationMemoryTracker,
    _round_stats,
    _stats,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from rich.console import Console

    from ..session import TaskGenerationTiming, TaskSessionError
    from ..utils.constants import RuntimeName

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

RUNTIME_TYPE: RuntimeName = "winml-runtime"
GGUF_SUFFIX = ".gguf"
# gguf-split names every shard <prefix>-<index>-of-<count>.gguf (five digits each).
_SPLIT_SHARD = re.compile(
    r"^(?P<prefix>.+)-\d{5}(?P<separator>-of-)(?P<count>\d{5})(?P<suffix>\.gguf)$",
    re.IGNORECASE,
)
_UNAVAILABLE_MODEL_METRICS = (
    "prefill_duration_ms",
    "prefill_tokens_per_second",
    "first_token_duration_ms",
    "detokenization_duration_ms",
    "model_ttft_duration_ms",
    "model_compute_duration_ms",
    "response_eval_duration_ms",
    "response_eval_tokens_per_second",
    "sequence_fetch_duration_ms",
)
_MEASURED_REQUEST_METRICS = (
    "template_duration_ms",
    "tokenization_duration_ms",
    "generator_create_duration_ms",
    "task_ttft_duration_ms",
    "task_generation_duration_ms",
    "task_stream_create_duration_ms",
    "task_stream_completion_duration_ms",
    "result_fetch_duration_ms",
    "request_ttft_duration_ms",
    "request_duration_ms",
    "steady_state_tpot_ms",
    "steady_state_decode_tokens_per_second",
)


# =============================================================================
# Input helpers
# =============================================================================


def gguf_shard_paths(entry: Path, *, require_exists: bool = True) -> list[Path]:
    """Return every shard of a GGUF model, given its entry file.

    A gguf-split name (``<prefix>-00001-of-00003.gguf``) expands to all shards
    in the same folder; any other name is a single-file model.

    Set ``require_exists=False`` to protect all expected shard paths before
    output files are opened. An evidence run checks the shards right after its
    start write, so a missing shard ends that run as failed with
    ``FileNotFoundError``.

    Raises:
        FileNotFoundError: A split shard is missing and ``require_exists`` is true.
    """
    match = _SPLIT_SHARD.match(entry.name)
    if match is None:
        return [entry]
    count = int(match["count"])
    shards = [
        entry.with_name(
            f"{match['prefix']}-{index:05d}{match['separator']}{match['count']}{match['suffix']}"
        )
        for index in range(1, count + 1)
    ]
    missing = [shard.name for shard in shards if not shard.is_file()] if require_exists else []
    if missing:
        raise FileNotFoundError(
            f"GGUF split shard(s) not found next to {entry}: {', '.join(missing)}"
        )
    return shards


def read_prompt_ids_file(path: Path) -> tuple[int, ...]:
    """Read exact prompt token IDs from a JSON array file.

    Raises:
        ValueError: The file cannot be read, is not JSON, or is not a non-empty
            array of non-negative integer token IDs.
    """
    try:
        data = json.loads(path.read_bytes())
    except OSError as exc:
        raise ValueError(f"could not be read: {exc}") from exc
    # UnicodeDecodeError: undecodable bytes; RecursionError: nested too deeply.
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise ValueError(f"not valid JSON: {exc}") from exc
    except ValueError as exc:  # valid JSON, but int() refuses an integer this long
        raise ValueError("contains an integer too large to be a token ID") from exc
    if (
        not isinstance(data, list)
        or not data
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in data)
    ):
        raise ValueError("expected a non-empty JSON array of non-negative integer token IDs")
    return tuple(data)


# =============================================================================
# Binding contracts
# =============================================================================


class TaskPerfConfig(Protocol):
    """The resolved benchmark request a :class:`TaskPerfBenchmark` reads.

    Every member is read-only, so a dataclass field satisfies it.
    """

    @property
    def model_path(self) -> Path:
        """The model the binding loads."""

    @property
    def device(self) -> str:
        """Device the engine targets."""

    @property
    def requested_device(self) -> str:
        """The ``--device`` value as passed."""

    @property
    def prompt(self) -> str:
        """Prompt text generated from on every request."""

    @property
    def apply_template(self) -> bool:
        """Render the prompt with the model's chat template."""

    @property
    def max_new_tokens(self) -> int:
        """Upper bound on generated tokens per request."""

    @property
    def iterations(self) -> int:
        """Timed requests."""

    @property
    def warmup(self) -> int:
        """Warmup requests excluded from statistics."""

    @property
    def memory(self) -> bool:
        """Record memory at the load and inference checkpoints."""

    @property
    def prompt_ids(self) -> tuple[int, ...] | None:
        """Exact prompt token IDs submitted instead of encoding ``prompt``."""

    @property
    def context_length(self) -> int | None:
        """Fixed sequence capacity the loaded session must report exactly."""

    @property
    def ignore_eos(self) -> bool:
        """Create the Task with an empty EOS set."""


class TaskGenerationSession(Protocol):
    """The Windows ML Task generation session a :class:`TaskPerfBenchmark` drives.

    The tokenizer opens before the model loads, so the benchmark can size the
    sequence capacity from the encoded prompt.
    """

    chat_template_applied: bool | None
    load_timings_ms: dict[str, float]

    @property
    def context_length(self) -> int | None:
        """Effective sequence capacity after :meth:`load`, or ``None`` when unknown."""

    @property
    def effective_device(self) -> str | None:
        """Device class the engine resolved to, or ``None`` when unknown."""

    @property
    def task_default_eos_token_ids(self) -> tuple[int, ...]:
        """EOS IDs the native Task stops on, read back after :meth:`load`."""

    def open_tokenizer(self) -> None:
        """Open the model's tokenizer, the first step of the load."""

    def encode_prompt(self, prompt: str, *, apply_template: bool = ...) -> list[int]:
        """Encode one user prompt and set :attr:`chat_template_applied`.

        With ``apply_template`` the prompt is rendered through the model's chat
        template when the model has one.
        """

    def load(
        self, *, context_length: int | None = ..., eos_token_ids: Sequence[int] | None = ...
    ) -> None:
        """Load the model and create its native Task session."""

    def generate_timed(
        self,
        prompt_token_ids: Iterable[int],
        *,
        max_new_tokens: int,
        clock: Callable[[], float] = ...,
    ) -> TaskGenerationTiming:
        """Run one greedy generation and measure Task stream latency."""

    def close(self) -> None:
        """Release native handles."""


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class TaskRequestSample:
    """Request timings observable at the native Task stream boundary."""

    kind: str
    index: int
    prompt_tokens: int
    generated_tokens: int
    template_duration_ms: float
    tokenization_duration_ms: float
    generator_create_duration_ms: float
    task_ttft_duration_ms: float | None
    task_generation_duration_ms: float
    task_stream_create_duration_ms: float
    task_stream_completion_duration_ms: float
    result_fetch_duration_ms: float
    decode_token_durations_ms: list[float]
    response_text: str = ""

    @property
    def request_ttft_duration_ms(self) -> float | None:
        """Prompt preparation and Task execution through the first update."""
        if self.task_ttft_duration_ms is None:
            return None
        return (
            self.template_duration_ms
            + self.tokenization_duration_ms
            + self.generator_create_duration_ms
            + self.task_ttft_duration_ms
        )

    @property
    def request_duration_ms(self) -> float:
        """Prompt preparation through the completed Task result fetch."""
        return (
            self.template_duration_ms
            + self.tokenization_duration_ms
            + self.generator_create_duration_ms
            + self.task_generation_duration_ms
        )

    @property
    def steady_state_tpot_ms(self) -> float | None:
        """Mean time between updates, including native sampling and text decode."""
        if not self.decode_token_durations_ms:
            return None
        return sum(self.decode_token_durations_ms) / len(self.decode_token_durations_ms)

    @property
    def steady_state_decode_tokens_per_second(self) -> float | None:
        """Update throughput after the first token, when intervals are measurable."""
        total_ms = sum(self.decode_token_durations_ms)
        return len(self.decode_token_durations_ms) * 1000.0 / total_ms if total_ms > 0 else None

    def to_dict(self) -> dict[str, Any]:
        """Serialize measured request spans and explicit unavailable legacy fields."""
        result: dict[str, Any] = {
            "kind": self.kind,
            "index": self.index,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "decode_token_durations_ms": [
                round(value, 3) for value in self.decode_token_durations_ms
            ],
            **dict.fromkeys(_UNAVAILABLE_MODEL_METRICS),
        }
        for metric in _MEASURED_REQUEST_METRICS:
            value = getattr(self, metric)
            precision = 2 if metric == "steady_state_decode_tokens_per_second" else 3
            result[metric] = round(value, precision) if value is not None else None
        return result


@dataclass
class TaskBenchmarkResult:
    """Canonical results from a Task generation benchmark.

    A binding subclass names its engine in ``BACKEND``, adds its own
    ``benchmark_info`` fields through :meth:`_backend_fields` and its console
    rows through :meth:`display_rows`.
    """

    BACKEND: ClassVar[str]

    config: TaskPerfConfig
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    prompt_tokens: int = 0
    generated_tokens: int = 0
    response_text: str = ""
    stop_reason: str | None = None
    context_length: int | None = None
    effective_device: str | None = None
    chat_template_applied: bool | None = None
    packages: dict[str, str | None] = field(default_factory=dict)
    load: dict[str, float | str | None] = field(default_factory=dict)
    requests: list[TaskRequestSample] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    memory_profile: dict[str, float | None] | None = None
    model_format: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable Task timing report."""
        cfg = self.config
        result: dict[str, Any] = {
            "schema_version": 3,
            "benchmark_info": {
                "runtime": RUNTIME_TYPE,
                "backend": self.BACKEND,
                "api": "windowsml.tasks",
                "timing_source": "task-stream",
                "model_format": self.model_format,
                "experimental": True,
                "model_id": str(cfg.model_path),
                "running_model_path": str(cfg.model_path),
                "ep": None,
                "device": cfg.requested_device,
                "effective_device": self.effective_device,
                "iterations": cfg.iterations,
                "warmup": cfg.warmup,
                "max_new_tokens": cfg.max_new_tokens,
                "ignore_eos": cfg.ignore_eos,
                "apply_template": cfg.apply_template,
                "chat_template_applied": self.chat_template_applied,
                # Exact token IDs are submitted instead of the prompt text.
                "prompt": cfg.prompt if cfg.prompt_ids is None else None,
                "prompt_tokens": self.prompt_tokens,
                "generated_tokens": self.generated_tokens,
                "stop_reason": self.stop_reason,
                "context_length": self.context_length,
                "packages": dict(self.packages),
                **self._backend_fields(),
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

    def _backend_fields(self) -> dict[str, Any]:
        """Engine-specific ``benchmark_info`` fields, placed after ``packages``."""
        return {}

    def display_rows(self) -> list[tuple[str, str]]:
        """Engine-specific ``(label, value)`` console rows, printed after ``Model:``."""
        return []

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


ConfigT = TypeVar("ConfigT", bound=TaskPerfConfig)
ResultT = TypeVar("ResultT", bound=TaskBenchmarkResult)


# =============================================================================
# Benchmark engine
# =============================================================================


class TaskPerfBenchmark(ABC, Generic[ConfigT, ResultT]):
    """Runs warmup + timed Task generations and aggregates LLM metrics.

    The tokenizer is opened before the model so the context capacity can be
    sized to the encoded prompt plus ``max_new_tokens``; without that hint an
    engine may reserve a KV cache for the model's full training context.

    A binding sets the class attributes and implements :meth:`_new_session`,
    :meth:`_new_memory_tracker`, :meth:`_new_result` and :meth:`_load_metrics`;
    the other hooks add nothing by default.

    Attributes:
        SESSION_ERROR: The binding's session error, raised when the prompt plus
            ``max_new_tokens`` exceeds ``--context-length``. :func:`run_task_perf`
            reports it, subclasses included, as a CLI error.
        LOAD_ERROR: Raised when the loaded session does not honor the protocol
            options; a subclass of ``SESSION_ERROR``.
        LOG_LABEL: Engine name in the run log line, e.g. ``"GGUF"``.

    Args:
        config: The resolved benchmark request.
        session: Pre-built session (dependency injection for tests). When
            ``None`` :meth:`_new_session` creates one, which is closed after the
            run.
        clock: Monotonic clock in seconds (injectable for tests).
    """

    SESSION_ERROR: ClassVar[type[TaskSessionError]]
    LOAD_ERROR: ClassVar[type[TaskSessionError]]
    LOG_LABEL: ClassVar[str]

    def __init__(
        self,
        config: ConfigT,
        *,
        session: TaskGenerationSession | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._config: ConfigT = config
        self._session = session
        self._owns_session = session is None
        self._clock = clock
        self._stop_reasons: list[tuple[str, str]] = []

    def run(self) -> ResultT:
        """Execute the benchmark and return aggregated metrics."""
        if self._session is None:
            self._session = self._new_session()
        try:
            return self._run_body(self._session)
        finally:
            if self._owns_session:
                self._session.close()

    # -- binding hooks -----------------------------------------------------------
    @abstractmethod
    def _new_session(self) -> TaskGenerationSession:
        """Create the session for a run without an injected one."""

    @abstractmethod
    def _new_memory_tracker(
        self, session: TaskGenerationSession
    ) -> _GenerationMemoryTracker | None:
        """Return the memory tracker for this run, or ``None`` to skip memory checkpoints."""

    @abstractmethod
    def _new_result(self, session: TaskGenerationSession, **fields: Any) -> ResultT:
        """Build the result from the shared ``fields`` plus the binding's own."""

    @abstractmethod
    def _load_metrics(self, session_load_ms: float) -> dict[str, float | str | None]:
        """Return the startup spans of the ``load`` block."""

    def _run_body(self, session: TaskGenerationSession) -> ResultT:
        """Run the benchmark on ``session``; a binding may wrap the run."""
        return self._run_benchmark_body(session)

    def _after_tokenizer_open(self) -> None:
        """Called once the tokenizer is open, before the prompt is encoded for the load."""

    def _after_model_load(self) -> None:
        """Called once the model is loaded, before the load is verified."""

    def _after_request(self, kind: str, index: int, timing: TaskGenerationTiming) -> None:
        """Called after each request finishes, before its sample is built."""

    def _verify_load(self, session: TaskGenerationSession) -> None:
        """Check the loaded session honors the protocol options before any request.

        Raises:
            TaskSessionError: ``LOAD_ERROR`` when the effective capacity differs
                from ``--context-length``, or the Task kept EOS IDs under
                ``--ignore-eos``.
        """
        config = self._config
        if config.context_length is not None and session.context_length != config.context_length:
            raise self.LOAD_ERROR(
                f"--context-length {config.context_length} resolved to an effective sequence "
                f"capacity of {session.context_length}; the Runtime did not apply the "
                "requested capacity."
            )
        if config.ignore_eos:
            eos_token_ids = session.task_default_eos_token_ids
            if eos_token_ids:
                raise self.LOAD_ERROR(
                    "--ignore-eos requested an empty EOS set, but the Task reports EOS "
                    f"token IDs {list(eos_token_ids)}."
                )

    # -- request loop ------------------------------------------------------------
    def _run_benchmark_body(self, session: TaskGenerationSession) -> ResultT:
        config = self._config
        memory_tracker = self._new_memory_tracker(session)
        if memory_tracker is not None:
            memory_tracker.record_baseline()

        load_start = self._clock()
        session.open_tokenizer()
        tokenizer_ready = self._clock()
        self._after_tokenizer_open()
        # Size the context to this benchmark (excluded from the load span).
        prompt_ids = self._prompt_ids(session)
        model_start = self._clock()
        self._load(session, prompt_ids)
        load_end = self._clock()
        self._after_model_load()
        self._verify_load(session)
        session_load_ms = ((tokenizer_ready - load_start) + (load_end - model_start)) * 1000.0
        if memory_tracker is not None:
            memory_tracker.record_after_load()

        total_runs = config.warmup + config.iterations
        logger.info(
            "%s perf: %d warmup + %d timed generations (max_new_tokens=%d)",
            self.LOG_LABEL,
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
        result = self._aggregate(session, samples, session_load_ms=session_load_ms)

        if memory_tracker is not None:
            memory_tracker.record_after_benchmark()
            result.memory_profile = memory_tracker.to_dict()
        return result

    def _prompt_ids(self, session: TaskGenerationSession) -> list[int]:
        """Return the exact prompt IDs, or encode the prompt text."""
        config = self._config
        if config.prompt_ids is not None:
            return list(config.prompt_ids)
        return session.encode_prompt(config.prompt, apply_template=config.apply_template)

    def _load(self, session: TaskGenerationSession, prompt_ids: Sequence[int]) -> None:
        """Load the model with the sequence capacity this run needs.

        Raises:
            TaskSessionError: ``SESSION_ERROR`` when the prompt plus
                ``max_new_tokens`` exceeds ``--context-length`` (checked before
                loading).
        """
        config = self._config
        needed = len(prompt_ids) + config.max_new_tokens
        context = needed if config.context_length is None else config.context_length
        if needed > context:
            raise self.SESSION_ERROR(
                f"--context-length {context} is smaller than the {len(prompt_ids)} prompt "
                f"tokens plus --max-new-tokens {config.max_new_tokens}."
            )
        session.load(context_length=context, eos_token_ids=() if config.ignore_eos else None)

    def _time_one_generation(
        self,
        session: TaskGenerationSession,
        *,
        kind: str,
        index: int,
    ) -> TaskRequestSample:
        """Run one generation and convert its spans to a canonical request sample.

        The tokenizer renders the chat template and tokenizes in one call, so that
        span is reported as ``tokenization_duration_ms``; ``template_duration_ms``
        is ``0``. Exact prompt IDs are not tokenized, so that span is ``0`` too.
        Resetting the sequence state for a new request is reported as
        ``generator_create_duration_ms``. :meth:`_after_request` sees each
        finished request first.
        """
        config = self._config
        tokenize_start = self._clock()
        prompt_ids = self._prompt_ids(session)
        tokenize_end = self._clock()
        timing = session.generate_timed(
            prompt_ids,
            max_new_tokens=config.max_new_tokens,
            clock=self._clock,
        )
        self._after_request(kind, index, timing)
        self._stop_reasons.append((kind, timing.stop_reason))
        if index == 0:
            logger.info("Model response (iteration 1): %s", timing.response_text)
        else:
            logger.debug("Model response (iteration %d): %s", index + 1, timing.response_text)
        return TaskRequestSample(
            kind=kind,
            index=index,
            prompt_tokens=timing.input_tokens,
            generated_tokens=timing.generated_tokens,
            template_duration_ms=0.0,
            tokenization_duration_ms=(
                (tokenize_end - tokenize_start) * 1000.0 if config.prompt_ids is None else 0.0
            ),
            generator_create_duration_ms=timing.reset_s * 1000.0,
            task_ttft_duration_ms=(
                timing.first_token_s * 1000.0 if timing.first_token_s is not None else None
            ),
            task_generation_duration_ms=timing.generation_s * 1000.0,
            task_stream_create_duration_ms=timing.stream_create_s * 1000.0,
            task_stream_completion_duration_ms=timing.completion_s * 1000.0,
            result_fetch_duration_ms=timing.result_fetch_s * 1000.0,
            decode_token_durations_ms=[value * 1000.0 for value in timing.decode_s],
            response_text=timing.response_text,
        )

    def _aggregate(
        self,
        session: TaskGenerationSession,
        samples: list[TaskRequestSample],
        *,
        session_load_ms: float,
    ) -> ResultT:
        """Aggregate canonical samples (warmup requests excluded from stats)."""
        timed = [sample for sample in samples if sample.kind == "timed"]
        first = samples[0] if samples else None
        first_ttft_ms = first.request_ttft_duration_ms if first is not None else None
        aggregate: dict[str, Any] = {
            "warmup_excluded": True,
            "warmup_request_count": sum(sample.kind == "warmup" for sample in samples),
            "timed_request_count": len(timed),
            "cold_start_ttft_duration_ms": (
                session_load_ms + first_ttft_ms if first_ttft_ms is not None else None
            ),
            "cold_start_total_duration_ms": (
                session_load_ms + first.request_duration_ms if first is not None else None
            ),
            **dict.fromkeys(_UNAVAILABLE_MODEL_METRICS),
        }
        for metric in _MEASURED_REQUEST_METRICS:
            values = [value for sample in timed if (value := getattr(sample, metric)) is not None]
            aggregate[metric] = _stats(values) if values else None
        timed_stops = [reason for kind, reason in self._stop_reasons if kind == "timed"]
        aggregate["stop_reason_counts"] = dict(sorted(Counter(timed_stops).items()))
        first_timed = timed[0] if timed else None
        return self._new_result(
            session,
            config=self._config,
            prompt_tokens=first_timed.prompt_tokens if first_timed else 0,
            generated_tokens=first_timed.generated_tokens if first_timed else 0,
            response_text=first_timed.response_text if first_timed else "",
            stop_reason=timed_stops[0] if timed_stops else None,
            context_length=getattr(session, "context_length", None),
            effective_device=getattr(session, "effective_device", None),
            chat_template_applied=getattr(session, "chat_template_applied", None),
            load=self._load_metrics(session_load_ms),
            requests=samples,
            aggregate=aggregate,
        )


# =============================================================================
# Reporting
# =============================================================================


def _zero_token_advice(result: TaskBenchmarkResult) -> str:
    """Suggest the input changes that can avoid an immediate end-of-sequence stop."""
    cfg = result.config
    choices = ["different token IDs" if cfg.prompt_ids is not None else "another --prompt"]
    if cfg.prompt_ids is None and result.chat_template_applied:
        choices.append("--no-apply-template")
    if not cfg.ignore_eos:
        choices.append("--ignore-eos")
    if len(choices) > 2:
        return f"Try {', '.join(choices[:-1])}, or {choices[-1]}."
    return f"Try {' or '.join(choices)}."


def display_task_report(result: TaskBenchmarkResult, console: Console) -> None:
    """Render a Task generation benchmark report to the console."""
    from rich.table import Table

    def format_metric(value: float | None) -> str:
        return f"{value:.2f}" if value is not None else "n/a"

    def format_mib(value: float | None) -> str:
        return f"{value:.1f}" if value is not None else "unavailable"

    cfg = result.config
    device = result.effective_device or cfg.device
    device_str = device if cfg.requested_device == device else f"{cfg.requested_device} -> {device}"
    console.print()
    console.print(f"[dim]Runtime:[/dim]   {RUNTIME_TYPE} ({result.BACKEND}, experimental)")
    console.print("[dim]API:[/dim]       Windows ML Tasks (task-stream timing)")
    console.print(f"[dim]Device:[/dim]    {device_str}")
    console.print(f"[dim]Model:[/dim]     {escape(str(cfg.model_path))}")
    for label, value in result.display_rows():
        console.print(f"[dim]{label}:[/dim] {escape(value)}")
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
    if "session_create_duration_ms" in load:
        # Only CGC times the first create_session apart: its GPU program compiles there.
        console.print(
            f"  set_model: {float(load.get('set_model_duration_ms') or 0.0):.2f} ms  |  "
            "First session (GPU compile): "
            f"{float(load.get('session_create_duration_ms') or 0.0):.2f} ms"
        )
    console.print(
        f"  Cold TTFT: {format_metric(aggregate.get('cold_start_ttft_duration_ms'))} ms  |  "
        f"Cold total: {format_metric(aggregate.get('cold_start_total_duration_ms'))} ms"
    )

    console.print()
    console.print("[bold]Task TTFT (ms)[/bold]")
    task_ttft = aggregate.get("task_ttft_duration_ms") or {}
    table = Table(show_header=True, header_style="bold cyan")
    for column in ["Avg", "P50", "P90", "P95", "P99", "Min", "Max"]:
        table.add_column(column, justify="right")
    table.add_row(
        *(
            format_metric(task_ttft.get(key))
            for key in ("mean", "p50", "p90", "p95", "p99", "min", "max")
        )
    )
    console.print(table)

    decode_tps = aggregate.get("steady_state_decode_tokens_per_second") or {}
    tpot = aggregate.get("steady_state_tpot_ms") or {}
    request_duration = aggregate.get("request_duration_ms") or {}
    console.print()
    console.print("[bold]Pure prefill:[/bold] unavailable (not exposed by Tasks)")
    console.print(
        "[bold]Decode (Task stream):[/bold] "
        f"{format_metric(decode_tps.get('mean'))} tokens/sec  |  "
        f"{format_metric(tpot.get('mean'))} ms/token (TPOT)"
    )
    console.print(
        f"[bold]Warm start:[/bold] {format_metric(request_duration.get('mean'))} "
        "ms avg per generation"
    )
    if cfg.warmup > 0:
        console.print(
            f"  [dim]Excluded first {cfg.warmup} warmup generation(s) from statistics[/dim]"
        )
    if result.generated_tokens == 0:
        console.print(
            "[yellow]Warning:[/yellow] the model stopped before generating a token, so "
            f"decode metrics are empty. {_zero_token_advice(result)}"
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
        # Present only when the binding tracked memory on a known GPU adapter.
        if "vram_local_after_inference_mb" in mem:
            console.print(
                "  GPU local/shared: "
                f"{format_mib(mem.get('vram_local_after_compile_mb'))}/"
                f"{format_mib(mem.get('vram_shared_after_compile_mb'))} MiB after load  |  "
                f"{format_mib(mem.get('vram_local_after_inference_mb'))}/"
                f"{format_mib(mem.get('vram_shared_after_inference_mb'))} MiB after inference"
            )
    console.print()


def write_task_report(result: TaskBenchmarkResult, output_path: str | Path) -> None:
    """Write the Task generation benchmark result to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)


# =============================================================================
# Entry point
# =============================================================================


def run_task_perf(
    benchmark: TaskPerfBenchmark[Any, ResultT],
    *,
    resolve_output_path: Callable[[], str | Path],
    console: Console,
    json_mode: bool,
) -> ResultT:
    """Run a Task generation benchmark, print its report, and persist JSON.

    Translates the binding's session failures (``SESSION_ERROR``, which covers
    not-installed and load errors) and a missing model file into ``click``
    errors so the CLI exits cleanly instead of dumping a traceback. Any other
    exception, a Ctrl+C included, propagates unchanged. A failed run writes no
    report.

    Args:
        benchmark: The binding's benchmark, ready to run.
        resolve_output_path: Returns the JSON report path; called only after the
            run succeeds.
        console: Console for the report and the saved-report line.
        json_mode: Print the JSON report instead of the console report.

    Returns:
        The benchmark result.

    Raises:
        click.ClickException: The run raised ``SESSION_ERROR`` or
            ``FileNotFoundError``.
    """
    try:
        result = benchmark.run()
    except (benchmark.SESSION_ERROR, FileNotFoundError) as exc:
        raise click.ClickException(str(exc)) from exc

    if json_mode:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        display_task_report(result, console)

    output_path = resolve_output_path()
    write_task_report(result, output_path)
    console.print(f"[green]Results saved to:[/green] {output_path}")
    return result
