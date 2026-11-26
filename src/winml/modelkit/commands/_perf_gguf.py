# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""GGUF generation benchmarking for ``winml perf`` (experimental).

``winml perf -m model.gguf`` benchmarks a GGUF decoder through Windows ML Tasks
and :class:`~winml.modelkit.session.WinMLLlamaSession`. The llama.cpp backend's
native Task stream owns sampling and detokenization. Schema version 3 measures
Task TTFT and successive stream-update arrivals; unavailable model-only phases
remain null in the ``load`` / ``requests`` / ``aggregate`` / ``memory`` layout.

This module binds the shared Task generation engine in
:mod:`._task_generation_perf` to llama.cpp: it creates the session, keeps the
``--evidence`` capture, and reports the llama.cpp packages, backends and load
spans.

llama.cpp support ships only in preview Windows ML packages that winml-cli does
not depend on (``windowsml`` with ``windowsml.runtime`` and ``windowsml.tasks`` plus
``windowsml-llama-core``). Missing packages surface as actionable errors; see
``docs/samples/gguf-llama-perf.md`` for the environment.

The ``perf`` command validates the ``.gguf`` input and delegates here via
:func:`run_gguf_perf`; this module is imported only for GGUF inputs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.markup import escape

from ..session import LlamaLoadError, LlamaSessionError, WinMLLlamaSession, llama_package_versions
from ._generation_metrics import (
    DEFAULT_PROMPT as _DEFAULT_PROMPT,
)
from ._generation_metrics import _GenerationMemoryTracker

# The ``name as name`` imports keep names that moved to the shared engine
# importable from this module.
from ._task_generation_perf import GGUF_SUFFIX as GGUF_SUFFIX
from ._task_generation_perf import (
    TaskBenchmarkResult,
    TaskPerfBenchmark,
    display_task_report,
    run_task_perf,
    write_task_report,
)
from ._task_generation_perf import gguf_shard_paths as gguf_shard_paths


if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.console import Console

    from ..session import TaskGenerationTiming
    from ._gguf_evidence import EvidenceRecorder
    from ._gguf_native_log import NativeLogCapture
    from ._task_generation_perf import TaskGenerationSession


# =============================================================================
# Constants
# =============================================================================

BACKEND_NAME = "llama.cpp"
MODEL_FORMAT = "gguf"


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
        prompt_ids: Exact prompt token IDs submitted on every request instead of
            encoding ``prompt``.
        context_length: Fixed sequence capacity; the run fails unless the Runtime
            reports exactly this capacity after load.
        ignore_eos: Create the Task with an empty EOS set so every request runs to
            ``max_new_tokens`` or the sequence capacity.
        evidence_path: Raw ``gguf-task-events-2`` evidence file.
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
    prompt_ids: tuple[int, ...] | None = None
    context_length: int | None = None
    ignore_eos: bool = False
    evidence_path: Path | None = None


@dataclass
class GGUFBenchmarkResult(TaskBenchmarkResult):
    """Canonical results from a GGUF generation benchmark.

    Attributes:
        llama_backends: llama.cpp backend families installed beside the Runtime.
    """

    BACKEND = BACKEND_NAME
    model_format: str = MODEL_FORMAT
    llama_backends: list[str] = field(default_factory=list)

    def _backend_fields(self) -> dict[str, Any]:
        return {"llama_backends": list(self.llama_backends)}


# =============================================================================
# Benchmark engine
# =============================================================================


class GGUFPerfBenchmark(TaskPerfBenchmark[GGUFPerfConfig, GGUFBenchmarkResult]):
    """Runs warmup + timed GGUF generations and aggregates LLM metrics.

    The tokenizer is opened before the model so the context capacity can be
    sized to the encoded prompt plus ``max_new_tokens``; without that hint
    llama.cpp may reserve a KV cache for the model's full training context.
    With ``config.evidence_path`` set, the run also writes its evidence; a
    failure is recorded there, then raised.

    Args:
        config: The resolved benchmark request.
        session: Pre-built session (dependency injection for tests). When
            ``None`` a :class:`WinMLLlamaSession` is created from ``config`` and
            closed after the run.
        clock: Monotonic clock in seconds (injectable for tests).
        log_capture: Native log capture for an evidence run (default: one bound
            to the loaded ``WinMLllama.dll``).
    """

    SESSION_ERROR = LlamaSessionError
    LOAD_ERROR = LlamaLoadError
    LOG_LABEL = "GGUF"

    def __init__(
        self,
        config: GGUFPerfConfig,
        *,
        session: TaskGenerationSession | None = None,
        clock: Callable[[], float] = time.perf_counter,
        log_capture: NativeLogCapture | None = None,
    ) -> None:
        super().__init__(config, session=session, clock=clock)
        self._log_capture = log_capture
        self._evidence: EvidenceRecorder | None = None

    def _new_session(self) -> TaskGenerationSession:
        # Only an evidence run asks for the stage's execution diagnostics: they can
        # move the llama.cpp context creation from the first request into Build.
        return WinMLLlamaSession(
            self._config.model_path,
            device=self._config.device,
            execution_diagnostics=self._config.evidence_path is not None,
        )

    def _new_memory_tracker(
        self, session: TaskGenerationSession
    ) -> _GenerationMemoryTracker | None:
        # An evidence run leaves memory sampling to an external observer.
        if self._config.memory and self._evidence is None:
            # llama.cpp does not report which adapter a GPU backend selected, so
            # only process RAM is tracked; VRAM fields are omitted, not guessed.
            return _GenerationMemoryTracker(adapter_luid=None)
        return None

    def _run_body(self, session: TaskGenerationSession) -> GGUFBenchmarkResult:
        if self._config.evidence_path is not None:
            return self._run_with_evidence(session)
        return self._run_benchmark_body(session)

    def _run_with_evidence(self, session: TaskGenerationSession) -> GGUFBenchmarkResult:
        """Run the benchmark while recording ``gguf-task-events-2`` evidence.

        The start write names this process before the Runtime is imported, and a
        missing model shard fails the run right after it. The benchmark body
        captures llama.cpp's log from model load through request 0 and records
        every finished request. The end write follows on every exit, after the
        Runtime log callback is restored and before the caller closes the session,
        with the execution diagnostics the session copied after Build.
        """
        from ._gguf_evidence import EvidenceRecorder
        from ._gguf_native_log import NativeLogCapture

        evidence = EvidenceRecorder(self._config, capture=self._log_capture or NativeLogCapture())
        self._evidence = evidence
        try:
            gguf_shard_paths(self._config.model_path)
            result = self._run_benchmark_body(session)
        except BaseException as exc:
            evidence.finish(exc, diagnostics=getattr(session, "execution_diagnostics", None))
            raise
        evidence.finish(None, diagnostics=getattr(session, "execution_diagnostics", None))
        return result

    def _after_tokenizer_open(self) -> None:
        if self._evidence is not None:
            # Capture llama.cpp's own log from the model load through request 0.
            self._evidence.capture.install()

    def _after_model_load(self) -> None:
        if self._evidence is not None:
            # A Ctrl+C that the log callback absorbed during the load stops the run.
            self._evidence.capture.raise_if_interrupted()

    def _after_request(self, kind: str, index: int, timing: TaskGenerationTiming) -> None:
        # An evidence run records each request and ends the native log capture
        # after request 0.
        evidence = self._evidence
        if evidence is not None:
            evidence.record_request(kind, index, timing)
            if index == 0:
                # Later requests run with the Runtime's own log callback.
                evidence.capture.restore()
                evidence.capture.raise_if_interrupted()

    def _new_result(self, session: TaskGenerationSession, **fields: Any) -> GGUFBenchmarkResult:
        payload = getattr(session, "payload", None)
        return GGUFBenchmarkResult(
            packages=llama_package_versions(),
            llama_backends=list(payload.backends) if payload is not None else [],
            **fields,
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
            "task_create_duration_ms": float(timings.get("task_create_duration_ms", 0.0)),
            "native_load_duration_ms": model_create_ms + pipeline_build_ms,
            "weight_upload_duration_ms": None,
            "weight_upload_estimate_duration_ms": None,
            "weight_upload_estimate_source": "unavailable",
        }


# =============================================================================
# Reporting
# =============================================================================

# A GGUF report is the shared Task generation report.
display_gguf_report = display_task_report
write_gguf_report = write_task_report


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
    instead of dumping a traceback. A failed evidence run saves only its
    evidence and exits non-zero.
    """
    result = run_task_perf(
        GGUFPerfBenchmark(config),
        resolve_output_path=lambda: config.output_path or gguf_output_path(config.model_path),
        console=console,
        json_mode=json_mode,
    )
    if config.evidence_path is not None:
        console.print(f"Evidence saved to: {escape(str(config.evidence_path))}", soft_wrap=True)
    return result
