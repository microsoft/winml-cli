# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CGC text generation benchmarking for ``winml perf`` (experimental).

``winml perf -m <file>.gguf --backend cgc`` and ``winml perf -m <checkpoint-folder>
--backend cgc`` benchmark GPU text generation through the model-backed Windows ML
Task API and :class:`~winml.modelkit.session.WinMLCGCSession`. This module binds
the shared Task generation engine in :mod:`._task_generation_perf` to CGC: it
creates the session on the pinned GPU, records VRAM on that GPU for
``--memory``, and reports the adapter, model format, vocabulary size, program
cache and the load spans of ``set_model`` and the first ``create_session``,
where the GPU program compiles.

CGC text generation needs the ``windowsml`` 2.8.183.dev0 preview, which winml-cli
does not depend on; see ``docs/samples/cgc-text-generation-perf.md``. Importing
this module imports neither Windows ML nor the llama.cpp binding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from .. import __version__
from ..session import CGCLoadError, CGCSessionError, WinMLCGCSession, cgc_package_versions
from ._generation_metrics import DEFAULT_PROMPT, _GenerationMemoryTracker
from ._task_generation_perf import (
    TaskBenchmarkResult,
    TaskPerfBenchmark,
    gguf_shard_paths,
    run_task_perf,
)


if TYPE_CHECKING:
    from rich.console import Console

    from ._task_generation_perf import TaskGenerationSession


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class CGCPerfConfig:
    """Resolved request for a CGC text generation benchmark.

    ``--evidence`` is llama.cpp-only, so there is no evidence path.

    Attributes:
        model_path: The ``.gguf`` file (first shard of a split model) or the
            safetensors checkpoint folder.
        model_format: ``"gguf"`` or ``"safetensors"``.
        output_path: JSON report path.
        device_luid: LUID of the GPU to pin; ``None`` selects the
            high-performance GPU.
        device: Device CGC targets; always ``gpu``.
        requested_device: The ``--device`` value as passed (``auto`` means ``gpu``).
        prompt: Prompt text generated from on every request.
        apply_template: Render the prompt with the model's chat template.
        max_new_tokens: Upper bound on generated tokens per request.
        iterations: Timed requests.
        warmup: Warmup requests excluded from statistics.
        memory: Record process RAM and the pinned GPU's memory at the load and
            inference checkpoints.
        prompt_ids: Exact prompt token IDs submitted on every request instead of
            encoding ``prompt``.
        context_length: Fixed logical capacity; the run fails unless Windows ML
            reports exactly this capacity after load.
        ignore_eos: Create the Task with an empty EOS set so every request runs to
            ``max_new_tokens`` or the logical capacity.
        program_cache_dir: Directory for compiled GPU programs
            (``--program-cache``); ``None`` compiles on every run.
    """

    model_path: Path
    model_format: Literal["gguf", "safetensors"]
    output_path: Path
    device_luid: str | None = None
    device: str = "gpu"
    requested_device: str = "auto"
    prompt: str = DEFAULT_PROMPT
    apply_template: bool = True
    max_new_tokens: int = 128
    iterations: int = 10
    warmup: int = 2
    memory: bool = False
    prompt_ids: tuple[int, ...] | None = None
    context_length: int | None = None
    ignore_eos: bool = False
    program_cache_dir: Path | None = None


@dataclass
class CGCBenchmarkResult(TaskBenchmarkResult):
    """Canonical results from a CGC text generation benchmark.

    Attributes:
        adapter: The pinned GPU: its ``name``, ``luid`` and how it was
            selected (``selection``).
        vocabulary_size: Tokenizer vocabulary size Windows ML reported, or
            ``None`` when it reported none.
        program_cache_dir: The ``--program-cache`` directory, or ``None`` when off.
    """

    BACKEND: ClassVar[str] = "cgc"

    adapter: dict[str, str] = field(default_factory=dict)
    vocabulary_size: int | None = None
    program_cache_dir: str | None = None

    def _backend_fields(self) -> dict[str, Any]:
        return {
            "adapter": dict(self.adapter),
            "vocabulary_size": self.vocabulary_size,
            "program_cache_dir": self.program_cache_dir,
        }

    def display_rows(self) -> list[tuple[str, str]]:
        """The pinned GPU, model format, logical capacity and program cache."""
        adapter = self.adapter
        capacity = (
            f"{self.context_length} tokens" if self.context_length is not None else "unavailable"
        )
        return [
            ("Adapter", f"{adapter['name']} ({adapter['luid']}, {adapter['selection']})"),
            ("Format", self.model_format),
            ("Capacity", capacity),
            ("Program cache", self.program_cache_dir or "off (cold compile every run)"),
        ]


# =============================================================================
# Benchmark engine
# =============================================================================


class CGCPerfBenchmark(TaskPerfBenchmark[CGCPerfConfig, CGCBenchmarkResult]):
    """Runs warmup + timed CGC generations on one GPU and aggregates LLM metrics.

    The run starts no threads, so every session call happens on the thread that
    calls :meth:`run`; Windows ML requires the thread that created the session.

    Args:
        config: The resolved benchmark request.
        session: Pre-built session (dependency injection for tests). When
            ``None`` a :class:`WinMLCGCSession` is created from ``config`` and
            closed after the run.
        clock: Monotonic clock in seconds (injectable for tests).
    """

    SESSION_ERROR = CGCSessionError
    LOAD_ERROR = CGCLoadError
    LOG_LABEL = "CGC"

    def _new_session(self) -> TaskGenerationSession:
        """Create the session on the requested GPU.

        Raises:
            FileNotFoundError: A split GGUF shard is missing; checked before
                Windows ML loads.
            CGCSessionError: The model-backed Task API is not installed, or no
                hardware GPU fits the request.
        """
        config = self._config
        if config.model_format == "gguf":
            gguf_shard_paths(config.model_path)
        return WinMLCGCSession(
            config.model_path,
            model_format=config.model_format,
            device_luid=config.device_luid,
            program_cache_dir=config.program_cache_dir,
        )

    def _new_memory_tracker(
        self, session: TaskGenerationSession
    ) -> _GenerationMemoryTracker | None:
        if not self._config.memory:
            return None
        # The session pinned the GPU, so its VRAM is tracked next to process RAM.
        return _GenerationMemoryTracker(adapter_luid=cast("WinMLCGCSession", session).adapter_luid)

    def _verify_load(self, session: TaskGenerationSession) -> None:
        """Check the loaded session honors the protocol options before any request.

        Raises:
            CGCLoadError: ``--context-length`` was requested but Windows ML
                reported no logical capacity to verify it against, or a shared
                check failed.
        """
        requested = self._config.context_length
        if requested is not None and session.context_length is None:
            raise CGCLoadError(
                f"--context-length {requested} could not be verified: Windows ML did not "
                "report the logical capacity."
            )
        super()._verify_load(session)

    def _new_result(self, session: TaskGenerationSession, **fields: Any) -> CGCBenchmarkResult:
        cgc_session = cast("WinMLCGCSession", session)
        config = self._config
        return CGCBenchmarkResult(
            packages={**cgc_package_versions(), "winml-cli": __version__},
            model_format=config.model_format,
            adapter={
                "name": cgc_session.adapter_name,
                "luid": cgc_session.adapter_luid,
                "selection": cgc_session.adapter_selection,
            },
            vocabulary_size=cgc_session.vocabulary_size,
            program_cache_dir=(
                None if config.program_cache_dir is None else str(config.program_cache_dir)
            ),
            **fields,
        )

    def _load_metrics(self, session_load_ms: float) -> dict[str, float | str | None]:
        """Return startup spans; exact weight-upload telemetry is not exposed.

        The first ``create_session`` compiles the GPU program, so it is
        reported apart from ``set_model``.
        """
        timings = getattr(self._session, "load_timings_ms", None) or {}
        model_create_ms = float(timings.get("model_create_duration_ms", 0.0))
        set_model_ms = float(timings.get("set_model_duration_ms", 0.0))
        session_create_ms = float(timings.get("session_create_duration_ms", 0.0))
        return {
            "session_load_duration_ms": session_load_ms,
            "runtime_create_duration_ms": float(timings.get("runtime_create_duration_ms", 0.0)),
            "target_create_duration_ms": float(timings.get("target_create_duration_ms", 0.0)),
            "model_create_duration_ms": model_create_ms,
            "set_model_duration_ms": set_model_ms,
            "session_create_duration_ms": session_create_ms,
            "native_load_duration_ms": model_create_ms + set_model_ms + session_create_ms,
            "weight_upload_duration_ms": None,
            "weight_upload_estimate_duration_ms": None,
            "weight_upload_estimate_source": "unavailable",
        }


# =============================================================================
# Entry point
# =============================================================================


def run_cgc_perf(
    config: CGCPerfConfig,
    *,
    console: Console,
    json_mode: bool,
) -> CGCBenchmarkResult:
    """Run a CGC text generation benchmark, print the report, and persist JSON.

    Translates session failures and a missing GGUF shard into ``click`` errors
    so the CLI exits cleanly instead of dumping a traceback; any other
    exception, a Ctrl+C included, propagates. A failed run closes the session
    and writes no report.
    """
    return run_task_perf(
        CGCPerfBenchmark(config),
        resolve_output_path=lambda: config.output_path,
        console=console,
        json_mode=json_mode,
    )
