# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Windows ML Task session for GGUF language models (experimental).

:class:`WinMLLlamaSession` composes a Runtime decoder pipeline into a native
``windowsml.tasks.TextGenerationTask``. ``Runtime.load_model`` selects llama.cpp
from the GGUF artifact. The Task drives generation and sampling; the CLI reads
its token stream and terminal result. The tokenizer and chat template come from
GGUF metadata through ``windowsml.runtime.Tokenizer``.

llama.cpp support is experimental and optional. It ships only in preview Windows
ML packages that winml-cli does not depend on:

* a preview ``windowsml`` build exposing ``windowsml.runtime`` and ``windowsml.tasks``;
* the ``windowsml-llama-core`` payload (llama.cpp adapter and CPU backend),
  plus a matching GPU backend built with ``Build-LlamaBackend.ps1`` for
  ``device="gpu"``.

Nothing in this module imports ``windowsml`` at import time. The Runtime is
imported on first use, and each missing piece is reported as an actionable
:class:`LlamaNotInstalledError`.

Timing is captured at Task stream boundaries, including native generation,
sampling and text decoding. First-token latency includes stream creation; later
tokens use update arrival intervals. Terminal reads and result retrieval are
included in total generation time. The Task exposes no separate prefill or
detokenization timing, so those model phases are not estimated.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Distributions that make up the llama.cpp path, reported in benchmark results.
_LLAMA_DISTRIBUTIONS: tuple[str, ...] = (
    "windowsml",
    "windowsml-llama-core",
    "windowsml-llama-cuda",
)
# Devices the llama.cpp backend can target through the Runtime.
_LLAMA_DEVICES: tuple[str, ...] = ("cpu", "gpu")

_LLAMA_CORE_DISTRIBUTION = "windowsml-llama-core"
# Payload installed by windowsml-llama-core into ``windowsml/lib``.
_LLAMA_ADAPTER_DLL = "WinMLRuntimeLlama.dll"
_LLAMA_BACKENDS_DIR = "LlamaBackends"
_CPU_BACKEND = "cpu"
_DOCS_HINT = "See docs/samples/gguf-llama-perf.md in the winml-cli repository for the setup."

# HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED): the Runtime has no usable backend
# for the requested target, e.g. --device gpu without a llama.cpp GPU backend.
_HRESULT_NOT_SUPPORTED = 0x80070032

# A stage descriptor reports UINT64_MAX for a dimension the backend left free.
_DYNAMIC_DIMENSION = 0xFFFFFFFFFFFFFFFF


# =============================================================================
# Errors
# =============================================================================


class LlamaSessionError(RuntimeError):
    """A GGUF Runtime session could not complete the requested operation."""


class LlamaNotInstalledError(LlamaSessionError):
    """The preview Runtime package or the llama.cpp payload is not installed."""


class LlamaLoadError(LlamaSessionError):
    """The GGUF model could not be loaded or its pipeline could not be built."""


# =============================================================================
# Data classes
# =============================================================================


@dataclass(frozen=True)
class LlamaGenerationTiming:
    """Task stream wall-clock spans (seconds) for one greedy generation.

    Attributes:
        input_tokens: Prompt token count reported by the Task result.
        generated_tokens: Tokens reported by the Task, excluding terminating EOS.
        reset_s: Resetting the native generation session between requests.
        stream_create_s: Creating the Task stream from encoded prompt tokens.
        first_token_s: From stream creation start through the first token update.
            ``None`` when no token is emitted. Includes sampling and text decoding.
        decode_s: Arrival intervals between successive token updates, including
            native generation, sampling, text decoding, and stream overhead.
        completion_s: From the final update (or stream creation completion) to
            the terminal stream read. Includes EOS detection and finalization.
        result_fetch_s: Reading and checking the terminal result.
        generation_s: Total from stream creation start through result retrieval.
        response_text: Canonical text returned by the Task result.
        stop_reason: Normalized Task finish reason.
    """

    input_tokens: int
    generated_tokens: int
    reset_s: float
    stream_create_s: float
    first_token_s: float | None
    decode_s: list[float]
    completion_s: float
    result_fetch_s: float
    generation_s: float
    response_text: str
    stop_reason: str


# Task finish reasons that end a generation normally, with the stop-reason
# names reported by the benchmark. ERROR and unknown reasons are failures.
_TASK_STOP_REASONS: dict[str, str] = {
    "EOS_TOKEN": "EOS",
    "STOP_TOKEN": "STOP_TOKEN",
    "MAX_TOKENS": "MAX_TOKENS",
    "SEQUENCE_CAPACITY": "CAPACITY",
    "CANCELED": "CANCELED",
}


@dataclass(frozen=True)
class LlamaRequestRecord:
    """Caller-observed ``time.perf_counter_ns`` points and Task outputs of one request.

    Every time point is read by the caller immediately around the Task call it
    names; no backend timing is involved. Failures are recorded rather than
    raised so a benchmark can keep every planned request.

    Attributes:
        input_ids: Token IDs passed to ``generate_tokens``.
        reset_start_ns: Immediately before the session reset.
        reset_end_ns: After the reset returned; ``None`` when it raised.
        reset_error: The reset failure; no request was submitted.
        reset_disclosure: Session state after the reset: ``state``,
            ``sequence_state`` (``disclosure``, ``session_state``,
            ``evaluated_token_count``, ``retained_token_count``) and
            ``retained_token_ids``, with enums as their names.
        disclosure_error: Why the state disclosure could not be read.
        submit_ns: Immediately before ``generate_tokens``.
        stream_created_ns: After ``generate_tokens`` returned the stream.
        events: ``(observed_ns, token_id)`` for each ``read_next()`` update.
        completed_ns: When ``read_next()`` returned the end of the stream.
        result_fetched_ns: After the terminal result was read and checked.
        result_token_ids: Token IDs reported by the Task result.
        prompt_token_count: Prompt token count reported by the Task result.
        finish_reason: Name of the Task finish reason.
        error_code: Task result error code.
        speculative_statistics: ``method`` name and the three draft counts.
        response_text: Canonical text returned by the Task result.
        error: The first failure after submission.
    """

    input_ids: tuple[int, ...]
    reset_start_ns: int
    reset_end_ns: int | None = None
    reset_error: BaseException | None = None
    reset_disclosure: dict[str, Any] | None = None
    disclosure_error: BaseException | None = None
    submit_ns: int | None = None
    stream_created_ns: int | None = None
    events: tuple[tuple[int, int], ...] = ()
    completed_ns: int | None = None
    result_fetched_ns: int | None = None
    result_token_ids: tuple[int, ...] | None = None
    prompt_token_count: int | None = None
    finish_reason: str | None = None
    error_code: int | None = None
    speculative_statistics: dict[str, Any] | None = None
    response_text: str | None = None
    error: BaseException | None = None

    @property
    def task_stop_reason(self) -> str | None:
        """Normalized stop reason; ``None`` for ``ERROR``, unknown reasons, or no result."""
        if self.finish_reason is None:
            return None
        return _TASK_STOP_REASONS.get(self.finish_reason)


@dataclass(frozen=True)
class LlamaPayload:
    """llama.cpp payload discovered beside the installed Runtime.

    Attributes:
        lib_dir: ``windowsml/lib`` directory, or ``None`` when ``windowsml``
            is not installed.
        adapter_present: Whether the Runtime llama.cpp adapter DLL is present.
        backends: Backend families found under ``LlamaBackends`` (e.g. ``cpu``,
            ``cuda``, ``vulkan``).
    """

    lib_dir: Path | None
    adapter_present: bool
    backends: tuple[str, ...]


# =============================================================================
# Package and payload discovery
# =============================================================================


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def llama_package_versions() -> dict[str, str | None]:
    """Return installed versions of the Windows ML llama.cpp distributions.

    Returns:
        Mapping of distribution name to version, ``None`` when not installed.
    """
    return {name: _distribution_version(name) for name in _LLAMA_DISTRIBUTIONS}


def _windowsml_package_dir() -> Path | None:
    """Locate the installed ``windowsml`` package without importing it."""
    spec = importlib.util.find_spec("windowsml")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def _discover_llama_payload(package_dir: Path | None) -> LlamaPayload:
    """Inspect the llama.cpp payload installed beside the Runtime.

    Args:
        package_dir: Directory of the ``windowsml`` package, or ``None``.

    Returns:
        The discovered payload. Missing directories yield an empty payload.
    """
    if package_dir is None:
        return LlamaPayload(lib_dir=None, adapter_present=False, backends=())
    lib_dir = package_dir / "lib"
    backends_dir = lib_dir / _LLAMA_BACKENDS_DIR
    backends: tuple[str, ...] = ()
    if backends_dir.is_dir():
        backends = tuple(sorted(entry.name for entry in backends_dir.iterdir() if entry.is_dir()))
    return LlamaPayload(
        lib_dir=lib_dir,
        adapter_present=(lib_dir / _LLAMA_ADAPTER_DLL).is_file(),
        backends=backends,
    )


def _install_hint(windowsml_version: str | None) -> str:
    if windowsml_version:
        return f'pip install --pre "windowsml-llama-core=={windowsml_version}"'
    return 'pip install --pre "windowsml[with-ort]" windowsml-llama-core'


def _require_llama_backend(
    device: str,
    *,
    package_dir: Path | None = None,
    versions: dict[str, str | None] | None = None,
) -> LlamaPayload:
    """Check that the llama.cpp payload (and a GPU backend for ``gpu``) is installed.

    The check is advisory where the layout is unknown: the payload counts as
    installed when either the ``windowsml-llama-core`` distribution or the
    adapter DLL is present, and a GPU request is rejected only when the
    ``LlamaBackends`` folder exists and holds nothing but the CPU backend.

    Args:
        device: Requested device, ``"cpu"`` or ``"gpu"``.
        package_dir: ``windowsml`` package directory (default: discovered).
        versions: Installed distribution versions (default: discovered).

    Returns:
        The discovered payload.

    Raises:
        LlamaNotInstalledError: The payload or the requested GPU backend is missing.
    """
    if package_dir is None:
        package_dir = _windowsml_package_dir()
    if versions is None:
        versions = llama_package_versions()
    payload = _discover_llama_payload(package_dir)
    if versions.get(_LLAMA_CORE_DISTRIBUTION) is None and not payload.adapter_present:
        raise LlamaNotInstalledError(
            "GGUF models need the llama.cpp backend for the Windows ML Runtime, which is "
            "not installed. Install the payload that matches your 'windowsml' version: "
            f"{_install_hint(versions.get('windowsml'))}. {_DOCS_HINT}"
        )
    if device == "gpu" and payload.backends and set(payload.backends) <= {_CPU_BACKEND}:
        raise LlamaNotInstalledError(
            "--device gpu needs a llama.cpp GPU backend, but only the CPU backend is "
            f"installed in '{payload.lib_dir}'. Build a matching GPU backend with "
            "Build-LlamaBackend.ps1 from the microsoft/WindowsML repository and provide "
            "its vendor runtime libraries. Use "
            f"--device cpu otherwise. {_DOCS_HINT}"
        )
    return payload


def _import_runtime_api() -> Any:
    """Import ``windowsml.runtime`` or raise :class:`LlamaNotInstalledError`."""
    try:
        return importlib.import_module("windowsml.runtime")
    except ImportError as exc:
        raise LlamaNotInstalledError(
            "GGUF models need the preview 'windowsml' package with the Runtime API "
            "(windowsml.runtime), which is not installed. Install the preview packages: "
            f"{_install_hint(None)}. {_DOCS_HINT}"
        ) from exc
    except FileNotFoundError as exc:  # missing WinMLRuntimeCore.dll payload
        raise LlamaNotInstalledError(
            "The installed 'windowsml' package does not ship the Runtime native library "
            "(WinMLRuntimeCore.dll). Install a preview build that includes the Runtime API. "
            f"{_DOCS_HINT}"
        ) from exc


def _import_tasks_api() -> Any:
    """Load the optional Task projection only when a GGUF session is built."""
    try:
        return importlib.import_module("windowsml.tasks")
    except (ImportError, FileNotFoundError) as exc:
        raise LlamaNotInstalledError(
            "GGUF generation needs the preview Windows ML Task API (windowsml.tasks) "
            f"and WinMLTasks.dll. Install a matching preview 'windowsml' package. {_DOCS_HINT}"
        ) from exc


# =============================================================================
# Native error translation
# =============================================================================


def _hresult(exc: BaseException) -> int | None:
    """Return a ``WinMLError``'s normalized 32-bit HRESULT, or ``None``."""
    hr = getattr(exc, "hresult", None)
    if isinstance(hr, int):
        return hr & 0xFFFFFFFF
    return None


@contextmanager
def _translate_native_errors(phase: str, *, device: str) -> Iterator[None]:
    """Translate native ``WinMLError``s into session errors with guidance.

    Exceptions without an HRESULT are re-raised unchanged so genuine bugs stay
    visible.
    """
    try:
        yield
    except LlamaSessionError:
        raise
    except Exception as exc:
        hr = _hresult(exc)
        if hr is None:
            raise
        hex_hr = f"0x{hr:08X}"
        if hr == _HRESULT_NOT_SUPPORTED and device == "gpu" and phase != "run":
            raise LlamaLoadError(
                f"The Windows ML Runtime has no usable llama.cpp GPU backend ({hex_hr}). "
                "Build a matching backend with Build-LlamaBackend.ps1 and provide its "
                f"vendor runtime libraries, or use --device cpu. {_DOCS_HINT}"
            ) from exc
        if phase == "run":
            raise LlamaSessionError(f"GGUF inference failed ({hex_hr}): {exc}") from exc
        raise LlamaLoadError(f"GGUF {phase} failed ({hex_hr}): {exc}") from exc


def _enum_name(value: Any) -> str:
    """Return a projected enum's name (or the value itself as text)."""
    return str(getattr(value, "name", value))


def _read_reset_disclosure(session: Any) -> tuple[dict[str, Any] | None, BaseException | None]:
    """Read the Task session state right after a reset.

    Returns ``(disclosure, None)``, or ``(None, error)`` when the projection or
    the native session cannot disclose its state; generation still proceeds.
    """
    try:
        state = _enum_name(session.state)
        sequence = session.sequence_state
        retained = [int(token) for token in session.retained_token_ids]
        disclosure = {
            "state": state,
            "sequence_state": {
                "disclosure": _enum_name(sequence.disclosure),
                "session_state": _enum_name(sequence.session_state),
                "evaluated_token_count": int(sequence.evaluated_token_count),
                "retained_token_count": int(sequence.retained_token_count),
            },
            "retained_token_ids": retained,
        }
    except Exception as exc:  # disclosure is diagnostic evidence
        return None, exc
    return disclosure, None


def _result_fields(result: Any) -> dict[str, Any]:
    """Copy the terminal Task result fields that a request record keeps."""
    statistics = getattr(result, "speculative_statistics", None)
    return {
        "result_token_ids": tuple(int(token) for token in result.token_ids),
        "prompt_token_count": int(result.prompt_token_count),
        "finish_reason": _enum_name(result.finish_reason),
        "error_code": int(result.error_code),
        "speculative_statistics": None
        if statistics is None
        else {
            "method": _enum_name(statistics.method),
            "verification_step_count": int(statistics.verification_step_count),
            "draft_token_count": int(statistics.draft_token_count),
            "accepted_draft_token_count": int(statistics.accepted_draft_token_count),
        },
        "response_text": str(result.text),
    }


def _close_quietly(resource: Any) -> None:
    close = getattr(resource, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:  # teardown is best-effort
        logger.debug("Failed to close %s", type(resource).__name__, exc_info=True)


def _fixed_size(dimension: int) -> int | None:
    """Return a concrete dimension, or ``None`` when the backend left it free."""
    if dimension in (0, _DYNAMIC_DIMENSION):
        return None
    return int(dimension)


# Stateful-stage draft-prediction options and the type each read-back takes.
_DRAFT_READBACKS: tuple[tuple[str, Callable[[Any], Any]], ...] = (
    ("model_draft_predictor_enabled", bool),
    ("draft_token_limit_hint", int),
    ("draft_predictor_path", str),
)


def _stage_readback(stage: Any, name: str) -> tuple[Any, str | None]:
    """Read one stage property, returning ``(value, None)`` or ``(None, reason)``.

    Read-backs are diagnostics, so any failure (a missing projection property,
    an unsupported native getter) is reported instead of raised.
    """
    try:
        return getattr(stage, name), None
    except Exception as exc:  # diagnostics never fail the run
        return None, f"{name}: {type(exc).__name__}: {exc}"


# =============================================================================
# Session
# =============================================================================


class WinMLLlamaSession:
    """Greedy GGUF generation through Windows ML Tasks over a Runtime pipeline.

    Lifecycle: :meth:`open_tokenizer` imports the Runtime and loads the GGUF
    tokenizer, :meth:`load` loads the model and builds the pipeline, and
    :meth:`generate_timed` runs one timed generation. The tokenizer is opened
    first so a caller can size the context capacity from the encoded prompt
    before the pipeline is built.

    Args:
        model_path: Path to a ``.gguf`` file (the first shard of a sharded model).
        device: ``"cpu"`` or ``"gpu"``. GPU needs a llama.cpp GPU backend beside
            the Runtime.
        runtime_api: Pre-imported ``windowsml.runtime`` module. When given, the
            payload check is skipped (used by tests to inject a fake Runtime).
        tasks_api: Pre-imported ``windowsml.tasks`` module (optional injection).
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        runtime_api: Any = None,
        tasks_api: Any = None,
    ) -> None:
        path = Path(model_path)
        if path.suffix.lower() != ".gguf":
            raise ValueError(f"WinMLLlamaSession requires a .gguf file, got '{model_path}'.")
        if device not in _LLAMA_DEVICES:
            raise ValueError(f"GGUF models support devices {_LLAMA_DEVICES}, got {device!r}.")
        self._model_path = path
        self._device = device
        self._wr: Any = runtime_api
        self._wt: Any = tasks_api
        self._check_payload = runtime_api is None
        self.payload: LlamaPayload | None = None
        self.load_timings_ms: dict[str, float] = {}
        self.chat_template_applied: bool | None = None

        self._runtime: Any = None
        self._tokenizer: Any = None
        self._tasks: Any = None
        self._task_config: Any = None
        self._task: Any = None
        self._task_session: Any = None
        self._stage_target: Any = None
        self._model: Any = None
        self._stage: Any = None
        self._pipeline: Any = None
        self._vocabulary_size: int | None = None
        self._sequence_capacity = 0
        self._effective_device: str | None = None
        self._resolved_target_kind: str | None = None
        self._initialization_spans: list[tuple[str, int, int]] = []

    # -- metadata -----------------------------------------------------------
    @property
    def model_path(self) -> Path:
        """The GGUF file this session loads."""
        return self._model_path

    @property
    def device(self) -> str:
        """The requested device (``cpu`` or ``gpu``)."""
        return self._device

    @property
    def effective_device(self) -> str | None:
        """Device class the decoder stage resolved to, or ``None`` before :meth:`load`."""
        return self._effective_device

    @property
    def context_length(self) -> int | None:
        """Sequence capacity reported by the built stage, or ``None`` before :meth:`load`."""
        return self._sequence_capacity or None

    @property
    def declared_context_length(self) -> int | None:
        """Sequence capacity the model declares, or ``None`` when the stage cannot report it."""
        unavailable: tuple[type[BaseException], ...] = (
            AttributeError,
            TypeError,
            *self._runtime_errors(),
        )
        try:
            return int(self._stage.declared_sequence_capacity)
        except unavailable:
            return None

    @property
    def task_default_eos_token_ids(self) -> tuple[int, ...]:
        """EOS IDs the native Task stops on, read back after :meth:`load`.

        Raises:
            LlamaSessionError: The session is not loaded.
        """
        if self._task is None:
            raise LlamaSessionError(
                "WinMLLlamaSession.load() must be called before reading the Task EOS set."
            )
        with _translate_native_errors("EOS read-back", device=self._device):
            return tuple(int(token) for token in self._task.default_eos_token_ids)

    @property
    def tokenizer_eos_token_ids(self) -> tuple[int, ...]:
        """EOS IDs declared by the GGUF tokenizer metadata.

        Raises:
            LlamaSessionError: The tokenizer is not open.
        """
        if self._tokenizer is None:
            raise LlamaSessionError(
                "WinMLLlamaSession.open_tokenizer() must be called before reading "
                "tokenizer EOS IDs."
            )
        return tuple(int(token) for token in self._tokenizer.eos_token_ids)

    @property
    def speculative_readback(self) -> dict[str, Any]:
        """Draft-prediction settings of the built stage; the CLI never sets them.

        Each value is ``None`` when its read fails, and ``reason`` joins the
        failures (``None`` when every read succeeded).
        """
        readback: dict[str, Any] = {}
        reasons: list[str] = []
        for name, convert in _DRAFT_READBACKS:
            value, reason = _stage_readback(self._stage, name)
            readback[name] = None if value is None else convert(value)
            if reason:
                reasons.append(reason)
        readback["reason"] = "; ".join(reasons) or None
        return readback

    @property
    def resolved_target_kind(self) -> str | None:
        """Execution-target kind the stage reports, without the requested-device fallback."""
        return self._resolved_target_kind

    @property
    def initialization_spans(self) -> list[tuple[str, int, int]]:
        """``(name, start_ns, end_ns)`` for each initialization step that completed."""
        return list(self._initialization_spans)

    # -- lifecycle ----------------------------------------------------------
    def open_tokenizer(self) -> None:
        """Import the Runtime, check the llama.cpp payload, and load the GGUF tokenizer.

        Raises:
            FileNotFoundError: The GGUF file does not exist.
            LlamaNotInstalledError: The preview packages or the payload are missing.
            LlamaLoadError: The Runtime failed to read the tokenizer.
        """
        if self._tokenizer is not None:
            return
        if not self._model_path.is_file():
            raise FileNotFoundError(f"GGUF model not found: {self._model_path}")
        if self._wr is None:
            self._wr = _import_runtime_api()
        if self._check_payload:
            self.payload = _require_llama_backend(self._device)

        runtime_start = time.perf_counter_ns()
        with _translate_native_errors("runtime creation", device=self._device):
            self._runtime = self._wr.Runtime()
        tokenizer_start = time.perf_counter_ns()
        self._initialization_spans.append(("runtime_create", runtime_start, tokenizer_start))
        with _translate_native_errors("tokenizer load", device=self._device):
            self._tokenizer = self._wr.Tokenizer.from_file(str(self._model_path))
        tokenizer_end = time.perf_counter_ns()
        self._initialization_spans.append(("tokenizer_open", tokenizer_start, tokenizer_end))
        self.load_timings_ms["runtime_create_duration_ms"] = (tokenizer_start - runtime_start) / 1e6
        self.load_timings_ms["tokenizer_create_duration_ms"] = (
            tokenizer_end - tokenizer_start
        ) / 1e6

    def load(
        self,
        *,
        context_length: int | None = None,
        eos_token_ids: Sequence[int] | None = None,
    ) -> None:
        """Load GGUF, build the decoder pipeline, and create its native Task session.

        Args:
            context_length: Sequence capacity hint (prompt plus generated tokens).
                ``None`` keeps the backend default, which can be the model's full
                training context and reserve a large KV cache.
            eos_token_ids: Explicit EOS set for the Task. ``None`` keeps the
                tokenizer's EOS IDs; an empty sequence disables EOS stopping so
                generation runs to its token limit or the sequence capacity.

        Raises:
            LlamaLoadError: Loading the model or building the pipeline failed.
        """
        if self._task_session is not None:
            return
        self.open_tokenizer()
        if self._wt is None:
            self._wt = _import_tasks_api()
        wr, runtime = self._wr, self._runtime
        wt = self._wt
        stage_target = model = builder = stage = pipeline = None
        tasks = configuration = task = task_session = None
        try:
            with _translate_native_errors("target creation", device=self._device):
                stage_target = self._create_target(wr, runtime)
            model_start = time.perf_counter_ns()
            with _translate_native_errors("model load", device=self._device):
                model = runtime.load_model(str(self._model_path))
            build_start = time.perf_counter_ns()
            self._initialization_spans.append(("model_load", model_start, build_start))
            with _translate_native_errors("pipeline build", device=self._device):
                builder = runtime.create_pipeline_builder()
                stage = builder.add_model_stage(model, stage_target, "decoder")
                # The Runtime publishes only outputs requested before Build.
                stage.request_output(0)
                if context_length:
                    stage.set_sequence_capacity_hint(int(context_length))
                pipeline = builder.build()
            build_end = time.perf_counter_ns()
            self._initialization_spans.append(("pipeline_build", build_start, build_end))
            _close_quietly(builder)
            builder = None

            self._token_input_contract(wr, stage)
            self._sequence_capacity = self._probe_sequence_capacity(wr, stage)
            self._vocabulary_size = self._discover_vocabulary(stage)
            resolved_target_kind = self._resolved_device(stage)
            effective_device = resolved_target_kind or self._device
            task_start = time.perf_counter_ns()
            with _translate_native_errors("Task creation", device=self._device):
                try:
                    tasks = wt.Tasks(runtime)
                except FileNotFoundError as exc:
                    raise LlamaNotInstalledError(
                        "The installed preview 'windowsml' package is missing WinMLTasks.dll. "
                        f"Reinstall the matching preview package. {_DOCS_HINT}"
                    ) from exc
                configuration = tasks.create_text_generation_configuration()
                configuration.configure_unified(
                    pipeline=pipeline,
                    token_input_stage=stage,
                    token_input_index=0,
                    output_stage=stage,
                    output_index=0,
                    output_kind=wt.TextGenerationOutputKind.LOGITS,
                    state_owner_stage=stage,
                    token_target=stage_target,
                    tokenizer=self._tokenizer,
                )
                configuration.validate()
                if eos_token_ids is None:
                    task = tasks.create_text_generation_task(self._tokenizer)
                else:
                    task = tasks.create_text_generation_task(
                        self._tokenizer, tuple(int(token) for token in eos_token_ids)
                    )
                task_created = time.perf_counter_ns()
                self._initialization_spans.append(("task_create", task_start, task_created))
                task_session = task.create_session(configuration)
            task_end = time.perf_counter_ns()
            self._initialization_spans.append(("session_create", task_created, task_end))
        except Exception:
            for resource in (
                task_session,
                task,
                configuration,
                tasks,
                pipeline,
                stage,
                builder,
                model,
                stage_target,
            ):
                _close_quietly(resource)
            raise

        self._stage_target = stage_target
        self._model = model
        self._stage = stage
        self._pipeline = pipeline
        self._tasks = tasks
        self._task_config = configuration
        self._task = task
        self._task_session = task_session
        self._effective_device = effective_device
        self._resolved_target_kind = resolved_target_kind
        self.load_timings_ms["model_create_duration_ms"] = (build_start - model_start) / 1e6
        self.load_timings_ms["pipeline_build_duration_ms"] = (build_end - build_start) / 1e6
        self.load_timings_ms["task_create_duration_ms"] = (task_end - task_start) / 1e6

    def close(self) -> None:
        """Release native handles (best-effort; safe to call more than once)."""
        for name in (
            "_task_session",
            "_task",
            "_task_config",
            "_tasks",
            "_tokenizer",
            "_pipeline",
            "_stage",
            "_model",
            "_stage_target",
            "_runtime",
        ):
            _close_quietly(getattr(self, name))
            setattr(self, name, None)

    def __enter__(self) -> WinMLLlamaSession:
        """Return the session for use as a context manager."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the session."""
        self.close()

    # -- prompting ------------------------------------------------------------
    def encode_prompt(self, prompt: str, *, apply_template: bool = True) -> list[int]:
        """Encode one user prompt, optionally through the model's chat template.

        Without a usable chat template the raw prompt is encoded instead and
        :attr:`chat_template_applied` is set to ``False``.

        Args:
            prompt: User prompt text.
            apply_template: Render the prompt as one user turn with the GGUF chat
                template; ``False`` encodes the text verbatim with special tokens.

        Returns:
            Prompt token IDs.
        """
        if self._tokenizer is None:
            self.open_tokenizer()
        if apply_template:
            # AttributeError covers Runtime builds without the conversation types.
            template_errors: tuple[type[BaseException], ...] = (
                AttributeError,
                *self._runtime_errors(),
            )
            try:
                token_ids = self._format_user_turn(prompt)
            except template_errors as exc:
                logger.info("GGUF chat template unavailable (%s); encoding the raw prompt", exc)
            else:
                self.chat_template_applied = True
                return token_ids
        self.chat_template_applied = False
        return [int(token) for token in self._tokenizer.encode(prompt, add_special_tokens=True)]

    def _format_user_turn(self, prompt: str) -> list[int]:
        """Render one user turn with the GGUF chat template and return its token IDs."""
        wr = self._wr
        request = wr.ChatCompletionRequest(
            messages=(wr.ChatCompletionMessage(role=wr.ChatRole.USER, content=prompt),),
        )
        with self._tokenizer.format_conversation(request) as formatted:
            return [int(token) for token in formatted.token_ids]

    # -- generation -----------------------------------------------------------
    def greedy_options(self, max_new_tokens: int) -> Any:
        """Greedy Task options with all eleven fields explicit (neutral filters, no stops)."""
        if self._wt is None:
            self._wt = _import_tasks_api()
        return self._wt.TextGenerationOptions(
            max_new_tokens=max_new_tokens,
            stop_token_ids=(),
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            min_p=0.0,
            repetition_penalty=1.0,
            repetition_penalty_window=0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            seed=0,
        )

    def check_prompt_ids(self, token_ids: Sequence[int]) -> None:
        """Reject prompt IDs that are empty, outside the vocabulary, or too long.

        Raises:
            LlamaSessionError: The IDs cannot be submitted to this session.
        """
        if not token_ids:
            raise LlamaSessionError("the tokenizer produced an empty prompt")
        self._validate_token_ids(token_ids, "prompt")
        capacity = self._sequence_capacity
        if capacity and len(token_ids) >= capacity:
            raise LlamaSessionError(
                f"the prompt ({len(token_ids)} tokens) does not fit the model sequence "
                f"capacity ({capacity} tokens)"
            )

    def generate_timed(
        self,
        prompt_token_ids: Iterable[int],
        *,
        max_new_tokens: int,
        clock: Callable[[], float] = time.perf_counter,
    ) -> LlamaGenerationTiming:
        """Read one native greedy generation and measure Task stream latency.

        Args:
            prompt_token_ids: Prompt token IDs, e.g. from :meth:`encode_prompt`.
            max_new_tokens: Upper bound on emitted tokens.
            clock: Monotonic clock in seconds (injectable for tests).

        Returns:
            Timing spans, emitted token count, text, and stop reason.

        Raises:
            ValueError: ``max_new_tokens`` is not positive.
            LlamaSessionError: The session is not loaded, the prompt is empty or
                too long, or inference failed.
        """
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self._task_session is None:
            raise LlamaSessionError("WinMLLlamaSession.load() must be called before generation.")
        token_ids = [int(token) for token in prompt_token_ids]
        self.check_prompt_ids(token_ids)

        options = self.greedy_options(max_new_tokens)
        with _translate_native_errors("run", device=self._device):
            reset_start = clock()
            self._task_session.reset()
            generation_start = clock()
            stream = self._task_session.generate_tokens(token_ids, options)
            try:
                create_end = clock()
                previous_arrival = create_end
                first_token_s: float | None = None
                emitted_ids: list[int] = []
                decode_s: list[float] = []
                while True:
                    update = stream.read_next()
                    arrival = clock()
                    if update is None:
                        completion_s = arrival - previous_arrival
                        break
                    if emitted_ids:
                        decode_s.append(arrival - previous_arrival)
                    else:
                        first_token_s = arrival - generation_start
                    emitted_ids.append(int(update.token_id))
                    previous_arrival = arrival
                result = stream.result
                result.raise_for_error()
                result_end = clock()
                stop_reason = self._task_stop_reason(result.finish_reason)
                result_ids = tuple(int(token) for token in result.token_ids)
                if result_ids != tuple(emitted_ids):
                    raise LlamaSessionError(
                        "The Task result token IDs disagree with its stream updates."
                    )
                if int(result.prompt_token_count) != len(token_ids):
                    raise LlamaSessionError(
                        "The Task result prompt count disagrees with the submitted prompt."
                    )
                if len(result_ids) > max_new_tokens:
                    raise LlamaSessionError(
                        "The Task result exceeds the requested generated token limit."
                    )
            finally:
                _close_quietly(stream)

        return LlamaGenerationTiming(
            input_tokens=int(result.prompt_token_count),
            generated_tokens=len(result_ids),
            reset_s=generation_start - reset_start,
            stream_create_s=create_end - generation_start,
            first_token_s=first_token_s,
            decode_s=decode_s,
            completion_s=completion_s,
            result_fetch_s=result_end - arrival,
            generation_s=result_end - generation_start,
            response_text=str(result.text),
            stop_reason=stop_reason,
        )

    def run_request(
        self,
        token_ids: Sequence[int],
        *,
        options: Any,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> LlamaRequestRecord:
        """Run one Task request and record caller-observed evidence.

        The order of clock reads is the measurement definition: the reset and
        the state disclosure happen before ``submit_ns``, which is read
        immediately before ``generate_tokens``; each ``read_next()`` update is
        one event stamped as soon as the call returns; ``completed_ns`` marks
        the end of the stream and ``result_fetched_ns`` the checked result.

        Args:
            token_ids: Prompt token IDs, submitted unchanged.
            options: Task generation options, e.g. from :meth:`greedy_options`.
            clock_ns: Monotonic integer-nanosecond clock (injectable for tests).

        Returns:
            The record. Reset, stream and result failures are recorded in it
            instead of raised, so the caller can keep every planned request.

        Raises:
            LlamaSessionError: :meth:`load` has not been called.
        """
        if self._task_session is None:
            raise LlamaSessionError("WinMLLlamaSession.load() must be called before generation.")
        session = self._task_session
        ids = tuple(int(token) for token in token_ids)
        reset_start_ns = clock_ns()
        try:
            session.reset()
        except Exception as exc:  # recorded: the session cannot take this request
            return LlamaRequestRecord(
                input_ids=ids, reset_start_ns=reset_start_ns, reset_error=exc
            )
        reset_end_ns = clock_ns()
        disclosure, disclosure_error = _read_reset_disclosure(session)

        fields: dict[str, Any] = {}
        events: list[tuple[int, int]] = []
        stream = None
        try:
            fields["submit_ns"] = clock_ns()
            stream = session.generate_tokens(ids, options)
            fields["stream_created_ns"] = clock_ns()
            while True:
                update = stream.read_next()
                observed_ns = clock_ns()
                if update is None:
                    fields["completed_ns"] = observed_ns
                    break
                events.append((observed_ns, int(update.token_id)))
            result = stream.result
            fields.update(_result_fields(result))
            result.raise_for_error()
            fields["result_fetched_ns"] = clock_ns()
        except Exception as exc:  # recorded: the caller decides what a failure means
            fields["error"] = exc
        finally:
            _close_quietly(stream)
        return LlamaRequestRecord(
            input_ids=ids,
            reset_start_ns=reset_start_ns,
            reset_end_ns=reset_end_ns,
            reset_disclosure=disclosure,
            disclosure_error=disclosure_error,
            events=tuple(events),
            **fields,
        )

    # -- internals --------------------------------------------------------------
    @staticmethod
    def _task_stop_reason(reason: Any) -> str:
        name = str(getattr(reason, "name", ""))
        if name not in _TASK_STOP_REASONS:
            raise LlamaSessionError(f"The Task ended with an unsuccessful finish reason: {name}.")
        return _TASK_STOP_REASONS[name]

    def _runtime_errors(self) -> tuple[type[BaseException], ...]:
        errors = tuple(
            error
            for error in (
                getattr(self._wr, "NotSupportedError", None),
                getattr(self._wr, "WinMLError", None),
            )
            if isinstance(error, type) and issubclass(error, BaseException)
        )
        return errors or (RuntimeError,)

    def _create_target(self, wr: Any, runtime: Any) -> Any:
        if self._device == "cpu":
            return runtime.create_cpu_target()
        preferences = getattr(wr, "ExecutionTargetPreference", None)
        if preferences is None:
            raise LlamaNotInstalledError(
                "The installed 'windowsml' package does not expose the generic GPU "
                f"execution target needed for --device gpu. {_DOCS_HINT}"
            )
        return runtime.create_target(wr.ExecutionTargetKind.GPU, preferences.DEFAULT)

    @staticmethod
    def _token_input_contract(wr: Any, stage: Any) -> tuple[Any, int]:
        with stage.schema() as schema:
            data_type, shape = schema.input_desc(0)
        if data_type not in (wr.TensorDataType.INT32, wr.TensorDataType.INT64):
            raise LlamaLoadError("the GGUF decoder token input must use INT32 or INT64")
        if len(shape) not in (1, 2):
            raise LlamaLoadError("the GGUF decoder token input must have rank 1 or 2")
        return data_type, len(shape)

    @staticmethod
    def _probe_sequence_capacity(wr: Any, stage: Any) -> int:
        try:
            return int(stage.sequence_capacity)
        except wr.NotSupportedError as exc:
            raise LlamaLoadError(
                "the GGUF stage does not expose Runtime sequence state; an autoregressive "
                "decoder is required"
            ) from exc

    def _discover_vocabulary(self, stage: Any) -> int | None:
        with stage.schema() as schema:
            _data_type, output_shape = schema.output_desc(0)
        vocabulary_size = _fixed_size(output_shape[-1]) if output_shape else None
        self._vocabulary_size = vocabulary_size
        eos_tokens = [int(token) for token in self._tokenizer.eos_token_ids]
        if not eos_tokens:
            raise LlamaLoadError("the GGUF tokenizer metadata exposes no EOS token")
        special = list(eos_tokens)
        bos_token = self._tokenizer.bos_token_id
        if bos_token is not None:
            special.append(int(bos_token))
        self._validate_token_ids(special, "tokenizer special")
        return vocabulary_size

    @staticmethod
    def _resolved_device(stage: Any) -> str | None:
        target = getattr(stage, "execution_target", None)
        if target is None:
            return None
        try:
            name = getattr(getattr(target, "kind", None), "name", None)
        finally:
            _close_quietly(target)
        return str(name).lower() if name else None

    def _validate_token_ids(self, token_ids: Sequence[int], label: str) -> None:
        vocabulary_size = self._vocabulary_size
        for token_id in token_ids:
            if token_id < 0 or (vocabulary_size is not None and token_id >= vocabulary_size):
                raise LlamaSessionError(
                    f"{label} token {token_id} is outside the model vocabulary of "
                    f"{vocabulary_size if vocabulary_size is not None else 'unknown'} entries"
                )


__all__ = [
    "LlamaGenerationTiming",
    "LlamaLoadError",
    "LlamaNotInstalledError",
    "LlamaPayload",
    "LlamaRequestRecord",
    "LlamaSessionError",
    "WinMLLlamaSession",
    "llama_package_versions",
]
