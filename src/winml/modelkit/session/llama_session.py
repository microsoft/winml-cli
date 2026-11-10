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
import importlib.util
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..sysinfo import format_pdh_luid
from .task_generation import (
    TaskGenerationTiming,
    TaskSessionError,
    _close_quietly,
    _distribution_version,
    _encode_user_prompt,
    _greedy_task_options,
    _hresult,
    _release_closed_interfaces,
    _run_timed_generation,
    _validate_token_ids,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence


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


class LlamaSessionError(TaskSessionError):
    """A GGUF Runtime session could not complete the requested operation."""


class LlamaNotInstalledError(LlamaSessionError):
    """The preview Runtime package or the llama.cpp payload is not installed."""


class LlamaLoadError(LlamaSessionError):
    """The GGUF model could not be loaded or its pipeline could not be built."""


# =============================================================================
# Data classes
# =============================================================================

# The Task stream timing every Task engine reports, under its llama.cpp name.
LlamaGenerationTiming = TaskGenerationTiming


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


@dataclass(frozen=True)
class LlamaExecutionDiagnostics:
    """The decoder stage's execution diagnostics, copied right after Build.

    Attributes:
        enabled: Whether the stage accepted the request, made before Build, to
            capture them.
        reason: Why they are missing, or ``None`` when they were read.
        coverage: How the stage's work is placed relative to its target's
            device: a ``PlacementCoverage`` name such as ``FULL``, or its number.
        work_unit: What the shares count: a ``StageWorkUnit`` name such as
            ``MODEL_LAYER``, or its number.
        work_shares: One entry per share, by descending unit count: its
            ``device_kind``, ``adapter_luid`` (``0xHHHHHHHH_0xLLLLLLLL``, or
            ``None`` for host code or an adapter the Runtime cannot identify),
            ``implementation`` and ``unit_count``.
        description: The Runtime's text for logging, or ``None`` when it gives
            none; its content is not part of its contract. On llama.cpp it holds
            the device and the offloaded layers and, when the Runtime records
            them, the context, batch, attention and KV-cache settings that
            llama.cpp logged while it created the context, unless a capturing
            log callback had replaced the Runtime's by then.
    """

    enabled: bool
    reason: str | None
    coverage: str | int | None = None
    work_unit: str | int | None = None
    work_shares: tuple[dict[str, Any], ...] = ()
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the fields as a JSON-ready object."""
        return {
            "enabled": self.enabled,
            "reason": self.reason,
            "coverage": self.coverage,
            "work_unit": self.work_unit,
            "work_shares": [dict(share) for share in self.work_shares],
            "description": self.description,
        }


# =============================================================================
# Package and payload discovery
# =============================================================================


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
        if hr == _HRESULT_NOT_SUPPORTED and device == "gpu":
            raise LlamaLoadError(
                f"The Windows ML Runtime has no usable llama.cpp GPU backend ({hex_hr}). "
                "Build a matching backend with Build-LlamaBackend.ps1 and provide its "
                f"vendor runtime libraries, or use --device cpu. {_DOCS_HINT}"
            ) from exc
        raise LlamaLoadError(f"GGUF {phase} failed ({hex_hr}): {exc}") from exc


def _fixed_size(dimension: int) -> int | None:
    """Return a concrete dimension, or ``None`` when the backend left it free."""
    if dimension in (0, _DYNAMIC_DIMENSION):
        return None
    return int(dimension)


def _enum_name(value: Any) -> str | int:
    """The name of a Runtime enum value, or the number a newer Runtime returned."""
    name = getattr(value, "name", None)
    return int(value) if name is None else str(name)


def _enable_execution_diagnostics(stage: Any) -> str | None:
    """Ask the stage to capture execution diagnostics at Build.

    Returns why it could not, or ``None``. Never raises: the diagnostics are
    evidence, not part of the load.
    """
    # An older projection has no such property, and assigning it would only add an attribute.
    if not isinstance(getattr(type(stage), "execution_diagnostics_enabled", None), property):
        return "The installed 'windowsml' package has no stage execution diagnostics."
    try:
        stage.execution_diagnostics_enabled = True
    except Exception as exc:  # an optional Runtime interface must not fail the load
        return f"The decoder stage refused execution diagnostics: {type(exc).__name__}: {exc}"
    return None


def _read_execution_diagnostics(stage: Any, refusal: str | None) -> LlamaExecutionDiagnostics:
    """Copy the built stage's execution diagnostics; a refusal or failed read is the reason."""
    if refusal is not None:
        return LlamaExecutionDiagnostics(enabled=False, reason=refusal)
    try:
        diagnostics = stage.execution_diagnostics
        return LlamaExecutionDiagnostics(
            enabled=True,
            reason=None,
            coverage=_enum_name(diagnostics.coverage),
            work_unit=_enum_name(diagnostics.work_unit),
            work_shares=tuple(
                {
                    "device_kind": _enum_name(share.device_kind),
                    "adapter_luid": (
                        None
                        if share.adapter_luid is None
                        else format_pdh_luid(str(share.adapter_luid))
                    ),
                    "implementation": _enum_name(share.implementation),
                    "unit_count": int(share.unit_count),
                }
                for share in diagnostics.work_shares
            ),
            description=diagnostics.description,
        )
    except Exception as exc:  # a failed read is evidence, not a load failure
        return LlamaExecutionDiagnostics(
            enabled=True,
            reason=f"Could not read the execution diagnostics: {type(exc).__name__}: {exc}",
        )


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
        execution_diagnostics: Ask the decoder stage, when the installed
            ``windowsml`` supports it, to capture its execution diagnostics at
            Build, and keep them, or why they are missing, in
            :attr:`execution_diagnostics`. A stage that accepts can create its
            llama.cpp context during Build instead of on the first request.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        runtime_api: Any = None,
        tasks_api: Any = None,
        execution_diagnostics: bool = False,
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
        self._capture_diagnostics = execution_diagnostics
        self.payload: LlamaPayload | None = None
        self.load_timings_ms: dict[str, float] = {}
        self.chat_template_applied: bool | None = None
        # Set by load() once the pipeline is built, when execution_diagnostics was requested.
        self.execution_diagnostics: LlamaExecutionDiagnostics | None = None

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

        runtime_start = time.perf_counter()
        with _translate_native_errors("runtime creation", device=self._device):
            self._runtime = self._wr.Runtime()
        tokenizer_start = time.perf_counter()
        with _translate_native_errors("tokenizer load", device=self._device):
            self._tokenizer = self._wr.Tokenizer.from_file(str(self._model_path))
        tokenizer_end = time.perf_counter()
        self.load_timings_ms["runtime_create_duration_ms"] = (
            tokenizer_start - runtime_start
        ) * 1000.0
        self.load_timings_ms["tokenizer_create_duration_ms"] = (
            tokenizer_end - tokenizer_start
        ) * 1000.0

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
            model_start = time.perf_counter()
            with _translate_native_errors("model load", device=self._device):
                model = runtime.load_model(str(self._model_path))
            build_start = time.perf_counter()
            with _translate_native_errors("pipeline build", device=self._device):
                builder = runtime.create_pipeline_builder()
                stage = builder.add_model_stage(model, stage_target, "decoder")
                # The Runtime publishes only outputs requested before Build.
                stage.request_output(0)
                if context_length:
                    stage.set_sequence_capacity_hint(int(context_length))
                refusal = (
                    _enable_execution_diagnostics(stage) if self._capture_diagnostics else None
                )
                pipeline = builder.build()
            build_end = time.perf_counter()
            if self._capture_diagnostics:
                self.execution_diagnostics = _read_execution_diagnostics(stage, refusal)
            _close_quietly(builder)
            builder = None

            self._token_input_contract(wr, stage)
            self._sequence_capacity = self._probe_sequence_capacity(wr, stage)
            self._vocabulary_size = self._discover_vocabulary(stage)
            effective_device = self._resolved_device(stage) or self._device
            task_start = time.perf_counter()
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
                task_session = task.create_session(configuration)
            task_end = time.perf_counter()
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
        self.load_timings_ms["model_create_duration_ms"] = (build_start - model_start) * 1000.0
        self.load_timings_ms["pipeline_build_duration_ms"] = (build_end - build_start) * 1000.0
        self.load_timings_ms["task_create_duration_ms"] = (task_end - task_start) * 1000.0

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
        _release_closed_interfaces()

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
        token_ids, self.chat_template_applied = _encode_user_prompt(
            self._wr,
            self._tokenizer,
            prompt,
            apply_template=apply_template,
            engine_label="GGUF",
        )
        return token_ids

    # -- generation -----------------------------------------------------------
    def greedy_options(self, max_new_tokens: int) -> Any:
        """Greedy Task options (neutral filters, no stops).

        Every field explicit; watermark disabled when the Task API declares it.
        """
        if self._wt is None:
            self._wt = _import_tasks_api()
        return _greedy_task_options(self._wt, max_new_tokens)

    def generate_timed(
        self,
        prompt_token_ids: Iterable[int],
        *,
        max_new_tokens: int,
        clock: Callable[[], float] = time.perf_counter,
    ) -> LlamaGenerationTiming:
        """Read one native greedy generation and measure Task stream latency.

        The session is reset first and its state is read after the reset, outside
        the timed spans. ``submit_time`` is read immediately before
        ``generate_tokens`` and one reading follows each ``read_next()`` return.

        Args:
            prompt_token_ids: Prompt token IDs, e.g. from :meth:`encode_prompt`.
            max_new_tokens: Upper bound on emitted tokens.
            clock: Monotonic clock in seconds (injectable for tests).

        Returns:
            Timing spans and the clock readings they come from, the emitted token
            IDs and count, text, stop reason, and the tokens left after the reset.

        Raises:
            ValueError: ``max_new_tokens`` is not positive.
            LlamaSessionError: The session is not loaded, the prompt is empty or
                too long, or inference failed.
        """
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self._task_session is None:
            raise LlamaSessionError("WinMLLlamaSession.load() must be called before generation.")
        return _run_timed_generation(
            self._task_session,
            self._wt,
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            sequence_capacity=self._sequence_capacity or None,
            vocabulary_size=self._vocabulary_size,
            clock=clock,
            error_type=LlamaSessionError,
            engine_label="GGUF",
        )

    # -- internals --------------------------------------------------------------
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
        _validate_token_ids(
            special,
            "tokenizer special",
            vocabulary_size=vocabulary_size,
            error_type=LlamaSessionError,
        )
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


__all__ = [
    "LlamaExecutionDiagnostics",
    "LlamaGenerationTiming",
    "LlamaLoadError",
    "LlamaNotInstalledError",
    "LlamaPayload",
    "LlamaSessionError",
    "WinMLLlamaSession",
    "llama_package_versions",
]
