# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Windows ML CGC text generation for GGUF files and safetensors checkpoints (experimental).

:class:`WinMLCGCSession` drives the model-backed Task API of the ``windowsml``
2.8.183.dev0 preview: ``Runtime.load_model`` loads a ``.gguf`` file or a
checkpoint folder, and ``TextGenerationConfiguration.set_model`` binds it to an
execution target pinned to one GPU through an owned DXCore adapter. The model
provider installed with Windows ML (CGC) serves the model, or rejects it; no
other engine is ever tried. Native failures are translated by HRESULT into
actionable errors.

Nothing here imports ``windowsml`` at import time. The 2.8-only modules are
imported by name on first use, so type checking against the pinned stable
``windowsml`` stays clean. Generation uses the Task stream timing core shared
with the llama.cpp session (:mod:`.task_generation`), so the timing spans are
wall clock at Task stream boundaries. Every Windows ML call must run on the
thread that created the session.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from .task_generation import (
    TaskGenerationTiming,
    TaskSessionError,
    _close_quietly,
    _distribution_version,
    _encode_user_prompt,
    _hresult,
    _run_timed_generation,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

_WINDOWSML = "windowsml"
# The 2.8 modules that carry the model-backed Task API.
_CGC_MODULES: tuple[str, ...] = ("windowsml.runtime", "windowsml.tasks")

# HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED): the CGC route does not serve the request.
_HRESULT_NOT_SUPPORTED = 0x80070032
# HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND) and HRESULT_FROM_WIN32(ERROR_PATH_NOT_FOUND).
_HRESULTS_NOT_FOUND = frozenset({0x80070002, 0x80070003})
# E_NOINTERFACE: the installed Windows ML lacks a model-backed interface.
_HRESULT_NO_INTERFACE = 0x80004002

# Phases in which ERROR_NOT_SUPPORTED means the CGC backend does not serve the
# model, mapped to the Windows ML call that returned it.
_MODEL_CALLS = {"model load": "load_model", "set_model": "set_model"}


# =============================================================================
# Errors
# =============================================================================


class CGCSessionError(TaskSessionError):
    """A CGC text generation session could not complete the requested operation."""


class CGCNotInstalledError(CGCSessionError):
    """The installed ``windowsml`` lacks the model-backed Task API or a native part of it."""


class CGCLoadError(CGCSessionError):
    """Windows ML could not pin the GPU, load the model or create its Task session."""


# =============================================================================
# Packages and imports
# =============================================================================


def cgc_package_versions() -> dict[str, str | None]:
    """Return the installed version of ``windowsml``, the one package CGC text generation uses.

    Returns:
        ``{"windowsml": version}``; the version is ``None`` when not installed.
    """
    return {_WINDOWSML: _distribution_version(_WINDOWSML)}


def _needs_preview_message() -> str:
    installed = _distribution_version(_WINDOWSML) or "not installed"
    return (
        "CGC text generation needs windowsml 2.8.183.dev0 or newer, which provides the "
        f"model-backed Task API; the installed windowsml is {installed}. Install it in a "
        "separate environment: see docs/samples/cgc-text-generation-perf.md in the "
        "winml-cli repository."
    )


def _missing_native_library(exc: BaseException) -> CGCNotInstalledError:
    return CGCNotInstalledError(
        f"The installed windowsml package is missing a native library: {exc}. "
        "Reinstall windowsml 2.8.183.dev0 or newer."
    )


def _not_installed_error(exc: BaseException) -> CGCNotInstalledError:
    """Explain why the Windows ML Runtime or Task API could not be imported.

    Only a missing ``windowsml`` module means the preview is absent. Any other
    import failure (a different ``DirectML.dll`` already loaded, a missing
    ``comtypes``, an incomplete CGC payload) keeps its own reason.
    """
    if isinstance(exc, FileNotFoundError):
        return _missing_native_library(exc)
    if isinstance(exc, ModuleNotFoundError) and exc.name in {_WINDOWSML, *_CGC_MODULES}:
        return CGCNotInstalledError(_needs_preview_message())
    return CGCNotInstalledError(f"Windows ML could not load its Runtime API: {exc}")


def _require_cgc_modules() -> None:
    """Check that ``windowsml.runtime`` and ``windowsml.tasks`` exist, without importing them.

    ``find_spec`` imports only the parent ``windowsml`` package; a missing
    parent counts as absent.

    Raises:
        CGCNotInstalledError: A module is absent, or ``windowsml`` cannot be imported.
    """
    try:
        found = all(importlib.util.find_spec(name) is not None for name in _CGC_MODULES)
    except (ImportError, FileNotFoundError) as exc:
        raise _not_installed_error(exc) from exc
    if not found:
        raise CGCNotInstalledError(_needs_preview_message())


def _import_cgc_apis() -> tuple[Any, Any]:
    """Import ``windowsml.runtime`` and ``windowsml.tasks``.

    Raises:
        CGCNotInstalledError: An import failed; the message keeps its reason.
    """
    try:
        runtime_api = importlib.import_module("windowsml.runtime")
        tasks_api = importlib.import_module("windowsml.tasks")
    except (ImportError, FileNotFoundError) as exc:
        raise _not_installed_error(exc) from exc
    return runtime_api, tasks_api


# =============================================================================
# GPU adapter
# =============================================================================


class _GPUAdapter(Protocol):
    """The owned GPU adapter a session pins; ``sysinfo.DXCoreAdapterHandle`` provides it.

    A protocol, not an import of ``DXCoreAdapterHandle``: ``sysinfo.device``
    imports this package, and CodeQL reports the cycle even under
    ``TYPE_CHECKING``.
    """

    @property
    def pointer(self) -> int:
        """``IDXCoreAdapter*`` address for ``Runtime.create_target_from_adapter``."""

    @property
    def name(self) -> str:
        """Driver description of the GPU."""

    @property
    def luid(self) -> str:
        """LUID of the GPU, formatted as ``0xHHHHHHHH_0xLLLLLLLL``."""

    @property
    def selection(self) -> str:
        """How the GPU was selected, e.g. ``device-luid`` or ``high-performance``."""

    def close(self) -> None:
        """Release the adapter reference."""


def open_gpu_adapter(device_luid: str | None) -> _GPUAdapter:
    """Own the GPU that ``device_luid`` names, else the high-performance GPU.

    The seam over :func:`winml.modelkit.sysinfo.open_gpu_adapter`, imported in
    the body so this module adds no import edge to ``sysinfo``.

    Raises:
        DXCoreAdapterError: DXCore failed, or no hardware GPU fits the request.
    """
    from ..sysinfo import open_gpu_adapter as open_dxcore_gpu_adapter

    return open_dxcore_gpu_adapter(device_luid)


def _elapsed_ms(start: float) -> float:
    """Wall-clock milliseconds since the ``time.perf_counter()`` reading ``start``."""
    return (time.perf_counter() - start) * 1000.0


def _positive(value: Any) -> int | None:
    """``value`` as a positive ``int``; ``None`` when it is missing or not positive."""
    if value is None:
        return None
    number = int(value)
    return number if number > 0 else None


# =============================================================================
# Session
# =============================================================================


class WinMLCGCSession:
    """Greedy text generation through the model-backed Windows ML Task API on one GPU.

    Lifecycle: the constructor checks that the preview modules exist and owns
    the GPU; :meth:`open_tokenizer` loads the model and binds it to that GPU
    (the tokenizer exists only after ``set_model``); :meth:`load` sets the
    logical capacity and creates the Task session, where the GPU program
    compiles; :meth:`generate_timed` runs one timed generation. Each native
    handle is kept the moment it exists, so :meth:`close` releases whatever a
    failed step acquired, the GPU adapter last.

    Args:
        model_path: A ``.gguf`` file or a safetensors checkpoint folder.
        model_format: ``"gguf"`` or ``"safetensors"``; selects the hints of
            the error messages.
        device_luid: LUID of the GPU to pin, as ``winml sys`` lists it; ``None``
            selects the high-performance GPU.
        program_cache_dir: Directory for compiled GPU programs, created when
            missing and never deleted; ``None`` compiles without a cache.
        runtime_api: Pre-imported ``windowsml.runtime`` module. When both
            modules are injected (tests inject fakes), the package check is
            skipped.
        tasks_api: Pre-imported ``windowsml.tasks`` module.

    Raises:
        CGCNotInstalledError: ``windowsml.runtime`` or ``windowsml.tasks`` is absent.
        CGCLoadError: No hardware GPU fits the request.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_format: Literal["gguf", "safetensors"],
        device_luid: str | None = None,
        program_cache_dir: Path | None = None,
        runtime_api: Any = None,
        tasks_api: Any = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._model_format: str = model_format
        self._program_cache_dir = program_cache_dir
        self._wr: Any = runtime_api
        self._wt: Any = tasks_api
        self.chat_template_applied: bool | None = None
        self.load_timings_ms: dict[str, float] = {}

        self._runtime: Any = None
        self._target: Any = None
        self._model: Any = None
        self._tasks: Any = None
        self._configuration: Any = None
        self._tokenizer: Any = None
        self._task: Any = None
        self._task_session: Any = None
        self._context_length: int | None = None
        self._vocabulary_size: int | None = None
        self._effective_device: str | None = None

        if runtime_api is None or tasks_api is None:
            _require_cgc_modules()
        from ..sysinfo import DXCoreAdapterError

        try:
            adapter = open_gpu_adapter(device_luid)
        except DXCoreAdapterError as exc:
            raise CGCLoadError(str(exc)) from exc
        self._adapter: _GPUAdapter | None = adapter
        self._adapter_name = adapter.name
        self._adapter_luid = adapter.luid
        self._adapter_selection = adapter.selection

    # -- metadata -----------------------------------------------------------
    @property
    def model_path(self) -> Path:
        """The ``.gguf`` file or checkpoint folder this session loads."""
        return self._model_path

    @property
    def model_format(self) -> str:
        """``"gguf"`` or ``"safetensors"``."""
        return self._model_format

    @property
    def adapter_name(self) -> str:
        """Driver description of the pinned GPU."""
        return self._adapter_name

    @property
    def adapter_luid(self) -> str:
        """LUID of the pinned GPU, formatted as ``0xHHHHHHHH_0xLLLLLLLL``."""
        return self._adapter_luid

    @property
    def adapter_selection(self) -> str:
        """How the GPU was selected: ``device-luid``, ``high-performance`` or ``only-gpu``."""
        return self._adapter_selection

    @property
    def effective_device(self) -> str | None:
        """``"gpu"`` once :meth:`load` created the Task session, else ``None``."""
        return self._effective_device

    @property
    def context_length(self) -> int | None:
        """The logical capacity Windows ML reported in :meth:`load`, or ``None``."""
        return self._context_length

    @property
    def vocabulary_size(self) -> int | None:
        """The tokenizer vocabulary size, or ``None`` when not read or not reported."""
        return self._vocabulary_size

    @property
    def task_default_eos_token_ids(self) -> tuple[int, ...]:
        """EOS IDs the native Task stops on, read back after :meth:`load`.

        Raises:
            CGCSessionError: The session is not loaded.
        """
        if self._task_session is None:
            raise CGCSessionError(
                "WinMLCGCSession.load() must be called before reading the Task EOS set."
            )
        with self._translate_cgc_errors("EOS read-back"):
            return tuple(int(token) for token in self._task.default_eos_token_ids)

    # -- lifecycle ----------------------------------------------------------
    def open_tokenizer(self) -> None:
        """Load the model, bind it to the pinned GPU and read the tokenizer it creates.

        Raises:
            CGCNotInstalledError: The installed ``windowsml`` lacks the
                model-backed Task API or one of its native libraries.
            CGCLoadError: Windows ML could not load or bind the model, the
                program cache cannot be created, or no tokenizer was created.
        """
        if self._tokenizer is not None:
            return
        adapter = self._adapter
        if adapter is None:
            raise CGCSessionError("The WinMLCGCSession is closed.")
        if self._wr is None or self._wt is None:
            runtime_api, tasks_api = _import_cgc_apis()
            self._wr = runtime_api if self._wr is None else self._wr
            self._wt = tasks_api if self._wt is None else self._wt
        wr, wt = self._wr, self._wt
        if not hasattr(wt.TextGenerationConfiguration, "set_model"):
            raise CGCNotInstalledError(_needs_preview_message())

        with self._translate_cgc_errors("runtime creation"):
            start = time.perf_counter()
            self._runtime = wr.Runtime()
            self.load_timings_ms["runtime_create_duration_ms"] = _elapsed_ms(start)
        with self._translate_cgc_errors("target creation"):
            start = time.perf_counter()
            self._target = self._runtime.create_target_from_adapter(adapter.pointer)
            self.load_timings_ms["target_create_duration_ms"] = _elapsed_ms(start)
            if self._target.kind != wr.ExecutionTargetKind.GPU:
                raise CGCLoadError(
                    f"The adapter {self._adapter_label} did not resolve to a GPU execution target."
                )
        with self._translate_cgc_errors("model load"):
            start = time.perf_counter()
            self._model = self._runtime.load_model(str(self._model_path))
            self.load_timings_ms["model_create_duration_ms"] = _elapsed_ms(start)
        with self._translate_cgc_errors("Task creation"):
            self._tasks = wt.Tasks(self._runtime)
            self._configuration = self._tasks.create_text_generation_configuration()

        cache_dir = self._program_cache_dir
        if cache_dir is not None:
            # Outside the translator: a FileNotFoundError here is no missing library.
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise CGCLoadError(
                    f"Could not create the --program-cache directory {cache_dir}: {exc}"
                ) from exc
        with self._translate_cgc_errors("set_model"):
            start = time.perf_counter()
            self._configuration.set_model(
                self._target,
                self._model,
                compiled_program_directory=None if cache_dir is None else str(cache_dir),
            )
            self.load_timings_ms["set_model_duration_ms"] = _elapsed_ms(start)
        with self._translate_cgc_errors("tokenizer read-back"):
            self._tokenizer = self._configuration.tokenizer
        if self._tokenizer is None:
            message = "Windows ML created no tokenizer for this model after set_model."
            if self._model_format == "safetensors":
                message += " The checkpoint folder needs tokenizer.json and tokenizer_config.json."
            raise CGCLoadError(message)
        self._vocabulary_size = self._read_vocabulary_size()

    def load(
        self,
        *,
        context_length: int | None = None,
        eos_token_ids: Sequence[int] | None = None,
    ) -> None:
        """Set the logical capacity and create the native Task session.

        The tokenizer is opened first when :meth:`open_tokenizer` has not run.
        The GPU program compiles in ``create_session``, which is timed alone.

        Args:
            context_length: Logical capacity (prompt plus generated tokens).
                ``None`` keeps the Windows ML default, which can be the model's
                full trained context. The capacity is read back either way.
            eos_token_ids: Explicit EOS set for the Task. ``None`` keeps the
                tokenizer's EOS IDs; an empty sequence disables EOS stopping so
                generation runs to its token limit or the logical capacity.

        Raises:
            CGCNotInstalledError: The installed ``windowsml`` lacks part of the
                model-backed Task API.
            CGCLoadError: Windows ML rejected the capacity, the Task or the
                session for this model and GPU.
        """
        if self._task_session is not None:
            return
        self.open_tokenizer()
        configuration = self._configuration
        with self._translate_cgc_errors("capacity"):
            if context_length is not None:
                configuration.set_logical_capacity(int(context_length))
            self._context_length = _positive(configuration.logical_capacity)
        if self._context_length is None:
            requested = (
                ""
                if context_length is None
                else f" after set_logical_capacity({int(context_length)})"
            )
            logger.warning(
                "Windows ML did not report the logical capacity%s; context_length is reported "
                "as null and the prompt-fit check is skipped.",
                requested,
            )
        with self._translate_cgc_errors("Task creation"):
            if eos_token_ids is None:
                self._task = self._tasks.create_text_generation_task(self._tokenizer)
            else:
                self._task = self._tasks.create_text_generation_task(
                    self._tokenizer, tuple(int(token) for token in eos_token_ids)
                )
        with self._translate_cgc_errors("create_session"):
            start = time.perf_counter()
            self._task_session = self._task.create_session(configuration)
            self.load_timings_ms["session_create_duration_ms"] = _elapsed_ms(start)
        self._effective_device = "gpu"

    def close(self) -> None:
        """Release every Windows ML handle, then the GPU adapter (safe to call more than once).

        The program cache directory is kept.
        """
        for name in (
            "_task_session",
            "_task",
            "_configuration",
            "_tasks",
            "_tokenizer",
            "_model",
            "_target",
            "_runtime",
        ):
            _close_quietly(getattr(self, name))
            setattr(self, name, None)
        # The Runtime keeps no reference to the adapter: release it after the target.
        adapter, self._adapter = self._adapter, None
        _close_quietly(adapter)

    def __enter__(self) -> WinMLCGCSession:
        """Return the session for use as a context manager."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the session."""
        self.close()

    # -- prompting ------------------------------------------------------------
    def encode_prompt(self, prompt: str, *, apply_template: bool = True) -> list[int]:
        """Encode one user prompt, optionally through the model's chat template.

        The tokenizer is opened first when no tokenizer is open yet. Without a
        usable chat template the raw prompt is encoded instead and
        :attr:`chat_template_applied` is set to ``False``.

        Args:
            prompt: User prompt text.
            apply_template: Render the prompt as one user turn with the model's
                chat template; ``False`` encodes the text verbatim with special
                tokens.

        Returns:
            Prompt token IDs.
        """
        if self._tokenizer is None:
            self.open_tokenizer()
        with self._translate_cgc_errors("prompt encoding"):
            token_ids, self.chat_template_applied = _encode_user_prompt(
                self._wr,
                self._tokenizer,
                prompt,
                apply_template=apply_template,
                engine_label="CGC",
            )
        return token_ids

    # -- generation -----------------------------------------------------------
    def generate_timed(
        self,
        prompt_token_ids: Iterable[int],
        *,
        max_new_tokens: int,
        clock: Callable[[], float] = time.perf_counter,
    ) -> TaskGenerationTiming:
        """Read one native greedy generation and measure Task stream latency.

        The prompt must fit below the logical capacity and inside the tokenizer
        vocabulary when Windows ML reported them.

        Args:
            prompt_token_ids: Prompt token IDs, e.g. from :meth:`encode_prompt`.
            max_new_tokens: Upper bound on emitted tokens.
            clock: Monotonic clock in seconds (injectable for tests).

        Returns:
            Timing spans and the clock readings they come from, the emitted token
            IDs and count, text, stop reason, and the tokens left after the reset.

        Raises:
            ValueError: ``max_new_tokens`` is not positive.
            CGCSessionError: The session is not loaded, the prompt is empty or
                does not fit, or inference failed.
        """
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self._task_session is None:
            raise CGCSessionError("WinMLCGCSession.load() must be called before generation.")
        return _run_timed_generation(
            self._task_session,
            self._wt,
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            sequence_capacity=self.context_length,
            vocabulary_size=self.vocabulary_size,
            clock=clock,
            error_type=CGCSessionError,
            engine_label="CGC",
        )

    # -- internals --------------------------------------------------------------
    @property
    def _adapter_label(self) -> str:
        return f"{self._adapter_name} ({self._adapter_luid})"

    def _read_vocabulary_size(self) -> int | None:
        """Read the tokenizer vocabulary size; ``None`` with a warning when unreported.

        Runs outside the error translator: a missing read-back only turns off
        the prompt token range check.
        """
        wr = self._wr
        try:
            vocabulary_size = _positive(self._tokenizer.vocabulary_identity.token_count)
        except (AttributeError, wr.NotSupportedError, wr.WinMLError):
            vocabulary_size = None
        if vocabulary_size is None:
            logger.warning(
                "Windows ML did not report the tokenizer vocabulary size; vocabulary_size is "
                "reported as null and prompt token IDs are not range-checked."
            )
        return vocabulary_size

    @contextmanager
    def _translate_cgc_errors(self, phase: str) -> Iterator[None]:
        """Translate a failure inside one Windows ML ``phase`` into a session error.

        Session errors pass through. A ``WinMLError`` maps by HRESULT, and a
        ``FileNotFoundError`` or ``ImportError`` means part of ``windowsml`` is
        missing. Anything else, ``KeyboardInterrupt`` included, propagates
        unchanged so genuine bugs stay visible.
        """
        try:
            yield
        except CGCSessionError:
            raise
        except Exception as exc:
            hr = _hresult(exc)
            if hr is not None:
                raise self._hresult_error(phase, hr, exc) from exc
            if isinstance(exc, FileNotFoundError):
                raise _missing_native_library(exc) from exc
            if isinstance(exc, ImportError):
                raise CGCNotInstalledError(
                    f"Windows ML could not load a native dependency during {phase}: {exc}"
                ) from exc
            raise

    def _hresult_error(self, phase: str, hr: int, exc: Exception) -> CGCSessionError:
        """The session error for a Windows ML call in ``phase`` that failed with ``hr``."""
        hex_hr = f"0x{hr:08X}"
        if hr == _HRESULT_NO_INTERFACE:
            return CGCNotInstalledError(_needs_preview_message())
        if hr in _HRESULTS_NOT_FOUND:
            return CGCLoadError(
                f"Windows ML could not find a model file ({hex_hr}): {exc}. A checkpoint folder "
                "needs config.json, its *.safetensors weights (or model.safetensors.index.json "
                "and every shard it lists), tokenizer.json and tokenizer_config.json."
            )
        if hr == _HRESULT_NOT_SUPPORTED and phase in _MODEL_CALLS:
            message = (
                f"Windows ML CGC cannot run this model on {self._adapter_label} "
                f"({_MODEL_CALLS[phase]} returned {hex_hr}): the installed CGC backend does "
                "not support it, and winml-cli does not fall back to another engine."
            )
            if self._model_format == "gguf":
                message += " To run this .gguf file on llama.cpp, omit --backend cgc."
            return CGCLoadError(message)
        if hr == _HRESULT_NOT_SUPPORTED and phase == "create_session":
            return CGCLoadError(
                f"The selected GPU {self._adapter_label} or the requested options cannot run "
                f"this model with Windows ML CGC (create_session returned {hex_hr}). Select "
                "another GPU with --device-luid (see 'winml sys'), or lower --context-length "
                "or --max-new-tokens."
            )
        return CGCLoadError(f"CGC {phase} failed ({hex_hr}): {exc}")


__all__ = [
    "CGCLoadError",
    "CGCNotInstalledError",
    "CGCSessionError",
    "WinMLCGCSession",
    "cgc_package_versions",
]
