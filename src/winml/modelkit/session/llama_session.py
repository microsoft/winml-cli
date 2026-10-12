# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Windows ML Runtime session for GGUF language models (experimental).

:class:`WinMLLlamaSession` runs a ``.gguf`` decoder through the Windows ML
Runtime pipeline API (``windowsml.runtime``). ``Runtime.load_model`` selects the
llama.cpp backend from the ``.gguf`` artifact, so the same pipeline primitives
that run ONNX models drive one ``decoder`` stage whose sequence state (the KV
cache) is owned by the backend. The tokenizer and chat template come from the
GGUF metadata through ``windowsml.runtime.Tokenizer``.

llama.cpp support is experimental and optional. It ships only in preview Windows
ML packages that winml-cli does not depend on:

* a preview ``windowsml`` build that exposes ``windowsml.runtime``; and
* the ``windowsml-llama-core`` payload (llama.cpp adapter and CPU backend),
  plus a GPU backend such as ``windowsml-llama-cuda`` for ``device="gpu"``.

Nothing in this module imports ``windowsml`` at import time. The Runtime is
imported on first use, and each missing piece is reported as an actionable
:class:`LlamaNotInstalledError`.

Timing is captured at the Runtime call boundaries: one ``Pipeline.run`` over the
whole prompt is the *prefill*, reading the final logits row and selecting the
first token is the *first token*, and each later step (run one token, read
logits, select the next token) is one *decode* step. Detokenization is timed
separately so prefill and decode spans reflect model compute.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import logging
import struct
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np


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
    """Wall-clock spans (seconds) captured for one greedy generation.

    Attributes:
        input_tokens: Prompt token count submitted in the prefill run.
        generated_tokens: Tokens emitted, excluding a terminating EOS token.
        reset_s: Resetting the pipeline sequence state and the token decoder.
        prefill_s: The single ``Pipeline.run`` over the whole prompt.
        first_token_s: Reading the final logits row and selecting the first token.
        decode_s: One entry per emitted token after the first: run the previous
            token, read logits, select the next token. A step that selects EOS
            emits nothing and is not recorded.
        detokenization_s: Total time spent turning emitted tokens into text.
        response_text: Concatenated decoded text.
        stop_reason: ``"MAX_TOKENS"``, ``"EOS"``, or ``"CAPACITY"``.
    """

    input_tokens: int
    generated_tokens: int
    reset_s: float
    prefill_s: float
    first_token_s: float
    decode_s: list[float]
    detokenization_s: float
    response_text: str
    stop_reason: str


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
            f"installed in '{payload.lib_dir}'. Install 'windowsml-llama-cuda' for NVIDIA "
            "GPUs, or build another backend family (for example Vulkan) with "
            "Build-LlamaBackend.ps1 from the microsoft/WindowsML repository. Use "
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
                "Install 'windowsml-llama-cuda' for NVIDIA GPUs or build a backend with "
                f"Build-LlamaBackend.ps1, or use --device cpu. {_DOCS_HINT}"
            ) from exc
        if phase == "run":
            raise LlamaSessionError(f"GGUF inference failed ({hex_hr}): {exc}") from exc
        raise LlamaLoadError(f"GGUF {phase} failed ({hex_hr}): {exc}") from exc


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


# =============================================================================
# Session
# =============================================================================


class WinMLLlamaSession:
    """Greedy text generation over a GGUF model through the Windows ML Runtime.

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
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        runtime_api: Any = None,
    ) -> None:
        path = Path(model_path)
        if path.suffix.lower() != ".gguf":
            raise ValueError(f"WinMLLlamaSession requires a .gguf file, got '{model_path}'.")
        if device not in _LLAMA_DEVICES:
            raise ValueError(f"GGUF models support devices {_LLAMA_DEVICES}, got {device!r}.")
        self._model_path = path
        self._device = device
        self._wr: Any = runtime_api
        self._check_payload = runtime_api is None
        self.payload: LlamaPayload | None = None
        self.load_timings_ms: dict[str, float] = {}
        self.chat_template_applied: bool | None = None

        self._runtime: Any = None
        self._tokenizer: Any = None
        self._decoder: Any = None
        self._stage_target: Any = None
        self._token_target: Any = None
        self._model: Any = None
        self._stage: Any = None
        self._pipeline: Any = None
        self._token_data_type: Any = None
        self._token_rank = 0
        self._vocabulary_size: int | None = None
        self._end_tokens: frozenset[int] = frozenset()
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

    def load(self, *, context_length: int | None = None) -> None:
        """Load the GGUF model and build a one-stage decoder pipeline.

        Args:
            context_length: Sequence capacity hint (prompt plus generated tokens).
                ``None`` keeps the backend default, which can be the model's full
                training context and reserve a large KV cache.

        Raises:
            LlamaLoadError: Loading the model or building the pipeline failed.
        """
        if self._pipeline is not None:
            return
        self.open_tokenizer()
        wr, runtime = self._wr, self._runtime
        stage_target = model = builder = stage = pipeline = decoder = token_target = None
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
                pipeline = builder.build()
            build_end = time.perf_counter()
            _close_quietly(builder)
            builder = None

            self._token_data_type, self._token_rank = self._token_input_contract(wr, stage)
            self._sequence_capacity = self._probe_sequence_capacity(wr, stage)
            self._vocabulary_size, self._end_tokens = self._discover_vocabulary(stage)
            decoder = self._tokenizer.create_decoder()
            token_target = runtime.create_cpu_target()
        except Exception:
            for resource in (decoder, pipeline, stage, builder, model, token_target, stage_target):
                _close_quietly(resource)
            raise

        self._stage_target = stage_target
        self._model = model
        self._stage = stage
        self._pipeline = pipeline
        self._decoder = decoder
        self._token_target = token_target
        self._effective_device = self._resolved_device(stage) or self._device
        self.load_timings_ms["model_create_duration_ms"] = (build_start - model_start) * 1000.0
        self.load_timings_ms["pipeline_build_duration_ms"] = (build_end - build_start) * 1000.0

    def close(self) -> None:
        """Release native handles (best-effort; safe to call more than once)."""
        for name in (
            "_decoder",
            "_tokenizer",
            "_pipeline",
            "_stage",
            "_model",
            "_token_target",
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
    def generate_timed(
        self,
        prompt_token_ids: Iterable[int],
        *,
        max_new_tokens: int,
        clock: Callable[[], float] = time.perf_counter,
    ) -> LlamaGenerationTiming:
        """Run one greedy generation and capture its phase timings.

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
        if self._pipeline is None:
            raise LlamaSessionError("WinMLLlamaSession.load() must be called before generation.")
        token_ids = [int(token) for token in prompt_token_ids]
        if not token_ids:
            raise LlamaSessionError("the tokenizer produced an empty prompt")
        self._validate_token_ids(token_ids, "prompt")
        capacity = self._sequence_capacity
        if capacity and len(token_ids) >= capacity:
            raise LlamaSessionError(
                f"the prompt ({len(token_ids)} tokens) does not fit the model sequence "
                f"capacity ({capacity} tokens)"
            )

        reset_start = clock()
        self._pipeline.reset()
        self._decoder.reset()
        prefill_start = clock()
        self._run_tokens(token_ids)
        first_token_start = clock()
        token_id = self._next_token()
        first_token_end = clock()

        decode_s: list[float] = []
        fragments: list[str] = []
        detokenization_s = 0.0
        stop_reason = "MAX_TOKENS"
        if token_id in self._end_tokens:
            stop_reason = "EOS"
        else:
            while True:
                detokenize_start = clock()
                fragments.append(self._decoder.decode_token(token_id))
                detokenization_s += clock() - detokenize_start
                if len(fragments) >= max_new_tokens:
                    break
                if capacity and self._stage.sequence_position + 1 >= capacity:
                    stop_reason = "CAPACITY"
                    break
                step_start = clock()
                self._run_tokens((token_id,))
                token_id = self._next_token()
                step_end = clock()
                if token_id in self._end_tokens:
                    stop_reason = "EOS"
                    break
                decode_s.append(step_end - step_start)

        return LlamaGenerationTiming(
            input_tokens=len(token_ids),
            generated_tokens=len(fragments),
            reset_s=prefill_start - reset_start,
            prefill_s=first_token_start - prefill_start,
            first_token_s=first_token_end - first_token_start,
            decode_s=decode_s,
            detokenization_s=detokenization_s,
            response_text="".join(fragments),
            stop_reason=stop_reason,
        )

    # -- internals --------------------------------------------------------------
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

    def _discover_vocabulary(self, stage: Any) -> tuple[int | None, frozenset[int]]:
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
        return vocabulary_size, frozenset(eos_tokens)

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

    def _run_tokens(self, token_ids: Sequence[int]) -> None:
        """Bind token IDs as decoder input 0 and run the pipeline once."""
        values = tuple(token_ids)
        shape = (len(values),) if self._token_rank == 1 else (1, len(values))
        element = "i" if self._token_data_type == self._wr.TensorDataType.INT32 else "q"
        payload = struct.pack(f"<{len(values)}{element}", *values)
        with (
            _translate_native_errors("run", device=self._device),
            self._runtime.create_tensor(
                self._token_data_type,
                shape,
                payload,
                target=self._token_target,
            ) as token_tensor,
        ):
            self._stage.bind_input(0, token_tensor)
            self._pipeline.run()

    def _next_token(self) -> int:
        """Read the final-position logits row and return its argmax."""
        with (
            _translate_native_errors("run", device=self._device),
            self._stage.output(0) as output_tensor,
        ):
            _data_type, shape = output_tensor.desc()
            if not shape or shape[-1] <= 0:
                raise LlamaSessionError("GGUF logits must have a vocabulary dimension")
            if len(shape) == 1:
                logits = output_tensor.to_numpy()
            else:
                starts = [0] * len(shape)
                extents = [1] * len(shape)
                starts[-2] = shape[-2] - 1
                extents[-1] = shape[-1]
                with output_tensor.region(
                    starts,
                    extents,
                    target=self._token_target,
                ) as final_position:
                    logits = final_position.to_numpy()
        array = np.asarray(logits)
        return int(array.reshape(-1, array.shape[-1])[-1].argmax())


__all__ = [
    "LlamaGenerationTiming",
    "LlamaLoadError",
    "LlamaNotInstalledError",
    "LlamaPayload",
    "LlamaSessionError",
    "WinMLLlamaSession",
    "llama_package_versions",
]
