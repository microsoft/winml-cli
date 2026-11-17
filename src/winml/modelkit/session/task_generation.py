# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Task stream timing core shared by the Windows ML Task text generation engines.

Every engine binding (llama.cpp through a Runtime pipeline, or CGC through a
model-backed configuration) ends with a native ``windowsml.tasks`` text
generation session. This module drives that session the same way for all of
them: it encodes the prompt, builds greedy options, runs one timed request and
checks its terminal result, raising the binding's own error type.

The module imports only the standard library. Callers pass the imported
``windowsml`` modules and the native handles, so importing it never loads
Windows ML.

Timing is captured at Task stream boundaries, including native generation,
sampling and text decoding. First-token latency includes stream creation; later
tokens use update arrival intervals. Terminal reads and result retrieval are
included in total generation time.
"""

from __future__ import annotations

import gc
import importlib.metadata
import inspect
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

logger = logging.getLogger(__name__)


# =============================================================================
# Errors
# =============================================================================


class TaskSessionError(RuntimeError):
    """A Windows ML Task generation session could not complete the requested operation."""


# =============================================================================
# Data classes
# =============================================================================


@dataclass(frozen=True)
class TaskGenerationTiming:
    """Task stream wall-clock spans (seconds) for one greedy generation.

    The spans are differences of ``clock`` readings. ``submit_time``,
    ``token_times`` and ``completed_time`` are the readings themselves; with the
    default ``time.perf_counter`` they are machine-wide QPC times that another
    process can align with its own samples. The session state is read after the
    reset and before ``submit_time``, outside every span.

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
        submit_time: Clock reading immediately before ``generate_tokens``.
        token_times: Clock reading as each token update returned from ``read_next()``.
        completed_time: Clock reading when ``read_next()`` returned the end of the stream.
        token_ids: Token ID of each update, in stream order.
        cached_tokens: Tokens the Task session still held right after the reset;
            ``0`` means the prompt was evaluated from an empty session, ``None``
            that the session could not report its state.
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
    submit_time: float = 0.0
    token_times: tuple[float, ...] = ()
    completed_time: float = 0.0
    token_ids: tuple[int, ...] = ()
    cached_tokens: int | None = None


# =============================================================================
# Packages, native errors and handles
# =============================================================================


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _hresult(exc: BaseException) -> int | None:
    """Return a ``WinMLError``'s normalized 32-bit HRESULT, or ``None``."""
    hr = getattr(exc, "hresult", None)
    if isinstance(hr, int):
        return hr & 0xFFFFFFFF
    return None


def _runtime_errors(runtime_api: Any) -> tuple[type[BaseException], ...]:
    """Error types the Runtime raises, or ``RuntimeError`` when it exposes none."""
    errors = tuple(
        error
        for error in (
            getattr(runtime_api, "NotSupportedError", None),
            getattr(runtime_api, "WinMLError", None),
        )
        if isinstance(error, type) and issubclass(error, BaseException)
    )
    return errors or (RuntimeError,)


def _cached_tokens_after_reset(session: Any) -> int | None:
    """Count the tokens a Task session still holds right after a reset.

    The largest of the evaluated count, the retained count and the retained IDs,
    so any token left behind reads above ``0``. ``None`` when the projection or
    the native session cannot report its state; generation still proceeds.
    """
    try:
        sequence = session.sequence_state
        return max(
            int(sequence.evaluated_token_count),
            int(sequence.retained_token_count),
            len(session.retained_token_ids),
        )
    except Exception:  # the count is evidence only, never a reason to fail the request
        return None


def _close_quietly(resource: Any) -> None:
    close = getattr(resource, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:  # teardown is best-effort
        logger.debug("Failed to close %s", type(resource).__name__, exc_info=True)


def _release_closed_interfaces() -> None:
    """Release the Windows ML interfaces that closed wrappers left in reference cycles.

    ``windowsml`` wraps each COM interface with ``ctypes.cast``, which records the
    pointer in its own ``_objects``, so closing a wrapper releases the interface
    only when the cycle collector next runs. Left to interpreter exit, comtypes no
    longer releases it, and Windows ML tears the native objects down while its
    DLLs unload, which can crash the process in the GPU driver (0xC0000409) after
    the run has finished.
    """
    gc.collect()


# =============================================================================
# Prompting
# =============================================================================


def _format_user_turn(runtime_api: Any, tokenizer: Any, prompt: str) -> list[int]:
    """Render one user turn with the model's chat template and return its token IDs."""
    request = runtime_api.ChatCompletionRequest(
        messages=(
            runtime_api.ChatCompletionMessage(role=runtime_api.ChatRole.USER, content=prompt),
        ),
    )
    with tokenizer.format_conversation(request) as formatted:
        return [int(token) for token in formatted.token_ids]


def _encode_user_prompt(
    runtime_api: Any,
    tokenizer: Any,
    prompt: str,
    *,
    apply_template: bool,
    engine_label: str,
) -> tuple[list[int], bool]:
    """Encode one user prompt, optionally through the model's chat template.

    Without a usable chat template the raw prompt is encoded instead.

    Args:
        runtime_api: Imported ``windowsml.runtime`` module (conversation types).
        tokenizer: Runtime tokenizer of the model.
        prompt: User prompt text.
        apply_template: Render the prompt as one user turn with the chat template;
            ``False`` encodes the text verbatim with special tokens.
        engine_label: Engine name in the log message, e.g. ``"GGUF"``.

    Returns:
        The prompt token IDs, and whether the chat template rendered them.
    """
    if apply_template:
        # AttributeError covers Runtime builds without the conversation types.
        template_errors: tuple[type[BaseException], ...] = (
            AttributeError,
            *_runtime_errors(runtime_api),
        )
        try:
            token_ids = _format_user_turn(runtime_api, tokenizer, prompt)
        except template_errors as exc:
            logger.info(
                "%s chat template unavailable (%s); encoding the raw prompt", engine_label, exc
            )
        else:
            return token_ids, True
    return [int(token) for token in tokenizer.encode(prompt, add_special_tokens=True)], False


# =============================================================================
# Greedy options
# =============================================================================


def _accepts_keyword(factory: Any, name: str) -> bool:
    """Whether ``factory`` declares ``name`` as a parameter a keyword can pass.

    Only a positional-or-keyword or keyword-only parameter counts; ``**kwargs``
    does not. A factory whose signature cannot be read counts as not declaring
    it, and a warning names the parameter that keeps its default.
    """
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError) as exc:
        logger.warning(
            "Cannot read the parameters of %r (%s); the %r option keeps its default",
            factory,
            exc,
            name,
        )
        return False
    parameter = parameters.get(name)
    return parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _greedy_task_options(tasks_api: Any, max_new_tokens: int) -> Any:
    """Greedy Task options with every field explicit (neutral filters, no stops).

    ``watermark`` is turned off when ``TextGenerationOptions`` declares it, as the
    2.8 Task API does; an options type without it gets the other eleven fields.
    """
    options_type = tasks_api.TextGenerationOptions
    values: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "stop_token_ids": (),
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "repetition_penalty_window": 0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "seed": 0,
    }
    if _accepts_keyword(options_type, "watermark"):
        values["watermark"] = False
    return options_type(**values)


# =============================================================================
# Results and validation
# =============================================================================


def _task_stop_reason(reason: Any, *, error_type: type[TaskSessionError]) -> str:
    """Normalize a Task finish reason; an unsuccessful one raises ``error_type``."""
    name = str(getattr(reason, "name", ""))
    normalized = {
        "EOS_TOKEN": "EOS",
        "STOP_TOKEN": "STOP_TOKEN",
        "MAX_TOKENS": "MAX_TOKENS",
        "SEQUENCE_CAPACITY": "CAPACITY",
        "CANCELED": "CANCELED",
    }
    if name not in normalized:
        raise error_type(f"The Task ended with an unsuccessful finish reason: {name}.")
    return normalized[name]


def _validate_token_ids(
    token_ids: Sequence[int],
    label: str,
    *,
    vocabulary_size: int | None,
    error_type: type[TaskSessionError],
) -> None:
    """Reject negative token IDs and, when the vocabulary size is known, IDs beyond it."""
    for token_id in token_ids:
        if token_id < 0 or (vocabulary_size is not None and token_id >= vocabulary_size):
            raise error_type(
                f"{label} token {token_id} is outside the model vocabulary of "
                f"{vocabulary_size if vocabulary_size is not None else 'unknown'} entries"
            )


@contextmanager
def _translate_run_errors(
    *, error_type: type[TaskSessionError], engine_label: str
) -> Iterator[None]:
    """Translate a native ``WinMLError`` raised during generation into ``error_type``.

    Session errors pass through, and exceptions without an HRESULT are re-raised
    unchanged so genuine bugs stay visible.
    """
    try:
        yield
    except TaskSessionError:
        raise
    except Exception as exc:
        hr = _hresult(exc)
        if hr is None:
            raise
        raise error_type(f"{engine_label} inference failed (0x{hr:08X}): {exc}") from exc


# =============================================================================
# Timed generation
# =============================================================================


def _run_timed_generation(
    task_session: Any,
    tasks_api: Any,
    prompt_token_ids: Iterable[int],
    *,
    max_new_tokens: int,
    sequence_capacity: int | None,
    vocabulary_size: int | None,
    clock: Callable[[], float],
    error_type: type[TaskSessionError],
    engine_label: str,
) -> TaskGenerationTiming:
    """Read one native greedy generation and measure Task stream latency.

    The session is reset first and its state is read after the reset, outside
    the timed spans. ``submit_time`` is read immediately before
    ``generate_tokens`` and one reading follows each ``read_next()`` return.

    Args:
        task_session: Native Task generation session.
        tasks_api: Imported ``windowsml.tasks`` module (greedy options).
        prompt_token_ids: Prompt token IDs.
        max_new_tokens: Upper bound on emitted tokens.
        sequence_capacity: Model sequence capacity the prompt must fit below;
            ``None`` (or ``0``) skips the check.
        vocabulary_size: Vocabulary size the prompt IDs must fit, or ``None``
            when unknown (only negative IDs are rejected).
        clock: Monotonic clock in seconds.
        error_type: Engine error type for every failure the core reports.
        engine_label: Engine name in native inference failures, e.g. ``"GGUF"``.

    Returns:
        Timing spans and the clock readings they come from, the emitted token
        IDs and count, text, stop reason, and the tokens left after the reset.

    Raises:
        ValueError: ``max_new_tokens`` is not positive.
        TaskSessionError: ``error_type`` when the prompt is empty, outside the
            vocabulary or too long for the capacity, the Task result disagrees
            with its stream, or inference failed.
    """
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    token_ids = [int(token) for token in prompt_token_ids]
    if not token_ids:
        raise error_type("the tokenizer produced an empty prompt")
    _validate_token_ids(token_ids, "prompt", vocabulary_size=vocabulary_size, error_type=error_type)
    if sequence_capacity and len(token_ids) >= sequence_capacity:
        raise error_type(
            f"the prompt ({len(token_ids)} tokens) does not fit the model sequence "
            f"capacity ({sequence_capacity} tokens)"
        )

    options = _greedy_task_options(tasks_api, max_new_tokens)
    with _translate_run_errors(error_type=error_type, engine_label=engine_label):
        reset_start = clock()
        task_session.reset()
        reset_end = clock()
        cached_tokens = _cached_tokens_after_reset(task_session)
        generation_start = clock()
        stream = task_session.generate_tokens(token_ids, options)
        try:
            create_end = clock()
            previous_arrival = create_end
            first_token_s: float | None = None
            emitted_ids: list[int] = []
            token_times: list[float] = []
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
                token_times.append(arrival)
                previous_arrival = arrival
            result = stream.result
            result.raise_for_error()
            result_end = clock()
            stop_reason = _task_stop_reason(result.finish_reason, error_type=error_type)
            result_ids = tuple(int(token) for token in result.token_ids)
            if result_ids != tuple(emitted_ids):
                raise error_type("The Task result token IDs disagree with its stream updates.")
            if int(result.prompt_token_count) != len(token_ids):
                raise error_type(
                    "The Task result prompt count disagrees with the submitted prompt."
                )
            if len(result_ids) > max_new_tokens:
                raise error_type("The Task result exceeds the requested generated token limit.")
        finally:
            _close_quietly(stream)

    return TaskGenerationTiming(
        input_tokens=int(result.prompt_token_count),
        generated_tokens=len(result_ids),
        reset_s=reset_end - reset_start,
        stream_create_s=create_end - generation_start,
        first_token_s=first_token_s,
        decode_s=decode_s,
        completion_s=completion_s,
        result_fetch_s=result_end - arrival,
        generation_s=result_end - generation_start,
        response_text=str(result.text),
        stop_reason=stop_reason,
        submit_time=generation_start,
        token_times=tuple(token_times),
        completed_time=arrival,
        token_ids=tuple(emitted_ids),
        cached_tokens=cached_tokens,
    )


__all__ = ["TaskGenerationTiming", "TaskSessionError"]
