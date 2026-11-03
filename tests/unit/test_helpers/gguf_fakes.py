# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test doubles for GGUF evidence.

A llama log API that calls back through ctypes, Task options, a native error
carrying an HRESULT, and ``generate_timed`` results read from a test clock.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import TYPE_CHECKING

from winml.modelkit.commands._gguf_native_log import LOG_CALLBACK
from winml.modelkit.session import LlamaGenerationTiming


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


# Stand-in address of the Runtime adapter's own llama log callback.
RUNTIME_CALLBACK = 0x7920


class HResultError(OSError):
    """A native failure that carries an HRESULT, like the projection's WinMLError."""

    def __init__(self, hresult: int, message: str = "native failure") -> None:
        super().__init__(message)
        self.hresult = hresult


@dataclasses.dataclass(frozen=True)
class FakeOptions:
    """``TextGenerationOptions`` with the projection's defaults."""

    max_new_tokens: int = 256
    stop_token_ids: tuple[int, ...] = ()
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    repetition_penalty_window: int = 0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int = 0


def make_timing(
    clock: Callable[[], float],
    *,
    tokens: Iterable[int] = (10, 11, 12),
    input_tokens: int = 4,
    cached_tokens: int | None = 0,
) -> LlamaGenerationTiming:
    """A request that ran to its token limit, read from ``clock`` like ``generate_timed``.

    ``clock`` is read in order at the reset start, the reset end, the submit,
    the stream creation, once per token update, the end of the stream and the
    checked result; the spans are differences of those readings.
    """
    token_ids = tuple(tokens)
    reset_start, reset_end, submit, created = clock(), clock(), clock(), clock()
    token_times = tuple(clock() for _ in token_ids)
    completed, fetched = clock(), clock()
    return LlamaGenerationTiming(
        input_tokens=input_tokens,
        generated_tokens=len(token_ids),
        reset_s=reset_end - reset_start,
        stream_create_s=created - submit,
        first_token_s=token_times[0] - submit if token_times else None,
        decode_s=[later - earlier for earlier, later in itertools.pairwise(token_times)],
        completion_s=completed - (token_times[-1] if token_times else created),
        result_fetch_s=fetched - completed,
        generation_s=fetched - submit,
        response_text="".join(f"<{token}>" for token in token_ids),
        stop_reason="MAX_TOKENS",
        submit_time=submit,
        token_times=token_times,
        completed_time=completed,
        token_ids=token_ids,
        cached_tokens=cached_tokens,
    )


class FakeLogApi:
    """In-memory ``llama_log_get``/``llama_log_set`` pair.

    ``emit`` invokes the installed callback through a real ctypes function
    pointer, exactly as native code would.
    """

    def __init__(self) -> None:
        self.callback: int | None = RUNTIME_CALLBACK
        self.user_data: int | None = None
        self.set_calls: list[tuple[int | None, int | None]] = []

    def get(self) -> tuple[int | None, int | None]:
        return self.callback, self.user_data

    def set(self, callback: int | None, user_data: int | None) -> None:
        self.set_calls.append((callback, user_data))
        self.callback = callback
        self.user_data = user_data

    def emit(self, level: int, text: bytes | None) -> None:
        assert self.callback not in (None, RUNTIME_CALLBACK), "no capture callback is installed"
        LOG_CALLBACK(self.callback)(level, text, None)
