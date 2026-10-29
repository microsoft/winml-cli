# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test doubles for GGUF evidence.

A llama log API that calls back through ctypes, Task options, a native error
carrying an HRESULT, and request records with generated time points.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any

from winml.modelkit.commands._gguf_native_log import LOG_CALLBACK
from winml.modelkit.session import LlamaRequestRecord


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


def make_record(
    *,
    input_ids: tuple[int, ...] | list[int] = (1, 2, 3, 4),
    tokens: Any = (10, 11, 12),
    start_ns: int = 1_000,
    **overrides: Any,
) -> LlamaRequestRecord:
    """A successful request record that ran to its token limit.

    Time points accumulate from ``start_ns``: reset start, reset end (+20),
    submit (+5), stream created (+30), one event per token (+50 each),
    completion (+70) and result fetch (+110). ``overrides`` replace fields.
    """
    tokens = tuple(tokens)
    count = len(tokens)
    times = list(itertools.accumulate([start_ns, 20, 5, 30, *[50] * count, 70, 110]))
    record = LlamaRequestRecord(
        input_ids=tuple(input_ids),
        reset_start_ns=times[0],
        reset_end_ns=times[1],
        reset_disclosure={
            "state": "EMPTY",
            "sequence_state": {
                "disclosure": "NONE",
                "session_state": "EMPTY",
                "evaluated_token_count": 0,
                "retained_token_count": 0,
            },
            "retained_token_ids": [],
        },
        submit_ns=times[2],
        stream_created_ns=times[3],
        events=tuple(zip(times[4 : 4 + count], tokens, strict=True)),
        completed_ns=times[4 + count],
        result_fetched_ns=times[5 + count],
        result_token_ids=tokens,
        prompt_token_count=len(input_ids),
        finish_reason="MAX_TOKENS",
        error_code=0,
        speculative_statistics={
            "method": "NONE",
            "verification_step_count": 0,
            "draft_token_count": 0,
            "accepted_draft_token_count": 0,
        },
        response_text="".join(f"<{token}>" for token in tokens),
    )
    return dataclasses.replace(record, **overrides)


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
