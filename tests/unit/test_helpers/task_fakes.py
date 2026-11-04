# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test doubles for the shared Windows ML Task stream core.

Greedy options in the 2.7 (eleven fields) and 2.8 (plus ``watermark``) shapes,
and a native Task session that returns one stream of token updates followed by
its terminal result. Nothing here imports ``windowsml`` or ``winml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class TaskOptions11:
    """``TextGenerationOptions`` as the 2.7 Task API declares it: eleven fields."""

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


@dataclass
class TaskOptions12(TaskOptions11):
    """``TextGenerationOptions`` as the 2.8 Task API declares it: watermark on by default."""

    watermark: bool = True


@dataclass
class FakeTaskResult:
    """Terminal Task result; a run to the token limit unless a test changes it."""

    token_ids: tuple[int, ...]
    prompt_token_count: int
    finish_reason: Any = field(default_factory=lambda: SimpleNamespace(name="MAX_TOKENS"))
    text: str = "ab"

    def raise_for_error(self) -> None:
        """Succeed: the fake result never carries a native error."""


class FakeTaskStream:
    """Task stream yielding one update per token ID, then the end of the stream.

    Args:
        token_ids: Token IDs the stream emits, also the result's token IDs.
        error: Raised by the first :meth:`read_next` call when given.
    """

    def __init__(
        self, token_ids: Sequence[int] = (7, 8), *, error: BaseException | None = None
    ) -> None:
        self.result = FakeTaskResult(tuple(token_ids), 0)
        self.closed = False
        self._updates = iter(tuple(token_ids))
        self._error = error

    def read_next(self) -> SimpleNamespace | None:
        """Return the next token update, or ``None`` at the end of the stream."""
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        token_id = next(self._updates, None)
        return None if token_id is None else SimpleNamespace(token_id=token_id)

    def close(self) -> None:
        """Mark the stream closed."""
        self.closed = True


class FakeTaskSession:
    """Native Task session that records its calls and returns one stream.

    It has no ``sequence_state``, so the tokens left after a reset read as unknown.

    Args:
        stream: Stream that :meth:`generate_tokens` returns (default: a new
            :class:`FakeTaskStream`).
        error: Raised by :meth:`generate_tokens` when given.
        calls: List that receives the name of each :meth:`reset` and
            :meth:`generate_tokens` call (default: a new list).
        closed: List that receives ``"session"`` on each :meth:`close` call, so
            a test can share it with other fakes to check the close order.
    """

    def __init__(
        self,
        *,
        stream: FakeTaskStream | None = None,
        error: BaseException | None = None,
        calls: list[str] | None = None,
        closed: list[str] | None = None,
    ) -> None:
        self.stream = FakeTaskStream() if stream is None else stream
        self.error = error
        self.calls: list[str] = [] if calls is None else calls
        self.options: Any = None
        self.token_ids: Sequence[int] | None = None
        self._closed = closed

    def reset(self) -> None:
        """Record the reset."""
        self.calls.append("reset")

    def generate_tokens(self, token_ids: Sequence[int], options: Any) -> FakeTaskStream:
        """Record the request and return the stream, or raise ``error``."""
        self.calls.append("generate_tokens")
        self.options = options
        self.token_ids = token_ids
        if self.error is not None:
            raise self.error
        self.stream.result.prompt_token_count = len(token_ids)
        return self.stream

    def close(self) -> None:
        """Record the close in the shared ``closed`` list, when one was given."""
        if self._closed is not None:
            self._closed.append("session")
