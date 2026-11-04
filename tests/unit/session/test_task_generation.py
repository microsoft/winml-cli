# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the Task stream core shared by the Windows ML Task engines.

The core imports only the standard library, so fake options, streams and
sessions stand in for ``windowsml.tasks``. The llama.cpp binding's behavior is
pinned through the session in ``test_llama_session.py``.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import itertools
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from tests.unit.test_helpers import FakeTaskSession, FakeTaskStream, TaskOptions11, TaskOptions12
from winml.modelkit.session import (
    LlamaGenerationTiming,
    LlamaSessionError,
    TaskGenerationTiming,
    TaskSessionError,
)
from winml.modelkit.session.task_generation import (
    _accepts_keyword,
    _greedy_task_options,
    _run_timed_generation,
    _task_stop_reason,
)


_LOGGER = "winml.modelkit.session.task_generation"


class _EngineError(TaskSessionError):
    """Error type an engine binding hands to the core."""


class _HResultError(Exception):
    hresult = 0x80004005


def _unreadable_factory() -> Any:
    def factory(**_kwargs: Any) -> None:
        """Accept any keyword; its signature cannot be read."""

    factory.__signature__ = "not a signature"  # inspect.signature raises TypeError
    return factory


def _generate(session: FakeTaskSession, *, max_new_tokens: int = 2) -> TaskGenerationTiming:
    return _run_timed_generation(
        session,
        SimpleNamespace(TextGenerationOptions=TaskOptions12),
        [1, 2],
        max_new_tokens=max_new_tokens,
        sequence_capacity=None,
        vocabulary_size=None,
        clock=itertools.count(step=1.0).__next__,
        error_type=_EngineError,
        engine_label="CGC",
    )


@pytest.mark.parametrize(
    ("factory", "declared"),
    [
        (TaskOptions12, True),
        (lambda *, watermark=True: None, True),
        (TaskOptions11, False),
        (lambda **kwargs: None, False),
        (lambda watermark=True, /: None, False),
        (_unreadable_factory(), False),
    ],
    ids=[
        "2.8-options",
        "keyword-only",
        "2.7-options",
        "var-keyword",
        "positional-only",
        "unreadable-signature",
    ],
)
def test_accepts_keyword_reads_only_declared_parameters(factory: Any, declared: bool) -> None:
    assert _accepts_keyword(factory, "watermark") is declared


@pytest.mark.parametrize(
    "factory",
    # A partial whose bound arguments do not fit its function raises ValueError.
    [_unreadable_factory(), functools.partial(TaskOptions12, unknown=1)],
    ids=["type-error", "value-error"],
)
def test_accepts_keyword_warns_when_the_signature_is_unreadable(
    factory: Any, caplog: pytest.LogCaptureFixture
) -> None:
    with pytest.raises((TypeError, ValueError)) as unreadable:
        inspect.signature(factory)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert _accepts_keyword(factory, "watermark") is False

    (record,) = [record for record in caplog.records if record.name == _LOGGER]
    assert record.levelno == logging.WARNING
    assert "'watermark'" in record.getMessage()
    assert str(unreadable.value) in record.getMessage()


def test_greedy_options_disable_watermark_only_when_declared() -> None:
    eleven = _greedy_task_options(SimpleNamespace(TextGenerationOptions=TaskOptions11), 8)
    twelve = _greedy_task_options(SimpleNamespace(TextGenerationOptions=TaskOptions12), 8)

    assert dataclasses.asdict(eleven) == {
        "max_new_tokens": 8,
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
    assert twelve.watermark is False
    assert dataclasses.asdict(twelve) == {**dataclasses.asdict(eleven), "watermark": False}


def test_generation_streams_greedy_options_and_maps_the_stop_reason() -> None:
    stream = FakeTaskStream()
    session = FakeTaskSession(stream=stream)

    timing = _generate(session)

    assert session.calls == ["reset", "generate_tokens"]
    assert session.options.watermark is False
    assert (timing.generated_tokens, timing.stop_reason, timing.token_ids) == (
        2,
        "MAX_TOKENS",
        (7, 8),
    )
    assert timing.cached_tokens is None
    assert stream.closed


def test_generation_rechecks_max_new_tokens() -> None:
    session = FakeTaskSession()
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        _generate(session, max_new_tokens=0)
    assert session.calls == []


@pytest.mark.parametrize("failing_call", ["generate_tokens", "read_next"])
def test_run_failures_raise_the_engine_error_with_its_label(failing_call: str) -> None:
    error = _HResultError()
    stream = FakeTaskStream(error=error if failing_call == "read_next" else None)
    session = FakeTaskSession(
        stream=stream, error=error if failing_call == "generate_tokens" else None
    )

    with pytest.raises(_EngineError, match=r"^CGC inference failed \(0x80004005\)") as excinfo:
        _generate(session)

    assert excinfo.value.__cause__ is error
    # A stream the Task handed out is closed even when reading it fails.
    assert stream.closed is (failing_call == "read_next")


def test_errors_without_an_hresult_propagate_unchanged() -> None:
    error = RuntimeError("not a native failure")
    with pytest.raises(RuntimeError) as excinfo:
        _generate(FakeTaskSession(error=error))
    assert excinfo.value is error


def test_unsuccessful_finish_reasons_raise_the_engine_error() -> None:
    stream = FakeTaskStream()
    stream.result.finish_reason = SimpleNamespace(name="ERROR")
    with pytest.raises(_EngineError, match="ERROR"):
        _generate(FakeTaskSession(stream=stream))
    assert stream.closed


def test_sequence_capacity_is_a_normal_stop_that_returns_capacity() -> None:
    # Returned, never raised, exactly as the llama.cpp session always mapped it.
    reason = SimpleNamespace(name="SEQUENCE_CAPACITY")
    assert _task_stop_reason(reason, error_type=_EngineError) == "CAPACITY"


def test_llama_names_alias_the_shared_types() -> None:
    assert LlamaGenerationTiming is TaskGenerationTiming
    assert issubclass(LlamaSessionError, TaskSessionError)
