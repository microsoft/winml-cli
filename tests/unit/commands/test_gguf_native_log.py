# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for capturing llama.cpp native log fragments during GGUF load and warmup 0."""

from __future__ import annotations

import itertools
import sys
from typing import Any

import pytest

from tests.unit.test_helpers import RUNTIME_CALLBACK, FakeLogApi
from winml.modelkit.commands._gguf_native_log import NativeLogCapture, loaded_module_handle


# ggml_log_level of a fragment that continues the previous one (GGML_LOG_LEVEL_CONT).
_CONT = 5


@pytest.fixture
def api() -> FakeLogApi:
    return FakeLogApi()


@pytest.fixture
def capture(api: FakeLogApi) -> NativeLogCapture:
    return NativeLogCapture(load_api=lambda: api, clock_ns=itertools.count(100, 10).__next__)


def _raise(exc: Exception) -> Any:
    def loader() -> Any:
        raise exc

    return loader


def _texts(capture: NativeLogCapture) -> list[str]:
    return [fragment["text"] for fragment in capture.to_dict()["fragments"]]


def test_install_swaps_the_callback_and_keeps_each_fragment_with_its_stamp(
    capture: NativeLogCapture, api: FakeLogApi
) -> None:
    capture.install()
    assert capture.installed
    assert api.callback != RUNTIME_CALLBACK
    # A progress line arrives as a head fragment followed by continuation fragments.
    emitted = [
        (2, "llama_model_load: loading"),
        (_CONT, "."),
        (_CONT, "."),
        (_CONT, " done\n"),
        (2, "next line\n"),
    ]
    for level, text in emitted:
        api.emit(level, text.encode())
    # Nothing is joined: each call keeps its level, its text as emitted and its own stamp.
    assert capture.to_dict()["fragments"] == [
        {"level": level, "text": text, "qpc_ns": 100 + 10 * index}
        for index, (level, text) in enumerate(emitted)
    ]


def test_restore_reinstates_and_verifies_runtime_callback(
    capture: NativeLogCapture, api: FakeLogApi
) -> None:
    capture.install()
    capture.restore()
    capture.restore()
    assert api.callback == RUNTIME_CALLBACK
    assert len(api.set_calls) == 2
    doc = capture.to_dict()
    assert (doc["intact"], doc["restored"]) == (True, True)


def test_restore_leaves_a_replaced_logger_alone(
    capture: NativeLogCapture, api: FakeLogApi
) -> None:
    capture.install()
    api.callback = 0x9999
    calls = len(api.set_calls)
    capture.restore()
    assert api.callback == 0x9999
    assert len(api.set_calls) == calls
    doc = capture.to_dict()
    assert (doc["intact"], doc["restored"]) == (False, False)


def test_a_failed_read_during_restore_leaves_intact_unknown(
    capture: NativeLogCapture, api: FakeLogApi
) -> None:
    capture.install()

    def get() -> tuple[int | None, int | None]:
        raise OSError("llama_log_get failed")

    api.get = get  # type: ignore[method-assign]
    calls = len(api.set_calls)
    capture.restore()  # the error is suppressed: the capture never fails the run
    assert len(api.set_calls) == calls
    doc = capture.to_dict()
    assert (doc["intact"], doc["restored"]) == (None, False)


def test_callback_tolerates_invalid_utf8_and_null_text(
    capture: NativeLogCapture, api: FakeLogApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    raised: list[Any] = []
    monkeypatch.setattr(sys, "unraisablehook", raised.append)
    capture.install()
    api.emit(2, b"tok \xff\xfe\n")
    api.emit(2, None)
    assert _texts(capture) == [b"tok \xff\xfe\n".decode("utf-8", "replace")]
    assert raised == []


def test_reason_explains_a_capture_that_was_never_installed(capture: NativeLogCapture) -> None:
    assert capture.to_dict()["reason"] == (
        "The run ended before the native log capture was installed."
    )
    capture.install()
    assert capture.to_dict()["reason"] is None


@pytest.mark.parametrize("failure", [KeyboardInterrupt, OSError])
def test_a_broken_swap_puts_the_runtime_callback_back(
    api: FakeLogApi, failure: type[BaseException]
) -> None:
    reads: list[int] = []
    read = api.get

    def get() -> tuple[int | None, int | None]:
        reads.append(1)
        if len(reads) == 2:  # the check right after llama_log_set
            raise failure
        return read()

    api.get = get  # type: ignore[method-assign]
    capture = NativeLogCapture(load_api=lambda: api)
    if failure is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            capture.install()
    else:
        capture.install()

    assert api.callback == RUNTIME_CALLBACK
    assert capture.installed is False
    assert capture.to_dict()["reason"]


def test_ctrl_c_in_the_callback_is_kept_for_the_caller(
    api: FakeLogApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    raised: list[Any] = []
    monkeypatch.setattr(sys, "unraisablehook", raised.append)
    ticks = itertools.count(100, 10)

    def clock() -> int:
        tick = next(ticks)
        if tick == 110:  # Ctrl+C lands while the second fragment is being recorded
            raise KeyboardInterrupt
        return tick

    capture = NativeLogCapture(load_api=lambda: api, clock_ns=clock)
    capture.install()
    for index in range(3):
        api.emit(2, f"l{index}\n".encode())

    assert raised == []
    assert _texts(capture) == ["l0\n", "l2\n"]
    with pytest.raises(KeyboardInterrupt):
        capture.raise_if_interrupted()
    capture.raise_if_interrupted()  # each Ctrl+C is raised once


@pytest.mark.parametrize(
    ("loader", "reason"),
    [
        (lambda: None, "not loaded"),
        (_raise(AttributeError("function 'llama_log_get' not found")), "llama_log_get"),
    ],
)
def test_unavailable_log_api_is_reported(loader: Any, reason: str) -> None:
    capture = NativeLogCapture(load_api=loader)
    capture.install()
    capture.restore()
    doc = capture.to_dict()
    assert doc["installed"] is False
    assert reason in doc["reason"]
    assert (doc["intact"], doc["restored"], doc["fragments"]) == (None, False, [])


def test_default_loader_reports_an_unloaded_library() -> None:
    capture = NativeLogCapture()
    capture.install()
    assert capture.installed is False
    assert capture.reason == (
        "WinMLllama.dll is not loaded in this process after the tokenizer opened."
    )


def test_loaded_module_handle_reads_the_live_process() -> None:
    handle = loaded_module_handle("kernel32.dll")
    assert handle is not None
    assert handle[1].lower().endswith("kernel32.dll")
    assert loaded_module_handle("not-a-loaded-module.dll") is None


def test_to_dict_holds_the_capture_state_and_the_raw_fragments(
    capture: NativeLogCapture, api: FakeLogApi
) -> None:
    capture.install()
    api.emit(2, b"a\n")
    capture.restore()
    assert capture.to_dict() == {
        "installed": True,
        "reason": None,
        "intact": True,
        "restored": True,
        "fragments": [{"level": 2, "text": "a\n", "qpc_ns": 100}],
    }
