# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for capturing llama.cpp native log lines during GGUF load and warmup 0."""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from typing import Any

import pytest

from tests.unit.test_helpers import RUNTIME_CALLBACK, FakeLogApi
from winml.modelkit.commands._gguf_native_log import (
    GGML_LOG_LEVEL_CONT,
    NATIVE_VALUE_KEYS,
    NativeLogCapture,
    join_continuations,
    parse_native_values,
)


@pytest.fixture
def api() -> FakeLogApi:
    return FakeLogApi()


@pytest.fixture
def capture(api: FakeLogApi) -> NativeLogCapture:
    return NativeLogCapture(
        load_api=lambda: ("C:/lib/WinMLllama.dll", api),
        clock_ns=itertools.count(100, 10).__next__,
    )


def _raise(exc: Exception) -> Any:
    def loader() -> Any:
        raise exc

    return loader


def test_install_saves_runtime_callback_and_captures_lines(
    capture: NativeLogCapture, api: FakeLogApi
) -> None:
    capture.install()
    assert capture.installed
    assert capture.library == "C:/lib/WinMLllama.dll"
    assert api.callback != RUNTIME_CALLBACK
    head, tail = "llama_model_load: loading", [".", ".", " done\n"]
    api.emit(2, head.encode())
    for part in tail:
        api.emit(GGML_LOG_LEVEL_CONT, part.encode())
    api.emit(2, b"next line\n")
    assert [(line["level"], line["text"]) for line in capture.lines(None)] == [
        (2, head + "".join(tail).rstrip("\r\n")),
        (2, "next line"),
    ]


def test_cont_without_previous_line_starts_a_line() -> None:
    assert join_continuations([(5, "x\n", 1)]) == [(5, "x", 1)]


def test_lines_are_phased_at_the_warmup0_submit(
    capture: NativeLogCapture, api: FakeLogApi
) -> None:
    capture.install()
    for index in range(3):  # stamped 100, 110, 120
        api.emit(2, f"l{index}\n".encode())
    assert [line["phase"] for line in capture.lines(110)] == ["load", "warmup-0", "warmup-0"]
    assert {line["phase"] for line in capture.lines(None)} == {"load"}
    assert list(capture.lines(None)[0]) == ["level", "text", "qpc_ns", "phase"]


def test_restore_reinstates_and_verifies_runtime_callback(
    capture: NativeLogCapture, api: FakeLogApi
) -> None:
    capture.install()
    capture.restore()
    capture.restore()
    assert api.callback == RUNTIME_CALLBACK
    assert len(api.set_calls) == 2
    assert capture.to_dict(None)["after_capture"] == {
        "policy": "runtime_callback_restored",
        "restored": True,
        "verified_by_llama_log_get": True,
        "intact_until_restore": True,
    }


def test_restore_leaves_a_replaced_logger_alone(
    capture: NativeLogCapture, api: FakeLogApi
) -> None:  # Review Focus 5
    capture.install()
    api.callback = 0x9999
    calls = len(api.set_calls)
    capture.restore()
    assert api.callback == 0x9999
    assert len(api.set_calls) == calls
    after = capture.to_dict(None)["after_capture"]
    assert (after["restored"], after["intact_until_restore"]) == (False, False)


def test_callback_tolerates_invalid_utf8_and_null_text(
    capture: NativeLogCapture, api: FakeLogApi, monkeypatch: pytest.MonkeyPatch
) -> None:  # Review Focus 4
    raised: list[Any] = []
    monkeypatch.setattr(sys, "unraisablehook", raised.append)
    capture.install()
    api.emit(2, b"tok \xff\xfe\n")
    api.emit(2, None)
    assert [line["text"] for line in capture.lines(None)] == [
        b"tok \xff\xfe".decode("utf-8", "replace")
    ]
    assert raised == []


def test_ctrl_c_in_the_callback_is_kept_for_the_caller(
    api: FakeLogApi, monkeypatch: pytest.MonkeyPatch
) -> None:  # Review Focus 5
    raised: list[Any] = []
    monkeypatch.setattr(sys, "unraisablehook", raised.append)
    ticks = itertools.count(100, 10)

    def clock() -> int:
        tick = next(ticks)
        if tick == 110:  # Ctrl+C lands while the second line is being recorded
            raise KeyboardInterrupt
        return tick

    capture = NativeLogCapture(load_api=lambda: ("C:/lib/WinMLllama.dll", api), clock_ns=clock)
    capture.install()
    for index in range(3):
        api.emit(2, f"l{index}\n".encode())

    assert raised == []
    assert [line["text"] for line in capture.lines(None)] == ["l0", "l2"]
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
    doc = capture.to_dict(None)
    assert doc["capture"]["installed"] is False
    assert reason in doc["capture"]["reason"]
    assert doc["lines"] == []
    assert doc["after_capture"]["policy"] == "runtime_default_unobserved"
    assert doc["after_capture"]["restored"] is False


def test_default_loader_reports_an_unloaded_library() -> None:
    capture = NativeLogCapture()
    capture.install()
    assert capture.installed is False
    assert capture.reason == (
        "WinMLllama.dll is not loaded in this process after the tokenizer opened."
    )


def test_to_dict_fixed_fields_and_lines_hash(capture: NativeLogCapture, api: FakeLogApi) -> None:
    capture.install()
    api.emit(2, b"a\n")
    doc = capture.to_dict(None)
    assert doc["window"] == "load_through_warmup_0"
    assert doc["capture"] == {
        "api": "llama_log_set",
        "library": "C:/lib/WinMLllama.dll",
        "installed": True,
        "reason": None,
    }
    assert doc["outside_capture"] == (
        "Backend registration and device lines are written to process stderr before the "
        "Runtime installs its logger."
    )
    assert doc["lines_sha256"] == hashlib.sha256(
        json.dumps(doc["lines"], separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def test_parse_native_values_reads_contract_keys_first_match_wins() -> None:
    values = {"n_ctx": 4096, "n_batch": 4096, "n_ubatch": 512, "flash_attn": "auto"}
    texts = [f"llama_context: {'n_ctx_seq':<21} = 1"]
    texts += [f"llama_context: {key:<21} = {value}" for key, value in values.items()]
    texts += [
        "load_tensors: offloaded 27/27 layers to GPU",
        "llama_kv_cache: size =   16.00 MiB (  4096 cells,   4 layers,  1/1 seqs), "
        "K (f16):    8.00 MiB, V (q8_0):    8.00 MiB",
        "llama_kv_cache: size =   22.00 MiB (  1024 cells,  22 layers,  1/1 seqs), "
        "K (q4_0):   11.00 MiB, V (q4_0):   11.00 MiB",
    ]
    parsed = parse_native_values([{"text": text} for text in texts])
    expected = {
        key: {"value": value, "line": index + 1}
        for index, (key, value) in enumerate(values.items())
    }
    expected.update(
        {
            "n_layer_offloaded": {"value": 27, "line": 5},
            "n_layer_total": {"value": 27, "line": 5},
            "type_k": {"value": "f16", "line": 6},
            "type_v": {"value": "q8_0", "line": 6},
        }
    )
    assert parsed == expected
    assert set(parsed) == set(NATIVE_VALUE_KEYS)
