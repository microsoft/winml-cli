# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for CGC text generation through the model-backed Windows ML Task API.

Fake ``windowsml.runtime`` and ``windowsml.tasks`` modules stand in for the
2.8.183.dev0 preview, and a fake GPU handle replaces the DXCore adapter, so no
Windows ML or DXCore code runs. The Task stream fakes are the ones the shared
core tests use.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import itertools
import logging
import sys
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.unit.test_helpers import FakeTaskSession, FakeTaskStream, TaskOptions12
from winml.modelkit.session import (
    CGCLoadError,
    CGCNotInstalledError,
    CGCSessionError,
    WinMLCGCSession,
    cgc_package_versions,
)
from winml.modelkit.session.cgc_session import _not_installed_error
from winml.modelkit.sysinfo import DXCoreAdapterError


_CGC_SESSION = importlib.import_module("winml.modelkit.session.cgc_session")
_LOGGER = "winml.modelkit.session.cgc_session"

_ADAPTER_POINTER = 0x5A
_ADAPTER_NAME = "Fake GPU"
_ADAPTER_LUID = "0x00000000_0x0000BEEF"
_CAPACITY = 64
_VOCABULARY = 151936
_RAW_PROMPT = [11, 12]
_TEMPLATED_PROMPT = [10, 11, 12, 13]
_TOKENIZER_EOS = (151645,)
_E_NOINTERFACE = 0x80004002
_E_FAIL = 0x80004005
_NO_ARGUMENT: Any = object()

_LOADED_HANDLES = ["task", "configuration", "tasks", "tokenizer", "model", "target", "runtime"]


# ---------------------------------------------------------------------------
# Fake windowsml.runtime and windowsml.tasks (2.8 model-backed Task API)
# ---------------------------------------------------------------------------


class _WinMLError(Exception):
    """``windowsml.runtime.WinMLError``: a failing HRESULT, normalized to 32 bits."""

    def __init__(self, hr: int, context: str = "") -> None:
        self.hresult = hr & 0xFFFFFFFF
        message = f"WinML call failed (HRESULT 0x{self.hresult:08X})"
        super().__init__(f"{message} in {context}" if context else message)


class _NotSupportedError(_WinMLError):
    """``windowsml.runtime.NotSupportedError``: an optional interface is missing."""


class _ExecutionTargetKind(IntEnum):
    CPU = 1
    GPU = 2
    NPU = 3


class _FakeApis:
    """Fake ``windowsml.runtime`` and ``windowsml.tasks`` modules of the 2.8 preview.

    The members the session drives append their name to ``calls`` and put their
    argument in ``received``; every closed handle appends its kind to ``closed``.

    Args:
        fail: Exception the named call raises after it is recorded.
        capacity: ``configuration.logical_capacity`` read-back.
        vocabulary: Tokenizer vocabulary size, or the exception reading it raises.
        chat_template: ``True`` renders the template. ``False`` raises
            ``NotSupportedError(E_NOINTERFACE)``, as the projection does for a
            tokenizer without the conversation formatter; an exception is
            raised as given, as a failing native ``FormatConversation`` is.
        target_kind: ``ExecutionTargetKind`` member the execution target reports.
        tokenizer: Whether ``set_model`` created a tokenizer.
        stream: Stream the Task session returns (default: a new one).
    """

    def __init__(
        self,
        *,
        fail: dict[str, BaseException] | None = None,
        capacity: int | None = _CAPACITY,
        vocabulary: int | BaseException = _VOCABULARY,
        chat_template: bool | BaseException = True,
        target_kind: str = "GPU",
        tokenizer: bool = True,
        stream: FakeTaskStream | None = None,
    ) -> None:
        self.fail = fail or {}
        self.capacity = capacity
        self.vocabulary = vocabulary
        self.chat_template = chat_template
        self.target_kind = target_kind
        self.tokenizer = tokenizer
        self.stream = stream
        self.calls: list[str] = []
        self.closed: list[str] = []
        self.received: dict[str, Any] = {}
        self.session: FakeTaskSession | None = None

    def record(self, call: str, argument: Any = _NO_ARGUMENT) -> None:
        """Record ``call`` and its argument, then raise the failure configured for it."""
        self.calls.append(call)
        if argument is not _NO_ARGUMENT:
            self.received[call] = argument
        if call in self.fail:
            raise self.fail[call]

    def build(self) -> tuple[SimpleNamespace, SimpleNamespace]:
        """Return the fake ``(windowsml.runtime, windowsml.tasks)`` modules."""
        runtime_api = SimpleNamespace(
            Runtime=lambda: _Runtime(self),
            WinMLError=_WinMLError,
            NotSupportedError=_NotSupportedError,
            ExecutionTargetKind=_ExecutionTargetKind,
            ChatCompletionRequest=SimpleNamespace,
            ChatCompletionMessage=SimpleNamespace,
            ChatRole=SimpleNamespace(USER="user"),
        )
        tasks_api = SimpleNamespace(
            Tasks=lambda runtime: _Tasks(self, runtime),
            TextGenerationConfiguration=_Configuration,
            TextGenerationOptions=TaskOptions12,
        )
        return runtime_api, tasks_api


def _require_type(value: Any, expected: type, name: str) -> None:
    # The real projection rejects foreign objects with TypeError.
    if not isinstance(value, expected):
        raise TypeError(f"{name} must be a {expected.__name__}")


class _Handle:
    """A Windows ML object whose ``close()`` appends its kind to ``closed``."""

    def __init__(self, apis: _FakeApis, closed_as: str) -> None:
        self._apis = apis
        self._closed_as = closed_as

    def close(self) -> None:
        self._apis.closed.append(self._closed_as)


class _Target(_Handle):
    def __init__(self, apis: _FakeApis) -> None:
        super().__init__(apis, "target")
        self.kind = _ExecutionTargetKind[apis.target_kind]


class _Model(_Handle):
    def __init__(self, apis: _FakeApis) -> None:
        super().__init__(apis, "model")


class _Runtime(_Handle):
    def __init__(self, apis: _FakeApis) -> None:
        super().__init__(apis, "runtime")
        apis.record("Runtime")

    def create_target_from_adapter(self, adapter: Any) -> _Target:
        self._apis.record("create_target_from_adapter", adapter)
        return _Target(self._apis)

    def load_model(self, path: str) -> _Model:
        self._apis.record("load_model", path)
        return _Model(self._apis)


class _FormattedConversation:
    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = tuple(token_ids)

    def __enter__(self) -> _FormattedConversation:
        return self

    def __exit__(self, *_exc: object) -> None:
        pass


class _Tokenizer(_Handle):
    def __init__(self, apis: _FakeApis) -> None:
        super().__init__(apis, "tokenizer")
        self.eos_token_ids = _TOKENIZER_EOS

    @property
    def vocabulary_identity(self) -> SimpleNamespace:
        self._apis.record("vocabulary_identity")
        if isinstance(self._apis.vocabulary, BaseException):
            raise self._apis.vocabulary
        return SimpleNamespace(token_count=self._apis.vocabulary, sha256=bytes(32))

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is True
        return list(_RAW_PROMPT)

    def format_conversation(self, request: Any) -> _FormattedConversation:
        template = self._apis.chat_template
        if template is False:
            raise _NotSupportedError(_E_NOINTERFACE, "IWinMLStructuredConversationFormatter")
        if isinstance(template, BaseException):
            raise template
        self._apis.received["format_conversation"] = request
        return _FormattedConversation(_TEMPLATED_PROMPT)


class _Configuration(_Handle):
    """``TextGenerationConfiguration``; its ``set_model`` marks the model-backed API."""

    def __init__(self, apis: _FakeApis) -> None:
        super().__init__(apis, "configuration")

    def set_model(
        self, target: _Target, model: _Model, *, compiled_program_directory: str | None = None
    ) -> None:
        _require_type(target, _Target, "target")
        _require_type(model, _Model, "model")
        # The session creates the program cache before binding the model.
        assert compiled_program_directory is None or Path(compiled_program_directory).is_dir()
        self._apis.record("set_model", {"compiled_program_directory": compiled_program_directory})

    @property
    def tokenizer(self) -> _Tokenizer | None:
        self._apis.record("tokenizer")
        return _Tokenizer(self._apis) if self._apis.tokenizer else None

    def set_logical_capacity(self, capacity: int) -> None:
        _require_type(capacity, int, "capacity")
        self._apis.record("set_logical_capacity", capacity)

    @property
    def logical_capacity(self) -> int | None:
        self._apis.record("logical_capacity")
        return self._apis.capacity


class _Task(_Handle):
    def __init__(self, apis: _FakeApis, tokenizer: _Tokenizer, eos_token_ids: Any) -> None:
        super().__init__(apis, "task")
        # An explicit set, even an empty one, replaces the tokenizer's EOS IDs.
        self._eos = tuple(tokenizer.eos_token_ids if eos_token_ids is None else eos_token_ids)

    @property
    def default_eos_token_ids(self) -> tuple[int, ...]:
        return self._eos

    def create_session(self, configuration: _Configuration) -> FakeTaskSession:
        _require_type(configuration, _Configuration, "configuration")
        self._apis.record("create_session")
        self._apis.session = FakeTaskSession(
            stream=self._apis.stream, calls=self._apis.calls, closed=self._apis.closed
        )
        return self._apis.session


class _Tasks(_Handle):
    def __init__(self, apis: _FakeApis, runtime: Any) -> None:
        super().__init__(apis, "tasks")
        _require_type(runtime, _Runtime, "runtime")
        apis.record("Tasks")

    def create_text_generation_configuration(self) -> _Configuration:
        self._apis.record("create_text_generation_configuration")
        return _Configuration(self._apis)

    def create_text_generation_task(
        self, tokenizer: _Tokenizer, eos_token_ids: Any = None
    ) -> _Task:
        _require_type(tokenizer, _Tokenizer, "tokenizer")
        self._apis.record("create_text_generation_task", eos_token_ids)
        return _Task(self._apis, tokenizer, eos_token_ids)


class _FakeAdapter:
    """The owned GPU handle ``open_gpu_adapter`` returns; ``close()`` appends ``"adapter"``."""

    pointer = _ADAPTER_POINTER
    name = _ADAPTER_NAME
    luid = _ADAPTER_LUID
    selection = "high-performance"

    def __init__(self, closed: list[str]) -> None:
        self._closed = closed

    def close(self) -> None:
        self._closed.append("adapter")


def _patch_adapter(monkeypatch: pytest.MonkeyPatch, apis: _FakeApis) -> None:
    """Replace the GPU selection with a fake handle that closes into ``apis.closed``."""

    def open_fake_adapter(device_luid: str | None) -> _FakeAdapter:
        apis.received["open_gpu_adapter"] = device_luid
        return _FakeAdapter(apis.closed)

    monkeypatch.setattr(_CGC_SESSION, "open_gpu_adapter", open_fake_adapter)


def _session(
    monkeypatch: pytest.MonkeyPatch,
    apis: _FakeApis,
    model_path: Path,
    *,
    model_format: str = "gguf",
    **options: Any,
) -> WinMLCGCSession:
    _patch_adapter(monkeypatch, apis)
    runtime_api, tasks_api = apis.build()
    return WinMLCGCSession(
        model_path,
        model_format=model_format,
        runtime_api=runtime_api,
        tasks_api=tasks_api,
        **options,
    )


def _needs_preview_text() -> str:
    return (
        "CGC text generation needs windowsml 2.8.183.dev0 or newer, which provides the "
        "model-backed Task API; the installed windowsml is "
        f"{importlib.metadata.version('windowsml')}. Install it in a separate environment: "
        "see docs/samples/cgc-text-generation-perf.md in the winml-cli repository."
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_session_drives_the_model_backed_task_api_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    apis = _FakeApis()
    model = tmp_path / "model.gguf"
    session = _session(monkeypatch, apis, model)

    session.open_tokenizer()
    assert (session.effective_device, session.context_length) == (None, None)
    session.load(context_length=_CAPACITY)
    timing = session.generate_timed(
        [1, 2], max_new_tokens=2, clock=itertools.count(step=1.0).__next__
    )

    assert apis.calls == [
        "Runtime",
        "create_target_from_adapter",
        "load_model",
        "Tasks",
        "create_text_generation_configuration",
        "set_model",
        "tokenizer",
        "vocabulary_identity",
        "set_logical_capacity",
        "logical_capacity",
        "create_text_generation_task",
        "create_session",
        "reset",
        "generate_tokens",
    ]
    assert apis.received["open_gpu_adapter"] is None
    assert apis.received["create_target_from_adapter"] == _ADAPTER_POINTER
    assert apis.received["load_model"] == str(model)
    assert apis.received["set_model"] == {"compiled_program_directory": None}
    assert apis.received["set_logical_capacity"] == _CAPACITY
    assert set(session.load_timings_ms) == {
        "runtime_create_duration_ms",
        "target_create_duration_ms",
        "model_create_duration_ms",
        "set_model_duration_ms",
        "session_create_duration_ms",
    }
    assert all(duration >= 0.0 for duration in session.load_timings_ms.values())
    assert (session.effective_device, session.context_length, session.vocabulary_size) == (
        "gpu",
        _CAPACITY,
        _VOCABULARY,
    )
    assert (session.model_path, session.model_format) == (model, "gguf")
    assert (session.adapter_name, session.adapter_luid, session.adapter_selection) == (
        _ADAPTER_NAME,
        _ADAPTER_LUID,
        "high-performance",
    )
    assert (timing.token_ids, timing.stop_reason) == ((7, 8), "MAX_TOKENS")


def test_close_releases_every_handle_once_with_the_adapter_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    apis = _FakeApis()
    with _session(monkeypatch, apis, tmp_path / "model.gguf") as session:
        session.open_tokenizer()
        session.load(context_length=_CAPACITY)
        session.generate_timed([1, 2], max_new_tokens=2)
    session.close()

    assert apis.closed == ["session", *_LOADED_HANDLES, "adapter"]
    assert apis.session is not None
    assert apis.session.stream.closed
    calls = list(apis.calls)
    with pytest.raises(CGCSessionError, match=r"^The WinMLCGCSession is closed\.$"):
        session.open_tokenizer()
    assert apis.calls == calls


@pytest.mark.parametrize(
    ("failure", "raised"),
    [(_WinMLError(_E_FAIL), CGCLoadError), (KeyboardInterrupt(), KeyboardInterrupt)],
    ids=["hresult", "KeyboardInterrupt"],
)
def test_failed_load_leaves_every_acquired_handle_closable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: BaseException,
    raised: type[BaseException],
) -> None:
    apis = _FakeApis(fail={"create_session": failure})
    session = _session(monkeypatch, apis, tmp_path / "model.gguf")
    session.open_tokenizer()

    with pytest.raises(raised):
        session.load(context_length=_CAPACITY)
    assert session.effective_device is None
    session.close()

    assert apis.closed == [*_LOADED_HANDLES, "adapter"]


@pytest.mark.parametrize(
    ("eos_token_ids", "received", "task_eos"),
    [((), (), ()), (None, None, _TOKENIZER_EOS)],
    ids=["ignore-eos", "tokenizer-eos"],
)
def test_ignore_eos_creates_the_task_with_an_empty_eos_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    eos_token_ids: tuple[int, ...] | None,
    received: tuple[int, ...] | None,
    task_eos: tuple[int, ...],
) -> None:
    apis = _FakeApis()
    session = _session(monkeypatch, apis, tmp_path / "model.gguf")
    with pytest.raises(
        CGCSessionError,
        match=r"^WinMLCGCSession\.load\(\) must be called before reading the Task EOS set\.$",
    ):
        _ = session.task_default_eos_token_ids

    # load() opens the tokenizer itself when the caller has not.
    session.load(context_length=_CAPACITY, eos_token_ids=eos_token_ids)

    assert apis.received["create_text_generation_task"] == received
    assert session.task_default_eos_token_ids == task_eos


def test_prompt_checks_use_the_read_backs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    apis = _FakeApis()
    session = _session(monkeypatch, apis, tmp_path / "model.gguf")
    session.load(context_length=_CAPACITY)

    with pytest.raises(
        CGCSessionError,
        match=rf"does not fit the model sequence capacity \({_CAPACITY} tokens\)",
    ):
        session.generate_timed(range(_CAPACITY), max_new_tokens=1)
    with pytest.raises(
        CGCSessionError, match=f"outside the model vocabulary of {_VOCABULARY} entries"
    ):
        session.generate_timed([1, _VOCABULARY], max_new_tokens=1)
    assert "generate_tokens" not in apis.calls


@pytest.mark.parametrize(
    ("capacity", "vocabulary"),
    [
        (None, _NotSupportedError(_E_NOINTERFACE)),
        (None, _WinMLError(_E_FAIL)),
        (None, AttributeError("vocabulary_identity")),
        (0, 0),
    ],
    ids=["not-supported", "winml-error", "no-attribute", "zero"],
)
def test_missing_read_backs_are_null_with_a_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capacity: int | None,
    vocabulary: int | BaseException,
) -> None:
    apis = _FakeApis(capacity=capacity, vocabulary=vocabulary)
    session = _session(monkeypatch, apis, tmp_path / "model.gguf")

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        session.load(context_length=_CAPACITY)

    assert (session.context_length, session.vocabulary_size) == (None, None)
    assert [
        (record.levelno, record.getMessage()) for record in caplog.records if record.name == _LOGGER
    ] == [
        (
            logging.WARNING,
            "Windows ML did not report the tokenizer vocabulary size; vocabulary_size is "
            "reported as null and prompt token IDs are not range-checked.",
        ),
        (
            logging.WARNING,
            f"Windows ML did not report the logical capacity after set_logical_capacity("
            f"{_CAPACITY}); context_length is reported as null and the prompt-fit check is "
            "skipped.",
        ),
    ]
    assert "logical capacity" in caplog.text
    assert "vocabulary size" in caplog.text
    # Generation proceeds without the checks the read-backs would have enabled.
    assert session.generate_timed([1, 2], max_new_tokens=2).generated_tokens == 2


@pytest.mark.parametrize(("capacity", "warned"), [(4096, False), (None, True)])
def test_without_a_context_length_the_default_capacity_is_read_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capacity: int | None,
    warned: bool,
) -> None:
    apis = _FakeApis(capacity=capacity)
    session = _session(monkeypatch, apis, tmp_path / "model.gguf")

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        session.load()

    assert "set_logical_capacity" not in apis.calls
    assert "logical_capacity" in apis.calls
    assert session.context_length == capacity
    messages = [record.getMessage() for record in caplog.records if record.name == _LOGGER]
    expected = (
        "Windows ML did not report the logical capacity; context_length is reported as "
        "null and the prompt-fit check is skipped."
    )
    assert messages == ([expected] if warned else [])


def test_program_cache_is_created_passed_as_text_and_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    apis = _FakeApis()
    cache = tmp_path / "cache" / "nested"
    session = _session(monkeypatch, apis, tmp_path / "model.gguf", program_cache_dir=cache)

    session.open_tokenizer()
    session.close()

    assert apis.received["set_model"] == {"compiled_program_directory": str(cache)}
    assert cache.is_dir()


def test_uncreatable_program_cache_is_a_load_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory", encoding="utf-8")
    apis = _FakeApis()
    session = _session(
        monkeypatch, apis, tmp_path / "model.gguf", program_cache_dir=blocker / "cache"
    )

    with pytest.raises(
        CGCLoadError, match=r"^Could not create the --program-cache directory"
    ) as excinfo:
        session.open_tokenizer()

    assert str(blocker / "cache") in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, OSError)
    assert "set_model" not in apis.calls
    session.close()
    assert apis.closed == ["configuration", "tasks", "model", "target", "runtime", "adapter"]


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "error", "model_format", "raises", "expected", "absent"),
    [
        (
            "set_model",
            _WinMLError(0x80070032),
            "gguf",
            CGCLoadError,
            [
                "cannot run this model on Fake GPU (0x00000000_0x0000BEEF)",
                "set_model returned 0x80070032",
                "omit --backend cgc",
            ],
            None,
        ),
        (
            "set_model",
            _WinMLError(0x80070032),
            "safetensors",
            CGCLoadError,
            ["does not support it"],
            "omit --backend cgc",
        ),
        (
            "load_model",
            _WinMLError(0x80070032),
            "safetensors",
            CGCLoadError,
            ["cannot run this model", "load_model returned 0x80070032"],
            "omit --backend cgc",
        ),
        (
            "create_session",
            _WinMLError(0x80070032),
            "safetensors",
            CGCLoadError,
            ["The selected GPU Fake GPU", "cannot run this model", "--device-luid"],
            "omit --backend cgc",
        ),
        (
            "load_model",
            _WinMLError(0x80070002),
            "safetensors",
            CGCLoadError,
            ["could not find a model file (0x80070002)", "tokenizer_config.json"],
            None,
        ),
        (
            "load_model",
            _WinMLError(0x80070003),
            "gguf",
            CGCLoadError,
            ["could not find a model file (0x80070003)"],
            None,
        ),
        (
            "set_model",
            _NotSupportedError(_E_NOINTERFACE),
            "gguf",
            CGCNotInstalledError,
            ["2.8.183.dev0"],
            None,
        ),
        (
            "Tasks",
            FileNotFoundError("WinMLTasks.dll"),
            "gguf",
            CGCNotInstalledError,
            ["missing a native library", "WinMLTasks.dll"],
            None,
        ),
        (
            "load_model",
            ImportError("onnxruntime-windowsml is malformed"),
            "safetensors",
            CGCNotInstalledError,
            ["during model load", "malformed"],
            None,
        ),
        (
            "load_model",
            _WinMLError(_E_FAIL),
            "gguf",
            CGCLoadError,
            ["CGC model load failed (0x80004005)"],
            "cannot run this model",
        ),
    ],
)
def test_windows_ml_errors_become_actionable_messages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    call: str,
    error: BaseException,
    model_format: str,
    raises: type[CGCSessionError],
    expected: list[str],
    absent: str | None,
) -> None:
    apis = _FakeApis(fail={call: error})
    session = _session(monkeypatch, apis, tmp_path / "model", model_format=model_format)

    with pytest.raises(raises) as excinfo:
        session.open_tokenizer()
        session.load(context_length=_CAPACITY)
    session.close()

    assert type(excinfo.value) is raises
    assert excinfo.value.__cause__ is error
    message = str(excinfo.value)
    for text in expected:
        assert text in message
    assert absent is None or absent not in message


@pytest.mark.parametrize("model_format", ["gguf", "safetensors"])
@pytest.mark.parametrize("hr", [0x80070032, _E_FAIL], ids=["not-supported", "e-fail"])
def test_set_model_rejection_keeps_the_hresult_and_native_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hr: int, model_format: str
) -> None:
    # The 2.8 CGC backend rejects an encoder-only checkpoint in set_model with
    # E_FAIL instead of ERROR_NOT_SUPPORTED; both mean it cannot run the model.
    error = _WinMLError(hr, "IWinMLModelTextGenerationConfiguration.SetModel")
    apis = _FakeApis(fail={"set_model": error})
    session = _session(monkeypatch, apis, tmp_path / "model", model_format=model_format)

    with pytest.raises(CGCLoadError) as excinfo:
        session.open_tokenizer()
    session.close()

    assert excinfo.value.__cause__ is error
    message = str(excinfo.value)
    assert f"cannot run this model on {_ADAPTER_NAME} ({_ADAPTER_LUID})" in message
    assert f"set_model returned 0x{hr:08X}" in message
    assert str(error) in message
    # Only a .gguf file has another engine to point to; none is tried.
    assert ("omit --backend cgc" in message) is (model_format == "gguf")
    assert apis.calls[-1] == "set_model"


def test_tasks_api_without_set_model_needs_the_preview(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    apis = _FakeApis()
    _patch_adapter(monkeypatch, apis)
    runtime_api, tasks_api = apis.build()
    # The configuration type of a Task API without the model-backed methods.
    tasks_api.TextGenerationConfiguration = type("TextGenerationConfiguration", (), {})
    session = WinMLCGCSession(
        tmp_path / "model.gguf",
        model_format="gguf",
        runtime_api=runtime_api,
        tasks_api=tasks_api,
    )

    with pytest.raises(CGCNotInstalledError) as excinfo:
        session.open_tokenizer()

    assert str(excinfo.value) == _needs_preview_text()
    assert apis.calls == []
    session.close()
    assert apis.closed == ["adapter"]


def test_stable_windowsml_fails_before_gpu_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The pinned stable windowsml has no runtime subpackage.
    monkeypatch.setitem(sys.modules, "windowsml.runtime", None)

    def select_gpu(device_luid: str | None) -> Any:
        pytest.fail("the GPU was selected before the windowsml check")

    monkeypatch.setattr(_CGC_SESSION, "open_gpu_adapter", select_gpu)

    with pytest.raises(CGCNotInstalledError) as excinfo:
        WinMLCGCSession(tmp_path / "model.gguf", model_format="gguf")

    message = str(excinfo.value)
    assert "2.8.183.dev0" in message
    assert importlib.metadata.version("windowsml") in message
    assert message == _needs_preview_text()


@pytest.mark.parametrize(
    ("error", "expected", "absent"),
    [
        (
            ImportError("A different DirectML.dll is already loaded."),
            "A different DirectML.dll is already loaded",
            "not installed",
        ),
        (
            ModuleNotFoundError("No module named 'comtypes'", name="comtypes"),
            "comtypes",
            "2.8.183.dev0",
        ),
        (
            ModuleNotFoundError("No module named 'windowsml.runtime'", name="windowsml.runtime"),
            "2.8.183.dev0",
            None,
        ),
        (FileNotFoundError("WinMLRuntimeCore.dll"), "missing a native library", None),
    ],
)
def test_import_failures_keep_their_reason(
    error: BaseException, expected: str, absent: str | None
) -> None:
    translated = _not_installed_error(error)

    assert isinstance(translated, CGCNotInstalledError)
    message = str(translated)
    assert expected in message
    assert absent is None or absent not in message


def test_non_gpu_target_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    apis = _FakeApis(target_kind="NPU")
    session = _session(monkeypatch, apis, tmp_path / "model.gguf")

    with pytest.raises(
        CGCLoadError,
        match=(
            r"^The adapter Fake GPU \(0x00000000_0x0000BEEF\) did not resolve to a GPU "
            r"execution target\.$"
        ),
    ):
        session.open_tokenizer()

    assert apis.calls == ["Runtime", "create_target_from_adapter"]
    session.close()
    assert apis.closed == ["target", "runtime", "adapter"]


def test_adapter_selection_errors_become_load_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    message = (
        "No hardware GPU has LUID 0x00000000_0x0000c003. Hardware GPUs: Fake GPU "
        "(0x00000000_0x0000BEEF). Run 'winml sys' to list adapters; a LUID changes after a "
        "reboot or a driver update."
    )
    received: list[str | None] = []

    def select_gpu(device_luid: str | None) -> Any:
        received.append(device_luid)
        raise DXCoreAdapterError(message)

    monkeypatch.setattr(_CGC_SESSION, "open_gpu_adapter", select_gpu)
    runtime_api, tasks_api = _FakeApis().build()

    with pytest.raises(CGCLoadError) as excinfo:
        WinMLCGCSession(
            tmp_path / "model.gguf",
            model_format="gguf",
            device_luid="0x00000000_0x0000c003",
            runtime_api=runtime_api,
            tasks_api=tasks_api,
        )

    assert str(excinfo.value) == message
    assert isinstance(excinfo.value.__cause__, DXCoreAdapterError)
    assert received == ["0x00000000_0x0000c003"]


@pytest.mark.parametrize(("model_format", "hint"), [("gguf", False), ("safetensors", True)])
def test_missing_tokenizer_is_a_load_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_format: str, hint: bool
) -> None:
    apis = _FakeApis(tokenizer=False)
    session = _session(monkeypatch, apis, tmp_path / "model", model_format=model_format)

    with pytest.raises(
        CGCLoadError, match=r"^Windows ML created no tokenizer for this model after set_model\."
    ) as excinfo:
        session.open_tokenizer()

    folder_hint = " The checkpoint folder needs tokenizer.json and tokenizer_config.json."
    assert str(excinfo.value).endswith(folder_hint) is hint
    session.close()
    assert apis.closed == ["configuration", "tasks", "model", "target", "runtime", "adapter"]


# ---------------------------------------------------------------------------
# Prompting and generation
# ---------------------------------------------------------------------------


def test_encode_prompt_opens_the_tokenizer_and_renders_one_user_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    apis = _FakeApis()
    session = _session(monkeypatch, apis, tmp_path / "model.gguf")
    assert session.chat_template_applied is None

    assert session.encode_prompt("Hello") == _TEMPLATED_PROMPT
    assert session.chat_template_applied is True
    request = apis.received["format_conversation"]
    assert [(message.role, message.content) for message in request.messages] == [("user", "Hello")]

    assert session.encode_prompt("Hello", apply_template=False) == _RAW_PROMPT
    assert session.chat_template_applied is False
    # The tokenizer opened once, on the first encode.
    assert apis.calls.count("Runtime") == 1
    assert apis.calls[-1] == "vocabulary_identity"


@pytest.mark.parametrize(
    "missing_template",
    [
        False,
        _WinMLError(_E_FAIL, "IWinMLStructuredConversationFormatter.FormatConversation"),
    ],
    ids=["no-formatter-interface", "formatter-failed"],
)
def test_missing_chat_template_encodes_the_raw_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, missing_template: bool | BaseException
) -> None:
    apis = _FakeApis(chat_template=missing_template)
    session = _session(monkeypatch, apis, tmp_path / "model.gguf")

    assert session.encode_prompt("Hello") == _RAW_PROMPT
    assert session.chat_template_applied is False


def test_generation_needs_a_positive_limit_and_a_loaded_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    apis = _FakeApis()
    session = _session(monkeypatch, apis, tmp_path / "model.gguf")

    with pytest.raises(ValueError, match=r"^max_new_tokens must be positive$"):
        session.generate_timed([1, 2], max_new_tokens=0)
    with pytest.raises(
        CGCSessionError, match=r"^WinMLCGCSession\.load\(\) must be called before generation\.$"
    ):
        session.generate_timed([1, 2], max_new_tokens=2)
    assert apis.calls == []


def test_generation_disables_watermark_and_names_cgc_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    error = _WinMLError(_E_FAIL)
    apis = _FakeApis(stream=FakeTaskStream(error=error))
    session = _session(monkeypatch, apis, tmp_path / "model.gguf")
    session.load(context_length=_CAPACITY)

    with pytest.raises(CGCSessionError, match=r"^CGC inference failed \(0x80004005\)") as excinfo:
        session.generate_timed([1, 2], max_new_tokens=2)

    assert type(excinfo.value) is CGCSessionError
    assert excinfo.value.__cause__ is error
    assert apis.session is not None
    assert apis.session.options.watermark is False


def test_package_versions_report_windowsml_only() -> None:
    assert set(cgc_package_versions()) == {"windowsml"}
    assert cgc_package_versions() == {"windowsml": importlib.metadata.version("windowsml")}
