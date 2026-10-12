# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the GGUF (llama.cpp) Windows ML Runtime session.

A fake ``windowsml.runtime`` module stands in for the preview Runtime, so these
tests need neither the preview packages nor a GGUF model. The fake decoder
always predicts ``last_token + 1`` and the tokenizer marks the last vocabulary
entry as EOS, which makes every stop condition deterministic.
"""

from __future__ import annotations

import struct
import sys
from itertools import count
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from winml.modelkit.session import (
    LlamaLoadError,
    LlamaNotInstalledError,
    LlamaSessionError,
    WinMLLlamaSession,
    llama_package_versions,
)
from winml.modelkit.session.llama_session import (
    _discover_llama_payload,
    _require_llama_backend,
)


if TYPE_CHECKING:
    from pathlib import Path


_VOCABULARY = 8
_EOS = _VOCABULARY - 1
_BOS = 1
_RAW_PROMPT = [1, 2]
_TEMPLATED_PROMPT = [1, 2, 3]
_DYNAMIC = 0xFFFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# Fake windowsml.runtime
# ---------------------------------------------------------------------------


class _WinMLError(Exception):
    def __init__(self, message: str = "", hresult: int | None = None) -> None:
        super().__init__(message)
        self.hresult = hresult


class _NotSupportedError(_WinMLError):
    pass


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class _DataTypes:
    INT32 = SimpleNamespace(name="INT32")
    INT64 = SimpleNamespace(name="INT64")
    FLOAT32 = SimpleNamespace(name="FLOAT32")


class _Target(_Closable):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = SimpleNamespace(name=kind)


class _Tensor(_Closable):
    def __init__(self, value: np.ndarray) -> None:
        super().__init__()
        self.value = value

    def desc(self) -> tuple[Any, tuple[int, ...]]:
        return _DataTypes.FLOAT32, tuple(self.value.shape)

    def to_numpy(self) -> np.ndarray:
        return self.value

    def region(self, starts: list[int], extents: list[int], *, target: Any) -> _Tensor:
        index = tuple(
            slice(start, start + extent) for start, extent in zip(starts, extents, strict=True)
        )
        return _Tensor(self.value[index])


class _Schema(_Closable):
    def __init__(self, token_type: Any, token_rank: int) -> None:
        super().__init__()
        self._token_type = token_type
        self._token_rank = token_rank

    def input_desc(self, index: int) -> tuple[Any, tuple[int, ...]]:
        assert index == 0
        shape = (_DYNAMIC,) if self._token_rank == 1 else (1, _DYNAMIC)
        return self._token_type, shape

    def output_desc(self, index: int) -> tuple[Any, tuple[int, ...]]:
        assert index == 0
        return _DataTypes.FLOAT32, (1, _DYNAMIC, _VOCABULARY)


class _Stage(_Closable):
    """Decoder stage that predicts ``last token + 1`` for every position."""

    def __init__(self, target: _Target, *, token_type: Any, token_rank: int) -> None:
        super().__init__()
        self._target_kind = target.kind.name
        self._token_type = token_type
        self._token_rank = token_rank
        self.requested_outputs: set[int] = set()
        self.capacity_hint: int | None = None
        self.default_capacity = 64
        self.sequence_position = 0
        self.bound: list[int] = []
        self.last_run: list[int] = []

    @property
    def sequence_capacity(self) -> int:
        return self.capacity_hint or self.default_capacity

    @property
    def execution_target(self) -> _Target:
        return _Target(self._target_kind)

    def request_output(self, index: int) -> None:
        self.requested_outputs.add(index)

    def set_sequence_capacity_hint(self, capacity: int) -> None:
        self.capacity_hint = capacity

    def schema(self) -> _Schema:
        return _Schema(self._token_type, self._token_rank)

    def bind_input(self, index: int, tensor: _Tensor) -> None:
        assert index == 0
        self.bound = [int(value) for value in tensor.value.reshape(-1)]

    def output(self, index: int) -> _Tensor:
        assert index in self.requested_outputs
        logits = np.zeros((1, len(self.last_run), _VOCABULARY), dtype=np.float32)
        for position, token in enumerate(self.last_run):
            logits[0, position, (token + 1) % _VOCABULARY] = 1.0
        return _Tensor(logits)


class _Pipeline(_Closable):
    def __init__(self, stage: _Stage) -> None:
        super().__init__()
        self._stage = stage
        self.runs: list[list[int]] = []

    def run(self) -> None:
        stage = self._stage
        stage.sequence_position += len(stage.bound)
        stage.last_run = list(stage.bound)
        self.runs.append(list(stage.bound))

    def reset(self) -> None:
        self._stage.sequence_position = 0


class _Builder(_Closable):
    def __init__(self, runtime: _Runtime) -> None:
        super().__init__()
        self._runtime = runtime

    def add_model_stage(self, model: Any, target: _Target, name: str) -> _Stage:
        assert name == "decoder"
        self._runtime.stage = _Stage(
            target,
            token_type=self._runtime.token_type,
            token_rank=self._runtime.token_rank,
        )
        return self._runtime.stage

    def build(self) -> _Pipeline:
        assert self._runtime.stage is not None
        self._runtime.pipeline = _Pipeline(self._runtime.stage)
        return self._runtime.pipeline


class _Decoder(_Closable):
    def decode_token(self, token: int) -> str:
        return f"<{token}>"

    def reset(self) -> None:
        pass


class _Formatted(_Closable):
    def __init__(self, token_ids: list[int]) -> None:
        super().__init__()
        self.token_ids = token_ids


class _Tokenizer(_Closable):
    def __init__(self, *, chat_template: bool) -> None:
        super().__init__()
        self._chat_template = chat_template
        self.eos_token_ids = (_EOS,)
        self.bos_token_id = _BOS
        self.requests: list[Any] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is True
        return list(_RAW_PROMPT)

    def format_conversation(self, request: Any) -> _Formatted:
        if not self._chat_template:
            raise _NotSupportedError("model has no chat template")
        self.requests.append(request)
        return _Formatted(list(_TEMPLATED_PROMPT))

    def create_decoder(self) -> _Decoder:
        return _Decoder()


class _Runtime(_Closable):
    def __init__(self, api: SimpleNamespace) -> None:
        super().__init__()
        self._api = api
        self.token_type = api.token_type
        self.token_rank = api.token_rank
        self.stage: _Stage | None = None
        self.pipeline: _Pipeline | None = None
        self.loaded: list[str] = []
        api.runtimes.append(self)

    def create_cpu_target(self) -> _Target:
        return _Target("CPU")

    def create_target(self, kind: Any, preference: Any) -> _Target:
        assert preference is self._api.ExecutionTargetPreference.DEFAULT
        return _Target(kind.name)

    def load_model(self, path: str) -> _Closable:
        if self._api.load_hresult is not None:
            raise _WinMLError("LoadModel failed", hresult=self._api.load_hresult)
        self.loaded.append(path)
        return _Closable()

    def create_pipeline_builder(self) -> _Builder:
        return _Builder(self)

    def create_tensor(
        self, data_type: Any, shape: tuple[int, ...], payload: bytes, *, target: Any
    ) -> _Tensor:
        element = "i" if data_type is _DataTypes.INT32 else "q"
        values = struct.unpack(f"<{len(payload) // struct.calcsize(element)}{element}", payload)
        return _Tensor(np.asarray(values, dtype=np.int64).reshape(shape))


def _runtime_api(
    *,
    chat_template: bool = True,
    load_hresult: int | None = None,
    token_type: Any = _DataTypes.INT32,
    token_rank: int = 2,
) -> SimpleNamespace:
    api = SimpleNamespace(
        WinMLError=_WinMLError,
        NotSupportedError=_NotSupportedError,
        TensorDataType=_DataTypes,
        ExecutionTargetKind=SimpleNamespace(
            CPU=SimpleNamespace(name="CPU"),
            GPU=SimpleNamespace(name="GPU"),
            NPU=SimpleNamespace(name="NPU"),
        ),
        ExecutionTargetPreference=SimpleNamespace(DEFAULT=object()),
        ChatRole=SimpleNamespace(USER="user", ASSISTANT="assistant"),
        ChatCompletionMessage=lambda role, content: SimpleNamespace(role=role, content=content),
        ChatCompletionRequest=lambda messages: SimpleNamespace(messages=messages),
        load_hresult=load_hresult,
        token_type=token_type,
        token_rank=token_rank,
        runtimes=[],
        tokenizers=[],
    )
    api.Runtime = lambda: _Runtime(api)

    def from_file(path: str) -> _Tokenizer:
        tokenizer = _Tokenizer(chat_template=chat_template)
        api.tokenizers.append(tokenizer)
        return tokenizer

    api.Tokenizer = SimpleNamespace(from_file=from_file)
    return api


def _gguf(tmp_path: Path) -> Path:
    model = tmp_path / "tiny-q4_k_m.gguf"
    model.write_bytes(b"GGUF")
    return model


def _session(
    tmp_path: Path, *, device: str = "cpu", **api_kwargs: Any
) -> tuple[WinMLLlamaSession, SimpleNamespace]:
    api = _runtime_api(**api_kwargs)
    return WinMLLlamaSession(_gguf(tmp_path), device=device, runtime_api=api), api


# ---------------------------------------------------------------------------
# Construction and loading
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_rejects_non_gguf_path(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"\.gguf"):
            WinMLLlamaSession(tmp_path / "model.onnx")

    @pytest.mark.parametrize("device", ["npu", "auto", "config"])
    def test_rejects_unsupported_device(self, tmp_path: Path, device: str) -> None:
        with pytest.raises(ValueError, match="cpu"):
            WinMLLlamaSession(_gguf(tmp_path), device=device)

    def test_missing_file_is_reported_before_runtime_import(self, tmp_path: Path) -> None:
        session = WinMLLlamaSession(tmp_path / "missing.gguf", runtime_api=_runtime_api())
        with pytest.raises(FileNotFoundError, match=r"missing\.gguf"):
            session.open_tokenizer()

    def test_missing_runtime_module_is_actionable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "windowsml.runtime", None)
        session = WinMLLlamaSession(_gguf(tmp_path))
        with pytest.raises(LlamaNotInstalledError, match="windowsml-llama-core"):
            session.open_tokenizer()


class TestLoad:
    def test_builds_one_decoder_stage_with_capacity_hint(self, tmp_path: Path) -> None:
        session, api = _session(tmp_path)
        session.load(context_length=40)

        runtime = api.runtimes[0]
        assert runtime.loaded == [str(session.model_path)]
        assert runtime.stage is not None
        assert runtime.stage.requested_outputs == {0}
        assert runtime.stage.capacity_hint == 40
        assert session.context_length == 40
        assert session.effective_device == "cpu"
        assert set(session.load_timings_ms) == {
            "runtime_create_duration_ms",
            "tokenizer_create_duration_ms",
            "model_create_duration_ms",
            "pipeline_build_duration_ms",
        }

    def test_without_hint_keeps_backend_capacity(self, tmp_path: Path) -> None:
        session, api = _session(tmp_path)
        session.load()
        assert api.runtimes[0].stage.capacity_hint is None
        assert session.context_length == 64

    def test_gpu_uses_generic_gpu_target(self, tmp_path: Path) -> None:
        session, _api = _session(tmp_path, device="gpu")
        session.load(context_length=16)
        assert session.effective_device == "gpu"

    def test_gpu_without_backend_hresult_is_actionable(self, tmp_path: Path) -> None:
        session, _api = _session(tmp_path, device="gpu", load_hresult=0x80070032)
        with pytest.raises(LlamaLoadError, match="GPU backend") as excinfo:
            session.load()
        assert "0x80070032" in str(excinfo.value)

    def test_other_native_load_failure_keeps_hresult(self, tmp_path: Path) -> None:
        session, _api = _session(tmp_path, load_hresult=0x80004005)
        with pytest.raises(LlamaLoadError, match="0x80004005"):
            session.load()

    @pytest.mark.parametrize("token_type", [_DataTypes.INT32, _DataTypes.INT64])
    @pytest.mark.parametrize("token_rank", [1, 2])
    def test_token_contract_variants_generate(
        self, tmp_path: Path, token_type: Any, token_rank: int
    ) -> None:
        session, _api = _session(tmp_path, token_type=token_type, token_rank=token_rank)
        session.load(context_length=32)
        timing = session.generate_timed(_TEMPLATED_PROMPT, max_new_tokens=8)
        assert timing.response_text == "<4><5><6>"

    def test_rejects_float_token_input(self, tmp_path: Path) -> None:
        session, _api = _session(tmp_path, token_type=_DataTypes.FLOAT32)
        with pytest.raises(LlamaLoadError, match="INT32 or INT64"):
            session.load()


# ---------------------------------------------------------------------------
# Prompting and generation
# ---------------------------------------------------------------------------


class TestEncodePrompt:
    def test_chat_template_renders_one_user_turn(self, tmp_path: Path) -> None:
        session, api = _session(tmp_path)
        assert session.encode_prompt("hello") == _TEMPLATED_PROMPT
        assert session.chat_template_applied is True
        (request,) = api.tokenizers[0].requests
        assert [(m.role, m.content) for m in request.messages] == [("user", "hello")]

    def test_raw_prompt_when_template_disabled(self, tmp_path: Path) -> None:
        session, _api = _session(tmp_path)
        assert session.encode_prompt("hello", apply_template=False) == _RAW_PROMPT
        assert session.chat_template_applied is False

    def test_falls_back_to_raw_prompt_without_template(self, tmp_path: Path) -> None:
        session, _api = _session(tmp_path, chat_template=False)
        assert session.encode_prompt("hello") == _RAW_PROMPT
        assert session.chat_template_applied is False


class TestGenerateTimed:
    def test_stops_at_eos_without_counting_it(self, tmp_path: Path) -> None:
        session, api = _session(tmp_path)
        session.load(context_length=32)
        timing = session.generate_timed(_TEMPLATED_PROMPT, max_new_tokens=16)

        assert timing.stop_reason == "EOS"
        assert timing.input_tokens == len(_TEMPLATED_PROMPT)
        assert timing.generated_tokens == 3
        assert timing.response_text == "<4><5><6>"
        # Prefill runs the whole prompt once; tokens 4 and 5 are fed back. Token
        # 6 selects EOS, so the step that ran it emits nothing and is not timed.
        assert api.runtimes[0].pipeline.runs == [_TEMPLATED_PROMPT, [4], [5], [6]]
        assert len(timing.decode_s) == 2

    def test_stops_at_max_new_tokens(self, tmp_path: Path) -> None:
        session, api = _session(tmp_path)
        session.load(context_length=32)
        timing = session.generate_timed(_RAW_PROMPT, max_new_tokens=2)

        assert timing.stop_reason == "MAX_TOKENS"
        assert timing.generated_tokens == 2
        assert timing.response_text == "<3><4>"
        # The last emitted token is never run.
        assert api.runtimes[0].pipeline.runs == [_RAW_PROMPT, [3]]
        assert len(timing.decode_s) == 1

    def test_stops_before_exceeding_capacity(self, tmp_path: Path) -> None:
        session, api = _session(tmp_path)
        session.load(context_length=len(_RAW_PROMPT) + 2)
        timing = session.generate_timed(_RAW_PROMPT, max_new_tokens=16)

        assert timing.stop_reason == "CAPACITY"
        assert timing.generated_tokens == 2
        assert api.runtimes[0].stage.sequence_position < session.context_length

    def test_immediate_eos_emits_nothing(self, tmp_path: Path) -> None:
        session, _api = _session(tmp_path)
        session.load(context_length=32)
        timing = session.generate_timed([_EOS - 1], max_new_tokens=4)

        assert (timing.stop_reason, timing.generated_tokens, timing.decode_s) == ("EOS", 0, [])
        assert timing.response_text == ""

    def test_each_request_resets_sequence_state(self, tmp_path: Path) -> None:
        session, _api = _session(tmp_path)
        session.load(context_length=32)
        first = session.generate_timed(_TEMPLATED_PROMPT, max_new_tokens=16)
        second = session.generate_timed(_TEMPLATED_PROMPT, max_new_tokens=16)
        assert first.response_text == second.response_text == "<4><5><6>"

    def test_phase_spans_follow_the_clock(self, tmp_path: Path) -> None:
        session, _api = _session(tmp_path)
        session.load(context_length=32)
        ticks = count()
        timing = session.generate_timed(
            _TEMPLATED_PROMPT, max_new_tokens=16, clock=lambda: float(next(ticks))
        )

        assert (timing.reset_s, timing.prefill_s, timing.first_token_s) == (1.0, 1.0, 1.0)
        assert timing.decode_s == [1.0, 1.0]
        assert timing.detokenization_s == 3.0

    def test_requires_load(self, tmp_path: Path) -> None:
        session, _api = _session(tmp_path)
        with pytest.raises(LlamaSessionError, match="load"):
            session.generate_timed(_TEMPLATED_PROMPT, max_new_tokens=4)

    @pytest.mark.parametrize("max_new_tokens", [0, -1])
    def test_rejects_non_positive_max_new_tokens(self, tmp_path: Path, max_new_tokens: int) -> None:
        session, _api = _session(tmp_path)
        session.load(context_length=32)
        with pytest.raises(ValueError, match="positive"):
            session.generate_timed(_TEMPLATED_PROMPT, max_new_tokens=max_new_tokens)

    def test_rejects_empty_and_oversized_prompts(self, tmp_path: Path) -> None:
        session, _api = _session(tmp_path)
        session.load(context_length=len(_TEMPLATED_PROMPT))
        with pytest.raises(LlamaSessionError, match="empty"):
            session.generate_timed([], max_new_tokens=4)
        with pytest.raises(LlamaSessionError, match="capacity"):
            session.generate_timed(_TEMPLATED_PROMPT, max_new_tokens=4)

    def test_rejects_tokens_outside_vocabulary(self, tmp_path: Path) -> None:
        session, _api = _session(tmp_path)
        session.load(context_length=32)
        with pytest.raises(LlamaSessionError, match="vocabulary"):
            session.generate_timed([_VOCABULARY], max_new_tokens=4)


class TestClose:
    def test_close_releases_handles_and_is_idempotent(self, tmp_path: Path) -> None:
        session, api = _session(tmp_path)
        session.load(context_length=32)
        runtime = api.runtimes[0]
        pipeline, stage = runtime.pipeline, runtime.stage

        session.close()
        session.close()

        assert runtime.closed
        assert pipeline.closed
        assert stage.closed
        assert api.tokenizers[0].closed
        with pytest.raises(LlamaSessionError, match="load"):
            session.generate_timed(_TEMPLATED_PROMPT, max_new_tokens=4)


# ---------------------------------------------------------------------------
# Payload discovery
# ---------------------------------------------------------------------------


def _windowsml_dir(tmp_path: Path, *, adapter: bool, backends: tuple[str, ...]) -> Path:
    package = tmp_path / "windowsml"
    lib = package / "lib"
    lib.mkdir(parents=True)
    if adapter:
        (lib / "WinMLRuntimeLlama.dll").write_bytes(b"")
    for family in backends:
        (lib / "LlamaBackends" / family).mkdir(parents=True)
    return package


_NO_LLAMA: dict[str, str | None] = {
    "windowsml": "2.7.2021a0",
    "windowsml-llama-core": None,
    "windowsml-llama-cuda": None,
}
_CORE_ONLY: dict[str, str | None] = {**_NO_LLAMA, "windowsml-llama-core": "2.7.2021a0"}


class TestPayload:
    def test_discovers_adapter_and_backends(self, tmp_path: Path) -> None:
        package = _windowsml_dir(tmp_path, adapter=True, backends=("cuda", "cpu"))
        payload = _discover_llama_payload(package)
        assert payload.adapter_present
        assert payload.backends == ("cpu", "cuda")
        assert payload.lib_dir == package / "lib"

    def test_missing_payload_names_matching_core_version(self, tmp_path: Path) -> None:
        package = _windowsml_dir(tmp_path, adapter=False, backends=())
        with pytest.raises(LlamaNotInstalledError, match=r"windowsml-llama-core==2\.7\.2021a0"):
            _require_llama_backend("cpu", package_dir=package, versions=_NO_LLAMA)

    @pytest.mark.parametrize(
        ("adapter", "versions"),
        [(True, _NO_LLAMA), (False, _CORE_ONLY)],
        ids=["adapter-dll", "core-distribution"],
    )
    def test_either_signal_counts_as_installed(
        self, tmp_path: Path, adapter: bool, versions: dict[str, str | None]
    ) -> None:
        package = _windowsml_dir(tmp_path, adapter=adapter, backends=("cpu",))
        payload = _require_llama_backend("cpu", package_dir=package, versions=versions)
        assert payload.backends == ("cpu",)

    def test_gpu_requires_a_non_cpu_backend(self, tmp_path: Path) -> None:
        package = _windowsml_dir(tmp_path, adapter=True, backends=("cpu",))
        with pytest.raises(LlamaNotInstalledError, match="windowsml-llama-cuda"):
            _require_llama_backend("gpu", package_dir=package, versions=_CORE_ONLY)

    def test_gpu_accepts_any_gpu_backend_family(self, tmp_path: Path) -> None:
        package = _windowsml_dir(tmp_path, adapter=True, backends=("cpu", "vulkan"))
        payload = _require_llama_backend("gpu", package_dir=package, versions=_CORE_ONLY)
        assert "vulkan" in payload.backends

    def test_unknown_backend_layout_does_not_block_gpu(self, tmp_path: Path) -> None:
        package = _windowsml_dir(tmp_path, adapter=True, backends=())
        payload = _require_llama_backend("gpu", package_dir=package, versions=_CORE_ONLY)
        assert payload.backends == ()

    def test_package_versions_cover_every_llama_distribution(self) -> None:
        assert set(llama_package_versions()) == {
            "windowsml",
            "windowsml-llama-core",
            "windowsml-llama-cuda",
        }
