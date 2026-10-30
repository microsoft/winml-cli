# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for GGUF benchmarking in ``winml perf`` without native dependencies.

A fake session returning clock-generated :class:`LlamaGenerationTiming` objects
makes aggregation deterministic without importing Windows ML Tasks. The session is tested in
``tests/unit/session/test_llama_session.py``.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import os
from io import StringIO
from itertools import accumulate, count, pairwise
from pathlib import Path
from random import Random
from statistics import fmean
from typing import Any
from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner
from rich.console import Console

from tests.unit.test_helpers import (
    RUNTIME_CALLBACK,
    FakeLogApi,
    FakeOptions,
    HResultError,
    make_record,
)
from winml.modelkit.commands import _gguf_evidence
from winml.modelkit.commands import _perf_gguf as perf_gguf
from winml.modelkit.commands._gguf_evidence import rows_path, token_ids_sha256
from winml.modelkit.commands._gguf_native_log import NativeLogCapture
from winml.modelkit.commands._perf_gguf import (
    GGUFBenchmarkResult,
    GGUFEvidenceError,
    GGUFPerfBenchmark,
    GGUFPerfConfig,
    _sample_from_record,
    display_gguf_report,
    gguf_output_path,
    is_gguf_model,
    run_gguf_perf,
    write_gguf_report,
)
from winml.modelkit.commands.perf import _resolve_runtime, perf
from winml.modelkit.session import (
    LlamaGenerationTiming,
    LlamaLoadError,
    LlamaNotInstalledError,
    LlamaRequestRecord,
    LlamaSessionError,
)


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


def _timing(
    *,
    generated_tokens: int = 3,
    input_tokens: int = 4,
    scale: float = 1.0,
    stop_reason: str = "MAX_TOKENS",
) -> LlamaGenerationTiming:
    """Derive session spans from generated native-operation timestamps."""
    random = Random(generated_tokens + input_tokens)
    durations = [random.uniform(0.001, 0.1) * scale for _ in range(generated_tokens + 4)]
    reset_end, create_end, *after_create = accumulate(durations)
    arrivals = after_create[:generated_tokens]
    terminal, result_end = after_create[generated_tokens:]
    return LlamaGenerationTiming(
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        reset_s=reset_end,
        stream_create_s=create_end - reset_end,
        first_token_s=arrivals[0] - reset_end if arrivals else None,
        decode_s=[end - start for start, end in pairwise(arrivals)],
        completion_s=terminal - (arrivals[-1] if arrivals else create_end),
        result_fetch_s=result_end - terminal,
        generation_s=result_end - reset_end,
        response_text="".join(chr(ord("a") + index) for index in range(generated_tokens)),
        stop_reason=stop_reason,
    )


class _FakeLlamaSession:
    """WinMLLlamaSession stand-in returning generated timings in order."""

    def __init__(
        self,
        timings: list[LlamaGenerationTiming],
        *,
        prompt_ids: list[int] | None = None,
        chat_template: bool = True,
        effective_context: int | None = None,
        eos_readback: tuple[int, ...] | None = None,
        check_error: Exception | None = None,
    ) -> None:
        self._timings = list(timings)
        self._prompt_ids = prompt_ids if prompt_ids is not None else list(range(4))
        self._chat_template = chat_template
        self._effective_context = effective_context
        self._eos_readback = eos_readback
        self._check_error = check_error
        self.load_kwargs: dict[str, Any] = {}
        self.checked_ids: list[int] | None = None
        self.submitted_ids: list[list[int]] = []
        self.chat_template_applied: bool | None = None
        self.context_length: int | None = None
        self.effective_device: str | None = None
        self.payload = None
        self.load_timings_ms = {
            name: float(index)
            for index, name in enumerate(
                (
                    "runtime_create_duration_ms",
                    "tokenizer_create_duration_ms",
                    "model_create_duration_ms",
                    "pipeline_build_duration_ms",
                    "task_create_duration_ms",
                ),
                start=1,
            )
        }
        self.calls: list[str] = []
        self.generate_kwargs: list[dict[str, Any]] = []
        self.closed = False

    def open_tokenizer(self) -> None:
        self.calls.append("open_tokenizer")

    def encode_prompt(self, prompt: str, *, apply_template: bool = True) -> list[int]:
        self.calls.append("encode_prompt")
        self.chat_template_applied = apply_template and self._chat_template
        return list(self._prompt_ids)

    def load(
        self, *, context_length: int | None = None, eos_token_ids: tuple[int, ...] | None = None
    ) -> None:
        self.calls.append("load")
        self.load_kwargs = {"context_length": context_length, "eos_token_ids": eos_token_ids}
        self.context_length = (
            self._effective_context if self._effective_context is not None else context_length
        )
        self.effective_device = "cpu"

    @property
    def task_default_eos_token_ids(self) -> tuple[int, ...]:
        if self._eos_readback is not None:
            return self._eos_readback
        return () if self.load_kwargs.get("eos_token_ids") == () else (7,)

    def check_prompt_ids(self, token_ids: list[int]) -> None:
        self.checked_ids = list(token_ids)
        if self._check_error is not None:
            raise self._check_error

    def generate_timed(self, prompt_token_ids: list[int], **kwargs: Any) -> LlamaGenerationTiming:
        self.generate_kwargs.append(kwargs)
        self.submitted_ids.append(list(prompt_token_ids))
        return self._timings.pop(0)

    def close(self) -> None:
        self.closed = True


def _config(tmp_path: Path, **overrides: Any) -> GGUFPerfConfig:
    values: dict[str, Any] = {
        "model_path": tmp_path / "tiny.gguf",
        "warmup": 1,
        "iterations": 2,
        "max_new_tokens": 3,
        "output_path": tmp_path / "report.json",
    }
    values.update(overrides)
    return GGUFPerfConfig(**values)


def _run_fake(
    tmp_path: Path, timings: list[LlamaGenerationTiming], **overrides: Any
) -> tuple[GGUFBenchmarkResult, _FakeLlamaSession]:
    session = _FakeLlamaSession(timings)
    ticks = count(step=0.001)
    benchmark = GGUFPerfBenchmark(
        _config(tmp_path, **overrides), session=session, clock=lambda: next(ticks)
    )
    return benchmark.run(), session


def _gguf(tmp_path: Path, name: str = "model.gguf") -> Path:
    model = tmp_path / name
    model.write_bytes(b"GGUF")
    return model


def _ids_file(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "ids.json"
    path.write_bytes(content)
    return path


_INITIALIZATION_SPANS = [
    (name, 100 + 10 * index, 110 + 10 * index)
    for index, name in enumerate(
        (
            "runtime_create",
            "tokenizer_open",
            "model_load",
            "pipeline_build",
            "task_create",
            "session_create",
        )
    )
]


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


class _FakeProtocolSession(_FakeLlamaSession):
    """Session double for evidence runs; ``failures`` maps a request position to a fault.

    ``ctrl_c_in_log`` ("load" or "warmup-0") makes llama.cpp log one line at that point
    while a Ctrl+C arrives, so the capture's log callback runs into the KeyboardInterrupt.
    """

    def __init__(
        self,
        *,
        failures: dict[int, str] | None = None,
        log_api: FakeLogApi | None = None,
        evidence_path: Path | None = None,
        effective_context: int | None = None,
        eos_readback: tuple[int, ...] | None = None,
        ctrl_c_in_log: str | None = None,
    ) -> None:
        super().__init__([], effective_context=effective_context, eos_readback=eos_readback)
        self._failures = dict(failures or {})
        self._ctrl_c_in_log = ctrl_c_in_log
        self.ctrl_c_pending = False
        self._log_api = log_api
        self._evidence_path = evidence_path
        self.initialization_spans = list(_INITIALIZATION_SPANS)
        self.declared_context_length = None
        self.tokenizer_eos_token_ids = (7,)
        self.speculative_readback = {
            "model_draft_predictor_enabled": False,
            "draft_token_limit_hint": 0,
            "draft_predictor_path": None,
            "reason": None,
        }
        self.resolved_target_kind = "cpu"
        self.callback_at_load: int | None = None
        self.callback_at_close: int | None = None
        self.callback_at_request: list[int | None] = []
        self.gc_enabled_at_request: list[bool] = []
        self.rows_at_request: list[int | None] = []

    def _log_callback(self) -> int | None:
        return self._log_api.callback if self._log_api is not None else None

    def _log_during_ctrl_c(self, point: str) -> None:
        if self._ctrl_c_in_log == point and self._log_api is not None:
            self.ctrl_c_pending = True
            self._log_api.emit(2, b"llama_context: n_ctx = 16\n")

    def load(self, **kwargs: Any) -> None:
        super().load(**kwargs)
        self.callback_at_load = self._log_callback()
        self._log_during_ctrl_c("load")

    def greedy_options(self, max_new_tokens: int) -> FakeOptions:
        return FakeOptions(max_new_tokens=max_new_tokens)

    def run_request(
        self, token_ids: list[int], *, options: FakeOptions, clock_ns: Any
    ) -> LlamaRequestRecord:
        position = len(self.gc_enabled_at_request)
        self.calls.append("run_request")
        self.gc_enabled_at_request.append(gc.isenabled())
        self.callback_at_request.append(self._log_callback())
        evidence = self._evidence_path
        self.rows_at_request.append(_line_count(rows_path(evidence)) if evidence else None)
        failure = self._failures.get(position)
        if failure == "raise":
            raise RuntimeError("native session crashed")
        if failure == "interrupt":
            raise KeyboardInterrupt
        record = make_record(
            input_ids=token_ids, tokens=range(options.max_new_tokens), start_ns=clock_ns()
        )
        if position == 0:
            self._log_during_ctrl_c("warmup-0")
        if failure == "error":
            return dataclasses.replace(
                record, finish_reason="ERROR", error=HResultError(0x80004005)
            )
        if failure == "reset":
            return LlamaRequestRecord(
                input_ids=tuple(token_ids),
                reset_start_ns=record.reset_start_ns,
                reset_error=HResultError(0x80004005),
            )
        return record

    def close(self) -> None:
        super().close()
        self.callback_at_close = self._log_callback()


def _protocol_benchmark(
    tmp_path: Path,
    *,
    failures: dict[int, str] | None = None,
    effective_context: int | None = None,
    eos_readback: tuple[int, ...] | None = None,
    owned: bool = False,
    ctrl_c_in_log: str | None = None,
    **config_overrides: Any,
) -> tuple[GGUFPerfBenchmark, _FakeProtocolSession, FakeLogApi]:
    """Build an evidence benchmark; ``owned`` leaves session creation to the benchmark."""
    values: dict[str, Any] = {
        "prompt_ids": (1, 2, 3, 4),
        "context_length": 16,
        "ignore_eos": True,
        "warmup": 1,
        "iterations": 2,
        "max_new_tokens": 3,
        "evidence_path": tmp_path / "ev.json",
        **config_overrides,
    }
    config = _config(tmp_path, **values)
    config.model_path.write_bytes(b"GGUF")
    api = FakeLogApi()
    session = _FakeProtocolSession(
        failures=failures,
        log_api=api,
        evidence_path=config.evidence_path,
        effective_context=effective_context,
        eos_readback=eos_readback,
        ctrl_c_in_log=ctrl_c_in_log,
    )
    ticks = count(1_000, 1_000)

    def clock_ns() -> int:
        return next(ticks)

    def capture_clock_ns() -> int:
        if session.ctrl_c_pending:
            session.ctrl_c_pending = False
            raise KeyboardInterrupt
        return clock_ns()

    benchmark = GGUFPerfBenchmark(
        config,
        session=None if owned else session,
        clock=lambda: clock_ns() / 1e9,
        clock_ns=clock_ns,
        log_capture=NativeLogCapture(
            load_api=lambda: ("C:/lib/WinMLllama.dll", api), clock_ns=capture_clock_ns
        ),
    )
    return benchmark, session, api


def _protocol_run(
    tmp_path: Path, **kwargs: Any
) -> tuple[dict[str, Any], _FakeProtocolSession, FakeLogApi]:
    benchmark, session, api = _protocol_benchmark(tmp_path, **kwargs)
    benchmark.run()
    return _read(tmp_path / "ev.json"), session, api


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _explode(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("must not be called on the evidence path")


# ---------------------------------------------------------------------------
# Benchmark engine
# ---------------------------------------------------------------------------


ADDITIVE_KEYS = {
    "input_mode",
    "prompt_ids_sha256",
    "context_length_requested",
    "eos_policy",
    "evidence_path",
    "evidence_status",
}


class TestProtocolOptions:
    def test_prompt_ids_skip_encoding_and_size_the_context(self, tmp_path: Path) -> None:
        ids = (5, 6, 7, 8, 9)
        result, session = _run_fake(
            tmp_path,
            [_timing(input_tokens=len(ids)) for _ in range(3)],
            prompt_ids=ids,
            apply_template=False,
        )
        assert "encode_prompt" not in session.calls
        assert session.submitted_ids == [list(ids)] * 3
        assert session.load_kwargs == {
            "context_length": len(ids) + result.config.max_new_tokens,
            "eos_token_ids": None,
        }
        assert session.checked_ids == list(ids)
        assert all(sample.tokenization_duration_ms == 0.0 for sample in result.requests)

    def test_context_length_overrides_the_derived_capacity(self, tmp_path: Path) -> None:
        _, session = _run_fake(tmp_path, [_timing()] * 3, context_length=64)
        assert session.load_kwargs["context_length"] == 64

    def test_context_length_mismatch_fails_before_any_request(self, tmp_path: Path) -> None:
        session = _FakeLlamaSession([_timing()] * 3, effective_context=32)
        with pytest.raises(LlamaLoadError, match="--context-length 64"):
            GGUFPerfBenchmark(_config(tmp_path, context_length=64), session=session).run()
        assert session.generate_kwargs == []

    def test_prompt_longer_than_context_length_fails_before_load(self, tmp_path: Path) -> None:
        session = _FakeLlamaSession([_timing()] * 3, prompt_ids=list(range(8)))
        with pytest.raises(LlamaSessionError, match="--context-length 10"):
            GGUFPerfBenchmark(
                _config(tmp_path, context_length=10, max_new_tokens=3), session=session
            ).run()
        assert "load" not in session.calls

    def test_ignore_eos_requests_and_verifies_an_empty_set(self, tmp_path: Path) -> None:
        _, session = _run_fake(tmp_path, [_timing()] * 3, ignore_eos=True)
        assert session.load_kwargs["eos_token_ids"] == ()
        bad = _FakeLlamaSession([_timing()] * 3, eos_readback=(7,))
        with pytest.raises(LlamaLoadError, match="--ignore-eos"):
            GGUFPerfBenchmark(_config(tmp_path, ignore_eos=True), session=bad).run()
        assert bad.generate_kwargs == []

    def test_out_of_vocabulary_ids_fail_before_any_request(self, tmp_path: Path) -> None:
        session = _FakeLlamaSession(
            [_timing()] * 3, check_error=LlamaSessionError("outside the model vocabulary")
        )
        with pytest.raises(LlamaSessionError, match="vocabulary"):
            GGUFPerfBenchmark(_config(tmp_path, prompt_ids=(1, 2)), session=session).run()
        assert session.generate_kwargs == []

    def test_schema3_adds_fields_only_with_protocol_options(self, tmp_path: Path) -> None:
        stock = _run_fake(tmp_path, [_timing()] * 3)[0].to_dict()["benchmark_info"]
        ids = (1, 2, 3, 4)
        protocol = _run_fake(
            tmp_path,
            [_timing()] * 3,
            prompt_ids=ids,
            context_length=16,
            ignore_eos=True,
            apply_template=False,
        )[0].to_dict()["benchmark_info"]
        assert ADDITIVE_KEYS.isdisjoint(stock)
        assert set(protocol) == set(stock) | ADDITIVE_KEYS
        assert {key: protocol[key] for key in ADDITIVE_KEYS} == {
            "input_mode": "token_ids",
            "prompt_ids_sha256": token_ids_sha256(ids),
            "context_length_requested": 16,
            "eos_policy": "ignore",
            "evidence_path": None,
            "evidence_status": None,
        }
        assert protocol["prompt"] is None
        assert stock["prompt"] == GGUFPerfConfig(model_path=Path("m.gguf")).prompt

    def test_text_mode_reports_the_digest_of_the_encoded_prompt(self, tmp_path: Path) -> None:
        info = _run_fake(tmp_path, [_timing()] * 3, context_length=64)[0].to_dict()
        benchmark_info = info["benchmark_info"]
        assert benchmark_info["input_mode"] == "text"
        assert benchmark_info["prompt_ids_sha256"] == token_ids_sha256(range(4))
        assert benchmark_info["prompt"] == GGUFPerfConfig(model_path=Path("m.gguf")).prompt


class TestEvidenceRun:
    def test_complete_run_writes_contract_evidence(self, tmp_path: Path) -> None:
        doc, session, _ = _protocol_run(tmp_path)

        assert (doc["status"], doc["failure_phase"], doc["error"]) == ("complete", None, None)
        assert [row["id"] for row in doc["requests"]] == ["warmup-0", "measured-0", "measured-1"]
        assert all(
            row["status"] == "succeeded" and row["stop_reason"] == "limit"
            for row in doc["requests"]
        )
        assert session.calls[:2] == ["open_tokenizer", "load"]
        assert session.load_kwargs == {"context_length": 16, "eos_token_ids": ()}
        assert [span["name"] for span in doc["initialization"]["spans"]] == [
            name for name, _, _ in session.initialization_spans
        ]
        assert doc["requests"] == [
            json.loads(line)
            for line in rows_path(tmp_path / "ev.json").read_text(encoding="utf-8").splitlines()
        ]

    def test_run_returns_schema3_samples_and_status(self, tmp_path: Path) -> None:
        benchmark, _, _ = _protocol_benchmark(tmp_path)
        result = benchmark.run()

        assert result.evidence_status == "complete"
        assert [(sample.kind, sample.index) for sample in result.requests] == [
            ("warmup", 0),
            ("timed", 1),
            ("timed", 2),
        ]
        assert result.aggregate["stop_reason_counts"] == {"MAX_TOKENS": 2}
        info = result.to_dict()["benchmark_info"]
        assert info["evidence_status"] == "complete"
        assert info["prompt_ids_sha256"] == token_ids_sha256((1, 2, 3, 4))

    def test_provisional_evidence_exists_before_model_hashing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}
        real = _gguf_evidence.model_inventory

        def inventory(path: Path) -> dict[str, Any]:
            seen.update(_read(tmp_path / "ev.json"))
            return real(path)

        monkeypatch.setattr(_gguf_evidence, "model_inventory", inventory)
        _protocol_run(tmp_path)

        assert seen["status"] == "running"
        assert seen["process"]["pid"] == os.getpid()

    def test_log_capture_spans_load_through_warmup0_only(self, tmp_path: Path) -> None:
        doc, session, _ = _protocol_run(tmp_path)

        assert session.callback_at_load not in (None, RUNTIME_CALLBACK)
        assert session.callback_at_request == [
            session.callback_at_load,
            RUNTIME_CALLBACK,
            RUNTIME_CALLBACK,
        ]
        assert doc["native_log"]["after_capture"]["restored"] is True

    def test_gc_is_disabled_only_for_measured_requests(self, tmp_path: Path) -> None:
        doc, session, _ = _protocol_run(tmp_path)

        assert session.gc_enabled_at_request == [True, False, False]
        assert gc.isenabled()
        assert doc["harness"]["gc"]["collections_between_requests"] == 2

    def test_rows_reach_disk_before_the_next_request(self, tmp_path: Path) -> None:
        _, session, _ = _protocol_run(tmp_path)

        assert session.rows_at_request == [0, 1, 2]

    def test_text_prompt_is_encoded_once(self, tmp_path: Path) -> None:
        doc, session, _ = _protocol_run(tmp_path, prompt_ids=None, context_length=None)

        assert session.calls.count("encode_prompt") == 1
        assert doc["workload"]["input_mode"] == "text"
        assert all(row["input_ids"] == session._prompt_ids for row in doc["requests"])

    def test_memory_tracker_is_never_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(perf_gguf, "_GenerationMemoryTracker", _explode)

        assert _protocol_run(tmp_path, memory=True)[0]["status"] == "complete"

    def test_schema3_samples_derive_from_qpc_records(self) -> None:
        record = make_record()
        sample = _sample_from_record(record, kind="timed", index=1)
        assert record.submit_ns is not None
        assert record.stream_created_ns is not None
        assert record.completed_ns is not None
        assert record.result_fetched_ns is not None
        assert record.reset_end_ns is not None

        assert (sample.kind, sample.index) == ("timed", 1)
        assert sample.prompt_tokens == record.prompt_token_count
        assert sample.generated_tokens == len(record.events)
        assert (sample.template_duration_ms, sample.tokenization_duration_ms) == (0.0, 0.0)
        assert sample.generator_create_duration_ms == (
            (record.reset_end_ns - record.reset_start_ns) / 1e6
        )
        assert sample.task_stream_create_duration_ms == (
            (record.stream_created_ns - record.submit_ns) / 1e6
        )
        assert sample.task_ttft_duration_ms == (record.events[0][0] - record.submit_ns) / 1e6
        assert sample.decode_token_durations_ms == [
            (later[0] - earlier[0]) / 1e6 for earlier, later in pairwise(record.events)
        ]
        assert sample.task_stream_completion_duration_ms == (
            (record.completed_ns - record.events[-1][0]) / 1e6
        )
        assert sample.result_fetch_duration_ms == (
            (record.result_fetched_ns - record.completed_ns) / 1e6
        )
        assert sample.task_generation_duration_ms == (
            (record.result_fetched_ns - record.submit_ns) / 1e6
        )
        assert sample.response_text == record.response_text

    def test_sample_without_updates_has_no_ttft(self) -> None:
        record = make_record(tokens=())
        sample = _sample_from_record(record, kind="warmup", index=0)
        assert record.completed_ns is not None
        assert record.stream_created_ns is not None

        assert sample.task_ttft_duration_ms is None
        assert sample.decode_token_durations_ms == []
        assert sample.task_stream_completion_duration_ms == (
            (record.completed_ns - record.stream_created_ns) / 1e6
        )


class TestEvidenceFailures:
    def test_failed_request_is_retained_and_later_requests_run(self, tmp_path: Path) -> None:
        doc, _, _ = _protocol_run(tmp_path, failures={1: "error"})

        assert doc["status"] == "partial"
        assert [row["status"] for row in doc["requests"]] == ["succeeded", "failed", "succeeded"]

    def test_reset_failure_marks_the_remaining_requests_incomplete(self, tmp_path: Path) -> None:
        with pytest.raises(GGUFEvidenceError, match="during measured"):
            _protocol_run(tmp_path, failures={1: "reset"})

        doc = _read(tmp_path / "ev.json")
        assert (doc["status"], doc["failure_phase"]) == ("failed", "measured")
        assert [row["status"] for row in doc["requests"]] == [
            "succeeded",
            "incomplete",
            "incomplete",
        ]
        assert doc["requests"][1]["error"]["hresult"] == "0x80004005"

    def test_context_mismatch_records_both_values_then_fails(self, tmp_path: Path) -> None:
        with pytest.raises(GGUFEvidenceError, match="during initialization"):
            _protocol_run(tmp_path, effective_context=32)

        doc = _read(tmp_path / "ev.json")
        assert doc["configuration"]["context"] == {
            "requested": 16,
            "effective": 32,
            "declared": None,
        }
        assert {row["status"] for row in doc["requests"]} == {"incomplete"}

    def test_missing_shard_is_a_configuration_failure(self, tmp_path: Path) -> None:
        benchmark, session, _ = _protocol_benchmark(
            tmp_path, model_path=tmp_path / "m-00001-of-00002.gguf"
        )

        with pytest.raises(GGUFEvidenceError, match="during configuration"):
            benchmark.run()

        assert _read(tmp_path / "ev.json")["failure_phase"] == "configuration"
        assert session.calls == []

    def test_unexpected_exception_restores_runtime_callback_before_close(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        benchmark, session, _ = _protocol_benchmark(tmp_path, failures={0: "raise"}, owned=True)
        monkeypatch.setattr(perf_gguf, "WinMLLlamaSession", lambda *_a, **_k: session)

        with pytest.raises(GGUFEvidenceError, match="during warmup"):
            benchmark.run()

        assert session.closed
        assert session.callback_at_close == RUNTIME_CALLBACK

    def test_error_names_a_cause_that_has_no_message(self) -> None:
        error = GGUFEvidenceError("warmup", RuntimeError(), Path("ev.json"))

        assert str(error) == (
            "GGUF evidence run failed during warmup: RuntimeError Evidence: ev.json"
        )

    def test_ctrl_c_during_load_logging_stops_the_run_after_load(self, tmp_path: Path) -> None:
        benchmark, session, api = _protocol_benchmark(tmp_path, ctrl_c_in_log="load")

        with pytest.raises(KeyboardInterrupt):
            benchmark.run()

        doc = _read(tmp_path / "ev.json")
        assert (doc["status"], doc["failure_phase"]) == ("failed", "initialization")
        assert doc["error"]["type"] == "KeyboardInterrupt"
        assert {row["status"] for row in doc["requests"]} == {"incomplete"}
        assert "run_request" not in session.calls
        assert api.callback == RUNTIME_CALLBACK

    def test_ctrl_c_during_warmup0_logging_keeps_its_row_and_stops(self, tmp_path: Path) -> None:
        benchmark, session, api = _protocol_benchmark(tmp_path, ctrl_c_in_log="warmup-0")

        with pytest.raises(KeyboardInterrupt):
            benchmark.run()

        doc = _read(tmp_path / "ev.json")
        assert (doc["status"], doc["failure_phase"]) == ("failed", "warmup")
        assert [row["status"] for row in doc["requests"]] == [
            "succeeded",
            "incomplete",
            "incomplete",
        ]
        assert session.gc_enabled_at_request == [True]
        assert api.callback == RUNTIME_CALLBACK

    def test_interrupt_writes_failed_evidence_and_propagates(self, tmp_path: Path) -> None:
        with pytest.raises(KeyboardInterrupt):
            _protocol_run(tmp_path, failures={1: "interrupt"})

        assert _read(tmp_path / "ev.json")["status"] == "failed"
        assert gc.isenabled()


class TestEvidenceCli:
    @pytest.mark.parametrize(
        ("failures", "exit_code", "status"),
        [({}, 0, "complete"), ({1: "error"}, 1, "partial"), ({1: "reset"}, 1, "failed")],
    )
    def test_exit_code_follows_the_evidence_status(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        failures: dict[int, str],
        exit_code: int,
        status: str,
    ) -> None:
        evidence = tmp_path / "ev.json"
        report = tmp_path / "report.json"
        session = _FakeProtocolSession(failures=failures, evidence_path=evidence)
        monkeypatch.setattr(perf_gguf, "WinMLLlamaSession", lambda *_a, **_k: session)
        args = [
            "-m",
            str(_gguf(tmp_path)),
            "--prompt-ids-file",
            str(_ids_file(tmp_path, b"[1, 2, 3, 4]")),
            "--context-length",
            "16",
            "--ignore-eos",
            "--evidence",
            str(evidence),
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--max-new-tokens",
            "3",
            "-o",
            str(report),
        ]
        result = CliRunner().invoke(perf, args)

        assert result.exit_code == exit_code, result.output
        assert _read(evidence)["status"] == status
        assert report.exists() is (status != "failed")
        assert f"(status: {status})" in result.output
        assert session.closed


class TestBenchmark:
    def test_sizes_context_from_prompt_and_loads_before_generating(self, tmp_path: Path) -> None:
        timings = [_timing() for _ in range(3)]
        result, session = _run_fake(tmp_path, timings)

        assert session.calls[:4] == ["open_tokenizer", "encode_prompt", "load", "encode_prompt"]
        assert session.context_length == timings[0].input_tokens + result.config.max_new_tokens
        assert all(
            kwargs["max_new_tokens"] == result.config.max_new_tokens
            for kwargs in session.generate_kwargs
        )
        assert not session.closed  # injected sessions belong to the caller

    def test_aggregates_timed_requests_only(self, tmp_path: Path) -> None:
        timings = [
            _timing(scale=100.0, stop_reason="EOS"),  # warmup, excluded
            *[_timing(scale=float(index)) for index in range(1, 3)],
        ]
        result, _session = _run_fake(tmp_path, timings)
        aggregate = result.aggregate

        assert aggregate["warmup_request_count"] == 1
        assert aggregate["timed_request_count"] == 2
        measured = timings[result.config.warmup :]
        assert aggregate["task_ttft_duration_ms"]["mean"] == pytest.approx(
            fmean(timing.first_token_s for timing in measured) * 1000
        )
        assert aggregate["task_generation_duration_ms"]["mean"] == pytest.approx(
            fmean(timing.generation_s for timing in measured) * 1000
        )
        assert aggregate["steady_state_tpot_ms"]["mean"] == pytest.approx(
            fmean(fmean(timing.decode_s) for timing in measured) * 1000
        )
        assert aggregate["steady_state_decode_tokens_per_second"]["mean"] == pytest.approx(
            fmean(len(timing.decode_s) / sum(timing.decode_s) for timing in measured)
        )
        assert aggregate["stop_reason_counts"] == {"MAX_TOKENS": 2}
        assert result.stop_reason == "MAX_TOKENS"
        assert (result.prompt_tokens, result.generated_tokens) == (
            measured[0].input_tokens,
            measured[0].generated_tokens,
        )

    def test_load_metrics_come_from_session_spans(self, tmp_path: Path) -> None:
        timings = [_timing(generated_tokens=1) for _ in range(3)]
        result, session = _run_fake(tmp_path, timings)

        for name, duration in session.load_timings_ms.items():
            assert result.load[name] == duration
        assert result.load["native_load_duration_ms"] == sum(
            session.load_timings_ms[name]
            for name in ("model_create_duration_ms", "pipeline_build_duration_ms")
        )
        assert result.load["weight_upload_duration_ms"] is None
        assert result.load["weight_upload_estimate_source"] == "unavailable"

    def test_request_timing_includes_stream_startup_and_terminal_work(self, tmp_path: Path) -> None:
        clock_step_s = 0.125
        ticks = count(step=clock_step_s)
        timing = _timing()
        session = _FakeLlamaSession([timing] * 3)
        benchmark = GGUFPerfBenchmark(_config(tmp_path), session=session, clock=lambda: next(ticks))
        result = benchmark.run()
        sample = result.requests[-1]

        assert sample.template_duration_ms == 0.0
        assert sample.tokenization_duration_ms == pytest.approx(clock_step_s * 1000)
        assert sample.generator_create_duration_ms == pytest.approx(timing.reset_s * 1000)
        assert sample.task_ttft_duration_ms == pytest.approx(timing.first_token_s * 1000)
        assert sample.task_generation_duration_ms == pytest.approx(timing.generation_s * 1000)
        assert sample.task_stream_create_duration_ms == pytest.approx(timing.stream_create_s * 1000)
        assert sample.task_stream_completion_duration_ms == pytest.approx(
            timing.completion_s * 1000
        )
        assert sample.result_fetch_duration_ms == pytest.approx(timing.result_fetch_s * 1000)
        assert sample.decode_token_durations_ms == pytest.approx(
            [duration * 1000 for duration in timing.decode_s]
        )
        assert sample.request_ttft_duration_ms == pytest.approx(
            (clock_step_s + timing.reset_s + timing.first_token_s) * 1000
        )
        assert sample.request_duration_ms == pytest.approx(
            (clock_step_s + timing.reset_s + timing.generation_s) * 1000
        )
        assert result.aggregate["cold_start_ttft_duration_ms"] == pytest.approx(
            result.load["session_load_duration_ms"] + result.requests[0].request_ttft_duration_ms
        )
        assert result.aggregate["cold_start_total_duration_ms"] == pytest.approx(
            result.load["session_load_duration_ms"] + result.requests[0].request_duration_ms
        )

    @pytest.mark.parametrize(
        "metric",
        [
            "prefill_duration_ms",
            "prefill_tokens_per_second",
            "first_token_duration_ms",
            "detokenization_duration_ms",
            "model_ttft_duration_ms",
            "model_compute_duration_ms",
            "response_eval_duration_ms",
            "response_eval_tokens_per_second",
            "sequence_fetch_duration_ms",
        ],
    )
    def test_unobservable_model_phases_are_null(self, tmp_path: Path, metric: str) -> None:
        result, _session = _run_fake(tmp_path, [_timing()] * 3)
        report = result.to_dict()

        assert report["aggregate"][metric] is None
        assert all(sample[metric] is None for sample in report["requests"])

    @pytest.mark.parametrize("generated_tokens", range(2))
    def test_no_decode_intervals_leave_decode_metrics_null(
        self, tmp_path: Path, generated_tokens: int
    ) -> None:
        timing = _timing(generated_tokens=generated_tokens, stop_reason="EOS")
        result, _session = _run_fake(tmp_path, [timing] * 3)
        report = result.to_dict()
        sample = report["requests"][-1]

        assert sample["decode_token_durations_ms"] == []
        for metric in ("steady_state_tpot_ms", "steady_state_decode_tokens_per_second"):
            assert sample[metric] is None
            assert report["aggregate"][metric] is None
        if generated_tokens == 0:
            assert sample["task_ttft_duration_ms"] is None
            assert sample["request_ttft_duration_ms"] is None
            assert report["aggregate"]["task_ttft_duration_ms"] is None
            assert report["aggregate"]["request_ttft_duration_ms"] is None
            assert report["aggregate"]["cold_start_ttft_duration_ms"] is None
        assert result.requests[-1].request_duration_ms == pytest.approx(
            (0.001 + timing.reset_s + timing.generation_s) * 1000
        )

    def test_unavailable_observations_do_not_dilute_timed_averages(self, tmp_path: Path) -> None:
        timings = [_timing(generated_tokens=index) for index in range(3)]
        result, _session = _run_fake(tmp_path, timings, warmup=0, iterations=len(timings))

        assert result.aggregate["task_ttft_duration_ms"]["mean"] == pytest.approx(
            fmean(timing.first_token_s for timing in timings[1:]) * 1000
        )
        assert result.aggregate["steady_state_tpot_ms"]["mean"] == pytest.approx(
            fmean(timings[-1].decode_s) * 1000
        )
        assert result.aggregate["cold_start_ttft_duration_ms"] is None

    def test_warmup_only_run_does_not_become_timed_data(self, tmp_path: Path) -> None:
        result, _session = _run_fake(tmp_path, [_timing()], warmup=1, iterations=0)

        assert result.aggregate["timed_request_count"] == 0
        assert result.aggregate["warmup_request_count"] == 1
        assert result.aggregate["task_generation_duration_ms"] is None
        assert result.aggregate["request_duration_ms"] is None
        assert result.aggregate["stop_reason_counts"] == {}

    def test_owned_session_is_closed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _FakeLlamaSession([_timing(generated_tokens=1)] * 3)
        monkeypatch.setattr(perf_gguf, "WinMLLlamaSession", lambda *_args, **_kwargs: session)

        GGUFPerfBenchmark(_config(tmp_path)).run()

        assert session.closed

    def test_memory_profile_tracks_ram_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rss = iter([100.0, 400.0, 420.0])
        monkeypatch.setattr(
            "winml.modelkit.commands._generation_metrics._get_rss_mb", lambda: next(rss)
        )
        timings = [_timing(generated_tokens=1) for _ in range(3)]
        result, _session = _run_fake(tmp_path, timings, memory=True)

        assert result.memory_profile is not None
        assert result.memory_profile["rss_model_load_delta_mb"] == 300.0
        assert not any(key.startswith("vram") for key in result.memory_profile)


class TestResultToDict:
    def test_reports_runtime_backend_and_format(self, tmp_path: Path) -> None:
        timings = [_timing() for _ in range(3)]
        result, _session = _run_fake(tmp_path, timings, requested_device="auto")
        info = result.to_dict()["benchmark_info"]

        assert info["runtime"] == "winml-runtime"
        assert info["backend"] == "llama.cpp"
        assert info["api"] == "windowsml.tasks"
        assert info["timing_source"] == "task-stream"
        assert info["model_format"] == "gguf"
        assert info["experimental"] is True
        assert info["ep"] is None
        assert (info["device"], info["effective_device"]) == ("auto", "cpu")
        assert info["context_length"] == 7
        assert info["chat_template_applied"] is True
        assert set(info["packages"]) == {
            "windowsml",
            "windowsml-llama-core",
            "windowsml-llama-cuda",
        }

    def test_retains_report_layout_with_task_timing_schema(self, tmp_path: Path) -> None:
        timings = [_timing() for _ in range(3)]
        result, _session = _run_fake(tmp_path, timings)
        report = result.to_dict()

        assert report["schema_version"] == 3
        assert {"load", "requests", "aggregate", "response_text"} <= set(report)
        assert len(report["requests"]) == 3
        assert report["requests"][0]["kind"] == "warmup"
        json.dumps(report)  # JSON-serializable


# ---------------------------------------------------------------------------
# Reporting and entry point
# ---------------------------------------------------------------------------


class TestReporting:
    def test_console_report_shows_task_metrics_and_device_resolution(self, tmp_path: Path) -> None:
        timings = [_timing() for _ in range(3)]
        result, _session = _run_fake(tmp_path, timings, requested_device="auto")
        buffer = StringIO()
        display_gguf_report(result, Console(file=buffer, width=200, color_system=None))
        output = buffer.getvalue()

        assert "llama.cpp, experimental" in output
        assert "auto -> cpu" in output
        assert "API:" in output
        assert "Windows ML Tasks" in output
        assert "Task TTFT" in output
        assert "Pure prefill: unavailable" in output
        assert "Decode (Task stream):" in output

    def test_console_report_warns_when_nothing_was_generated(self, tmp_path: Path) -> None:
        empty = _timing(generated_tokens=0, stop_reason="EOS")
        result, _session = _run_fake(tmp_path, [empty] * 3)
        buffer = StringIO()
        display_gguf_report(result, Console(file=buffer, width=200, color_system=None))

        output = buffer.getvalue()
        assert "stopped before generating a token" in output
        assert "Cold TTFT: n/a" in output
        assert "n/a tokens/sec" in output
        assert "n/a ms/token" in output

    def test_write_report_round_trips(self, tmp_path: Path) -> None:
        timings = [_timing() for _ in range(3)]
        result, _session = _run_fake(tmp_path, timings)
        path = tmp_path / "nested" / "report.json"
        write_gguf_report(result, path)

        assert json.loads(path.read_text(encoding="utf-8"))["benchmark_info"]["backend"] == (
            "llama.cpp"
        )

    def test_output_path_uses_model_stem(self) -> None:
        path = gguf_output_path(Path("models") / "qwen2.5-0.5b-instruct-q4_k_m.gguf")
        assert path.parent.name == "qwen2.5-0.5b-instruct-q4_k_m"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("model.gguf", True), ("MODEL.GGUF", True), ("model.onnx", False), ("org/model", False)],
    )
    def test_is_gguf_model(self, value: str, expected: bool) -> None:
        assert is_gguf_model(value) is expected


class TestRunGGUFPerf:
    @pytest.mark.parametrize(
        "error",
        (LlamaNotInstalledError, LlamaLoadError, LlamaSessionError, FileNotFoundError),
    )
    def test_session_failures_become_click_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[Exception]
    ) -> None:
        def fail(_self: GGUFPerfBenchmark) -> GGUFBenchmarkResult:
            raise error(error.__name__)

        monkeypatch.setattr(GGUFPerfBenchmark, "run", fail)
        console = Console(file=StringIO())
        with pytest.raises(click.ClickException, match=error.__name__):
            run_gguf_perf(_config(tmp_path), console=console, json_mode=False)

    def test_json_mode_prints_and_saves_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        timings = [_timing() for _ in range(3)]
        result, _session = _run_fake(tmp_path, timings)
        monkeypatch.setattr(GGUFPerfBenchmark, "run", lambda _self: result)
        config = _config(tmp_path)

        run_gguf_perf(config, console=Console(file=StringIO()), json_mode=True)

        assert json.loads(capsys.readouterr().out)["benchmark_info"]["model_format"] == "gguf"
        assert config.output_path is not None
        assert config.output_path.is_file()


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


class TestCliDispatch:
    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    @pytest.fixture
    def capture_run(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        """Patch run_gguf_perf and capture the GGUFPerfConfig it receives."""
        captured: dict = {}

        def fake_run(config: GGUFPerfConfig, *, console: Console, json_mode: bool) -> MagicMock:
            captured["config"] = config
            captured["json_mode"] = json_mode
            return MagicMock()

        monkeypatch.setattr(perf_gguf, "run_gguf_perf", fake_run)
        return captured

    def test_auto_runtime_dispatches_gguf_with_generation_defaults(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        model = _gguf(tmp_path)
        result = runner.invoke(perf, ["-m", str(model)])

        assert result.exit_code == 0, result.output
        config = capture_run["config"]
        assert config.model_path == model
        assert (config.device, config.requested_device) == ("cpu", "auto")
        assert (config.iterations, config.warmup, config.max_new_tokens) == (10, 2, 128)
        assert config.apply_template is True
        assert config.output_path.parent.name == "model"

    def test_explicit_options_are_forwarded(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        model = _gguf(tmp_path)
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Name three primary colors.", encoding="utf-8")
        output = tmp_path / "out.json"
        result = runner.invoke(
            perf,
            [
                "-m",
                str(model),
                "--runtime",
                "winml-runtime",
                "--device",
                "gpu",
                "--prompt-file",
                str(prompt_file),
                "--no-apply-template",
                "--max-new-tokens",
                "16",
                "--iterations",
                "3",
                "--warmup",
                "0",
                "--no-memory",
                "-o",
                str(output),
                "--format",
                "json",
            ],
        )

        assert result.exit_code == 0, result.output
        config = capture_run["config"]
        assert (config.device, config.requested_device) == ("gpu", "gpu")
        assert config.prompt == "Name three primary colors."
        assert config.apply_template is False
        assert (config.iterations, config.warmup, config.max_new_tokens) == (3, 0, 16)
        assert config.memory is False
        assert config.output_path == output
        assert capture_run["json_mode"] is True

    @pytest.mark.parametrize("runtime", ["winml-ort", "ort-genai"])
    def test_rejects_non_runtime_api_runtimes(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, runtime: str
    ) -> None:
        result = runner.invoke(perf, ["-m", str(_gguf(tmp_path)), "--runtime", runtime])

        assert result.exit_code == 2, result.output
        assert f"--runtime {runtime} cannot load" in result.output
        assert "config" not in capture_run

    @pytest.mark.parametrize("device", ["npu", "config"])
    def test_rejects_unsupported_devices(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, device: str
    ) -> None:
        result = runner.invoke(perf, ["-m", str(_gguf(tmp_path)), "--device", device])

        assert result.exit_code == 2, result.output
        assert "not supported for GGUF models" in result.output
        assert "config" not in capture_run

    @pytest.mark.parametrize(
        ("args", "flag"),
        [
            (["--ep", "cpu"], "--ep"),
            (["--ep-options", "key=value"], "--ep-options"),
            (["--backend", "ort"], "--backend"),
            (["--module", "LlamaAttention"], "--module"),
            (["--submodel", "decoder"], "--submodel"),
        ],
    )
    def test_rejects_options_that_select_another_target(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, args: list[str], flag: str
    ) -> None:
        result = runner.invoke(perf, ["-m", str(_gguf(tmp_path)), *args])

        assert result.exit_code == 2, result.output
        assert f"{flag} cannot be used with a GGUF model" in result.output
        assert "config" not in capture_run

    def test_missing_file_is_a_usage_error(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        result = runner.invoke(perf, ["-m", str(tmp_path / "missing.gguf")])

        assert result.exit_code == 2, result.output
        assert "GGUF file not found" in result.output
        assert "config" not in capture_run

    def test_warns_about_ignored_build_options(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        result = runner.invoke(
            perf, ["-m", str(_gguf(tmp_path)), "--precision", "w8a16", "--batch-size", "2"]
        )

        assert result.exit_code == 0, result.output
        assert "ignored for GGUF models" in result.output
        assert "--precision" in result.output
        assert "--batch-size" in result.output

    def test_ignored_build_config_does_not_load_ort_configuration(
        self,
        runner: CliRunner,
        tmp_path: Path,
        capture_run: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ignored ONNX configuration cannot initialize unrelated runtime dependencies."""
        from winml.modelkit.utils import cli as cli_utils

        def unexpected_config_load(_path: Path) -> None:
            pytest.fail("GGUF must not load the ignored ONNX build configuration")

        monkeypatch.setattr(cli_utils, "load_build_config", unexpected_config_load)
        config_path = tmp_path / "onnx-build.json"
        config_path.write_text(
            json.dumps({"compile": {"device": "npu", "execution_provider": "qnn"}}),
            encoding="utf-8",
        )
        result = runner.invoke(perf, ["-m", str(_gguf(tmp_path)), "--config", str(config_path)])

        assert result.exit_code == 0, result.output
        assert "--config" in result.output and "ignored for GGUF models" in result.output
        assert capture_run["config"].device == "cpu"

    def test_prompt_and_prompt_file_are_exclusive(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("hello", encoding="utf-8")
        result = runner.invoke(
            perf,
            ["-m", str(_gguf(tmp_path)), "--prompt", "hi", "--prompt-file", str(prompt_file)],
        )

        assert result.exit_code == 2, result.output
        assert "mutually exclusive" in result.output

    def test_protocol_options_are_forwarded(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        ids = list(range(10))
        evidence = tmp_path / "run" / "task-events.json"
        result = runner.invoke(
            perf,
            [
                "-m",
                str(_gguf(tmp_path)),
                "--prompt-ids-file",
                str(_ids_file(tmp_path, json.dumps(ids).encode())),
                "--context-length",
                "64",
                "--ignore-eos",
                "--evidence",
                str(evidence),
                "--max-new-tokens",
                "8",
                "-o",
                str(tmp_path / "r.json"),
            ],
        )

        assert result.exit_code == 0, result.output
        config = capture_run["config"]
        assert (
            config.prompt_ids,
            config.context_length,
            config.ignore_eos,
            config.evidence_path,
            config.memory,
            config.apply_template,
        ) == (tuple(ids), 64, True, evidence, False, False)

    def test_prompt_ids_file_accepts_a_utf8_bom(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        path = _ids_file(tmp_path, b"\xef\xbb\xbf" + json.dumps([1, 2]).encode())
        result = runner.invoke(perf, ["-m", str(_gguf(tmp_path)), "--prompt-ids-file", str(path)])

        assert result.exit_code == 0, result.output
        assert capture_run["config"].prompt_ids == (1, 2)

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            (b"[]", "at least one token ID"),
            (b"[true]", "element 0"),
            (b"[1, 1.5]", "element 1"),
            (b"[-1]", "element 0"),
            (b'{"ids": [1]}', "JSON array of token IDs"),
            (b"1,2", "not valid JSON"),
            (b"\xff[1]", "not valid JSON"),
        ],
    )
    def test_prompt_ids_file_rejects_malformed_content(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, content: bytes, message: str
    ) -> None:
        path = _ids_file(tmp_path, content)
        result = runner.invoke(perf, ["-m", str(_gguf(tmp_path)), "--prompt-ids-file", str(path)])

        assert result.exit_code == 2, result.output
        assert f"--prompt-ids-file {path}" in result.output
        assert message in result.output
        assert "config" not in capture_run

    @pytest.mark.parametrize(
        ("extra", "message"),
        [
            (["--prompt", "hi"], "cannot be combined"),
            (["--prompt-file", "<file>"], "cannot be combined"),
            (["--apply-template"], "--apply-template cannot be used with --prompt-ids-file"),
        ],
    )
    def test_prompt_ids_conflicts_are_usage_errors(
        self,
        runner: CliRunner,
        tmp_path: Path,
        capture_run: dict,
        extra: list[str],
        message: str,
    ) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("hello", encoding="utf-8")
        extra = [str(prompt_file) if value == "<file>" else value for value in extra]
        ids = _ids_file(tmp_path, b"[1, 2]")
        result = runner.invoke(
            perf, ["-m", str(_gguf(tmp_path)), "--prompt-ids-file", str(ids), *extra]
        )

        assert result.exit_code == 2, result.output
        assert message in result.output
        assert "config" not in capture_run

    def test_explicit_no_apply_template_is_allowed_with_prompt_ids(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        ids = _ids_file(tmp_path, b"[1, 2]")
        result = runner.invoke(
            perf,
            ["-m", str(_gguf(tmp_path)), "--prompt-ids-file", str(ids), "--no-apply-template"],
        )

        assert result.exit_code == 0, result.output
        assert capture_run["config"].apply_template is False

    def test_prompt_ids_plus_max_new_tokens_must_fit_context_length(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        ids = _ids_file(tmp_path, json.dumps(list(range(10))).encode())
        result = runner.invoke(
            perf,
            [
                "-m",
                str(_gguf(tmp_path)),
                "--prompt-ids-file",
                str(ids),
                "--max-new-tokens",
                "8",
                "--context-length",
                "16",
            ],
        )

        assert result.exit_code == 2, result.output
        assert "--context-length 16" in result.output
        assert "config" not in capture_run

    @pytest.mark.parametrize(
        ("extra", "message"),
        [
            (["--memory"], "--memory cannot be used with --evidence"),
            (["--warmup", "0"], "--evidence needs at least one warmup request"),
        ],
    )
    def test_evidence_rejects_in_process_memory_and_zero_warmup(
        self,
        runner: CliRunner,
        tmp_path: Path,
        capture_run: dict,
        extra: list[str],
        message: str,
    ) -> None:
        evidence = tmp_path / "ev.json"
        result = runner.invoke(
            perf, ["-m", str(_gguf(tmp_path)), "--evidence", str(evidence), *extra]
        )

        assert result.exit_code == 2, result.output
        assert message in result.output
        assert "config" not in capture_run

    @pytest.mark.parametrize("extra", [[], ["--no-memory"]])
    def test_evidence_turns_off_default_memory_sampling(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, extra: list[str]
    ) -> None:
        evidence = tmp_path / "ev.json"
        result = runner.invoke(
            perf, ["-m", str(_gguf(tmp_path)), "--evidence", str(evidence), *extra]
        )

        assert result.exit_code == 0, result.output
        assert capture_run["config"].memory is False

    @pytest.mark.parametrize("existing", ["ev.json", "ev.rows.jsonl"])
    def test_evidence_must_be_new_unless_overwrite(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, existing: str
    ) -> None:
        (tmp_path / existing).write_text("{}", encoding="utf-8")
        args = ["-m", str(_gguf(tmp_path)), "--evidence", str(tmp_path / "ev.json")]

        result = runner.invoke(perf, args)
        assert result.exit_code == 1, result.output
        assert "already exists" in result.output
        assert "config" not in capture_run

        result = runner.invoke(perf, [*args, "--overwrite"])
        assert result.exit_code == 0, result.output
        assert capture_run["config"].evidence_path == tmp_path / "ev.json"

    @pytest.mark.parametrize("output", ["x.json", "x.rows.jsonl", "X.JSON", "x.json.tmp"])
    def test_evidence_and_output_must_differ(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, output: str
    ) -> None:
        result = runner.invoke(
            perf,
            [
                "-m",
                str(_gguf(tmp_path)),
                "--evidence",
                str(tmp_path / "x.json"),
                "-o",
                str(tmp_path / output),
            ],
        )

        assert result.exit_code == 2, result.output
        assert "--evidence and --output must name different files" in result.output
        assert "config" not in capture_run

    @pytest.mark.parametrize(
        ("option", "victim", "message"),
        [
            ("--evidence", "model.gguf", "--evidence would overwrite the --model input"),
            ("--evidence", "MODEL.GGUF", "--evidence would overwrite the --model input"),
            ("--evidence", "ids.json", "--evidence would overwrite the --prompt-ids-file input"),
            ("--evidence", "ids", "--evidence rows file would overwrite the --prompt-ids-file"),
            ("-o", "model.gguf", "--output would overwrite the --model input"),
            ("-o", "ids.json", "--output would overwrite the --prompt-ids-file input"),
        ],
    )
    def test_outputs_never_overwrite_the_model_or_the_ids_file(
        self,
        runner: CliRunner,
        tmp_path: Path,
        capture_run: dict,
        option: str,
        victim: str,
        message: str,
    ) -> None:
        model = _gguf(tmp_path)
        ids = _ids_file(tmp_path, b"[1, 2]")
        target = tmp_path / victim
        if victim == "ids":
            # The evidence rows file (ids.rows.jsonl) is the IDs file.
            ids = ids.rename(tmp_path / "ids.rows.jsonl")
            target = tmp_path / "ids.json"
        protected = {path: path.read_bytes() for path in (model, ids)}
        result = runner.invoke(
            perf,
            [
                "-m",
                str(model),
                "--prompt-ids-file",
                str(ids),
                option,
                str(target),
                "--overwrite",
            ],
        )

        assert result.exit_code == 2, result.output
        assert message in result.output
        assert "config" not in capture_run
        assert {path: path.read_bytes() for path in protected} == protected

    @pytest.mark.parametrize(
        "flag",
        [
            ["--prompt-ids-file", "<ids>"],
            ["--context-length", "8"],
            ["--ignore-eos"],
            ["--evidence", "<ev>"],
        ],
    )
    @pytest.mark.parametrize("kind", ["onnx", "genai"])
    def test_gguf_only_options_are_rejected_for_other_runtimes(
        self,
        runner: CliRunner,
        tmp_path: Path,
        capture_run: dict,
        flag: list[str],
        kind: str,
    ) -> None:
        if kind == "onnx":
            model = tmp_path / "model.onnx"
            model.write_bytes(b"")
        else:
            model = tmp_path / "bundle"
            model.mkdir()
            (model / "genai_config.json").write_text("{}", encoding="utf-8")
        values = {"<ids>": str(_ids_file(tmp_path, b"[1]")), "<ev>": str(tmp_path / "ev.json")}
        flag = [values.get(value, value) for value in flag]
        result = runner.invoke(perf, ["-m", str(model), *flag])

        assert result.exit_code == 2, result.output
        assert "only apply to GGUF models" in result.output
        assert "config" not in capture_run


class TestAutoRuntime:
    @pytest.mark.parametrize("name", ["model.gguf", "Model-00001-of-00003.GGUF"])
    def test_selects_runtime_api_for_gguf(self, name: str) -> None:
        assert _resolve_runtime("auto", name) == "winml-runtime"

    @pytest.mark.parametrize("runtime", ["winml-ort", "ort-genai", "winml-runtime"])
    def test_preserves_explicit_runtime(self, runtime: str) -> None:
        assert _resolve_runtime(runtime, "model.gguf") == runtime
