# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for gguf-task-events-1.1 evidence rows, writer and documents."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_helpers import FakeLogApi, FakeOptions, HResultError, make_timing
from winml.modelkit.commands import _gguf_evidence
from winml.modelkit.commands._gguf_evidence import (
    MISSING_LOAD,
    MISSING_PREFILL_TPS,
    MISSING_READY,
    EvidenceRecorder,
    EvidenceWriter,
    error_object,
    incomplete_row,
    planned_requests,
    request_row,
    rows_path,
    run_status,
    token_ids_sha256,
)
from winml.modelkit.commands._gguf_identity import (
    clock_info,
    model_inventory,
    process_identity,
    producer_info,
)
from winml.modelkit.commands._gguf_native_log import NATIVE_VALUE_KEYS, NativeLogCapture


if TYPE_CHECKING:
    from collections.abc import Callable


def _clock(start_ns: int = 1_000, step_ns: int = 50) -> Callable[[], float]:
    """A ``time.perf_counter()`` stand-in reading ``start_ns``, then every ``step_ns``."""
    ticks = itertools.count(start_ns, step_ns)
    return lambda: next(ticks) / 1e9


# ---------------------------------------------------------------------------
# Request rows
# ---------------------------------------------------------------------------


def test_row_keeps_the_generate_timed_readings_as_integer_nanoseconds() -> None:
    start = 345_678_123_456_789  # about four days of QPC uptime, in nanoseconds
    timing = make_timing(_clock(start, 50), tokens=(10, 11, 12), cached_tokens=0)
    # make_timing reads the clock at the reset start and end, the submit, the stream
    # creation, each token, the end of the stream and the checked result.
    assert request_row("warmup", 2, timing) == {
        "id": "warmup-2",
        "kind": "warmup",
        "index": 2,
        "status": "succeeded",
        "submit_ns": start + 100,
        "completed_ns": start + 350,
        "output_events": [
            {"observed_ns": start + 200, "token_id": 10},
            {"observed_ns": start + 250, "token_id": 11},
            {"observed_ns": start + 300, "token_id": 12},
        ],
        "stop_reason": "MAX_TOKENS",
        "prompt_tokens": 4,
        "cached_tokens": 0,
        "prefill": None,
        "missing_reasons": {"prefill_tps": MISSING_PREFILL_TPS},
        "error": None,
    }


def test_incomplete_row_has_the_same_fields() -> None:
    row = incomplete_row(kind="measured", index=4)
    assert set(row) == set(request_row("measured", 0, make_timing(_clock())))
    assert (row["id"], row["status"], row["output_events"], row["error"]) == (
        "measured-4",
        "incomplete",
        [],
        None,
    )


def test_run_status_rules() -> None:
    ok, bad = {"status": "succeeded"}, {"status": "failed"}
    assert [
        run_status([ok, ok], failure_phase=None),
        run_status([ok, bad], failure_phase=None),
        run_status([ok], failure_phase="finalization"),
    ] == ["complete", "partial", "failed"]


def test_hashes_paths_and_plan_follow_the_contract(tmp_path: Path) -> None:
    ids = list(range(1024))
    assert token_ids_sha256(ids) == hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()
    ).hexdigest()
    assert rows_path(tmp_path / "task-events.json") == tmp_path / "task-events.rows.jsonl"
    assert planned_requests(2, 3) == [
        ("warmup", 0),
        ("warmup", 1),
        ("measured", 0),
        ("measured", 1),
        ("measured", 2),
    ]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def test_write_is_atomic_and_rejects_nan(tmp_path: Path) -> None:
    path = tmp_path / "out" / "ev.json"
    writer = EvidenceWriter(path)
    writer.write({"a": 1})
    with pytest.raises(ValueError, match="JSON"):
        writer.write({"a": float("nan")})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}
    writer.close()


def test_rows_are_lf_terminated_and_flushed(tmp_path: Path) -> None:
    path = tmp_path / "ev.json"
    writer = EvidenceWriter(path)
    row = {"id": "warmup-0", "n": 1}
    writer.append_row(row)
    assert rows_path(path).read_bytes() == json.dumps(row, separators=(",", ":")).encode() + b"\n"
    writer.close()


def test_writer_truncates_stale_rows(tmp_path: Path) -> None:  # Review Focus 3
    path = tmp_path / "ev.json"
    rows_path(path).write_text("stale\n", encoding="utf-8")
    writer = EvidenceWriter(path)
    assert rows_path(path).read_bytes() == b""
    writer.close()


def test_final_write_retries_while_a_reader_holds_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # Review Focus 1
    calls: list[Path] = []
    real = Path.replace

    def flaky(self: Path, target: Path) -> Path:
        calls.append(target)
        if len(calls) <= 2:
            raise PermissionError(13, "in use")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    path = tmp_path / "ev.json"
    writer = EvidenceWriter(path, sleep=lambda _seconds: None)
    writer.write({"status": "complete"})
    writer.close()
    assert len(calls) == 3
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "complete"}


def test_final_write_gives_up_after_the_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # Review Focus 1
    def always_busy(self: Path, target: Path) -> Path:
        raise PermissionError(13, "in use")

    monkeypatch.setattr(Path, "replace", always_busy)
    sleeps: list[float] = []
    writer = EvidenceWriter(tmp_path / "ev.json", sleep=sleeps.append)
    with pytest.raises(PermissionError):
        writer.write({})
    writer.close()
    assert sleeps == [0.05] * 19


# ---------------------------------------------------------------------------
# Evidence documents
# ---------------------------------------------------------------------------


def _cfg(tmp_path: Path, **overrides: Any) -> SimpleNamespace:
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"GGUF" + bytes(range(64)))
    values: dict[str, Any] = {
        "model_path": model,
        "device": "cpu",
        "evidence_path": tmp_path / "ev.json",
        "prompt_ids": (1, 2, 3, 4),
        "prompt": None,
        "context_length": 16,
        "ignore_eos": True,
        "warmup": 1,
        "iterations": 2,
        "max_new_tokens": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _session_ns() -> SimpleNamespace:
    return SimpleNamespace(
        context_length=16,
        declared_context_length=None,
        task_default_eos_token_ids=(),
        tokenizer_eos_token_ids=(7,),
        speculative_readback={
            "model_draft_predictor_enabled": False,
            "draft_token_limit_hint": 0,
            "draft_predictor_path": None,
            "reason": None,
        },
        resolved_target_kind="cpu",
    )


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _no_capture() -> NativeLogCapture:
    return NativeLogCapture(load_api=lambda: None)


def _run_all(recorder: EvidenceRecorder, cfg: SimpleNamespace) -> None:
    clock = _clock(10_000)
    for kind, index in planned_requests(cfg.warmup, cfg.iterations):
        recorder.record_request(kind, index, make_timing(clock))


class _RaisingEosSession(SimpleNamespace):
    @property
    def task_default_eos_token_ids(self) -> tuple[int, ...]:
        raise HResultError(0x80004005, "read-back failed")


class TestEvidenceRecorder:
    def test_provisional_document_identifies_the_process(self, tmp_path: Path) -> None:
        EvidenceRecorder(_cfg(tmp_path), clock_ns=lambda: 42)
        doc = _read(tmp_path / "ev.json")
        assert (doc["schema_version"], doc["host_api"], doc["status"], doc["load"]) == (
            "gguf-task-events-1.1",
            "windowsml.tasks",
            "running",
            None,
        )
        assert "path_id" not in doc
        assert (doc["requests"], doc["missing_reasons"]) == ([], {})
        assert (doc["failure_phase"], doc["error"]) == (None, None)
        assert doc["process"] == process_identity()
        assert doc["clock"] == clock_info(42)
        assert doc["producer"] == producer_info()
        for key in ("workload", "configuration", "initialization", "native_log", "cache_state"):
            assert doc[key] == {}
        assert list(doc) == [
            "schema_version", "producer", "host_api", "process", "clock", "status",
            "failure_phase", "error", "workload", "configuration", "initialization",
            "requests", "native_log", "load", "missing_reasons", "cache_state",
        ]

    def test_complete_document_records_requested_and_effective_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLAMA_ARG_SPEC_TYPE", "none")
        cfg = _cfg(tmp_path)
        recorder = EvidenceRecorder(cfg)
        recorder.record_configuration()
        recorder.begin_initialization()
        recorder.record_prompt(cfg.prompt_ids)
        session = _session_ns()
        recorder.record_load(session, requested_context=16)
        options = FakeOptions(max_new_tokens=3)
        recorder.record_options(options)
        _run_all(recorder, cfg)
        recorder.record_loaded_modules("after_warmup0")
        recorder.record_loaded_modules("at_end")
        recorder.record_hashed_modules()
        status = recorder.finish(
            capture=_no_capture(), initialization_spans=[("runtime_create", 5, 6)]
        )
        doc = _read(cfg.evidence_path)
        assert status == doc["status"] == recorder.status == "complete"
        assert (doc["failure_phase"], doc["error"]) == (None, None)
        assert doc["workload"] == {
            "input_mode": "token_ids",
            "input_ids_sha256": token_ids_sha256(cfg.prompt_ids),
            "input_token_count": 4,
            "max_new_tokens": 3,
            "context_length": 16,
            "ignore_eos": True,
            "warmup": 1,
            "iterations": 2,
            "sampling": {**dataclasses.asdict(options), "stop_token_ids": []},
        }
        c = doc["configuration"]
        assert c["model"] == model_inventory(cfg.model_path)
        assert c["environment"]["LLAMA_ARG_SPEC_TYPE"] == "none"
        assert c["context"] == {"requested": 16, "effective": 16, "declared": None}
        assert c["eos"] == {
            "requested_policy": "ignore",
            "task_default_eos_token_ids": [],
            "tokenizer_eos_token_ids": [7],
            "stop_token_ids": [],
            "reason": None,
        }
        assert c["speculative"] == {
            **session.speculative_readback,
            "source": "stage_readback_not_set_by_cli",
        }
        assert c["device"] == {
            "requested": "cpu",
            "effective_target_kind": "cpu",
            "physical_adapter": {
                "effective": "unknown",
                "reason": (
                    "ExecutionTarget exposes only kind; the benchmark attributes the adapter "
                    "externally."
                ),
            },
        }
        assert c["sampling"] == {
            "effective": "unverified",
            "reason": "Windows ML 2.7.2021a0 has no TextGenerationOptions read-back.",
        }
        assert c["in_process_resource_work"] is False
        assert set(c["native"]["unknown"]) == {"threads", *NATIVE_VALUE_KEYS}
        assert c["native"]["values"] == {}
        assert doc["initialization"]["spans"] == [
            {"name": "runtime_create", "start_ns": 5, "end_ns": 6}
        ]
        assert doc["initialization"]["ready"] is None
        assert doc["missing_reasons"] == {"load": MISSING_LOAD, "ready": MISSING_READY}
        assert MISSING_LOAD == (
            "Windows ML 2.7.2021a0 does not expose an initialization-to-ready signal; Task "
            "session creation is not Ready."
        )
        assert MISSING_READY == (
            "TextGenerationSessionState.READY is logical; the native context is created "
            "lazily on the first request."
        )
        assert doc["cache_state"]["file"] == doc["cache_state"]["compilation"] == "unknown"
        sidecar = rows_path(cfg.evidence_path).read_text(encoding="utf-8").splitlines()
        assert [json.loads(line) for line in sidecar] == doc["requests"]

    def test_native_values_and_phases_come_from_the_capture(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path, warmup=1, iterations=0)
        api = FakeLogApi()
        stamps = iter([500, 50_000])
        capture = NativeLogCapture(
            load_api=lambda: ("C:/lib/WinMLllama.dll", api), clock_ns=lambda: next(stamps)
        )
        capture.install()
        recorder = EvidenceRecorder(cfg)
        api.emit(2, b"llama_context: n_ctx                 = 4096\n")
        recorder.record_request("warmup", 0, make_timing(_clock(10_000)))
        api.emit(2, b"llama_context: n_ctx                 = 9\n")
        capture.restore()
        recorder.finish(capture=capture, initialization_spans=[])
        doc = _read(cfg.evidence_path)
        assert recorder.warmup0_submit_ns == 10_100  # the third reading: after the reset
        assert [line["phase"] for line in doc["native_log"]["lines"]] == ["load", "warmup-0"]
        assert doc["configuration"]["native"]["values"]["n_ctx"] == {"value": 4096, "line": 0}
        assert doc["configuration"]["native"]["unknown"]["n_batch"] == (
            "No matching libllama log line in the capture window."
        )

    def test_eos_readback_failure_is_recorded_with_reason(self, tmp_path: Path) -> None:
        values = {
            key: value
            for key, value in vars(_session_ns()).items()
            if key != "task_default_eos_token_ids"
        }
        recorder = EvidenceRecorder(_cfg(tmp_path))
        recorder.record_load(_RaisingEosSession(**values), requested_context=16)
        recorder.finish(capture=_no_capture(), initialization_spans=[])
        eos = _read(tmp_path / "ev.json")["configuration"]["eos"]
        assert eos["task_default_eos_token_ids"] is None
        assert eos["reason"] == "HResultError: read-back failed"

    @pytest.mark.parametrize(
        ("phase", "first_error"), [("measured", True), ("initialization", False)]
    )
    def test_failure_fills_remaining_planned_rows_as_incomplete(
        self, tmp_path: Path, phase: str, first_error: bool
    ) -> None:
        cfg = _cfg(tmp_path)
        recorder = EvidenceRecorder(cfg)
        recorder.record_configuration()
        recorder.record_request("warmup", 0, make_timing(_clock()))
        exc = HResultError(0x80004005, "session unusable")
        recorder.fail(phase, exc)
        recorder.fail("finalization", RuntimeError("later"))  # only the first failure counts
        status = recorder.finish(capture=_no_capture(), initialization_spans=[])
        doc = _read(cfg.evidence_path)
        rows = doc["requests"]
        assert [row["status"] for row in rows] == ["succeeded", "incomplete", "incomplete"]
        assert [row["id"] for row in rows] == ["warmup-0", "measured-0", "measured-1"]
        assert rows[1]["error"] == (error_object(exc) if first_error else None)
        assert rows[2]["error"] is None
        assert (status, doc["failure_phase"], doc["error"]) == (
            "failed",
            phase,
            error_object(exc),
        )
        assert recorder.cause is exc
        sidecar = rows_path(cfg.evidence_path).read_text(encoding="utf-8").splitlines()
        assert [json.loads(line) for line in sidecar] == rows

    def test_modules_and_ort_sidecar_are_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        modules = [r"C:\x\onnxruntime.dll", r"C:\x\kernel32.dll"]
        monkeypatch.setattr(_gguf_evidence, "list_loaded_modules", lambda: modules)
        recorder = EvidenceRecorder(_cfg(tmp_path))
        recorder.record_loaded_modules("after_warmup0")
        recorder.record_loaded_modules("at_end")
        recorder.finish(capture=_no_capture(), initialization_spans=[])
        c = _read(tmp_path / "ev.json")["configuration"]
        assert c["loaded_modules"] == {"after_warmup0": modules, "at_end": modules}
        assert c["ort_sidecar_loaded"] is True

    def test_initialization_starts_after_the_process_record(self, tmp_path: Path) -> None:
        recorder = EvidenceRecorder(_cfg(tmp_path), clock_ns=itertools.count(1_000, 7).__next__)
        recorder.begin_initialization()
        recorder.finish(capture=_no_capture(), initialization_spans=[])
        doc = _read(tmp_path / "ev.json")
        assert doc["clock"]["started_qpc_ns"] < doc["initialization"]["init_start_ns"]
