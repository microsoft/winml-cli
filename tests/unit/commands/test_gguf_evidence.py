# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for gguf-task-events-2 evidence rows, writes and documents."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_helpers import RUNTIME_CALLBACK, FakeLogApi, make_timing
from winml.modelkit import __version__
from winml.modelkit.commands._gguf_evidence import EvidenceRecorder, _write_atomic, request_row
from winml.modelkit.commands._gguf_native_log import NativeLogCapture
from winml.modelkit.commands._perf_gguf import GGUFPerfConfig
from winml.modelkit.session import LlamaExecutionDiagnostics, llama_package_versions


if TYPE_CHECKING:
    from collections.abc import Callable


def _clock(start_ns: int = 1_000, step_ns: int = 50) -> Callable[[], float]:
    """A ``time.perf_counter()`` stand-in reading ``start_ns``, then every ``step_ns``."""
    ticks = itertools.count(start_ns, step_ns)
    return lambda: next(ticks) / 1e9


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Request rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cached_tokens", [5, None])
def test_row_keeps_the_generate_timed_readings_as_integer_nanoseconds(
    cached_tokens: int | None,
) -> None:
    start = 345_678_123_456_789  # about four days of QPC uptime, in nanoseconds
    timing = make_timing(_clock(start, 50), tokens=(10, 11, 12), cached_tokens=cached_tokens)
    # make_timing reads the clock at the reset start and end, the submit, the stream
    # creation, each token, the end of the stream and the checked result.
    assert request_row("timed", 2, timing) == {
        "kind": "timed",
        "index": 2,
        "submit_ns": start + 100,
        "completed_ns": start + 350,
        "output_events": [
            {"observed_ns": start + 200, "token_id": 10},
            {"observed_ns": start + 250, "token_id": 11},
            {"observed_ns": start + 300, "token_id": 12},
        ],
        "stop_reason": "MAX_TOKENS",
        "prompt_tokens": 4,
        # A count above 0 shows prompt reuse; None that the session state was unreadable.
        "cached_tokens": cached_tokens,
    }


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def test_write_is_atomic_and_rejects_nan(tmp_path: Path) -> None:
    path = tmp_path / "ev.json"
    _write_atomic(path, {"a": 1})
    with pytest.raises(ValueError, match="JSON"):
        _write_atomic(path, {"a": float("nan")})
    assert _read(path) == {"a": 1}


def test_write_retries_while_another_process_holds_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    real = Path.replace

    def flaky(self: Path, target: Path) -> Path:
        calls.append(target)
        if len(calls) <= 2:
            raise PermissionError(13, "in use")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    path = tmp_path / "ev.json"
    _write_atomic(path, {"status": "complete"}, sleep=lambda _seconds: None)
    assert len(calls) == 3
    assert _read(path) == {"status": "complete"}


def test_write_gives_up_after_the_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def always_busy(self: Path, target: Path) -> Path:
        raise PermissionError(13, "in use")

    monkeypatch.setattr(Path, "replace", always_busy)
    sleeps: list[float] = []
    with pytest.raises(PermissionError):
        _write_atomic(tmp_path / "ev.json", {}, sleep=sleeps.append)
    assert sleeps == [0.05] * 19


# ---------------------------------------------------------------------------
# Evidence documents
# ---------------------------------------------------------------------------


def _config(tmp_path: Path) -> GGUFPerfConfig:
    return GGUFPerfConfig(
        model_path=tmp_path / "tiny.gguf",
        evidence_path=tmp_path / "runs" / "ev.json",
        prompt_ids=(1, 2, 3, 4),
        context_length=16,
        ignore_eos=True,
        warmup=1,
        iterations=2,
        max_new_tokens=3,
    )


class TestEvidenceRecorder:
    def test_start_write_names_only_the_schema_status_and_process(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        assert config.evidence_path is not None
        EvidenceRecorder(config, capture=NativeLogCapture(load_api=lambda: None))
        # The missing folder is created.
        assert _read(config.evidence_path) == {
            "schema_version": "gguf-task-events-2",
            "status": "running",
            "pid": os.getpid(),
        }

    @pytest.mark.parametrize("error", [None, RuntimeError("native session crashed")])
    def test_end_write_keeps_the_finished_requests_and_the_native_log(
        self, tmp_path: Path, error: BaseException | None
    ) -> None:
        config = _config(tmp_path)
        assert config.evidence_path is not None
        api = FakeLogApi()
        stamps = itertools.count(500, 100)
        capture = NativeLogCapture(load_api=lambda: api, clock_ns=stamps.__next__)
        capture.install()
        recorder = EvidenceRecorder(config, capture=capture)
        clock = _clock(10_000)
        # The second request could not read its session state; null survives the write.
        timings = [make_timing(clock), make_timing(clock, cached_tokens=None)]
        api.emit(2, b"llama_model_loader: loaded meta data\n")
        api.emit(2, b"llama_context: n_ctx = 16\n")
        recorder.record_request("warmup", 0, timings[0])
        recorder.record_request("timed", 1, timings[1])
        recorder.finish(error)

        doc = _read(config.evidence_path)
        assert list(doc) == [
            "schema_version",
            "status",
            "pid",
            "error",
            "workload",
            "packages",
            "requests",
            "native_log",
            "execution_diagnostics",
        ]
        assert (doc["schema_version"], doc["pid"]) == ("gguf-task-events-2", os.getpid())
        # No diagnostics were handed over, as when the run ends before its pipeline is built.
        assert doc["execution_diagnostics"] is None
        if error is None:
            assert (doc["status"], doc["error"]) == ("complete", None)
        else:
            assert (doc["status"], doc["error"]) == (
                "failed",
                {"type": "RuntimeError", "message": "native session crashed"},
            )
        assert doc["workload"] == {
            "input_ids_sha256": hashlib.sha256(b"[1,2,3,4]").hexdigest(),
            "input_token_count": 4,
            "max_new_tokens": 3,
            "context_length": 16,
            "ignore_eos": True,
            "warmup": 1,
            "iterations": 2,
        }
        assert doc["packages"] == {**llama_package_versions(), "winml-cli": __version__}
        assert doc["requests"] == [
            request_row("warmup", 0, timings[0]),
            request_row("timed", 1, timings[1]),
        ]
        # finish() put the Runtime callback back before the end write, which holds the
        # captured fragments as emitted, each with its own stamp.
        assert api.callback == RUNTIME_CALLBACK
        assert doc["native_log"] == {
            "installed": True,
            "reason": None,
            "intact": True,
            "restored": True,
            "fragments": [
                {"level": 2, "text": "llama_model_loader: loaded meta data\n", "qpc_ns": 500},
                {"level": 2, "text": "llama_context: n_ctx = 16\n", "qpc_ns": 600},
            ],
        }

    def test_end_write_keeps_the_execution_diagnostics(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        assert config.evidence_path is not None
        share = {
            "device_kind": "GPU",
            "adapter_luid": "0x00000001_0x0001A2B3",
            "implementation": "VENDOR_LIBRARY",
            "unit_count": 17,
        }
        diagnostics = LlamaExecutionDiagnostics(
            enabled=True,
            reason=None,
            coverage="FULL",
            work_unit="MODEL_LAYER",
            work_shares=(share,),
            description="device=GPU0; offloaded layers=17/17\n",
        )
        recorder = EvidenceRecorder(config, capture=NativeLogCapture(load_api=lambda: None))
        recorder.finish(None, diagnostics=diagnostics)

        assert _read(config.evidence_path)["execution_diagnostics"] == {
            "enabled": True,
            "reason": None,
            "coverage": "FULL",
            "work_unit": "MODEL_LAYER",
            "work_shares": [share],
            "description": "device=GPU0; offloaded layers=17/17\n",
        }

    def test_text_prompt_workload_has_no_token_ids(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        config.prompt_ids = None
        assert config.evidence_path is not None
        EvidenceRecorder(config, capture=NativeLogCapture(load_api=lambda: None)).finish(None)

        workload = _read(config.evidence_path)["workload"]
        assert (workload["input_ids_sha256"], workload["input_token_count"]) == (None, None)
