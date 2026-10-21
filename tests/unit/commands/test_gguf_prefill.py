# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Generated native-writer evidence validates correlation independently of Tasks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from random import Random
from typing import Any

import pytest

from winml.modelkit.commands._gguf_prefill import NativePrefillTrace, PrefillTraceError


ENV = "WINMLCLI_GGUF_PREFILL_CONTEXT"


@pytest.fixture
def trace(tmp_path: Path) -> NativePrefillTrace:
    model = tmp_path / "generated.gguf"
    model.write_bytes(Random(19).randbytes(128))
    return NativePrefillTrace(tmp_path / "events.jsonl", model)


def _native_event() -> dict[str, Any]:
    """A fake native writer snapshots the live CLI context and backend clock."""
    context = json.loads(Path(os.environ[ENV]).read_text(encoding="utf-8"))
    assert context["active"]
    random = Random(context["request"]["index"])
    start = random.randrange(1_000_000, 10_000_000)
    completed = start + random.randrange(1000, 1_000_000)
    return {
        "schema_version": context["event_schema_version"],
        **{key: context[key] for key in ("run_id", "pid", "model_sha256")},
        **context["request"],
        "phase": "prefill",
        "duration_ns": completed - start,
        "boundary": "backend_prompt_start_to_synchronized_completion_before_sampling",
        "synchronized": True,
        "before_first_sampling": True,
        "clock_domain": "generated-native-monotonic-clock",
        "clock_monotonic": True,
        "source": "generated-native-prefill-observer",
        "backend_build": hashlib.sha256(b"generated-build").hexdigest(),
    }


def _append(trace: NativePrefillTrace, event: dict[str, Any]) -> None:
    with trace.path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")


def test_native_duration_and_provenance_round_trip(trace: NativePrefillTrace) -> None:
    ids = [Random(index).randrange(10000) for index in range(37)]
    with trace:
        trace.prepare_request("timed", 0, ids)
        event = _native_event()
        _append(trace, event)
        trace.finish_request()
        assert not json.loads(trace.context_path.read_text())["active"]
        observation = trace.read_observations()[("timed", 0)]
        assert observation.duration_ns == event["duration_ns"]
        assert observation.input_tokens == len(ids)
        assert observation.to_dict()["boundary"] == event["boundary"]
        assert (
            trace.metadata()["trace_sha256"] == hashlib.sha256(trace.path.read_bytes()).hexdigest()
        )
        assert (
            event["prompt_sha256"]
            == hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        )


def test_missing_native_events_remain_absent(trace: NativePrefillTrace) -> None:
    with trace:
        for index in range(3):
            trace.prepare_request("timed", index, list(range(index + 1)))
            if index == 1:
                _append(trace, _native_event())
            trace.finish_request()
        assert set(trace.read_observations()) == {("timed", 1)}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "previous-run"),
        ("pid", -1),
        ("request_id", "unknown"),
        ("model_sha256", "different"),
        ("prompt_sha256", "different"),
        ("prompt_tokens", 2),
        ("prompt_tokens", True),
        ("kind", "warmup"),
        ("index", True),
        ("phase", "decode"),
        ("schema_version", "unknown"),
        ("duration_ns", 0),
        ("duration_ns", -1),
        ("duration_ns", True),
        ("duration_ns", float("nan")),
        ("duration_ns", float("inf")),
        ("duration_ns", 2**63),
        ("synchronized", False),
        ("synchronized", 1),
        ("before_first_sampling", False),
        ("boundary", "submit_to_first_token"),
        ("source", ""),
        ("backend_build", None),
        ("clock_domain", " "),
        ("clock_monotonic", False),
    ],
)
def test_invalid_native_evidence_is_not_silently_used(
    trace: NativePrefillTrace,
    field: str,
    value: Any,
) -> None:
    with trace:
        trace.prepare_request("timed", 0, [7])
        event = _native_event()
        event[field] = value
        _append(trace, event)
        trace.finish_request()
        with pytest.raises(PrefillTraceError):
            trace.read_observations()
    assert trace.path.is_file()  # Invalid original evidence is preserved for diagnosis.


def test_duplicate_event_is_not_a_second_sample(trace: NativePrefillTrace) -> None:
    with trace:
        trace.prepare_request("timed", 0, [7])
        event = _native_event()
        _append(trace, event)
        _append(trace, event)
        trace.finish_request()
        trace.prepare_request("timed", 1, [7])
        trace.finish_request()
        with pytest.raises(PrefillTraceError, match="Duplicate"):
            trace.read_observations()


def test_source_build_drift_is_rejected(trace: NativePrefillTrace) -> None:
    with trace:
        for index in range(2):
            trace.prepare_request("timed", index, [7])
            event = _native_event()
            event["backend_build"] += str(index)
            _append(trace, event)
            trace.finish_request()
        with pytest.raises(PrefillTraceError, match="changed within"):
            trace.read_observations()


@pytest.mark.parametrize("payload", [b"{", b"null\n", b"\xff\n", b"\n"])
def test_malformed_trace_is_explicit_error(trace: NativePrefillTrace, payload: bytes) -> None:
    with trace:
        trace.prepare_request("timed", 0, [7])
        trace.finish_request()
        trace.path.write_bytes(payload)
        with pytest.raises(PrefillTraceError):
            trace.read_observations()


def test_trace_size_is_bounded(trace: NativePrefillTrace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("winml.modelkit.commands._gguf_prefill._MAX_TRACE_BYTES", 8)
    with trace:
        trace.path.write_bytes(b"x" * 9)
        with pytest.raises(PrefillTraceError, match="limit"):
            trace.read_observations()


def test_context_is_restored_after_generation_failure(
    trace: NativePrefillTrace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV, "previous-context")
    with pytest.raises(RuntimeError, match="generation failed"), trace:
        trace.prepare_request("timed", 0, [7])
        raise RuntimeError("generation failed")
    assert os.environ[ENV] == "previous-context"
    assert not json.loads(trace.context_path.read_text())["active"]


def test_existing_trace_is_not_overwritten_and_failed_enter_releases_lock(
    trace: NativePrefillTrace,
    tmp_path: Path,
) -> None:
    before = Random(23).randbytes(20)
    trace.path.write_bytes(before)
    with pytest.raises(PrefillTraceError, match="fresh"), trace:
        pass
    assert trace.path.read_bytes() == before
    with NativePrefillTrace(tmp_path / "new.jsonl", trace.model_path) as other:
        assert other.read_observations() == {}


def test_nested_or_overlapping_context_is_rejected(
    trace: NativePrefillTrace, tmp_path: Path
) -> None:
    with trace:
        with (
            pytest.raises(PrefillTraceError, match="already in use"),
            NativePrefillTrace(
                tmp_path / "nested.jsonl",
                trace.model_path,
            ),
        ):
            pass
        trace.prepare_request("timed", 0, [7])
        with pytest.raises(PrefillTraceError, match="serialized"):
            trace.prepare_request("timed", 1, [7])
        with pytest.raises(PrefillTraceError, match="after all"):
            trace.read_observations()
        trace.finish_request()
        assert trace.read_observations() == {}


def test_cancelled_model_hash_releases_context_lock(
    trace: NativePrefillTrace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = hashlib.file_digest

    def cancelled(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(hashlib, "file_digest", cancelled)
    with pytest.raises(KeyboardInterrupt), trace:
        pass
    monkeypatch.setattr(hashlib, "file_digest", original)
    with NativePrefillTrace(tmp_path / "after-cancel.jsonl", trace.model_path) as other:
        assert other.read_observations() == {}
