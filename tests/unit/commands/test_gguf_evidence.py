# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for gguf-task-events-1.1 evidence rows, writer and documents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_helpers import HResultError, make_record
from winml.modelkit.commands._gguf_evidence import (
    MISSING_PREFILL_TPS,
    STOP_REASONS,
    EvidenceWriter,
    incomplete_row,
    planned_requests,
    request_row,
    rows_path,
    run_status,
    token_ids_sha256,
)


def _row(**overrides: Any) -> dict[str, Any]:
    return request_row(
        make_record(**overrides), kind="measured", index=0, max_new_tokens=3, ignore_eos=False
    )


# ---------------------------------------------------------------------------
# Request rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("finish", "stop"), list(STOP_REASONS.items()))
def test_row_maps_native_finish_to_contract_stop_reason(finish: str, stop: str) -> None:
    row = _row(finish_reason=finish)
    assert (row["finish_reason"], row["stop_reason"]) == (finish, stop)
    assert STOP_REASONS == {
        "MAX_TOKENS": "limit",
        "EOS_TOKEN": "eos",
        "STOP_TOKEN": "stop_token",
        "SEQUENCE_CAPACITY": "capacity",
        "CANCELED": "cancelled",
        "ERROR": "error",
    }


def test_row_copies_qpc_events_without_merging() -> None:
    record = make_record()
    row = request_row(record, kind="warmup", index=2, max_new_tokens=3, ignore_eos=True)
    assert (row["id"], row["kind"], row["index"], row["observation"], row["status"]) == (
        "warmup-2",
        "warmup",
        2,
        "caller",
        "succeeded",
    )
    assert row["output_events"] == [
        {"observed_ns": observed, "token_ids": [token]} for observed, token in record.events
    ]
    assert row["reset"] == {
        "start_ns": record.reset_start_ns,
        "end_ns": record.reset_end_ns,
        **record.reset_disclosure,
    }
    assert row["prefill"] is None
    assert row["missing_reasons"] == {"prefill_tps": MISSING_PREFILL_TPS}
    assert (
        row["truncated"],
        row["truncated_source"],
        row["cached_tokens"],
        row["cached_tokens_source"],
    ) == (False, "task_result", 0, "task_state_after_reset")
    assert MISSING_PREFILL_TPS == (
        "Windows ML 2.7.2021a0 Task/Runtime does not expose synchronized prefill timing."
    )


def test_row_keys_follow_the_contract_order() -> None:
    assert list(_row()) == [
        "id",
        "kind",
        "index",
        "status",
        "observation",
        "input_ids",
        "actual_input_tokens",
        "reset",
        "reset_completed_before_submit",
        "submit_ns",
        "stream_created_ns",
        "output_events",
        "completed_ns",
        "result_fetched_ns",
        "finish_reason",
        "error_code",
        "stop_reason",
        "truncated",
        "truncated_source",
        "cached_tokens",
        "cached_tokens_source",
        "speculative_statistics",
        "prefill",
        "missing_reasons",
        "consistency_failures",
        "error",
    ]


def test_truncated_and_cached_tokens_are_derived() -> None:
    assert _row(finish_reason="SEQUENCE_CAPACITY")["truncated"] is True
    short = _row(prompt_token_count=3)
    assert short["truncated"] is True
    assert short["consistency_failures"] == ["prompt_count_differs"]
    kept = dict(make_record().reset_disclosure, retained_token_ids=[1, 2])
    kept["sequence_state"] = dict(kept["sequence_state"], retained_token_count=2)
    assert _row(reset_disclosure=kept)["cached_tokens"] == 2
    unknown = _row(reset_disclosure=None, disclosure_error=HResultError(0x80004002, "no QI"))
    assert unknown["cached_tokens"] is None
    assert unknown["reset"] is None
    assert unknown["reset_completed_before_submit"] is True
    assert "reset_disclosure" in unknown["missing_reasons"]


@pytest.mark.parametrize(
    ("overrides", "ignore_eos", "code"),
    [
        ({"result_token_ids": (10, 11, 99)}, False, "stream_result_ids_differ"),
        ({"prompt_token_count": 5}, False, "prompt_count_differs"),
        ({"tokens": (10, 11, 12, 13)}, False, "output_exceeds_limit"),
        ({"tokens": (10, 11)}, True, "ignore_eos_limit_not_reached"),
    ],
)
def test_consistency_failures_fail_the_row(
    overrides: dict[str, Any], ignore_eos: bool, code: str
) -> None:
    # prompt_token_count 5 exceeds the submitted 4 IDs, so truncation is not reported.
    row = request_row(
        make_record(**overrides), kind="measured", index=0, max_new_tokens=3, ignore_eos=ignore_eos
    )
    assert row["consistency_failures"] == [code]
    assert row["status"] == "failed"


def test_early_stops_still_succeed() -> None:
    assert _row(finish_reason="EOS_TOKEN")["status"] == "succeeded"


def test_native_error_fails_the_row_with_hresult() -> None:
    row = _row(error=HResultError(0x80004005, "boom"), result_fetched_ns=None)
    assert row["status"] == "failed"
    assert row["error"] == {"type": "HResultError", "message": "boom", "hresult": "0x80004005"}


def test_missing_result_fails_the_row_without_consistency_checks() -> None:
    row = _row(
        result_token_ids=None, prompt_token_count=None, finish_reason=None,
        error=RuntimeError("stream failed"),
    )
    assert row["status"] == "failed"
    assert row["consistency_failures"] == []
    assert (row["truncated"], row["stop_reason"]) == (None, None)


def test_incomplete_row_has_the_same_fields() -> None:
    row = incomplete_row(kind="measured", index=4, input_ids=[1, 2])
    assert set(row) == set(_row())
    assert (
        row["status"],
        row["input_ids"],
        row["reset_completed_before_submit"],
        row["output_events"],
        row["error"],
    ) == ("incomplete", [1, 2], False, [], None)
    assert row["id"] == "measured-4"


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
