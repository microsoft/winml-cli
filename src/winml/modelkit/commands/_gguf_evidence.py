# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""``gguf-task-events-1.1`` evidence for GGUF protocol runs of ``winml perf``.

An evidence run writes one JSON document (provisional at start, final at the
end, each write atomic) and a ``.rows.jsonl`` sidecar that receives each
request row as soon as the request finishes, so a crash still leaves the
completed rows behind. A benchmark harness recomputes every metric from these
raw rows; nothing here aggregates.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from ..session import LlamaRequestRecord


EVIDENCE_SCHEMA_VERSION = "gguf-task-events-1.1"
HOST_API = "windowsml.tasks"
MISSING_LOAD = (
    "Windows ML 2.7.2021a0 does not expose an initialization-to-ready signal; "
    "Task session creation is not Ready."
)
MISSING_READY = (
    "TextGenerationSessionState.READY is logical; the native context is created lazily "
    "on the first request."
)
MISSING_PREFILL_TPS = (
    "Windows ML 2.7.2021a0 Task/Runtime does not expose synchronized prefill timing."
)
# Native Task finish reasons and the evidence stop reasons they map to.
STOP_REASONS: dict[str, str] = {
    "MAX_TOKENS": "limit",
    "EOS_TOKEN": "eos",
    "STOP_TOKEN": "stop_token",
    "SEQUENCE_CAPACITY": "capacity",
    "CANCELED": "cancelled",
    "ERROR": "error",
}

# A reader holding the evidence file open makes the atomic replace fail on Windows.
_REPLACE_ATTEMPTS = 20
_REPLACE_INTERVAL_S = 0.05


def rows_path(evidence_path: Path) -> Path:
    """The rows sidecar of an evidence file (``task-events.json`` -> ``task-events.rows.jsonl``)."""
    return evidence_path.with_name(evidence_path.stem + ".rows.jsonl")


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """SHA-256 of the compact JSON array of token IDs (the benchmark's prompt hash)."""
    return hashlib.sha256(json.dumps(list(token_ids), separators=(",", ":")).encode()).hexdigest()


def error_object(exc: BaseException) -> dict[str, Any]:
    """Describe a failure; ``hresult`` is set when the error carries an integer HRESULT."""
    hresult = getattr(exc, "hresult", None)
    text = None
    if isinstance(hresult, int) and not isinstance(hresult, bool):
        text = f"0x{hresult & 0xFFFFFFFF:08X}"
    return {"type": type(exc).__name__, "message": str(exc), "hresult": text}


def planned_requests(warmup: int, iterations: int) -> list[tuple[str, int]]:
    """``(kind, index)`` of every planned request: all warmups, then the measured ones."""
    return [("warmup", index) for index in range(warmup)] + [
        ("measured", index) for index in range(iterations)
    ]


def _reset_object(record: LlamaRequestRecord) -> dict[str, Any] | None:
    if record.reset_disclosure is None:
        return None
    return {
        "start_ns": record.reset_start_ns,
        "end_ns": record.reset_end_ns,
        **record.reset_disclosure,
    }


def _cached_tokens(disclosure: Mapping[str, Any] | None) -> int | None:
    """0 for a verified-empty session after reset; otherwise the largest disclosed count.

    The count is at least 1 whenever any token was evaluated or retained, so a
    session that was not cleared can never read as prompt-reuse free.
    """
    if disclosure is None:
        return None
    sequence = disclosure["sequence_state"]
    retained_ids = list(disclosure["retained_token_ids"])
    evaluated = int(sequence["evaluated_token_count"])
    retained = int(sequence["retained_token_count"])
    if evaluated == 0 and retained == 0 and not retained_ids:
        return 0
    return max(retained, evaluated, len(retained_ids))


def _consistency_failures(
    record: LlamaRequestRecord, *, max_new_tokens: int, ignore_eos: bool
) -> list[str]:
    result_ids = list(record.result_token_ids or ())
    failures: list[str] = []
    if [token for _, token in record.events] != result_ids:
        failures.append("stream_result_ids_differ")
    if record.prompt_token_count != len(record.input_ids):
        failures.append("prompt_count_differs")
    if len(result_ids) > max_new_tokens:
        failures.append("output_exceeds_limit")
    if ignore_eos and record.finish_reason == "MAX_TOKENS" and len(result_ids) != max_new_tokens:
        failures.append("ignore_eos_limit_not_reached")
    return failures


def _base_row(kind: str, index: int, input_ids: Sequence[int]) -> dict[str, Any]:
    """Every row key in contract order, with the values of a request that never ran."""
    return {
        "id": f"{kind}-{index}",
        "kind": kind,
        "index": index,
        "status": "incomplete",
        "observation": "caller",
        "input_ids": list(input_ids),
        "actual_input_tokens": None,
        "reset": None,
        "reset_completed_before_submit": False,
        "submit_ns": None,
        "stream_created_ns": None,
        "output_events": [],
        "completed_ns": None,
        "result_fetched_ns": None,
        "finish_reason": None,
        "error_code": None,
        "stop_reason": None,
        "truncated": None,
        "truncated_source": "task_result",
        "cached_tokens": None,
        "cached_tokens_source": "task_state_after_reset",
        "speculative_statistics": None,
        "prefill": None,
        "missing_reasons": {"prefill_tps": MISSING_PREFILL_TPS},
        "consistency_failures": [],
        "error": None,
    }


def request_row(
    record: LlamaRequestRecord, *, kind: str, index: int, max_new_tokens: int, ignore_eos: bool
) -> dict[str, Any]:
    """Build the evidence row of one executed request.

    The row fails when the request raised, reported ``ERROR``, produced no
    result, or failed a consistency check; early stops (EOS, stop token,
    capacity, cancellation) still succeed and keep their stop reason.
    """
    has_result = record.result_token_ids is not None
    failures = (
        _consistency_failures(record, max_new_tokens=max_new_tokens, ignore_eos=ignore_eos)
        if has_result
        else []
    )
    error = record.error if record.error is not None else record.reset_error
    failed = (
        error is not None or bool(failures) or record.finish_reason == "ERROR" or not has_result
    )
    truncated = None
    if has_result:
        truncated = (record.prompt_token_count or 0) < len(record.input_ids) or (
            record.finish_reason == "SEQUENCE_CAPACITY"
        )
    row = _base_row(kind, index, record.input_ids)
    row.update(
        {
            "status": "failed" if failed else "succeeded",
            "actual_input_tokens": record.prompt_token_count,
            "reset": _reset_object(record),
            "reset_completed_before_submit": record.reset_end_ns is not None,
            "submit_ns": record.submit_ns,
            "stream_created_ns": record.stream_created_ns,
            "output_events": [
                {"observed_ns": observed_ns, "token_ids": [token]}
                for observed_ns, token in record.events
            ],
            "completed_ns": record.completed_ns,
            "result_fetched_ns": record.result_fetched_ns,
            "finish_reason": record.finish_reason,
            "error_code": record.error_code,
            "stop_reason": STOP_REASONS.get(record.finish_reason) if record.finish_reason else None,
            "truncated": truncated,
            "cached_tokens": _cached_tokens(record.reset_disclosure),
            "speculative_statistics": record.speculative_statistics,
            "consistency_failures": failures,
            "error": None if error is None else error_object(error),
        }
    )
    if record.disclosure_error is not None:
        row["missing_reasons"]["reset_disclosure"] = (
            f"{type(record.disclosure_error).__name__}: {record.disclosure_error}"
        )
    return row


def incomplete_row(
    *,
    kind: str,
    index: int,
    input_ids: Sequence[int],
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Row of a planned request that never ran (same keys as an executed row)."""
    row = _base_row(kind, index, input_ids)
    row["error"] = error
    return row


def run_status(rows: Sequence[Mapping[str, Any]], *, failure_phase: str | None) -> str:
    """``failed`` when a phase failed, ``complete`` when every row succeeded, else ``partial``."""
    if failure_phase is not None:
        return "failed"
    return "complete" if all(row["status"] == "succeeded" for row in rows) else "partial"


def _try_replace(source: Path, target: Path, *, last: bool) -> bool:
    try:
        source.replace(target)
    except PermissionError:
        if last:
            raise
        return False
    return True


class EvidenceWriter:
    """Atomic evidence writes plus the append-only rows sidecar.

    The sidecar is truncated when the writer is created, so a rerun never
    mixes in rows from an earlier run at the same location.

    Args:
        path: Evidence file path; its folder is created when missing.
        sleep: Delay between replace attempts (injectable for tests).
    """

    def __init__(self, path: Path, *, sleep: Callable[[float], None] = time.sleep) -> None:
        self.path = path
        self._sleep = sleep
        path.parent.mkdir(parents=True, exist_ok=True)
        self.rows_path = rows_path(path)
        self._rows = self.rows_path.open("w", encoding="utf-8", newline="\n")

    def write(self, document: Mapping[str, Any]) -> None:
        """Replace the evidence file with ``document``, retrying while a reader holds it.

        Raises:
            ValueError: The document contains NaN or infinity.
            PermissionError: The file stayed locked for the whole retry budget.
        """
        text = json.dumps(document, indent=2, allow_nan=False) + "\n"
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        for attempt in range(_REPLACE_ATTEMPTS):
            if _try_replace(temporary, self.path, last=attempt == _REPLACE_ATTEMPTS - 1):
                return
            self._sleep(_REPLACE_INTERVAL_S)

    def append_row(self, row: Mapping[str, Any]) -> None:
        """Append one request row to the sidecar and flush it (no fsync)."""
        self._rows.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        self._rows.flush()

    def close(self) -> None:
        """Close the rows sidecar; safe to call more than once."""
        self._rows.close()
