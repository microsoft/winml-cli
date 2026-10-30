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

import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from ._gguf_identity import (
    clock_info,
    environment_snapshot,
    hash_modules,
    list_loaded_modules,
    model_inventory,
    package_versions,
    process_identity,
    producer_info,
    read_manifests,
    windowsml_package_dir,
)
from ._gguf_native_log import NATIVE_VALUE_KEYS, parse_native_values


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from ..session import LlamaRequestRecord, WinMLLlamaSession
    from ._gguf_native_log import NativeLogCapture


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

CACHE_STATE_EVIDENCE = (
    "Every model shard was hashed before initialization, which can warm the OS file cache; "
    "Windows ML 2.7.2021a0 reports no file or compilation cache state."
)
_PHYSICAL_ADAPTER_REASON = (
    "ExecutionTarget exposes only kind; the benchmark attributes the adapter externally."
)
_SAMPLING_REASON = "Windows ML 2.7.2021a0 has no TextGenerationOptions read-back."
_THREADS_UNKNOWN = "No Task/Runtime API and no libllama log line."
_NATIVE_LINE_MISSING = "No matching libllama log line in the capture window."
# Top-level, workload and configuration keys in contract order.
_WORKLOAD_KEYS = (
    "input_mode",
    "input_ids_sha256",
    "input_token_count",
    "max_new_tokens",
    "context_length",
    "ignore_eos",
    "warmup",
    "iterations",
    "sampling",
)
_CONFIGURATION_KEYS = (
    "model",
    "context",
    "speculative",
    "device",
    "eos",
    "sampling",
    "native",
    "packages",
    "manifests",
    "loaded_modules",
    "hashed_modules",
    "ort_sidecar_loaded",
    "environment",
    "in_process_resource_work",
)
# Failure phases during which a planned request was being executed.
_REQUEST_PHASES = frozenset({"warmup", "measured"})

# A reader holding the evidence file open makes the atomic replace fail on Windows.
_REPLACE_ATTEMPTS = 20
_REPLACE_INTERVAL_S = 0.05


def rows_path(evidence_path: Path) -> Path:
    """The rows sidecar of an evidence file (``task-events.json`` -> ``task-events.rows.jsonl``)."""
    return evidence_path.with_name(evidence_path.stem + ".rows.jsonl")


def temporary_path(evidence_path: Path) -> Path:
    """The file each evidence write goes to before it replaces the evidence file."""
    return evidence_path.with_name(evidence_path.name + ".tmp")


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
        temporary = temporary_path(self.path)
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


# =============================================================================
# Evidence document
# =============================================================================


class EvidenceRunConfig(Protocol):
    """The benchmark configuration fields an evidence run reads."""

    @property
    def model_path(self) -> Path:
        """The ``.gguf`` entry file (the first shard of a split model)."""

    @property
    def device(self) -> str:
        """The device the llama.cpp backend targets (``cpu`` or ``gpu``)."""

    @property
    def evidence_path(self) -> Path | None:
        """The evidence file to write."""

    @property
    def prompt_ids(self) -> tuple[int, ...] | None:
        """Exact prompt token IDs, or ``None`` when the prompt text is encoded."""

    @property
    def context_length(self) -> int | None:
        """The ``--context-length`` value, or ``None`` when it was not given."""

    @property
    def ignore_eos(self) -> bool:
        """Whether the Task is created with an empty EOS set."""

    @property
    def warmup(self) -> int:
        """Planned warmup requests."""

    @property
    def iterations(self) -> int:
        """Planned measured requests."""

    @property
    def max_new_tokens(self) -> int:
        """Token limit of each request."""


def _ordered(values: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    ordered = {key: values[key] for key in keys if key in values}
    ordered.update({key: value for key, value in values.items() if key not in ordered})
    return ordered


def _read_token_ids(read: Callable[[], Sequence[int]]) -> tuple[list[int] | None, str | None]:
    try:
        return [int(token) for token in read()], None
    except Exception as exc:  # a failed read-back is evidence, not a run failure
        return None, f"{type(exc).__name__}: {exc}"


class EvidenceRecorder:
    """Assemble a ``gguf-task-events-1.1`` document while a protocol run proceeds.

    The provisional document (status ``running``, identifying this process) is
    written on construction. Each phase adds what it observed; :meth:`finish`
    fills planned requests that never ran with ``incomplete`` rows and writes
    the final document.

    Args:
        config: The run configuration; ``evidence_path`` must be set.
        clock_ns: Integer-nanosecond clock for the start and init timestamps.
        writer: Evidence writer (default: one for ``config.evidence_path``).

    Raises:
        ValueError: ``config.evidence_path`` is not set.
    """

    def __init__(
        self,
        config: EvidenceRunConfig,
        *,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        writer: EvidenceWriter | None = None,
    ) -> None:
        if config.evidence_path is None:
            raise ValueError("EvidenceRecorder needs config.evidence_path")
        self._config = config
        self._clock_ns = clock_ns
        started_qpc_ns = clock_ns()
        self._header: dict[str, Any] = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "producer": producer_info(),
            "host_api": HOST_API,
            "process": process_identity(),
            "clock": clock_info(started_qpc_ns),
        }
        self._writer = writer if writer is not None else EvidenceWriter(config.evidence_path)
        self._status = "running"
        self._failure_phase: str | None = None
        self._cause: BaseException | None = None
        self._workload: dict[str, Any] = {}
        self._configuration: dict[str, Any] = {}
        self._initialization: dict[str, Any] = {}
        self._rows: list[dict[str, Any]] = []
        self._harness: dict[str, Any] = {}
        self._native_log: dict[str, Any] = {}
        self._missing_reasons: dict[str, str] = {}
        self._cache_state: dict[str, Any] = {}
        self._prompt_ids: list[int] | None = None
        self._init_start_ns: int | None = None
        self._package_dir: Path | None = None
        self.warmup0_submit_ns: int | None = None
        self._writer.write(self._document())

    # -- state ------------------------------------------------------------------
    @property
    def path(self) -> Path:
        """The evidence file."""
        return self._writer.path

    @property
    def status(self) -> str:
        """``running`` until :meth:`finish`, then ``complete``, ``partial`` or ``failed``."""
        return self._status

    @property
    def failure_phase(self) -> str | None:
        """Phase of the first recorded failure."""
        return self._failure_phase

    @property
    def cause(self) -> BaseException | None:
        """The first recorded failure."""
        return self._cause

    @property
    def error(self) -> dict[str, Any] | None:
        """The first recorded failure as an evidence error object."""
        return None if self._cause is None else error_object(self._cause)

    # -- phases -----------------------------------------------------------------
    def record_configuration(self) -> None:
        """Record the requested workload and the facts known before initialization.

        Model shards are hashed last because a missing shard fails this phase.
        """
        config = self._config
        self._workload.update(
            {
                "input_mode": "token_ids" if config.prompt_ids is not None else "text",
                "max_new_tokens": config.max_new_tokens,
                "context_length": config.context_length,
                "ignore_eos": config.ignore_eos,
                "warmup": config.warmup,
                "iterations": config.iterations,
            }
        )
        if config.prompt_ids is not None:
            self.record_prompt(config.prompt_ids)
        self._package_dir = windowsml_package_dir()
        self._configuration.update(
            {
                "device": {"requested": config.device},
                "packages": package_versions(),
                "manifests": read_manifests(self._package_dir),
                "environment": environment_snapshot(os.environ),
                "in_process_resource_work": False,
            }
        )
        self._configuration["model"] = model_inventory(Path(config.model_path))

    def begin_initialization(self) -> None:
        """Mark the start of initialization (before the Runtime is imported)."""
        self._init_start_ns = self._clock_ns()

    def record_prompt(self, prompt_ids: Sequence[int]) -> None:
        """Record the token IDs every request submits."""
        self._prompt_ids = [int(token) for token in prompt_ids]
        self._workload["input_ids_sha256"] = token_ids_sha256(self._prompt_ids)
        self._workload["input_token_count"] = len(self._prompt_ids)

    def record_load(self, session: WinMLLlamaSession, *, requested_context: int) -> None:
        """Record what the built session reports back about its configuration."""
        readback = session.speculative_readback
        self._configuration["context"] = {
            "requested": requested_context,
            "effective": session.context_length,
            "declared": session.declared_context_length,
        }
        self._configuration["speculative"] = {
            "model_draft_predictor_enabled": readback.get("model_draft_predictor_enabled"),
            "draft_token_limit_hint": readback.get("draft_token_limit_hint"),
            "draft_predictor_path": readback.get("draft_predictor_path"),
            "source": "stage_readback_not_set_by_cli",
            "reason": readback.get("reason"),
        }
        device = self._configuration.setdefault("device", {"requested": self._config.device})
        device["effective_target_kind"] = session.resolved_target_kind
        device["physical_adapter"] = {"effective": "unknown", "reason": _PHYSICAL_ADAPTER_REASON}
        task_eos, task_reason = _read_token_ids(lambda: session.task_default_eos_token_ids)
        tokenizer_eos, tokenizer_reason = _read_token_ids(lambda: session.tokenizer_eos_token_ids)
        reasons = [reason for reason in (task_reason, tokenizer_reason) if reason]
        self._configuration["eos"] = {
            "requested_policy": "ignore" if self._config.ignore_eos else "default",
            "task_default_eos_token_ids": task_eos,
            "tokenizer_eos_token_ids": tokenizer_eos,
            "stop_token_ids": [],
            "reason": "; ".join(reasons) or None,
        }
        self._configuration["sampling"] = {"effective": "unverified", "reason": _SAMPLING_REASON}

    def record_options(self, options: Any) -> None:
        """Record every field of the Task options each request sends."""
        sampling = dataclasses.asdict(options)
        sampling["stop_token_ids"] = list(sampling["stop_token_ids"])
        self._workload["sampling"] = sampling

    def record_request(self, kind: str, index: int, record: LlamaRequestRecord) -> dict[str, Any]:
        """Add one executed request to the document and the rows sidecar."""
        row = request_row(
            record,
            kind=kind,
            index=index,
            max_new_tokens=self._config.max_new_tokens,
            ignore_eos=self._config.ignore_eos,
        )
        self._append(row)
        if (kind, index) == ("warmup", 0):
            self.warmup0_submit_ns = record.submit_ns
        return row

    def record_loaded_modules(self, stage: Literal["after_warmup0", "at_end"]) -> None:
        """Record the modules loaded in this process now."""
        modules = list_loaded_modules()
        self._configuration.setdefault("loaded_modules", {})[stage] = modules
        if stage == "at_end":
            self._configuration["ort_sidecar_loaded"] = any(
                Path(module).name.lower() == "onnxruntime.dll" for module in modules
            )

    def record_hashed_modules(self) -> None:
        """Hash the loaded ``windowsml`` modules and compare them with the manifests."""
        modules = self._configuration.get("loaded_modules", {}).get("at_end")
        if modules is None:
            modules = list_loaded_modules()
        package_dir = self._package_dir
        if package_dir is None:
            package_dir = windowsml_package_dir()
        manifests = self._configuration.get("manifests")
        if manifests is None:
            manifests = read_manifests(package_dir)
        self._configuration["hashed_modules"] = hash_modules(modules, package_dir, manifests)

    def fail(self, phase: str, exc: BaseException) -> None:
        """Record a failure; only the first one determines the phase and error."""
        if self._failure_phase is None:
            self._failure_phase = phase
            self._cause = exc

    def finish(
        self,
        *,
        capture: NativeLogCapture,
        initialization_spans: Sequence[tuple[str, int, int]],
        gc_collections: int,
    ) -> str:
        """Complete the document, write it atomically, and return the run status.

        Planned requests that never ran become ``incomplete`` rows (in the
        document and the sidecar); the first carries the error when the run
        failed during a request phase.
        """
        planned = planned_requests(self._config.warmup, self._config.iterations)
        for position, (kind, index) in enumerate(planned[len(self._rows) :]):
            error = None
            if position == 0 and self._failure_phase in _REQUEST_PHASES and self._cause:
                error = error_object(self._cause)
            planned_ids = self._prompt_ids or []
            self._append(incomplete_row(kind=kind, index=index, input_ids=planned_ids, error=error))
        self._native_log = capture.to_dict(self.warmup0_submit_ns)
        values = parse_native_values(self._native_log["lines"])
        missing = _NATIVE_LINE_MISSING if capture.installed else capture.reason or ""
        self._configuration["native"] = {
            "source": "native-log",
            "values": values,
            "unknown": {
                "threads": _THREADS_UNKNOWN,
                **{key: missing for key in NATIVE_VALUE_KEYS if key not in values},
            },
        }
        self._initialization = {
            "init_start_ns": self._init_start_ns,
            "spans": [
                {"name": name, "start_ns": start, "end_ns": end}
                for name, start, end in initialization_spans
            ],
            "ready": None,
        }
        self._harness = {
            "gc": {
                "policy": "disabled_during_measured_requests",
                "collections_between_requests": gc_collections,
            }
        }
        self._missing_reasons = {"load": MISSING_LOAD, "ready": MISSING_READY}
        self._cache_state = {
            "file": "unknown",
            "compilation": "unknown",
            "evidence": CACHE_STATE_EVIDENCE,
        }
        self._status = run_status(self._rows, failure_phase=self._failure_phase)
        try:
            self._writer.write(self._document())
        finally:
            self._writer.close()
        return self._status

    # -- internals --------------------------------------------------------------
    def _append(self, row: dict[str, Any]) -> None:
        self._rows.append(row)
        self._writer.append_row(row)

    def _document(self) -> dict[str, Any]:
        return {
            **self._header,
            "status": self._status,
            "failure_phase": self._failure_phase,
            "error": self.error,
            "workload": _ordered(self._workload, _WORKLOAD_KEYS),
            "configuration": _ordered(self._configuration, _CONFIGURATION_KEYS),
            "initialization": self._initialization,
            "requests": list(self._rows),
            "harness": self._harness,
            "native_log": self._native_log,
            "load": None,
            "missing_reasons": self._missing_reasons,
            "cache_state": self._cache_state,
        }
