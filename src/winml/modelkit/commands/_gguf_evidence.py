# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""``gguf-task-events-2`` evidence for GGUF runs of ``winml perf --evidence``.

The evidence is one JSON document, written twice, each time to a temporary
file that then replaces it. The start write names this process before the
Runtime is imported, so an external observer can bind to it before the model
loads. The end write, made also when the run fails, adds the outcome, the
workload as the CLI applied it, the installed package versions, the raw clock
readings of every finished request and the native log capture. A benchmark
harness recomputes every metric from these readings; nothing here aggregates.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import TYPE_CHECKING, Any

from .. import __version__
from ..session import llama_package_versions


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from ..session import LlamaGenerationTiming
    from ._gguf_native_log import NativeLogCapture
    from ._perf_gguf import GGUFPerfConfig


EVIDENCE_SCHEMA_VERSION = "gguf-task-events-2"

# Windows refuses to replace a file that another process (a reader, antivirus) holds.
_REPLACE_ATTEMPTS = 20
_REPLACE_INTERVAL_S = 0.05


def temporary_path(evidence_path: Path) -> Path:
    """The file each evidence write goes to before it replaces the evidence file."""
    return evidence_path.with_name(evidence_path.name + ".tmp")


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """SHA-256 of the compact JSON array of token IDs (the benchmark's prompt hash)."""
    return hashlib.sha256(json.dumps(list(token_ids), separators=(",", ":")).encode()).hexdigest()


def _ns(seconds: float) -> int:
    """A ``time.perf_counter()`` reading as integer nanoseconds on the same QPC clock."""
    return round(seconds * 1e9)


def request_row(kind: str, index: int, timing: LlamaGenerationTiming) -> dict[str, Any]:
    """The evidence row of a finished request, from its ``generate_timed`` readings.

    ``index`` is the request's position in the run, as in the schema 3 report.
    """
    return {
        "kind": kind,
        "index": index,
        "submit_ns": _ns(timing.submit_time),
        "completed_ns": _ns(timing.completed_time),
        "output_events": [
            {"observed_ns": _ns(observed), "token_id": token}
            for observed, token in zip(timing.token_times, timing.token_ids, strict=True)
        ],
        "stop_reason": timing.stop_reason,
        "prompt_tokens": timing.input_tokens,
        "cached_tokens": timing.cached_tokens,
    }


def _try_replace(source: Path, target: Path, *, last: bool) -> bool:
    try:
        source.replace(target)
    except PermissionError:
        if last:
            raise
        return False
    return True


def _write_atomic(
    path: Path, document: Mapping[str, Any], *, sleep: Callable[[float], None] = time.sleep
) -> None:
    """Replace ``path`` with ``document``, retrying while another process holds the file.

    Raises:
        ValueError: The document contains NaN or infinity.
        PermissionError: The file stayed locked for the whole retry budget.
    """
    temporary = temporary_path(path)
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(_REPLACE_ATTEMPTS):
        if _try_replace(temporary, path, last=attempt == _REPLACE_ATTEMPTS - 1):
            return
        sleep(_REPLACE_INTERVAL_S)


def _header(status: str) -> dict[str, Any]:
    return {"schema_version": EVIDENCE_SCHEMA_VERSION, "status": status, "pid": os.getpid()}


class EvidenceRecorder:
    """Write the evidence of one GGUF ``--evidence`` run.

    Construction makes the start write (status ``running``); :meth:`finish`
    makes the end write.

    Args:
        config: The run configuration. ``evidence_path`` names the file (its
            folder is created when missing); the end write echoes the workload.
        capture: The native log capture; the end write holds its fragments.

    Raises:
        ValueError: ``config.evidence_path`` is not set.
    """

    def __init__(self, config: GGUFPerfConfig, *, capture: NativeLogCapture) -> None:
        if config.evidence_path is None:
            raise ValueError("EvidenceRecorder needs config.evidence_path")
        self.path = config.evidence_path
        self.capture = capture
        self._config = config
        self._rows: list[dict[str, Any]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(self.path, _header("running"))

    def record_request(self, kind: str, index: int, timing: LlamaGenerationTiming) -> None:
        """Add the row of one finished request."""
        self._rows.append(request_row(kind, index, timing))

    def finish(self, error: BaseException | None) -> None:
        """Restore the Runtime log callback, then make the end write.

        Args:
            error: What ended the run, or ``None`` when every request finished.
        """
        self.capture.restore()
        config = self._config
        ids = config.prompt_ids
        failure = None if error is None else {"type": type(error).__name__, "message": str(error)}
        # Fragments stamped before the first request's submit belong to the model load.
        first_submit_ns = self._rows[0]["submit_ns"] if self._rows else None
        document = {
            **_header("complete" if error is None else "failed"),
            "error": failure,
            "workload": {
                "input_ids_sha256": None if ids is None else token_ids_sha256(ids),
                "input_token_count": None if ids is None else len(ids),
                "max_new_tokens": config.max_new_tokens,
                "context_length": config.context_length,
                "ignore_eos": config.ignore_eos,
                "warmup": config.warmup,
                "iterations": config.iterations,
            },
            "packages": {**llama_package_versions(), "winml-cli": __version__},
            "requests": self._rows,
            "native_log": self.capture.to_dict(first_submit_ns),
        }
        _write_atomic(self.path, document)
