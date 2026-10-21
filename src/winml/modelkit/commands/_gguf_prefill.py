# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Opt-in ingestion of correlated, instrumented native GGUF prefill events.

This is a CLI-owned file protocol, not an API supported by stock Windows ML
packages. Only a native writer implementing the documented synchronized,
pre-sampling boundary can produce these measurements. No host span is a fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


_CONTEXT_ENV = "WINMLCLI_GGUF_PREFILL_CONTEXT"
_CONTEXT_SCHEMA = "winml-gguf-prefill-context-1"
_EVENT_SCHEMA = "winml-gguf-prefill-event-1"
_BOUNDARY = "backend_prompt_start_to_synchronized_completion_before_sampling"
_MAX_TRACE_BYTES = 16 * 1024 * 1024
_TRACE_LOCK = threading.Lock()


class PrefillTraceError(ValueError):
    """Opt-in native evidence is invalid or cannot be safely correlated."""


@dataclass(frozen=True)
class NativePrefillObservation:
    """One validated backend duration; it is never a caller/Task clock delta."""

    duration_ns: int
    input_tokens: int
    source: str
    backend_build: str
    clock_domain: str

    def to_dict(self) -> dict[str, Any]:
        """Preserve native precision and the explicit timing boundary."""
        return {
            "duration_ns": self.duration_ns,
            "input_tokens": self.input_tokens,
            "source": self.source,
            "backend_build": self.backend_build,
            "clock_domain": self.clock_domain,
            "clock_monotonic": True,
            "boundary": _BOUNDARY,
            "synchronized": True,
            "before_first_sampling": True,
        }


class NativePrefillTrace:
    """Reserve fresh evidence and publish serialized request correlation.

    Native instrumentation reads the context file before prompt execution and
    appends one completed event before the Task request returns. Reading the
    trace after all requests does not poll or extend recorded Task timers.
    """

    def __init__(self, path: Path, model_path: Path) -> None:
        self.path = Path(path).absolute()
        self.context_path = self.path.with_name(self.path.name + ".context.json")
        self.model_path = Path(model_path).resolve()
        self.run_id = uuid4().hex
        self._context: dict[str, Any] = {}
        self._requests: dict[str, dict[str, Any]] = {}
        self._active = False
        self._entered = False
        self._previous_env: str | None = None
        self._trace_sha256: str | None = None

    def __enter__(self) -> NativePrefillTrace:
        if not _TRACE_LOCK.acquire(blocking=False):
            raise PrefillTraceError(
                "Native prefill request context is already in use in this process"
            )
        try:
            if (
                self.path.resolve() == self.model_path
                or self.context_path.resolve() == self.model_path
            ):
                raise PrefillTraceError("Native prefill files must not overwrite the model")
            if self.path.exists() or self.context_path.exists():
                raise PrefillTraceError("Native prefill trace and context paths must both be fresh")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.model_path.open("rb") as stream:
                model_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            self._context = {
                "schema_version": _CONTEXT_SCHEMA,
                "event_schema_version": _EVENT_SCHEMA,
                "trace_path": str(self.path),
                "run_id": self.run_id,
                "pid": os.getpid(),
                "model_sha256": model_sha256,
                "active": False,
                "request": None,
            }
            # Exclusive creation prevents accidental import/overwrite of old evidence.
            with self.path.open("xb"):
                pass
            with self.context_path.open("x", encoding="utf-8") as stream:
                json.dump(self._context, stream)
            self._previous_env = os.environ.get(_CONTEXT_ENV)
            os.environ[_CONTEXT_ENV] = str(self.context_path)
            self._entered = True
            return self
        except BaseException as exc:
            _TRACE_LOCK.release()
            if isinstance(exc, PrefillTraceError):
                raise
            if isinstance(exc, (OSError, ValueError)):
                raise PrefillTraceError(f"Cannot prepare native prefill evidence: {exc}") from exc
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self._active:
                self.finish_request()
        finally:
            if self._previous_env is None:
                os.environ.pop(_CONTEXT_ENV, None)
            else:
                os.environ[_CONTEXT_ENV] = self._previous_env
            self._entered = False
            _TRACE_LOCK.release()

    def _publish(self) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.context_path.parent,
                prefix=self.context_path.name + ".",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(self._context, stream)
            temporary.replace(self.context_path)
        except OSError as exc:
            raise PrefillTraceError(f"Cannot publish native request context: {exc}") from exc
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def prepare_request(self, kind: str, index: int, prompt_ids: list[int]) -> None:
        """Publish correlation before generation, outside the recorded Task span."""
        if not self._entered or self._active:
            raise PrefillTraceError(
                "Native prefill requests must be serialized in an active context"
            )
        if kind not in ("warmup", "timed") or type(index) is not int or index < 0:
            raise PrefillTraceError("Invalid native prefill request identity")
        if any(row["kind"] == kind and row["index"] == index for row in self._requests.values()):
            raise PrefillTraceError("Duplicate native prefill request identity")
        if not prompt_ids or any(type(token) is not int or token < 0 for token in prompt_ids):
            raise PrefillTraceError(
                "Native prefill correlation requires actual nonnegative token IDs"
            )
        payload = json.dumps(prompt_ids, separators=(",", ":")).encode("utf-8")
        request_id = uuid4().hex
        request = {
            "request_id": request_id,
            "kind": kind,
            "index": index,
            "prompt_tokens": len(prompt_ids),
            "prompt_sha256": hashlib.sha256(payload).hexdigest(),
        }
        self._requests[request_id] = request
        self._context.update(active=True, request=request)
        self._active = True
        self._publish()

    def finish_request(self) -> None:
        """Deactivate correlation after the Task terminal result is observed."""
        if not self._entered or not self._active:
            raise PrefillTraceError("No active native prefill request")
        self._context.update(active=False, request=None)
        self._active = False
        self._publish()

    def read_observations(self) -> dict[tuple[str, int], NativePrefillObservation]:
        """Validate supplied events; missing events are unavailable, not errors."""
        if not self._entered or self._active:
            raise PrefillTraceError("Read native events after all active requests complete")
        try:
            with self.path.open("rb") as stream:
                raw = stream.read(_MAX_TRACE_BYTES + 1)
        except OSError as exc:
            raise PrefillTraceError(f"Cannot read native prefill trace: {exc}") from exc
        if len(raw) > _MAX_TRACE_BYTES:
            raise PrefillTraceError("Native prefill trace exceeds the 16 MiB limit")
        self._trace_sha256 = hashlib.sha256(raw).hexdigest()
        result: dict[tuple[str, int], NativePrefillObservation] = {}
        producer: tuple[str, str, str] | None = None
        try:
            lines = raw.decode("utf-8").splitlines()
            if len(lines) > len(self._requests):
                raise PrefillTraceError("Native trace contains more events than requests")
            for line in lines:
                event = json.loads(line)
                request, observation = self._validate_event(event)
                key = (request["kind"], request["index"])
                if key in result:
                    raise PrefillTraceError("Duplicate native prefill event")
                current = (observation.source, observation.backend_build, observation.clock_domain)
                if producer is not None and producer != current:
                    raise PrefillTraceError(
                        "Native prefill source/build/clock changed within a run"
                    )
                producer = current
                result[key] = observation
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PrefillTraceError(f"Malformed native prefill JSONL: {exc}") from exc
        return result

    def _validate_event(self, event: Any) -> tuple[dict[str, Any], NativePrefillObservation]:
        if not isinstance(event, dict):
            raise PrefillTraceError("Native prefill event must be a JSON object")
        if event.get("schema_version") != _EVENT_SCHEMA or event.get("phase") != "prefill":
            raise PrefillTraceError("Unsupported native prefill event schema/phase")
        for key in ("run_id", "pid", "model_sha256"):
            if (
                type(event.get(key)) is not type(self._context[key])
                or event[key] != self._context[key]
            ):
                raise PrefillTraceError(f"Native prefill event {key} differs from the active run")
        request_id = event.get("request_id")
        request = self._requests.get(request_id) if isinstance(request_id, str) else None
        if request is None:
            raise PrefillTraceError("Native prefill event belongs to an unknown request")
        for key in ("kind", "index", "prompt_sha256", "prompt_tokens"):
            if type(event.get(key)) is not type(request[key]) or event[key] != request[key]:
                raise PrefillTraceError(f"Native prefill event {key} differs from submitted input")
        if (
            event.get("boundary") != _BOUNDARY
            or event.get("synchronized") is not True
            or event.get("before_first_sampling") is not True
            or event.get("clock_monotonic") is not True
        ):
            raise PrefillTraceError("Native prefill event does not establish the required boundary")
        duration = event.get("duration_ns")
        if type(duration) is not int or not 0 < duration < 2**63:
            raise PrefillTraceError("Native prefill duration_ns must be a positive int64")
        for key in ("source", "backend_build", "clock_domain"):
            if not isinstance(event.get(key), str) or not event[key].strip():
                raise PrefillTraceError(f"Native prefill event requires {key}")
        return request, NativePrefillObservation(
            duration_ns=duration,
            input_tokens=event["prompt_tokens"],
            source=event["source"],
            backend_build=event["backend_build"],
            clock_domain=event["clock_domain"],
        )

    def metadata(self) -> dict[str, Any]:
        """Return provenance without treating a boundary declaration as verification."""
        return {
            "enabled": True,
            "source": "external_native_trace",
            "external_native_writer_required": True,
            "event_schema_version": _EVENT_SCHEMA,
            "boundary": _BOUNDARY,
            "run_id": self.run_id,
            "trace_path": str(self.path),
            "trace_sha256": self._trace_sha256,
            "context_path": str(self.context_path),
            "model_sha256": self._context.get("model_sha256"),
            "model_hash_scope": "entrypoint_file",
            "model_hash_before_load": True,
            "load_cache_caveat": (
                "Model provenance hashing reads the file before load; not a cold-cache run"
            ),
            "instrumented": True,
            "percentile_method": "nearest-rank",
        }
