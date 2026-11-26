# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Text generation benchmarking of an OpenAI-compatible server (experimental).

``winml perf --endpoint <base-url>`` times a server that is already running,
such as llama.cpp's ``llama-server``, through the standard streaming
``/v1/chat/completions`` API. The server owns the model, its chat template and
its tokenizer; this module only sends text prompts and measures what arrives.

The report is the schema 3 generation layout that GGUF and CGC Task runs use
(:class:`~._task_generation_perf.GenerationBenchmarkResult`), with
``benchmark_info.runtime: openai-endpoint`` and ``timing_source: http-stream``.
The client measures the shared request metrics:

* ``request_ttft_duration_ms``: from sending the request to the first streamed
  event that carries generated text;
* ``request_duration_ms``: from sending the request to the end of the stream;
* ``decode_token_durations_ms``: gaps between events that carry generated text;
* ``steady_state_decode_tokens_per_second`` and ``steady_state_tpot_ms``:
  ``generated_tokens - 1`` over the time from the first to the last text event.
  Speculative-decoding servers may stream several tokens per event, so these
  come from the token count, not the number of events. At one token per event
  they equal the Task stream values.

Task and model spans the client cannot observe are null. What only a server
reports (cached prompt tokens, llama.cpp-style ``timings`` with draft tokens)
goes in additive fields and blocks, never under a schema 3 metric name. A
``provenance`` block (host, run labels, run index, file hashes, server model
list) lets reports from different machines and runs be compared.
"""

from __future__ import annotations

import contextlib
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from importlib import metadata
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar

import click
from rich.markup import escape

from .. import __version__
from ._generation_metrics import aggregate_requests, request_counts
from ._openai_chat_client import (
    ChatStreamTiming,
    EndpointError,
    OpenAIChatClient,
    scrub_url,
)
from ._task_generation_perf import (
    _MEASURED_REQUEST_METRICS,
    _UNAVAILABLE_MODEL_METRICS,
    GenerationBenchmarkResult,
    _request_report_fields,
    run_task_perf,
)


if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from rich.console import Console

logger = logging.getLogger(__name__)

RUNTIME = "openai-endpoint"
_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
# A drive letter, a leading separator, '.' or '~', or any backslash marks a file path.
_PATH_LIKE = re.compile(r"^(?:[A-Za-z]:|[\\/~.])|\\")
_MODEL_FILE_SUFFIXES = (".gguf", ".onnx", ".safetensors", ".bin")

# Schema 3 request metrics the client measures; the other schema 3 metrics are null.
_CLIENT_METRICS = (
    "request_ttft_duration_ms",
    "request_duration_ms",
    "steady_state_tpot_ms",
    "steady_state_decode_tokens_per_second",
)
# Server ``timings`` rates summarized under ``aggregate.server``.
_SERVER_RATE_METRICS = (
    "prefill_tokens_per_second",
    "decode_tokens_per_second",
    "draft_acceptance_rate",
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EndpointPerfConfig:
    """Resolved request for an endpoint text generation benchmark.

    Attributes:
        base_url: The server's OpenAI base URL, for example ``http://127.0.0.1:8080/v1``.
        model: The ``model`` field to send; ``None`` uses the only model the
            server lists.
        prompt: Text sent as the single user message of every request.
        prompt_source: Where the prompt came from (``--prompt`` or ``--prompt-file``).
        max_new_tokens: ``max_tokens`` of every request.
        iterations: Timed requests.
        warmup: Warmup requests excluded from statistics.
        extra_body: JSON object merged into every request body.
        run_labels: Free-form ``KEY=VALUE`` labels recorded in the report.
        run_index: Position of this run in a repeated series, when given.
        file_hashes: Labelled SHA-256 hashes of files the user named.
        output_path: The JSON report path; ``None`` picks a default once the
            served model is known.
    """

    base_url: str
    model: str | None = None
    prompt: str = ""
    prompt_source: str = "--prompt"
    max_new_tokens: int = 128
    iterations: int = 10
    warmup: int = 2
    extra_body: dict[str, Any] = field(default_factory=dict)
    run_labels: dict[str, str] = field(default_factory=dict)
    run_index: int | None = None
    file_hashes: dict[str, dict[str, Any]] = field(default_factory=dict)
    output_path: Path | None = None


def report_model_id(model_id: str) -> str:
    """Return a served model id fit for a shared report.

    Servers such as ``llama-server`` report the model's file path as its id
    unless given an alias; such an id is reduced to its file name so reports
    carry no local folders. Other ids (for example ``org/model``) are kept.
    """
    if _PATH_LIKE.search(model_id) or model_id.lower().endswith(_MODEL_FILE_SUFFIXES):
        return re.split(r"[\\/]", model_id.rstrip("\\/"))[-1] or model_id
    return model_id


def _report_model_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Keep the descriptive fields of a ``/models`` entry, without local paths."""
    kept: dict[str, Any] = {
        key: entry[key] for key in ("object", "owned_by", "created") if key in entry
    }
    if entry.get("id") is not None:
        kept["id"] = report_model_id(str(entry["id"]))
    if isinstance(entry.get("meta"), dict):
        kept["meta"] = {
            key: value
            for key, value in entry["meta"].items()
            if isinstance(value, int | float | bool)
            or value is None
            or (isinstance(value, str) and len(value) <= 64 and not _PATH_LIKE.search(value))
        }
    return kept


def default_report_slug(model_id: str) -> str:
    """Return a file-system safe slug for a served model id.

    A model id that is a file path (as ``llama-server`` reports by default) is
    reduced to its file stem.
    """
    name = report_model_id(model_id)
    if name.lower().endswith(_MODEL_FILE_SUFFIXES):
        name = name.rsplit(".", 1)[0]
    slug = _SLUG_UNSAFE.sub("_", name).strip("._") or "model"
    return f"endpoint-{slug}"


# =============================================================================
# Request samples
# =============================================================================


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _per_second(tokens: float | None, duration_ms: float | None) -> float | None:
    if tokens is None or duration_ms is None or duration_ms <= 0:
        return None
    return tokens * 1000.0 / duration_ms


def normalize_server_timings(timings: dict[str, Any] | None) -> dict[str, float | None]:
    """Map a llama.cpp-style ``timings`` object to the ``server`` report block.

    llama.cpp splits prefill from decode at a different point than the schema 3
    ``prefill_*`` and ``response_eval_*`` metrics, so these stay server fields.
    The first generated token comes from the prompt batch, so ``predicted_ms``
    spans ``predicted_n - 1`` decode steps, as llama.cpp's own rate counts them.
    Unknown or missing keys give ``None``; the raw object is kept separately.
    """
    t = timings or {}
    prompt_n = _number(t.get("prompt_n"))
    prompt_ms = _number(t.get("prompt_ms"))
    predicted_n = _number(t.get("predicted_n"))
    predicted_ms = _number(t.get("predicted_ms"))
    decode_n = predicted_n - 1 if predicted_n is not None and predicted_n >= 2 else None
    draft_n = _number(t.get("draft_n"))
    accepted = _number(t.get("draft_n_accepted"))
    return {
        "prefill_tokens": prompt_n,
        "prefill_duration_ms": prompt_ms,
        "prefill_tokens_per_second": _per_second(prompt_n, prompt_ms),
        "decode_tokens": decode_n,
        "decode_duration_ms": predicted_ms,
        "decode_tokens_per_second": _per_second(decode_n, predicted_ms),
        "draft_tokens": draft_n,
        "draft_tokens_accepted": accepted,
        "draft_acceptance_rate": (
            accepted / draft_n
            if draft_n is not None and accepted is not None and draft_n > 0
            else None
        ),
    }


@dataclass
class EndpointRequestSample:
    """One request, as measured at the client and reported by the server."""

    kind: str
    index: int
    request_ttft_duration_ms: float | None
    request_duration_ms: float
    token_event_arrivals_ms: list[float]
    prompt_tokens: int | None
    cached_prompt_tokens: int | None
    generated_tokens: int
    generated_tokens_source: str
    stop_reason: str | None
    server_timings: dict[str, Any] | None
    response_text: str = ""
    reasoning_text: str = ""

    @classmethod
    def from_timing(cls, kind: str, index: int, timing: ChatStreamTiming) -> EndpointRequestSample:
        """Build a sample from one streamed response.

        The generated token count comes from ``usage``, else the server
        ``timings``, else the number of streamed text events.
        """
        usage = timing.usage or {}
        prompt_tokens = usage.get("prompt_tokens")
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        generated = usage.get("completion_tokens")
        source = "usage"
        predicted = (timing.server_timings or {}).get("predicted_n")
        if not isinstance(generated, int):
            if isinstance(predicted, int):
                generated, source = predicted, "server_timings"
            else:
                generated, source = len(timing.token_event_arrivals_s), "stream_events"
        return cls(
            kind=kind,
            index=index,
            request_ttft_duration_ms=(
                None if timing.first_token_s is None else timing.first_token_s * 1000.0
            ),
            request_duration_ms=timing.end_s * 1000.0,
            token_event_arrivals_ms=[value * 1000.0 for value in timing.token_event_arrivals_s],
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            cached_prompt_tokens=cached if isinstance(cached, int) else None,
            generated_tokens=generated,
            generated_tokens_source=source,
            stop_reason=timing.finish_reason,
            server_timings=timing.server_timings,
            response_text=timing.response_text,
            reasoning_text=timing.reasoning_text,
        )

    @property
    def decode_token_durations_ms(self) -> list[float]:
        """Gaps between streamed events that carried generated text."""
        return [later - earlier for earlier, later in pairwise(self.token_event_arrivals_ms)]

    @property
    def decode_duration_ms(self) -> float | None:
        """Time from the first to the last event that carried generated text."""
        if len(self.token_event_arrivals_ms) < 2:
            return None
        return self.token_event_arrivals_ms[-1] - self.token_event_arrivals_ms[0]

    @property
    def decode_tokens(self) -> int | None:
        """Tokens generated after the first one."""
        return self.generated_tokens - 1 if self.generated_tokens >= 2 else None

    @property
    def steady_state_decode_tokens_per_second(self) -> float | None:
        """Client-side decode throughput after the first token."""
        tokens = self.decode_tokens
        return _per_second(None if tokens is None else float(tokens), self.decode_duration_ms)

    @property
    def steady_state_tpot_ms(self) -> float | None:
        """Mean time per token after the first one."""
        tokens = self.decode_tokens
        duration = self.decode_duration_ms
        return duration / tokens if tokens and duration is not None and duration > 0 else None

    @property
    def tokens_per_event(self) -> float | None:
        """Generated tokens per streamed text event (above 1 when the server batches tokens)."""
        events = len(self.token_event_arrivals_ms)
        return self.generated_tokens / events if events else None

    @property
    def server(self) -> dict[str, float | None]:
        """Normalized server timings."""
        return normalize_server_timings(self.server_timings)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the schema 3 request fields plus the endpoint-only fields."""
        tokens_per_event = self.tokens_per_event
        result = _request_report_fields(self)
        result.update(
            {
                "cached_prompt_tokens": self.cached_prompt_tokens,
                "generated_tokens_source": self.generated_tokens_source,
                "stop_reason": self.stop_reason,
                "token_event_count": len(self.token_event_arrivals_ms),
                "tokens_per_event": (
                    None if tokens_per_event is None else round(tokens_per_event, 3)
                ),
                "server": {
                    key: None if value is None else round(value, 3)
                    for key, value in self.server.items()
                },
                "server_timings": self.server_timings,
                "response_text": self.response_text,
            }
        )
        if self.reasoning_text:
            result["reasoning_text"] = self.reasoning_text
        return result


def aggregate_endpoint_requests(samples: list[EndpointRequestSample]) -> dict[str, Any]:
    """Summarize the timed requests in the schema 3 aggregate layout.

    Schema 3 metrics the client cannot observe are null. ``server`` summarizes
    the server ``timings`` rates, and ``totals`` pools tokens and time across
    requests, which weighs long responses more than averaging per-request rates.
    """
    timed = [sample for sample in samples if sample.kind == "timed"]

    def total(values: list[float | int | None]) -> float | None:
        present = [value for value in values if value is not None]
        return float(sum(present)) if present else None

    decode_pairs = [
        (sample.decode_tokens, sample.decode_duration_ms)
        for sample in timed
        if sample.decode_tokens is not None and sample.decode_duration_ms is not None
    ]
    decode_tokens = total([tokens for tokens, _ in decode_pairs])
    decode_ms = total([duration for _, duration in decode_pairs])
    draft = total([sample.server["draft_tokens"] for sample in timed])
    accepted = total([sample.server["draft_tokens_accepted"] for sample in timed])
    server_views = [SimpleNamespace(kind=sample.kind, **sample.server) for sample in samples]
    return {
        **request_counts(samples),
        # No model load is observed, so there is no cold start.
        "cold_start_ttft_duration_ms": None,
        "cold_start_total_duration_ms": None,
        **dict.fromkeys(_UNAVAILABLE_MODEL_METRICS),
        **dict.fromkeys(_MEASURED_REQUEST_METRICS),
        **aggregate_requests(samples, (*_CLIENT_METRICS, "tokens_per_event")),
        "stop_reason_counts": dict(
            sorted(Counter(sample.stop_reason or "none" for sample in timed).items())
        ),
        "server": aggregate_requests(server_views, _SERVER_RATE_METRICS),
        "totals": {
            "prompt_tokens": total([sample.prompt_tokens for sample in timed]),
            "cached_prompt_tokens": total([sample.cached_prompt_tokens for sample in timed]),
            "generated_tokens": total([sample.generated_tokens for sample in timed]),
            "decode_tokens": decode_tokens,
            "decode_duration_ms": decode_ms,
            "steady_state_decode_tokens_per_second": _per_second(decode_tokens, decode_ms),
            "draft_tokens": draft,
            "draft_tokens_accepted": accepted,
            "draft_acceptance_rate": accepted / draft if draft and accepted is not None else None,
        },
    }


# =============================================================================
# Result
# =============================================================================


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"winml-cli": __version__}
    try:
        versions["httpx"] = metadata.version("httpx")
    except metadata.PackageNotFoundError:
        versions["httpx"] = None
    return versions


def _format_metric(value: float | None, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


@dataclass
class EndpointBenchmarkResult(GenerationBenchmarkResult[EndpointPerfConfig]):
    """Schema 3 results of an endpoint text generation benchmark."""

    RUNTIME: ClassVar[str] = RUNTIME
    API: ClassVar[str] = "openai.chat.completions"
    TIMING_SOURCE: ClassVar[str] = "http-stream"
    TTFT_METRIC: ClassVar[str] = "request_ttft_duration_ms"
    TTFT_TITLE: ClassVar[str] = "Time to first token"
    DECODE_LABEL: ClassVar[str] = "Decode (client)"

    model: str = ""
    served_model: str | None = None
    server_models: list[dict[str, Any]] = field(default_factory=list)
    system_fingerprint: str | None = None
    host: dict[str, Any] = field(default_factory=dict)
    requests: list[EndpointRequestSample] = field(default_factory=list)
    packages: dict[str, str | None] = field(default_factory=_package_versions)

    @property
    def measured_model(self) -> str:
        """The model the server answered with, else the requested one.

        Some servers, ``llama-server`` among them, answer a request for an
        unknown model with the model they serve, so the request alone does
        not identify what was measured.
        """
        return self.served_model or self.model

    @property
    def model_mismatch(self) -> bool:
        """Whether the server answered with a different model than requested."""
        return self.served_model is not None and self.served_model != self.model

    def _run_fields(self) -> dict[str, Any]:
        return {"model_id": report_model_id(self.measured_model)}

    def _backend_fields(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "requested_model": report_model_id(self.model),
            "endpoint": scrub_url(cfg.base_url),
            "prompt_source": cfg.prompt_source,
            "request_parameters": {
                "temperature": cfg.extra_body.get("temperature", 0),
                "extra_body": dict(cfg.extra_body),
            },
        }

    def _report_sections(self) -> dict[str, Any]:
        """The ``provenance`` block: run identity, host and server."""
        cfg = self.config
        return {
            "provenance": {
                "run_index": cfg.run_index,
                "run_labels": dict(cfg.run_labels),
                "file_hashes": dict(cfg.file_hashes),
                "host": self.host,
                "server": {
                    "endpoint": scrub_url(cfg.base_url),
                    "system_fingerprint": self.system_fingerprint,
                    "models": [_report_model_entry(entry) for entry in self.server_models],
                },
            }
        }

    def identity_rows(self) -> list[tuple[str, str]]:
        """The runtime, the API, the endpoint and the served model."""
        return [
            ("Runtime", f"{self.RUNTIME} (experimental)"),
            ("API", "OpenAI chat completions (streamed, client-side timing)"),
            ("Endpoint", scrub_url(self.config.base_url)),
            ("Model", report_model_id(self.measured_model)),
        ]

    def display_rows(self) -> list[tuple[str, str]]:
        """The server build and the run labels, when known."""
        cfg = self.config
        rows = []
        if self.system_fingerprint:
            rows.append(("Server build", self.system_fingerprint))
        if cfg.run_labels or cfg.run_index is not None:
            labels = ", ".join(f"{key}={value}" for key, value in cfg.run_labels.items())
            index = f"run {cfg.run_index}" if cfg.run_index is not None else ""
            rows.append(("Run", ", ".join(part for part in (index, labels) if part)))
        return rows

    def display_sections(self, console: Console) -> None:
        """Print server timings, speculative decoding and prompt-cache notes."""
        aggregate = self.aggregate
        server = aggregate.get("server") or {}
        totals = aggregate.get("totals") or {}
        decode = server.get("decode_tokens_per_second")
        prefill = server.get("prefill_tokens_per_second")
        per_event = aggregate.get("tokens_per_event") or {}
        if not (
            decode
            or prefill
            or totals.get("draft_tokens")
            or totals.get("cached_prompt_tokens")
            or self.model_mismatch
        ):
            return
        console.print()
        if self.model_mismatch:
            console.print(
                f"[yellow]Warning:[/yellow] requested model "
                f"'{escape(report_model_id(self.model))}', but the server answered with "
                f"'{escape(report_model_id(self.measured_model))}'; the report names the model "
                "the server answered with."
            )
        if decode or prefill:
            console.print(
                "[bold]Server timings:[/bold] "
                f"decode {_format_metric((decode or {}).get('mean'))} tokens/sec  |  "
                f"prefill {_format_metric((prefill or {}).get('mean'))} tokens/sec"
            )
        if totals.get("draft_tokens"):
            rate = (totals.get("draft_acceptance_rate") or 0.0) * 100
            console.print(
                "[bold]Speculative decoding:[/bold] "
                f"{int(totals.get('draft_tokens_accepted') or 0)} of {int(totals['draft_tokens'])} "
                f"draft tokens accepted ({_format_metric(rate, 1)}%)"
                f"  |  {_format_metric(per_event.get('mean'))} tokens per streamed event"
            )
        if totals.get("cached_prompt_tokens"):
            console.print(
                "[yellow]Note:[/yellow] the server reused cached prompt tokens on timed requests, "
                "so TTFT excludes part of the prefill. To measure a full prefill, turn off the "
                'server\'s prompt cache (llama-server: send "cache_prompt": false with '
                "--extra-body; see the OpenAI-compatible server sample for shell quoting)."
            )


# =============================================================================
# Benchmark
# =============================================================================


def resolve_model(config_model: str | None, models: list[dict[str, Any]]) -> str:
    """Pick the ``model`` to send: the requested one, or the only one listed.

    Raises:
        click.UsageError: No model was requested and the server lists none or several.
    """
    if config_model:
        return config_model
    ids = [str(entry["id"]) for entry in models if entry.get("id")]
    if len(ids) == 1:
        return ids[0]
    if not ids:
        raise click.UsageError(
            "The server lists no models. Pass the served model name with --endpoint-model."
        )
    raise click.UsageError(
        "The server lists several models; choose one with --endpoint-model: " + ", ".join(ids)
    )


class EndpointPerfBenchmark:
    """Runs warmup and timed requests against one server, one at a time.

    A runner for :func:`~._task_generation_perf.run_task_perf`.

    Args:
        config: The resolved benchmark request.
        client: A pre-built client (dependency injection for tests). When
            ``None`` one is created from ``config`` and closed after the run.
        collect_host: Returns the host description (injectable for tests).
        on_request: Called with ``(number, total, kind)`` before each request.
        status: Context manager held open while requests run, such as a
            console spinner.
    """

    SESSION_ERROR: ClassVar[type[Exception]] = EndpointError

    def __init__(
        self,
        config: EndpointPerfConfig,
        *,
        client: OpenAIChatClient | None = None,
        collect_host: Callable[[], dict[str, Any]] | None = None,
        on_request: Callable[[int, int, str], None] | None = None,
        status: AbstractContextManager[Any] | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._owns_client = client is None
        self._collect_host = collect_host
        self._on_request = on_request
        self._status = status
        # The measured model once a run finishes; it names the default report.
        self.model: str | None = None

    def run(self) -> EndpointBenchmarkResult:
        """Execute the benchmark.

        Raises:
            EndpointError: The server failed a request.
            click.UsageError: The served model could not be resolved.
        """
        client = self._client or OpenAIChatClient(self._config.base_url)
        try:
            with self._status or contextlib.nullcontext():
                return self._run(client)
        finally:
            if self._owns_client:
                client.close()

    def _run(self, client: OpenAIChatClient) -> EndpointBenchmarkResult:
        config = self._config
        try:
            server_models = client.list_models()
        except EndpointError:
            if not config.model:
                raise
            logger.warning("Could not list the server's models; using --endpoint-model.")
            server_models = []
        model = self.model = resolve_model(config.model, server_models)
        if self._collect_host is None:
            from ._run_provenance import collect_host_info

            host = collect_host_info()
        else:
            host = self._collect_host()

        messages = [{"role": "user", "content": config.prompt}]
        total = config.warmup + config.iterations
        samples: list[EndpointRequestSample] = []
        fingerprint: str | None = None
        served: set[str] = set()
        for index in range(total):
            kind = "warmup" if index < config.warmup else "timed"
            if self._on_request is not None:
                self._on_request(index + 1, total, kind)
            timing = client.stream_chat(
                model=model,
                messages=messages,
                max_tokens=config.max_new_tokens,
                extra_body=config.extra_body,
            )
            fingerprint = timing.system_fingerprint or fingerprint
            if timing.response_model:
                served.add(timing.response_model)
            samples.append(EndpointRequestSample.from_timing(kind, index, timing))
        if len(served) > 1:
            logger.warning(
                "The server answered with several models (%s); reporting the requested model.",
                ", ".join(sorted(report_model_id(name) for name in served)),
            )
        served_model = served.pop() if len(served) == 1 else None
        self.model = served_model or model
        first_timed = next((sample for sample in samples if sample.kind == "timed"), None)
        return EndpointBenchmarkResult(
            config=config,
            model=model,
            served_model=served_model,
            server_models=server_models,
            system_fingerprint=fingerprint,
            host=host,
            prompt_tokens=first_timed.prompt_tokens if first_timed else 0,
            generated_tokens=first_timed.generated_tokens if first_timed else 0,
            response_text=first_timed.response_text if first_timed else "",
            stop_reason=first_timed.stop_reason if first_timed else None,
            requests=samples,
            aggregate=aggregate_endpoint_requests(samples),
        )


def run_endpoint_perf(
    config: EndpointPerfConfig,
    *,
    console: Console,
    json_mode: bool,
    resolve_output_path: Callable[[str], Path],
    client: OpenAIChatClient | None = None,
) -> EndpointBenchmarkResult:
    """Run an endpoint benchmark through the shared generation report path.

    Args:
        config: The resolved benchmark request.
        console: Console for progress, the report and the saved-report line.
        json_mode: Print the JSON report instead of the console report.
        resolve_output_path: Returns the report path for the served model id;
            used only when ``config.output_path`` is ``None``.
        client: A pre-built client (dependency injection for tests).
    """
    status = None if json_mode else console.status("Connecting to the server")

    def progress(number: int, total: int, kind: str) -> None:
        if status is not None:
            status.update(f"Request {number} of {total} ({kind})")

    benchmark = EndpointPerfBenchmark(config, client=client, on_request=progress, status=status)

    def output_path() -> Path:
        # Called only after a successful run, which has resolved the served model.
        return config.output_path or resolve_output_path(benchmark.model or "")

    return run_task_perf(
        benchmark, resolve_output_path=output_path, console=console, json_mode=json_mode
    )
