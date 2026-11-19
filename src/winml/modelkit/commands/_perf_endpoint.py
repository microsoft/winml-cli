# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Text generation benchmarking of an OpenAI-compatible server (experimental).

``winml perf --endpoint <base-url>`` times a server that is already running,
such as llama.cpp's ``llama-server``, through the standard streaming
``/v1/chat/completions`` API. The server owns the model, its chat template and
its tokenizer; this module only sends text prompts and measures what arrives.

Per request it records, at the client:

* time to first token (TTFT): from sending the request to the first streamed
  event that carries generated text;
* the arrival of every event that carries generated text, and the request
  duration;
* decode throughput: ``(completion_tokens - 1)`` over the time from the first to
  the last text event. Speculative-decoding servers may stream several tokens
  per event, so throughput is computed from the token count, never from event
  gaps.

It also keeps what the server reports: ``usage`` token counts, cached prompt
tokens, ``finish_reason``, the ``system_fingerprint``, and llama.cpp-style
``timings`` (server-side prefill and decode spans and, with speculative
decoding, drafted and accepted draft tokens).

The report uses ``schema_version`` 4. It adds a ``provenance`` block (host,
GPU drivers, run labels, run index, file hashes, server model list) so reports
from different machines and runs can be combined and compared.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from rich.markup import escape

from .. import __version__
from ._generation_metrics import _round_stats, _stats
from ._openai_chat_client import (
    ChatStreamTiming,
    EndpointError,
    OpenAIChatClient,
    scrub_url,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.console import Console

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 4
RUNTIME = "openai-endpoint"
REPORT_SCHEMA_PATH = Path(__file__).parent / "assets" / "perf-endpoint-report.schema.json"
_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
# A drive letter, a leading separator, '.' or '~', or any backslash marks a file path.
_PATH_LIKE = re.compile(r"^(?:[A-Za-z]:|[\\/~.])|\\")
_MODEL_FILE_SUFFIXES = (".gguf", ".onnx", ".safetensors", ".bin")

# Per-request metrics summarized over the timed requests.
_AGGREGATED_METRICS = (
    "ttft_ms",
    "request_duration_ms",
    "decode_tokens_per_second",
    "tokens_per_event",
    "prompt_tokens",
    "completion_tokens",
    "server_prefill_tokens_per_second",
    "server_decode_tokens_per_second",
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
    """Map a llama.cpp-style ``timings`` object to report fields.

    Unknown or missing keys give ``None``; the raw object is kept separately.
    """
    t = timings or {}
    prompt_n = _number(t.get("prompt_n"))
    prompt_ms = _number(t.get("prompt_ms"))
    predicted_n = _number(t.get("predicted_n"))
    predicted_ms = _number(t.get("predicted_ms"))
    draft_n = _number(t.get("draft_n"))
    accepted = _number(t.get("draft_n_accepted"))
    return {
        "server_prefill_tokens": prompt_n,
        "server_prefill_ms": prompt_ms,
        "server_prefill_tokens_per_second": _per_second(prompt_n, prompt_ms),
        "server_decode_tokens": predicted_n,
        "server_decode_ms": predicted_ms,
        "server_decode_tokens_per_second": _per_second(predicted_n, predicted_ms),
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
    """One timed request, as measured at the client and reported by the server."""

    kind: str
    index: int
    ttft_ms: float | None
    request_duration_ms: float
    token_event_arrivals_ms: list[float]
    prompt_tokens: int | None
    cached_prompt_tokens: int | None
    completion_tokens: int | None
    completion_tokens_source: str
    finish_reason: str | None
    server_timings: dict[str, Any] | None
    response_text: str = ""
    reasoning_text: str = ""

    @classmethod
    def from_timing(cls, kind: str, index: int, timing: ChatStreamTiming) -> EndpointRequestSample:
        """Build a sample from one streamed response."""
        usage = timing.usage or {}
        prompt_tokens = usage.get("prompt_tokens")
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        completion = usage.get("completion_tokens")
        source = "usage"
        predicted = (timing.server_timings or {}).get("predicted_n")
        if not isinstance(completion, int):
            if isinstance(predicted, int):
                completion, source = predicted, "server_timings"
            else:
                completion, source = len(timing.token_event_arrivals_s), "stream_events"
        return cls(
            kind=kind,
            index=index,
            ttft_ms=None if timing.first_token_s is None else timing.first_token_s * 1000.0,
            request_duration_ms=timing.end_s * 1000.0,
            token_event_arrivals_ms=[value * 1000.0 for value in timing.token_event_arrivals_s],
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            cached_prompt_tokens=cached if isinstance(cached, int) else None,
            completion_tokens=completion,
            completion_tokens_source=source,
            finish_reason=timing.finish_reason,
            server_timings=timing.server_timings,
            response_text=timing.response_text,
            reasoning_text=timing.reasoning_text,
        )

    @property
    def token_event_count(self) -> int:
        """Streamed events that carried generated text."""
        return len(self.token_event_arrivals_ms)

    @property
    def decode_duration_ms(self) -> float | None:
        """Time from the first to the last event that carried generated text."""
        if self.token_event_count < 2:
            return None
        return self.token_event_arrivals_ms[-1] - self.token_event_arrivals_ms[0]

    @property
    def decode_tokens(self) -> int | None:
        """Tokens generated after the first one."""
        if self.completion_tokens is None or self.completion_tokens < 2:
            return None
        return self.completion_tokens - 1

    @property
    def decode_tokens_per_second(self) -> float | None:
        """Client-side decode throughput after the first token."""
        tokens = self.decode_tokens
        return _per_second(None if tokens is None else float(tokens), self.decode_duration_ms)

    @property
    def tokens_per_event(self) -> float | None:
        """Generated tokens per streamed text event (above 1 when the server batches tokens)."""
        if self.completion_tokens is None or not self.token_event_count:
            return None
        return self.completion_tokens / self.token_event_count

    @property
    def server(self) -> dict[str, float | None]:
        """Normalized server timings."""
        return normalize_server_timings(self.server_timings)

    def metric(self, name: str) -> float | None:
        """Return a per-request metric by its report name."""
        if name in self.server:
            return self.server[name]
        value = getattr(self, name)
        return None if value is None else float(value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the request for the report."""
        arrivals = self.token_event_arrivals_ms
        result: dict[str, Any] = {
            "kind": self.kind,
            "index": self.index,
            "prompt_tokens": self.prompt_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "completion_tokens_source": self.completion_tokens_source,
            "finish_reason": self.finish_reason,
            "ttft_ms": _round(self.ttft_ms),
            "request_duration_ms": _round(self.request_duration_ms),
            "decode_duration_ms": _round(self.decode_duration_ms),
            "decode_tokens_per_second": _round(self.decode_tokens_per_second, 2),
            "token_event_count": self.token_event_count,
            "tokens_per_event": _round(self.tokens_per_event),
            "token_event_intervals_ms": [
                round(later - earlier, 3) for earlier, later in pairwise(arrivals)
            ],
            **{key: _round(value) for key, value in self.server.items()},
            "server_timings": self.server_timings,
            "response_text": self.response_text,
        }
        if self.reasoning_text:
            result["reasoning_text"] = self.reasoning_text
        return result


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def aggregate_samples(samples: list[EndpointRequestSample]) -> dict[str, Any]:
    """Summarize the timed requests; warmup requests are excluded.

    Each metric gets the percentile summary of the requests that report it, or
    ``None`` when none do. ``totals`` pools tokens and time across requests,
    which weighs long responses more than averaging per-request rates does.
    """
    timed = [sample for sample in samples if sample.kind == "timed"]
    aggregate: dict[str, Any] = {
        "warmup_excluded": True,
        "warmup_request_count": sum(sample.kind == "warmup" for sample in samples),
        "timed_request_count": len(timed),
    }
    for name in _AGGREGATED_METRICS:
        values = [value for sample in timed if (value := sample.metric(name)) is not None]
        aggregate[name] = _round_stats(_stats(values)) if values else None

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
    aggregate["totals"] = {
        "prompt_tokens": total([sample.prompt_tokens for sample in timed]),
        "cached_prompt_tokens": total([sample.cached_prompt_tokens for sample in timed]),
        "completion_tokens": total([sample.completion_tokens for sample in timed]),
        "decode_tokens": decode_tokens,
        "decode_duration_ms": _round(decode_ms),
        "decode_tokens_per_second": _round(_per_second(decode_tokens, decode_ms), 2),
        "draft_tokens": draft,
        "draft_tokens_accepted": accepted,
        "draft_acceptance_rate": (
            _round(accepted / draft, 4) if draft and accepted is not None else None
        ),
    }
    reasons: dict[str, int] = {}
    for sample in timed:
        key = sample.finish_reason or "none"
        reasons[key] = reasons.get(key, 0) + 1
    aggregate["finish_reason_counts"] = dict(sorted(reasons.items()))
    return aggregate


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


@dataclass
class EndpointBenchmarkResult:
    """Canonical results of an endpoint text generation benchmark."""

    config: EndpointPerfConfig
    model: str
    server_models: list[dict[str, Any]]
    system_fingerprint: str | None
    host: dict[str, Any]
    requests: list[EndpointRequestSample]
    aggregate: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    packages: dict[str, str | None] = field(default_factory=_package_versions)

    @property
    def first_timed(self) -> EndpointRequestSample | None:
        """The first timed request, whose response the report shows."""
        return next((sample for sample in self.requests if sample.kind == "timed"), None)

    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON report (``schema_version`` 4)."""
        cfg = self.config
        first = self.first_timed
        return {
            "schema_version": SCHEMA_VERSION,
            "benchmark_info": {
                "runtime": RUNTIME,
                "api": "openai.chat.completions",
                "timing_source": "http-stream",
                "experimental": True,
                "endpoint": scrub_url(cfg.base_url),
                "model_id": report_model_id(self.model),
                "iterations": cfg.iterations,
                "warmup": cfg.warmup,
                "max_new_tokens": cfg.max_new_tokens,
                "prompt": cfg.prompt,
                "prompt_source": cfg.prompt_source,
                "request_parameters": {"temperature": 0, "extra_body": dict(cfg.extra_body)},
                "packages": dict(self.packages),
                "timestamp": self.timestamp,
            },
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
            },
            "requests": [sample.to_dict() for sample in self.requests],
            "aggregate": self.aggregate,
            "response_text": first.response_text if first else "",
        }


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

    Args:
        config: The resolved benchmark request.
        client: A pre-built client (dependency injection for tests). When
            ``None`` one is created from ``config`` and closed after the run.
        collect_host: Returns the host description (injectable for tests).
        on_request: Called with ``(number, total, kind)`` before each request.
    """

    def __init__(
        self,
        config: EndpointPerfConfig,
        *,
        client: OpenAIChatClient | None = None,
        collect_host: Callable[[], dict[str, Any]] | None = None,
        on_request: Callable[[int, int, str], None] | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._owns_client = client is None
        self._collect_host = collect_host
        self._on_request = on_request

    def run(self) -> EndpointBenchmarkResult:
        """Execute the benchmark.

        Raises:
            EndpointError: The server failed a request.
            click.UsageError: The served model could not be resolved.
        """
        client = self._client or OpenAIChatClient(self._config.base_url)
        try:
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
        model = resolve_model(config.model, server_models)
        if self._collect_host is None:
            from ._run_provenance import collect_host_info

            host = collect_host_info()
        else:
            host = self._collect_host()

        messages = [{"role": "user", "content": config.prompt}]
        total = config.warmup + config.iterations
        samples: list[EndpointRequestSample] = []
        fingerprint: str | None = None
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
            samples.append(EndpointRequestSample.from_timing(kind, index, timing))
        return EndpointBenchmarkResult(
            config=config,
            model=model,
            server_models=server_models,
            system_fingerprint=fingerprint,
            host=host,
            requests=samples,
            aggregate=aggregate_samples(samples),
        )


# =============================================================================
# Reporting
# =============================================================================


def _fmt(value: float | None, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


def display_endpoint_report(result: EndpointBenchmarkResult, console: Console) -> None:
    """Render the endpoint benchmark report to the console."""
    from rich.table import Table

    cfg = result.config
    aggregate = result.aggregate
    totals = aggregate["totals"]
    first = result.first_timed
    rows = [
        ("Runtime", f"{RUNTIME} (experimental)"),
        ("API", "OpenAI chat completions (streamed, client-side timing)"),
        ("Endpoint", scrub_url(cfg.base_url)),
        ("Model", report_model_id(result.model)),
    ]
    if result.system_fingerprint:
        rows.append(("Server build", result.system_fingerprint))
    if cfg.run_labels or cfg.run_index is not None:
        labels = ", ".join(f"{key}={value}" for key, value in cfg.run_labels.items())
        index = f"run {cfg.run_index}" if cfg.run_index is not None else ""
        rows.append(("Run", ", ".join(part for part in (index, labels) if part)))
    width = max(len(label) for label, _ in rows) + 3

    console.print()
    for label, value in rows:
        console.print(f"[dim]{label}:[/dim]" + " " * (width - len(label) - 1) + escape(value))
    if first is not None:
        console.print(
            "[dim]Prompt:[/dim]" + " " * (width - 7) + f"{first.prompt_tokens} tokens   "
            f"[dim]Generated:[/dim] {first.completion_tokens} tokens "
            f"(max_new_tokens={cfg.max_new_tokens}, finish={first.finish_reason})"
        )
        if first.response_text:
            console.print()
            console.print("[bold]Response[/bold]")
            console.print(first.response_text, markup=False)

    console.print()
    console.print("[bold]Time to first token (ms)[/bold]")
    table = Table(show_header=True, header_style="bold cyan")
    for column in ("Avg", "P50", "P90", "P95", "P99", "Min", "Max"):
        table.add_column(column, justify="right")
    ttft = aggregate.get("ttft_ms") or {}
    table.add_row(
        *(_fmt(ttft.get(key)) for key in ("mean", "p50", "p90", "p95", "p99", "min", "max"))
    )
    console.print(table)

    decode = aggregate.get("decode_tokens_per_second") or {}
    duration = aggregate.get("request_duration_ms") or {}
    per_event = aggregate.get("tokens_per_event") or {}
    console.print()
    console.print(
        f"[bold]Decode (client):[/bold] {_fmt(decode.get('mean'))} tokens/sec avg  |  "
        f"{_fmt(totals.get('decode_tokens_per_second'))} tokens/sec pooled"
    )
    server_decode = aggregate.get("server_decode_tokens_per_second")
    server_prefill = aggregate.get("server_prefill_tokens_per_second")
    if server_decode or server_prefill:
        console.print(
            "[bold]Server timings:[/bold] "
            f"decode {_fmt((server_decode or {}).get('mean'))} tokens/sec  |  "
            f"prefill {_fmt((server_prefill or {}).get('mean'))} tokens/sec"
        )
    if totals.get("draft_tokens"):
        console.print(
            "[bold]Speculative decoding:[/bold] "
            f"{int(totals['draft_tokens_accepted'] or 0)} of {int(totals['draft_tokens'])} "
            f"draft tokens accepted ({_fmt((totals['draft_acceptance_rate'] or 0) * 100, 1)}%)"
            f"  |  {_fmt(per_event.get('mean'))} tokens per streamed event"
        )
    console.print(f"[bold]Request duration:[/bold] {_fmt(duration.get('mean'))} ms avg")
    if cfg.warmup > 0:
        console.print(f"  [dim]Excluded first {cfg.warmup} warmup request(s) from statistics[/dim]")
    if totals.get("cached_prompt_tokens"):
        console.print(
            "[yellow]Note:[/yellow] the server reused cached prompt tokens on timed requests, so "
            "TTFT excludes part of the prefill. To measure a full prefill, turn off the server's "
            "prompt cache (llama-server: --extra-body '{\"cache_prompt\": false}')."
        )
    if first is not None and not first.completion_tokens:
        console.print(
            "[yellow]Warning:[/yellow] the server generated no tokens, so decode metrics are empty."
        )
    console.print()


def write_endpoint_report(result: EndpointBenchmarkResult, output_path: Path) -> None:
    """Write the report JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result.to_dict(), stream, indent=2)


def run_endpoint_perf(
    config: EndpointPerfConfig,
    *,
    console: Console,
    json_mode: bool,
    resolve_output_path: Callable[[str], Path],
    client: OpenAIChatClient | None = None,
) -> EndpointBenchmarkResult:
    """Run an endpoint benchmark, print the report, and write the JSON report.

    Server failures become ``click`` errors so the CLI exits cleanly; a failed
    run writes no report.

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

    benchmark = EndpointPerfBenchmark(config, client=client, on_request=progress)
    try:
        if status is not None:
            with status:
                result = benchmark.run()
        else:
            result = benchmark.run()
    except EndpointError as exc:
        raise click.ClickException(str(exc)) from exc

    if json_mode:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        display_endpoint_report(result, console)
    output_path = config.output_path or resolve_output_path(result.model)
    write_endpoint_report(result, output_path)
    console.print(f"[green]Results saved to:[/green] {output_path}")
    return result
