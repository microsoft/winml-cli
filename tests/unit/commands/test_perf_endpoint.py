# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ``winml perf --endpoint`` (OpenAI-compatible server benchmarking).

An ``httpx.MockTransport`` plays the server, so no network or model is needed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import jsonschema
import pytest
from click.testing import CliRunner

from winml.modelkit.commands import _perf_endpoint as perf_endpoint
from winml.modelkit.commands import _run_provenance as run_provenance
from winml.modelkit.commands._openai_chat_client import (
    ChatStreamTiming,
    EndpointError,
    OpenAIChatClient,
    iter_sse_data,
    scrub_url,
)
from winml.modelkit.commands._perf_endpoint import (
    REPORT_SCHEMA_PATH,
    EndpointPerfBenchmark,
    EndpointPerfConfig,
    EndpointRequestSample,
    aggregate_samples,
    default_report_slug,
    resolve_model,
)
from winml.modelkit.commands.perf import perf


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

BASE_URL = "http://127.0.0.1:8080/v1"
_HOST = {
    "platform": "test",
    "machine": "AMD64",
    "python": "3.11",
    "os": None,
    "cpus": None,
    "gpus": [{"name": "Test GPU", "driverVersion": "1.2.3"}],
    "ram_mib": 1024,
    "errors": [],
}


# ---------------------------------------------------------------------------
# Fake server
# ---------------------------------------------------------------------------


def _chunk(content: str | None = None, **extra: Any) -> dict[str, Any]:
    choices = [] if content is None else [{"index": 0, "delta": {"content": content}}]
    return {"object": "chat.completion.chunk", "model": "test-model", "choices": choices, **extra}


def _final(
    *, completion: int, prompt: int = 5, cached: int = 0, timings: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    finish = {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
    usage: dict[str, Any] = {
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "prompt_tokens_details": {"cached_tokens": cached},
        }
    }
    if timings is not None:
        usage["timings"] = timings
    return [finish, _chunk(**usage)]


def _sse(events: list[dict[str, Any]]) -> bytes:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode() + (
        b"data: [DONE]\n\n"
    )


class FakeServer:
    """Answers ``/models`` and ``/chat/completions`` and records request bodies."""

    def __init__(
        self,
        events: list[dict[str, Any]],
        *,
        models: list[str] | None = None,
        chat_status: int = 200,
    ) -> None:
        self.events = events
        self.models = ["test-model"] if models is None else models
        self.chat_status = chat_status
        self.bodies: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.headers.append(request.headers)
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": name} for name in self.models]})
        self.bodies.append(json.loads(request.content))
        if self.chat_status != 200:
            return httpx.Response(self.chat_status, text="model not loaded")
        return httpx.Response(
            200, content=_sse(self.events), headers={"Content-Type": "text/event-stream"}
        )

    def client(self, **kwargs: Any) -> OpenAIChatClient:
        return OpenAIChatClient(BASE_URL, transport=httpx.MockTransport(self.handle), **kwargs)


def _ticker(step: float = 0.01) -> Callable[[], float]:
    """A clock that advances ``step`` seconds per call."""
    now = [0.0]

    def tick() -> float:
        now[0] += step
        return now[0]

    return tick


_SPEC_TIMINGS = {
    "prompt_n": 5,
    "prompt_ms": 10.0,
    "predicted_n": 6,
    "predicted_ms": 60.0,
    "draft_n": 8,
    "draft_n_accepted": 6,
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


def test_iter_sse_data_joins_multiline_data_and_stops_at_done() -> None:
    lines = [
        ": keep-alive",
        "data: a",
        "data: b",
        "",
        "event: x",
        "data: c",
        "",
        "data: [DONE]",
        "",
    ]
    assert list(iter_sse_data(iter(lines))) == ["a\nb", "c"]


def test_scrub_url_drops_credentials_query_and_fragment() -> None:
    assert scrub_url("https://user:secret@host:8443/v1?key=abc#frag") == "https://host:8443/v1"


def test_stream_chat_times_text_events_and_reads_usage_and_timings() -> None:
    events = [
        _chunk(choices=[{"index": 0, "delta": {"role": "assistant", "content": None}}]),
        _chunk(choices=[{"index": 0, "delta": {"reasoning_content": "think"}}]),
        _chunk("Hel"),
        _chunk("lo"),
        *_final(completion=6, timings=_SPEC_TIMINGS),
    ]
    events[-1]["system_fingerprint"] = "build-1"
    server = FakeServer(events)
    with server.client() as client:
        timing = client.stream_chat(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=6,
            extra_body={"temperature": 0.5, "top_k": 1},
            clock=_ticker(),
        )

    assert timing.response_text == "Hello"
    assert timing.reasoning_text == "think"
    # The role-only event carries no text, so the reasoning event is the first token.
    assert len(timing.token_event_arrivals_s) == 3
    assert timing.first_token_s == timing.token_event_arrivals_s[0]
    assert timing.finish_reason == "length"
    assert timing.usage is not None
    assert timing.usage["completion_tokens"] == 6
    assert timing.server_timings == _SPEC_TIMINGS
    assert timing.system_fingerprint == "build-1"
    body = server.bodies[0]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["max_tokens"] == 6
    # --extra-body is merged last and can override the defaults.
    assert body["temperature"] == 0.5
    assert body["top_k"] == 1


def test_api_key_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    server = FakeServer(_final(completion=1))
    with server.client() as client:
        client.list_models()
    assert server.headers[0]["Authorization"] == "Bearer sk-test"


def test_stream_chat_raises_on_http_error_status() -> None:
    server = FakeServer([], chat_status=503)
    with server.client() as client, pytest.raises(EndpointError, match="HTTP 503"):
        client.stream_chat(model="m", messages=[], max_tokens=1)


def test_stream_chat_raises_on_streamed_error_event() -> None:
    server = FakeServer([{"error": {"message": "context too long"}}])
    with server.client() as client, pytest.raises(EndpointError, match="context too long"):
        client.stream_chat(model="m", messages=[], max_tokens=1)


# ---------------------------------------------------------------------------
# Samples and aggregation
# ---------------------------------------------------------------------------


def _timing(
    arrivals: list[float],
    *,
    completion: int | None,
    timings: dict[str, Any] | None = None,
    cached: int | None = None,
) -> ChatStreamTiming:
    usage: dict[str, Any] | None = None
    if completion is not None:
        usage = {"prompt_tokens": 5, "completion_tokens": completion}
        if cached is not None:
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return ChatStreamTiming(
        first_token_s=arrivals[0] if arrivals else None,
        token_event_arrivals_s=arrivals,
        end_s=(arrivals[-1] if arrivals else 0.0) + 0.01,
        usage=usage,
        server_timings=timings,
        finish_reason="length",
    )


def test_decode_rate_uses_token_count_not_event_count() -> None:
    # Three events carrying nine tokens: a server that streams several tokens per event.
    sample = EndpointRequestSample.from_timing("timed", 0, _timing([0.1, 0.2, 0.3], completion=9))
    assert sample.ttft_ms == pytest.approx(100.0)
    assert sample.decode_duration_ms == pytest.approx(200.0)
    assert sample.decode_tokens_per_second == pytest.approx(40.0)
    assert sample.tokens_per_event == pytest.approx(3.0)


def test_completion_tokens_fall_back_to_server_timings_then_events() -> None:
    from_timings = EndpointRequestSample.from_timing(
        "timed", 0, _timing([0.1, 0.2], completion=None, timings={"predicted_n": 7})
    )
    from_events = EndpointRequestSample.from_timing(
        "timed", 0, _timing([0.1, 0.2], completion=None)
    )
    assert (from_timings.completion_tokens, from_timings.completion_tokens_source) == (
        7,
        "server_timings",
    )
    assert (from_events.completion_tokens, from_events.completion_tokens_source) == (
        2,
        "stream_events",
    )


def test_aggregate_excludes_warmup_and_pools_draft_acceptance() -> None:
    samples = [
        EndpointRequestSample.from_timing(
            "warmup", 0, _timing([5.0, 6.0], completion=2, timings={"draft_n": 100})
        ),
        EndpointRequestSample.from_timing(
            "timed",
            1,
            _timing([0.1, 0.2], completion=3, timings={"draft_n": 10, "draft_n_accepted": 9}),
        ),
        EndpointRequestSample.from_timing(
            "timed",
            2,
            _timing([0.1, 0.5], completion=5, timings={"draft_n": 30, "draft_n_accepted": 3}),
        ),
    ]
    aggregate = aggregate_samples(samples)
    assert aggregate["warmup_request_count"] == 1
    assert aggregate["timed_request_count"] == 2
    # Pooled: 12 of 40 accepted, not the mean of 0.9 and 0.1.
    assert aggregate["totals"]["draft_acceptance_rate"] == pytest.approx(0.3)
    assert aggregate["draft_acceptance_rate"]["mean"] == pytest.approx(0.5)
    # Pooled decode: (2 + 4) tokens over (100 + 400) ms.
    assert aggregate["totals"]["decode_tokens_per_second"] == pytest.approx(12.0)


def test_aggregate_metrics_are_none_without_server_timings() -> None:
    sample = EndpointRequestSample.from_timing("timed", 0, _timing([0.1, 0.2], completion=2))
    aggregate = aggregate_samples([sample])
    assert aggregate["server_decode_tokens_per_second"] is None
    assert aggregate["draft_acceptance_rate"] is None
    assert aggregate["totals"]["draft_acceptance_rate"] is None


# ---------------------------------------------------------------------------
# Model resolution and report naming
# ---------------------------------------------------------------------------


def test_resolve_model_prefers_request_then_single_listed_model() -> None:
    assert resolve_model("chosen", [{"id": "a"}, {"id": "b"}]) == "chosen"
    assert resolve_model(None, [{"id": "only"}]) == "only"


@pytest.mark.parametrize("models", [[], [{"id": "a"}, {"id": "b"}]])
def test_resolve_model_needs_endpoint_model_when_ambiguous(models: list[dict[str, Any]]) -> None:
    with pytest.raises(Exception, match="--endpoint-model"):
        resolve_model(None, models)


@pytest.mark.parametrize(
    ("model_id", "slug"),
    [
        (r"C:\models\qwen2.5-1.5b-instruct-q4_k_m.gguf", "endpoint-qwen2.5-1.5b-instruct-q4_k_m"),
        ("org/model:latest", "endpoint-org_model_latest"),
    ],
)
def test_default_report_slug_is_file_system_safe(model_id: str, slug: str) -> None:
    assert default_report_slug(model_id) == slug


def test_report_strips_local_folders_from_path_like_model_ids() -> None:
    model_path = r"C:\Users\someone\models\target.gguf"
    server = FakeServer([_chunk("a"), *_final(completion=1)], models=[model_path])
    config = EndpointPerfConfig(base_url=BASE_URL, prompt="hi", iterations=1, warmup=0)
    result = EndpointPerfBenchmark(config, client=server.client(), collect_host=lambda: _HOST).run()
    report = result.to_dict()

    # The full id is still what the server expects in requests.
    assert server.bodies[0]["model"] == model_path
    assert report["benchmark_info"]["model_id"] == "target.gguf"
    assert report["provenance"]["server"]["models"] == [{"id": "target.gguf"}]
    assert "someone" not in json.dumps(report)
    # Hub-style ids are not paths.
    assert perf_endpoint.report_model_id("org/model") == "org/model"


# ---------------------------------------------------------------------------
# Benchmark and report
# ---------------------------------------------------------------------------


def _report_schema() -> dict[str, Any]:
    return json.loads(REPORT_SCHEMA_PATH.read_text(encoding="utf-8"))


def test_benchmark_report_matches_schema() -> None:
    server = FakeServer([_chunk("a"), _chunk("b"), *_final(completion=6, timings=_SPEC_TIMINGS)])
    config = EndpointPerfConfig(
        base_url=BASE_URL,
        prompt="hello",
        max_new_tokens=6,
        iterations=2,
        warmup=1,
        run_labels={"server": "llama.cpp"},
        run_index=3,
    )
    result = EndpointPerfBenchmark(config, client=server.client(), collect_host=lambda: _HOST).run()
    report = result.to_dict()

    jsonschema.validate(report, _report_schema())
    assert len(server.bodies) == 3
    assert server.bodies[0]["messages"] == [{"role": "user", "content": "hello"}]
    assert report["schema_version"] == 4
    assert report["benchmark_info"]["model_id"] == "test-model"
    assert report["provenance"]["run_index"] == 3
    assert report["provenance"]["run_labels"] == {"server": "llama.cpp"}
    assert report["aggregate"]["timed_request_count"] == 2
    assert report["aggregate"]["totals"]["draft_acceptance_rate"] == pytest.approx(0.75)


def test_hash_files_records_name_size_and_sha256_without_folder(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"abc")
    hashed = run_provenance.hash_files({"model": model})
    assert hashed == {
        "model": {
            "name": "model.gguf",
            "size_bytes": 3,
            "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        }
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_server(monkeypatch: pytest.MonkeyPatch) -> FakeServer:
    server = FakeServer([_chunk("a"), _chunk("b"), *_final(completion=2, cached=4)])
    monkeypatch.setattr(perf_endpoint, "OpenAIChatClient", lambda base_url: server.client())
    monkeypatch.setattr(run_provenance, "collect_host_info", lambda: _HOST)
    return server


def test_cli_endpoint_run_writes_report(fake_server: FakeServer, tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    model = tmp_path / "model.gguf"
    model.write_bytes(b"abc")
    result = CliRunner().invoke(
        perf,
        [
            "--endpoint",
            BASE_URL,
            "--prompt",
            "hi",
            "--max-new-tokens",
            "2",
            "--iterations",
            "2",
            "--warmup",
            "0",
            "--extra-body",
            '{"cache_prompt": false}',
            "--run-label",
            "config=plain",
            "--run-index",
            "1",
            "--hash-file",
            f"model={model}",
            "-o",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text(encoding="utf-8"))
    jsonschema.validate(report, _report_schema())
    assert len(fake_server.bodies) == 2
    assert fake_server.bodies[0]["cache_prompt"] is False
    assert report["benchmark_info"]["request_parameters"]["extra_body"] == {"cache_prompt": False}
    assert report["provenance"]["file_hashes"]["model"]["name"] == "model.gguf"
    assert report["aggregate"]["totals"]["cached_prompt_tokens"] == 8
    assert "cached prompt tokens" in result.output


def test_cli_endpoint_defaults_to_ten_iterations_and_two_warmup(
    fake_server: FakeServer, tmp_path: Path
) -> None:
    result = CliRunner().invoke(perf, ["--endpoint", BASE_URL, "-o", str(tmp_path / "r.json")])
    assert result.exit_code == 0, result.output
    assert len(fake_server.bodies) == 12


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["-m", "model.gguf"], "--model cannot be used with --endpoint"),
        (["--device", "gpu"], "--device cannot be used with --endpoint"),
        (["--no-apply-template"], "--apply-template/--no-apply-template cannot be used"),
        (["--extra-body", "[1]"], "--extra-body must be a JSON object"),
        (["--extra-body", "{bad"], "--extra-body is not valid JSON"),
        (["--run-label", "novalue"], "--run-label 'novalue' must be KEY=VALUE"),
        (["--run-label", "a=1", "--run-label", "a=2"], "repeats the key 'a'"),
    ],
)
def test_cli_endpoint_rejects_invalid_options(args: list[str], message: str) -> None:
    result = CliRunner().invoke(perf, ["--endpoint", BASE_URL, *args])
    assert result.exit_code == 2
    assert message in result.output


@pytest.mark.parametrize(
    "url", ["localhost:8080", "http://127.0.0.1:abc/v1", "http://127.0.0.1:99999/v1"]
)
def test_cli_endpoint_rejects_invalid_url(url: str) -> None:
    result = CliRunner().invoke(perf, ["--endpoint", url])
    assert result.exit_code == 2
    assert "URL" in result.output
    assert "Traceback" not in result.output


def test_cli_endpoint_only_options_need_endpoint() -> None:
    result = CliRunner().invoke(perf, ["-m", "model.onnx", "--run-label", "a=1"])
    assert result.exit_code == 2
    assert "--run-label only applies with --endpoint" in result.output


def test_cli_endpoint_server_failure_writes_no_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = FakeServer([], chat_status=500)
    monkeypatch.setattr(perf_endpoint, "OpenAIChatClient", lambda base_url: server.client())
    monkeypatch.setattr(run_provenance, "collect_host_info", lambda: _HOST)
    output = tmp_path / "report.json"
    result = CliRunner().invoke(perf, ["--endpoint", BASE_URL, "-o", str(output)])
    assert result.exit_code == 1
    assert "HTTP 500" in result.output
    assert not output.exists()
