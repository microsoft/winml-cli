# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Streaming client for OpenAI-compatible chat completions servers.

``winml perf --endpoint`` times a running server through the standard
``/v1/chat/completions`` streaming API. This module sends one request, records
when each server-sent event arrives, and collects what the server reports about
the request: ``usage``, ``finish_reason`` and, when the server adds one, a
llama.cpp-style ``timings`` object with prefill, decode and speculative-decoding
counters.

It imports only ``httpx`` and the standard library.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    import httpx


API_KEY_ENV = "OPENAI_API_KEY"
_ERROR_BODY_LIMIT = 500
# Reasoning servers stream generated reasoning in a separate delta field.
_REASONING_FIELDS = ("reasoning_content", "reasoning")


class EndpointError(RuntimeError):
    """The server could not be reached, refused a request, or streamed an error."""


def scrub_url(url: str) -> str:
    """Return ``url`` without credentials, query string or fragment, for reports."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:  # out of range or not a number
        port = None
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def normalize_base_url(url: str) -> str:
    """Validate an OpenAI base URL such as ``http://127.0.0.1:8080/v1``.

    Raises:
        ValueError: The URL is not an absolute http(s) URL with a valid port.
    """
    import httpx

    url = url.strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(
            f"'{url}' is not an http(s) URL. Pass the server's OpenAI base URL, "
            "for example http://127.0.0.1:8080/v1."
        )
    try:
        _ = parts.port
        httpx.URL(url)
    except (ValueError, httpx.InvalidURL) as exc:
        shown = urlunsplit((parts.scheme, parts.netloc.rpartition("@")[2], parts.path, "", ""))
        raise ValueError(f"'{shown}' is not a valid URL: {exc}") from exc
    return url.rstrip("/")


@dataclass
class ChatStreamTiming:
    """One streamed chat completion, timed at the client.

    Times are seconds since the request was sent.

    Attributes:
        first_token_s: Arrival of the first event carrying generated text, or
            ``None`` when the server generated nothing.
        token_event_arrivals_s: Arrival of every event carrying generated text.
            A speculative-decoding server may send several tokens per event.
        end_s: When the stream ended.
        response_text: Concatenated ``content`` deltas.
        reasoning_text: Concatenated reasoning deltas, when the server streams them.
        finish_reason: The ``finish_reason`` of the last choice that reported one.
        usage: The ``usage`` object, when the server reported one.
        server_timings: The ``timings`` object, when the server reported one.
        response_model: The ``model`` field of the streamed events.
        system_fingerprint: The ``system_fingerprint`` field, which some
            servers set to their build.
    """

    first_token_s: float | None
    token_event_arrivals_s: list[float]
    end_s: float
    response_text: str = ""
    reasoning_text: str = ""
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    server_timings: dict[str, Any] | None = None
    response_model: str | None = None
    system_fingerprint: str | None = None


@dataclass
class _StreamState:
    first_token_s: float | None = None
    arrivals: list[float] = field(default_factory=list)
    content: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    server_timings: dict[str, Any] | None = None
    response_model: str | None = None
    system_fingerprint: str | None = None

    def add_event(self, event: Mapping[str, Any], arrival_s: float) -> None:
        if "error" in event:
            raise EndpointError(f"The server streamed an error: {_short(event['error'])}")
        if isinstance(event.get("model"), str):
            self.response_model = event["model"]
        if isinstance(event.get("system_fingerprint"), str):
            self.system_fingerprint = event["system_fingerprint"]
        if isinstance(event.get("usage"), dict):
            self.usage = dict(event["usage"])
        if isinstance(event.get("timings"), dict):
            self.server_timings = dict(event["timings"])
        generated = False
        for choice in event.get("choices") or ():
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if isinstance(text, str) and text:
                self.content.append(text)
                generated = True
            for name in _REASONING_FIELDS:
                value = delta.get(name)
                if isinstance(value, str) and value:
                    self.reasoning.append(value)
                    generated = True
            if choice.get("finish_reason") is not None:
                self.finish_reason = str(choice["finish_reason"])
        if generated:
            if self.first_token_s is None:
                self.first_token_s = arrival_s
            self.arrivals.append(arrival_s)


def _short(value: object) -> str:
    text = value if isinstance(value, str) else json.dumps(value)
    return text if len(text) <= _ERROR_BODY_LIMIT else text[:_ERROR_BODY_LIMIT] + "..."


def iter_sse_data(lines: Iterator[str]) -> Iterator[str]:
    """Yield the ``data`` payload of each server-sent event, stopping at ``[DONE]``.

    Multi-line ``data`` fields are joined with newlines, per the SSE format;
    comments and other fields are skipped.
    """
    data: list[str] = []
    for line in lines:
        if line == "":
            if data:
                payload = "\n".join(data)
                data = []
                if payload.strip() == "[DONE]":
                    return
                yield payload
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        if name == "data":
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        payload = "\n".join(data)
        if payload.strip() != "[DONE]":
            yield payload


class OpenAIChatClient:
    """Sends streaming chat completions to one OpenAI-compatible server.

    Args:
        base_url: The OpenAI base URL, for example ``http://127.0.0.1:8080/v1``.
            Requests go to ``<base_url>/chat/completions`` and ``<base_url>/models``.
        api_key: Bearer token; ``None`` reads ``OPENAI_API_KEY`` and sends no
            ``Authorization`` header when it is unset.
        timeout_s: Seconds to wait to connect and between streamed events.
        transport: An ``httpx`` transport (dependency injection for tests).
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        import httpx

        self.base_url = normalize_base_url(base_url)
        key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        headers = {"Accept": "text/event-stream"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=min(timeout_s, 30.0)),
            transport=transport,
        )

    def close(self) -> None:
        """Close the connection pool."""
        self._client.close()

    def __enter__(self) -> OpenAIChatClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def list_models(self) -> list[dict[str, Any]]:
        """Return the ``data`` entries of ``GET <base_url>/models``.

        Raises:
            EndpointError: The server is unreachable or did not return a model list.
        """
        import httpx

        url = f"{self.base_url}/models"
        try:
            response = self._client.get(url, headers={"Accept": "application/json"})
        except httpx.HTTPError as exc:
            raise EndpointError(f"Could not reach {scrub_url(url)}: {exc}") from exc
        if response.status_code != 200:
            raise EndpointError(
                f"GET {scrub_url(url)} returned HTTP {response.status_code}: "
                f"{_short(response.text)}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise EndpointError(f"GET {scrub_url(url)} did not return JSON.") from exc
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise EndpointError(f"GET {scrub_url(url)} returned no 'data' model list.")
        return [entry for entry in data if isinstance(entry, dict)]

    def stream_chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int,
        extra_body: Mapping[str, Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> ChatStreamTiming:
        """Send one greedy streaming chat completion and time its events.

        The request asks for ``temperature`` 0 and a final ``usage`` event;
        ``extra_body`` keys are merged last, so they can override either.

        Raises:
            EndpointError: The request failed, the server returned a non-200
                status, or the stream carried an error event.
        """
        import httpx

        body: dict[str, Any] = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            **(extra_body or {}),
        }
        url = f"{self.base_url}/chat/completions"
        state = _StreamState()
        start = clock()
        try:
            with self._client.stream("POST", url, json=body) as response:
                if response.status_code != 200:
                    response.read()
                    raise EndpointError(
                        f"POST {scrub_url(url)} returned HTTP {response.status_code}: "
                        f"{_short(response.text)}"
                    )
                for payload in iter_sse_data(response.iter_lines()):
                    arrival = clock() - start
                    try:
                        event = json.loads(payload)
                    except ValueError as exc:
                        raise EndpointError(
                            f"The server streamed an event that is not JSON: {_short(payload)}"
                        ) from exc
                    if isinstance(event, dict):
                        state.add_event(event, arrival)
        except httpx.HTTPError as exc:
            raise EndpointError(f"Request to {scrub_url(url)} failed: {exc}") from exc
        end = clock() - start
        return ChatStreamTiming(
            first_token_s=state.first_token_s,
            token_event_arrivals_s=state.arrivals,
            end_s=end,
            response_text="".join(state.content),
            reasoning_text="".join(state.reasoning),
            finish_reason=state.finish_reason,
            usage=state.usage,
            server_timings=state.server_timings,
            response_model=state.response_model,
            system_fingerprint=state.system_fingerprint,
        )
