# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Security middleware for local WinML HTTP servers."""

from __future__ import annotations

import ipaddress
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from starlette.responses import PlainTextResponse


if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send


class SameOriginMiddleware:
    """Reject browser requests whose origin differs from the target server."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and not _is_same_origin(scope):
            response = PlainTextResponse("Cross-origin requests are not allowed.", status_code=403)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _is_same_origin(scope: Scope) -> bool:
    headers = {key.lower(): value for key, value in scope.get("headers", [])}
    origin_header = headers.get(b"origin")
    if origin_header is None:
        return True

    try:
        origin = urlsplit(origin_header.decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        return False
    if origin.scheme not in {"http", "https"} or origin.path or origin.query or origin.fragment:
        return False
    if origin.username is not None or origin.password is not None:
        return False
    if not _is_allowed_http_host(origin.hostname):
        return False
    try:
        origin_port = _effective_port(origin.port, origin.scheme)
    except ValueError:
        return False

    target_scheme = scope.get("scheme", "http").lower()
    if target_scheme not in {"http", "https"}:
        return False
    host_header = headers.get(b"host")
    if host_header is None:
        server = scope.get("server")
        if server is None:
            return False
        target_host, target_port = server
        target_host = _normalize_host(target_host)
        target_port = _effective_port(target_port, target_scheme)
    else:
        try:
            target = urlsplit(f"{target_scheme}://{host_header.decode('ascii')}")
        except (UnicodeDecodeError, ValueError):
            return False
        if (
            target.path
            or target.query
            or target.fragment
            or target.username is not None
            or target.password is not None
        ):
            return False
        target_host = _normalize_host(target.hostname)
        try:
            target_port = _effective_port(target.port, target_scheme)
        except ValueError:
            return False

    return (
        origin.scheme == target_scheme
        and _normalize_host(origin.hostname) == target_host
        and origin_port == target_port
    )


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def _effective_port(port: int | None, scheme: str) -> int:
    return _default_port(scheme) if port is None else port


def _normalize_host(hostname: str | None) -> str | None:
    return hostname.rstrip(".").lower() if hostname is not None else None


def _is_allowed_http_host(hostname: str | None) -> bool:
    normalized = _normalize_host(hostname)
    if normalized is None:
        return False
    if normalized == "localhost":
        return True
    try:
        ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return True
