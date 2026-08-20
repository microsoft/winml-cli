# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Security middleware for local WinML HTTP servers."""

from __future__ import annotations

import ipaddress
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from starlette._utils import get_route_path
from starlette.responses import PlainTextResponse


if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from starlette.types import ASGIApp, Receive, Scope, Send


class SameOriginMiddleware:
    """Enforce browser-origin and local-only route boundaries."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        loopback_only_routes: Mapping[str, Collection[str] | None] | None = None,
    ) -> None:
        self.app = app
        self._loopback_only_routes = {
            prefix.rstrip("/"): (
                None if methods is None else frozenset(method.upper() for method in methods)
            )
            for prefix, methods in (loopback_only_routes or {}).items()
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            if self._requires_loopback(scope) and not _is_loopback_client(scope):
                response = PlainTextResponse(
                    "This route is available only to local clients.",
                    status_code=403,
                )
                await response(scope, receive, send)
                return
            if not _is_same_origin(scope):
                response = PlainTextResponse(
                    "Cross-origin requests are not allowed.",
                    status_code=403,
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)

    def _requires_loopback(self, scope: Scope) -> bool:
        path = get_route_path(scope)
        method = scope.get("method", "").upper()
        for prefix, methods in self._loopback_only_routes.items():
            if path != prefix and not path.startswith(f"{prefix}/"):
                continue
            if methods is None or method in methods:
                return True
        return False


def _is_loopback_client(scope: Mapping[str, Any]) -> bool:
    client = scope.get("client")
    if not isinstance(client, (tuple, list)) or not client:
        return False
    host = client[0]
    if not isinstance(host, str):
        return False
    try:
        address = ipaddress.ip_address(host.split("%", maxsplit=1)[0])
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback


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
