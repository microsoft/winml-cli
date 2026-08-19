# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Security tests for browser access to the local serve APIs."""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from winml.modelkit.serve.app import create_app
from winml.modelkit.serve.cli_api import app as cli_app


if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastapi import FastAPI


@pytest.fixture(params=["cli", "inference"])
def protected_app(request: pytest.FixtureRequest) -> Iterator[tuple[FastAPI, AsyncMock]]:
    invoke = AsyncMock()
    invoke.return_value = {
        "command": "sys",
        "exit_code": 0,
        "result": {},
        "stdout": "",
        "stderr": "",
        "duration_ms": 1.0,
    }
    target = (
        "winml.modelkit.serve.cli_api._run_with_semaphore"
        if request.param == "cli"
        else "winml.modelkit.serve.app._run_with_semaphore"
    )

    with ExitStack() as stack:
        stack.enter_context(patch(target, invoke))
        app = cli_app if request.param == "cli" else create_app(model_path=None, mode="multi")
        yield app, invoke


def test_cross_origin_preflight_is_rejected(
    protected_app: tuple[FastAPI, AsyncMock],
) -> None:
    app, invoke = protected_app

    response = TestClient(app).options(
        "/v1/cli/sys",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 403
    assert "access-control-allow-origin" not in response.headers
    invoke.assert_not_awaited()


def test_cross_origin_post_is_rejected_before_cli_dispatch(
    protected_app: tuple[FastAPI, AsyncMock],
) -> None:
    app, invoke = protected_app

    response = TestClient(app).post(
        "/v1/cli/sys",
        headers={"Origin": "https://evil.example"},
        json={"args": {}},
    )

    assert response.status_code == 403
    assert "access-control-allow-origin" not in response.headers
    invoke.assert_not_awaited()


def test_same_origin_post_reaches_cli_dispatch(
    protected_app: tuple[FastAPI, AsyncMock],
) -> None:
    app, invoke = protected_app

    response = TestClient(app).post(
        "/v1/cli/sys",
        headers={"Origin": "http://testserver"},
        json={"args": {}},
    )

    assert response.status_code == 200
    invoke.assert_awaited_once_with("sys", {})
