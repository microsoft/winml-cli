# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Security boundaries for the local CLI HTTP API."""

from __future__ import annotations

from typing import Any

import click
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import winml.modelkit.serve.app as serve_app_module
import winml.modelkit.serve.cli_api as cli_api
from winml.modelkit.loader import load_hf_config, load_hf_model
from winml.modelkit.serve._security import _is_same_origin


def _successful_cli_response(command: str) -> cli_api.CliResponse:
    return cli_api.CliResponse(
        command=command,
        exit_code=0,
        result=None,
        stdout="",
        stderr="",
        duration_ms=0,
    )


class TestOriginProtection:
    """Browser origins cannot cross the local HTTP trust boundary."""

    def test_phase_zero_rejects_foreign_origin_before_dispatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        invocations: list[str] = []

        def fake_invoke(command: str, _args: dict[str, Any]) -> cli_api.CliResponse:
            invocations.append(command)
            return _successful_cli_response(command)

        monkeypatch.setattr(cli_api, "_invoke", fake_invoke)

        with TestClient(cli_api.app) as client:
            response = client.post(
                "/v1/cli/sys",
                headers={"Origin": "https://example.invalid"},
                json={"args": {}},
            )

        assert response.status_code == 403
        assert "access-control-allow-origin" not in response.headers
        assert invocations == []

    @pytest.mark.parametrize(
        ("origin", "host", "scheme", "expected"),
        [
            ("http://127.0.0.1:8000", "127.0.0.1:8000", "http", True),
            ("https://localhost", "localhost", "https", True),
            ("http://[::1]:8000", "[::1]:8000", "http", True),
            ("http://LOCALHOST", "localhost:80", "http", True),
            ("http://localhost:0", "localhost:80", "http", False),
            ("http://127.0.0.1:8001", "127.0.0.1:8000", "http", False),
            ("http://192.0.2.10:8000", "192.0.2.10:8000", "http", True),
            ("http://example.invalid", "127.0.0.1:8000", "http", False),
            ("http://user@localhost", "localhost", "http", False),
            ("http://localhost", "[::1", "http", False),
            ("null", "127.0.0.1:8000", "http", False),
        ],
    )
    def test_origin_parser(
        self,
        origin: str,
        host: str,
        scheme: str,
        expected: bool,
    ) -> None:
        scope = {
            "type": "http",
            "scheme": scheme,
            "headers": [
                (b"host", host.encode("ascii")),
                (b"origin", origin.encode("ascii")),
            ],
        }

        assert _is_same_origin(scope) is expected

    def test_phase_zero_rejects_foreign_origin_preflight(self) -> None:
        with TestClient(cli_api.app) as client:
            response = client.options(
                "/v1/cli/build",
                headers={
                    "Origin": "https://example.invalid",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )

        assert response.status_code == 403
        assert "access-control-allow-origin" not in response.headers

    def test_phase_zero_allows_same_origin_ip_host(self) -> None:
        with TestClient(cli_api.app, base_url="http://192.0.2.10:8000") as client:
            response = client.get(
                "/v1/health",
                headers={"Origin": "http://192.0.2.10:8000"},
            )

        assert response.status_code == 200

    def test_phase_zero_rejects_dns_rebinding_origin(self) -> None:
        with TestClient(cli_api.app) as client:
            response = client.get(
                "/v1/health",
                headers={
                    "Host": "example.invalid",
                    "Origin": "http://example.invalid",
                },
            )

        assert response.status_code == 403

    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"Origin": "http://127.0.0.1:8000"},
        ],
    )
    def test_phase_zero_allows_local_clients(self, headers: dict[str, str]) -> None:
        with TestClient(cli_api.app, base_url="http://127.0.0.1:8000") as client:
            response = client.get("/v1/health", headers=headers)

        assert response.status_code == 200

    def test_model_server_rejects_foreign_origin(self) -> None:
        app = serve_app_module.create_app(model_path="unused")
        client = TestClient(app)
        try:
            response = client.get(
                "/openapi.json",
                headers={"Origin": "https://example.invalid"},
            )
        finally:
            client.close()

        assert response.status_code == 403
        assert "access-control-allow-origin" not in response.headers


class TestCliHttpPolicy:
    """HTTP dispatch exposes only the explicitly safe CLI capability set."""

    @pytest.mark.parametrize("key", ["trust_remote_code", "trust-remote-code"])
    def test_remote_code_opt_in_is_rejected_before_dispatch(
        self,
        key: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        invocations: list[tuple[str, dict[str, Any]]] = []

        def fake_invoke(command: str, args: dict[str, Any]) -> cli_api.CliResponse:
            invocations.append((command, args))
            return _successful_cli_response(command)

        monkeypatch.setattr(cli_api, "_invoke", fake_invoke)

        with TestClient(cli_api.app) as client:
            response = client.post(
                "/v1/cli/build",
                json={"args": {key: True}},
            )

        assert response.status_code == 400
        assert invocations == []

    def test_model_server_rejects_commands_outside_allowlist(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        invocations: list[str] = []

        def fake_invoke(command: str, _args: dict[str, Any]) -> cli_api.CliResponse:
            invocations.append(command)
            return _successful_cli_response(command)

        monkeypatch.setattr(cli_api, "_invoke", fake_invoke)
        app = FastAPI()
        serve_app_module._register_routes(app, mode="single")

        with TestClient(app) as client:
            response = client.post(
                "/v1/cli/eval",
                json={"args": {}},
            )

        assert response.status_code == 404
        assert invocations == []

    def test_http_invocation_blocks_nested_remote_code_and_resets_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        forwarded_values: list[bool] = []

        class RecordingAutoConfig:
            @staticmethod
            def from_pretrained(_model_id: str, **kwargs: Any) -> object:
                forwarded_values.append(kwargs["trust_remote_code"])
                return object()

        def raw_config_loader(
            _model_id: str,
            **_kwargs: Any,
        ) -> tuple[dict[str, str], dict[str, Any]]:
            return {"model_type": "unit-test"}, {}

        @click.group()
        def test_cli() -> None:
            pass

        @test_cli.command("build")
        def build_command() -> None:
            load_hf_config(
                RecordingAutoConfig,
                "owner/model",
                trust_remote_code=True,
                raw_config_loader=raw_config_loader,
            )

        monkeypatch.setattr(cli_api, "winml_cli", test_cli)

        response = cli_api._invoke("build", {})

        assert response.exit_code != 0
        assert forwarded_values == []

        load_hf_config(
            RecordingAutoConfig,
            "owner/model",
            trust_remote_code=True,
            raw_config_loader=raw_config_loader,
        )
        assert forwarded_values == [True]

    def test_preloaded_config_cannot_bypass_http_remote_code_policy(self) -> None:
        with (
            cli_api._disable_remote_code_execution(),
            pytest.raises(PermissionError, match="disabled for HTTP"),
        ):
            load_hf_model(
                "owner/model",
                trust_remote_code=True,
                hf_config=object(),
            )
