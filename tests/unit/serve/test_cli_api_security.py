# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Security boundaries for the local CLI HTTP API."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from typing import Any

import click
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import winml.modelkit.serve.app as serve_app_module
import winml.modelkit.serve.cli_api as cli_api
from winml.modelkit.inference import InferenceEngine, PredictionResult
from winml.modelkit.loader import load_hf_config, load_hf_model
from winml.modelkit.serve._security import _is_same_origin
from winml.modelkit.serve.manager import ModelSlot


def _successful_cli_response(command: str) -> cli_api.CliResponse:
    return cli_api.CliResponse(
        command=command,
        exit_code=0,
        result=None,
        stdout="",
        stderr="",
        duration_ms=0,
    )


class _DemoPolicyParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.content_security_policy: str | None = None
        self.server_url_attributes: dict[str, str | None] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if tag == "meta" and attributes.get("http-equiv", "").lower() == "content-security-policy":
            self.content_security_policy = attributes.get("content")
        if tag == "input" and attributes.get("id") == "serverUrl":
            self.server_url_attributes = attributes


def _evaluate_demo_server_url(html: str, document_url: str) -> str:
    match = re.search(r"^const serverUrl = (?P<expression>.+);\r?$", html, re.MULTILINE)
    assert match is not None

    node = shutil.which("node")
    assert node is not None
    script = (
        f"const window = {{ location: new URL({json.dumps(document_url)}) }};\n"
        f"const serverUrl = {match.group('expression')};\n"
        "process.stdout.write(serverUrl);"
    )
    result = subprocess.run(  # noqa: S603 -- fixed Node executable and repository JS
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


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

        with TestClient(cli_api.app, client=("127.0.0.1", 50000)) as client:
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
        with TestClient(cli_api.app, client=("127.0.0.1", 50000)) as client:
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
        with TestClient(
            cli_api.app,
            base_url="http://127.0.0.1:8000",
            client=("127.0.0.1", 50000),
        ) as client:
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


class TestLoopbackProtection:
    """Only clients on this machine can invoke privileged HTTP routes."""

    def test_remote_client_cannot_invoke_cli_without_origin(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        invocations: list[str] = []

        def fake_invoke(command: str, _args: dict[str, Any]) -> cli_api.CliResponse:
            invocations.append(command)
            return _successful_cli_response(command)

        monkeypatch.setattr(cli_api, "_invoke", fake_invoke)

        with TestClient(cli_api.app, client=("192.0.2.20", 50000)) as client:
            response = client.post(
                "/v1/cli/compile",
                json={
                    "args": {
                        "model": "model.onnx",
                        "compiler": "qairt",
                        "qnn_sdk_root": r"\\attacker\share\sdk",
                    }
                },
            )

        assert response.status_code == 403
        assert invocations == []

    def test_remote_client_cannot_bypass_cli_guard_through_mount(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        invocations: list[str] = []

        def fake_invoke(command: str, _args: dict[str, Any]) -> cli_api.CliResponse:
            invocations.append(command)
            return _successful_cli_response(command)

        monkeypatch.setattr(cli_api, "_invoke", fake_invoke)
        parent = FastAPI()
        parent.mount("/proxy", cli_api.app)

        with TestClient(parent, client=("192.0.2.20", 50000)) as client:
            response = client.post("/proxy/v1/cli/sys", json={"args": {}})

        assert response.status_code == 403
        assert invocations == []

    def test_remote_client_cannot_use_management_route_without_origin(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        invocations: list[str] = []

        async def fake_run(
            command: str,
            _args: dict[str, Any],
        ) -> cli_api.CliResponse:
            invocations.append(command)
            return _successful_cli_response(command)

        monkeypatch.setattr(serve_app_module, "_run_with_semaphore", fake_run)
        app = serve_app_module.create_app(model_path="unused")
        client = TestClient(app, client=("192.0.2.20", 50000))
        try:
            cli_response = client.post("/v1/cli/sys", json={"args": {}})
            ep_response = client.post("/v1/ep", json={"ep": "cpu"})
            load_response = client.post("/v1/models", json={"model_id": "owner/model"})
            unload_response = client.delete("/v1/models/owner/model")
            read_only_response = client.get("/v1/models")
            inference_response = client.post(
                "/v1/predict",
                json={"inputs": {"text": "hello"}, "params": {}},
            )
        finally:
            client.close()

        assert cli_response.status_code == 403
        assert ep_response.status_code == 403
        assert load_response.status_code == 403
        assert unload_response.status_code == 403
        assert read_only_response.status_code == 503
        assert inference_response.status_code == 503
        assert invocations == []

    @pytest.mark.parametrize("client_host", ["127.0.0.1", "::1"])
    def test_loopback_client_can_invoke_cli_without_origin(
        self,
        client_host: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            cli_api,
            "_invoke",
            lambda command, _args: _successful_cli_response(command),
        )

        with TestClient(cli_api.app, client=(client_host, 50000)) as client:
            response = client.post("/v1/cli/sys", json={"args": {}})

        assert response.status_code == 200

    @pytest.mark.parametrize(
        ("path", "request_kwargs"),
        [
            (
                "/v1/predict?model_id=owner/unloaded-model",
                {"json": {"inputs": {"text": "hello"}, "params": {}}},
            ),
            (
                "/v1/predict/file",
                {
                    "files": {"file": ("input.bin", b"data")},
                    "data": {"model_id": "owner/unloaded-model"},
                },
            ),
        ],
    )
    def test_remote_inference_cannot_lazy_load_model(
        self,
        path: str,
        request_kwargs: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        load_calls: list[str] = []

        def record_load(
            _engine: InferenceEngine,
            model_path: str,
            **_kwargs: Any,
        ) -> None:
            load_calls.append(str(model_path))
            raise AssertionError("remote request reached model loading")

        monkeypatch.setattr(InferenceEngine, "load", record_load)
        app = serve_app_module.create_app(model_path=None, mode="multi")

        with TestClient(
            app,
            client=("192.0.2.20", 50000),
            raise_server_exceptions=False,
        ) as client:
            response = client.post(path, **request_kwargs)

        assert response.status_code == 400
        assert "not loaded" in response.text
        assert load_calls == []

    def test_loopback_inference_retains_lazy_load(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        load_calls: list[str] = []

        def fake_load(
            engine: InferenceEngine,
            model_path: str,
            *,
            task: str | None = None,
            device: str = "auto",
            **_kwargs: Any,
        ) -> None:
            load_calls.append(str(model_path))
            engine._model = object()
            engine._model_id = str(model_path)
            engine._model_path = str(model_path)
            engine._task = task or "unit-test"
            engine._device = device
            engine._user_input_schema = None

        def fake_predict(
            engine: InferenceEngine,
            *,
            inputs: dict[str, Any],
            task: str | None = None,
            **_kwargs: Any,
        ) -> PredictionResult:
            assert inputs == {"text": "hello"}
            return PredictionResult(
                task=task or engine.task or "unit-test",
                model_id=engine.model_id,
                device=engine.device,
                predictions={},
                latency_ms=0,
            )

        monkeypatch.setattr(InferenceEngine, "load", fake_load)
        monkeypatch.setattr(InferenceEngine, "predict", fake_predict)
        app = serve_app_module.create_app(model_path=None, mode="multi")

        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            response = client.post(
                "/v1/predict?model_id=owner/local-model",
                json={"inputs": {"text": "hello"}, "params": {}},
            )

        assert response.status_code == 200
        assert load_calls == ["owner/local-model"]

    def test_remote_inference_can_use_loaded_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        load_calls: list[str] = []

        def record_load(
            _engine: InferenceEngine,
            model_path: str,
            **_kwargs: Any,
        ) -> None:
            load_calls.append(str(model_path))
            raise AssertionError("loaded slot unexpectedly reloaded")

        def fake_predict(
            engine: InferenceEngine,
            *,
            inputs: dict[str, Any],
            task: str | None = None,
            **_kwargs: Any,
        ) -> PredictionResult:
            assert inputs == {"text": "hello"}
            return PredictionResult(
                task=task or engine.task or "unit-test",
                model_id=engine.model_id,
                device=engine.device,
                predictions={},
                latency_ms=0,
            )

        monkeypatch.setattr(InferenceEngine, "load", record_load)
        monkeypatch.setattr(InferenceEngine, "predict", fake_predict)
        app = serve_app_module.create_app(model_path=None, mode="multi")

        with TestClient(app, client=("192.0.2.20", 50000)) as client:
            engine = InferenceEngine()
            engine._model = object()
            engine._model_id = "owner/loaded-model"
            engine._model_path = "owner/loaded-model"
            engine._task = "unit-test"
            engine._device = "cpu"
            engine._user_input_schema = None
            client.app.state.manager._slots["owner/loaded-model"] = ModelSlot(
                model_id="owner/loaded-model",
                engine=engine,
            )

            response = client.post(
                "/v1/predict?model_id=owner/loaded-model",
                json={"inputs": {"text": "hello"}, "params": {}},
            )

        assert response.status_code == 200
        assert load_calls == []


class TestDemoSecurity:
    """The bundled UI cannot select a server outside its own origin."""

    def test_demo_locks_server_selection_to_document_origin(self) -> None:
        with TestClient(cli_api.app, client=("127.0.0.1", 50000)) as client:
            response = client.get("/demo")

        parser = _DemoPolicyParser()
        parser.feed(response.text)

        assert response.status_code == 200
        assert parser.content_security_policy == "connect-src 'self'"
        assert parser.server_url_attributes is not None
        assert "readonly" in parser.server_url_attributes
        assert _evaluate_demo_server_url(response.text, str(response.url)) == "http://testserver"

    def test_demo_preserves_mount_path_in_server_url(self) -> None:
        parent = FastAPI()
        parent.mount("/proxy", cli_api.app)

        with TestClient(parent, client=("127.0.0.1", 50000)) as client:
            response = client.get("/proxy/demo")

        assert response.status_code == 200
        assert _evaluate_demo_server_url(response.text, str(response.url)) == (
            "http://testserver/proxy"
        )


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

        with TestClient(cli_api.app, client=("127.0.0.1", 50000)) as client:
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
