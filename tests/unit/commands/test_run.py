# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for winml run command.

Tests the CLI interface, output formatting, model matching, param parsing,
and print helpers using Click's CliRunner.
All inference is mocked — no real models are loaded.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.run import (
    _format_text,
    _models_match,
    _parse_param_value,
    _print_result,
    _try_server_predict,
    run,
)


if TYPE_CHECKING:
    from pathlib import Path

_ENGINE_PATH = "winml.modelkit.serve.engine.InferenceEngine"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _make_mock_engine(result_dict: dict) -> MagicMock:
    """Build a mock InferenceEngine whose predict() returns *result_dict*."""
    mock_result = MagicMock()
    mock_result.model_dump.return_value = result_dict
    engine = MagicMock()
    engine.predict.return_value = mock_result
    return engine


_MINIMAL_RESULT: dict = {
    "task": "test",
    "device": "cpu",
    "ep": "",
    "latency_ms": 0,
    "predictions": [],
}


# =============================================================================
# _parse_param_value
# =============================================================================


class TestParseParamValue:
    def test_int(self) -> None:
        assert _parse_param_value("42") == 42
        assert isinstance(_parse_param_value("42"), int)

    def test_negative_int(self) -> None:
        assert _parse_param_value("-3") == -3

    def test_zero(self) -> None:
        assert _parse_param_value("0") == 0
        assert isinstance(_parse_param_value("0"), int)

    def test_float(self) -> None:
        assert _parse_param_value("3.14") == pytest.approx(3.14)
        assert isinstance(_parse_param_value("3.14"), float)

    def test_float_with_dot_zero(self) -> None:
        """'1.0' is float, not int (int('1.0') raises ValueError)."""
        assert _parse_param_value("1.0") == 1.0
        assert isinstance(_parse_param_value("1.0"), float)

    def test_bool_true(self) -> None:
        assert _parse_param_value("true") is True
        assert _parse_param_value("True") is True
        assert _parse_param_value("TRUE") is True

    def test_bool_false(self) -> None:
        assert _parse_param_value("false") is False
        assert _parse_param_value("False") is False

    def test_string(self) -> None:
        assert _parse_param_value("hello") == "hello"

    def test_empty_string(self) -> None:
        assert _parse_param_value("") == ""


# =============================================================================
# _models_match
# =============================================================================


class TestModelsMatch:
    def test_exact_match(self) -> None:
        assert _models_match("microsoft/resnet-50", "microsoft/resnet-50") is True

    def test_basename_match(self) -> None:
        assert _models_match("/cache/resnet-50", "hub/resnet-50") is True

    def test_no_match(self) -> None:
        assert _models_match("microsoft/resnet-50", "google/vit-base") is False

    def test_empty_server_model(self) -> None:
        assert _models_match("", "microsoft/resnet-50") is False

    def test_hf_org_name_differs_but_basename_same(self) -> None:
        assert _models_match("org-a/bert-base", "org-b/bert-base") is True

    def test_local_path_vs_hf_id(self) -> None:
        assert _models_match("/home/user/.cache/bert-base", "microsoft/bert-base") is True

    def test_both_empty(self) -> None:
        assert _models_match("", "") is False

    def test_single_segment_exact(self) -> None:
        assert _models_match("resnet-50", "resnet-50") is True

    def test_single_segment_no_match(self) -> None:
        assert _models_match("resnet-50", "vit-base") is False


# =============================================================================
# _format_text
# =============================================================================


class TestFormatText:
    def test_classification_output(self) -> None:
        result = {
            "task": "text-classification",
            "model_id": "ProsusAI/finbert",
            "device": "cpu",
            "ep": "",
            "latency_ms": 123.4,
            "predictions": [
                {"label": "neutral", "score": 0.89},
                {"label": "positive", "score": 0.06},
            ],
        }
        text = _format_text(result)
        assert "Task:    text-classification" in text
        assert "Model:   ProsusAI/finbert" in text
        assert "neutral" in text
        assert "positive" in text
        assert "Latency: 123.4ms" in text

    def test_with_ep(self) -> None:
        result = {
            "task": "image-classification",
            "model_id": "resnet",
            "device": "npu",
            "ep": "QNNExecutionProvider",
            "latency_ms": 50.0,
            "predictions": [],
        }
        text = _format_text(result)
        assert "Device:  npu (QNNExecutionProvider)" in text

    def test_no_ep(self) -> None:
        result = {
            "task": "test",
            "device": "cpu",
            "ep": "",
            "latency_ms": 10.0,
            "predictions": [],
        }
        text = _format_text(result)
        assert "Device:  cpu" in text
        assert "(" not in text.split("Device:")[1].split("\n")[0]

    def test_non_list_predictions(self) -> None:
        result = {**_MINIMAL_RESULT, "predictions": "raw output string"}
        text = _format_text(result)
        assert "Output: raw output string" in text

    def test_no_model_id(self) -> None:
        text = _format_text(_MINIMAL_RESULT)
        assert "Model:" not in text

    def test_score_four_decimal_places(self) -> None:
        result = {
            **_MINIMAL_RESULT,
            "predictions": [{"label": "cat", "score": 0.123456789}],
        }
        text = _format_text(result)
        assert "0.1235" in text

    def test_prediction_numbering(self) -> None:
        result = {
            **_MINIMAL_RESULT,
            "predictions": [
                {"label": "a", "score": 0.5},
                {"label": "b", "score": 0.3},
                {"label": "c", "score": 0.2},
            ],
        }
        text = _format_text(result)
        lines = text.split("\n")
        result_lines = [line for line in lines if line.strip().startswith(("1.", "2.", "3."))]
        assert len(result_lines) == 3

    def test_model_path_fallback(self) -> None:
        result = {**_MINIMAL_RESULT, "model_path": "/tmp/my-model.onnx"}  # noqa: S108
        text = _format_text(result)
        assert "Model:   /tmp/my-model.onnx" in text

    def test_missing_keys_use_defaults(self) -> None:
        text = _format_text({})
        assert "Task:" in text
        assert "Latency: 0.0ms" in text

    def test_prediction_without_label_uses_index(self) -> None:
        result = {**_MINIMAL_RESULT, "predictions": [{"score": 0.9}]}
        text = _format_text(result)
        assert "1" in text
        assert "0.9000" in text


# =============================================================================
# _print_result
# =============================================================================


class TestPrintResult:
    def test_json_to_file(self, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        data = {"task": "test", "predictions": []}
        _print_result(data, output_format="json", output_path=str(out))
        assert json.loads(out.read_text()) == data

    def test_text_to_file(self, tmp_path: Path) -> None:
        out = tmp_path / "out.txt"
        _print_result(_MINIMAL_RESULT, output_format="text", output_path=str(out))
        assert "Task:    test" in out.read_text()

    def test_text_to_stdout(self) -> None:
        r = CliRunner()
        with r.isolated_filesystem():
            _print_result(_MINIMAL_RESULT, output_format="text", output_path=None)

    def test_click_echo_oserror_fallback(self) -> None:
        """When click.echo raises OSError (Windows error 6), falls back to sys.stdout."""
        mock_stdout = MagicMock()
        with (
            patch("winml.modelkit.commands.run.click") as mock_click,
            patch("winml.modelkit.commands.run.sys") as mock_sys,
        ):
            mock_click.echo.side_effect = OSError("Windows error 6")
            mock_sys.stdout = mock_stdout
            _print_result(_MINIMAL_RESULT, output_format="text", output_path=None)

        mock_stdout.write.assert_called_once()
        mock_stdout.flush.assert_called_once()


# =============================================================================
# CLI command
# =============================================================================


class TestRunCLI:
    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(run, ["--help"])
        assert result.exit_code == 0
        assert "Run one-shot inference" in result.output

    def test_help_shows_param_option(self, runner: CliRunner) -> None:
        result = runner.invoke(run, ["--help"])
        assert "-P" in result.output
        assert "KEY=VALUE" in result.output

    def test_missing_model_path(self, runner: CliRunner) -> None:
        result = runner.invoke(run, ["--text", "test"])
        assert result.exit_code != 0

    def test_missing_input(self, runner: CliRunner) -> None:
        result = runner.invoke(run, ["--model", "some-model"])
        assert result.exit_code != 0

    def test_embedded_text_inference(self, runner: CliRunner) -> None:
        """--text → engine.predict(text=...) path."""
        engine = _make_mock_engine(
            {
                **_MINIMAL_RESULT,
                "task": "text-classification",
                "predictions": [{"label": "positive", "score": 0.95}],
            }
        )

        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "test-model", "--text", "hello world"],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "positive" in result.output
        engine.load.assert_called_once_with(
            "test-model",
            task=None,
            device="auto",
            ep=None,
        )
        engine.predict.assert_called_once_with(files=None, text="hello world")

    def test_embedded_file_inference(self, runner: CliRunner, tmp_path: Path) -> None:
        """--file → engine.predict(files=[...]) path."""
        img = tmp_path / "cat.jpg"
        img.write_bytes(b"\xff\xd8fake-jpeg")

        engine = _make_mock_engine(
            {
                **_MINIMAL_RESULT,
                "predictions": [{"label": "cat", "score": 0.99}],
            }
        )

        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "resnet", "--file", str(img)],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "cat" in result.output
        engine.predict.assert_called_once()
        call_kwargs = engine.predict.call_args.kwargs
        assert call_kwargs["files"] == [b"\xff\xd8fake-jpeg"]

    def test_multiple_files(self, runner: CliRunner, tmp_path: Path) -> None:
        """Multiple --file flags pass a list of bytes to engine.predict."""
        f1 = tmp_path / "a.jpg"
        f2 = tmp_path / "b.jpg"
        f1.write_bytes(b"img-a")
        f2.write_bytes(b"img-b")

        engine = _make_mock_engine(_MINIMAL_RESULT)

        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "llava", "--file", str(f1), "--file", str(f2), "--text", "Compare"],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        call_kwargs = engine.predict.call_args.kwargs
        assert call_kwargs["files"] == [b"img-a", b"img-b"]
        assert call_kwargs["text"] == "Compare"

    def test_file_not_found_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(
            run,
            ["--model", "model", "--file", "/nonexistent/path.jpg"],
        )
        assert result.exit_code == 2
        assert "file not found" in result.output

    def test_json_output_format(self, runner: CliRunner) -> None:
        engine = _make_mock_engine(
            {
                "task": "test",
                "predictions": [{"label": "a", "score": 0.5}],
            }
        )

        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "model", "--text", "hello", "--format", "json"],
            )

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["task"] == "test"

    def test_output_to_file(self, runner: CliRunner, tmp_path: Path) -> None:
        out = tmp_path / "result.json"
        engine = _make_mock_engine({"task": "test", "predictions": []})

        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                [
                    "--model",
                    "model",
                    "--text",
                    "hello",
                    "--format",
                    "json",
                    "-o",
                    str(out),
                ],
            )

        assert result.exit_code == 0
        assert json.loads(out.read_text())["task"] == "test"

    def test_load_error_exits_3(self, runner: CliRunner) -> None:
        engine = MagicMock()
        engine.load.side_effect = RuntimeError("bad model")

        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "bad-model", "--text", "hello"],
            )

        assert result.exit_code == 3
        assert "Error loading model" in result.output

    def test_predict_error_exits_4(self, runner: CliRunner) -> None:
        engine = MagicMock()
        engine.predict.side_effect = RuntimeError("inference failed")

        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "model", "--text", "hello"],
            )

        assert result.exit_code == 4
        assert "Error during inference" in result.output

    def test_task_forwarded_to_load(self, runner: CliRunner) -> None:
        engine = _make_mock_engine(_MINIMAL_RESULT)

        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                [
                    "--model",
                    "model.onnx",
                    "--text",
                    "hello",
                    "--task",
                    "image-classification",
                ],
            )

        assert result.exit_code == 0
        engine.load.assert_called_once_with(
            "model.onnx",
            task="image-classification",
            device="auto",
            ep=None,
        )

    def test_device_and_ep_forwarded(self, runner: CliRunner) -> None:
        engine = _make_mock_engine({**_MINIMAL_RESULT, "device": "gpu", "ep": "dml"})

        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                [
                    "--model",
                    "model",
                    "--text",
                    "hello",
                    "--device",
                    "gpu",
                    "--ep",
                    "dml",
                ],
            )

        assert result.exit_code == 0
        engine.load.assert_called_once_with(
            "model",
            task=None,
            device="gpu",
            ep="dml",
        )

    # ---- -P / --param tests ----

    def test_single_param(self, runner: CliRunner) -> None:
        """-P max_new_tokens=100 is forwarded to engine.predict."""
        engine = _make_mock_engine(_MINIMAL_RESULT)

        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "model", "--text", "hello", "-P", "max_new_tokens=100"],
            )

        assert result.exit_code == 0
        kwargs = engine.predict.call_args.kwargs
        assert kwargs["max_new_tokens"] == 100

    def test_multiple_params(self, runner: CliRunner) -> None:
        engine = _make_mock_engine(_MINIMAL_RESULT)

        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                [
                    "--model",
                    "model",
                    "--text",
                    "hello",
                    "-P",
                    "max_new_tokens=50",
                    "-P",
                    "temperature=0.7",
                    "-P",
                    "do_sample=true",
                ],
            )

        assert result.exit_code == 0
        kwargs = engine.predict.call_args.kwargs
        assert kwargs["max_new_tokens"] == 50
        assert kwargs["temperature"] == pytest.approx(0.7)
        assert kwargs["do_sample"] is True

    def test_param_top_k(self, runner: CliRunner) -> None:
        """-P top_k=10 passes top_k to engine.predict."""
        engine = _make_mock_engine(_MINIMAL_RESULT)

        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "model", "--text", "hello", "-P", "top_k=10"],
            )

        assert result.exit_code == 0
        assert engine.predict.call_args.kwargs["top_k"] == 10

    def test_invalid_param_format_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(
            run,
            ["--model", "model", "--text", "hello", "-P", "badformat"],
        )
        assert result.exit_code == 2
        assert "invalid --param format" in result.output


# =============================================================================
# Auto-connect
# =============================================================================


class TestAutoConnect:
    @staticmethod
    def _make_httpx_client_mock(
        health_json: dict,
        predict_json: dict | None = None,
        *,
        health_status: int = 200,
    ) -> MagicMock:
        """Build a mock that behaves like ``httpx.Client`` used as context manager."""
        client = MagicMock()
        health_resp = MagicMock(status_code=health_status)
        health_resp.json.return_value = health_json
        client.get.return_value = health_resp

        if predict_json is not None:
            predict_resp = MagicMock()
            predict_resp.json.return_value = predict_json
            client.post.return_value = predict_resp

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=client)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    def test_auto_connect_delegates_when_server_matches(self, runner: CliRunner) -> None:
        server_response = {
            "task": "text-classification",
            "predictions": [{"label": "pos", "score": 0.9}],
            "latency_ms": 5.0,
            "device": "cpu",
            "ep": "",
        }
        ctx = self._make_httpx_client_mock(
            health_json={"model_id": "my-model"},
            predict_json=server_response,
        )

        with patch("httpx.Client", return_value=ctx):
            result = runner.invoke(run, ["--model", "my-model", "--text", "hello", "--connect"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "pos" in result.output

    def test_auto_connect_falls_back_when_no_server(self, runner: CliRunner) -> None:
        engine = _make_mock_engine(_MINIMAL_RESULT)

        with (
            patch("httpx.Client", side_effect=ConnectionError("refused")),
            patch(_ENGINE_PATH, return_value=engine),
        ):
            result = runner.invoke(run, ["--model", "model", "--text", "hello", "--connect"])

        assert result.exit_code == 0
        engine.load.assert_called_once()

    def test_auto_connect_skipped_when_model_mismatch(self, runner: CliRunner) -> None:
        engine = _make_mock_engine(_MINIMAL_RESULT)
        ctx = self._make_httpx_client_mock(
            health_json={"model_id": "different-model"},
        )

        with (
            patch("httpx.Client", return_value=ctx),
            patch(_ENGINE_PATH, return_value=engine),
        ):
            result = runner.invoke(run, ["--model", "my-model", "--text", "hello", "--connect"])

        assert result.exit_code == 0
        engine.load.assert_called_once()


# =============================================================================
# _try_server_predict (unit-level, no CLI)
# =============================================================================

_DEFAULT_KWARGS: dict = {"top_k": 5}


class TestTryServerPredict:
    """Direct unit tests for _try_server_predict — isolated from CLI."""

    @staticmethod
    def _make_client_mock(
        health_json: dict,
        predict_json: dict | None = None,
        *,
        health_status: int = 200,
    ) -> MagicMock:
        client = MagicMock()
        health_resp = MagicMock(status_code=health_status)
        health_resp.json.return_value = health_json
        client.get.return_value = health_resp

        if predict_json is not None:
            predict_resp = MagicMock()
            predict_resp.json.return_value = predict_json
            client.post.return_value = predict_resp

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=client)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    def test_returns_none_when_httpx_not_installed(self) -> None:
        with patch.dict("sys.modules", {"httpx": None}):
            result = _try_server_predict(
                port=8000,
                model_path="m",
                files=(),
                text="t",
                pipeline_kwargs=_DEFAULT_KWARGS,
            )
        assert result is None

    def test_returns_none_on_non_200_health(self) -> None:
        ctx = self._make_client_mock(health_json={}, health_status=503)
        with patch("httpx.Client", return_value=ctx):
            result = _try_server_predict(
                port=8000,
                model_path="m",
                files=(),
                text="t",
                pipeline_kwargs=_DEFAULT_KWARGS,
            )
        assert result is None

    def test_returns_none_on_model_mismatch(self) -> None:
        ctx = self._make_client_mock(health_json={"model_id": "other-model"})
        with patch("httpx.Client", return_value=ctx):
            result = _try_server_predict(
                port=8000,
                model_path="my-model",
                files=(),
                text="t",
                pipeline_kwargs=_DEFAULT_KWARGS,
            )
        assert result is None

    def test_model_path_key_fallback(self) -> None:
        """Health endpoint returns model_path instead of model_id."""
        expected = {"predictions": [{"label": "ok", "score": 1.0}]}
        ctx = self._make_client_mock(
            health_json={"model_path": "my-model"},
            predict_json=expected,
        )
        with patch("httpx.Client", return_value=ctx):
            result = _try_server_predict(
                port=8000,
                model_path="my-model",
                files=(),
                text="hello",
                pipeline_kwargs=_DEFAULT_KWARGS,
            )
        assert result == expected

    def test_text_input_posts_to_predict(self) -> None:
        expected = {"predictions": []}
        ctx = self._make_client_mock(
            health_json={"model_id": "m"},
            predict_json=expected,
        )
        with patch("httpx.Client", return_value=ctx):
            result = _try_server_predict(
                port=9000,
                model_path="m",
                files=(),
                text="some text",
                pipeline_kwargs={"top_k": 3},
            )

        assert result == expected
        client = ctx.__enter__.return_value
        client.post.assert_called_once()
        call_kwargs = client.post.call_args
        assert "/v1/predict" in call_kwargs.args[0]
        body = call_kwargs.kwargs["json"]
        assert body["text"] == "some text"
        assert body["params"]["top_k"] == 3

    def test_text_input_forwards_extra_kwargs(self) -> None:
        """Extra pipeline_kwargs are included in the JSON body."""
        expected = {"predictions": []}
        ctx = self._make_client_mock(
            health_json={"model_id": "m"},
            predict_json=expected,
        )
        with patch("httpx.Client", return_value=ctx):
            _try_server_predict(
                port=8000,
                model_path="m",
                files=(),
                text="text",
                pipeline_kwargs={"top_k": 5, "max_new_tokens": 100, "temperature": 0.7},
            )

        client = ctx.__enter__.return_value
        body = client.post.call_args.kwargs["json"]
        assert body["params"]["max_new_tokens"] == 100
        assert body["params"]["temperature"] == 0.7

    def test_single_file_posts_to_predict_file(self, tmp_path: Path) -> None:
        img = tmp_path / "photo.png"
        img.write_bytes(b"fake-png")

        expected = {"predictions": [{"label": "cat", "score": 0.9}]}
        ctx = self._make_client_mock(
            health_json={"model_id": "m"},
            predict_json=expected,
        )
        with patch("httpx.Client", return_value=ctx):
            result = _try_server_predict(
                port=8000,
                model_path="m",
                files=(str(img),),
                text=None,
                pipeline_kwargs=_DEFAULT_KWARGS,
            )

        assert result == expected
        client = ctx.__enter__.return_value
        call_args = client.post.call_args
        assert "/v1/predict/file" in call_args.args[0]

    def test_single_file_forwards_kwargs_as_params_json(self, tmp_path: Path) -> None:
        img = tmp_path / "img.jpg"
        img.write_bytes(b"fake")

        ctx = self._make_client_mock(
            health_json={"model_id": "m"},
            predict_json={"predictions": []},
        )
        with patch("httpx.Client", return_value=ctx):
            _try_server_predict(
                port=8000,
                model_path="m",
                files=(str(img),),
                text=None,
                pipeline_kwargs={"top_k": 3, "threshold": 0.5},
            )

        client = ctx.__enter__.return_value
        form_data = client.post.call_args.kwargs["data"]
        params = json.loads(form_data["params"])
        assert params["top_k"] == 3
        assert params["threshold"] == 0.5

    def test_multiple_files_posts_to_predict_json(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.jpg"
        f2 = tmp_path / "b.jpg"
        f1.write_bytes(b"img-a")
        f2.write_bytes(b"img-b")

        expected = {"predictions": []}
        ctx = self._make_client_mock(
            health_json={"model_id": "m"},
            predict_json=expected,
        )
        with patch("httpx.Client", return_value=ctx):
            result = _try_server_predict(
                port=8000,
                model_path="m",
                files=(str(f1), str(f2)),
                text="Compare",
                pipeline_kwargs=_DEFAULT_KWARGS,
            )

        assert result == expected
        client = ctx.__enter__.return_value
        call_args = client.post.call_args
        assert "/v1/predict" in call_args.args[0]
        body = call_args.kwargs["json"]
        assert len(body["files"]) == 2
        assert body["text"] == "Compare"

    def test_custom_port(self) -> None:
        expected = {"predictions": []}
        ctx = self._make_client_mock(
            health_json={"model_id": "m"},
            predict_json=expected,
        )
        with patch("httpx.Client", return_value=ctx):
            _try_server_predict(
                port=1234,
                model_path="m",
                files=(),
                text="t",
                pipeline_kwargs=_DEFAULT_KWARGS,
            )
        client = ctx.__enter__.return_value
        url = client.get.call_args.args[0]
        assert ":1234" in url

    def test_returns_none_on_connection_error(self) -> None:
        with patch("httpx.Client", side_effect=ConnectionError("refused")):
            result = _try_server_predict(
                port=8000,
                model_path="m",
                files=(),
                text="t",
                pipeline_kwargs=_DEFAULT_KWARGS,
            )
        assert result is None
