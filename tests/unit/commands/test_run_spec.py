# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Spec-based test suite for ``winml run`` command.

Test cases are derived systematically from ``winml run --help``.
Each class maps to a specific CLI option or interaction.
All inference is mocked — no real models are loaded.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.run import (
    _build_example_command,
    _coerce_value,
    _format_text,
    _models_match,
    _parse_heuristic,
    _print_result,
    _resolve_shortcuts,
    run,
)
from winml.modelkit.inference import InputField


if TYPE_CHECKING:
    from pathlib import Path

_ENGINE_PATH = "winml.modelkit.inference.InferenceEngine"

_MINIMAL_RESULT: dict = {
    "task": "test",
    "device": "cpu",
    "ep": "",
    "latency_ms": 0,
    "predictions": [],
}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _make_mock_engine(
    result_dict: dict,
    *,
    schema: list[InputField] | None = None,
    task: str | None = None,
    params: list[dict] | None = None,
) -> MagicMock:
    """Build a mock InferenceEngine whose predict() returns *result_dict*."""
    mock_result = MagicMock()
    mock_result.model_dump.return_value = result_dict
    engine = MagicMock()
    engine.predict.return_value = mock_result
    engine.user_input_schema = schema
    engine.task = task or result_dict.get("task")
    engine.model_id = result_dict.get("model_id")
    engine.model_path = result_dict.get("model_path", "test-model")
    engine.pipeline_params = params
    return engine


# =============================================================================
# TC-01: --model option (required)
# =============================================================================


class TestModelOption:
    """--model / -m is required and accepts HF ID, dir, or .onnx path."""

    def test_missing_model_exits_nonzero(self, runner: CliRunner) -> None:
        """Omitting --model must fail (it's required)."""
        result = runner.invoke(run, ["--text", "hello"])
        assert result.exit_code != 0

    def test_short_flag_m(self, runner: CliRunner) -> None:
        """-m is the short form of --model."""
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["-m", "my-model", "--text", "hi"])
        assert result.exit_code == 0
        engine.load.assert_called_once()
        assert engine.load.call_args.args[0] == "my-model"

    def test_model_value_forwarded_to_engine(self, runner: CliRunner) -> None:
        """The raw model string is passed to engine.load()."""
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            runner.invoke(run, ["--model", "microsoft/resnet-50", "--text", "x"])
        engine.load.assert_called_once_with(
            "microsoft/resnet-50", task=None, device="auto", ep=None
        )


# =============================================================================
# TC-02: --file / -f option
# =============================================================================


class TestFileOption:
    def test_file_reads_bytes(self, runner: CliRunner, tmp_path: Path) -> None:
        """--file reads file content as bytes and passes to predict."""
        img = tmp_path / "cat.jpg"
        img.write_bytes(b"\xff\xd8fake-jpeg")
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--file", str(img)])
        assert result.exit_code == 0
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["file"] == b"\xff\xd8fake-jpeg"

    def test_short_flag_f(self, runner: CliRunner, tmp_path: Path) -> None:
        """-f is the short form of --file."""
        img = tmp_path / "a.png"
        img.write_bytes(b"png-data")
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "-f", str(img)])
        assert result.exit_code == 0

    def test_file_not_found_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(run, ["--model", "m", "--file", "/no/such/file.jpg"])
        assert result.exit_code == 2
        assert "file not found" in result.output.lower()

    def test_file_is_directory_exits_2(self, runner: CliRunner, tmp_path: Path) -> None:
        """--file pointing to a directory should fail."""
        d = tmp_path / "subdir"
        d.mkdir()
        result = runner.invoke(run, ["--model", "m", "--file", str(d)])
        assert result.exit_code == 2
        assert "file not found" in result.output.lower()

    def test_multiple_files_exits_2(self, runner: CliRunner, tmp_path: Path) -> None:
        """--file is documented for a single file; multiple → error."""
        a = tmp_path / "a.jpg"
        b = tmp_path / "b.jpg"
        a.write_bytes(b"a")
        b.write_bytes(b"b")
        result = runner.invoke(run, ["--model", "m", "--file", str(a), "--file", str(b)])
        assert result.exit_code == 2
        assert "only one file" in result.output.lower()


# =============================================================================
# TC-03: --text / -t option
# =============================================================================


class TestTextOption:
    def test_text_passed_to_predict(self, runner: CliRunner) -> None:
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--text", "hello world"])
        assert result.exit_code == 0
        assert engine.predict.call_args.kwargs["inputs"]["text"] == "hello world"

    def test_short_flag_t(self, runner: CliRunner) -> None:
        """-t is the short form of --text."""
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "-t", "hi"])
        assert result.exit_code == 0
        assert engine.predict.call_args.kwargs["inputs"]["text"] == "hi"

    def test_empty_text_string(self, runner: CliRunner) -> None:
        """--text '' should pass empty string (has_inputs is True)."""
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--text", ""])
        assert result.exit_code == 0
        assert engine.predict.call_args.kwargs["inputs"]["text"] == ""

    def test_text_ambiguous_with_multiple_text_fields(self, runner: CliRunner) -> None:
        """--text with schema having >1 text fields → ambiguous error."""
        schema = [
            InputField(name="question", type="text", required=True),
            InputField(name="context", type="text", required=True),
        ]
        engine = _make_mock_engine(_MINIMAL_RESULT, schema=schema)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--text", "Who?"])
        assert result.exit_code == 2
        assert "ambiguous" in result.output.lower()

    def test_text_not_supported_for_image_only_task(self, runner: CliRunner) -> None:
        """--text on a task with no text input → error."""
        schema = [
            InputField(name="image", type="image", required=True),
        ]
        engine = _make_mock_engine(_MINIMAL_RESULT, schema=schema)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--text", "hello"])
        assert result.exit_code == 2
        assert "not supported" in result.output.lower()


# =============================================================================
# TC-04: --input / -I option
# =============================================================================


class TestInputOption:
    def test_basic_name_value(self, runner: CliRunner) -> None:
        """-I question='Who?' passes named input."""
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "-I", "question=Who?"])
        assert result.exit_code == 0
        assert engine.predict.call_args.kwargs["inputs"]["question"] == "Who?"

    def test_value_containing_equals(self, runner: CliRunner) -> None:
        """Value with '=' should be preserved: -I text='a=b'."""
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "-I", "text=a=b"])
        assert result.exit_code == 0
        assert engine.predict.call_args.kwargs["inputs"]["text"] == "a=b"

    def test_missing_equals_exits_2(self, runner: CliRunner) -> None:
        """-I without = is invalid."""
        result = runner.invoke(run, ["--model", "m", "-I", "badformat"])
        assert result.exit_code == 2
        assert "invalid --input format" in result.output.lower()

    def test_repeatable_multiple_inputs(self, runner: CliRunner) -> None:
        """Multiple -I options are all forwarded."""
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "m", "-I", "question=Who?", "-I", "context=Tim Cook is CEO"],
            )
        assert result.exit_code == 0
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["question"] == "Who?"
        assert inputs["context"] == "Tim Cook is CEO"

    def test_at_file_syntax_binary_type(self, runner: CliRunner, tmp_path: Path) -> None:
        """-I image=@photo.jpg reads file bytes when schema type is binary."""
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"jpeg-data")
        schema = [InputField(name="image", type="image", required=True)]
        engine = _make_mock_engine(_MINIMAL_RESULT, schema=schema)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "-I", f"image=@{img}"])
        assert result.exit_code == 0
        assert engine.predict.call_args.kwargs["inputs"]["image"] == b"jpeg-data"

    def test_at_file_not_found_errors(self, runner: CliRunner) -> None:
        """-I image=@nonexistent.jpg with binary schema type → error."""
        schema = [InputField(name="image", type="image", required=True)]
        engine = _make_mock_engine(_MINIMAL_RESULT, schema=schema)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "-I", "image=@/no/such/file.jpg"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_at_prefix_text_type_is_literal(self, runner: CliRunner) -> None:
        """-I text=@file.txt with text type → literal string '@file.txt'."""
        schema = [InputField(name="text", type="text", required=True)]
        engine = _make_mock_engine(_MINIMAL_RESULT, schema=schema)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "-I", "text=@file.txt"])
        assert result.exit_code == 0
        assert engine.predict.call_args.kwargs["inputs"]["text"] == "@file.txt"

    def test_json_value_parsed(self, runner: CliRunner) -> None:
        """-I labels='["a","b"]' with json type → parsed list."""
        schema = [
            InputField(name="text", type="text", required=True),
            InputField(name="labels", type="json", required=True),
        ]
        engine = _make_mock_engine(_MINIMAL_RESULT, schema=schema)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "m", "-I", "text=hello", "-I", 'labels=["a","b"]'],
            )
        assert result.exit_code == 0
        assert engine.predict.call_args.kwargs["inputs"]["labels"] == ["a", "b"]

    def test_invalid_json_errors(self, runner: CliRunner) -> None:
        """-I data='{bad' with json type → error."""
        schema = [InputField(name="data", type="json", required=True)]
        engine = _make_mock_engine(_MINIMAL_RESULT, schema=schema)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "-I", "data={bad"])
        assert result.exit_code != 0
        assert "invalid json" in result.output.lower()

    def test_number_coercion(self, runner: CliRunner) -> None:
        """-I threshold=0.5 with number type → float."""
        schema = [
            InputField(name="text", type="text", required=True),
            InputField(name="threshold", type="number", required=False),
        ]
        engine = _make_mock_engine(_MINIMAL_RESULT, schema=schema)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "m", "-I", "text=hi", "-I", "threshold=0.5"],
            )
        assert result.exit_code == 0
        assert engine.predict.call_args.kwargs["inputs"]["threshold"] == pytest.approx(0.5)

    def test_invalid_number_errors(self, runner: CliRunner) -> None:
        """-I val=abc with number type → error."""
        schema = [InputField(name="val", type="number", required=True)]
        engine = _make_mock_engine(_MINIMAL_RESULT, schema=schema)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "-I", "val=abc"])
        assert result.exit_code != 0
        assert "expected number" in result.output.lower()

    def test_boolean_coercion(self, runner: CliRunner) -> None:
        """-I flag=true with boolean type → True."""
        schema = [
            InputField(name="text", type="text", required=True),
            InputField(name="flag", type="boolean", required=False),
        ]
        engine = _make_mock_engine(_MINIMAL_RESULT, schema=schema)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "m", "-I", "text=hi", "-I", "flag=true"],
            )
        assert result.exit_code == 0
        assert engine.predict.call_args.kwargs["inputs"]["flag"] is True

    def test_invalid_boolean_errors(self, runner: CliRunner) -> None:
        """-I flag=maybe with boolean type → error."""
        schema = [InputField(name="flag", type="boolean", required=True)]
        engine = _make_mock_engine(_MINIMAL_RESULT, schema=schema)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "-I", "flag=maybe"])
        assert result.exit_code != 0
        assert "expected true/false" in result.output.lower()

    def test_empty_value(self, runner: CliRunner) -> None:
        """-I text= passes empty string as value."""
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "-I", "text="])
        assert result.exit_code == 0
        assert engine.predict.call_args.kwargs["inputs"]["text"] == ""


# =============================================================================
# TC-05: -P / --param option
# =============================================================================


class TestParamOption:
    def test_int_value(self, runner: CliRunner) -> None:
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run, ["--model", "m", "--text", "hi", "-P", "max_new_tokens=100"]
            )
        assert result.exit_code == 0
        assert engine.predict.call_args.kwargs["max_new_tokens"] == 100

    def test_float_value(self, runner: CliRunner) -> None:
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--text", "hi", "-P", "temperature=0.7"])
        assert result.exit_code == 0
        assert engine.predict.call_args.kwargs["temperature"] == pytest.approx(0.7)

    def test_bool_value(self, runner: CliRunner) -> None:
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--text", "hi", "-P", "do_sample=true"])
        assert result.exit_code == 0
        assert engine.predict.call_args.kwargs["do_sample"] is True

    def test_string_value(self, runner: CliRunner) -> None:
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--text", "hi", "-P", "mode=greedy"])
        assert result.exit_code == 0
        assert engine.predict.call_args.kwargs["mode"] == "greedy"

    def test_value_with_equals(self, runner: CliRunner) -> None:
        """-P key=a=b → key='a=b' (splits on first = only)."""
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--text", "hi", "-P", "key=a=b"])
        assert result.exit_code == 0
        assert engine.predict.call_args.kwargs["key"] == "a=b"

    def test_missing_equals_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(run, ["--model", "m", "--text", "hi", "-P", "badformat"])
        assert result.exit_code == 2
        assert "invalid --param format" in result.output.lower()

    def test_multiple_params(self, runner: CliRunner) -> None:
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                [
                    "--model",
                    "m",
                    "--text",
                    "hi",
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

    def test_long_form_param(self, runner: CliRunner) -> None:
        """--param is the long form of -P."""
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--text", "hi", "--param", "top_k=5"])
        assert result.exit_code == 0
        assert engine.predict.call_args.kwargs["top_k"] == 5


# =============================================================================
# TC-06: --task option
# =============================================================================


class TestTaskOption:
    def test_task_forwarded_to_engine(self, runner: CliRunner) -> None:
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            runner.invoke(
                run,
                ["--model", "m", "--text", "hi", "--task", "text-classification"],
            )
        assert engine.load.call_args.kwargs["task"] == "text-classification"

    def test_task_default_none(self, runner: CliRunner) -> None:
        """Without --task, None is passed (auto-detect)."""
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            runner.invoke(run, ["--model", "m", "--text", "hi"])
        assert engine.load.call_args.kwargs["task"] is None


# =============================================================================
# TC-07: --device option
# =============================================================================


class TestDeviceOption:
    @pytest.mark.parametrize("device", ["auto", "cpu", "gpu", "npu"])
    def test_valid_device_values(self, runner: CliRunner, device: str) -> None:
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--text", "hi", "--device", device])
        assert result.exit_code == 0
        assert engine.load.call_args.kwargs["device"] == device

    def test_default_is_auto(self, runner: CliRunner) -> None:
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            runner.invoke(run, ["--model", "m", "--text", "hi"])
        assert engine.load.call_args.kwargs["device"] == "auto"

    def test_invalid_device_exits_nonzero(self, runner: CliRunner) -> None:
        result = runner.invoke(run, ["--model", "m", "--text", "hi", "--device", "tpu"])
        assert result.exit_code != 0

    def test_device_case_insensitive(self, runner: CliRunner) -> None:
        """Click Choice with case_sensitive=False allows 'CPU'."""
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--text", "hi", "--device", "CPU"])
        assert result.exit_code == 0


# =============================================================================
# TC-08: --ep option
# =============================================================================


class TestEPOption:
    @pytest.mark.parametrize(
        "ep_val",
        [
            "qnn",
            "openvino",
            "vitisai",
            "QNNExecutionProvider",
            "OpenVINOExecutionProvider",
            "VitisAIExecutionProvider",
        ],
    )
    def test_valid_ep_values(self, runner: CliRunner, ep_val: str) -> None:
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--text", "hi", "--ep", ep_val])
        assert result.exit_code == 0, f"--ep {ep_val} failed: {result.output}"

    def test_invalid_ep_exits_nonzero(self, runner: CliRunner) -> None:
        result = runner.invoke(run, ["--model", "m", "--text", "hi", "--ep", "cuda"])
        assert result.exit_code != 0

    def test_ep_default_none(self, runner: CliRunner) -> None:
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            runner.invoke(run, ["--model", "m", "--text", "hi"])
        assert engine.load.call_args.kwargs["ep"] is None


# =============================================================================
# TC-09: --schema option
# =============================================================================


class TestSchemaOption:
    def test_schema_prints_and_exits_without_inference(self, runner: CliRunner) -> None:
        """--schema prints schema and exits; no predict() call."""
        schema = [InputField(name="image", type="image", required=True)]
        engine = _make_mock_engine(
            _MINIMAL_RESULT,
            schema=schema,
            params=[{"name": "top_k", "type": "integer", "default": 5}],
        )
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--schema"])
        assert result.exit_code == 0
        engine.predict.assert_not_called()
        assert "inputs" in result.output.lower()

    def test_schema_shows_parameters(self, runner: CliRunner) -> None:
        engine = _make_mock_engine(
            _MINIMAL_RESULT,
            params=[{"name": "top_k", "type": "integer", "default": 5}],
        )
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--schema"])
        assert result.exit_code == 0
        assert "top_k" in result.output

    def test_schema_without_inputs_still_works(self, runner: CliRunner) -> None:
        """--schema alone (no --text/--file/--input) should succeed."""
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--schema"])
        assert result.exit_code == 0


# =============================================================================
# TC-10: --format option
# =============================================================================


class TestFormatOption:
    def test_default_is_text(self, runner: CliRunner) -> None:
        engine = _make_mock_engine(
            {**_MINIMAL_RESULT, "predictions": [{"label": "a", "score": 0.5}]}
        )
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--text", "hi"])
        assert result.exit_code == 0
        assert "Task:" in result.output  # text format has this header

    def test_json_format(self, runner: CliRunner) -> None:
        engine = _make_mock_engine({"task": "test", "predictions": []})
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--text", "hi", "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["task"] == "test"

    def test_invalid_format_exits_nonzero(self, runner: CliRunner) -> None:
        result = runner.invoke(run, ["--model", "m", "--text", "hi", "--format", "xml"])
        assert result.exit_code != 0


# =============================================================================
# TC-11: --output / -o option
# =============================================================================


class TestOutputOption:
    def test_output_to_file_json(self, runner: CliRunner, tmp_path: Path) -> None:
        out = tmp_path / "result.json"
        engine = _make_mock_engine({"task": "test", "predictions": []})
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "m", "--text", "hi", "--format", "json", "-o", str(out)],
            )
        assert result.exit_code == 0
        assert json.loads(out.read_text())["task"] == "test"

    def test_output_to_file_text(self, runner: CliRunner, tmp_path: Path) -> None:
        out = tmp_path / "result.txt"
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "m", "--text", "hi", "--format", "text", "-o", str(out)],
            )
        assert result.exit_code == 0
        content = out.read_text()
        assert "Task:" in content

    def test_output_nothing_to_stdout_when_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """When -o is specified, nothing should go to stdout."""
        out = tmp_path / "result.json"
        engine = _make_mock_engine({"task": "test", "predictions": []})
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "m", "--text", "hi", "--format", "json", "-o", str(out)],
            )
        assert result.exit_code == 0
        # stdout should be empty (no result text)
        assert result.output.strip() == ""


# =============================================================================
# TC-12: --port option
# =============================================================================


class TestPortOption:
    def test_default_port_is_8000(self, runner: CliRunner) -> None:
        """Default port is 8000 per help text."""
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with (
            patch("httpx.Client", side_effect=ConnectionError),
            patch(_ENGINE_PATH, return_value=engine),
        ):
            result = runner.invoke(run, ["--model", "m", "--text", "hi", "--connect"])
        assert result.exit_code == 0

    def test_custom_port_forwarded(self, runner: CliRunner) -> None:
        """--port 9999 changes the port used for server health check."""
        _make_mock_engine(_MINIMAL_RESULT)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        health = MagicMock(status_code=200)
        health.json.return_value = {"model_id": "m"}
        mock_client.get.return_value = health
        predict_resp = MagicMock()
        predict_resp.json.return_value = _MINIMAL_RESULT
        mock_client.post.return_value = predict_resp

        with patch("httpx.Client", return_value=mock_client):
            result = runner.invoke(
                run, ["--model", "m", "--text", "hi", "--connect", "--port", "9999"]
            )
        assert result.exit_code == 0
        url = mock_client.get.call_args.args[0]
        assert ":9999" in url


# =============================================================================
# TC-13: --connect option
# =============================================================================


class TestConnectOption:
    @staticmethod
    def _server_mock(
        health_json: dict,
        predict_json: dict | None = None,
        health_status: int = 200,
    ) -> MagicMock:
        client = MagicMock()
        health_resp = MagicMock(status_code=health_status)
        health_resp.json.return_value = health_json
        client.get.return_value = health_resp
        if predict_json is not None:
            pred_resp = MagicMock()
            pred_resp.json.return_value = predict_json
            client.post.return_value = pred_resp
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=client)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    def test_connect_routes_to_server(self, runner: CliRunner) -> None:
        server_resp = {**_MINIMAL_RESULT, "predictions": [{"label": "ok", "score": 1.0}]}
        ctx = self._server_mock(health_json={"model_id": "m"}, predict_json=server_resp)
        with patch("httpx.Client", return_value=ctx):
            result = runner.invoke(run, ["--model", "m", "--text", "hello", "--connect"])
        assert result.exit_code == 0
        assert "ok" in result.output

    def test_connect_fallback_when_no_server(self, runner: CliRunner) -> None:
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with (
            patch("httpx.Client", side_effect=ConnectionError("refused")),
            patch(_ENGINE_PATH, return_value=engine),
        ):
            result = runner.invoke(run, ["--model", "m", "--text", "hello", "--connect"])
        assert result.exit_code == 0
        engine.load.assert_called_once()

    def test_connect_fallback_on_model_mismatch(self, runner: CliRunner) -> None:
        engine = _make_mock_engine(_MINIMAL_RESULT)
        ctx = self._server_mock(health_json={"model_id": "other-model"})
        with (
            patch("httpx.Client", return_value=ctx),
            patch(_ENGINE_PATH, return_value=engine),
        ):
            result = runner.invoke(run, ["--model", "my-model", "--text", "hi", "--connect"])
        assert result.exit_code == 0
        engine.load.assert_called_once()

    def test_connect_without_inputs_goes_to_embedded(self, runner: CliRunner) -> None:
        """--connect without any input → embedded path (hint shown)."""
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--connect"])
        assert result.exit_code == 0
        # No predict should have been called (no inputs)
        engine.predict.assert_not_called()

    def test_connect_file_plus_input_forwards_to_server(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """--connect + --file + --input → forwards via server with inputs form field."""
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"data")
        engine = _make_mock_engine(_MINIMAL_RESULT)
        ctx = self._server_mock(
            health_json={"model_id": "m"},
            predict_json=_MINIMAL_RESULT,
        )
        with (
            patch("httpx.Client", return_value=ctx),
            patch(_ENGINE_PATH, return_value=engine),
        ):
            result = runner.invoke(
                run,
                ["--model", "m", "--file", str(img), "-I", "question=What?", "--connect"],
            )
        assert result.exit_code == 0
        # Should have been forwarded to server, not fallen back to embedded
        engine.load.assert_not_called()


# =============================================================================
# TC-14: Input conflicts and interactions
# =============================================================================


class TestInputConflicts:
    def test_input_param_collision_exits_2(self, runner: CliRunner) -> None:
        """-I key=val and -P key=val → collision error."""
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "m", "-I", "top_k=5", "-P", "top_k=10"],
            )
        assert result.exit_code == 2
        assert "specified as both" in result.output.lower()

    def test_file_and_text_combined(self, runner: CliRunner, tmp_path: Path) -> None:
        """--file + --text → both in inputs dict (no schema, no conflict)."""
        img = tmp_path / "a.jpg"
        img.write_bytes(b"img")
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--file", str(img), "--text", "Describe"])
        assert result.exit_code == 0
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["file"] == b"img"
        assert inputs["text"] == "Describe"

    def test_file_and_input_same_field_conflicts(self, runner: CliRunner, tmp_path: Path) -> None:
        """--file + -I image=@other.jpg → conflict for same field."""
        img1 = tmp_path / "a.jpg"
        img2 = tmp_path / "b.jpg"
        img1.write_bytes(b"a")
        img2.write_bytes(b"b")
        schema = [InputField(name="image", type="image", required=True)]
        engine = _make_mock_engine(_MINIMAL_RESULT, schema=schema)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "m", "--file", str(img1), "-I", f"image=@{img2}"],
            )
        assert result.exit_code == 2
        assert "specified twice" in result.output.lower()

    def test_text_and_input_text_conflicts(self, runner: CliRunner) -> None:
        """--text + -I text=... → conflict."""
        schema = [InputField(name="text", type="text", required=True)]
        engine = _make_mock_engine(_MINIMAL_RESULT, schema=schema)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--text", "hi", "-I", "text=bye"])
        assert result.exit_code == 2
        assert "specified twice" in result.output.lower()

    def test_file_not_supported_by_task(self, runner: CliRunner, tmp_path: Path) -> None:
        """--file on a text-only task → error."""
        img = tmp_path / "a.jpg"
        img.write_bytes(b"data")
        schema = [InputField(name="text", type="text", required=True)]
        engine = _make_mock_engine(_MINIMAL_RESULT, schema=schema)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--file", str(img)])
        assert result.exit_code == 2
        assert "not supported" in result.output.lower()

    def test_file_ambiguous_multiple_binary_fields(self, runner: CliRunner, tmp_path: Path) -> None:
        """--file with >1 binary schema fields → ambiguous error."""
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"data")
        schema = [
            InputField(name="image_0", type="image", required=True),
            InputField(name="image_1", type="image", required=True),
        ]
        engine = _make_mock_engine(_MINIMAL_RESULT, schema=schema)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--file", str(img)])
        assert result.exit_code == 2
        assert "ambiguous" in result.output.lower()


# =============================================================================
# TC-15: No-input behavior
# =============================================================================


class TestNoInputBehavior:
    def test_no_inputs_shows_hint_exits_0(self, runner: CliRunner) -> None:
        engine = _make_mock_engine(_MINIMAL_RESULT)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m"])
        assert result.exit_code == 0
        engine.predict.assert_not_called()
        assert "--input" in result.output or "--schema" in result.output


# =============================================================================
# TC-16: Error exit codes
# =============================================================================


class TestExitCodes:
    def test_model_load_error_exits_3(self, runner: CliRunner) -> None:
        engine = MagicMock()
        engine.load.side_effect = RuntimeError("corrupt model")
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--text", "hi"])
        assert result.exit_code == 3
        assert "error loading model" in result.output.lower()

    def test_predict_error_exits_4(self, runner: CliRunner) -> None:
        engine = MagicMock()
        engine.predict.side_effect = RuntimeError("inference boom")
        engine.user_input_schema = None
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "--text", "hi"])
        assert result.exit_code == 4
        assert "error during inference" in result.output.lower()

    def test_coerce_error_exit_code_consistency(self, runner: CliRunner) -> None:
        """Input coercion errors (from _coerce_value) should exit with code 2,
        matching other input validation errors."""
        schema = [InputField(name="val", type="number", required=True)]
        engine = _make_mock_engine(_MINIMAL_RESULT, schema=schema)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "m", "-I", "val=abc"])
        assert result.exit_code == 2, (
            f"Expected exit code 2 for input validation error, got {result.exit_code}. "
            f"Output: {result.output}"
        )


# =============================================================================
# TC-17: --connect + --schema interaction
# =============================================================================


class TestConnectSchemaInteraction:
    def test_schema_with_connect_and_inputs_server_available(self, runner: CliRunner) -> None:
        """--connect --schema --text 'hi' with server available:
        --schema should take priority and print schema (not run inference)."""
        server_resp = {**_MINIMAL_RESULT, "predictions": [{"label": "ok", "score": 1.0}]}
        ctx = MagicMock()
        client = MagicMock()
        health = MagicMock(status_code=200)
        health.json.return_value = {"model_id": "m"}
        client.get.return_value = health
        pred = MagicMock()
        pred.json.return_value = server_resp
        client.post.return_value = pred
        ctx.__enter__ = MagicMock(return_value=client)
        ctx.__exit__ = MagicMock(return_value=False)

        engine = _make_mock_engine(
            _MINIMAL_RESULT,
            schema=[InputField(name="text", type="text", required=True)],
            params=[{"name": "top_k", "type": "integer"}],
        )
        with (
            patch("httpx.Client", return_value=ctx),
            patch(_ENGINE_PATH, return_value=engine),
        ):
            result = runner.invoke(run, ["--model", "m", "--text", "hi", "--schema", "--connect"])
        # --schema takes priority over --connect: should show schema, not inference
        assert "inputs" in result.output.lower() or "Inputs" in result.output, (
            f"Expected schema output, but got inference result instead. Output: {result.output}"
        )


# =============================================================================
# TC-18: _parse_heuristic edge cases
# =============================================================================


class TestParseHeuristic:
    def test_at_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "data.txt"
        f.write_bytes(b"hello")
        result = _parse_heuristic(f"@{f}")
        assert result == b"hello"

    def test_at_nonexistent_file_returns_string(self) -> None:
        """@nonexistent.txt → returns literal string (no error).
        This may be surprising to users who expect an error."""
        result = _parse_heuristic("@/no/such/file.txt")
        assert result == "@/no/such/file.txt"

    def test_json_string_parsed(self) -> None:
        result = _parse_heuristic('["a","b"]')
        assert result == ["a", "b"]

    def test_plain_string(self) -> None:
        result = _parse_heuristic("hello world")
        assert result == "hello world"

    def test_number_string_parsed_as_json(self) -> None:
        """'42' is valid JSON, so parsed to int."""
        result = _parse_heuristic("42")
        assert result == 42

    def test_boolean_string_parsed_as_json(self) -> None:
        """'true' is valid JSON, parsed to Python True."""
        result = _parse_heuristic("true")
        assert result is True


# =============================================================================
# TC-19: _coerce_value edge cases
# =============================================================================


class TestCoerceValue:
    def test_text_type_no_json_parsing(self) -> None:
        """Text type: '42' stays as string '42', not parsed as int."""
        result = _coerce_value("42", "text", "field")
        assert result == "42"
        assert isinstance(result, str)

    def test_number_int_coercion(self) -> None:
        result = _coerce_value("42", "number", "field")
        assert result == 42
        assert isinstance(result, int)

    def test_number_float_coercion(self) -> None:
        result = _coerce_value("3.14", "number", "field")
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_boolean_true(self) -> None:
        assert _coerce_value("true", "boolean", "f") is True
        assert _coerce_value("True", "boolean", "f") is True
        assert _coerce_value("TRUE", "boolean", "f") is True

    def test_boolean_false(self) -> None:
        assert _coerce_value("false", "boolean", "f") is False

    def test_unknown_type_returns_string(self) -> None:
        """Unknown field type falls through and returns string as-is."""
        result = _coerce_value("something", "unknown_type", "field")
        assert result == "something"


# =============================================================================
# TC-20: _format_text edge cases
# =============================================================================


class TestFormatTextEdgeCases:
    def test_dict_predictions(self) -> None:
        """Predictions as dict → 'Output:' section."""
        result = _format_text({**_MINIMAL_RESULT, "predictions": {"answer": "42"}})
        assert "Output:" in result
        assert "answer: 42" in result

    def test_empty_dict_result(self) -> None:
        """Completely empty dict should not crash."""
        text = _format_text({})
        assert "Task:" in text

    def test_predictions_without_score(self) -> None:
        """Prediction dicts without 'score' key."""
        result = _format_text({**_MINIMAL_RESULT, "predictions": [{"label": "cat"}]})
        # Should still display, just without score formatting
        assert "cat" in result


# =============================================================================
# TC-21: _models_match edge cases
# =============================================================================


class TestModelsMatchEdgeCases:
    def test_both_empty(self) -> None:
        assert _models_match("", "") is False

    def test_server_empty(self) -> None:
        assert _models_match("", "model") is False

    def test_exact_match(self) -> None:
        assert _models_match("microsoft/resnet-50", "microsoft/resnet-50") is True

    def test_basename_match_different_org(self) -> None:
        assert _models_match("org-a/bert-base", "org-b/bert-base") is True

    def test_no_match(self) -> None:
        assert _models_match("resnet-50", "vit-base") is False


# =============================================================================
# TC-22: _resolve_shortcuts edge cases (no schema)
# =============================================================================


class TestResolveShortcutsNoSchema:
    def test_file_uses_fallback_name_file(self) -> None:
        """With no schema, --file maps to key 'file'."""
        result = _resolve_shortcuts([b"data"], None, {}, None)
        assert result == {"file": b"data"}

    def test_text_uses_fallback_name_text(self) -> None:
        result = _resolve_shortcuts([], "hello", {}, None)
        assert result == {"text": "hello"}

    def test_file_and_text_combined(self) -> None:
        result = _resolve_shortcuts([b"data"], "hello", {}, None)
        assert result == {"file": b"data", "text": "hello"}

    def test_file_conflict_with_named_input(self) -> None:
        """--file + -I file=@... → conflict."""
        with pytest.raises(Exception, match="specified twice"):
            _resolve_shortcuts([b"data"], None, {"file": b"other"}, None)

    def test_text_conflict_with_named_input(self) -> None:
        with pytest.raises(Exception, match="specified twice"):
            _resolve_shortcuts([], "hello", {"text": "world"}, None)


# =============================================================================
# TC-23: _build_example_command
# =============================================================================


class TestBuildExampleCommand:
    def test_returns_none_without_schema(self) -> None:
        assert _build_example_command("m", None, None) is None

    def test_single_image_uses_file_shortcut(self) -> None:
        schema = [InputField(name="image", type="image", required=True)]
        cmd = _build_example_command("m", schema, None)
        assert "--file" in cmd

    def test_single_text_uses_text_shortcut(self) -> None:
        schema = [InputField(name="text", type="text", required=True)]
        cmd = _build_example_command("m", schema, None)
        assert "--text" in cmd

    def test_multiple_text_fields_use_input(self) -> None:
        """With >1 required text fields, shortcuts can't be used."""
        schema = [
            InputField(name="question", type="text", required=True),
            InputField(name="context", type="text", required=True),
        ]
        cmd = _build_example_command("m", schema, None)
        assert "-I question=" in cmd
        assert "-I context=" in cmd
        assert "--text" not in cmd

    def test_task_param_included_in_command(self) -> None:
        """When task= is provided, --task should appear in the example command."""
        schema = [InputField(name="text", type="text", required=True)]
        cmd = _build_example_command("m", schema, None, task="sentence-similarity")
        assert "--task sentence-similarity" in cmd

    def test_task_none_omits_flag(self) -> None:
        """When task=None, --task should not appear."""
        schema = [InputField(name="text", type="text", required=True)]
        cmd = _build_example_command("m", schema, None, task=None)
        assert "--task" not in cmd


# =============================================================================
# TC-23b: _format_text segmentation-specific formatting
# =============================================================================


class TestFormatTextSegmentation:
    def test_segmentation_header(self) -> None:
        """Segmentation tasks should show 'Results (area coverage):' header."""
        result = _format_text(
            {
                **_MINIMAL_RESULT,
                "task": "image-segmentation",
                "predictions": [{"label": "shirt", "score": 0.5}],
            }
        )
        assert "Results (area coverage):" in result

    def test_segmentation_percentage_format(self) -> None:
        """Segmentation scores should display as percentages."""
        result = _format_text(
            {
                **_MINIMAL_RESULT,
                "task": "image-segmentation",
                "predictions": [{"label": "shirt", "score": 0.5}],
            }
        )
        assert "50.0%" in result

    def test_semantic_segmentation_alias(self) -> None:
        """'semantic-segmentation' should also trigger segmentation formatting."""
        result = _format_text(
            {
                **_MINIMAL_RESULT,
                "task": "semantic-segmentation",
                "predictions": [{"label": "pants", "score": 0.25}],
            }
        )
        assert "Results (area coverage):" in result
        assert "25.0%" in result

    def test_non_segmentation_uses_decimal(self) -> None:
        """Non-segmentation tasks should use decimal score format."""
        result = _format_text(
            {
                **_MINIMAL_RESULT,
                "task": "image-classification",
                "predictions": [{"label": "cat", "score": 0.9}],
            }
        )
        assert "0.9000" in result
        assert "Results:" in result
        assert "area coverage" not in result

    def test_none_score_shows_dash(self) -> None:
        """score=None should display as '—'."""
        result = _format_text(
            {
                **_MINIMAL_RESULT,
                "predictions": [{"label": "x", "score": None}],
            }
        )
        assert "—" in result


# =============================================================================
# TC-23c: _print_result does not mutate input
# =============================================================================


class TestPrintResultNoMutation:
    def test_text_format_does_not_pop_mask(self, tmp_path: Path) -> None:
        """_print_result in text mode should not mutate the original result dict."""
        result = {
            **_MINIMAL_RESULT,
            "task": "image-segmentation",
            "predictions": [
                {"label": "shirt", "score": 0.8, "mask": "base64encodeddata"},
            ],
        }
        out = tmp_path / "out.txt"
        _print_result(result, output_format="text", output_path=out)
        # Original should still have mask
        assert result["predictions"][0]["mask"] == "base64encodeddata"

    def test_json_format_includes_mask(self, tmp_path: Path) -> None:
        """_print_result in json mode should include mask data."""
        result = {
            **_MINIMAL_RESULT,
            "task": "image-segmentation",
            "predictions": [
                {"label": "shirt", "score": 0.8, "mask": "base64data"},
            ],
        }
        out = tmp_path / "out.json"
        _print_result(result, output_format="json", output_path=out)
        assert "base64data" in out.read_text()


# =============================================================================
# TC-24: --help option
# =============================================================================


class TestHelpOption:
    def test_help_shows_all_options(self, runner: CliRunner) -> None:
        result = runner.invoke(run, ["--help"])
        assert result.exit_code == 0
        for option in [
            "--model",
            "--file",
            "--text",
            "--input",
            "--task",
            "--device",
            "--ep",
            "--param",
            "--schema",
            "--format",
            "--output",
            "--port",
            "--connect",
        ]:
            assert option in result.output, f"Missing option {option} in help"

    def test_help_shows_short_flags(self, runner: CliRunner) -> None:
        result = runner.invoke(run, ["--help"])
        for flag in ["-m", "-f", "-t", "-I", "-P", "-o"]:
            assert flag in result.output, f"Missing short flag {flag} in help"
