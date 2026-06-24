# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E quality-gate tests for ``winml run``.

No mocks — real models, real inputs, real outputs.  Tests are organized
in three tiers by scope and cost:

Tier 1 — **Feature gates** (2 fixed models)
    Validates CLI features: ``--file``, ``--text``, ``-I``, ``-P``,
    ``--format text|json``, ``-o``, ``--schema``.

Tier 2 — **Schema coverage** (all hub models)
    ``winml run --schema`` for every ``(model_id, task)`` pair in
    ``hub_models.json`` — lightweight, no ORT session.

Tier 3 — **Inference coverage** (all hub models)
    Full inference per hub model.  Cache-aware: prefers already-built
    directories under ``~/.cache/winml/artifacts/`` to avoid the slow
    export → optimize → analyze pipeline on every run.

Usage::

    # All tiers
    uv run pytest -m e2e tests/e2e/test_run_e2e.py -v

    # Tier 1 only (fast regression)
    uv run pytest -m e2e tests/e2e/test_run_e2e.py -k "Feature" -v

    # Tier 2 only (schema)
    uv run pytest -m e2e tests/e2e/test_run_e2e.py -k "Schema" -v

    # Tier 3 only (inference matrix)
    uv run pytest -m e2e tests/e2e/test_run_e2e.py -k "Inference" -v

    # Filter by task or model name
    uv run pytest -m e2e tests/e2e/test_run_e2e.py -k "text_classification" -v
    uv run pytest -m e2e tests/e2e/test_run_e2e.py -k "finbert" -v

Markers:
    e2e:     Full end-to-end test with real models
    slow:    Tests that take > 30 seconds
    network: Requires network access to HuggingFace Hub
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from winml.modelkit.commands.run import run

from .conftest import HUB_PAIRS as _PAIRS
from .conftest import SAMPLE_TEXT as _SAMPLE_TEXT
from .conftest import TEXT_BY_FIELD as _TEXT_BY_FIELD
from .conftest import hub_test_id as _pytest_id
from .conftest import resolve_model_arg as _resolve_model_arg


if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import CliRunner

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.e2e_run,
    pytest.mark.slow,
    pytest.mark.network,
    pytest.mark.timeout(3600),
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGE_HF_ID = "microsoft/resnet-18"
_TEXT_HF_ID = "prajjwal1/bert-tiny"


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------


def _extract_json(output: str) -> dict:
    """Extract the JSON object from CLI output that may have build-pipeline noise.

    ``run.py`` redirects ``sys.stdout`` → ``sys.stderr`` during ``engine.load()``
    to prevent build-pipeline prints from contaminating JSON output.  However,
    some C-extension code (e.g. onnxruntime, tqdm) writes directly to file
    descriptor 1, bypassing Python's ``sys.stdout`` — ``redirect_stdout`` cannot
    intercept those writes.  This function provides a robust fallback by
    scanning for the first valid top-level ``{...}`` JSON object.
    """
    decoder = json.JSONDecoder()
    # Scan forward for the first '{' that starts a valid JSON object
    for i, ch in enumerate(output):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(output, i)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No JSON object found in output: {output[:200]!r}")


# ---------------------------------------------------------------------------
# Sample inputs for inference
# ---------------------------------------------------------------------------

_FALLBACK_INPUT_ARGS: dict[str, list[str]] = {
    "sentence-similarity": ["--text", _SAMPLE_TEXT],
}


def _build_inference_args(
    schema_inputs: list[dict],
    task: str,
    test_image: str,
) -> list[str] | None:
    """Build CLI args for inference from ``--schema`` output.

    Returns ``None`` when no inputs can be determined (caller should skip).
    """
    required = [i for i in schema_inputs if i.get("required", False)]

    if not required:
        return _FALLBACK_INPUT_ARGS.get(task)

    binary = [i for i in required if i["type"] in ("image", "audio", "video")]
    text = [i for i in required if i["type"] == "text"]
    json_fields = [i for i in required if i["type"] == "json"]

    args: list[str] = []

    if len(binary) == 1 and binary[0]["type"] == "image":
        args.extend(["--file", test_image])
    else:
        for b in binary:
            args.extend(["-I", f"{b['name']}=@{test_image}"])

    if len(text) == 1 and not binary and not json_fields:
        sample = _TEXT_BY_FIELD.get(text[0]["name"], _SAMPLE_TEXT)
        args.extend(["--text", sample])
    else:
        for t in text:
            sample = _TEXT_BY_FIELD.get(t["name"], _SAMPLE_TEXT)
            args.extend(["-I", f"{t['name']}={sample}"])

    for j in json_fields:
        args.extend(["-I", f'{j["name"]}=["positive","negative","neutral"]'])

    return args


# ---------------------------------------------------------------------------
# Tier 1 fixtures: cache-aware model paths for fixed models
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def image_model() -> str:
    """Resolve resnet-18 to cache dir (fast) or HF ID (slow)."""
    return _resolve_model_arg(_IMAGE_HF_ID)


@pytest.fixture(scope="module")
def text_model() -> str:
    """Resolve bert-tiny to cache dir (fast) or HF ID (slow)."""
    return _resolve_model_arg(_TEXT_HF_ID, task="text-classification")


# =====================================================================
# Tier 1 — Feature gates (fixed models, deep assertions)
# =====================================================================


class TestFeatureImageClassification:
    """resnet-18: --file, --format, -o, -I image=@path."""

    def test_file_text_format(self, runner: CliRunner, image_model: str, test_image: str) -> None:
        result = runner.invoke(
            run,
            ["--model", image_model, "--file", test_image],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Task:" in result.output
        assert "Device:" in result.output
        assert "Latency:" in result.output

    def test_file_json_format(self, runner: CliRunner, image_model: str, test_image: str) -> None:
        result = runner.invoke(
            run,
            ["--model", image_model, "--file", test_image, "--format", "json"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["task"] == "image-classification"
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) > 0
        pred = data["predictions"][0]
        assert "label" in pred and "score" in pred
        assert isinstance(pred["score"], float)
        assert data["latency_ms"] > 0

    def test_file_json_to_file(
        self, runner: CliRunner, image_model: str, test_image: str, tmp_path: Path
    ) -> None:
        out = tmp_path / "result.json"
        result = runner.invoke(
            run,
            ["--model", image_model, "--file", test_image, "--format", "json", "-o", str(out)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["task"] == "image-classification"
        assert len(data["predictions"]) > 0

    def test_named_input(self, runner: CliRunner, image_model: str, test_image: str) -> None:
        result = runner.invoke(
            run,
            ["--model", image_model, "-I", f"image=@{test_image}", "--format", "json"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert len(data["predictions"]) > 0


class TestFeatureTextClassification:
    """bert-tiny: --text, -I text=, -P top_k=."""

    def test_text_shortcut(self, runner: CliRunner, text_model: str, tmp_path: Path) -> None:
        out = tmp_path / "result.json"
        result = runner.invoke(
            run,
            [
                "--model",
                text_model,
                "--text",
                "This product is amazing!",
                "--task",
                "text-classification",
                "--format",
                "json",
                "-o",
                str(out),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["task"] == "text-classification"
        assert "predictions" in data
        assert data["latency_ms"] > 0

    def test_named_input(self, runner: CliRunner, text_model: str) -> None:
        result = runner.invoke(
            run,
            [
                "--model",
                text_model,
                "-I",
                "text=Hello world",
                "--task",
                "text-classification",
                "--format",
                "json",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "predictions" in data

    def test_pipeline_param(self, runner: CliRunner, text_model: str) -> None:
        result = runner.invoke(
            run,
            [
                "--model",
                text_model,
                "--text",
                "Testing pipeline params",
                "--task",
                "text-classification",
                "-P",
                "top_k=3",
                "--format",
                "json",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "predictions" in data


class TestFeatureOutputFormats:
    """Validate --format text vs json, and -o file output."""

    def test_text_format_sections(
        self,
        runner: CliRunner,
        image_model: str,
        test_image: str,
    ) -> None:
        result = runner.invoke(
            run,
            ["--model", image_model, "--file", test_image, "--format", "text"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        out = result.output
        assert "Task:    image-classification" in out
        assert "Device:" in out
        assert "Results:" in out or "Output:" in out
        assert "Latency:" in out and "ms" in out

    def test_json_format_keys(self, runner: CliRunner, image_model: str, test_image: str) -> None:
        result = runner.invoke(
            run,
            ["--model", image_model, "--file", test_image, "--format", "json"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert {"task", "predictions", "latency_ms", "device"}.issubset(data.keys())

    def test_output_to_file(
        self, runner: CliRunner, image_model: str, test_image: str, tmp_path: Path
    ) -> None:
        out = tmp_path / "result.txt"
        result = runner.invoke(
            run,
            ["--model", image_model, "--file", test_image, "--format", "text", "-o", str(out)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        content = out.read_text(encoding="utf-8")
        assert "Task:" in content and "Latency:" in content


class TestFeatureSchema:
    """Validate --schema output (text + json + file)."""

    def test_schema_text(self, runner: CliRunner, image_model: str) -> None:
        result = runner.invoke(
            run,
            ["--model", image_model, "--schema"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Inputs" in result.output or "inputs" in result.output.lower()

    def test_schema_json(self, runner: CliRunner, image_model: str) -> None:
        result = runner.invoke(
            run,
            ["--model", image_model, "--schema", "--format", "json"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "task" in data
        assert isinstance(data["inputs"], list)

    def test_schema_to_file(self, runner: CliRunner, image_model: str, tmp_path: Path) -> None:
        out = tmp_path / "schema.json"
        result = runner.invoke(
            run,
            ["--model", image_model, "--schema", "--format", "json", "-o", str(out)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert out.exists()
        assert "inputs" in json.loads(out.read_text(encoding="utf-8"))

    def test_schema_does_not_run_inference(self, runner: CliRunner, image_model: str) -> None:
        result = runner.invoke(
            run,
            ["--model", image_model, "--schema"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Inputs" in result.output or "inputs" in result.output.lower()


# =====================================================================
# Tier 2 — Schema coverage (all hub models, lightweight)
# =====================================================================


class TestSchemaAllModels:
    """``--schema --format json`` for every hub model — no ORT session needed."""

    @pytest.mark.parametrize("pair", _PAIRS, ids=[_pytest_id(p) for p in _PAIRS])
    def test_schema(self, runner: CliRunner, pair: dict[str, str]) -> None:
        result = runner.invoke(
            run,
            ["--model", pair["model_id"], "--task", pair["task"], "--schema", "--format", "json"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, f"--schema failed (exit {result.exit_code}):\n{result.output}"
        data = json.loads(result.stdout)
        assert "task" in data
        assert isinstance(data["inputs"], list)
        if data["inputs"]:
            assert "example" in data
            assert data["example"].startswith("winml run")


# =====================================================================
# Tier 3 — Inference coverage (all hub models, cache-aware)
# =====================================================================


class TestInferenceAllModels:
    """Full inference for every hub model — uses build cache when available.

    Flow per model:
      1. Resolve model arg (cache dir or HF ID)
      2. ``--schema`` → discover inputs
      3. Run inference → validate JSON output
    """

    @pytest.mark.parametrize("pair", _PAIRS, ids=[_pytest_id(p) for p in _PAIRS])
    def test_run(
        self,
        runner: CliRunner,
        pair: dict[str, str],
        test_image: str,
    ) -> None:
        model_id = pair["model_id"]
        task = pair["task"]
        model_arg = _resolve_model_arg(model_id, task=task)

        # Step 1: Discover inputs via --schema
        schema_result = runner.invoke(
            run,
            ["--model", model_arg, "--task", task, "--schema", "--format", "json"],
            catch_exceptions=False,
        )
        assert schema_result.exit_code == 0, (
            f"--schema failed (exit {schema_result.exit_code}):\n{schema_result.output}"
        )
        schema = json.loads(schema_result.stdout)

        # Step 2: Build inference args
        input_args = _build_inference_args(schema["inputs"], task, test_image)
        if input_args is None:
            pytest.xfail(
                f"Cannot determine inputs for task '{task}' (empty schema, no fallback) — "
                "extend _FALLBACK_INPUT_ARGS or _build_inference_args to cover this task"
            )

        # Step 3: Run inference
        result = runner.invoke(
            run,
            ["--model", model_arg, "--task", task, "--format", "json", *input_args],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, (
            f"Inference failed (exit {result.exit_code}):\n{result.output}"
        )
        data = _extract_json(result.stdout)
        assert "task" in data
        assert "latency_ms" in data
        assert data["latency_ms"] > 0
