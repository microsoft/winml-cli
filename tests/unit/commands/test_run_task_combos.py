# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Model x Task combination tests for ``winml run`` parameter parsing.

Verifies that different task types from TASK_REGISTRY correctly parse
inputs via --file/--text shortcuts and --input named inputs, and that
the resolved inputs dict matches what the InferenceEngine expects.

All inference is mocked — these tests validate CLI → engine.predict()
argument plumbing, not actual model inference.

Test matrix (one representative task per input pattern):

┌──────────────────────────────────┬────────────────────┬──────────────────────────────────┐
│ Task                             │ Input pattern       │ CLI invocation                   │
├──────────────────────────────────┼────────────────────┼──────────────────────────────────┤
│ image-classification             │ single image        │ --file cat.jpg                   │
│ text-classification              │ single text         │ --text "hello"                   │
│ question-answering               │ text + text         │ -I question=... -I context=...   │
│ zero-shot-classification         │ text + json         │ --text "..." -I labels='[...]'   │
│ visual-question-answering        │ image + text        │ --file img.jpg --text "What?"    │
│ keypoint-matching                │ image pair          │ -I image_0=@a -I image_1=@b      │
│ mask-generation                  │ image + opt json    │ --file img.jpg [-I points=...]   │
│ image-to-text                    │ image + opt text    │ --file img.jpg [-I prompt="..."]  │
│ text-generation                  │ text + -P params    │ --text "Once" -P max_new_tokens=…│
│ (no schema / raw)                │ heuristic           │ -I tensor='[1,2,3]'              │
└──────────────────────────────────┴────────────────────┴──────────────────────────────────┘
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.run import run
from winml.modelkit.inference import TASK_REGISTRY, InputField


if TYPE_CHECKING:
    from pathlib import Path

_ENGINE_PATH = "winml.modelkit.inference.InferenceEngine"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _make_engine(
    task: str,
    *,
    schema: list[InputField] | None = "auto",
    params: list[dict] | None = None,
) -> MagicMock:
    """Build a mock engine for the given task.

    If schema="auto", resolves from TASK_REGISTRY.
    """
    if schema == "auto":
        spec = TASK_REGISTRY.get(task)
        schema = spec.user_inputs if spec else None

    result = MagicMock()
    result.model_dump.return_value = {
        "task": task,
        "device": "cpu",
        "ep": "",
        "latency_ms": 10.0,
        "predictions": [{"label": "ok", "score": 0.99}],
    }
    engine = MagicMock()
    engine.predict.return_value = result
    engine.user_input_schema = schema
    engine.task = task
    engine.model_id = f"mock/{task}-model"
    engine.model_path = f"mock/{task}-model"
    engine.pipeline_params = params
    return engine


# =====================================================================
# 1. Single image task: image-classification
# =====================================================================


class TestImageClassification:
    """image-classification: one required image input."""

    TASK = "image-classification"

    def test_file_shortcut(self, runner: CliRunner, tmp_path: Path) -> None:
        """--file maps to the 'image' schema field."""
        img = tmp_path / "cat.jpg"
        img.write_bytes(b"\xff\xd8cat-jpeg")
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "resnet", "--file", str(img)])
        assert result.exit_code == 0, result.output
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["image"] == b"\xff\xd8cat-jpeg"

    def test_named_input(self, runner: CliRunner, tmp_path: Path) -> None:
        """-I image=@path uses @-syntax for file input."""
        img = tmp_path / "dog.png"
        img.write_bytes(b"png-data")
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "resnet", "-I", f"image=@{img}"])
        assert result.exit_code == 0, result.output
        assert engine.predict.call_args.kwargs["inputs"]["image"] == b"png-data"

    def test_text_rejected(self, runner: CliRunner) -> None:
        """--text on an image-only task → error."""
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "resnet", "--text", "hello"])
        assert result.exit_code == 2
        assert "not supported" in result.output.lower()

    def test_schema_json(self, runner: CliRunner) -> None:
        """--schema --format json returns correct schema structure."""
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "resnet", "--schema", "--format", "json"])
        assert result.exit_code == 0
        schema = json.loads(result.output)
        assert schema["task"] == self.TASK
        assert len(schema["inputs"]) == 1
        assert schema["inputs"][0]["name"] == "image"
        assert schema["inputs"][0]["type"] == "image"
        assert schema["inputs"][0]["required"] is True


# =====================================================================
# 2. Single text task: text-classification
# =====================================================================


class TestTextClassification:
    """text-classification: one required text input."""

    TASK = "text-classification"

    def test_text_shortcut(self, runner: CliRunner) -> None:
        """--text maps to the 'text' schema field."""
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "bert", "--text", "This movie is great"])
        assert result.exit_code == 0, result.output
        assert engine.predict.call_args.kwargs["inputs"]["text"] == "This movie is great"

    def test_named_input(self, runner: CliRunner) -> None:
        """-I text='...' works identically."""
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "bert", "-I", "text=This movie is great"])
        assert result.exit_code == 0, result.output
        assert engine.predict.call_args.kwargs["inputs"]["text"] == "This movie is great"

    def test_file_rejected(self, runner: CliRunner, tmp_path: Path) -> None:
        """--file on a text-only task → error."""
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"data")
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "bert", "--file", str(img)])
        assert result.exit_code == 2
        assert "not supported" in result.output.lower()


# =====================================================================
# 3. Text + text task: question-answering
# =====================================================================


class TestQuestionAnswering:
    """question-answering: two required text inputs (question, context)."""

    TASK = "question-answering"

    def test_named_inputs(self, runner: CliRunner) -> None:
        """-I question=... -I context=... are both forwarded."""
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                [
                    "--model",
                    "roberta-qa",
                    "-I",
                    "question=Who is the CEO?",
                    "-I",
                    "context=Tim Cook is the CEO of Apple.",
                ],
            )
        assert result.exit_code == 0, result.output
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["question"] == "Who is the CEO?"
        assert inputs["context"] == "Tim Cook is the CEO of Apple."

    def test_text_shortcut_ambiguous(self, runner: CliRunner) -> None:
        """--text is ambiguous (two text fields) → error."""
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "roberta-qa", "--text", "Who?"])
        assert result.exit_code == 2
        assert "ambiguous" in result.output.lower()


# =====================================================================
# 4. Text + json task: zero-shot-classification
# =====================================================================


class TestZeroShotClassification:
    """zero-shot-classification: text + json candidate_labels."""

    TASK = "zero-shot-classification"

    def test_text_shortcut_plus_json_input(self, runner: CliRunner) -> None:
        """--text for the single text field + -I labels for json field."""
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                [
                    "--model",
                    "bart-mnli",
                    "--text",
                    "I love programming",
                    "-I",
                    'candidate_labels=["positive","negative","neutral"]',
                ],
            )
        assert result.exit_code == 0, result.output
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["text"] == "I love programming"
        assert inputs["candidate_labels"] == ["positive", "negative", "neutral"]

    def test_all_named_inputs(self, runner: CliRunner) -> None:
        """Both inputs via -I."""
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                [
                    "--model",
                    "bart-mnli",
                    "-I",
                    "text=Great weather",
                    "-I",
                    'candidate_labels=["weather","sports"]',
                ],
            )
        assert result.exit_code == 0, result.output
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["text"] == "Great weather"
        assert inputs["candidate_labels"] == ["weather", "sports"]

    def test_invalid_json_labels(self, runner: CliRunner) -> None:
        """Invalid JSON for candidate_labels → error exit 2."""
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                [
                    "--model",
                    "bart-mnli",
                    "--text",
                    "test",
                    "-I",
                    "candidate_labels=not-json",
                ],
            )
        assert result.exit_code == 2
        assert "invalid json" in result.output.lower()


# =====================================================================
# 5. Image + text task: visual-question-answering
# =====================================================================


class TestVisualQuestionAnswering:
    """visual-question-answering: required image + required text (question)."""

    TASK = "visual-question-answering"

    def test_file_and_text_shortcuts(self, runner: CliRunner, tmp_path: Path) -> None:
        """--file + --text map to image and question fields."""
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"vqa-image")
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "vilt", "--file", str(img), "--text", "What color?"],
            )
        assert result.exit_code == 0, result.output
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["image"] == b"vqa-image"
        assert inputs["question"] == "What color?"

    def test_all_named_inputs(self, runner: CliRunner, tmp_path: Path) -> None:
        """Both via -I (no shortcuts)."""
        img = tmp_path / "img.png"
        img.write_bytes(b"png")
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                [
                    "--model",
                    "vilt",
                    "-I",
                    f"image=@{img}",
                    "-I",
                    "question=What is this?",
                ],
            )
        assert result.exit_code == 0, result.output
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["image"] == b"png"
        assert inputs["question"] == "What is this?"


# =====================================================================
# 6. Image pair task: keypoint-matching
# =====================================================================


class TestKeypointMatching:
    """keypoint-matching: two required image inputs (image_0, image_1)."""

    TASK = "keypoint-matching"

    def test_named_file_inputs(self, runner: CliRunner, tmp_path: Path) -> None:
        """Must use -I with @-syntax for each image."""
        a = tmp_path / "a.jpg"
        b = tmp_path / "b.jpg"
        a.write_bytes(b"img-a")
        b.write_bytes(b"img-b")
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                [
                    "--model",
                    "superglue",
                    "-I",
                    f"image_0=@{a}",
                    "-I",
                    f"image_1=@{b}",
                ],
            )
        assert result.exit_code == 0, result.output
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["image_0"] == b"img-a"
        assert inputs["image_1"] == b"img-b"

    def test_file_shortcut_ambiguous(self, runner: CliRunner, tmp_path: Path) -> None:
        """--file is ambiguous (two image fields) → error."""
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"data")
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "superglue", "--file", str(img)])
        assert result.exit_code == 2
        assert "ambiguous" in result.output.lower()


# =====================================================================
# 7. Image + optional json: mask-generation (SAM)
# =====================================================================


class TestMaskGeneration:
    """mask-generation: required image + optional json inputs."""

    TASK = "mask-generation"

    def test_image_only(self, runner: CliRunner, tmp_path: Path) -> None:
        """--file alone (optional json inputs omitted)."""
        img = tmp_path / "scene.jpg"
        img.write_bytes(b"scene-data")
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "sam", "--file", str(img)])
        assert result.exit_code == 0, result.output
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["image"] == b"scene-data"

    def test_image_with_optional_points(self, runner: CliRunner, tmp_path: Path) -> None:
        """--file + -I input_points=[[100,200]]."""
        img = tmp_path / "scene.jpg"
        img.write_bytes(b"scene")
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                [
                    "--model",
                    "sam",
                    "--file",
                    str(img),
                    "-I",
                    "input_points=[[100,200]]",
                ],
            )
        assert result.exit_code == 0, result.output
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["image"] == b"scene"
        assert inputs["input_points"] == [[100, 200]]


# =====================================================================
# 8. Image + optional text: image-to-text
# =====================================================================


class TestImageToText:
    """image-to-text: required image + optional text prompt."""

    TASK = "image-to-text"

    def test_image_only(self, runner: CliRunner, tmp_path: Path) -> None:
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"photo")
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "blip", "--file", str(img)])
        assert result.exit_code == 0, result.output
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["image"] == b"photo"

    def test_image_with_prompt(self, runner: CliRunner, tmp_path: Path) -> None:
        """--file + -I prompt='Describe this image'."""
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"photo")
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                [
                    "--model",
                    "blip",
                    "--file",
                    str(img),
                    "-I",
                    "prompt=Describe this image",
                ],
            )
        assert result.exit_code == 0, result.output
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["image"] == b"photo"
        assert inputs["prompt"] == "Describe this image"

    def test_text_shortcut_maps_to_prompt_field(self, runner: CliRunner, tmp_path: Path) -> None:
        """--text maps to the single text field 'prompt' (even though it's optional)."""
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"photo")
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "blip", "--file", str(img), "--text", "A photo of"],
            )
        assert result.exit_code == 0, result.output
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["prompt"] == "A photo of"


# =====================================================================
# 9. Text + pipeline params: text-generation
# =====================================================================


class TestTextGeneration:
    """text-generation: text input with -P pipeline parameters."""

    TASK = "text-generation"

    def test_text_with_params(self, runner: CliRunner) -> None:
        """--text + -P max_new_tokens + -P temperature."""
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                [
                    "--model",
                    "gpt2",
                    "--text",
                    "Once upon a time",
                    "-P",
                    "max_new_tokens=100",
                    "-P",
                    "temperature=0.7",
                    "-P",
                    "do_sample=true",
                ],
            )
        assert result.exit_code == 0, result.output
        kwargs = engine.predict.call_args.kwargs
        assert kwargs["inputs"]["text"] == "Once upon a time"
        assert kwargs["max_new_tokens"] == 100
        assert kwargs["temperature"] == pytest.approx(0.7)
        assert kwargs["do_sample"] is True

    def test_text_param_collision_detected(self, runner: CliRunner) -> None:
        """--text 'hello' -P text=100 → collision error (BUG-6 was fixed)."""
        engine = _make_engine(self.TASK)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(
                run,
                ["--model", "gpt2", "--text", "hello", "-P", "text=100"],
            )
        assert result.exit_code == 2
        assert "specified as both" in result.output.lower()


# =====================================================================
# 10. No schema (raw / unregistered task): heuristic parsing
# =====================================================================


class TestNoSchemaHeuristic:
    """Unregistered task: schema is None, heuristic parsing applies."""

    def test_text_as_string(self, runner: CliRunner) -> None:
        """Plain text passed through as-is."""
        engine = _make_engine("custom-task", schema=None)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "custom", "-I", "text=hello world"])
        assert result.exit_code == 0, result.output
        assert engine.predict.call_args.kwargs["inputs"]["text"] == "hello world"

    def test_json_value_parsed(self, runner: CliRunner) -> None:
        """JSON string auto-parsed to list."""
        engine = _make_engine("custom-task", schema=None)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "custom", "-I", "data=[1,2,3]"])
        assert result.exit_code == 0, result.output
        assert engine.predict.call_args.kwargs["inputs"]["data"] == [1, 2, 3]

    def test_number_parsed(self, runner: CliRunner) -> None:
        """Numeric string auto-parsed as int/float."""
        engine = _make_engine("custom-task", schema=None)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "custom", "-I", "threshold=0.5"])
        assert result.exit_code == 0, result.output
        assert engine.predict.call_args.kwargs["inputs"]["threshold"] == pytest.approx(0.5)

    def test_at_file_reads_bytes(self, runner: CliRunner, tmp_path: Path) -> None:
        """@path reads file bytes even without schema."""
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00\x01\x02")
        engine = _make_engine("custom-task", schema=None)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "custom", "-I", f"blob=@{f}"])
        assert result.exit_code == 0, result.output
        assert engine.predict.call_args.kwargs["inputs"]["blob"] == b"\x00\x01\x02"

    def test_text_shortcut_fallback(self, runner: CliRunner) -> None:
        """--text without schema uses 'text' as key."""
        engine = _make_engine("custom-task", schema=None)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "custom", "--text", "raw input"])
        assert result.exit_code == 0, result.output
        assert engine.predict.call_args.kwargs["inputs"]["text"] == "raw input"

    def test_file_shortcut_fallback(self, runner: CliRunner, tmp_path: Path) -> None:
        """--file without schema uses 'file' as key."""
        f = tmp_path / "photo.jpg"
        f.write_bytes(b"jpeg")
        engine = _make_engine("custom-task", schema=None)
        with patch(_ENGINE_PATH, return_value=engine):
            result = runner.invoke(run, ["--model", "custom", "--file", str(f)])
        assert result.exit_code == 0, result.output
        assert engine.predict.call_args.kwargs["inputs"]["file"] == b"jpeg"
