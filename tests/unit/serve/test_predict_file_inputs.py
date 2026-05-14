# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for POST /v1/predict/file with the ``inputs`` form field.

Validates that extra named inputs (JSON, text, base64 binary) can be
forwarded alongside the uploaded file, enabling zero-shot tasks and
mask-generation with spatial parameters via the multipart endpoint.

All inference is mocked — no real models are loaded.
"""

from __future__ import annotations

import base64
import json
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from winml.modelkit.inference import TASK_REGISTRY, InputField, Prediction, PredictionResult
from winml.modelkit.serve.app import _register_routes
from winml.modelkit.serve.manager import SingleModelManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OK_RESULT = PredictionResult(
    task="test",
    model_id="mock/model",
    device="cpu",
    ep=None,
    predictions=[Prediction(label="ok", score=0.99)],
    latency_ms=10.0,
)


def _make_engine(
    task: str,
    *,
    schema: list[InputField] | None = "auto",
) -> MagicMock:
    """Build a mock engine for the given task."""
    if schema == "auto":
        spec = TASK_REGISTRY.get(task)
        schema = spec.user_inputs if spec else None

    engine = MagicMock()
    engine.predict.return_value = _OK_RESULT
    engine.user_input_schema = schema
    engine.task = task
    engine.model_id = f"mock/{task}-model"
    engine.model_path = f"mock/{task}-model"
    engine.pipeline_params = None
    engine.request_count = 0
    engine.memory_mb = 0.0
    engine.last_request_at = None
    engine.latency_stats = {
        "mean_ms": 0,
        "min_ms": 0,
        "max_ms": 0,
        "p50_ms": 0,
        "p90_ms": 0,
        "p95_ms": 0,
        "p99_ms": 0,
        "sample_count": 0,
    }
    return engine


def _make_app(engine: MagicMock) -> FastAPI:
    """Build a minimal FastAPI app with the given mocked engine."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import time

        app.state.start_time = time.time()
        app.state.manager = SingleModelManager(engine, idle_timeout_sec=0)
        yield
        app.state.manager.shutdown()

    app = FastAPI(lifespan=lifespan)
    _register_routes(app, mode="single")
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPredictFileWithInputs:
    """POST /v1/predict/file with the ``inputs`` form field."""

    def test_zero_shot_image_classification(self) -> None:
        """file=@img.jpg + inputs={"candidate_labels": [...]} works."""
        engine = _make_engine("zero-shot-image-classification")
        app = _make_app(engine)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict/file",
                files={"file": ("img.jpg", b"\xff\xd8fake-jpeg", "image/jpeg")},
                data={
                    "inputs": json.dumps({"candidate_labels": ["cat", "dog"]}),
                },
            )
        assert resp.status_code == 200, resp.text
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["image"] == b"\xff\xd8fake-jpeg"
        assert inputs["candidate_labels"] == ["cat", "dog"]

    def test_zero_shot_object_detection(self) -> None:
        """file + candidate_labels via inputs field."""
        engine = _make_engine("zero-shot-object-detection")
        app = _make_app(engine)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict/file",
                files={"file": ("img.jpg", b"jpeg-data", "image/jpeg")},
                data={
                    "inputs": json.dumps({"candidate_labels": ["person", "car"]}),
                },
            )
        assert resp.status_code == 200, resp.text
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["image"] == b"jpeg-data"
        assert inputs["candidate_labels"] == ["person", "car"]

    def test_zero_shot_audio_classification(self) -> None:
        """file + candidate_labels for audio zero-shot."""
        engine = _make_engine("zero-shot-audio-classification")
        app = _make_app(engine)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict/file",
                files={"file": ("sound.wav", b"wav-data", "audio/wav")},
                data={
                    "inputs": json.dumps({"candidate_labels": ["music", "speech", "noise"]}),
                },
            )
        assert resp.status_code == 200, resp.text
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["audio"] == b"wav-data"
        assert inputs["candidate_labels"] == ["music", "speech", "noise"]

    def test_mask_generation_with_spatial(self) -> None:
        """file + input_points/input_labels via inputs field."""
        engine = _make_engine("mask-generation")
        app = _make_app(engine)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict/file",
                files={"file": ("scene.jpg", b"scene", "image/jpeg")},
                data={
                    "inputs": json.dumps(
                        {
                            "input_points": [[100, 200]],
                            "input_labels": [1],
                        }
                    ),
                },
            )
        assert resp.status_code == 200, resp.text
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["image"] == b"scene"
        assert inputs["input_points"] == [[100, 200]]
        assert inputs["input_labels"] == [1]

    def test_empty_inputs_field_is_fine(self) -> None:
        """Default empty inputs={} should not break existing behavior."""
        engine = _make_engine("image-classification")
        app = _make_app(engine)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict/file",
                files={"file": ("img.jpg", b"jpeg", "image/jpeg")},
            )
        assert resp.status_code == 200, resp.text
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["image"] == b"jpeg"

    def test_collision_file_and_inputs_same_key(self) -> None:
        """Providing 'image' in both file upload and inputs field → 400."""
        engine = _make_engine("image-classification")
        app = _make_app(engine)
        img_b64 = base64.b64encode(b"other-image").decode()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict/file",
                files={"file": ("img.jpg", b"jpeg", "image/jpeg")},
                data={
                    "inputs": json.dumps({"image": img_b64}),
                },
            )
        assert resp.status_code == 400
        assert "image" in resp.json()["detail"].lower()

    def test_collision_inputs_and_params(self) -> None:
        """Same key in inputs and params → 400."""
        engine = _make_engine("zero-shot-image-classification")
        app = _make_app(engine)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict/file",
                files={"file": ("img.jpg", b"jpeg", "image/jpeg")},
                data={
                    "inputs": json.dumps(
                        {
                            "candidate_labels": ["a", "b"],
                            "top_k": 3,
                        }
                    ),
                    "params": json.dumps({"top_k": 5}),
                },
            )
        assert resp.status_code == 400
        assert "top_k" in resp.json()["detail"]

    def test_invalid_inputs_json(self) -> None:
        """Malformed inputs JSON string → 422."""
        engine = _make_engine("image-classification")
        app = _make_app(engine)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict/file",
                files={"file": ("img.jpg", b"jpeg", "image/jpeg")},
                data={"inputs": "not-valid-json{"},
            )
        assert resp.status_code == 422

    def test_file_with_text_and_inputs(self) -> None:
        """file + text shortcut + extra inputs all merge correctly."""
        engine = _make_engine("zero-shot-image-classification")
        app = _make_app(engine)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict/file",
                files={"file": ("img.jpg", b"jpeg", "image/jpeg")},
                data={
                    "text": "should be ignored — no text field in schema",
                    "inputs": json.dumps({"candidate_labels": ["a", "b"]}),
                },
            )
        # text is ignored because schema has 0 text fields (only image + json)
        assert resp.status_code == 200, resp.text
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["image"] == b"jpeg"
        assert inputs["candidate_labels"] == ["a", "b"]

    def test_keypoint_matching_still_errors(self) -> None:
        """2 binary fields → 400 even with inputs field (must use JSON endpoint)."""
        engine = _make_engine("keypoint-matching")
        app = _make_app(engine)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict/file",
                files={"file": ("a.jpg", b"img-a", "image/jpeg")},
                data={
                    "inputs": json.dumps(
                        {
                            "image_1": base64.b64encode(b"img-b").decode(),
                        }
                    ),
                },
            )
        # _build_file_inputs errors: 2 binary fields
        assert resp.status_code == 400
        assert "binary input" in resp.json()["detail"].lower()
