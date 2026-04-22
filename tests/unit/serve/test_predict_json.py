# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for POST /v1/predict (JSON named-inputs endpoint).

Validates base64 binary decoding, input/params collision detection,
and basic text/QA inference via the JSON endpoint.

All inference is mocked — no real models are loaded.
"""

from __future__ import annotations

import base64
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
    ep="",
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


class TestPredictJsonBasic:
    """POST /v1/predict — basic input forwarding."""

    def test_text_classification(self) -> None:
        """Single text input is forwarded to engine.predict."""
        engine = _make_engine("text-classification")
        app = _make_app(engine)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict",
                json={"inputs": {"text": "hello world"}, "params": {}},
            )
        assert resp.status_code == 200, resp.text
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["text"] == "hello world"

    def test_question_answering(self) -> None:
        """Multiple text inputs (question + context) are forwarded."""
        engine = _make_engine("question-answering")
        app = _make_app(engine)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict",
                json={
                    "inputs": {
                        "question": "Who is the CEO?",
                        "context": "Tim Cook is the CEO of Apple.",
                    },
                    "params": {},
                },
            )
        assert resp.status_code == 200, resp.text
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["question"] == "Who is the CEO?"
        assert inputs["context"] == "Tim Cook is the CEO of Apple."

    def test_pipeline_params_forwarded(self) -> None:
        """params dict is unpacked as kwargs to engine.predict."""
        engine = _make_engine("text-classification")
        app = _make_app(engine)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict",
                json={
                    "inputs": {"text": "hi"},
                    "params": {"top_k": 3, "threshold": 0.5},
                },
            )
        assert resp.status_code == 200, resp.text
        call_kwargs = engine.predict.call_args.kwargs
        assert call_kwargs["top_k"] == 3
        assert call_kwargs["threshold"] == 0.5


class TestPredictJsonBase64Decode:
    """POST /v1/predict — base64 decoding for binary-typed inputs."""

    def test_image_base64_decoded(self) -> None:
        """Image input sent as base64 string is decoded to bytes."""
        engine = _make_engine("image-classification")
        app = _make_app(engine)
        raw_bytes = b"\xff\xd8fake-jpeg"
        encoded = base64.b64encode(raw_bytes).decode()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict",
                json={"inputs": {"image": encoded}, "params": {}},
            )
        assert resp.status_code == 200, resp.text
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["image"] == raw_bytes

    def test_audio_base64_decoded(self) -> None:
        """Audio input sent as base64 string is decoded to bytes."""
        engine = _make_engine("audio-classification")
        app = _make_app(engine)
        raw_bytes = b"wav-data"
        encoded = base64.b64encode(raw_bytes).decode()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict",
                json={"inputs": {"audio": encoded}, "params": {}},
            )
        assert resp.status_code == 200, resp.text
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["audio"] == raw_bytes

    def test_invalid_base64_returns_400(self) -> None:
        """Non-base64 string for a binary field → 400."""
        engine = _make_engine("image-classification")
        app = _make_app(engine)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict",
                json={"inputs": {"image": "not-valid-base64!@#"}, "params": {}},
            )
        assert resp.status_code == 400
        assert "base64" in resp.json()["detail"].lower()

    def test_text_field_not_decoded(self) -> None:
        """Text-typed inputs are passed through as-is, not base64-decoded."""
        engine = _make_engine("text-classification")
        app = _make_app(engine)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict",
                json={"inputs": {"text": "aGVsbG8="}, "params": {}},
            )
        assert resp.status_code == 200, resp.text
        inputs = engine.predict.call_args.kwargs["inputs"]
        # "aGVsbG8=" is valid base64 but should NOT be decoded for text type
        assert inputs["text"] == "aGVsbG8="


class TestPredictJsonCollision:
    """POST /v1/predict — input/params collision detection."""

    def test_same_key_in_inputs_and_params(self) -> None:
        """Key present in both inputs and params → 400."""
        engine = _make_engine("text-classification")
        app = _make_app(engine)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict",
                json={
                    "inputs": {"text": "hi", "top_k": 5},
                    "params": {"top_k": 3},
                },
            )
        assert resp.status_code == 400
        assert "top_k" in resp.json()["detail"]

    def test_no_collision_when_keys_disjoint(self) -> None:
        """Disjoint inputs and params keys → 200."""
        engine = _make_engine("text-classification")
        app = _make_app(engine)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict",
                json={
                    "inputs": {"text": "hello"},
                    "params": {"top_k": 5},
                },
            )
        assert resp.status_code == 200, resp.text


class TestPredictJsonZeroShot:
    """POST /v1/predict — zero-shot tasks with mixed input types."""

    def test_zero_shot_image_classification(self) -> None:
        """image (base64) + candidate_labels (JSON list) via JSON endpoint."""
        engine = _make_engine("zero-shot-image-classification")
        app = _make_app(engine)
        img_b64 = base64.b64encode(b"jpeg-data").decode()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict",
                json={
                    "inputs": {
                        "image": img_b64,
                        "candidate_labels": ["cat", "dog"],
                    },
                    "params": {},
                },
            )
        assert resp.status_code == 200, resp.text
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["image"] == b"jpeg-data"
        assert inputs["candidate_labels"] == ["cat", "dog"]

    def test_zero_shot_text_classification(self) -> None:
        """text + candidate_labels via JSON endpoint."""
        engine = _make_engine("zero-shot-classification")
        app = _make_app(engine)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/predict",
                json={
                    "inputs": {
                        "text": "I love programming",
                        "candidate_labels": ["tech", "sports", "food"],
                    },
                    "params": {},
                },
            )
        assert resp.status_code == 200, resp.text
        inputs = engine.predict.call_args.kwargs["inputs"]
        assert inputs["text"] == "I love programming"
        assert inputs["candidate_labels"] == ["tech", "sports", "food"]
