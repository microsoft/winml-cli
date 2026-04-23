# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""E2E quality-gate tests for ``winml serve``.

No mocks — real models, real HTTP requests, real predictions.  Tests are
organized in four tiers by scope and cost:

Tier 1 — **Endpoint feature gates** (2 fixed models)
    Validates every REST endpoint: ``/v1/health``, ``/v1/predict``,
    ``/v1/predict/file``, ``/v1/schema``, ``/v1/tools``, ``/v1/mcp-schema``,
    ``/v1/hub``, ``/v1/logs``, ``/v1/resources``, ``/v1/ep``.

Tier 2 — **Schema coverage** (all hub models)
    ``GET /v1/schema`` for every ``(model_id, task)`` pair in
    ``hub_models.json`` — lightweight, validates schema discovery via HTTP.

Tier 3 — **Inference coverage** (all hub models)
    Full ``POST /v1/predict`` or ``/v1/predict/file`` per hub model.
    Cache-aware: prefers already-built directories to avoid slow rebuilds.

Tier 4 — **Pipeline parameters** (fixed models)
    Validates that ``params`` are forwarded and affect predictions
    (e.g. ``top_k`` limits output length).

Usage::

    # All tiers
    uv run pytest -m e2e tests/e2e/test_serve_e2e.py -v

    # Tier 1 only (fast regression)
    uv run pytest -m e2e tests/e2e/test_serve_e2e.py -k "Feature" -v

    # Tier 2 only (schema)
    uv run pytest -m e2e tests/e2e/test_serve_e2e.py -k "SchemaAll" -v

    # Tier 3 only (inference matrix)
    uv run pytest -m e2e tests/e2e/test_serve_e2e.py -k "InferenceAll" -v

    # Filter by task or model name
    uv run pytest -m e2e tests/e2e/test_serve_e2e.py -k "text_classification" -v
    uv run pytest -m e2e tests/e2e/test_serve_e2e.py -k "finbert" -v

Markers:
    e2e:     Full end-to-end test with real models
    slow:    Tests that take > 30 seconds
    network: Requires network access to HuggingFace Hub
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from winml.modelkit.serve import create_app


pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.network, pytest.mark.timeout(3600)]


# ---------------------------------------------------------------------------
# Constants — fixed P0 models for Tier 1 / Tier 4
# ---------------------------------------------------------------------------

_IMAGE_HF_ID = "microsoft/resnet-18"
_TEXT_HF_ID = "prajjwal1/bert-tiny"

_SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog."


# ---------------------------------------------------------------------------
# Hub model parametrization (shared with test_run_e2e.py)
# ---------------------------------------------------------------------------

_HUB_JSON = (
    Path(__file__).resolve().parents[2] / "src" / "winml" / "modelkit" / "data" / "hub_models.json"
)
_HUB_DATA = json.loads(_HUB_JSON.read_text(encoding="utf-8"))


def _unique_pairs() -> list[dict[str, str]]:
    """Deduplicate ``(model_id, task)`` — keep first occurrence."""
    seen: set[tuple[str, str]] = set()
    pairs: list[dict[str, str]] = []
    for entry in _HUB_DATA["models"]:
        key = (entry["model_id"], entry["task"])
        if key not in seen:
            seen.add(key)
            pairs.append({"model_id": entry["model_id"], "task": entry["task"]})
    return pairs


_PAIRS = _unique_pairs()


def _pytest_id(pair: dict[str, str]) -> str:
    """Readable pytest ID, e.g. ``finbert-text_classification``."""
    short = pair["model_id"].rsplit("/", 1)[-1]
    task = pair["task"].replace("-", "_")
    return f"{short}-{task}"


# ---------------------------------------------------------------------------
# Cache-aware model resolution (same logic as test_run_e2e.py)
# ---------------------------------------------------------------------------


def _find_cache_dir(model_id: str, task: str | None = None) -> Path | None:
    from winml.modelkit.cache import get_cache_dir, model_id_to_slug
    from winml.modelkit.inference.engine import _find_build_artifacts

    slug = model_id_to_slug(model_id)
    cache_dir = get_cache_dir() / "artifacts" / slug
    if not cache_dir.is_dir():
        return None
    try:
        _find_build_artifacts(cache_dir, task=task)
        return cache_dir
    except FileNotFoundError:
        return None


def _resolve_model_arg(model_id: str, task: str | None = None) -> str:
    """Return the cache directory (fast) or HF model ID (slow rebuild)."""
    cache_dir = _find_cache_dir(model_id, task=task)
    if cache_dir is not None:
        return str(cache_dir)
    return model_id


# ---------------------------------------------------------------------------
# Sample inputs per task — used to build POST /v1/predict JSON bodies
# ---------------------------------------------------------------------------

_TEXT_BY_FIELD: dict[str, str] = {
    "question": "What is the capital of France?",
    "context": (
        "Paris is the capital of France. "
        "It is known for the Eiffel Tower and its rich cultural heritage."
    ),
    "text_1": "A man is eating food.",
    "text_2": "A man is eating a piece of bread.",
}


def _build_predict_body(
    schema_inputs: list[dict],
    task: str,
    test_image_b64: str,
) -> dict[str, Any] | None:
    """Build POST /v1/predict JSON body from schema discovery output.

    Returns ``None`` when no inputs can be determined (caller should skip).
    """
    required = [i for i in schema_inputs if i.get("required", False)]

    if not required:
        # sentence-similarity fallback
        if task == "sentence-similarity":
            return {
                "inputs": {"text_1": _TEXT_BY_FIELD["text_1"], "text_2": _TEXT_BY_FIELD["text_2"]},
            }
        return None

    inputs: dict[str, Any] = {}

    for field in required:
        name = field["name"]
        ftype = field["type"]
        if ftype in ("image", "audio", "video"):
            inputs[name] = test_image_b64
        elif ftype == "text":
            inputs[name] = _TEXT_BY_FIELD.get(name, _SAMPLE_TEXT)
        elif ftype == "json":
            inputs[name] = ["positive", "negative", "neutral"]

    if not inputs:
        return None

    return {"inputs": inputs}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def test_image(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Generate a 224x224 random RGB JPEG (reused across the module)."""
    import numpy as np
    from PIL import Image

    d = tmp_path_factory.mktemp("serve_e2e_assets")
    img_path = d / "test_image.jpg"
    rng = np.random.RandomState(42)
    arr = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(img_path), format="JPEG")
    return str(img_path)


@pytest.fixture(scope="module")
def test_image_bytes(test_image: str) -> bytes:
    """Raw bytes of the test JPEG image."""
    return Path(test_image).read_bytes()


@pytest.fixture(scope="module")
def test_image_b64(test_image_bytes: bytes) -> str:
    """Base64-encoded test image for JSON predict requests."""
    return base64.b64encode(test_image_bytes).decode()


@pytest.fixture(scope="module")
def image_model() -> str:
    """Resolve resnet-18 to cache dir (fast) or HF ID (slow)."""
    return _resolve_model_arg(_IMAGE_HF_ID)


@pytest.fixture(scope="module")
def text_model() -> str:
    """Resolve bert-tiny to cache dir (fast) or HF ID (slow)."""
    return _resolve_model_arg(_TEXT_HF_ID, task="text-classification")


@pytest.fixture(scope="module")
def image_client(image_model: str):
    """TestClient wrapping a Phase 1 serve app with an image model loaded."""
    app = create_app(model_path=image_model)
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def text_client(text_model: str):
    """TestClient wrapping a Phase 1 serve app with a text model loaded."""
    app = create_app(model_path=text_model, task="text-classification")
    with TestClient(app) as client:
        yield client


# =====================================================================
# Tier 1 — Feature gates: endpoint validation with fixed models
# =====================================================================


class TestFeatureHealth:
    """GET /v1/health — liveness and model metadata."""

    def test_health_image_model(self, image_client: TestClient) -> None:
        resp = image_client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["version"]
        assert data["mode"] == "single"
        assert data["task"] == "image-classification"
        assert data["uptime_sec"] >= 0

    def test_health_text_model(self, text_client: TestClient) -> None:
        resp = text_client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["task"] == "text-classification"


class TestFeaturePredictFile:
    """POST /v1/predict/file — image classification via file upload."""

    def test_upload_returns_predictions(
        self,
        image_client: TestClient,
        test_image_bytes: bytes,
    ) -> None:
        resp = image_client.post(
            "/v1/predict/file",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"] == "image-classification"
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) > 0
        pred = data["predictions"][0]
        assert "label" in pred and "score" in pred
        assert isinstance(pred["score"], float)
        assert data["latency_ms"] > 0

    def test_upload_with_model_id_underscore(
        self,
        image_client: TestClient,
        test_image_bytes: bytes,
    ) -> None:
        """model_id="_" (default) should route to the only loaded model."""
        resp = image_client.post(
            "/v1/predict/file",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
            data={"model_id": "_"},
        )
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) > 0

    def test_upload_with_task_hint(
        self,
        image_client: TestClient,
        test_image_bytes: bytes,
    ) -> None:
        resp = image_client.post(
            "/v1/predict/file",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
            data={"task": "image-classification"},
        )
        assert resp.status_code == 200
        assert resp.json()["task"] == "image-classification"


class TestFeaturePredictJson:
    """POST /v1/predict — JSON named inputs."""

    def test_text_classification(self, text_client: TestClient) -> None:
        resp = text_client.post(
            "/v1/predict",
            json={"inputs": {"text": "This product is amazing!"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"] == "text-classification"
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) > 0
        assert data["latency_ms"] > 0

    def test_image_classification_base64(
        self,
        image_client: TestClient,
        test_image_b64: str,
    ) -> None:
        resp = image_client.post(
            "/v1/predict",
            json={"inputs": {"image": test_image_b64}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"] == "image-classification"
        assert len(data["predictions"]) > 0

    def test_empty_inputs_returns_error(self, text_client: TestClient) -> None:
        resp = text_client.post(
            "/v1/predict",
            json={"inputs": {}},
        )
        assert resp.status_code in (400, 422, 500)

    def test_missing_inputs_field_returns_422(self, text_client: TestClient) -> None:
        resp = text_client.post("/v1/predict", json={})
        assert resp.status_code == 422


class TestFeatureSchema:
    """GET /v1/schema — request/response schema discovery."""

    def test_schema_image_model(self, image_client: TestClient) -> None:
        resp = image_client.get("/v1/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"] == "image-classification"
        assert isinstance(data["user_inputs"], list)
        assert len(data["user_inputs"]) > 0
        names = {inp["name"] for inp in data["user_inputs"]}
        assert "image" in names
        assert "endpoints" in data

    def test_schema_text_model(self, text_client: TestClient) -> None:
        resp = text_client.get("/v1/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"] == "text-classification"
        names = {inp["name"] for inp in data["user_inputs"]}
        assert "text" in names

    def test_schema_task_override(self, image_client: TestClient) -> None:
        """?task= query param overrides schema resolution."""
        resp = image_client.get("/v1/schema?task=object-detection")
        assert resp.status_code == 200
        assert resp.json()["task"] == "object-detection"


class TestFeatureTools:
    """GET /v1/tools — OpenAI function-calling tool definitions."""

    def test_tools_image_model(self, image_client: TestClient) -> None:
        resp = image_client.get("/v1/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data
        assert isinstance(data["tools"], list)
        assert len(data["tools"]) > 0
        tool = data["tools"][0]
        assert "type" in tool
        assert tool["type"] == "function"
        assert "function" in tool
        fn = tool["function"]
        assert "name" in fn
        assert "parameters" in fn

    def test_tools_text_model(self, text_client: TestClient) -> None:
        resp = text_client.get("/v1/tools")
        assert resp.status_code == 200
        assert len(resp.json()["tools"]) > 0


class TestFeatureMcpSchema:
    """GET /v1/mcp-schema — MCP-compatible tool definitions."""

    def test_mcp_schema(self, image_client: TestClient) -> None:
        resp = image_client.get("/v1/mcp-schema")
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data
        assert isinstance(data["tools"], list)
        assert len(data["tools"]) > 0
        mcp_tool = data["tools"][0]
        assert "name" in mcp_tool
        assert "description" in mcp_tool
        assert "inputSchema" in mcp_tool
        assert "server_info" in data
        assert data["server_info"]["name"] == "ModelKit Inference"


class TestFeatureHub:
    """GET /v1/hub — model catalog."""

    def test_hub_returns_models(self, image_client: TestClient) -> None:
        resp = image_client.get("/v1/hub")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0
        # Every entry must have model_id and task
        for m in data["models"]:
            assert "model_id" in m
            assert "task" in m
            assert "source" in m


class TestFeatureLogs:
    """GET /v1/logs — ring buffer log polling."""

    def test_logs_returns_structure(self, image_client: TestClient) -> None:
        resp = image_client.get("/v1/logs")
        assert resp.status_code == 200
        data = resp.json()
        assert "lines" in data
        assert "latest_seq" in data
        assert isinstance(data["lines"], list)
        assert isinstance(data["latest_seq"], int)

    def test_logs_after_filter(self, image_client: TestClient) -> None:
        """?after=N filters to lines with seq > N."""
        resp = image_client.get("/v1/logs?after=999999")
        assert resp.status_code == 200
        assert resp.json()["lines"] == []


class TestFeatureResources:
    """GET /v1/resources — runtime memory + request stats."""

    def test_resources(self, image_client: TestClient) -> None:
        resp = image_client.get("/v1/resources")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ready", "loading", "unloaded")
        assert "uptime_sec" in data
        assert data["uptime_sec"] >= 0
        assert "memory_mb" in data
        assert "request_count" in data


class TestFeatureEpSwitch:
    """POST /v1/ep — switch execution provider."""

    def test_switch_to_cpu(self, image_client: TestClient) -> None:
        resp = image_client.post("/v1/ep", json={"ep": "cpu"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["ep"] == "cpu"

    def test_switch_invalid_ep(self, image_client: TestClient) -> None:
        resp = image_client.post("/v1/ep", json={"ep": "nonexistent"})
        assert resp.status_code == 422


class TestFeatureModels:
    """GET /v1/models — list loaded models."""

    def test_list_models(self, image_client: TestClient) -> None:
        resp = image_client.get("/v1/models")
        assert resp.status_code == 200
        models = resp.json()
        assert isinstance(models, list)
        assert len(models) >= 1
        m = models[0]
        assert "model_id" in m
        assert "status" in m
        assert m["status"] == "ready"


class TestFeatureOutputConsistency:
    """Cross-endpoint consistency: predictions from /predict and /predict/file should match."""

    def test_file_vs_json_same_predictions(
        self,
        image_client: TestClient,
        test_image_bytes: bytes,
        test_image_b64: str,
    ) -> None:
        """Both endpoints should return the same top-1 label for the same image."""
        resp_file = image_client.post(
            "/v1/predict/file",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")},
        )
        resp_json = image_client.post(
            "/v1/predict",
            json={"inputs": {"image": test_image_b64}},
        )
        assert resp_file.status_code == 200
        assert resp_json.status_code == 200
        preds_file = resp_file.json()["predictions"]
        preds_json = resp_json.json()["predictions"]
        assert preds_file[0]["label"] == preds_json[0]["label"]


# =====================================================================
# Tier 2 — Schema coverage (all hub models via HTTP)
# =====================================================================


class TestSchemaAllModels:
    """``GET /v1/schema`` for every hub model."""

    @pytest.mark.parametrize("pair", _PAIRS, ids=[_pytest_id(p) for p in _PAIRS])
    def test_schema(self, pair: dict[str, str]) -> None:
        model_arg = _resolve_model_arg(pair["model_id"], task=pair["task"])
        app = create_app(model_path=model_arg, task=pair["task"])
        with TestClient(app) as client:
            resp = client.get("/v1/schema")
        assert resp.status_code == 200, f"GET /v1/schema failed ({resp.status_code}):\n{resp.text}"
        data = resp.json()
        assert data["task"] == pair["task"]
        assert isinstance(data["user_inputs"], list)
        assert "endpoints" in data


# =====================================================================
# Tier 3 — Inference coverage (all hub models, cache-aware)
# =====================================================================


class TestInferenceAllModels:
    """Full ``POST /v1/predict`` for every hub model.

    Flow per model:
      1. Create a serve app with the model
      2. ``GET /v1/schema`` → discover inputs
      3. ``POST /v1/predict`` → validate JSON response
    """

    @pytest.mark.parametrize("pair", _PAIRS, ids=[_pytest_id(p) for p in _PAIRS])
    def test_predict(self, pair: dict[str, str], test_image_b64: str) -> None:
        model_id = pair["model_id"]
        task = pair["task"]
        model_arg = _resolve_model_arg(model_id, task=task)

        app = create_app(model_path=model_arg, task=task)
        with TestClient(app) as client:
            # Step 1: Discover inputs via GET /v1/schema
            schema_resp = client.get("/v1/schema")
            assert schema_resp.status_code == 200, (
                f"GET /v1/schema failed ({schema_resp.status_code}):\n{schema_resp.text}"
            )
            schema = schema_resp.json()

            # Step 2: Build predict body
            body = _build_predict_body(schema["user_inputs"], task, test_image_b64)
            if body is None:
                pytest.skip(
                    f"Cannot determine inputs for task '{task}' (empty schema, no fallback)"
                )

            # Step 3: Run inference
            resp = client.post("/v1/predict", json=body)
            assert resp.status_code == 200, (
                f"POST /v1/predict failed ({resp.status_code}):\n{resp.text}"
            )
            data = resp.json()
            assert "task" in data
            assert "latency_ms" in data
            assert data["latency_ms"] > 0

            # Step 4: Validate response after inference
            health_resp = client.get("/v1/health")
            assert health_resp.status_code == 200
            assert health_resp.json()["status"] == "ready"


# =====================================================================
# Tier 4 — Pipeline parameters (fixed models, params forwarding)
# =====================================================================


class TestPipelineParams:
    """Validate that ``params`` in predict body are forwarded correctly."""

    def test_top_k_limits_predictions(self, text_client: TestClient) -> None:
        """top_k=1 should return at most 1 prediction."""
        resp = text_client.post(
            "/v1/predict",
            json={
                "inputs": {"text": "This is a great product!"},
                "params": {"top_k": 1},
            },
        )
        assert resp.status_code == 200
        preds = resp.json()["predictions"]
        assert isinstance(preds, list)
        assert len(preds) == 1

    def test_top_k_3_returns_up_to_3(self, text_client: TestClient) -> None:
        resp = text_client.post(
            "/v1/predict",
            json={
                "inputs": {"text": "The market is doing well today."},
                "params": {"top_k": 3},
            },
        )
        assert resp.status_code == 200
        preds = resp.json()["predictions"]
        assert isinstance(preds, list)
        assert len(preds) <= 3

    def test_params_empty_dict_ok(self, text_client: TestClient) -> None:
        """Empty params {} should work fine (default behavior)."""
        resp = text_client.post(
            "/v1/predict",
            json={"inputs": {"text": "Hello world"}, "params": {}},
        )
        assert resp.status_code == 200
        assert resp.json()["latency_ms"] > 0


# =====================================================================
# Error handling — negative cases
# =====================================================================


class TestErrorHandling:
    """Validate proper error responses for malformed requests."""

    def test_predict_file_no_file(self, image_client: TestClient) -> None:
        """Missing file field → 422."""
        resp = image_client.post("/v1/predict/file")
        assert resp.status_code == 422

    def test_predict_file_too_large(self, image_client: TestClient) -> None:
        """File > 20 MB → 413."""
        huge = b"\x00" * (21 * 1024 * 1024)
        resp = image_client.post(
            "/v1/predict/file",
            files={"file": ("big.bin", huge, "application/octet-stream")},
        )
        assert resp.status_code == 413

    def test_predict_json_inputs_params_collision(self, text_client: TestClient) -> None:
        """Same key in inputs and params → 400."""
        resp = text_client.post(
            "/v1/predict",
            json={
                "inputs": {"text": "hi", "top_k": 5},
                "params": {"top_k": 3},
            },
        )
        assert resp.status_code == 400
        assert "top_k" in resp.json()["detail"]

    def test_predict_json_invalid_base64_image(self, image_client: TestClient) -> None:
        """Non-base64 string for image field → 400."""
        resp = image_client.post(
            "/v1/predict",
            json={"inputs": {"image": "not-valid-base64!@#$"}},
        )
        assert resp.status_code == 400

    def test_switch_ep_invalid(self, image_client: TestClient) -> None:
        """Unknown EP name → 422."""
        resp = image_client.post("/v1/ep", json={"ep": "banana"})
        assert resp.status_code == 422

    def test_unload_model_single_mode(self, image_client: TestClient) -> None:
        """DELETE /v1/models/{id} in single mode → 400."""
        resp = image_client.delete("/v1/models/some-model")
        assert resp.status_code == 400

    def test_load_model_single_mode(self, image_client: TestClient) -> None:
        """POST /v1/models in single mode → 400."""
        resp = image_client.post(
            "/v1/models",
            json={"model_id": "some/model"},
        )
        assert resp.status_code == 400
