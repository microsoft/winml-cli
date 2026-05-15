# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Phase 1+ FastAPI inference application.

Activated when ``winml serve <model_path>`` is given a model argument.

Endpoints:
  GET  /v1/health             — liveness + model info
  POST /v1/predict            — image file upload OR JSON tensor inputs
  POST /v1/ep                 — switch execution provider (Phase 1, P0)
  GET  /v1/resources          — runtime memory + request stats (Phase 2)
  GET  /v1/models             — list all loaded models (Phase 3)
  GET  /v1/hub                — curated WinML Hub model catalog

EP shorthand mapping (cpu / dml / qnn / openvino) is handled here
in the serve layer, not inside InferenceEngine or WinMLSession.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.resources
import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..inference import InferenceEngine
from ..inference.types import PredictionResult
from .cli_api import CliRequest, CliResponse, _run_with_semaphore
from .manager import ModelSlotManager, SingleModelManager
from .schema import (
    EpSwitchRequest,
    HealthResponse,
    LatencyStats,
    ModelInfo,
    ModelLoadRequest,
    ModelStatsResponse,
    PredictJsonRequest,
    ResourceResponse,
    ToolsResponse,
)
from .schema_generator import APISchemaGenerator


if TYPE_CHECKING:
    from ..utils.constants import EPNameOrAlias

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Log capture — ring buffer of recent log lines for GET /v1/logs polling
# ---------------------------------------------------------------------------


class _RingHandler(logging.Handler):
    """Capture modelkit log lines into a fixed-size deque."""

    def __init__(self, maxlen: int = 200) -> None:
        super().__init__()
        self._buf: deque[dict] = deque(maxlen=maxlen)
        self._seq: int = 0

    def emit(self, record: logging.LogRecord) -> None:
        self._seq += 1
        self._buf.append(
            {
                "seq": self._seq,
                "ts": round(record.created, 3),
                "level": record.levelname,
                "name": record.name.removeprefix("winml.modelkit."),
                "msg": self.format(record),
            }
        )

    def since(self, after_seq: int) -> list[dict]:
        return [e for e in self._buf if e["seq"] > after_seq]


_log_handler = _RingHandler()
_log_handler.setFormatter(logging.Formatter("%(message)s"))
# Attach to modelkit root logger so all sub-loggers feed into the ring
logging.getLogger("winml.modelkit").addHandler(_log_handler)
logging.getLogger("winml.modelkit").setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    *,
    model_path: str | None,
    task: str | None = None,
    device: str = "auto",
    ep: EPNameOrAlias | None = None,
    idle_timeout_sec: float = 0.0,
    mode: str = "single",
    memory_budget_mb: float = 4096.0,
) -> FastAPI:
    """Build and return the FastAPI application.

    Args:
        model_path: HF model ID, build output dir, or .onnx file.
        task: Explicit task (required for raw .onnx).
        device: Target device ("auto", "cpu", "gpu", "npu").
        ep: Explicit EP short name.
        idle_timeout_sec: Phase 2 idle unload (0 = disabled).
        mode: "single" (Phase 1/2) | "multi" (Phase 3).
        memory_budget_mb: Phase 3 memory cap.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.start_time = time.time()
        if mode == "multi":
            mgr = ModelSlotManager(
                memory_budget_mb=memory_budget_mb,
                idle_timeout_sec=idle_timeout_sec,
                default_device=device,
            )
            if model_path:
                logger.info("Pre-loading model: %s", model_path)
                async with mgr.borrow(model_path):
                    pass
            else:
                logger.info("Multi-model server started (empty — load via POST /v1/models)")
            app.state.manager = mgr
        else:
            engine = InferenceEngine()
            engine.load(model_path, task=task, device=device, ep=ep)
            app.state.manager = SingleModelManager(engine, idle_timeout_sec=idle_timeout_sec)

        logger.info("Model ready")
        yield
        app.state.manager.shutdown()

    app = FastAPI(
        title="WinML CLI Inference Server",
        version=__version__,
        description=(
            "Local REST API for WinML model inference.\n\n"
            "- **Phase 0** `POST /v1/cli/{command}` — CLI wrapper\n"
            "- **Phase 1** `POST /v1/predict` — warm single-model inference\n"
            "- **Phase 2** `GET /v1/resources` — resource monitoring\n"
            "- **Phase 3** `GET /v1/models` — multi-model management"
        ),
        lifespan=lifespan,
    )
    # Permissive CORS for local dev server; no credentials to protect.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve demo UI at /demo
    _static_dir = Path(__file__).parent / "static"
    if _static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

        @app.get("/demo", include_in_schema=False)
        async def demo_ui() -> FileResponse:
            return FileResponse(str(_static_dir / "index.html"))

    _register_routes(app, mode=mode)
    return app


def _register_routes(app: FastAPI, *, mode: str) -> None:
    # ------------------------------------------------------------------
    # Local helpers (closure over app)
    # ------------------------------------------------------------------
    def _get_mgr() -> SingleModelManager | ModelSlotManager:
        mgr = getattr(app.state, "manager", None)
        if mgr is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")
        return mgr

    def _get_start_time() -> float:
        return getattr(app.state, "start_time", time.time())

    def _load_all_manifests() -> list[dict]:
        return [_manifest_from_engine(e) for e in _get_mgr().get_all_engines()]

    def _load_manifest(model_id: str | None = None) -> dict | None:
        engine = _get_mgr().get_engine(model_id)
        return _manifest_from_engine(engine) if engine else None

    # ------------------------------------------------------------------
    # GET /v1/health
    # ------------------------------------------------------------------
    @app.get("/v1/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        mgr = _get_mgr()
        models = await mgr.list_models()
        first = models[0] if models else {}
        status = first.get("status", "loading") if models else "loading"
        return HealthResponse(
            status=status,
            version=__version__,
            mode=mode,
            model_id=first.get("model_id"),
            task=first.get("task"),
            device=first.get("device"),
            ep=first.get("ep"),
            uptime_sec=round(time.time() - _get_start_time(), 1),
        )

    # ------------------------------------------------------------------
    # POST /v1/predict/file — single file upload
    # ------------------------------------------------------------------
    @app.post(
        "/v1/predict/file",
        response_model=PredictionResult,
        tags=["inference"],
        summary="Single file upload inference (image, audio, …)",
    )
    async def predict_file(
        file: UploadFile = File(..., description="Media file (image, audio, …)"),
        model_id: str = Form("_", description="Model ID (multi-model mode). Omit to auto-route."),
        task: str | None = Form(
            None, description="Task hint for routing, e.g. 'image-classification'"
        ),
        text: str | None = Form(None, description="Text input for multimodal tasks"),
        inputs: str = Form(
            "{}",
            description=(
                "JSON object of additional named inputs beyond file/text. "
                'E.g. {"candidate_labels": ["cat","dog"]} for zero-shot tasks. '
                "Binary values must be base64-encoded."
            ),
        ),
        params: str = Form(
            "{}",
            description='JSON pipeline parameters (e.g. {"top_k": 5, "threshold": 0.5})',
        ),
    ) -> PredictionResult:
        mgr = _get_mgr()
        data = await file.read()
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 20 MB)")
        try:
            extra_inputs = json.loads(inputs)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid inputs JSON: {exc}") from exc
        try:
            pipe_params = json.loads(params)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid params JSON: {exc}") from exc
        try:
            async with mgr.borrow(model_id, task=task) as engine:
                loop = asyncio.get_running_loop()
                base_inputs = _build_file_inputs(data, text, engine.user_input_schema)

                # Decode base64 for any binary-typed extra inputs
                extra = _decode_rest_inputs(extra_inputs, engine.user_input_schema)

                # Collision: same key in file/text auto-mapped AND explicit inputs
                collision = set(base_inputs.keys()) & set(extra.keys())
                if collision:
                    key = sorted(collision)[0]
                    raise ValueError(
                        f"'{key}' provided by both file/text upload and inputs field. "
                        "Remove the duplicate."
                    )

                merged = {**base_inputs, **extra}

                # Collision: inputs vs params
                collision = set(merged.keys()) & set(pipe_params.keys())
                if collision:
                    key = sorted(collision)[0]
                    raise ValueError(
                        f"'{key}' specified in both inputs and params. "
                        "Use inputs for model inputs and params for pipeline parameters."
                    )

                task_override = task
                return await loop.run_in_executor(
                    None,
                    lambda: engine.predict(inputs=merged, task=task_override, **pipe_params),
                )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ------------------------------------------------------------------
    # POST /v1/predict — JSON (named inputs)
    # ------------------------------------------------------------------
    @app.post(
        "/v1/predict",
        response_model=PredictionResult,
        tags=["inference"],
        summary="JSON inference — named inputs with optional pipeline params",
    )
    async def predict_json(
        request: PredictJsonRequest,
        model_id: str = "_",
    ) -> PredictionResult:
        mgr = _get_mgr()
        try:
            async with mgr.borrow(model_id, task=request.task) as engine:
                loop = asyncio.get_running_loop()
                task_override = request.task

                # Decode base64 for binary inputs based on schema
                inputs = _decode_rest_inputs(request.inputs, engine.user_input_schema)

                # Check inputs/params collision
                collision = set(inputs.keys()) & set(request.params.keys())
                if collision:
                    key = sorted(collision)[0]
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"'{key}' specified in both inputs and params. "
                            "Use inputs for model inputs and params for pipeline parameters."
                        ),
                    )

                pipe_params = dict(request.params)
                return await loop.run_in_executor(
                    None,
                    lambda: engine.predict(inputs=inputs, task=task_override, **pipe_params),
                )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ------------------------------------------------------------------
    # GET /v1/tools — OpenAI tool definitions
    # ------------------------------------------------------------------
    @app.get(
        "/v1/tools",
        response_model=ToolsResponse,
        tags=["inference"],
        summary="OpenAI-compatible tool definitions for the loaded model",
    )
    async def get_tools() -> ToolsResponse:
        """Return OpenAI function-calling tool definitions for all loaded models."""
        manifests = _load_all_manifests()
        if not manifests:
            raise HTTPException(
                status_code=503,
                detail="No model loaded. Please load a model first.",
            )

        try:
            tools = []
            for manifest in manifests:
                tools.extend(APISchemaGenerator(manifest).generate_tools_list())
            logger.info("Generated %d tools from %d model(s)", len(tools), len(manifests))
            return ToolsResponse(tools=tools)
        except (KeyError, ValueError, TypeError) as e:
            logger.error("Error generating tools: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Error generating tools: {e!s}",
            ) from e

    # ------------------------------------------------------------------
    # GET /v1/mcp-schema — MCP tool definitions
    # ------------------------------------------------------------------
    @app.get(
        "/v1/mcp-schema",
        tags=["inference"],
        summary="MCP-compatible tool definitions for Claude integration",
    )
    async def get_mcp_schema() -> dict[str, Any]:
        """Return MCP-compatible tool definitions for all loaded models."""
        manifests = _load_all_manifests()
        if not manifests:
            raise HTTPException(
                status_code=503,
                detail="No model loaded. Please load a model first.",
            )

        try:
            mcp_tools = []
            model_ids = []
            tasks = []
            for manifest in manifests:
                openai_tools = APISchemaGenerator(manifest).generate_tools_list()
                for tool in openai_tools:
                    fn = tool.get("function", {})
                    mcp_tools.append(
                        {
                            "name": fn.get("name", "unknown"),
                            "description": fn.get("description", ""),
                            "inputSchema": fn.get("parameters", {}),
                        }
                    )
                model_ids.append(manifest.get("model_id", "unknown"))
                tasks.append(manifest.get("task", "unknown"))

            logger.info("Generated %d MCP tools from %d model(s)", len(mcp_tools), len(manifests))
            return {
                "tools": mcp_tools,
                "server_info": {
                    "name": "WinML CLI Inference",
                    "version": __version__,
                    "models": [
                        {"model_id": mid, "task": t}
                        for mid, t in zip(model_ids, tasks, strict=True)
                    ],
                },
            }
        except (KeyError, ValueError, TypeError) as e:
            logger.error("Error generating MCP schema: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Error generating MCP schema: {e!s}",
            ) from e

    # ------------------------------------------------------------------
    # POST /v1/ep — switch EP
    # ------------------------------------------------------------------
    @app.post("/v1/ep", tags=["management"], summary="Switch execution provider")
    async def switch_ep(request: EpSwitchRequest) -> dict[str, Any]:
        # Pydantic already validates ep against the EPAlias Literal (rejects
        # unknown values with a 422 at parse time), so no extra check needed.
        ep = request.ep
        mgr = _get_mgr()
        if not isinstance(mgr, SingleModelManager):
            raise HTTPException(
                status_code=400,
                detail="EP switching is only supported in single-model mode",
            )
        await mgr.switch_ep(ep)
        return {"status": "ok", "ep": ep}

    # ------------------------------------------------------------------
    # GET /v1/resources
    # ------------------------------------------------------------------
    @app.get("/v1/resources", response_model=ResourceResponse, tags=["management"])
    async def resources() -> ResourceResponse:
        mgr = _get_mgr()
        models = await mgr.list_models()
        first = models[0] if models else {}
        engine = mgr.get_engine()
        last_at: str | None = None
        if engine and engine.last_request_at:
            last_at = engine.last_request_at.isoformat()
        return ResourceResponse(
            model_id=first.get("model_id"),
            task=first.get("task"),
            device=first.get("device"),
            ep=first.get("ep"),
            status=first.get("status", "unloaded"),
            memory_mb=round(first.get("memory_mb", 0.0), 1),
            uptime_sec=round(time.time() - _get_start_time(), 1),
            request_count=first.get("request_count", 0),
            last_request_at=last_at,
        )

    # ------------------------------------------------------------------
    # GET /v1/models
    # ------------------------------------------------------------------
    @app.get(
        "/v1/models",
        response_model=list[ModelInfo],
        tags=["management"],
        summary="List all loaded models",
    )
    async def list_models() -> list[ModelInfo]:
        mgr = _get_mgr()
        models = await mgr.list_models()
        return [ModelInfo(**m) for m in models]

    # ------------------------------------------------------------------
    # POST /v1/models — load a new model (multi-model)
    # ------------------------------------------------------------------
    @app.post(
        "/v1/models",
        tags=["management"],
        summary="Load a model into the slot manager",
    )
    async def load_model(request: ModelLoadRequest) -> dict[str, Any]:
        mgr = _get_mgr()
        if not isinstance(mgr, ModelSlotManager):
            raise HTTPException(
                status_code=400,
                detail="Model loading is only supported in multi-model mode (--multi)",
            )
        try:
            await mgr.load_model(
                request.model_id,
                task=request.task,
                device=request.device if request.device != "auto" else None,
                ep=request.ep,
                alias=request.alias,
                description=request.description,
            )
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"status": "ok", "model_id": request.model_id}

    # ------------------------------------------------------------------
    # DELETE /v1/models/{model_id} — unload a model (multi-model)
    # ------------------------------------------------------------------
    @app.delete(
        "/v1/models/{model_id:path}",
        tags=["management"],
        summary="Unload a model from the slot manager",
    )
    async def unload_model(model_id: str) -> dict[str, Any]:
        mgr = _get_mgr()
        if not isinstance(mgr, ModelSlotManager):
            raise HTTPException(
                status_code=400,
                detail="Model unloading is only supported in multi-model mode (--multi)",
            )
        try:
            await mgr.unload_model(model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded") from None
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"status": "ok", "model_id": model_id}

    # ------------------------------------------------------------------
    # GET /v1/models/{model_id}/stats — live perf stats
    # ------------------------------------------------------------------
    @app.get(
        "/v1/models/{model_id:path}/stats",
        response_model=ModelStatsResponse,
        tags=["management"],
        summary="Live latency stats for a loaded model",
    )
    async def model_stats(model_id: str) -> ModelStatsResponse:
        mgr = _get_mgr()
        try:
            engine, status = await mgr.get_model_stats(model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded") from None

        last_at = engine.last_request_at.isoformat() if engine.last_request_at else None
        return ModelStatsResponse(
            model_id=model_id,
            status=status,
            request_count=engine.request_count,
            memory_mb=round(engine.memory_mb, 1),
            latency=LatencyStats(**engine.latency_stats),
            last_request_at=last_at,
        )

    # ------------------------------------------------------------------
    # GET /v1/models/{model_id}/schema — request/response schema for a model
    # ------------------------------------------------------------------
    @app.get(
        "/v1/models/{model_id:path}/schema",
        tags=["discovery"],
        summary="Request/response schema and examples for a loaded model",
    )
    async def model_schema(model_id: str, task: str | None = None) -> dict[str, Any]:
        """Return the request/response schema for a specific model.

        User inputs from TASK_REGISTRY, pipeline parameters discovered
        from the loaded pipeline's _sanitize_parameters signature.
        Optional ``task`` query param overrides the model's default task
        for input schema resolution (e.g. sentence-similarity vs feature-extraction).
        """
        manifest = _load_manifest(model_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")
        engine = _get_mgr().get_engine(model_id)
        return _build_model_schema(manifest, engine, task_override=task)

    # ------------------------------------------------------------------
    # GET /v1/schema — request/response schema (single-model shortcut)
    # ------------------------------------------------------------------
    @app.get(
        "/v1/schema",
        tags=["discovery"],
        summary="Request/response schema for the current model",
    )
    async def current_schema(task: str | None = None) -> dict[str, Any]:
        """Shortcut: returns schema for the first/only loaded model."""
        manifest = _load_manifest()
        if manifest is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        engine = _get_mgr().get_engine()
        return _build_model_schema(manifest, engine, task_override=task)

    # ------------------------------------------------------------------
    # GET /v1/hub — built-in catalog + user cached models
    # ------------------------------------------------------------------
    @app.get("/v1/hub", tags=["management"], summary="Model catalog (built-in + user cache)")
    async def hub_catalog() -> dict[str, Any]:
        from ..cache.model import list_cached_models
        from ..loader.task import TASK_ABBREV

        # Built-in models
        pkg = importlib.resources.files("winml.modelkit.data")
        catalog = json.loads((pkg / "hub_models.json").read_text(encoding="utf-8"))
        builtin_models = catalog.get("models", [])
        builtin_ids = {m["model_id"] for m in builtin_models}

        # Scan user cache
        abbrev_to_task = {v: k for k, v in TASK_ABBREV.items()}
        cached = list_cached_models()
        cached_slugs: set[str] = set()
        for entry in cached:
            cached_slugs.add(entry["model_slug"])

        # Tag built-in models: "built-in" or "built-in+cached"
        for m in builtin_models:
            slug = m["model_id"].replace("/", "_")
            if slug in cached_slugs:
                m["source"] = "built-in+cached"
            else:
                m["source"] = "built-in"

        # Add user-cache-only models (not in built-in list)
        seen: set[str] = set()
        cache_only: list[dict[str, Any]] = []
        for entry in cached:
            slug = entry["model_slug"]
            if slug in seen:
                continue
            seen.add(slug)
            model_id = slug.replace("_", "/", 1)
            if model_id in builtin_ids:
                continue
            task = abbrev_to_task.get(entry["task_abbrev"], entry["task_abbrev"])
            cache_only.append(
                {
                    "model_id": model_id,
                    "task": task,
                    "model_type": "",
                    "source": "user-cache",
                    "cache_path": str(Path(entry["path"]).parent),
                }
            )

        return {
            "version": catalog.get("version", "unknown"),
            "models": builtin_models + cache_only,
        }

    # ------------------------------------------------------------------
    # GET /v1/logs — ring buffer polling for live log lines
    # ------------------------------------------------------------------
    @app.get("/v1/logs", tags=["management"], summary="Poll recent modelkit log lines")
    async def get_logs(after: int = 0) -> dict[str, Any]:
        """Return log lines with seq > after.  Poll every ~500ms to get live output."""
        return {"lines": _log_handler.since(after), "latest_seq": _log_handler._seq}

    # ------------------------------------------------------------------
    # POST /v1/cli/{command} — CLI wrapper (available in all modes)
    # ------------------------------------------------------------------
    @app.post(
        "/v1/cli/{command}",
        response_model=CliResponse,
        tags=["cli"],
        summary="Run any winml CLI command",
    )
    async def cli_command(command: str, request: CliRequest) -> CliResponse:
        """Proxy to the CLI wrapper — available in all server modes."""
        return await _run_with_semaphore(command, request.args)


# ---------------------------------------------------------------------------
# Manifest helpers (module-level — no manager dependency)
# ---------------------------------------------------------------------------


def _manifest_from_engine(engine: InferenceEngine) -> dict:
    """Build manifest dict from engine, trying build_manifest.json first."""
    if engine.model_path:
        manifest_file = Path(engine.model_path) / "build_manifest.json"
        if manifest_file.exists():
            try:
                return json.loads(manifest_file.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load manifest: %s", e)

    return {
        "model_id": engine.model_id or "unknown",
        "task": engine.task or "unknown",
        "parameters": {},
        "model_io": {},
        "processing": {},
    }


def _build_model_schema(
    manifest: dict,
    engine: InferenceEngine | None = None,
    task_override: str | None = None,
) -> dict[str, Any]:
    """Build request/response schema from TASK_REGISTRY + engine.

    User inputs come from TASK_REGISTRY[task]. Pipeline parameters come from
    the engine's _discover_pipeline_params (if available).
    When *task_override* is provided, user_inputs are resolved from that task
    instead (e.g. ``sentence-similarity`` for a ``feature-extraction`` model).
    """
    from ..inference.tasks import TASK_REGISTRY

    task = task_override or manifest.get("task", "")
    model_id = manifest.get("model_id", "unknown")

    # User inputs from registry
    user_inputs: list[dict[str, Any]] = []
    spec = TASK_REGISTRY.get(task)
    if spec:
        for f in spec.user_inputs:
            entry: dict[str, Any] = {
                "name": f.name,
                "type": f.type,
                "required": f.required,
                "description": f.description,
            }
            if not f.required and f.default is not None:
                entry["default"] = f.default
            user_inputs.append(entry)

    # Pipeline parameters from engine
    parameters: list[dict] = []
    if engine is not None and engine.pipeline_params:
        parameters = engine.pipeline_params

    return {
        "model_id": model_id,
        "task": task,
        "user_inputs": user_inputs,
        "parameters": parameters,
        "endpoints": {
            "predict": "/v1/predict",
            "predict_file": "/v1/predict/file",
            "schema": "/v1/schema",
            "tools": "/v1/tools",
        },
    }


# ---------------------------------------------------------------------------
# Input helpers for REST endpoints
# ---------------------------------------------------------------------------


def _decode_rest_inputs(
    inputs: dict[str, Any],
    schema: list | None,
) -> dict[str, Any]:
    """Decode base64 strings for binary-typed inputs based on schema.

    Raises ValueError on invalid base64.
    """
    if not schema:
        return inputs

    from ..inference.tasks import BINARY_TYPES

    schema_map = {f.name: f for f in schema}
    result = dict(inputs)
    for name, value in result.items():
        field = schema_map.get(name)
        if field and field.type in BINARY_TYPES and isinstance(value, str):
            try:
                result[name] = base64.b64decode(value)
            except (ValueError, base64.binascii.Error) as exc:
                raise ValueError(f"Invalid base64 for input '{name}': {exc}") from exc
    return result


def _build_file_inputs(
    data: bytes,
    text: str | None,
    schema: list | None,
) -> dict[str, Any]:
    """Build inputs dict from a file upload + optional text.

    Maps the uploaded file to the sole binary-typed input in the schema.
    Raises ValueError on ambiguity.
    """
    from ..inference.tasks import BINARY_TYPES

    if schema:
        binary_fields = [f for f in schema if f.type in BINARY_TYPES]
        if len(binary_fields) != 1:
            raise ValueError(
                f"File upload requires exactly one binary input in schema. "
                f"Found {len(binary_fields)}. Use POST /v1/predict with JSON instead."
            )
        inputs: dict[str, Any] = {binary_fields[0].name: data}
        if text is not None:
            text_fields = [f for f in schema if f.type == "text"]
            if len(text_fields) == 1:
                inputs[text_fields[0].name] = text
            elif len(text_fields) > 1:
                raise ValueError(
                    "Ambiguous: multiple text inputs in schema. "
                    "Use POST /v1/predict with JSON to specify named inputs."
                )
    else:
        # No schema — use generic names
        inputs = {"file": data}
        if text is not None:
            inputs["text"] = text

    return inputs


def print_startup_banner(
    *,
    host: str,
    port: int,
    model_path: str,
    task: str | None,
    device: str,
    ep: EPNameOrAlias | None,
) -> None:
    """Print Phase 1+ startup banner to stdout."""
    from rich.console import Console

    console = Console()
    console.print()
    console.print("[bold]WinML CLI Inference Server[/bold]")
    console.print(f"Model:   {model_path or '(none — load via POST /v1/models)'}")
    if task:
        console.print(f"Task:    {task}")
    console.print(f"Device:  {device}")
    if ep:
        console.print(f"EP:      {ep}")
    console.print()
    console.print(f"API:     http://{host}:{port}")
    console.print(f"Docs:    http://{host}:{port}/docs")
    console.print(f"Demo:    http://{host}:{port}/demo")
    console.print()
    console.print("Ready. Press [bold]Ctrl+C[/bold] to stop.")
    console.print()
