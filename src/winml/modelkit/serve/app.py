# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Phase 1+ FastAPI inference application.

Activated when ``wmk serve <model_path>`` is given a model argument.

Endpoints:
  GET  /v1/health             — liveness + model info
  POST /v1/predict            — image file upload OR JSON tensor inputs
  POST /v1/ep                 — switch execution provider (Phase 1, P0)
  GET  /v1/resources          — runtime memory + request stats (Phase 2)
  GET  /v1/models             — list all loaded models (Phase 3)
  GET  /v1/hub                — curated WinML Hub model catalog

EP shorthand mapping (cpu / dml / qnn / openvino / cuda) is handled here
in the serve layer, not inside InferenceEngine or WinMLSession.
"""

from __future__ import annotations

import asyncio
import importlib.resources
import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from .cli_api import CliRequest, CliResponse, _run_with_semaphore
from .engine import InferenceEngine
from .manager import ModelSlotManager, SingleModelManager
from .schema import (
    EpSwitchRequest,
    HealthResponse,
    LatencyStats,
    ModelInfo,
    ModelLoadRequest,
    ModelStatsResponse,
    PredictionResult,
    PredictJsonRequest,
    ResourceResponse,
)


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
                "name": record.name.removeprefix("modelkit."),
                "msg": self.format(record),
            }
        )

    def since(self, after_seq: int) -> list[dict]:
        return [e for e in self._buf if e["seq"] > after_seq]


_log_handler = _RingHandler()
_log_handler.setFormatter(logging.Formatter("%(message)s"))
# Attach to modelkit root logger so all sub-loggers feed into the ring
logging.getLogger("modelkit").addHandler(_log_handler)
logging.getLogger("modelkit").setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Valid EP shorthands accepted by POST /v1/ep
# ---------------------------------------------------------------------------
_VALID_EPS = {"cpu", "dml", "qnn", "openvino", "cuda", "auto"}

# ---------------------------------------------------------------------------
# Global manager — set by create_app() before uvicorn starts
# ---------------------------------------------------------------------------
_manager: SingleModelManager | ModelSlotManager | None = None
_start_time = time.time()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    *,
    model_path: str | None,
    task: str | None = None,
    device: str = "auto",
    ep: str | None = None,
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
    global _manager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _manager
        if mode == "multi":
            mgr = ModelSlotManager(
                memory_budget_mb=memory_budget_mb,
                idle_timeout_sec=idle_timeout_sec,
                default_device=device,
            )
            if model_path:
                # Pre-load the initial model if given
                logger.info("Pre-loading model: %s", model_path)
                async with mgr.borrow(model_path):
                    pass
            else:
                logger.info("Multi-model server started (empty — load via POST /v1/models)")
            _manager = mgr
        else:
            engine = InferenceEngine()
            engine.load(model_path, task=task, device=device, ep=ep)
            _manager = SingleModelManager(engine, idle_timeout_sec=idle_timeout_sec)
        logger.info("Model ready")
        yield
        # Shutdown: unload
        if isinstance(_manager, SingleModelManager):
            _manager._engine.unload()

    app = FastAPI(
        title="ModelKit Inference Server",
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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
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
    # GET /v1/health
    # ------------------------------------------------------------------
    @app.get("/v1/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        mgr = _get_manager()
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
            uptime_sec=round(time.time() - _start_time, 1),
        )

    # ------------------------------------------------------------------
    # POST /v1/predict — file upload
    # ------------------------------------------------------------------
    @app.post(
        "/v1/predict/file",
        response_model=PredictionResult,
        tags=["inference"],
        summary="Image file upload inference",
    )
    async def predict_file(
        file: UploadFile = File(..., description="Image file (JPEG, PNG, …)"),
        top_k: int = Form(5, ge=1, le=100),
        model_id: str = Form("_", description="Model ID (multi-model mode). Omit to auto-route."),
        task: str | None = Form(
            None, description="Task hint for routing, e.g. 'image-classification'"
        ),
    ) -> PredictionResult:
        mgr = _get_manager()
        data = await file.read()
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 20 MB)")
        try:
            async with mgr.borrow(model_id, task=task) as engine:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, lambda: engine.predict(image_bytes=data, top_k=top_k)
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ------------------------------------------------------------------
    # POST /v1/predict — JSON tensors
    # ------------------------------------------------------------------
    @app.post(
        "/v1/predict",
        response_model=PredictionResult,
        tags=["inference"],
        summary="JSON tensor inputs inference",
    )
    async def predict_json(
        request: PredictJsonRequest,
        model_id: str = "_",
    ) -> PredictionResult:
        mgr = _get_manager()
        try:
            async with mgr.borrow(model_id, task=request.task) as engine:
                loop = asyncio.get_event_loop()
                if request.text is not None:
                    return await loop.run_in_executor(
                        None, lambda: engine.predict(text=request.text, top_k=request.top_k)
                    )
                if request.inputs is None:
                    raise HTTPException(status_code=422, detail="Provide 'text' or 'inputs'")
                return await loop.run_in_executor(
                    None,
                    lambda: engine.predict(tensor_inputs=request.inputs, top_k=request.top_k),
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ------------------------------------------------------------------
    # POST /v1/ep — switch EP (Phase 1, P0 priority)
    # ------------------------------------------------------------------
    @app.post("/v1/ep", tags=["management"], summary="Switch execution provider")
    async def switch_ep(request: EpSwitchRequest) -> dict[str, Any]:
        ep = request.ep.lower()
        if ep not in _VALID_EPS:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown EP '{ep}'. Valid: {sorted(_VALID_EPS)}",
            )
        mgr = _get_manager()
        if not isinstance(mgr, SingleModelManager):
            raise HTTPException(
                status_code=400,
                detail="EP switching is only supported in single-model mode",
            )
        async with mgr._lock:
            mgr._engine.switch_ep(ep)
        return {"status": "ok", "ep": ep}

    # ------------------------------------------------------------------
    # GET /v1/resources — Phase 2
    # ------------------------------------------------------------------
    @app.get("/v1/resources", response_model=ResourceResponse, tags=["management"])
    async def resources() -> ResourceResponse:
        mgr = _get_manager()
        models = await mgr.list_models()
        first = models[0] if models else {}
        engine: InferenceEngine | None = None
        if isinstance(mgr, SingleModelManager):
            engine = mgr._engine
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
            uptime_sec=round(time.time() - _start_time, 1),
            request_count=first.get("request_count", 0),
            last_request_at=last_at,
        )

    # ------------------------------------------------------------------
    # GET /v1/models — Phase 3
    # ------------------------------------------------------------------
    @app.get(
        "/v1/models",
        response_model=list[ModelInfo],
        tags=["management"],
        summary="List all loaded models (Phase 3)",
    )
    async def list_models() -> list[ModelInfo]:
        mgr = _get_manager()
        models = await mgr.list_models()
        return [ModelInfo(**m) for m in models]

    # ------------------------------------------------------------------
    # POST /v1/models — load a new model (Phase 3 multi-model)
    # ------------------------------------------------------------------
    @app.post(
        "/v1/models",
        tags=["management"],
        summary="Load a model into the slot manager (Phase 3)",
    )
    async def load_model(request: ModelLoadRequest) -> dict[str, Any]:
        mgr = _get_manager()
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
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"status": "ok", "model_id": request.model_id}

    # ------------------------------------------------------------------
    # DELETE /v1/models/{model_id} — unload a model (Phase 3)
    # ------------------------------------------------------------------
    @app.delete(
        "/v1/models/{model_id:path}",
        tags=["management"],
        summary="Unload a model from the slot manager (Phase 3)",
    )
    async def unload_model(model_id: str) -> dict[str, Any]:
        mgr = _get_manager()
        if not isinstance(mgr, ModelSlotManager):
            raise HTTPException(
                status_code=400,
                detail="Model unloading is only supported in multi-model mode (--multi)",
            )
        async with mgr._lock:
            slot = mgr._slots.get(model_id)
            if slot is None:
                raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")
            if slot.refcount > 0:
                raise HTTPException(
                    status_code=409,
                    detail=f"Model '{model_id}' is in use (refcount={slot.refcount})",
                )
            slot.engine.unload()
            del mgr._slots[model_id]
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
        mgr = _get_manager()
        engine: InferenceEngine | None = None
        status = "unknown"

        if isinstance(mgr, SingleModelManager):
            engine = mgr._engine
            status = "ready" if engine.is_loaded else "unloaded"
        elif isinstance(mgr, ModelSlotManager):
            async with mgr._lock:
                slot = mgr._slots.get(model_id)
            if slot is None:
                raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded")
            engine = slot.engine
            status = "ready" if engine.is_loaded else "unloaded"

        if engine is None:
            raise HTTPException(status_code=404, detail="Model not found")

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
    # GET /v1/hub — serve curated model catalog from hub_models.json
    # ------------------------------------------------------------------
    @app.get("/v1/hub", tags=["management"], summary="Curated WinML Hub model catalog")
    async def hub_catalog() -> dict[str, Any]:
        pkg = importlib.resources.files("winml.modelkit.data")
        data = (pkg / "hub_models.json").read_text(encoding="utf-8")
        return json.loads(data)

    # ------------------------------------------------------------------
    # GET /v1/logs — ring buffer polling for live log lines
    # ------------------------------------------------------------------
    @app.get("/v1/logs", tags=["management"], summary="Poll recent modelkit log lines")
    async def get_logs(after: int = 0) -> dict[str, Any]:
        """Return log lines with seq > after.  Poll every ~500ms to get live output."""
        return {"lines": _log_handler.since(after), "latest_seq": _log_handler._seq}

    # ------------------------------------------------------------------
    # POST /v1/cli/{command} — Phase 0 CLI wrapper (available in all modes)
    # ------------------------------------------------------------------
    @app.post(
        "/v1/cli/{command}",
        response_model=CliResponse,
        tags=["cli"],
        summary="Run any wmk CLI command (Phase 0 compatibility)",
    )
    async def cli_command(command: str, request: CliRequest) -> CliResponse:
        """Proxy to the Phase 0 CLI wrapper — available in all server modes."""
        return await _run_with_semaphore(command, request.args)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_manager() -> SingleModelManager | ModelSlotManager:
    if _manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return _manager


def print_startup_banner(
    *,
    host: str,
    port: int,
    model_path: str,
    task: str | None,
    device: str,
    ep: str | None,
) -> None:
    """Print Phase 1+ startup banner to stdout."""
    from rich.console import Console

    console = Console()
    console.print()
    console.print("[bold]ModelKit Inference Server[/bold]")
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
