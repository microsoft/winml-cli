# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Pydantic request/response models for winml serve (Phase 1+).

Phase 0 schemas are in cli_api.py.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class EpSwitchRequest(BaseModel):
    """POST /v1/ep — switch execution provider."""

    ep: str = Field(..., description="EP short name: cpu, dml, qnn, openvino, cuda")


class PredictJsonRequest(BaseModel):
    """POST /v1/predict — named inputs + pipeline parameters.

    Binary inputs (image, audio, video) are sent as raw base64 strings.
    The server decodes them based on the task's user_inputs schema.

    Example::

        {
            "inputs": {
                "question": "Who is the CEO?",
                "context": "Tim Cook is the CEO of Apple Inc."
            },
            "params": {"top_k": 5}
        }
    """

    inputs: dict[str, Any] = Field(
        ...,
        description=(
            "Named inputs: {name: value, ...}. "
            "Binary inputs (image/audio/video) as base64 strings. "
            "Text as strings. JSON as objects/arrays. Numbers and booleans as-is."
        ),
    )
    task: str | None = Field(
        None,
        description=(
            "Task hint for model routing (multi-model mode). "
            "When model_id is omitted, the server picks the loaded model whose task matches."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Pipeline parameters forwarded to inference (e.g. top_k, max_new_tokens, temperature)."
        ),
    )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """GET /v1/health response."""

    status: str = Field(..., description="ok | loading | unloaded")
    version: str
    mode: str = Field(..., description="cli-wrapper | single | multi")
    model_id: str | None = None
    task: str | None = None
    device: str | None = None
    ep: str | None = None
    uptime_sec: float


class ResourceResponse(BaseModel):
    """GET /v1/resources response (Phase 2)."""

    model_id: str | None = None
    task: str | None = None
    device: str | None = None
    ep: str | None = None
    status: str = Field(..., description="ready | loading | unloaded")
    memory_mb: float = 0.0
    uptime_sec: float
    request_count: int = 0
    last_request_at: str | None = None


class ModelInfo(BaseModel):
    """Entry in GET /v1/models response (Phase 3)."""

    model_id: str
    task: str | None = None
    device: str | None = None
    ep: str | None = None
    status: str
    refcount: int = 0
    memory_mb: float = 0.0
    request_count: int = 0
    last_used_at: str | None = None
    alias: str | None = Field(None, description="Short name for programmatic routing via task hint")
    description: str | None = Field(
        None, description="Human-readable capability description for LLM agent routing"
    )


class ModelLoadRequest(BaseModel):
    """POST /v1/models — load a new model into the slot manager."""

    model_id: str = Field(..., description="HF model ID, build output dir, or .onnx path")
    task: str | None = Field(None, description="Task (required for raw .onnx)")
    device: str = Field("auto", description="auto | cpu | gpu | npu")
    ep: str | None = Field(None, description="Explicit EP short name")
    alias: str | None = Field(
        None,
        description=(
            "Short routing name for programmatic agents. "
            "Pass alias as the 'task' hint in predict requests to target this model directly. "
            "Example: 'finbert-financial' to distinguish from another text-classification model."
        ),
    )
    description: str | None = Field(
        None,
        description=(
            "Human-readable description of what this model does. "
            "Returned in GET /v1/models so LLM agents can choose the right model by capability. "
            "Example: 'Financial sentiment — positive/negative/neutral for earnings reports.'"
        ),
    )


class LatencyStats(BaseModel):
    """Per-model live latency statistics (rolling last-200 requests)."""

    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    sample_count: int = 0


class ModelStatsResponse(BaseModel):
    """GET /v1/models/{model_id}/stats — live perf stats for one model."""

    model_id: str
    status: str
    request_count: int = 0
    memory_mb: float = 0.0
    latency: LatencyStats = Field(default_factory=LatencyStats)
    last_request_at: str | None = None


class ToolsResponse(BaseModel):
    """GET /v1/tools — OpenAI-compatible tool definitions response."""

    tools: list[dict] = Field(..., description="Array of OpenAI function-calling tool definitions")
