# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Pydantic request/response models for wmk serve (Phase 1+).

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
    """POST /v1/predict — image (base64), text, or raw tensor inputs."""

    image_bytes: str | None = Field(None, description="Base64-encoded image data (JPEG, PNG, …)")
    inputs: dict[str, list[Any]] | None = Field(
        None, description="Map of input_name → nested list (numpy-serialisable)"
    )
    text: str | None = Field(None, description="Text input for NLP tasks")
    top_k: int = Field(5, ge=1, le=100, description="Top-K results for classification")
    task: str | None = Field(
        None,
        description=(
            "Task hint for model routing (multi-model mode). "
            "When model_id is omitted, the server picks the loaded model whose task matches. "
            "Example: 'image-classification', 'text-classification', 'object-detection'."
        ),
    )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class Prediction(BaseModel):
    """Single classification prediction."""

    label: str
    score: float


class PredictionResult(BaseModel):
    """POST /v1/predict response."""

    task: str
    model_id: str | None = None
    device: str
    ep: str | None = None
    predictions: list[Prediction] | dict[str, Any] = Field(
        ..., description="list[Prediction] for classification; raw dict for other tasks"
    )
    latency_ms: float


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


# ---------------------------------------------------------------------------
# OpenAI-compatible chat completions (LLM support)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """Single message in a chat conversation (OpenAI format)."""

    role: str = Field(..., description="Message role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message text content")


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions — OpenAI-compatible LLM request."""

    model: str = Field(..., description="Model ID or name")
    messages: list[ChatMessage] = Field(..., description="Conversation history (newest last)")
    max_tokens: int | None = Field(None, ge=1, description="Maximum tokens to generate")
    temperature: float | None = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float | None = Field(None, ge=0.0, le=1.0, description="Nucleus sampling (top-p)")
    stream: bool = Field(False, description="If true, stream tokens via SSE")


class ChatCompletionChoice(BaseModel):
    """One completion choice in a non-streaming response."""

    index: int
    message: ChatMessage
    finish_reason: str | None = None  # "stop", "length", "content_filter", etc.


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """POST /v1/chat/completions — full (non-streaming) response."""

    id: str = Field(..., description="Completion ID (chatcmpl-...)")
    object: str = Field(
        "chat.completion",
        description="Object type (always 'chat.completion' for non-streaming)",
    )
    created: int = Field(..., description="Unix timestamp of creation")
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage | None = None


class ChatCompletionChunkDelta(BaseModel):
    """Delta (incremental change) in a streaming chunk."""

    role: str | None = None  # Only in first chunk
    content: str | None = None  # Token text (or empty string)


class ChatCompletionChunkChoice(BaseModel):
    """One choice in a streaming chunk."""

    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None  # Only in final chunk


class ChatCompletionChunk(BaseModel):
    """SSE chunk for streaming chat completions."""

    id: str
    object: str = Field(
        "chat.completion.chunk",
        description="Object type (always 'chat.completion.chunk' for streaming)",
    )
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class ToolsResponse(BaseModel):
    """GET /v1/tools — OpenAI-compatible tool definitions response."""

    tools: list[dict] = Field(..., description="Array of OpenAI function-calling tool definitions")
