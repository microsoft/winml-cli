# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Core inference result types.

Lightweight Pydantic models shared by InferenceEngine and the REST layer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """Single classification prediction."""

    label: str
    score: float


class PredictionResult(BaseModel):
    """Structured inference result."""

    task: str
    model_id: str | None = None
    device: str
    ep: str | None = None
    predictions: list[Prediction] | dict[str, Any] = Field(
        ..., description="list[Prediction] for classification; raw dict for other tasks"
    )
    latency_ms: float
