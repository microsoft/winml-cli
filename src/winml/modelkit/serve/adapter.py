# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unified model inference adapters.

Design:
1. A single Adapter handles all inference scenarios
2. The inference engine is driven by manifest.engine.format (ONNX, GenAI, …)
3. Pre/post-processing is driven by TaskHandler (image, text, tensor)
4. LLM tasks use LLMAdapter which adds streaming support

Flow:
  Handler.preprocess() → Engine.infer() → Handler.postprocess()
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from .engine_factory import create_inference_engine
from .handlers import resolve_handler
from .schema import PredictionResult


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


class ModelAdapter(ABC):
    """Unified model inference interface."""

    @abstractmethod
    def predict(self, inputs: dict) -> PredictionResult:
        """Synchronous inference."""

    async def predict_streaming(self, inputs: dict) -> AsyncGenerator[dict, None]:
        """Streaming inference (default: yields the full result at once).

        Non-LLM models use this default implementation.
        LLM models override this for real token-by-token streaming.
        """
        result = self.predict(inputs)
        yield result


class UnifiedAdapter(ModelAdapter):
    """General-purpose inference adapter for any engine and task type.

    Handles all inference models (classification, detection, segmentation,
    text-to-text, etc.) with pluggable engines and task handlers.

    Flow:
      inputs → Handler.preprocess() → Engine.infer() → Handler.postprocess() → predictions

    Args:
        manifest: Build-generated model metadata
        model_path: Directory containing model files
        device: Target device (auto/cpu/gpu/npu)
    """

    def __init__(self, manifest: dict, model_path: Path, device: str = "auto"):
        self.manifest = manifest
        self.model_path = Path(model_path)
        self.device = device

        # Create inference engine based on manifest.engine.format
        self.engine = create_inference_engine(manifest, self.model_path, device)

        # Resolve task handler for pre/post-processing
        task = manifest.get("task", "image-classification")
        self.handler = resolve_handler(task, processor=None, model=None)

    def predict(self, inputs: dict) -> PredictionResult:
        """Synchronous inference: preprocess → infer → postprocess.

        Args:
            inputs: May contain image_bytes, text, tensor_inputs, top_k, etc.

        Returns:
            PredictionResult with predictions and latency
        """
        latency_start = time.time()

        # 1. Preprocess (handler supports image_bytes, text, or tensor_inputs)
        model_inputs = self.handler.preprocess(
            image_bytes=inputs.get("image_bytes"),
            text=inputs.get("text"),
            tensor_inputs=inputs.get("tensor_inputs"),
        )

        # 2. Infer (engine returns raw output arrays or dict)
        raw_output = self.engine.infer(model_inputs)

        # 3. Postprocess (handler converts raw output to Prediction objects)
        task_params = self._extract_task_params(inputs)
        predictions = self.handler.postprocess(raw_output, **task_params)

        latency_ms = (time.time() - latency_start) * 1000

        return PredictionResult(predictions=predictions, latency_ms=latency_ms)

    def _extract_task_params(self, inputs: dict) -> dict:
        """Extract task-specific params from inputs and manifest defaults."""
        task_params = {}
        if "top_k" in inputs:
            task_params["top_k"] = inputs["top_k"]
        else:
            manifest_params = self.manifest.get("parameters", {})
            task_params["top_k"] = manifest_params.get("top_k", 5)
        return task_params


class LLMAdapter(UnifiedAdapter):
    """Adapter for text-generation / LLM models with streaming support.

    Differences from UnifiedAdapter:
    1. Validates that the engine format is onnxruntime_genai
    2. predict() waits for full generation (for tool calling)
    3. predict_streaming() yields tokens one by one (for SSE)

    Usage:
    - Tool calling (Claude, MCP) → predict() [synchronous]
    - Web frontend → predict_streaming() [SSE]
    """

    def __init__(self, manifest: dict, model_path: Path, device: str = "auto"):
        super().__init__(manifest, model_path, device)

        engine_format = manifest.get("engine", {}).get("format")
        if engine_format != "onnxruntime_genai":
            raise ValueError(
                f"LLMAdapter requires engine.format='onnxruntime_genai', "
                f"got '{engine_format}'. Use UnifiedAdapter for other engines."
            )

    def predict(self, inputs: dict) -> PredictionResult:
        """Synchronous full generation — waits for complete output.

        Args:
            inputs: Must contain 'text' or 'prompt'

        Returns:
            PredictionResult with complete generated text
        """
        latency_start = time.time()

        prompt = inputs.get("text") or inputs.get("prompt")
        if not prompt:
            raise ValueError("LLM requires 'text' or 'prompt' input")

        generation_params = self._extract_generation_params(inputs)
        text = self.engine.infer(prompt, **generation_params)

        latency_ms = (time.time() - latency_start) * 1000
        return PredictionResult(predictions={"text": text}, latency_ms=latency_ms)

    async def predict_streaming(self, inputs: dict) -> AsyncGenerator[dict, None]:
        """Token-by-token streaming generation for SSE.

        Args:
            inputs: Must contain 'text' or 'prompt'

        Yields:
            {"token": "word", "latency_ms": 123.45}
        """
        latency_start = time.time()

        prompt = inputs.get("text") or inputs.get("prompt")
        if not prompt:
            raise ValueError("LLM requires 'text' or 'prompt' input")

        generation_params = self._extract_generation_params(inputs)

        async for token in self.engine.infer_streaming(prompt, **generation_params):
            latency_ms = (time.time() - latency_start) * 1000
            yield {"token": token, "latency_ms": latency_ms}

    def _extract_generation_params(self, inputs: dict) -> dict:
        """Extract generation params from inputs with manifest defaults."""
        manifest_params = self.manifest.get("parameters", {})
        return {
            "max_tokens": inputs.get("max_tokens", manifest_params.get("max_tokens", 100)),
            "temperature": inputs.get("temperature", manifest_params.get("temperature", 0.7)),
            "top_p": inputs.get("top_p", manifest_params.get("top_p", 0.9)),
            "top_k": inputs.get("top_k", manifest_params.get("top_k", 40)),
        }


class AdapterFactory:
    """Factory that creates the appropriate adapter based on the manifest.

    Routing:
    - task = "text-generation" → LLMAdapter (with streaming)
    - other tasks → UnifiedAdapter (standard inference)
    """

    @staticmethod
    def create(manifest: dict, model_path: Path, device: str = "auto") -> ModelAdapter:
        """Create an adapter from the model manifest.

        Args:
            manifest: Model metadata (task, engine info, etc.)
            model_path: Directory containing model files
            device: Target device

        Returns:
            ModelAdapter instance (UnifiedAdapter or LLMAdapter)

        Raises:
            ValueError: If the task is not recognized
        """
        task = manifest.get("task")

        if task == "text-generation":
            return LLMAdapter(manifest, model_path, device)

        if task in [
            "image-classification",
            "object-detection",
            "image-segmentation",
            "text-classification",
            "sentiment-analysis",
            "token-classification",
            "image-to-text",
        ]:
            return UnifiedAdapter(manifest, model_path, device)

        raise ValueError(
            f"Unknown task '{task}'. Supported tasks: "
            f"image-classification, object-detection, text-generation, etc."
        )
