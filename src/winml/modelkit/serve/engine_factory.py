# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Inference engine factory — creates the right engine from a build manifest.

Design:
1. Multiple engine backends (ONNX Runtime, GenAI, future others)
2. Engine selection driven by manifest.engine.format
3. Each engine owns its own initialization and inference logic
4. Unified interface (infer, infer_streaming)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from ..models.winml.base import WinMLPreTrainedModel


class InferenceEngineBase(ABC):
    """Abstract interface for inference engines."""

    @abstractmethod
    def infer(self, inputs: dict[str, Any]) -> Any:
        """Synchronous inference.

        Args:
            inputs: Pre-processed inputs (typically numpy arrays or torch tensors)

        Returns:
            Raw model output (list, dict, or other structure)
        """

    async def infer_streaming(self, *args, **kwargs) -> AsyncGenerator:
        """Streaming inference (optional, only some engines support this).

        Only LLM-related engines need to implement this method.

        Raises:
            NotImplementedError: If the engine does not support streaming
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support streaming inference")


class WinMLRuntimeEngine(InferenceEngineBase):
    """ONNX Runtime engine via WinMLSession.

    Handles standard ONNX models (classification, detection, segmentation, etc.).
    Executes inference through WinMLPreTrainedModel / WinMLSession with
    automatic execution provider selection (DML, QNN, CPU).

    Args:
        model: Initialized WinMLPreTrainedModel instance
    """

    def __init__(self, model: WinMLPreTrainedModel):
        self.model = model
        self.session = getattr(model, "_session", None)
        if self.session is None:
            raise ValueError("Model does not have a valid _session attribute")

    def infer(self, inputs: dict[str, Any]) -> Any:
        """Run inference through WinML.

        Args:
            inputs: Pre-processed inputs (numpy arrays or torch tensors).
                    Keys should match the model's input names.

        Returns:
            Model output (torch tensor or numpy array)
        """
        import torch

        tensor_inputs = {
            k: torch.from_numpy(v) if hasattr(v, "shape") else v for k, v in inputs.items()
        }

        return self.model(**tensor_inputs)


class GenAIEngine(InferenceEngineBase):
    """ONNX Runtime GenAI engine for LLM models.

    Provides LLM-optimized inference including:
    - Efficient token-by-token generation
    - Automatic KV cache management
    - Streaming generation support

    Args:
        model_path: Directory containing the GenAI model (genai_model/)
        device: Target device (gpu/cpu/npu)
    """

    def __init__(self, model_path: Path, device: str = "gpu"):
        from .genai_engine import GenAIEngineImpl

        self.model_path = Path(model_path)
        self.device = device

        self.genai = GenAIEngineImpl(model_path=self.model_path, device=device)

    def infer(self, prompt: str, **kwargs) -> str:
        """Synchronous full generation.

        Args:
            prompt: Input prompt
            **kwargs: Generation params (max_tokens, temperature, top_p, etc.)

        Returns:
            Complete generated text
        """
        return self.genai.generate(prompt, **kwargs)

    async def infer_streaming(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Streaming generation — yields one token at a time.

        Args:
            prompt: Input prompt
            **kwargs: Generation params (max_tokens, temperature, top_p, etc.)

        Yields:
            Each generated token
        """
        async for token in self.genai.generate_streaming(prompt, **kwargs):
            yield token


def create_inference_engine(
    manifest: dict,
    model_path: Path,
    device: str = "auto",
) -> InferenceEngineBase:
    """Create the appropriate inference engine from a build manifest.

    Args:
        manifest: Model metadata (contains engine.format and other config)
        model_path: Directory containing model files
        device: Target device (auto/cpu/gpu/npu)

    Returns:
        InferenceEngineBase instance

    Raises:
        ValueError: If the engine format is unsupported or config is invalid
    """
    engine_config = manifest.get("engine", {})
    engine_format = engine_config.get("format", "onnxruntime")

    if engine_format == "onnxruntime_genai":
        genai_model_path = engine_config.get("genai_model_path", "genai_model")
        return GenAIEngine(model_path=model_path / genai_model_path, device=device)

    if engine_format == "onnxruntime":
        from ..models.winml import get_winml_class

        task = manifest.get("task", "")
        onnx_path = model_path / "model.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(f"Model file not found: {onnx_path}")

        winml_class = get_winml_class(None, task)
        model = winml_class(onnx_path=onnx_path, config=None, device=device)

        return WinMLRuntimeEngine(model=model)

    raise ValueError(
        f"Unknown engine format '{engine_format}'. "
        f"Supported formats: onnxruntime, onnxruntime_genai"
    )
