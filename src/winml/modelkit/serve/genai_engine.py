# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""GenAI engine wrapper — high-level interface for onnxruntime_genai.

Provides:
1. Session management (initialization and lifecycle)
2. Tokenizer management (encoding and decoding)
3. Synchronous generation (full output)
4. Streaming generation (token by token)
5. WinML execution provider support
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..winml import register_execution_providers


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


try:
    import onnxruntime_genai as ort_genai
except ImportError:
    # onnxruntime_genai is optional - only needed for LLM inference
    ort_genai = None


class GenAIEngineImpl:
    """Wrapper around ONNX Runtime GenAI.

    Responsibilities:
    1. Initialize GenAI Session and Tokenizer
    2. Manage generation parameters
    3. Provide generate() (sync) and generate_streaming() (async)

    Args:
        model_path: Directory containing the GenAI model
                    (decoder_model_merged.onnx + tokenizer)
        device: Target device (gpu/cpu/npu)
    """

    def __init__(self, model_path: Path, device: str = "gpu"):
        if ort_genai is None:
            raise ImportError(
                "onnxruntime_genai is required for LLM inference. "
                "Install it with: pip install onnxruntime-genai"
            )

        self.model_path = Path(model_path)
        self.device = device

        # Register WinML execution providers
        try:
            eps = register_execution_providers(ort_genai=True)
            print(f"Registered execution providers: {eps}")
        except Exception as e:
            print(f"Warning: Failed to register WinML execution providers: {e}")

        # Select execution provider and create session
        ep = self._select_execution_provider(device)
        self.session = ort_genai.Create(
            model_folder=str(self.model_path),
            execution_providers=[ep],
        )

        self.tokenizer = ort_genai.Tokenizer(model_folder=str(self.model_path))

    def _select_execution_provider(self, device: str) -> tuple:
        """Select the appropriate execution provider.

        Args:
            device: Target device (gpu/cpu/npu)

        Returns:
            (provider_name, options_dict) tuple
        """
        if device == "gpu":
            return ("DmlExecutionProvider", {})
        if device == "npu":
            return ("QNNExecutionProvider", {})
        return ("CPUExecutionProvider", {})

    def generate(self, prompt: str, **kwargs) -> str:
        """Synchronous full generation.

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
                - max_tokens: int (default 100)
                - temperature: float (default 0.7)
                - top_p: float (default 0.9)
                - top_k: int (default 40)
                - repetition_penalty: float (default 1.0)

        Returns:
            Complete generated text
        """
        input_ids = self.tokenizer.encode(prompt)

        search_options = ort_genai.SearchOptions()
        search_options.max_length = kwargs.get("max_tokens", 100)

        if "temperature" in kwargs:
            search_options.temperature = kwargs["temperature"]
        if "top_p" in kwargs:
            search_options.top_p = kwargs["top_p"]
        if "top_k" in kwargs:
            search_options.top_k = kwargs["top_k"]

        output_ids = self.session.generate(input_ids, search_options)
        return self.tokenizer.decode(output_ids)

    async def generate_streaming(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Streaming generation — yields one token at a time.

        Args:
            prompt: Input prompt
            **kwargs: Generation parameters (same as generate())

        Yields:
            Each generated token as a string
        """
        input_ids = self.tokenizer.encode(prompt)

        search_options = ort_genai.SearchOptions()
        search_options.max_length = kwargs.get("max_tokens", 100)

        if "temperature" in kwargs:
            search_options.temperature = kwargs["temperature"]
        if "top_p" in kwargs:
            search_options.top_p = kwargs["top_p"]
        if "top_k" in kwargs:
            search_options.top_k = kwargs["top_k"]

        # NOTE: streaming API depends on ort_genai version;
        # assumes generate_search() returns a token ID iterator
        token_generator = self.session.generate_search(input_ids, search_options)

        for token_id in token_generator:
            token_str = self.tokenizer.decode([token_id])
            yield token_str
