# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""APISchemaGenerator - Auto-generate OpenAI tool definitions from manifest.

Fully manifest-driven: derives all API schemas from build_manifest.json fields.
No hardcoded model names, architectures, or task-specific logic.

Design principle: Single source of truth is the manifest. All schema generation
rules are derived from manifest fields: task, parameters, model_io, processing,
engine.format, and model_id.
"""

from __future__ import annotations

import re
from typing import Any


class APISchemaGenerator:
    """Generate OpenAI-compatible tool definitions from build manifest.

    The manifest contains:
    - task: e.g. "image-classification", "text-generation", "object-detection"
    - model_id: HuggingFace model ID or custom name
    - parameters: dict of parameter definitions (name → {"type": ..., "default": ...})
    - model_io: dict with "inputs" and "outputs" (shapes, dtypes)
    - processing: dict with "preprocessing" and "postprocessing" config
    - engine.format: "onnxruntime" or "onnxruntime_genai"
    """

    def __init__(self, manifest: dict) -> None:
        """Initialize from build_manifest.json content."""
        self.manifest = manifest
        self.task: str = manifest.get("task", "")
        self.model_id: str | None = manifest.get("model_id")
        self.parameters: dict = manifest.get("parameters", {})
        self.model_io: dict = manifest.get("model_io", {})
        self.processing: dict = manifest.get("processing", {})
        self.engine_format: str = manifest.get("engine", {}).get("format", "onnxruntime")

    # ---------------------------------------------------------------
    # Task detection helpers (manifest-driven, no hardcoding)
    # ---------------------------------------------------------------

    def _is_image_task(self) -> bool:
        """Detect image-input tasks from manifest.task."""
        image_prefixes = (
            "image-classification",
            "image-segmentation",
            "image-to-text",
            "object-detection",
            "semantic-segmentation",
        )
        return any(self.task.startswith(p) for p in image_prefixes) or self.task.startswith(
            "image-"
        )

    def _is_llm_task(self) -> bool:
        """Detect LLM (text-generation) tasks."""
        return self.task == "text-generation" or self.engine_format == "onnxruntime_genai"

    def _is_text_task(self) -> bool:
        """Detect text-input tasks (non-LLM)."""
        text_tasks = ("text-classification", "sentiment-analysis", "token-classification")
        return self.task in text_tasks or ("text" in self.task and not self._is_llm_task())

    # ---------------------------------------------------------------
    # Tool name and description builders (manifest-derived)
    # ---------------------------------------------------------------

    def _build_tool_name(self) -> str:
        """Derive tool name from task + model_id.

        Example: task="image-classification", model_id="microsoft/resnet-50"
        → "classify_image_microsoft_resnet_50"

        No hardcoded model-specific logic — purely mechanical transformation.
        """
        # Task part: extract verb from task name
        task_parts = self.task.lower().split("-")

        # For "image-classification" → "classify_image"
        # For "object-detection" → "detect_object"
        # For "text-generation" → "generate_text"
        if len(task_parts) >= 2:
            verb = task_parts[-1]  # last part
            noun = "-".join(task_parts[:-1])  # everything else
            task_suffix = f"{verb}_{noun}"
        else:
            task_suffix = self.task.replace("-", "_")

        # Model part: sanitize model_id
        if self.model_id:
            # Replace /, -, . with _
            model_part = re.sub(r"[/-]", "_", self.model_id).lower()
            # Remove duplicate underscores
            model_part = re.sub(r"_+", "_", model_part)
            tool_name = f"{task_suffix}_{model_part}"
        else:
            tool_name = task_suffix

        # Ensure valid Python identifier
        tool_name = re.sub(r"[^a-z0-9_]", "_", tool_name)
        tool_name = re.sub(r"_+", "_", tool_name)
        tool_name = tool_name.strip("_")

        return tool_name or "tool"

    def _build_description(self) -> str:
        """Derive description from task + model_id."""
        # Humanize task name
        task_display = self.task.replace("-", " ").title()

        if self.model_id:
            return f"{task_display} using {self.model_id}"
        return task_display

    # ---------------------------------------------------------------
    # Parameter schema builder (manifest-driven)
    # ---------------------------------------------------------------

    def _build_parameters_schema(self) -> dict:
        """Build OpenAI function parameters schema from manifest.

        Returns: {"type": "object", "properties": {...}, "required": [...]}

        Strategy:
        1. Add task-specific required input (image_bytes, prompt, messages)
        2. Copy manifest["parameters"] as optional properties
        3. For LLM: always add max_tokens, temperature, top_p
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        # Step 1: Task-specific required inputs
        if self._is_image_task():
            properties["image_bytes"] = {
                "type": "string",
                "description": "Base64-encoded image data",
            }
            required.append("image_bytes")

        elif self._is_llm_task():
            # LLM: support both messages array (Claude/OpenAI style) and prompt string
            properties["messages"] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"role": {"type": "string"}, "content": {"type": "string"}},
                },
                "description": "Chat message history (OpenAI format)",
            }
            properties["prompt"] = {
                "type": "string",
                "description": "Text prompt (alternative to messages)",
            }
            # Tool schema doesn't support conditional required, so keep both
            # optional and let the handler decide at runtime.

        elif self._is_text_task():
            properties["prompt"] = {
                "type": "string",
                "description": "Text input",
            }
            required.append("prompt")

        # Step 2: Copy manifest["parameters"] as optional properties
        type_mapping = {"int": "integer", "float": "number", "str": "string", "bool": "boolean"}

        for param_name, param_def in self.parameters.items():
            if param_name in properties:
                continue  # Skip if already added above

            param_type = param_def.get("type", "string")
            openai_type = type_mapping.get(param_type, "string")

            prop: dict[str, Any] = {"type": openai_type}

            if "description" in param_def:
                prop["description"] = param_def["description"]
            if "default" in param_def:
                prop["default"] = param_def["default"]
            if "min" in param_def:
                prop["minimum"] = param_def["min"]
            if "max" in param_def:
                prop["maximum"] = param_def["max"]

            properties[param_name] = prop

        # Step 3: LLM always adds generation parameters
        if self._is_llm_task():
            # Add max_tokens
            if "max_tokens" not in properties:
                default_max = self.parameters.get("max_tokens", {}).get("default", 100)
                properties["max_tokens"] = {
                    "type": "integer",
                    "description": "Maximum tokens to generate",
                    "default": default_max,
                    "minimum": 1,
                }

            # Add temperature
            if "temperature" not in properties:
                default_temp = self.parameters.get("temperature", {}).get("default", 0.7)
                properties["temperature"] = {
                    "type": "number",
                    "description": "Sampling temperature (0-2)",
                    "default": default_temp,
                    "minimum": 0.0,
                    "maximum": 2.0,
                }

            # Add top_p
            if "top_p" not in properties:
                default_top_p = self.parameters.get("top_p", {}).get("default", 0.9)
                properties["top_p"] = {
                    "type": "number",
                    "description": "Nucleus sampling parameter",
                    "default": default_top_p,
                    "minimum": 0.0,
                    "maximum": 1.0,
                }

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    # ---------------------------------------------------------------
    # Tool generation
    # ---------------------------------------------------------------

    def generate_openai_tool(
        self,
        *,
        include_streaming_extension: bool = False,
        streaming_endpoint: str = "/v1/predict/stream",
    ) -> dict:
        """Generate OpenAI function-calling tool definition.

        Returns the standard OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {...}
            }
        }

        If include_streaming_extension=True (for LLM), adds:
        "x-streaming": "/v1/predict/stream"
        """
        tool = {
            "type": "function",
            "function": {
                "name": self._build_tool_name(),
                "description": self._build_description(),
                "parameters": self._build_parameters_schema(),
            },
        }

        if include_streaming_extension:
            tool["x-streaming"] = streaming_endpoint

        return tool

    def generate_tools_list(self) -> list[dict]:
        """Generate tool definitions for the loaded model.

        For inference models: returns single tool without streaming.
        For LLM models: returns tool with streaming extension.
        """
        return [
            self.generate_openai_tool(
                include_streaming_extension=self._is_llm_task(),
            )
        ]
