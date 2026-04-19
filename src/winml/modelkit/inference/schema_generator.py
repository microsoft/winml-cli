# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""APISchemaGenerator — registry-driven OpenAI tool generation.

Generates OpenAI function-calling tool definitions from TASK_REGISTRY.
User inputs come from the registry; pipeline parameters from the manifest.
"""

from __future__ import annotations

import re
from typing import Any

from .tasks import BINARY_TYPES, TASK_REGISTRY


# user_inputs type → JSON Schema for tool generation
_TYPE_MAP: dict[str, dict[str, Any]] = {
    "text": {"type": "string"},
    "image": {"type": "string", "contentEncoding": "base64"},
    "audio": {"type": "string", "contentEncoding": "base64"},
    "video": {"type": "string", "contentEncoding": "base64"},
    "json": {"oneOf": [{"type": "object"}, {"type": "array"}]},
    "number": {"type": "number"},
    "boolean": {"type": "boolean"},
}


class APISchemaGenerator:
    """Generate OpenAI-compatible tool definitions from TASK_REGISTRY + manifest.

    The registry provides user_inputs (what the model needs).
    The manifest provides model_id, task, and optional pipeline parameters.
    """

    def __init__(self, manifest: dict) -> None:
        self.manifest = manifest
        self.task: str = manifest.get("task", "")
        self.model_id: str | None = manifest.get("model_id")

    # ---------------------------------------------------------------
    # Tool name and description (manifest-derived)
    # ---------------------------------------------------------------

    def _build_tool_name(self) -> str:
        """Derive tool name from task + model_id.

        Convention: task_model, hyphens→underscores, non-alphanum removed,
        truncated to 64 chars.

        Example: task="question-answering", model_id="deepset/roberta-base-squad2"
        → "question_answering_roberta_base_squad2"
        """
        task_part = self.task.replace("-", "_")

        if self.model_id:
            short = self.model_id.rsplit("/", 1)[-1]
            model_part = re.sub(r"[^a-z0-9]", "_", short.lower())
            model_part = re.sub(r"_+", "_", model_part).strip("_")
            name = f"{task_part}_{model_part}"
        else:
            name = task_part

        name = re.sub(r"[^a-z0-9_]", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name[:64] or "tool"

    def _build_description(self) -> str:
        """Derive description from task + model_id."""
        task_display = self.task.replace("-", " ").title()
        if self.model_id:
            return f"{task_display} using {self.model_id}"
        return task_display

    # ---------------------------------------------------------------
    # Parameter schema (registry-driven)
    # ---------------------------------------------------------------

    def _build_parameters_schema(self) -> dict:
        """Build OpenAI function parameters schema from TASK_REGISTRY.

        Returns: {"type": "object", "properties": {...}, "required": [...]}
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        spec = TASK_REGISTRY.get(self.task)
        if spec:
            for field in spec.user_inputs:
                prop = dict(_TYPE_MAP.get(field.type, {"type": "string"}))
                # Binary types get an encoding hint prepended to description
                if field.type in BINARY_TYPES:
                    prop["description"] = f"Base64-encoded {field.type} bytes — {field.description}"
                else:
                    prop["description"] = field.description
                if not field.required and field.default is not None:
                    prop["default"] = field.default
                properties[field.name] = prop
                if field.required:
                    required.append(field.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    # ---------------------------------------------------------------
    # Tool generation
    # ---------------------------------------------------------------

    def generate_openai_tool(self) -> dict:
        """Generate OpenAI function-calling tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self._build_tool_name(),
                "description": self._build_description(),
                "parameters": self._build_parameters_schema(),
            },
        }

    def generate_tools_list(self) -> list[dict]:
        """Generate tool definitions for the loaded model."""
        return [self.generate_openai_tool()]
