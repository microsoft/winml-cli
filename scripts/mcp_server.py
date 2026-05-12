#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Standalone MCP server for WinML CLI inference.

This script is intentionally self-contained and does NOT import from
winml.modelkit, avoiding heavy ML dependency imports (PyTorch etc.)
that would cause Claude Desktop MCP connection timeouts.

Schema-driven: queries ``/v1/models`` at startup and fetches per-model
schemas via ``/v1/models/{model_id}/schema``.  Each loaded model gets a
uniquely-named tool whose handler and docstring are generated directly
from the schema — no hardcoded task-category branching.

Usage:
    python scripts/mcp_server.py
    python scripts/mcp_server.py --server-url http://localhost:9000
"""

from __future__ import annotations

import json
import logging
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

import httpx
from mcp.server import FastMCP


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stderr)

# Binary input types that require file reading
_BINARY_TYPES = frozenset({"image", "audio", "video"})

_MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


# ---------------------------------------------------------------------------
# Tool naming (mirrors schema_generator._build_tool_name logic)
# ---------------------------------------------------------------------------


def _build_tool_name(task: str, model_id: str) -> str:
    """Derive a unique tool name from task + model_id.

    Convention: ``task_model``, hyphens→underscores, non-alphanum removed,
    truncated to 64 chars.

    Examples:
        ("image-classification", "microsoft/resnet-50")
        → "image_classification_resnet_50"
    """
    task_part = task.replace("-", "_")

    if model_id:
        short = model_id.rsplit("/", 1)[-1]
        model_part = re.sub(r"[^a-z0-9]", "_", short.lower())
        model_part = re.sub(r"_+", "_", model_part).strip("_")
        name = f"{task_part}_{model_part}"
    else:
        name = task_part

    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:64] or "predict"


# ---------------------------------------------------------------------------
# Tool docstring generation (schema-driven)
# ---------------------------------------------------------------------------


def _build_tool_doc(
    task: str,
    model_id: str,
    user_inputs: list[dict],
    has_binary: bool,
) -> str:
    """Generate a rich tool docstring from the model schema."""
    task_display = task.replace("-", " ").title()
    lines = [f"{task_display} using {model_id}.", ""]

    if user_inputs:
        lines.append("Model inputs:")
        for inp in user_inputs:
            req = "required" if inp.get("required") else "optional"
            desc = inp.get("description", inp["name"])
            lines.append(f"  {inp['name']} ({inp['type']}, {req}): {desc}")
        lines.append("")

    # Usage hints
    lines.append("Args:")
    if has_binary:
        lines.append("    file_path: Absolute path to the input file (image, audio, or video)")
    text_fields = [f for f in user_inputs if f["type"] == "text"]
    if len(text_fields) == 1:
        lines.append(f"    text: {text_fields[0].get('description', 'Text input')}")
    non_shortcut = [
        f
        for f in user_inputs
        if f["type"] not in _BINARY_TYPES and not (f["type"] == "text" and len(text_fields) == 1)
    ]
    if non_shortcut:
        example_keys = ", ".join(f'"{f["name"]}": ...' for f in non_shortcut)
        lines.append(f"    inputs_json: JSON with named inputs, e.g. {{{example_keys}}}")
    lines.append('    params_json: JSON pipeline parameters (e.g. {"top_k": 5})')

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Media type guessing (for multipart uploads)
# ---------------------------------------------------------------------------


_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
}


def _guess_media_type(path: Path) -> str:
    return _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


# ---------------------------------------------------------------------------
# Server creation
# ---------------------------------------------------------------------------


def create_server(model_url: str) -> FastMCP:
    """Create MCP server with schema-driven per-model tools."""
    model_url = model_url.rstrip("/")
    mcp = FastMCP("winmlcli-inference")

    # -- Static tool: list all loaded models --------------------------------

    @mcp.tool()
    async def list_models() -> str:
        """List all loaded models with their task, status, and routing info."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{model_url}/v1/models")
                resp.raise_for_status()
                return json.dumps(resp.json(), indent=2)
        except httpx.HTTPError as e:
            return f"Error: {e}"

    # -- Dynamic per-model tools -------------------------------------------

    models = _fetch_models(model_url)
    if models is None:
        # Server unreachable — register a generic fallback tool
        _register_fallback_predict(mcp, model_url)
    elif models:
        for model_info in models:
            if model_info.get("status") != "ready":
                continue
            model_id = model_info.get("model_id", "unknown")
            task = model_info.get("task", "")
            schema = _fetch_model_schema(model_url, model_id)
            user_inputs = schema.get("user_inputs", []) if schema else []
            _register_model_tool(mcp, model_url, model_id, task, user_inputs)
    else:
        # Server reachable but no models loaded yet — register fallback
        logger.info("Server reachable but no models loaded — registering fallback tool")
        _register_fallback_predict(mcp, model_url)

    return mcp


# ---------------------------------------------------------------------------
# Data fetching (best-effort, sync at startup)
# ---------------------------------------------------------------------------


def _fetch_models(model_url: str) -> list[dict] | None:
    """Fetch /v1/models to get all loaded models.

    Returns:
        list[dict]: List of model info dicts (may be empty).
        None: Server is unreachable.
    """
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{model_url}/v1/models")
            resp.raise_for_status()
            models = resp.json()
            if isinstance(models, list):
                logger.info("Found %d model(s) on %s", len(models), model_url)
                return models
    except httpx.HTTPError as e:
        logger.info("Cannot reach %s: %s (using fallback tools)", model_url, e)
    return None


def _fetch_model_schema(model_url: str, model_id: str) -> dict | None:
    """Fetch /v1/models/{model_id}/schema for a specific model."""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{model_url}/v1/models/{model_id}/schema")
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as e:
        logger.warning("Cannot fetch schema for '%s': %s", model_id, e)
    return None


# ---------------------------------------------------------------------------
# Schema-driven tool registration
# ---------------------------------------------------------------------------


def _register_model_tool(
    mcp: FastMCP,
    model_url: str,
    model_id: str,
    task: str,
    user_inputs: list[dict],
) -> None:
    """Register one tool for a model.  Handler params derived from schema."""
    tool_name = _build_tool_name(task, model_id)
    has_binary = any(f["type"] in _BINARY_TYPES for f in user_inputs)
    doc = _build_tool_doc(task, model_id, user_inputs, has_binary)

    async def _handler(
        file_path: str | None = None,
        text: str | None = None,
        inputs_json: str = "{}",
        params_json: str = "{}",
    ) -> str:
        try:
            pipe_params = json.loads(params_json)
        except json.JSONDecodeError as e:
            return f"Error: invalid params_json: {e}"
        try:
            extra_inputs = json.loads(inputs_json)
        except json.JSONDecodeError as e:
            return f"Error: invalid inputs_json: {e}"

        # -- Route 1: file_path provided → multipart /v1/predict/file ------
        if file_path:
            return await _predict_file(
                model_url,
                model_id,
                file_path,
                text,
                pipe_params,
            )

        # -- Route 2: no file → JSON /v1/predict ---------------------------
        inputs: dict = {}

        # Map `text` shortcut to the first text-type field in schema
        if text is not None:
            text_fields = [f for f in user_inputs if f["type"] == "text"]
            if len(text_fields) == 1:
                inputs[text_fields[0]["name"]] = text
            else:
                # Multiple or zero text fields — fall through to inputs_json
                inputs["text"] = text

        inputs.update(extra_inputs)

        if not inputs:
            return "Error: provide file_path, text, or inputs_json"

        return await _predict_json(model_url, model_id, inputs, pipe_params)

    _handler.__name__ = tool_name
    _handler.__doc__ = doc
    mcp.tool()(_handler)
    logger.info("Registered tool '%s' (%s, %s)", tool_name, task, model_id)


# ---------------------------------------------------------------------------
# Predict helpers (shared by model tools and fallback)
# ---------------------------------------------------------------------------


async def _predict_file(
    model_url: str,
    model_id: str,
    file_path: str,
    text: str | None,
    pipe_params: dict,
) -> str:
    """Send a file to /v1/predict/file and return the JSON result."""
    path = Path(file_path)
    if not path.is_file():
        return f"Error: file not found: {file_path}"
    data = path.read_bytes()
    if len(data) > _MAX_FILE_SIZE:
        return f"Error: file too large (max {_MAX_FILE_SIZE // (1024 * 1024)} MB)"
    try:
        form_data: dict[str, str] = {
            "model_id": model_id,
            "params": json.dumps(pipe_params),
        }
        if text is not None:
            form_data["text"] = text
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{model_url}/v1/predict/file",
                files={"file": (path.name, data, _guess_media_type(path))},
                data=form_data,
            )
            resp.raise_for_status()
            return json.dumps(resp.json(), indent=2)
    except httpx.HTTPError as e:
        return f"Error: {e}"


async def _predict_json(
    model_url: str,
    model_id: str,
    inputs: dict,
    pipe_params: dict,
) -> str:
    """Send named inputs to /v1/predict and return the JSON result."""
    try:
        payload = {"inputs": inputs, "params": pipe_params}
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{model_url}/v1/predict",
                params={"model_id": model_id},
                json=payload,
            )
            resp.raise_for_status()
            return json.dumps(resp.json(), indent=2)
    except httpx.HTTPError as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Fallback: generic predict when server is unreachable at startup
# ---------------------------------------------------------------------------


def _register_fallback_predict(mcp: FastMCP, model_url: str) -> None:
    @mcp.tool()
    async def predict(
        file_path: str | None = None,
        text: str | None = None,
        inputs_json: str = "{}",
        params_json: str = "{}",
    ) -> str:
        """Run inference on the WinML CLI server.

        Use list_models first to discover loaded models and their schemas.

        Args:
            file_path: Absolute path to a local file (image, audio, or video)
            text: Text input for text-based models
            inputs_json: JSON object with named inputs for multi-input models
            params_json: JSON pipeline parameters (e.g. {"top_k": 5})
        """
        try:
            pipe_params = json.loads(params_json)
        except json.JSONDecodeError as e:
            return f"Error: invalid params_json: {e}"
        try:
            extra_inputs = json.loads(inputs_json)
        except json.JSONDecodeError as e:
            return f"Error: invalid inputs_json: {e}"

        if file_path:
            return await _predict_file(model_url, "_", file_path, text, pipe_params)

        inputs: dict = {}
        if text is not None:
            inputs["text"] = text
        inputs.update(extra_inputs)

        if not inputs:
            return "Error: provide file_path, text, or inputs_json"

        return await _predict_json(model_url, "_", inputs, pipe_params)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run the MCP server."""
    parser = ArgumentParser(description="WinML CLI MCP Server (standalone)")
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="Base URL of the WinML CLI service (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    server = create_server(args.server_url)
    logger.info("Starting MCP server (WinML CLI: %s)", args.server_url)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
