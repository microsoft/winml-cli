#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Standalone MCP server for ModelKit inference.

This script is intentionally self-contained and does NOT import from
winml.modelkit, avoiding heavy ML dependency imports (PyTorch etc.)
that would cause Claude Desktop MCP connection timeouts.

Multi-model aware: queries /v1/models at startup and generates a
uniquely-named tool per loaded model (e.g. classify_image_resnet_50,
detect_object_yolov8).  If the server is unreachable, registers a
generic fallback tool.

Usage:
    python scripts/mcp_server.py
    python scripts/mcp_server.py --model-url http://localhost:9000
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

# Task category constants (mirrors schema_generator.py, no hardcoded models)
_IMAGE_PREFIXES = (
    "image-classification",
    "image-segmentation",
    "image-to-text",
    "object-detection",
    "semantic-segmentation",
)
_TEXT_TASKS = ("text-classification", "sentiment-analysis", "token-classification")


def _is_image_task(task: str) -> bool:
    return any(task.startswith(p) for p in _IMAGE_PREFIXES) or task.startswith("image-")


def _is_text_task(task: str) -> bool:
    return task in _TEXT_TASKS or "text" in task


# ---------------------------------------------------------------------------
# Tool naming (mirrors schema_generator._build_tool_name logic)
# ---------------------------------------------------------------------------


def _build_tool_name(task: str, model_id: str) -> str:
    """Derive a unique tool name from task + model_id.

    Examples:
        ("image-classification", "facebook/convnext-tiny-224")
        → "classify_image_facebook_convnext_tiny_224"

        ("object-detection", "facebook/detr-resnet-50")
        → "detect_object_facebook_detr_resnet_50"
    """
    task_parts = task.lower().split("-")
    if len(task_parts) >= 2:
        verb = task_parts[-1]
        noun = "_".join(task_parts[:-1])
        task_prefix = f"{verb}_{noun}"
    else:
        task_prefix = task.replace("-", "_")

    if model_id:
        model_part = re.sub(r"[/\-.]", "_", model_id).lower()
        model_part = re.sub(r"_+", "_", model_part)
        name = f"{task_prefix}_{model_part}"
    else:
        name = task_prefix

    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "predict"


# ---------------------------------------------------------------------------
# Server creation
# ---------------------------------------------------------------------------


def create_server(model_url: str) -> FastMCP:
    """Create MCP server with per-model tools from ModelKit."""
    model_url = model_url.rstrip("/")
    mcp = FastMCP("modelkit-inference")

    # -- Static tool: list all loaded models --------------------------------

    @mcp.tool()
    async def list_models() -> str:
        """List all loaded models with their task, status, and routing info."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{model_url}/v1/models")
                resp.raise_for_status()
                return json.dumps(resp.json(), indent=2)
        except Exception as e:
            return f"Error: {e}"

    # -- Dynamic per-model tools -------------------------------------------

    models = _fetch_models(model_url)
    if models:
        for model_info in models:
            if model_info.get("status") != "ready":
                continue
            task = model_info.get("task", "")
            model_id = model_info.get("model_id", "unknown")
            _register_model_tool(mcp, model_url, task, model_id)
    else:
        _register_fallback_predict(mcp, model_url)

    return mcp


def _fetch_models(model_url: str) -> list[dict] | None:
    """Fetch /v1/models to get all loaded models (best-effort)."""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{model_url}/v1/models")
            resp.raise_for_status()
            models = resp.json()
            if isinstance(models, list) and models:
                logger.info("Found %d model(s) on %s", len(models), model_url)
                return models
    except Exception as e:
        logger.info("Cannot reach %s: %s (using fallback tools)", model_url, e)
    return None


# ---------------------------------------------------------------------------
# Per-model tool registration (dispatches by task type)
# ---------------------------------------------------------------------------


def _register_model_tool(mcp: FastMCP, model_url: str, task: str, model_id: str) -> None:
    """Register one tool for a specific model, named and typed by its task."""
    tool_name = _build_tool_name(task, model_id)
    task_display = task.replace("-", " ").title()

    if _is_image_task(task):
        _register_image_tool(mcp, model_url, model_id, tool_name, task_display)
    elif _is_text_task(task):
        _register_text_tool(mcp, model_url, model_id, tool_name, task_display)
    else:
        logger.warning("Skipping unknown task '%s' for model '%s'", task, model_id)
        return

    logger.info("Registered tool '%s' (%s, %s)", tool_name, task, model_id)


# ---------------------------------------------------------------------------
# Image tool
# ---------------------------------------------------------------------------


def _register_image_tool(
    mcp: FastMCP, model_url: str, model_id: str, tool_name: str, task_display: str
) -> None:
    async def _handler(image_path: str, top_k: int = 5) -> str:
        path = Path(image_path)
        if not path.is_file():
            return f"Error: File not found: {image_path}"
        image_data = path.read_bytes()
        if len(image_data) > 20 * 1024 * 1024:
            return "Error: Image too large (max 20 MB)"
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{model_url}/v1/predict/file",
                    files={"file": (path.name, image_data, _guess_media_type(path))},
                    data={"top_k": str(top_k), "model_id": model_id},
                )
                resp.raise_for_status()
                return json.dumps(resp.json(), indent=2)
        except Exception as e:
            return f"Error: {e}"

    _handler.__name__ = tool_name
    _handler.__doc__ = (
        f"{task_display} using {model_id}.\n\n"
        "Analyzes a local image file.\n\n"
        "Args:\n"
        "    image_path: Absolute path to the image file (JPEG, PNG, etc.)\n"
        "    top_k: Number of top results to return (default: 5)"
    )
    mcp.tool()(_handler)


# ---------------------------------------------------------------------------
# Text tool
# ---------------------------------------------------------------------------


def _register_text_tool(
    mcp: FastMCP, model_url: str, model_id: str, tool_name: str, task_display: str
) -> None:
    async def _handler(text: str, top_k: int = 5) -> str:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{model_url}/v1/predict",
                    json={"text": text, "top_k": top_k, "task": model_id},
                )
                resp.raise_for_status()
                return json.dumps(resp.json(), indent=2)
        except Exception as e:
            return f"Error: {e}"

    _handler.__name__ = tool_name
    _handler.__doc__ = (
        f"{task_display} using {model_id}.\n\n"
        "Args:\n"
        "    text: The text to analyze\n"
        "    top_k: Number of top results to return (default: 5)"
    )
    mcp.tool()(_handler)


# ---------------------------------------------------------------------------
# Fallback: generic predict when server is unreachable at startup
# ---------------------------------------------------------------------------


def _register_fallback_predict(mcp: FastMCP, model_url: str) -> None:
    @mcp.tool()
    async def predict(
        image_path: str | None = None, text: str | None = None, top_k: int = 5
    ) -> str:
        """Run inference on the ModelKit server.

        For image tasks, provide image_path (absolute file path).
        For text tasks, provide text.
        Use list_models first to discover loaded models.

        Args:
            image_path: Absolute path to image file (for image tasks)
            text: Input text (for text/NLP tasks)
            top_k: Number of top results (default: 5)
        """
        if image_path:
            path = Path(image_path)
            if not path.is_file():
                return f"Error: File not found: {image_path}"
            image_data = path.read_bytes()
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        f"{model_url}/v1/predict/file",
                        files={"file": (path.name, image_data, _guess_media_type(path))},
                        data={"top_k": str(top_k)},
                    )
                    resp.raise_for_status()
                    return json.dumps(resp.json(), indent=2)
            except Exception as e:
                return f"Error: {e}"
        elif text:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        f"{model_url}/v1/predict",
                        json={"text": text, "top_k": top_k},
                    )
                    resp.raise_for_status()
                    return json.dumps(resp.json(), indent=2)
            except Exception as e:
                return f"Error: {e}"
        else:
            return "Error: Provide either 'image_path' or 'text'"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _guess_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }.get(suffix, "application/octet-stream")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run the MCP server."""
    parser = ArgumentParser(description="ModelKit MCP Server (standalone)")
    parser.add_argument(
        "--model-url",
        default="http://localhost:8000",
        help="Base URL of the ModelKit service (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    server = create_server(args.model_url)
    logger.info("Starting MCP server (ModelKit: %s)", args.model_url)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
