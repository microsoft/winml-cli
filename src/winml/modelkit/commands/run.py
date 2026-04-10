# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""One-shot inference command — wmk run.

Phase 2 auto-connect: checks if ``wmk serve`` is already running at
localhost:<port>.  If the warm server has the same model loaded, routes the
request there (zero load overhead).  Otherwise falls back to embedded
inference (load → predict → exit).

Usage:
    wmk run microsoft/resnet-50 --input cat.jpg
    wmk run ./build/resnet50/ --input cat.jpg --device gpu
    wmk run model.onnx --task image-classification --input photo.png
    wmk run microsoft/resnet-50 --input "A photo of a cat" --task text-classification
    wmk run microsoft/resnet-50 --input cat.jpg --format json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click


logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8000
_CONNECT_TIMEOUT = 0.5  # seconds to wait for health check


@click.command("run")
@click.argument("model_path")
@click.option(
    "--input",
    "-i",
    "input_data",
    required=True,
    help="Input: image file path, or text string for NLP tasks",
)
@click.option("--task", default=None, help="Task type (auto-detected when possible)")
@click.option(
    "--device",
    default="auto",
    show_default=True,
    help="Device: auto, cpu, gpu, npu",
)
@click.option("--ep", default=None, help="Explicit execution provider short name")
@click.option(
    "--top-k",
    default=5,
    type=int,
    show_default=True,
    help="Top-K results for classification",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Write output to file instead of stdout",
)
@click.option(
    "--port",
    default=_DEFAULT_PORT,
    type=int,
    show_default=True,
    help="Port to check for a running wmk serve instance (Phase 2 auto-connect)",
)
@click.option(
    "--no-connect",
    is_flag=True,
    default=False,
    help="Disable auto-connect; always use embedded inference",
)
@click.pass_context
def run(
    ctx: click.Context,
    model_path: str,
    input_data: str,
    task: str | None,
    device: str,
    ep: str | None,
    top_k: int,
    output_format: str,
    output: str | None,
    port: int,
    no_connect: bool,
) -> None:
    r"""Run one-shot inference on a model.

    Automatically connects to a running ``wmk serve`` instance when available
    (Phase 2 auto-connect), falling back to embedded inference otherwise.

    Examples:
    \b
        # Image classification from file
        wmk run microsoft/resnet-50 --input cat.jpg

        # From build output directory
        wmk run ./build/resnet50/ --input cat.jpg --device gpu

        # Raw ONNX file
        wmk run model.onnx --task image-classification --input photo.png

        # JSON output
        wmk run microsoft/resnet-50 --input cat.jpg --format json

        # Disable auto-connect (always embedded)
        wmk run microsoft/resnet-50 --input cat.jpg --no-connect
    """
    if ctx.obj and ctx.obj.get("debug"):
        logging.getLogger("modelkit").setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Phase 2: try auto-connect to running wmk serve
    # ------------------------------------------------------------------
    if not no_connect:
        result = _try_server_predict(
            port=port,
            model_path=model_path,
            input_data=input_data,
            top_k=top_k,
        )
        if result is not None:
            _print_result(result, output_format=output_format, output_path=output)
            return

    # ------------------------------------------------------------------
    # Embedded inference (Phase 1 fallback)
    # ------------------------------------------------------------------
    from ..serving.engine import InferenceEngine

    engine = InferenceEngine()
    try:
        engine.load(model_path, task=task, device=device, ep=ep)
    except Exception as exc:
        click.echo(f"Error loading model: {exc}", err=True)
        sys.exit(3)

    input_path = Path(input_data)
    try:
        if input_path.exists() and input_path.is_file():
            image_bytes = input_path.read_bytes()
            result = engine.predict(image_bytes=image_bytes, top_k=top_k)
        else:
            result = engine.predict(text=input_data, top_k=top_k)
    except Exception as exc:
        click.echo(f"Error during inference: {exc}", err=True)
        sys.exit(4)

    _print_result(result.model_dump(), output_format=output_format, output_path=output)


# ---------------------------------------------------------------------------
# Phase 2: auto-connect helpers
# ---------------------------------------------------------------------------


def _try_server_predict(
    *,
    port: int,
    model_path: str,
    input_data: str,
    top_k: int,
) -> dict | None:
    """Check if wmk serve is running and route the request there.

    Returns the parsed JSON response dict, or None if unavailable.
    """
    try:
        import httpx
    except ImportError:
        logger.debug("httpx not installed — skipping auto-connect")
        return None

    base_url = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(timeout=_CONNECT_TIMEOUT) as client:
            health = client.get(f"{base_url}/v1/health")
            if health.status_code != 200:
                return None
            info = health.json()
            server_model = info.get("model_id") or info.get("model_path", "")
            # Only delegate if server has the same model loaded
            if not _models_match(server_model, model_path):
                logger.debug(
                    "Server model '%s' != requested '%s' — using embedded",
                    server_model,
                    model_path,
                )
                return None

            # Route request to server
            input_path = Path(input_data)
            if input_path.exists() and input_path.is_file():
                with input_path.open("rb") as f:
                    resp = client.post(
                        f"{base_url}/v1/predict/file",
                        files={"file": (input_path.name, f, "application/octet-stream")},
                        data={"top_k": str(top_k)},
                        timeout=60,
                    )
            else:
                resp = client.post(
                    f"{base_url}/v1/predict",
                    json={"inputs": {"text": [input_data]}, "top_k": top_k},
                    timeout=60,
                )
            resp.raise_for_status()
            logger.debug("Auto-connected to wmk serve at %s", base_url)
            return resp.json()
    except Exception as exc:
        logger.debug("Auto-connect failed (%s) — using embedded inference", exc)
        return None


def _models_match(server_model: str, requested: str) -> bool:
    """Loose comparison: match on base name or exact string."""
    if not server_model:
        return False
    if server_model == requested:
        return True
    return Path(server_model).name == Path(requested).name


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _print_result(
    result: dict,
    *,
    output_format: str,
    output_path: str | None,
) -> None:
    text = json.dumps(result, indent=2) if output_format == "json" else _format_text(result)

    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
    else:
        click.echo(text)


def _format_text(result: dict) -> str:
    lines: list[str] = []
    task = result.get("task", "")
    model_id = result.get("model_id") or result.get("model_path", "")
    device = result.get("device", "")
    ep = result.get("ep", "")
    latency = result.get("latency_ms", 0)

    lines.append(f"Task:    {task}")
    if model_id:
        lines.append(f"Model:   {model_id}")
    lines.append(f"Device:  {device}" + (f" ({ep})" if ep else ""))
    lines.append("")

    predictions = result.get("predictions", [])
    if isinstance(predictions, list):
        lines.append("Results:")
        for i, p in enumerate(predictions, 1):
            label = p.get("label", str(i))
            score = p.get("score", 0.0)
            lines.append(f"  {i:2d}. {label:<30s} {score:.4f}")
    else:
        lines.append(f"Output: {predictions}")

    lines.append("")
    lines.append(f"Latency: {latency:.1f}ms")
    return "\n".join(lines)
