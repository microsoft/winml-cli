# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""One-shot inference command — winml run.

Usage:
    winml run microsoft/resnet-50 --input cat.jpg
    winml run ./build/resnet50/ --input cat.jpg --device gpu
    winml run model.onnx --task image-classification --input photo.png
    winml run microsoft/resnet-50 --input "A photo of a cat" --task text-classification
    winml run microsoft/resnet-50 --input cat.jpg --format json
    winml run model --input "Once upon" -P max_new_tokens=100 -P temperature=0.7
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import click


logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8000
_CONNECT_TIMEOUT = 0.5  # seconds to wait for health check


def _parse_param_value(value: str) -> int | float | bool | str:
    """Auto-parse a string value to int, float, bool, or str."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value


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
    "-P",
    "--param",
    "params",
    multiple=True,
    help="Pipeline parameter as KEY=VALUE (repeatable). "
    "Values auto-parsed as int/float/bool/str. "
    "E.g. -P max_new_tokens=100 -P temperature=0.7",
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
    help="Port to check for a running winml serve instance (auto-connect)",
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
    params: tuple[str, ...],
    output_format: str,
    output: str | None,
    port: int,
    no_connect: bool,
) -> None:
    r"""Run one-shot inference on a model.

    Automatically connects to a running ``winml serve`` instance when available,
    falling back to embedded inference otherwise.

    Examples:
    \b
        # Image classification from file
        winml run microsoft/resnet-50 --input cat.jpg

        # From build output directory
        winml run ./build/resnet50/ --input cat.jpg --device gpu

        # Raw ONNX file
        winml run model.onnx --task image-classification --input photo.png

        # JSON output
        winml run microsoft/resnet-50 --input cat.jpg --format json

        # Extra pipeline parameters
        winml run model --input "Once upon" -P max_new_tokens=100 -P temperature=0.7

        # Disable auto-connect (always embedded)
        winml run microsoft/resnet-50 --input cat.jpg --no-connect
    """
    if ctx.obj and ctx.obj.get("debug"):
        logging.getLogger("modelkit").setLevel(logging.DEBUG)

    # Build pipeline_kwargs from -P/--param entries
    pipeline_kwargs: dict[str, Any] = {}
    for p in params:
        if "=" not in p:
            click.echo(f"Error: invalid --param format: '{p}'. Use KEY=VALUE.", err=True)
            ctx.exit(2)
        k, v = p.split("=", 1)
        pipeline_kwargs[k] = _parse_param_value(v)

    # ------------------------------------------------------------------
    # Try auto-connect to running winml serve
    # ------------------------------------------------------------------
    if not no_connect:
        result = _try_server_predict(
            port=port,
            model_path=model_path,
            input_data=input_data,
            pipeline_kwargs=pipeline_kwargs,
        )
        if result is not None:
            _print_result(result, output_format=output_format, output_path=output)
            return

    # ------------------------------------------------------------------
    # Embedded inference fallback
    # ------------------------------------------------------------------
    from ..serve.engine import InferenceEngine

    engine = InferenceEngine()
    try:
        engine.load(model_path, task=task, device=device, ep=ep)
    except Exception as exc:
        click.echo(f"Error loading model: {exc}", err=True)
        ctx.exit(3)

    input_path = Path(input_data)
    try:
        if input_path.exists() and input_path.is_file():
            image_bytes = input_path.read_bytes()
            result = engine.predict(image_bytes=image_bytes, **pipeline_kwargs)
        else:
            result = engine.predict(text=input_data, **pipeline_kwargs)
    except Exception as exc:
        click.echo(f"Error during inference: {exc}", err=True)
        ctx.exit(4)

    _print_result(result.model_dump(), output_format=output_format, output_path=output)


# ---------------------------------------------------------------------------
# Auto-connect helpers
# ---------------------------------------------------------------------------


def _try_server_predict(
    *,
    port: int,
    model_path: str,
    input_data: str,
    pipeline_kwargs: dict[str, Any],
) -> dict | None:
    """Check if winml serve is running and route the request there.

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
            if not _models_match(server_model, model_path):
                logger.debug(
                    "Server model '%s' != requested '%s' — using embedded",
                    server_model,
                    model_path,
                )
                return None

            input_path = Path(input_data)
            if input_path.exists() and input_path.is_file():
                with input_path.open("rb") as f:
                    resp = client.post(
                        f"{base_url}/v1/predict/file",
                        files={"file": (input_path.name, f, "application/octet-stream")},
                        data={"params": json.dumps(pipeline_kwargs)},
                        timeout=60,
                    )
            else:
                resp = client.post(
                    f"{base_url}/v1/predict",
                    json={"text": input_data, "params": pipeline_kwargs},
                    timeout=60,
                )
            resp.raise_for_status()
            logger.debug("Auto-connected to winml serve at %s", base_url)
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
        try:
            click.echo(text)
        except OSError:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()


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
