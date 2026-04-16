# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""One-shot inference command — winml run.

Usage:
    winml run --model microsoft/resnet-50 --file cat.jpg
    winml run --model ./build/resnet50/ --file cat.jpg --device gpu
    winml run --model model.onnx --task image-classification --file photo.png
    winml run --model whisper --file speech.wav
    winml run --model llava --file img.jpg --text "What is this?"
    winml run --model bert --text "Hello world" --task text-classification
    winml run --model model --text "Once upon" -P max_new_tokens=100 -P temperature=0.7
    winml run --model microsoft/resnet-50 --file cat.jpg --format json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

from ..utils import cli as cli_utils


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
@cli_utils.model_option(required=True)
@click.option(
    "--file",
    "-f",
    "files",
    multiple=True,
    help="Input media file: image, audio, or video (repeatable)",
)
@click.option(
    "--text",
    "-t",
    default=None,
    help="Text input for NLP / multimodal tasks",
)
@click.option("--task", default=None, help="Task type (auto-detected when possible)")
@cli_utils.device_option(required=False, default="auto", include_auto=True)
@cli_utils.ep_option(required=False)
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
    "--connect",
    is_flag=True,
    default=False,
    help="Auto-connect to a running winml serve instance instead of embedded inference",
)
@click.pass_context
def run(
    ctx: click.Context,
    model: str,
    files: tuple[str, ...],
    text: str | None,
    task: str | None,
    device: str,
    ep: str | None,
    params: tuple[str, ...],
    output_format: str,
    output: str | None,
    port: int,
    connect: bool,
) -> None:
    r"""Run one-shot inference on a model.

    Uses embedded inference by default. Pass ``--connect`` to route
    through a running ``winml serve`` instance instead.

    Examples:
    \b
        # Image classification
        winml run --model microsoft/resnet-50 --file cat.jpg

        # From build output directory
        winml run --model ./build/resnet50/ --file cat.jpg --device gpu

        # Audio (speech recognition)
        winml run --model openai/whisper --file speech.wav

        # Multimodal (image + text)
        winml run --model llava --file img.jpg --text "Describe this image"

        # Multiple images
        winml run --model llava --file a.jpg --file b.jpg --text "Compare"

        # Text only
        winml run --model bert --text "Hello world"

        # Extra pipeline parameters
        winml run --model model --text "Once upon" -P max_new_tokens=100 -P temperature=0.7

        # Route through a running serve instance
        winml run --model microsoft/resnet-50 --file cat.jpg --connect
    """
    if not files and text is None:
        click.echo("Error: provide at least --file or --text.", err=True)
        ctx.exit(2)

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

    # Read file bytes
    file_bytes_list: list[bytes] = []
    for fp in files:
        p = Path(fp)
        if not p.exists() or not p.is_file():
            click.echo(f"Error: file not found: {fp}", err=True)
            ctx.exit(2)
        file_bytes_list.append(p.read_bytes())

    # ------------------------------------------------------------------
    # Try auto-connect to running winml serve
    # ------------------------------------------------------------------
    if connect:
        result = _try_server_predict(
            port=port,
            model_path=model,
            files=files,
            text=text,
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
        engine.load(model, task=task, device=device, ep=ep)
    except Exception as exc:
        click.echo(f"Error loading model: {exc}", err=True)
        ctx.exit(3)

    try:
        result = engine.predict(
            files=file_bytes_list or None,
            text=text,
            **pipeline_kwargs,
        )
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
    files: tuple[str, ...],
    text: str | None,
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

            if len(files) == 1:
                # Single file → multipart upload
                fp = Path(files[0])
                with fp.open("rb") as f:
                    resp = client.post(
                        f"{base_url}/v1/predict/file",
                        files={"file": (fp.name, f, "application/octet-stream")},
                        data={"params": json.dumps(pipeline_kwargs)},
                        timeout=60,
                    )
            else:
                # Multiple files or text-only → JSON endpoint
                import base64

                payload: dict[str, Any] = {"params": pipeline_kwargs}
                if files:
                    payload["files"] = [
                        base64.b64encode(Path(fp).read_bytes()).decode() for fp in files
                    ]
                if text is not None:
                    payload["text"] = text
                resp = client.post(
                    f"{base_url}/v1/predict",
                    json=payload,
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
