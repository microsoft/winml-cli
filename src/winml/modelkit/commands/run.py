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

    # Named inputs (any model):
    winml run --model roberta-qa --input question="Who?" --input context="Tim Cook is..."
    winml run --model sam --file img.jpg --input input_points='[[100,200]]'

    # Schema discovery:
    winml run --model ./build/roberta-qa/ --schema
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import click

from ..utils import cli as cli_utils


if TYPE_CHECKING:
    from ..utils.constants import EPNameOrAlias


logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8000
_CONNECT_TIMEOUT = 0.5  # seconds to wait for health check


# ---------------------------------------------------------------------------
# Input parsing helpers
# ---------------------------------------------------------------------------


def _parse_param_value(value: str) -> int | float | bool | str:
    """Auto-parse a -P value to int, float, bool, or str."""
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


def _coerce_value(value: str, field_type: str, name: str) -> Any:
    """Coerce a raw --input string value based on the schema type."""
    from ..inference.tasks import BINARY_TYPES

    if field_type in BINARY_TYPES:
        if not value.startswith("@"):
            raise click.ClickException(
                f"--input {name}: file inputs must use @path syntax (e.g. --input {name}=@file.jpg)"
            )
        path = Path(value[1:])
        if not path.exists():
            raise click.ClickException(f"--input {name}: file not found: {path}")
        return path.read_bytes()
    if field_type == "text":
        return value  # always string, never JSON-parsed
    if field_type == "json":
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"--input {name}: invalid JSON: {e}") from e
    if field_type == "number":
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            raise click.ClickException(f"--input {name}: expected number, got '{value}'") from None
    if field_type == "boolean":
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        raise click.ClickException(f"--input {name}: expected true/false, got '{value}'")
    return value


def _parse_heuristic(value: str) -> Any:
    """No schema: @path → bytes; valid JSON → parse; otherwise → string."""
    if value.startswith("@"):
        path = Path(value[1:])
        if path.exists():
            return path.read_bytes()
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        pass
    return value


def _coerce_inputs(
    raw: dict[str, str],
    schema: list | None,
) -> dict[str, Any]:
    """Coerce raw --input string values based on schema types.

    When schema is None (unregistered task / raw tensor), uses heuristics.
    """
    if schema is None:
        return {k: _parse_heuristic(v) for k, v in raw.items()}

    schema_map = {f.name: f for f in schema}
    result: dict[str, Any] = {}
    for name, value in raw.items():
        field = schema_map.get(name)
        if field is None:
            # Unknown input — will be caught by _validate_inputs later
            result[name] = _parse_heuristic(value)
        else:
            result[name] = _coerce_value(value, field.type, name)
    return result


def _resolve_shortcuts(
    file_bytes_list: list[bytes],
    text: str | None,
    named_inputs: dict[str, Any],
    schema: list | None,
) -> dict[str, Any]:
    """Merge --file/--text shortcuts with --input into a unified inputs dict.

    Raises ClickException on ambiguity or conflict.
    """
    from ..inference.tasks import BINARY_TYPES

    inputs = dict(named_inputs)

    if file_bytes_list and len(file_bytes_list) > 1:
        raise click.ClickException(
            f"--file accepts only one file via shortcut (got {len(file_bytes_list)}). "
            "Use --input for multiple file inputs "
            "(e.g. --input image_0=@a.jpg --input image_1=@b.jpg)."
        )

    if file_bytes_list:
        if schema is None:
            # No schema — use 'file' as fallback name
            if "file" in inputs:
                raise click.ClickException(
                    "Input 'file' specified twice (via --file and --input file=@...)."
                )
            inputs["file"] = file_bytes_list[0]
        else:
            binary_fields = [f for f in schema if f.type in BINARY_TYPES]
            if len(binary_fields) == 0:
                raise click.ClickException(
                    "--file not supported: task has no file-type input. "
                    "Use --input to specify inputs."
                )
            if len(binary_fields) > 1:
                names = [f.name for f in binary_fields]
                raise click.ClickException(
                    f"--file is ambiguous for this task (has {len(binary_fields)} file inputs: "
                    f"{', '.join(names)}). Use --input to specify: "
                    + " ".join(f"--input {n}=@path" for n in names)
                )
            name = binary_fields[0].name
            if name in inputs:
                raise click.ClickException(
                    f"Input '{name}' specified twice (via --file and --input {name}=@...)."
                )
            inputs[name] = file_bytes_list[0]

    if text is not None:
        if schema is None:
            if "text" in inputs:
                raise click.ClickException(
                    "Input 'text' specified twice (via --text and --input text=...)."
                )
            inputs["text"] = text
        else:
            text_fields = [f for f in schema if f.type == "text"]
            if len(text_fields) == 0:
                raise click.ClickException(
                    "--text not supported: task has no text-type input. "
                    "Use --input to specify inputs."
                )
            if len(text_fields) > 1:
                names = [f.name for f in text_fields]
                raise click.ClickException(
                    f"--text is ambiguous for this task (has {len(text_fields)} text inputs: "
                    f"{', '.join(names)}). Use --input to specify: "
                    + " ".join(f'--input {n}="..."' for n in names)
                )
            name = text_fields[0].name
            if name in inputs:
                raise click.ClickException(
                    f"Input '{name}' specified twice (via --text and --input {name}=...)."
                )
            inputs[name] = text

    return inputs


# ---------------------------------------------------------------------------
# Schema display
# ---------------------------------------------------------------------------


def _print_schema(
    engine: Any,
    *,
    output_format: str = "text",
    output_path: Path | None = None,
) -> None:
    """Print model schema (inputs + parameters) and exit.

    Respects --format (text/json) and -o (output file).
    """
    if output_format == "json":
        text = json.dumps(_schema_to_dict(engine), indent=2)
    else:
        text = _schema_to_text(engine)

    if output_path:
        output_path.write_text(text, encoding="utf-8")
    else:
        try:
            click.echo(text)
        except OSError:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()


def _schema_to_dict(engine: Any) -> dict[str, Any]:
    """Build schema as a JSON-serialisable dict."""
    schema = engine.user_input_schema
    params = engine.pipeline_params

    result: dict[str, Any] = {
        "model": engine.model_id or engine.model_path or "unknown",
        "task": engine.task or "unknown",
    }

    if schema:
        result["inputs"] = [
            {
                "name": f.name,
                "type": f.type,
                "required": f.required,
                "description": f.description,
                **({"default": f.default} if f.default is not None else {}),
            }
            for f in schema
        ]
    else:
        result["inputs"] = []

    result["parameters"] = params or []

    example = _build_example_command(engine.model_path or "MODEL", schema, params, task=engine.task)
    if example:
        result["example"] = example

    return result


def _schema_to_text(engine: Any) -> str:
    """Build schema as human-readable text."""
    lines: list[str] = []
    lines.append("")
    lines.append(f"Model: {engine.model_id or engine.model_path or 'unknown'}")
    lines.append(f"Task:  {engine.task or 'unknown'}")
    lines.append("")

    schema = engine.user_input_schema
    if schema:
        lines.append("Inputs (--input / -I):")
        for f in schema:
            if f.required:
                req = "(required)"
            elif f.default is not None:
                req = f"(default: {f.default})"
            else:
                req = "(optional)"
            lines.append(f"  {f.name:<16s} {f.type:<8s} {req}  {f.description}")
    else:
        lines.append("Inputs: (no schema — pass inputs directly)")

    lines.append("")
    params = engine.pipeline_params
    if params:
        lines.append("Parameters (-P):")
        for p in params:
            default_str = f"(default: {p['default']})" if "default" in p else ""
            lines.append(f"  {p['name']:<16s} {p['type']:<8s} {default_str}")
    else:
        lines.append("Parameters: (none discovered)")

    example = _build_example_command(engine.model_path or "MODEL", schema, params, task=engine.task)
    if example:
        lines.append("")
        lines.append("Example:")
        lines.append(f"  {example}")
    lines.append("")
    return "\n".join(lines)


# Placeholder values for --schema example, keyed by InputField.type
_EXAMPLE_VALUES: dict[str, str] = {
    "image": "@image.jpg",
    "audio": "@audio.wav",
    "video": "@video.mp4",
    "text": '"hello world"',
    "json": '\'["a","b"]\'',
    "number": "0.5",
    "boolean": "true",
}


def _build_example_command(
    model_path: str,
    schema: list | None,
    params: list[dict] | None,
    task: str | None = None,
) -> str | None:
    """Build a ready-to-copy example command from schema + params."""
    if not schema:
        return None

    from ..inference.tasks import BINARY_TYPES

    parts = [f"winml run --model {model_path}"]
    if task:
        parts.append(f"--task {task}")

    # Try shortcuts first (--file / --text) for simple models
    binary_fields = [f for f in schema if f.type in BINARY_TYPES and f.required]
    text_fields = [f for f in schema if f.type == "text" and f.required]
    other_fields = [
        f for f in schema if f.required and f.type not in BINARY_TYPES and f.type != "text"
    ]

    used_shortcut_names: set[str] = set()
    if len(binary_fields) == 1 and not other_fields:
        # Single binary → --file shortcut
        parts.append(f"--file {_EXAMPLE_VALUES[binary_fields[0].type][1:]}")  # strip @
        used_shortcut_names.add(binary_fields[0].name)
    if len(text_fields) == 1 and not other_fields:
        # Single text → --text shortcut
        parts.append(f"--text {_EXAMPLE_VALUES['text']}")
        used_shortcut_names.add(text_fields[0].name)

    # Remaining required inputs via --input
    for f in schema:
        if not f.required or f.name in used_shortcut_names:
            continue
        val = _EXAMPLE_VALUES.get(f.type, '"..."')
        parts.append(f"-I {f.name}={val}")

    # Add a representative -P param with a sample value
    if params:
        for p in params:
            sample = p.get("sample_value")
            if sample is not None:
                parts.append(f"-P {p['name']}={sample}")
                break

    return " ".join(parts)


def _print_input_hint(engine: Any) -> None:
    """Print available inputs as a hint when no inputs are provided."""
    click.echo()
    click.echo(f"Model: {engine.model_id or 'unknown'}")
    click.echo(f"Task:  {engine.task or 'unknown'}")
    click.echo()

    schema = engine.user_input_schema
    if schema:
        click.echo("Inputs (--input / -I):")
        for f in schema:
            if f.required:
                req = "(required)"
            elif f.default is not None:
                req = f"(default: {f.default})"
            else:
                req = "(optional)"
            click.echo(f"  {f.name:<16s} {f.type:<8s} {req}  {f.description}")
        click.echo()
        click.echo(
            "Hint: run with --input to start inference, "
            "or --schema to also show pipeline parameters (-P)."
        )
    else:
        click.echo("Error: provide at least --input. Use --schema to see available inputs.")
    click.echo()


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@click.command("run")
@cli_utils.model_option(required=True)
@click.option(
    "--file",
    "-f",
    "files",
    multiple=True,
    help="Input media file: image, audio, or video (shortcut for single-file models)",
)
@click.option(
    "--text",
    "-t",
    default=None,
    help="Text input (shortcut for single-text models)",
)
@click.option(
    "--input",
    "-I",
    "input_args",
    multiple=True,
    help="Named input as NAME=VALUE (repeatable). "
    "File inputs: -I image=@photo.jpg  "
    'Text inputs: -I question="Who?"  '
    'JSON inputs: -I labels=\'["a","b"]\'',
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
    "--schema",
    "show_schema",
    is_flag=True,
    default=False,
    help="Print model schema (inputs + parameters) and exit",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format",
)
@cli_utils.output_option("Write output to file instead of stdout")
@click.option(
    "--port",
    default=_DEFAULT_PORT,
    type=int,
    show_default=True,
    help="Port for auto-connect to a running winml serve instance",
)
@click.option(
    "--host",
    "connect_host",
    default="127.0.0.1",
    show_default=True,
    help="Host for auto-connect (use with --connect for remote servers)",
)
@click.option(
    "--connect",
    is_flag=True,
    default=False,
    help="Auto-connect to a running winml serve instance instead of embedded inference",
)
@cli_utils.skip_build_option()
@click.pass_context
def run(
    ctx: click.Context,
    model: str,
    files: tuple[str, ...],
    text: str | None,
    input_args: tuple[str, ...],
    task: str | None,
    device: str,
    ep: EPNameOrAlias | None,
    params: tuple[str, ...],
    show_schema: bool,
    output_format: str,
    output: Path | None,
    port: int,
    connect_host: str,
    connect: bool,
    skip_build: bool,
) -> None:
    r"""Run one-shot inference on a model.

    Uses embedded inference by default. Pass ``--connect`` to route
    through a running ``winml serve`` instance instead.

    Examples:
    \b
        # Image classification (shortcut)
        winml run --model microsoft/resnet-50 --file cat.jpg

        # Named inputs (any model)
        winml run --model roberta-qa -I question="Who?" -I context="Tim Cook is..."

        # Mixed: shortcut + named input
        winml run --model vilt --file photo.jpg -I question="What color?"

        # Schema discovery
        winml run --model ./build/roberta-qa/ --schema

        # Extra pipeline parameters
        winml run --model model --text "Once upon" -P max_new_tokens=100
    """
    if ctx.obj and ctx.obj.get("debug"):
        logging.getLogger("winml.modelkit").setLevel(logging.DEBUG)

    # Parse -P/--param entries
    pipeline_kwargs: dict[str, Any] = {}
    for p in params:
        if "=" not in p:
            click.echo(f"Error: invalid --param format: '{p}'. Use KEY=VALUE.", err=True)
            ctx.exit(2)
        k, v = p.split("=", 1)
        pipeline_kwargs[k] = _parse_param_value(v)

    # Parse --input entries (raw strings, coerced after model load)
    raw_inputs: dict[str, str] = {}
    for inp in input_args:
        if "=" not in inp:
            click.echo(f"Error: invalid --input format: '{inp}'. Use NAME=VALUE.", err=True)
            ctx.exit(2)
        k, v = inp.split("=", 1)
        raw_inputs[k] = v

    # Read file bytes (for --file shortcut)
    file_bytes_list: list[bytes] = []
    for fp in files:
        file_path = Path(fp)
        if not file_path.exists() or not file_path.is_file():
            click.echo(f"Error: file not found: {fp}", err=True)
            ctx.exit(2)
        file_bytes_list.append(file_path.read_bytes())

    if len(file_bytes_list) > 1:
        click.echo(
            f"Error: --file accepts only one file (got {len(file_bytes_list)}). "
            "Use --input for multiple file inputs (e.g. -I image_0=@a.jpg -I image_1=@b.jpg).",
            err=True,
        )
        ctx.exit(2)

    # Check if any input was provided
    has_inputs = bool(file_bytes_list) or text is not None or bool(raw_inputs)

    # ------------------------------------------------------------------
    # Try auto-connect to running winml serve
    # ------------------------------------------------------------------
    # --schema always uses the embedded path (needs local engine for
    # param discovery), so skip auto-connect when schema is requested.
    if connect and has_inputs and not show_schema:
        result = _try_server_predict(
            host=connect_host,
            port=port,
            model_path=model,
            file_paths=files,
            text=text,
            raw_inputs=raw_inputs,
            pipeline_kwargs=pipeline_kwargs,
        )
        if result is not None:
            _print_result(result, output_format=output_format, output_path=output)
            return

    if connect and not has_inputs and not show_schema:
        click.echo(
            "Warning: --connect ignored — no inputs provided. "
            "Add --text, --file, or --input to run inference via the server.",
            err=True,
        )

    # ------------------------------------------------------------------
    # Embedded inference
    # ------------------------------------------------------------------
    from ..inference import InferenceEngine

    engine = InferenceEngine()

    # --schema: lightweight load (no model build / ORT session) and exit
    if show_schema:
        try:
            engine.load_schema_only(model, task=task, device=device, ep=ep)
        except (OSError, ValueError, RuntimeError) as exc:
            click.echo(f"Error loading model: {exc}", err=True)
            ctx.exit(3)
        _print_schema(engine, output_format=output_format, output_path=output)
        return

    try:
        # Redirect stdout → stderr during model load so that build-pipeline
        # prints (from optimum, onnxruntime, etc.) don't contaminate
        # structured output (--format json) or text output parsing.
        with contextlib.redirect_stdout(sys.stderr):
            engine.load(model, task=task, device=device, ep=ep, skip_build=skip_build)
    except (OSError, ValueError, RuntimeError) as exc:
        click.echo(f"Error loading model: {exc}", err=True)
        ctx.exit(3)

    # No inputs: print hint and exit
    if not has_inputs:
        _print_input_hint(engine)
        ctx.exit(0)

    # Coerce --input values based on schema types
    schema = engine.user_input_schema
    try:
        coerced_inputs = _coerce_inputs(raw_inputs, schema)
    except click.ClickException as exc:
        click.echo(f"Error: {exc.format_message()}", err=True)
        ctx.exit(2)

    # Merge --file/--text shortcuts with --input
    try:
        inputs = _resolve_shortcuts(file_bytes_list, text, coerced_inputs, schema)
    except click.ClickException as exc:
        click.echo(f"Error: {exc.format_message()}", err=True)
        ctx.exit(2)

    # Check input / -P collision (after shortcuts are resolved so that
    # --file and --text shortcut keys are included in the check)
    collision = set(inputs.keys()) & set(pipeline_kwargs.keys())
    if collision:
        key = sorted(collision)[0]
        click.echo(
            f"Error: '{key}' specified as both input and -P. "
            f"Use --input for model inputs and -P for pipeline parameters.",
            err=True,
        )
        ctx.exit(2)

    try:
        prediction = engine.predict(inputs=inputs, **pipeline_kwargs)
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        click.echo(f"Error during inference: {exc}", err=True)
        ctx.exit(4)

    _print_result(prediction.model_dump(), output_format=output_format, output_path=output)


# ---------------------------------------------------------------------------
# Auto-connect helpers
# ---------------------------------------------------------------------------


def _resolve_text_field_via_schema(client: Any, base_url: str) -> str:
    """Probe ``GET /v1/schema`` to find the correct field name for ``--text``.

    Falls back to ``"text"`` when the schema is unavailable or the task has
    no text-type input.  This mirrors the embedded-path logic in
    ``_resolve_shortcuts`` so that ``--connect --text`` works for tasks
    whose text field is not named ``"text"`` (e.g. question-answering
    expects ``"question"``).
    """
    try:
        resp = client.get(f"{base_url}/v1/schema", timeout=2)
        if resp.status_code == 200:
            schema = resp.json()
            user_inputs = schema.get("user_inputs", [])
            text_fields = [f for f in user_inputs if f.get("type") == "text"]
            if len(text_fields) == 1:
                return str(text_fields[0]["name"])
    except Exception:
        logger.debug("Schema probe failed; falling back to field name 'text'", exc_info=True)
    return "text"


def _try_server_predict(
    *,
    host: str,
    port: int,
    model_path: str,
    file_paths: tuple[str, ...],
    text: str | None,
    raw_inputs: dict[str, str],
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

    base_url = f"http://{host}:{port}"
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

            # Route 1: file present → multipart /v1/predict/file
            #   The server resolves the correct schema field name via
            #   _build_file_inputs, so we don't need to know it here.
            #   The endpoint supports optional `text` and `inputs` form
            #   fields for multimodal and zero-shot tasks.
            if file_paths:
                fp = Path(file_paths[0])
                form_data: dict[str, str] = {
                    "params": json.dumps(pipeline_kwargs),
                }
                if text is not None:
                    form_data["text"] = text
                if raw_inputs:
                    form_data["inputs"] = json.dumps(raw_inputs)
                with fp.open("rb") as f:
                    resp = client.post(
                        f"{base_url}/v1/predict/file",
                        files={"file": (fp.name, f, "application/octet-stream")},
                        data=form_data,
                        timeout=60,
                    )
                resp.raise_for_status()
                logger.debug("Auto-connected to winml serve at %s", base_url)
                return cast("dict[Any, Any]", resp.json())

            # Route 2: no file → JSON /v1/predict with named inputs
            #   Coerce raw CLI strings (JSON arrays, numbers, booleans)
            #   so the server receives properly typed values.
            inputs: dict[str, Any] = {}
            inputs.update({k: _parse_heuristic(v) for k, v in raw_inputs.items()})
            if text is not None:
                # Resolve the correct field name via the server schema so
                # that --text works for tasks whose text field is not named
                # "text" (e.g. question-answering expects "question").
                text_field_name = _resolve_text_field_via_schema(client, base_url)
                inputs[text_field_name] = text

            payload: dict[str, Any] = {"inputs": inputs, "params": pipeline_kwargs}
            resp = client.post(
                f"{base_url}/v1/predict",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            logger.debug("Auto-connected to winml serve at %s", base_url)
            return cast("dict[Any, Any]", resp.json())
    except (httpx.HTTPError, OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.debug("Auto-connect failed (%s) — using embedded inference", exc)
        return None


def _models_match(server_model: str, requested: str) -> bool:
    """Loose comparison: match on base name or exact string."""
    if not server_model:
        return False
    if server_model == requested:
        return True
    if Path(server_model).name == Path(requested).name:
        logger.warning(
            "Auto-connect: basename match '%s' ≈ '%s' (org may differ)",
            server_model,
            requested,
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _print_result(
    result: dict,
    *,
    output_format: str,
    output_path: Path | None,
) -> None:
    if output_format == "json":
        text = json.dumps(result, indent=2)
    else:
        # Strip base64 mask data from text display — too large for terminal.
        # Work on a shallow copy to avoid mutating the caller's dict.
        import copy

        display_result = copy.copy(result)
        preds = display_result.get("predictions")
        if isinstance(preds, list):
            display_result["predictions"] = [
                {k: v for k, v in p.items() if k != "mask"} if isinstance(p, dict) else p
                for p in preds
            ]
        text = _format_text(display_result)

    if output_path:
        output_path.write_text(text, encoding="utf-8")
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
    is_segmentation = task in {"image-segmentation", "semantic-segmentation"}
    if isinstance(predictions, list) and predictions and isinstance(predictions[0], dict):
        # Classification-style: list of {label, score, ...}
        has_scores = "score" in predictions[0]
        lines.append("Results:" if not is_segmentation else "Results (area coverage):")
        for i, p in enumerate(predictions, 1):
            label = p.get("label", str(i))
            if has_scores:
                score = p.get("score")
                if score is not None:
                    score_str = f"{score:5.1%}" if is_segmentation else f"{score:.4f}"
                else:
                    score_str = "—"
                lines.append(f"  {i:2d}. {label:<30s} {score_str}")
            else:
                lines.append(f"  {i:2d}. {p}")
    elif isinstance(predictions, dict):
        lines.append("Output:")
        for k, v in predictions.items():
            lines.append(f"  {k}: {v}")
    else:
        lines.append(f"Output: {predictions}")

    lines.append("")
    lines.append(f"Latency: {latency:.1f}ms")
    return "\n".join(lines)
