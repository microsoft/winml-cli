# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Phase 0: CLI-as-API FastAPI application.

Wraps every winml CLI command as a REST endpoint using Click's CliRunner.
Each request is stateless — the model loads, runs, and unloads within a
single CliRunner.invoke() call.

Response always includes a structured ``result`` field (parsed JSON) when
the command supports JSON output, in addition to raw ``stdout`` for debugging.

Usage:
    winml serve                   # launches this app on port 8000
    GET  http://localhost:8000/docs        # Swagger UI
    GET  http://localhost:8000/v1/health
    POST http://localhost:8000/v1/cli/sys
    POST http://localhost:8000/v1/cli/perf
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __version__
from ..cli import main as winml_cli


# ---------------------------------------------------------------------------
# Semaphore groups
# Heavy commands load large models — serialize them to avoid OOM.
# Light commands are cheap — allow moderate concurrency.
# ---------------------------------------------------------------------------
_HEAVY_COMMANDS = {"build", "perf", "compile", "quantize", "optimize"}
_LIGHT_COMMANDS = {"analyze", "config", "export", "inspect", "sys"}
_ALL_COMMANDS = _HEAVY_COMMANDS | _LIGHT_COMMANDS

_heavy_semaphore = asyncio.Semaphore(1)
_light_semaphore = asyncio.Semaphore(2)  # inspect can trigger AutoProcessor — limit concurrency
_SEMAPHORE_TIMEOUT_SEC = 120
_EXEC_TIMEOUT_SEC = 600  # max time for a single CLI command execution

_start_time = time.time()

# ---------------------------------------------------------------------------
# JSON extraction strategies
#
# Each strategy describes how to obtain structured JSON from a command:
#
#   "format_flag"  — inject --format json (if user didn't already set it)
#   "stdout"       — stdout is always JSON; parse directly
#   "output_file"  — inject --output <tempfile>; read file after invocation
#   None           — no JSON output; result will be null
# ---------------------------------------------------------------------------
_JSON_STRATEGY: dict[str, str | None] = {
    "sys": "format_flag",  # --format [text|json|compact]
    "inspect": "format_flag",  # --format [table|json]
    "config": "stdout",  # always prints JSON to stdout
    "analyze": "output_file",  # --output PATH writes JSON
    "perf": "output_file",  # --output PATH writes JSON
    "build": None,
    "compile": None,
    "export": None,
    "optimize": None,
    "quantize": None,
}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class CliRequest(BaseModel):
    """Generic CLI invocation request.

    Pass CLI options as a flat dict. Keys use underscores (converted to
    ``--kebab-case`` flags automatically). Boolean ``True`` adds the flag;
    ``False`` omits it.

    Examples::

        {"model": "model.onnx", "ep": "qnn", "verbose": True}
        → winml analyze --model model.onnx --ep qnn --verbose

        {"model": "m.onnx", "no_quant": True, "output_dir": "out"}
        → winml build --model m.onnx --no-quant --output-dir out
    """

    args: dict[str, Any] = {}


class CliResponse(BaseModel):
    """Result of a CLI command invocation.

    ``result`` contains the parsed JSON output when the command supports it
    (sys, inspect, config, analyze, perf). For other commands it is ``null``.
    ``stdout`` always contains the raw text output for debugging.
    """

    command: str
    exit_code: int
    result: dict[str, Any] | list[Any] | None
    stdout: str
    stderr: str
    duration_ms: float


class HealthResponse(BaseModel):
    """Server health and capability information."""

    status: str
    version: str
    mode: str
    uptime_sec: float
    commands: list[str]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WinML CLI API",
    description=(
        "Phase 0 — CLI Wrapper. Each endpoint invokes a `winml` CLI command and "
        "returns structured JSON in `result` (where supported) plus raw "
        "`stdout`/`stderr`. Use `POST /v1/cli/{command}` with a JSON body "
        '`{"args": {"option": "value", ...}}`.'
    ),
    version=__version__,
)

# Permissive CORS for local dev server; no credentials to protect.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    @app.get("/demo", include_in_schema=False)
    async def demo_ui() -> FileResponse:
        """Serve the demo UI."""
        return FileResponse(str(_static_dir / "index.html"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _args_to_flags(args: dict[str, Any]) -> list[str]:
    """Convert a JSON args dict to a Click-compatible CLI flag list.

    Rules:
    - Keys are converted from snake_case to --kebab-case.
    - bool True  → adds the flag (e.g. ``--verbose``)
    - bool False → omits the flag entirely
    - None       → omits the flag entirely
    - Other      → ``--flag value``
    """
    flags: list[str] = []
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                flags.append(flag)
        elif value is not None:
            flags.extend([flag, str(value)])
    return flags


def _extract_json_from_stdout(stdout: str) -> dict[str, Any] | list[Any] | None:
    """Try to parse the last JSON object/array from stdout.

    Some commands print rich text before JSON.  We search backwards for
    the outermost ``{``/``[`` that, together with a matching closing
    delimiter at or after it, forms valid JSON.  This avoids the edge
    case where ``rfind('{')`` and ``rfind('}')`` pick delimiters from
    different JSON fragments.
    """
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        # Walk backwards through every occurrence of end_char
        pos = len(stdout)
        while True:
            end = stdout.rfind(end_char, 0, pos)
            if end == -1:
                break
            # Try every start_char from rightmost to leftmost up to `end`
            start = end
            while True:
                start = stdout.rfind(start_char, 0, start)
                if start == -1:
                    break
                try:
                    return cast(
                        "dict[str, Any] | list[Any] | None",
                        json.loads(stdout[start : end + 1]),
                    )
                except json.JSONDecodeError:
                    continue
            pos = end  # try next (earlier) end_char
    return None


def _invoke(command: str, args: dict[str, Any]) -> CliResponse:
    """Invoke a winml CLI command via CliRunner and return a CliResponse.

    Applies the per-command JSON extraction strategy to populate ``result``.
    """
    from click.testing import CliRunner

    strategy = _JSON_STRATEGY.get(command)
    effective_args = dict(args)
    tmp_output: Path | None = None

    # Prepare args based on strategy
    if strategy == "format_flag" and "format" not in effective_args:
        effective_args["format"] = "json"

    elif strategy == "output_file" and "output" not in effective_args:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
            tmp_output = Path(tmp_file.name)
        effective_args["output"] = str(tmp_output)

    runner = CliRunner()
    cli_args = [command, *_args_to_flags(effective_args)]

    t0 = time.perf_counter()
    result = runner.invoke(winml_cli, cli_args, catch_exceptions=True)
    duration_ms = (time.perf_counter() - t0) * 1000

    # Build stderr from unhandled exceptions
    stderr = ""
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        import traceback

        stderr = "".join(
            traceback.format_exception(
                type(result.exception),
                result.exception,
                result.exception.__traceback__,
            )
        )

    # Extract structured JSON result
    parsed: dict[str, Any] | list[Any] | None = None
    if result.exit_code == 0:
        if strategy in ("format_flag", "stdout"):
            parsed = _extract_json_from_stdout(result.output or "")
        elif strategy == "output_file" and tmp_output is not None and tmp_output.exists():
            try:
                parsed = json.loads(tmp_output.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                parsed = None

    # Clean up temp file
    if tmp_output is not None and tmp_output.exists():
        tmp_output.unlink(missing_ok=True)

    return CliResponse(
        command=command,
        exit_code=result.exit_code,
        result=parsed,
        stdout=result.output or "",
        stderr=stderr,
        duration_ms=round(duration_ms, 2),
    )


async def _run_with_semaphore(command: str, args: dict[str, Any]) -> CliResponse:
    """Acquire the appropriate semaphore and invoke the command."""
    sem = _heavy_semaphore if command in _HEAVY_COMMANDS else _light_semaphore
    try:
        await asyncio.wait_for(sem.acquire(), timeout=_SEMAPHORE_TIMEOUT_SEC)
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Server busy. Command '{command}' is waiting for a slot. Try again later.",
        ) from exc
    try:
        # CliRunner is synchronous — run in executor to avoid blocking event loop
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, _invoke, command, args),
            timeout=_EXEC_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Command '{command}' timed out after {_EXEC_TIMEOUT_SEC}s.",
        ) from exc
    finally:
        sem.release()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/v1/health", response_model=HealthResponse, tags=["Meta"])
async def health() -> HealthResponse:
    """Server health check."""
    return HealthResponse(
        status="ok",
        version=__version__,
        mode="cli-wrapper",
        uptime_sec=round(time.time() - _start_time, 1),
        commands=sorted(_ALL_COMMANDS),
    )


@app.post(
    "/v1/cli/{command}",
    response_model=CliResponse,
    tags=["CLI"],
    summary="Invoke a winml CLI command",
    responses={
        200: {"description": "Command invoked (check exit_code for success/failure)"},
        404: {"description": "Unknown command name"},
        503: {"description": "Server busy — heavy command slot is occupied"},
    },
)
async def run_cli_command(command: str, body: CliRequest) -> CliResponse:
    """Invoke any `winml` CLI command over HTTP.

    The `args` dict maps directly to CLI options:

    ```json
    POST /v1/cli/analyze
    {"args": {"model": "./model.onnx", "ep": "qnn", "verbose": true}}
    ```

    is equivalent to `winml analyze --model ./model.onnx --ep qnn --verbose`.

    **Structured result**: Commands that produce JSON output (sys, inspect,
    config, analyze, perf) automatically return parsed data in ``result``.
    All other commands return ``result: null`` with raw text in ``stdout``.

    **HTTP status**: always 200. Check ``exit_code`` (0 = success).
    Failures are reported in ``stderr`` and ``exit_code != 0``.
    """
    if command not in _ALL_COMMANDS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown command '{command}'. Valid commands: {sorted(_ALL_COMMANDS)}",
        )
    return await _run_with_semaphore(command, body.args)


# ---------------------------------------------------------------------------
# Startup banner (called from winml serve)
# ---------------------------------------------------------------------------


def print_startup_banner(host: str, port: int) -> None:  # noqa: D103
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print()
    console.print(
        Panel.fit(
            f"[bold]WinML CLI API Server[/bold]\n"
            f"[dim]Mode:[/dim]    CLI Wrapper (Phase 0)\n"
            f"[dim]Version:[/dim] {__version__}\n\n"
            f"[dim]API:[/dim]     [link]http://{host}:{port}[/link]\n"
            f"[dim]Docs:[/dim]    [link]http://{host}:{port}/docs[/link]\n\n"
            f"[dim]Commands:[/dim] {', '.join(sorted(_ALL_COMMANDS))}\n\n"
            f"[dim]Ready. Press Ctrl+C to stop.[/dim]",
            border_style="blue",
        )
    )
    console.print()
