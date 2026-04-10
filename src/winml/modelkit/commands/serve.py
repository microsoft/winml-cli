# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Model serving command for ModelKit CLI.

Phase 0 (no model arg): CLI Wrapper — all wmk commands as POST /v1/cli/{command}.
Phase 1 (model arg):    Single-model inference server — warm session, <50ms.
Phase 3 (--multi):      Multi-model server — refcount + LRU memory management.

Usage:
    wmk serve                              # Phase 0: CLI wrapper
    wmk serve microsoft/resnet-50         # Phase 1: HF model
    wmk serve ./build/resnet50/           # Phase 1: build output dir
    wmk serve model.onnx --task image-classification  # Phase 1: raw ONNX
    wmk serve microsoft/resnet-50 --multi # Phase 3: multi-model manager
"""

from __future__ import annotations

import logging
import sys

import click


logger = logging.getLogger(__name__)


@click.command("serve")
@click.argument("model_path", required=False, default=None)
@click.option("--task", default=None, help="Task type (required for raw .onnx files)")
@click.option(
    "--device",
    default="auto",
    show_default=True,
    help="Device: auto, cpu, gpu, npu",
)
@click.option("--ep", default=None, help="Explicit execution provider short name")
@click.option(
    "--port",
    default=8000,
    type=int,
    show_default=True,
    help="Port to listen on",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Host address to bind",
)
@click.option(
    "--idle-timeout",
    default=0.0,
    type=float,
    show_default=True,
    help="Phase 2: unload session after N seconds idle (0 = never)",
)
@click.option(
    "--multi",
    is_flag=True,
    default=False,
    help="Phase 3: enable multi-model manager",
)
@click.option(
    "--memory-budget",
    default=4096.0,
    type=float,
    show_default=True,
    help="Phase 3: memory budget in MB for multi-model manager",
)
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    hidden=True,
    help="Dev mode: auto-reload on file changes",
)
@click.pass_context
def serve(
    ctx: click.Context,
    model_path: str | None,
    task: str | None,
    device: str,
    ep: str | None,
    port: int,
    host: str,
    idle_timeout: float,
    multi: bool,
    memory_budget: float,
    reload: bool,
) -> None:
    r"""Start ModelKit as a local REST API server.

    Without MODEL_PATH starts in Phase 0 (CLI wrapper mode).
    With MODEL_PATH starts in Phase 1/3 (inference server mode).

    Examples:
    \b
        # Phase 0: all wmk commands as HTTP endpoints
        wmk serve

        # Phase 1: warm single-model inference
        wmk serve microsoft/resnet-50

        # Phase 1: from build output directory
        wmk serve ./build/resnet50/

        # Phase 1: raw ONNX file (task required)
        wmk serve model.onnx --task image-classification

        # Phase 2: with idle timeout
        wmk serve microsoft/resnet-50 --idle-timeout 300

        # Phase 3: multi-model
        wmk serve microsoft/resnet-50 --multi

        # Custom host/port
        wmk serve microsoft/resnet-50 --host 0.0.0.0 --port 9000
    """
    try:
        import uvicorn
    except ImportError:
        click.echo(
            "Error: uvicorn is required. Install with: pip install uvicorn[standard]",
            err=True,
        )
        sys.exit(1)

    if ctx.obj and ctx.obj.get("debug"):
        logging.getLogger("modelkit").setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Phase 0: no model AND not --multi → CLI wrapper
    # ------------------------------------------------------------------
    if model_path is None and not multi:
        try:
            from ..serve.cli_api import app
            from ..serve.cli_api import print_startup_banner as _banner0
        except ImportError as e:
            click.echo(f"Error: Failed to load serving module: {e}", err=True)
            sys.exit(1)
        _banner0(host=host, port=port)
        uvicorn.run(app, host=host, port=port, reload=reload, log_level="warning")
        return

    # ------------------------------------------------------------------
    # Phase 1 / 3: model given OR --multi (empty slot manager)
    # ------------------------------------------------------------------
    try:
        from ..serve.app import create_app
        from ..serve.app import print_startup_banner as _banner1
    except ImportError as e:
        click.echo(f"Error: Failed to load inference serving module: {e}", err=True)
        sys.exit(1)

    mode = "multi" if multi else "single"
    inference_app = create_app(
        model_path=model_path,
        task=task,
        device=device,
        ep=ep,
        idle_timeout_sec=idle_timeout,
        mode=mode,
        memory_budget_mb=memory_budget,
    )

    _banner1(
        host=host,
        port=port,
        model_path=model_path,
        task=task,
        device=device,
        ep=ep,
    )

    uvicorn.run(
        inference_app,
        host=host,
        port=port,
        reload=reload,
        log_level="warning",
    )
