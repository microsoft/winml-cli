# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Model serving command for WinML CLI.

Usage:
    winml serve                                             # Phase 0: CLI wrapper
    winml serve --model microsoft/resnet-50                 # Phase 1: HF model
    winml serve --model ./build/resnet50/                   # Phase 1: build output dir
    winml serve --model model.onnx --task image-classification  # Phase 1: raw ONNX
    winml serve --model microsoft/resnet-50 --multi         # Phase 3: multi-model manager
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import click

from ..utils import cli as cli_utils


if TYPE_CHECKING:
    from ..utils.constants import EPNameOrAlias


@click.command("serve")
@cli_utils.model_option(required=False)
@click.option("--task", default=None, help="Task type (required for raw .onnx files)")
@cli_utils.device_option(required=False, default="auto", include_auto=True)
@cli_utils.ep_option(required=False)
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
    help="Unload session after N seconds idle (0 = never)",
)
@click.option(
    "--multi/--no-multi",
    default=False,
    show_default=True,
    help="Enable multi-model manager",
)
@click.option(
    "--memory-budget",
    default=4096.0,
    type=float,
    show_default=True,
    help="Memory budget in MB for multi-model manager",
)
@click.option(
    "--auto-reload/--no-auto-reload",
    default=False,
    show_default=True,
    hidden=True,
    help="Dev mode: auto-reload on file changes",
)
@click.pass_context
def serve(
    ctx: click.Context,
    model: str | None,
    task: str | None,
    device: str,
    ep: EPNameOrAlias | None,
    port: int,
    host: str,
    idle_timeout: float,
    multi: bool,
    memory_budget: float,
    auto_reload: bool,
) -> None:
    r"""Start WinML CLI as a local REST API server.

    Without --model starts in Phase 0 (CLI wrapper mode).
    With --model starts in Phase 1/3 (inference server mode).

    Examples:
    \b
        # Phase 0: all winml commands as HTTP endpoints
        winml serve

        # Phase 1: warm single-model inference
        winml serve --model microsoft/resnet-50

        # Phase 1: from build output directory
        winml serve --model ./build/resnet50/

        # Phase 1: raw ONNX file (task required)
        winml serve --model model.onnx --task image-classification

        # Phase 2: with idle timeout
        winml serve --model microsoft/resnet-50 --idle-timeout 300

        # Phase 3: multi-model
        winml serve --model microsoft/resnet-50 --multi

        # Custom host/port
        winml serve --model microsoft/resnet-50 --host 0.0.0.0 --port 9000
    """
    try:
        import uvicorn
    except ImportError as e:
        raise click.ClickException(
            "uvicorn is required. Install with: pip install uvicorn[standard]"
        ) from e

    if ctx.obj and ctx.obj.get("debug"):
        logging.getLogger("modelkit").setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Phase 0: no model AND not --multi → CLI wrapper
    # ------------------------------------------------------------------
    if model is None and not multi:
        try:
            from ..serve.cli_api import app
            from ..serve.cli_api import print_startup_banner as _banner0
        except ImportError as e:
            raise click.ClickException(f"Failed to load serving module: {e}") from e
        _banner0(host=host, port=port)
        uvicorn.run(app, host=host, port=port, reload=auto_reload, log_level="warning")
        return

    # ------------------------------------------------------------------
    # Phase 1 / 3: model given OR --multi (empty slot manager)
    # ------------------------------------------------------------------
    try:
        from ..serve.app import create_app
        from ..serve.app import print_startup_banner as _banner1
    except ImportError as e:
        raise click.ClickException(f"Failed to load inference serving module: {e}") from e

    mode = "multi" if multi else "single"
    inference_app = create_app(
        model_path=model,
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
        model_path=model,
        task=task,
        device=device,
        ep=ep,
    )

    uvicorn.run(
        inference_app,
        host=host,
        port=port,
        reload=auto_reload,
        log_level="warning",
    )
