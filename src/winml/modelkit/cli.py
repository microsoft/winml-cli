# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML ModelKit CLI - Universal ONNX export from command line.

This module provides the main CLI entry point for ModelKit with automatic
command discovery from the commands/ directory.

Usage:
    winml --version
    winml --help
    winml export --model MODEL --output PATH [--backend BACKEND] [--verbose]

Entry Points:
    - Standalone CLI: winml
    - Module execution: python -m winml.modelkit
"""

from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path

import click

from . import __version__


logger = logging.getLogger(__name__)


class _LazyGroup(click.Group):
    """Click group that discovers and imports commands lazily.

    Command modules under ``commands/`` are imported only when the user
    actually invokes them (or asks for ``--help``).  This avoids pulling
    in heavy ML dependencies (torch, transformers, …) for lightweight
    sub-commands like ``winml sys``.
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self._commands_dir = Path(__file__).parent / "commands"
        # Lazily-discovered module names (without .py, excluding _private)
        self._lazy_names: list[str] | None = None

    # ------------------------------------------------------------------
    def _scan_names(self) -> list[str]:
        if self._lazy_names is None:
            if self._commands_dir.exists():
                self._lazy_names = [
                    f.stem for f in self._commands_dir.glob("*.py") if not f.name.startswith("_")
                ]
            else:
                self._lazy_names = []
        return self._lazy_names

    # ------------------------------------------------------------------
    def list_commands(self, ctx: click.Context) -> list[str]:
        # Merge eagerly-registered commands (if any) with lazy ones
        eager = set(super().list_commands(ctx))
        lazy = set(self._scan_names())
        return sorted(eager | lazy)

    # ------------------------------------------------------------------
    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        # Already registered?
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd

        # Lazy-load from commands/<cmd_name>.py
        if cmd_name not in self._scan_names():
            return None

        try:
            module = import_module(
                f".commands.{cmd_name}",
                package=__package__,
            )
        except ImportError as exc:
            logger.warning("Failed to import command module %s: %s", cmd_name, exc)
            return None
        except Exception as exc:
            logger.error("Error loading command %s: %s", cmd_name, exc)
            return None

        # Find the Click command in the module
        discovered: click.Command | None = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, click.Group):
                discovered = attr
                break
            if isinstance(attr, click.Command) and discovered is None:
                discovered = attr

        if discovered is not None:
            self.add_command(discovered, name=cmd_name)
        return discovered


@click.group(cls=_LazyGroup)
@click.version_option(version=__version__, prog_name="winml")
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug logging",
)
@click.pass_context
def main(ctx: click.Context, debug: bool) -> None:
    """WML ModelKit - Accelerate Model Deployment on WinML.

    Universal ONNX export with QNN and OpenVINO backend support.
    """
    # Configure logging based on debug flag
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Store debug flag in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


if __name__ == "__main__":
    main()
