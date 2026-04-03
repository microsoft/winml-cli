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


@click.group()
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


def _discover_commands() -> None:
    """Auto-discover Click commands from commands/ directory.

    This function scans the commands/ directory for Python modules and
    registers any Click commands found. Commands are registered using
    the module filename as the command name.

    Command Discovery Rules:
    - Skips files starting with underscore (_)
    - Looks for any object that is a click.Command instance
    - Uses module filename (without .py) as command name
    """
    commands_dir = Path(__file__).parent / "commands"

    # Early exit if commands directory doesn't exist
    if not commands_dir.exists():
        logger.debug("Commands directory not found: %s", commands_dir)
        return

    # Scan for Python modules
    for py_file in commands_dir.glob("*.py"):
        # Skip private modules
        if py_file.name.startswith("_"):
            continue

        module_name = py_file.stem
        try:
            # Import the module
            module = import_module(
                f".commands.{module_name}",
                package=__package__,
            )

            # Find Click command in module
            # Prefer click.Group over click.Command for hierarchical commands
            discovered_command = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, click.Group):
                    discovered_command = attr
                    break
                if isinstance(attr, click.Command) and discovered_command is None:
                    discovered_command = attr

            if discovered_command:
                # Register command with module name
                main.add_command(discovered_command, name=module_name)
                logger.debug("Discovered command: %s", module_name)

        except ImportError as e:
            logger.warning("Failed to import command module %s: %s", module_name, e)
        except Exception as e:
            logger.error("Error loading command %s: %s", module_name, e)


# Discover and register commands at module load time
_discover_commands()


if __name__ == "__main__":
    main()
