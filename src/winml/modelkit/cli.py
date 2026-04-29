# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML ModelKit CLI - Universal ONNX export from command line.

This module provides the main CLI entry point for ModelKit with lazy
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

import ast
import logging
from importlib import import_module
from pathlib import Path

import click

from . import __version__
from .telemetry import ActionGroup
from .telemetry import telemetry as _telemetry_mod
from .utils.logging import configure_logging


logger = logging.getLogger(__name__)

_COMMANDS_DIR = Path(__file__).parent / "commands"

# Commands that are temporarily disabled from the CLI surface.
# The modules remain on disk so tests and internal imports still work;
# they simply do not appear in ``winml --help`` or accept user invocations.
_DISABLED_COMMANDS: frozenset[str] = frozenset({"run", "serve"})


def _parse_click_help(path: Path) -> str:
    """Extract short help from a command module without importing it.

    Parses the module's AST to find the first decorated function's docstring,
    which Click uses as the command help text.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):
        return ""

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            docstring = ast.get_docstring(node)
            if docstring:
                # Return first line only (Click's short help)
                return docstring.split("\n")[0]
    return ""


class LazyGroup(ActionGroup):
    """Click group that defers command module imports until invoked.

    Instead of importing every command module at startup, this group reads
    command names from the filesystem and only imports a module when the
    user actually invokes that command. Help text is extracted via AST
    parsing (no module execution).

    Extends :class:`ActionGroup` so every resolved subcommand is also
    auto-instrumented with ModelKit telemetry.
    """

    def list_commands(self, ctx: click.Context) -> list[str]:
        """Return command names from filesystem — no module imports."""
        if not _COMMANDS_DIR.exists():
            return []
        return sorted(
            p.stem
            for p in _COMMANDS_DIR.glob("*.py")
            if not p.name.startswith("_") and p.stem not in _DISABLED_COMMANDS
        )

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Import command module only when the command is actually invoked."""
        if cmd_name in _DISABLED_COMMANDS:
            ctx.fail(
                f"'winml {cmd_name}' is currently disabled. "
                f"Use 'winml eval' for model evaluation instead."
            )

        try:
            module = import_module(
                f".commands.{cmd_name}",
                package=__package__,
            )
        except ImportError as e:
            logger.warning("Failed to import command module %s: %s", cmd_name, e)
            return None
        except Exception as e:
            logger.error("Error loading command %s: %s", cmd_name, e)
            return None

        # Find Click command in module (prefer Group over Command)
        discovered = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, click.Group):
                return attr
            if isinstance(attr, click.Command) and discovered is None:
                discovered = attr
        return discovered

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Format command list using AST-parsed help (no module imports)."""
        commands = []
        for cmd_name in self.list_commands(ctx):
            help_text = _parse_click_help(_COMMANDS_DIR / f"{cmd_name}.py")
            commands.append((cmd_name, help_text))

        if commands:
            limit = max(1, formatter.width - 6 - max(len(name) for name, _ in commands))
            rows = []
            for name, help_text in commands:
                short = help_text[:limit].rstrip() if help_text else ""
                rows.append((name, short))

            with formatter.section("Commands"):
                formatter.write_dl(rows)


@click.group(cls=LazyGroup)
@click.version_option(version=__version__, prog_name="winml")
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity (-v=INFO, -vv=DEBUG)",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Quiet mode - errors only",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Alias for -vv (DEBUG logging)",
    hidden=True,
)
@click.pass_context
def main(ctx: click.Context, verbose: int, quiet: bool, debug: bool) -> None:
    """WML ModelKit - Accelerate Model Deployment on WinML.

    Universal ONNX export with QNN and OpenVINO backend support.
    """
    # --debug is a backward-compat alias for -vv
    if debug:
        verbose = max(verbose, 2)

    configure_logging(verbosity=verbose, quiet=quiet)

    # Store verbosity in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug or verbose >= 2
    ctx.obj["verbosity"] = verbose
    ctx.obj["quiet"] = quiet

    ctx.call_on_close(_shutdown_telemetry)


def _shutdown_telemetry() -> None:
    # Only flush if a subcommand actually materialized the singleton.
    # Calling `get_or_init()` here unconditionally would build a fresh
    # Telemetry on the way out — which can trigger first-run consent
    # resolution during process shutdown if the iKey is non-empty.
    instance = _telemetry_mod._INSTANCE
    if instance is None:
        return
    try:
        instance.shutdown()
    except Exception:
        # Telemetry shutdown must never affect the CLI exit code; swallow
        # any error from a half-initialized singleton or transport flush.
        pass


if __name__ == "__main__":
    main()
