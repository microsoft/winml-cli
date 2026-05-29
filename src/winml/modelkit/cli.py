# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""WinML CLI - Universal ONNX export from command line.

This module provides the main entry point for WinML CLI with lazy
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
from .utils.cli import verbosity_options
from .utils.logging import configure_logging, flush_ort_startup_logs


logger = logging.getLogger(__name__)

_COMMANDS_DIR = Path(__file__).parent / "commands"

# 5-row block-letter art for "WinML CLI".  '#' = filled pixel, ' ' = empty.
# All letters use the same █ character so identical shapes (i vs I) look
# consistent regardless of horizontal position.
_LETTER_ART: dict[str, list[str]] = {
    "W": ["#   #", "#   #", "# # #", "## ##", "#   #"],
    "i": ["###", " # ", " # ", " # ", "###"],
    "n": ["#   #", "##  #", "# # #", "#  ##", "#   #"],
    "M": ["#   #", "## ##", "# # #", "#   #", "#   #"],
    "L": ["#    ", "#    ", "#    ", "#    ", "#####"],
    "C": ["####", "#   ", "#   ", "#   ", "####"],
    "I": ["###", " # ", " # ", " # ", "###"],
}
# Two word segments; rendered with a wider gap between them.
_SEGMENTS: list[list[str]] = [list("WinML"), list("CLI")]
_LETTER_GAP = "  "  # between letters within a word
_WORD_GAP = "    "  # between words

# Gradient stops (left → right across the full banner width).
_GRADIENT: list[tuple[float, tuple[int, int, int]]] = [
    (0.00, (0, 230, 255)),  # cyan
    (0.25, (0, 100, 255)),  # blue
    (0.50, (130, 0, 255)),  # purple
    (0.75, (255, 0, 180)),  # pink
    (1.00, (255, 80, 80)),  # red
]


def _gradient_color(t: float) -> tuple[int, int, int]:
    for i in range(len(_GRADIENT) - 1):
        t0, c0 = _GRADIENT[i]
        t1, c1 = _GRADIENT[i + 1]
        if t <= t1:
            s = (t - t0) / (t1 - t0)
            return (
                round(c0[0] + s * (c1[0] - c0[0])),
                round(c0[1] + s * (c1[1] - c0[1])),
                round(c0[2] + s * (c1[2] - c0[2])),
            )
    return _GRADIENT[-1][1]


def _print_banner(version: str, *, _console: object | None = None) -> None:
    """Print the WinML CLI gradient banner to stderr using Rich."""
    from rich.console import Console  # lazy import - keeps startup fast
    from rich.text import Text

    # Compute total art width across both word segments.
    art_w = len(_WORD_GAP) * (len(_SEGMENTS) - 1)
    for seg in _SEGMENTS:
        art_w += len(_LETTER_GAP) * (len(seg) - 1)
        art_w += sum(len(_LETTER_ART[ch][0]) for ch in seg)
    bar_w = art_w + 4
    margin = "  "

    con = _console or Console(stderr=True, highlight=False)
    con.print()

    for row_idx in range(5):
        line = Text(margin)
        col = 0
        for seg_idx, seg in enumerate(_SEGMENTS):
            if seg_idx > 0:
                line.append(_WORD_GAP)
                col += len(_WORD_GAP)
            for letter_idx, letter in enumerate(seg):
                if letter_idx > 0:
                    line.append(_LETTER_GAP)
                    col += len(_LETTER_GAP)
                for ch in _LETTER_ART[letter][row_idx]:
                    if ch == "#":
                        r, g, b = _gradient_color(col / max(art_w - 1, 1))
                        line.append("█", style=f"bold rgb({r},{g},{b})")
                    else:
                        line.append(" ")
                    col += 1
        con.print(line)

    con.print()
    bar = Text(margin)
    for i in range(bar_w):
        r, g, b = _gradient_color(i / max(bar_w - 1, 1))
        bar.append("─", style=f"rgb({r},{g},{b})")
    con.print(bar)

    con.print()
    con.print(f"{margin}[bold rgb(160,100,255)]Windows ML  ·  Model Conversion & Optimization[/]")
    con.print(f"{margin}[dim]v{version}  ·  CPU · GPU · NPU[/]")
    con.print()


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
    auto-instrumented with WinML CLI telemetry.
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

    def resolve_command(self, ctx: click.Context, args: list[str]):
        """Seed ``self.commands`` so Click can emit a did-you-mean hint on typos."""
        # Click's NoSuchCommand exception uses self.commands to find suggestions.
        for name in self.list_commands(ctx):
            self.commands.setdefault(name, None)  # type: ignore[arg-type]
        return super().resolve_command(ctx, args)

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Emit banner to stderr, then delegate to normal help formatting."""
        _print_banner(__version__)
        super().format_help(ctx, formatter)

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


@click.group(
    cls=LazyGroup,
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(version=__version__, prog_name="winml")
@verbosity_options
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Alias for -vv (DEBUG logging)",
    hidden=True,
)
@click.pass_context
def main(ctx: click.Context, verbose: int, quiet: bool, debug: bool) -> None:
    """WinML CLI - Accelerate Model Deployment on WinML.

    Universal ONNX export with various WinML execution providers support.
    """
    # --debug is a backward-compat alias for -vv
    if debug:
        verbose = max(verbose, 2)

    configure_logging(verbosity=verbose, quiet=quiet)

    # Replay ORT native stderr captured during onnxruntime import.
    # onnxruntime is imported at module level in constants.py (before configure_logging
    # runs), so any native C++ messages are buffered.  Flushing here — after
    # configure_logging — ensures they are emitted at the correct log level.
    flush_ort_startup_logs()

    # Store verbosity in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug or verbose >= 2
    ctx.obj["verbosity"] = verbose
    ctx.obj["quiet"] = quiet

    ctx.call_on_close(_shutdown_telemetry)

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(0)


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
