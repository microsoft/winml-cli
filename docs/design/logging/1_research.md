# Logging System Design Research

**Module**: `modelkit` CLI (`wmk`)
**Date**: 2026-03-16
**Status**: Research

---

## 1. Problem Statement

The current `wmk` CLI uses a binary `--debug` flag that toggles between `INFO` and `DEBUG`. This is non-standard, lacks granularity, and doesn't follow established Python CLI conventions. We need a proper verbosity system that:

- Follows POSIX and Python CLI conventions (`-v`, `-vv`, `-q`)
- Integrates cleanly with Python's `logging` module
- Works well with Click
- Scales across all `modelkit` submodules

### Current State (cli.py)

```python
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging")
def main(ctx, debug):
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
```

**Problems**: Default is `INFO` (too noisy for normal use), no `-v`/`-q` convention, no intermediate verbosity, debug format used at all levels.

---

## 2. Standard Verbosity Conventions

### 2.1 De Facto Standard Mapping

The universally accepted convention for CLI tools maps repeated `-v` flags to Python logging levels:

| Flags | Verbosity | Logging Level | Numeric | Purpose |
|-------|-----------|---------------|---------|---------|
| `-q` / `--quiet` | -1 | `ERROR` | 40 | Errors only |
| *(default)* | 0 | `WARNING` | 30 | Warnings and errors |
| `-v` | 1 | `INFO` | 20 | Informational progress |
| `-vv` | 2 | `DEBUG` | 10 | Full debug output |

**The formula**: `log_level = 30 - (verbosity * 10)`, clamped to `[10, 40]`.

This is derived from Python's logging level numbering (DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50) and the fact that they are spaced exactly 10 apart.

### 2.2 Default Level: WARNING, Not INFO

The default (no flags) should be `WARNING`. This is the Python `logging` module's own default and the POSIX expectation: a well-behaved CLI should be silent on success, reporting only warnings and errors. Users who want progress feedback explicitly opt in with `-v`.

### 2.3 Extended Verbosity (Optional)

Some tools support three or more `-v` levels. Two approaches:

**Approach A: Custom VERBOSE level (pip)**
pip defines a custom `VERBOSE = 15` level between DEBUG and INFO:

| Flags | Level |
|-------|-------|
| *(default)* | WARNING (30) |
| `-v` | VERBOSE (15) |
| `-vv` | DEBUG (10) |

**Approach B: Ansible's 5-level system**
Ansible supports `-v` through `-vvvvv`, each level revealing more detail (task results, input params, connection details, SSH protocol dumps).

**Recommendation**: For ModelKit, the standard 4-level system (`-q`, default, `-v`, `-vv`) is sufficient. Two verbose levels plus a quiet mode covers all practical needs. Avoid custom log levels unless there is a demonstrated need.

---

## 3. How Popular Python CLIs Handle This

### 3.1 pip

- **Flags**: `-v` (additive, up to 3x), `-q` (additive, up to 3x)
- **Levels**: Custom `VERBOSE=15` between DEBUG and INFO
- `-v` shows subprocess output; `-vv` shows full DEBUG; `-vvv` same as `-vv` (capped)
- `-q` reduces to ERROR; `-qq` to CRITICAL; `-qqq` to silent
- **Takeaway**: Additive quiet is a nice touch but adds complexity. Custom levels are worth it only at pip's scale.

### 3.2 Ruff

- **Flags**: `--verbose` / `-v`, `--quiet` / `-q`, `--silent` / `-s`
- Three explicit tiers: verbose, quiet (diagnostics only), silent (no output)
- No counting/additive flags
- **Takeaway**: Simple three-tier approach. Clean for tools where "how much debug info" is less important than "show diagnostics or not."

### 3.3 Ansible

- **Flags**: `-v` through `-vvvvv` (5 levels)
- Level 0: task names/status; Level 1: return values; Level 2: input params; Level 3: connection details; Level 4: SSH protocol dumps
- **Takeaway**: Fine-grained verbosity makes sense for complex orchestration tools. Overkill for most CLIs.

### 3.4 HTTPie

- **Flag**: `--verbose` / `-v` (boolean, not counting)
- Toggles display of request headers/body alongside response
- Not a logging-level control but a content-display toggle
- **Takeaway**: Some tools use `-v` as a feature toggle rather than log-level control. Keep these concepts separate.

### 3.5 pytest

- **Flags**: `-v` (additive), `-q` (additive), `--no-header`
- `-v` increases test output detail; `-vv` shows full assertion diffs
- Separate `--log-cli-level` for actual Python logging control
- **Takeaway**: Distinguishes between "output verbosity" and "log level." Worth considering but adds complexity.

### 3.6 uvicorn

- **Flag**: `--log-level` with explicit choices (critical, error, warning, info, debug, trace)
- No `-v`/`-q` shorthand
- Adds custom `TRACE` level below DEBUG
- **Takeaway**: Explicit `--log-level` is useful for server-type tools where users need precise control.

### Summary Table

| Tool | `-v` counting | `-q` flag | Custom levels | Default |
|------|:---:|:---:|:---:|---------|
| pip | Yes (3x) | Yes (3x) | VERBOSE=15 | WARNING |
| ruff | No (boolean) | Yes | No | WARNING |
| ansible | Yes (5x) | No | No | Level 0 |
| httpie | No (boolean) | No | No | Normal |
| pytest | Yes (2x) | Yes (2x) | No | WARNING |
| uvicorn | No | No | TRACE | INFO |

---

## 4. Quiet Flag Design

### 4.1 `-q` and `-v` Interaction

The cleanest pattern shares a single `verbosity` destination:

```python
# argparse version (conceptual)
parser.add_argument('-v', '--verbose', action='count', default=0, dest='verbosity')
parser.add_argument('-q', '--quiet', action='store_const', const=-1, dest='verbosity')
```

With Click, this requires a small callback since `count=True` doesn't natively support negative values:

```python
@click.option('-v', '--verbose', count=True, help="Increase verbosity (-v for info, -vv for debug)")
@click.option('-q', '--quiet', is_flag=True, help="Suppress all output except errors")
def main(verbose, quiet):
    if quiet:
        verbosity = -1
    else:
        verbosity = verbose
```

### 4.2 Mutual Exclusivity

`-v` and `-q` should be mutually exclusive. If both are passed, two strategies:

1. **Last wins** (complex to implement in Click)
2. **`-q` always wins** (simpler, safer -- errors should never be suppressed by accident)
3. **Error out** (strictest, most explicit)

**Recommendation**: `-q` overrides `-v`. Simple, predictable, safe.

---

## 5. Python Logging Best Practices

### 5.1 Module-Level Loggers

Every module should create its own logger:

```python
# modelkit/export/io.py
import logging

logger = logging.getLogger(__name__)
# __name__ == "modelkit.export.io"
```

**Why**: The logging module builds a hierarchy using dot notation. `getLogger("modelkit.export.io")` is a child of `"modelkit.export"`, which is a child of `"modelkit"`. Setting the level on `"modelkit"` propagates to all children. This is the single most important logging pattern.

**Rules**:
- Use `logger = logging.getLogger(__name__)` at module top level
- Never call `logging.basicConfig()` in library/module code -- only in the CLI entry point
- Never use `print()` for diagnostic output; use `logger.info()` / `logger.debug()`
- Use lazy formatting: `logger.debug("Loading %s", model_name)` not `logger.debug(f"Loading {model_name}")`

### 5.2 Configure Only at Entry Point

All logging configuration happens once, in the CLI entry point (`cli.py`):

```python
def _configure_logging(verbosity: int) -> None:
    """Configure root logger based on CLI verbosity."""
    base_level = logging.WARNING  # 30
    level = max(logging.DEBUG, base_level - (verbosity * 10))

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",  # Simple for WARNING default
        stream=sys.stderr,
    )
```

### 5.3 Lazy String Formatting

```python
# GOOD - string formatting only happens if the message will be emitted
logger.debug("Exported %d nodes from %s", count, model_path)

# BAD - f-string is always evaluated, even at WARNING level
logger.debug(f"Exported {count} nodes from {model_path}")

# BAD - .format() is always evaluated
logger.debug("Exported {} nodes from {}".format(count, model_path))
```

This matters for debug messages in hot paths where the formatting cost is non-trivial (e.g., stringifying large objects).

---

## 6. stderr vs stdout

### 6.1 The Rule

| Stream | Content |
|--------|---------|
| **stdout** | Program output (data, results, generated configs) |
| **stderr** | Everything else (logs, progress, warnings, errors) |

**Why**: Users pipe stdout to files or other programs. Mixing logs into stdout breaks pipelines:

```bash
# This must work cleanly:
wmk config --model bert-base-uncased > config.yaml

# Logs go to stderr, config YAML goes to stdout
# User can redirect independently:
wmk export --model bert 2>export.log
```

### 6.2 Implementation

Python's `logging.basicConfig()` defaults to `stderr` -- no configuration needed. But be explicit:

```python
logging.basicConfig(
    level=level,
    format=fmt,
    stream=sys.stderr,  # Explicit is better than implicit
)
```

For `click.echo()` (used for program output), it goes to stdout by default. Use `click.echo(..., err=True)` for diagnostic messages outside the logging system.

---

## 7. Log Format Best Practices

### 7.1 Format by Level

Different verbosity levels warrant different formats:

```python
def _configure_logging(verbosity: int) -> None:
    level = max(logging.DEBUG, logging.WARNING - (verbosity * 10))

    if verbosity >= 2:  # DEBUG
        fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
        datefmt = "%H:%M:%S"
    elif verbosity >= 1:  # INFO
        fmt = "%(levelname)s: %(message)s"
        datefmt = None
    else:  # WARNING (default) and QUIET (ERROR)
        fmt = "%(levelname)s: %(message)s"
        datefmt = None

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, stream=sys.stderr)
```

**Rationale**:
- At WARNING/ERROR: Users want terse output. No timestamps, no module names.
- At INFO: Level prefix helps distinguish info from warnings.
- At DEBUG: Full context (timestamp, module name) is essential for diagnosis.

### 7.2 Format Considerations

- **Timestamps**: Only at DEBUG level. Users running `-vv` are diagnosing timing issues.
- **Module names**: Only at DEBUG level. `modelkit.export.io` tells developers where the message originates.
- **Level names**: Always include. Distinguishes warnings from info from errors.
- **Color**: Consider using `click.style()` or a custom formatter for colored level names in terminal output. Not essential for v1.

---

## 8. Click Integration Patterns

### 8.1 Pattern: count=True with Callback

The most Pythonic Click pattern for verbosity:

```python
import logging
import sys

import click


def _configure_logging(verbosity: int) -> None:
    """Set up root logger from CLI verbosity level.

    Mapping:
        -q       -> ERROR   (40)  errors only
        default  -> WARNING (30)  warnings + errors
        -v       -> INFO    (20)  progress messages
        -vv      -> DEBUG   (10)  full diagnostics
    """
    level = max(logging.DEBUG, logging.WARNING - (verbosity * 10))

    if verbosity >= 2:
        fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
        datefmt = "%H:%M:%S"
    else:
        fmt = "%(levelname)s: %(message)s"
        datefmt = None

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        stream=sys.stderr,
    )


@click.group()
@click.version_option(version=__version__, prog_name="wmk")
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v info, -vv debug).")
@click.option("-q", "--quiet", is_flag=True, help="Only show errors.")
@click.pass_context
def main(ctx: click.Context, verbose: int, quiet: bool) -> None:
    """WML ModelKit - Accelerate Model Deployment on WinML."""
    verbosity = -1 if quiet else min(verbose, 2)
    _configure_logging(verbosity)

    ctx.ensure_object(dict)
    ctx.obj["verbosity"] = verbosity
```

### 8.2 Pattern: Environment Variable Override

Support a `WMK_LOG_LEVEL` environment variable for CI/scripting:

```python
import os

def _configure_logging(verbosity: int) -> None:
    env_level = os.environ.get("WMK_LOG_LEVEL")
    if env_level is not None:
        level = getattr(logging, env_level.upper(), None)
        if level is None:
            raise click.BadParameter(f"Invalid WMK_LOG_LEVEL: {env_level}")
    else:
        level = max(logging.DEBUG, logging.WARNING - (verbosity * 10))
    # ... rest of configuration
```

### 8.3 Pattern: Reusable Decorator

For projects with many Click groups, extract the options into a reusable decorator:

```python
import functools


def verbosity_options(func):
    """Add standard -v/-q verbosity options to a Click command."""
    @click.option("-v", "--verbose", count=True, help="Increase verbosity.")
    @click.option("-q", "--quiet", is_flag=True, help="Only show errors.")
    @functools.wraps(func)
    def wrapper(*args, verbose, quiet, **kwargs):
        verbosity = -1 if quiet else min(verbose, 2)
        _configure_logging(verbosity)
        return func(*args, **kwargs)
    return wrapper
```

### 8.4 Anti-Patterns to Avoid

```python
# BAD: Don't use click-log or click-logging libraries
# They are unmaintained (last release 2018/2020), add unnecessary
# dependency, and the stdlib pattern is simple enough.

# BAD: Don't use --log-level with free-text input
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", ...]))
# This is verbose and unfamiliar to most CLI users.

# BAD: Don't configure logging per-subcommand
# Configure once in the group, propagate via context.

# BAD: Don't use logging.root directly
logging.root.setLevel(...)  # Use logging.basicConfig() instead
```

---

## 9. Structured Logging Considerations

### 9.1 When to Use JSON Logging

JSON/structured logging (via `structlog` or stdlib's `logging.handlers`) is appropriate for:
- **Server applications** where logs are consumed by aggregators (ELK, Datadog)
- **Long-running services** where machine-parseable output enables alerting
- **Multi-service architectures** where correlated log analysis is needed

It is **not appropriate** as a default for CLI tools because:
- CLI output is read by humans in a terminal
- JSON is unreadable without `jq` or similar
- CLI invocations are short-lived, not aggregated

### 9.2 If We Ever Need It

Offer a `--log-format json` flag that swaps the formatter:

```python
if log_format == "json":
    import json

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            return json.dumps({
                "ts": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            })

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logging.root.addHandler(handler)
    logging.root.setLevel(level)
```

**Recommendation**: Do not implement JSON logging now. It adds complexity with no current use case. Revisit if ModelKit is used in CI/CD pipelines that need machine-parseable logs.

---

## 10. Third-Party Library Noise

### 10.1 The Problem

At DEBUG level, third-party libraries (transformers, urllib3, onnxruntime) flood stderr with their own debug messages. This makes our debug output unusable.

### 10.2 The Solution

After `basicConfig()`, raise the level for noisy third-party loggers:

```python
def _configure_logging(verbosity: int) -> None:
    level = max(logging.DEBUG, logging.WARNING - (verbosity * 10))

    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)

    # Silence noisy third-party loggers even at DEBUG
    for noisy in ("urllib3", "transformers", "onnxruntime", "filelock"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))
```

This keeps our `modelkit.*` loggers at the requested level while preventing third-party noise. Users who truly need transformers debug output can use `WMK_LOG_LEVEL` or configure those loggers separately.

---

## 11. Recommendation: Specific Pattern for ModelKit

### 11.1 Verbosity Flags

```
wmk [global options] <command> [command options]

Global options:
  -v, --verbose    Increase verbosity (use -vv for maximum detail)
  -q, --quiet      Only show errors
  --version        Show version
  --help           Show help
```

| Invocation | Verbosity | Level | What the user sees |
|------------|-----------|-------|--------------------|
| `wmk -q export ...` | -1 | ERROR | Only errors |
| `wmk export ...` | 0 | WARNING | Warnings and errors |
| `wmk -v export ...` | 1 | INFO | Progress: "Exporting model...", "Config generated" |
| `wmk -vv export ...` | 2 | DEBUG | Full diagnostics: shapes, paths, timing |

### 11.2 Recommended Implementation

```python
"""modelkit/cli.py -- CLI entry point."""
from __future__ import annotations

import logging
import os
import sys
from importlib import import_module
from pathlib import Path

import click

from . import __version__

logger = logging.getLogger(__name__)

_NOISY_LOGGERS = ("urllib3", "transformers", "onnxruntime", "filelock", "PIL")


def _configure_logging(verbosity: int) -> None:
    """Configure root logger based on CLI verbosity.

    Verbosity mapping:
        -1 (quiet)   -> ERROR   (40)
         0 (default) -> WARNING (30)
         1 (-v)      -> INFO    (20)
         2 (-vv)     -> DEBUG   (10)
    """
    # Environment variable override for CI / scripting
    env_level = os.environ.get("WMK_LOG_LEVEL")
    if env_level is not None:
        level = getattr(logging, env_level.upper(), None)
        if level is None:
            click.echo(
                f"WARNING: Invalid WMK_LOG_LEVEL={env_level!r}, ignoring.",
                err=True,
            )
            level = logging.WARNING
    else:
        level = max(logging.DEBUG, logging.WARNING - (verbosity * 10))

    # Format: terse for normal use, detailed for debug
    if level <= logging.DEBUG:
        fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
        datefmt = "%H:%M:%S"
    else:
        fmt = "%(levelname)s: %(message)s"
        datefmt = None

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        stream=sys.stderr,
    )

    # Suppress third-party noise at DEBUG level
    if level <= logging.DEBUG:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)


@click.group()
@click.version_option(version=__version__, prog_name="wmk")
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (-v for info, -vv for debug).",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    default=False,
    help="Only show errors.",
)
@click.pass_context
def main(ctx: click.Context, verbose: int, quiet: bool) -> None:
    """WML ModelKit - Accelerate Model Deployment on WinML.

    Universal ONNX export with QNN and OpenVINO backend support.
    """
    verbosity = -1 if quiet else min(verbose, 2)
    _configure_logging(verbosity)

    ctx.ensure_object(dict)
    ctx.obj["verbosity"] = verbosity
```

### 11.3 Module Logger Convention

Every module in the project should follow this pattern:

```python
# Top of every .py file in modelkit/
import logging

logger = logging.getLogger(__name__)

# Usage:
logger.debug("Shape resolved: %s -> %s", input_name, shape)
logger.info("Exported model to %s", output_path)
logger.warning("Preprocessor config not found, using defaults")
logger.error("Failed to load model: %s", exc)
```

### 11.4 Migration Checklist

1. Replace `--debug` flag with `-v`/`-q` in `cli.py`
2. Change default level from `INFO` to `WARNING`
3. Add `_configure_logging()` helper
4. Add `WMK_LOG_LEVEL` environment variable support
5. Add third-party logger suppression
6. Audit all `print()` calls in modelkit -- convert diagnostics to `logger.*`
7. Audit all `click.echo()` calls -- ensure program output goes to stdout, diagnostics to stderr
8. Add `logger = logging.getLogger(__name__)` to any module that lacks it

### 11.5 What NOT to Do

- **No custom log levels**. The standard 5 levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) are sufficient.
- **No `structlog` or `loguru`**. Zero dependencies for logging; the stdlib is enough.
- **No `click-log` or `click-logging`**. Unmaintained, unnecessary abstraction.
- **No `--log-level` option**. The `-v`/`-q` pattern is simpler and more conventional. The `WMK_LOG_LEVEL` env var covers the power-user case.
- **No JSON output** (for now). Revisit if CI pipeline integration demands it.
- **No per-subcommand logging configuration**. Configure once in the group, propagate via context.

---

## Sources

- [Configuring CLI output verbosity with logging and argparse](https://xahteiwi.eu/resources/hints-and-kinks/python-cli-logging-options/)
- [How to Set Logging Levels via Command Line in Python](https://signoz.io/guides/how-to-set-logging-level-from-command-line/)
- [Click Documentation: Options (count=True)](https://click.palletsprojects.com/en/stable/options/)
- [pip PR #9450: Add VERBOSE log level for -v](https://github.com/pypa/pip/pull/9450)
- [pip CLI Documentation](https://pip.pypa.io/en/stable/cli/pip/)
- [Logging HOWTO -- Python 3.14 documentation](https://docs.python.org/3/howto/logging.html)
- [Logging -- The Hitchhiker's Guide to Python](https://docs.python-guide.org/writing/logging/)
- [How and when to use stdout and stderr?](https://julienharbulot.com/python-cli-streams.html)
- [structlog Logging Best Practices](https://www.structlog.org/en/stable/logging-best-practices.html)
- [Verbosity In Ansible](https://www.builddevops.com/post/verbosity-in-ansible)
- [Ruff Configuration](https://docs.astral.sh/ruff/configuration/)
- [click-log Documentation](https://click-log.readthedocs.io/en/stable/)
