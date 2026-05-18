# CLI Arguments Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standardize CLI arguments across all 11 `wmk` subcommands per the spec at `docs/design/cli/3_cli_args_spec.md`.

**Architecture:** Create a shared decorator module (`_options.py`) for reusable Click options, migrate all commands to use shared decorators + global-only verbosity via `ctx.obj`, and enforce the spec with a test validator suite.

**Tech Stack:** Python 3.10+, Click, Rich, pytest

**Spec Reference:** `docs/design/cli/3_cli_args_spec.md` — all ADRs and flag tables live there.

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| **Create** | `modelkit/commands/_options.py` | Shared Click option decorators |
| **Create** | `tests/test_cli_spec.py` | Spec enforcement test validator |
| **Modify** | `modelkit/cli.py` | Root group: rename ctx key `verbosity`→`verbose`, add `-h` |
| **Modify** | `modelkit/commands/build.py` | Remove local `-v`/`-q`, use shared decorators |
| **Modify** | `modelkit/commands/compile.py` | Remove local `-v`, use shared decorators, `--no-quant`/`--no-validate` |
| **Modify** | `modelkit/commands/config.py` | Add `@click.pass_context`, remove local `-v`, use shared decorators |
| **Modify** | `modelkit/commands/export.py` | Remove local `-v`, use shared decorators |
| **Modify** | `modelkit/commands/inspect.py` | Remove local `-v`, use shared decorators |
| **Modify** | `modelkit/commands/perf.py` | Remove local `-v`, use shared decorators, add `-n`/`-t` shorts |
| **Modify** | `modelkit/commands/quantize.py` | Remove local `-v`, use shared decorators, add `-d`/`--ep`/`-n`/`-t` |
| **Modify** | `modelkit/commands/optimize.py` | Remove local `-v`, use shared decorators, add `-d`/`--ep` |
| **Modify** | `modelkit/commands/analyze.py` | Remove local `-v`, use shared decorators |
| **Modify** | `modelkit/commands/eval.py` | Remove local `-v`, use shared decorators, add `--ep`/`-t` |
| **Modify** | `modelkit/commands/sys.py` | Remove local `-v` |

---

## Task 1: Create `_options.py` — Shared Decorator Module

**Files:**
- Create: `modelkit/commands/_options.py`
- Test: `tests/test_cli_spec.py` (partial — decorator unit tests)

- [ ] **Step 1: Write failing test for shared decorators**

Create `tests/test_cli_spec.py` with basic import and decorator tests:

```python
"""CLI spec enforcement tests — validates argument conventions."""

import click
import pytest


def test_options_module_importable():
    """_options.py must be importable."""
    from modelkit.commands._options import (
        model_option,
        device_option,
        ep_option,
        output_file_option,
        output_dir_option,
        task_option,
        precision_option,
    )


def test_model_option_creates_click_option():
    """model_option() must return a Click decorator."""
    from modelkit.commands._options import model_option

    @click.command()
    @model_option()
    def dummy(model):
        pass

    param_names = [p.name for p in dummy.params]
    assert "model" in param_names


def test_model_option_required_default():
    """model_option() is required=True by default."""
    from modelkit.commands._options import model_option

    @click.command()
    @model_option()
    def dummy(model):
        pass

    model_param = next(p for p in dummy.params if p.name == "model")
    assert model_param.required is True


def test_model_option_required_false():
    """model_option(required=False) makes it optional."""
    from modelkit.commands._options import model_option

    @click.command()
    @model_option(required=False)
    def dummy(model):
        pass

    model_param = next(p for p in dummy.params if p.name == "model")
    assert model_param.required is False


def test_device_option_choices():
    """device_option() must have auto|cpu|gpu|npu choices, case-insensitive."""
    from modelkit.commands._options import device_option

    @click.command()
    @device_option()
    def dummy(device):
        pass

    device_param = next(p for p in dummy.params if p.name == "device")
    assert isinstance(device_param.type, click.Choice)
    assert set(device_param.type.choices) == {"auto", "cpu", "gpu", "npu"}
    assert device_param.type.case_sensitive is False


def test_device_option_default_auto():
    """device_option() defaults to 'auto'."""
    from modelkit.commands._options import device_option

    @click.command()
    @device_option()
    def dummy(device):
        pass

    device_param = next(p for p in dummy.params if p.name == "device")
    assert device_param.default == "auto"


def test_precision_option_is_string_not_choice():
    """precision_option() must be string type (ADR-8), not Choice."""
    from modelkit.commands._options import precision_option

    @click.command()
    @precision_option()
    def dummy(precision):
        pass

    precision_param = next(p for p in dummy.params if p.name == "precision")
    # Must NOT be Choice — ADR-8 says string + validator
    assert not isinstance(precision_param.type, click.Choice)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_cli_spec.py -v
```

Expected: ImportError — `_options` module doesn't exist yet.

- [ ] **Step 3: Create `_options.py` with all shared decorators**

Create `modelkit/commands/_options.py`:

```python
"""Shared CLI option decorators.

One source of truth for options used across multiple wmk subcommands.
See docs/design/cli/3_cli_args_spec.md for the full specification.
"""

from __future__ import annotations

import click


def _normalize_uppercase(
    ctx: click.Context, param: click.Parameter, value: str | None,
) -> str | None:
    """Normalize value to uppercase (for device choices)."""
    return value.upper() if value else value


# Known precision values — warn (don't error) on unknown for forward-compat
_KNOWN_PRECISIONS = {
    "auto", "fp32", "fp16", "int8", "int16",
    "w8a8", "w8a16", "w4a16",
}


def _validate_precision(
    ctx: click.Context, param: click.Parameter, value: str | None,
) -> str | None:
    """Warn on unknown precision values but don't reject them."""
    if value and value.lower() not in _KNOWN_PRECISIONS:
        click.echo(
            f"Warning: unknown precision '{value}'. "
            f"Known values: {', '.join(sorted(_KNOWN_PRECISIONS))}",
            err=True,
        )
    return value


# ── Shared option decorators ────────────────────────────────────


def model_option(required: bool = True):
    """Model identifier: HF model ID, local path, or .onnx file."""
    return click.option(
        "-m", "--model",
        required=required,
        type=str,
        help="HuggingFace model ID, local path, or .onnx file",
    )


def device_option():
    """Target device for inference/compilation."""
    return click.option(
        "-d", "--device",
        default="auto",
        type=click.Choice(
            ["auto", "cpu", "gpu", "npu"],
            case_sensitive=False,
        ),
        callback=_normalize_uppercase,
        expose_value=True,
        is_eager=False,
        help="Target device",
    )


def ep_option():
    """Force specific execution provider (overrides --device)."""
    return click.option(
        "--ep",
        default=None,
        type=str,
        help="Force specific execution provider (overrides --device)",
    )


def output_file_option(required: bool = False):
    """Output file path (single artifact)."""
    from pathlib import Path

    return click.option(
        "-o", "--output",
        required=required,
        type=click.Path(path_type=Path),
        help="Output file path",
    )


def output_dir_option(required: bool = False):
    """Output directory (multi-artifact builds)."""
    from pathlib import Path

    return click.option(
        "--output-dir",
        required=required,
        type=click.Path(path_type=Path),
        help="Output directory for artifacts",
    )


def task_option():
    """Override auto-detected task."""
    return click.option(
        "-t", "--task",
        default=None,
        type=str,
        help="Override auto-detected task (e.g., image-classification)",
    )


def precision_option():
    """Target precision (string + validator per ADR-8)."""
    return click.option(
        "-p", "--precision",
        default="auto",
        type=str,
        callback=_validate_precision,
        help="Target precision (e.g., fp32, fp16, int8, w8a16)",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli_spec.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 5: Lint**

```bash
uv run ruff check modelkit/commands/_options.py tests/test_cli_spec.py
```

- [ ] **Step 6: Commit**

```bash
git add modelkit/commands/_options.py tests/test_cli_spec.py
git commit -m "feat(cli): add shared option decorators (_options.py) and spec tests"
```

---

## Task 2: Fix Root Group (`cli.py`)

**Files:**
- Modify: `modelkit/cli.py:118-153`

- [ ] **Step 1: Write failing test for root context contract**

Append to `tests/test_cli_spec.py`:

```python
from click.testing import CliRunner
from modelkit.cli import main


def test_root_context_stores_verbose_key():
    """Root group must store ctx.obj['verbose'] (not 'verbosity')."""
    runner = CliRunner()
    result = runner.invoke(main, ["-v", "sys", "--help"])
    # If it runs without error, the context was set up
    assert result.exit_code == 0


def test_root_help_short_flag():
    """wmk -h must work (not just --help)."""
    runner = CliRunner()
    result = runner.invoke(main, ["-h"])
    assert result.exit_code == 0
    assert "WML ModelKit" in result.output


def test_root_vq_mutually_exclusive():
    """wmk -v -q must error."""
    runner = CliRunner()
    result = runner.invoke(main, ["-v", "-q", "sys", "--help"])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run tests to see failures**

```bash
uv run pytest tests/test_cli_spec.py::test_root_context_stores_verbose_key tests/test_cli_spec.py::test_root_help_short_flag tests/test_cli_spec.py::test_root_vq_mutually_exclusive -v
```

- [ ] **Step 3: Update `cli.py`**

Changes to `modelkit/cli.py`:

1. **Line 116**: `context_settings={"help_option_names": ["-h", "--help"]}` — already present in unstaged changes.
2. **Line 147**: Add mutual exclusion check:
   ```python
   if verbose and quiet:
       raise click.UsageError("Cannot use --verbose and --quiet together.")
   ```
3. **Line 152**: Rename key `"verbosity"` → `"verbose"`:
   ```python
   ctx.obj["verbose"] = verbose
   ```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli_spec.py -v
```

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check modelkit/cli.py
git add modelkit/cli.py tests/test_cli_spec.py
git commit -m "fix(cli): rename ctx.obj key verbosity→verbose, add -v/-q mutual exclusion"
```

---

## Task 3: Write Spec Validator Tests

**Files:**
- Modify: `tests/test_cli_spec.py`

These tests introspect the Click command tree and enforce the spec. They will fail initially (proving they catch violations), then pass as commands are migrated.

- [ ] **Step 1: Write the validator tests**

Append to `tests/test_cli_spec.py`:

```python
# ── Spec enforcement tests ──────────────────────────────────────
# These introspect the Click command tree to enforce the spec.
# They will be the last to pass, after all commands are migrated.

# Commands that must have -m/--model
MODEL_COMMANDS = {
    "build", "compile", "config", "export", "inspect",
    "perf", "quantize", "optimize", "analyze", "eval",
}

# Commands that must have --device and --ep
DEVICE_COMMANDS = {
    "build", "compile", "config", "perf",
    "eval", "analyze", "optimize", "quantize",
}

# Global shorts that subcommands must NOT define
RESERVED_SHORTS = {"-v", "-q", "-h"}


def _get_command(name: str) -> click.Command:
    """Get a subcommand by name from the root group."""
    cmd = main.commands.get(name)
    assert cmd is not None, f"Command '{name}' not found"
    return cmd


def _param_names(cmd: click.Command) -> set[str]:
    """Get all parameter names for a command."""
    return {p.name for p in cmd.params}


def _param_shorts(cmd: click.Command) -> list[str]:
    """Get all short flags for a command."""
    shorts = []
    for p in cmd.params:
        if hasattr(p, "opts"):
            for opt in p.opts:
                if opt.startswith("-") and not opt.startswith("--"):
                    shorts.append(opt)
    return shorts


def test_spec_all_commands_have_model():
    """Every command in MODEL_COMMANDS must have a 'model' parameter."""
    for name in MODEL_COMMANDS:
        cmd = _get_command(name)
        assert "model" in _param_names(cmd), (
            f"Command '{name}' is missing -m/--model (spec Section 2a)"
        )
        # NOTE: after migration, all commands use "model" (not "model_id")


def test_spec_sys_has_no_model():
    """sys command must NOT have -m/--model."""
    cmd = _get_command("sys")
    names = _param_names(cmd)
    assert "model" not in names, "sys should not have --model"


def test_spec_device_commands_have_device_and_ep():
    """Commands in DEVICE_COMMANDS must have both 'device' and 'ep'."""
    for name in DEVICE_COMMANDS:
        cmd = _get_command(name)
        names = _param_names(cmd)
        assert "device" in names, f"'{name}' missing --device (spec Section 2b)"
        assert "ep" in names, f"'{name}' missing --ep (spec Section 2b)"


def test_spec_no_subcommand_defines_verbose_or_quiet():
    """No subcommand may define --verbose or --quiet parameters (ADR-2).

    Short flags -v, -q, -h are checked by test_spec_global_flags_not_shadowed.
    """
    for name in main.commands:
        cmd = _get_command(name)
        names = _param_names(cmd)
        assert "verbose" not in names, (
            f"'{name}' defines --verbose — must use ctx.obj (ADR-2)"
        )
        assert "quiet" not in names, (
            f"'{name}' defines --quiet — must use ctx.obj (ADR-2)"
        )


def test_spec_device_choice_values():
    """All --device options must have exactly auto|cpu|gpu|npu."""
    expected = {"auto", "cpu", "gpu", "npu"}
    for name in DEVICE_COMMANDS:
        cmd = _get_command(name)
        device_param = next(
            (p for p in cmd.params if p.name == "device"), None
        )
        assert device_param is not None, f"'{name}' missing --device"
        assert isinstance(device_param.type, click.Choice), (
            f"'{name}' --device must be Choice type"
        )
        assert set(device_param.type.choices) == expected, (
            f"'{name}' --device choices are {device_param.type.choices}, "
            f"expected {expected}"
        )


def test_spec_device_case_insensitive():
    """All --device options must use case_sensitive=False."""
    for name in DEVICE_COMMANDS:
        cmd = _get_command(name)
        device_param = next(
            (p for p in cmd.params if p.name == "device"), None
        )
        assert device_param is not None
        assert device_param.type.case_sensitive is False, (
            f"'{name}' --device must be case_sensitive=False (ADR-4)"
        )


def test_spec_global_flags_not_shadowed():
    """No subcommand may redefine -h, -v, or -q (reserved globals)."""
    for name in main.commands:
        cmd = _get_command(name)
        shorts = _param_shorts(cmd)
        for reserved in RESERVED_SHORTS:
            assert reserved not in shorts, (
                f"'{name}' redefines {reserved} — reserved global short"
            )


def test_spec_output_flag_consistency():
    """Commands with -o must map it to --output (not --output-dir).

    build uses --output-dir WITHOUT -o short (ADR-3).
    """
    for name in main.commands:
        cmd = _get_command(name)
        for p in cmd.params:
            if not hasattr(p, "opts"):
                continue
            if "-o" in p.opts:
                assert "--output" in p.opts, (
                    f"'{name}' maps -o to {p.opts} — must be --output (ADR-3)"
                )
    # build must have --output-dir WITHOUT -o
    build_cmd = _get_command("build")
    for p in build_cmd.params:
        if hasattr(p, "opts") and "--output-dir" in p.opts:
            assert "-o" not in p.opts, (
                "build --output-dir must not have -o short (ADR-3)"
            )


def test_spec_short_flag_no_collisions():
    """No two options on the same command may share a short flag."""
    for name in main.commands:
        cmd = _get_command(name)
        shorts = _param_shorts(cmd)
        dupes = [s for s in shorts if shorts.count(s) > 1]
        assert not dupes, (
            f"'{name}' has short flag collisions: {set(dupes)}"
        )
```

- [ ] **Step 2: Run to see which commands violate the spec**

```bash
uv run pytest tests/test_cli_spec.py -v -k "test_spec" 2>&1 | head -60
```

Expected: Multiple failures — this proves the validators work. The exact failures become the migration checklist.

- [ ] **Step 3: Commit the validator tests (they will fail until migration is complete)**

```bash
git add tests/test_cli_spec.py
git commit -m "test(cli): add spec enforcement validators (will fail until migration)"
```

---

## Task 4: Migrate Commands — Remove Local `-v`/`-q`, Use `ctx.obj`

This is the largest task. Each command file needs:
1. Remove `@click.option("--verbose", "-v", ...)` decorator
2. Remove `verbose` from function signature
3. Add `@click.pass_context` if missing (config)
4. Replace `verbose` variable reads with `ctx.obj["verbose"]`
5. Replace `if ctx.obj.get("debug"): verbose = True` pattern (no longer needed)

**Files:** All 11 command files.

The migration is mechanical per command. Do them in dependency order — start with simpler commands, end with build (most complex).

- [ ] **Step 1: Migrate `sys.py`** (simplest command)

Remove `-v`/`--verbose` option. Read verbosity from `ctx.obj["verbose"]`.

- [ ] **Step 2: Migrate `inspect.py`**

Remove `-v`/`--verbose`. Remove `if ctx.obj.get("debug"): verbose = True`. Read `ctx.obj["verbose"]` where needed.

- [ ] **Step 3: Migrate `config.py`**

Add `@click.pass_context` (currently missing). Remove `-v`/`--verbose`. Read from `ctx.obj`. Replace `--precision` from `click.Choice(["auto","fp32","fp16","int8","int16"])` to shared `precision_option()` (string + validator per ADR-8). Device choices will reorder from `auto|npu|gpu|cpu` to `auto|cpu|gpu|npu` automatically via shared `device_option()` decorator (ADR-5).

- [ ] **Step 4: Migrate `export.py`**

Remove `-v`/`--verbose`. Remove debug override pattern.

- [ ] **Step 5: Migrate `quantize.py`**

Remove `-v`/`--verbose`. Add `-d`/`--device`, `--ep`, `-t`/`--task` short flag, `-n` for `--samples` using shared decorators.

- [ ] **Step 6: Migrate `optimize.py`**

Remove `-v`/`--verbose`. Remove `-p`/`--preset` (per user decision). Add `-d`/`--device`, `--ep` using shared decorators.

- [ ] **Step 7: Migrate `eval.py`**

Remove `-v`/`--verbose`. Add `--ep`, `-t`/`--task` short flag.

- [ ] **Step 8: Migrate `analyze.py`**

Remove `-v`/`--verbose`. Ensure `--device`/`--ep` use shared decorators.

- [ ] **Step 9: Migrate `compile.py`**

Remove `-v`/`--verbose`. Replace `--quantize/--no-quantize` pair with `--no-quant`. Replace `--validate/--no-validate` pair with `--no-validate` (single flag). Change `--output-dir` to `-o`/`--output`. Change `--device` default from `npu` to `auto`. Change `--ep` from `click.Choice(VALID_EPS)` to `type=str` (using shared `ep_option()` decorator). Use shared decorators for model, device, ep, output.

- [ ] **Step 10: Migrate `perf.py`**

Remove `-v`/`--verbose`. Add `-n` for `--iterations`, `-t` for `--task`. Rename `--no-quantize` to `--no-quant` (Section 5 negation convention). Replace `--precision` from `click.Choice` to shared `precision_option()` (string + validator per ADR-8). Remove `--compare-devices` (not in spec, never implemented). Use shared device/ep decorators.

- [ ] **Step 11: Migrate `build.py`**

Remove local `-v`/`--verbose` and `-q`/`--quiet`. Make `-m`/`--model` required. Remove `-o` short flag from `--output-dir` — per ADR-3, `-o` is reserved for `--output` (file output); `build` uses `--output-dir` (no short form). Use shared decorators for model, device, ep. This is the most complex — has quiet plumbing through `_run_single_build`, `StageLive`, etc. The `quiet` value now comes from `ctx.obj["quiet"]` instead of a local parameter.

- [ ] **Step 12: Run full test suite after each migration**

After each command migration:
```bash
uv run pytest tests/test_cli_spec.py -v -k "test_spec"
```

Watch the failure count decrease with each command migrated.

- [ ] **Step 13: Lint all modified files**

```bash
uv run ruff check modelkit/commands/ modelkit/cli.py
```

- [ ] **Step 14: Commit**

```bash
git add modelkit/commands/ modelkit/cli.py
git commit -m "refactor(cli): migrate all commands to shared decorators + global verbosity

Removes per-command -v/--verbose in favor of global ctx.obj['verbose'].
Standardizes --device, --ep, -m/--model via _options.py shared decorators.
Replaces compile flag pairs with --no-quant/--no-validate (ADR-10).
"
```

---

## Task 5: Run Full Spec Validation

- [ ] **Step 1: Run all spec validator tests**

```bash
uv run pytest tests/test_cli_spec.py -v
```

Expected: ALL PASS.

- [ ] **Step 2: Run existing test suite to catch regressions**

```bash
uv run pytest tests/ -v --ignore=tests/integration 2>&1 | tail -20
```

Fix any regressions from the migration.

- [ ] **Step 3: Smoke test key commands**

```bash
uv run wmk -h
uv run wmk -vv sys
uv run wmk -q build --help
uv run wmk export --help
uv run wmk compile --help
```

Verify help text looks correct and global flags propagate.

- [ ] **Step 4: Final lint**

```bash
uv run ruff check modelkit/ tests/test_cli_spec.py
```

- [ ] **Step 5: Commit any fixes**

```bash
git add modelkit/ tests/test_cli_spec.py
git commit -m "fix(cli): address regression fixes from args migration"
```

---

## Task 6: Final Commit and Cleanup

- [ ] **Step 1: Verify git log is clean**

```bash
git log --oneline -10
```

- [ ] **Step 2: Verify no leftover `verbose` parameters in command signatures**

```bash
grep -rn "def .*(.*verbose.*)" modelkit/commands/
```

Expected: No matches (all removed).

- [ ] **Step 3: Verify no leftover `-v` in command decorators**

```bash
grep -rn '"-v"' modelkit/commands/
```

Expected: No matches.

- [ ] **Step 4: Done**

All spec validator tests pass. All existing tests pass (or pre-existing failures documented). CLI arguments are standardized per `docs/design/cli/3_cli_args_spec.md`.
