# CLI Arguments Specification

## Table of Contents

- [Overview](#overview)
- [Architectural Decision Records](#architectural-decision-records)
- [Global Options](#1-global-options)
- [Shared Options](#2-shared-options)
- [Command-Specific Options](#3-command-specific-options)
- [Short Flag Registry](#4-short-flag-registry)
- [Negation Convention](#5-negation-convention)
- [Implementation Architecture](#6-implementation-architecture)
- [References](#references)

---

## Overview

### Purpose

This document specifies the **argument conventions** for the `wmk` CLI. It defines what flags exist, how they're named, where they live, and how they compose across the 11 subcommands.

The existing CLI PRD (`1_prd.md`) covers the framework — plugin discovery, global debug, lazy imports. This spec fills the gap: **the argument contract** that all commands must follow.

### Scope

- 11 subcommands: build, compile, config, export, inspect, perf, quantize, optimize, analyze, eval, sys
- Global options on root `wmk` group
- Shared options via reusable decorators
- Command-specific options
- Test-time enforcement of the spec

### Design Principles

1. **One source of truth**: Shared options defined once in `_options.py`
2. **Predictable**: Same flag name always means the same thing
3. **Minimal surprise**: Follows Click/Python CLI conventions (pip, ruff, uv)
4. **Extensible**: New commands inherit shared options; test validator catches omissions
5. **Clean slate**: No backward-compatibility constraints (pre-1.0 internal tool)

---

## Architectural Decision Records

### ADR-1: Breaking Change Strategy

**Context**: CLI args grew organically with inconsistencies across 11 subcommands.

| Option | Description |
|--------|-------------|
| A. Non-breaking | Add missing flags, deprecate old names with warnings |
| B. Breaking + deprecation | Rename flags, one release cycle of warnings |
| **C. Clean slate** | **Redesign freely, no backward compat** |

**Decision**: C — wmk is internal/pre-1.0, no external consumers depend on exact flag names.

---

### ADR-2: Verbosity Scope

**Context**: Should `-v`/`-q` be global-only or per-command?

| Option | Description |
|--------|-------------|
| **A. Global only** | **Root `wmk` level, inherited via `ctx.obj`** |
| B. Both levels, merged | Root + subcommand, values add up |

**Decision**: A — Pythonic convention (pip, ruff, uv, black all do this). Click's `ctx.obj` was designed for this pattern. No use case for per-command verbosity in wmk.

**Supersedes**: The PRD (`1_prd.md`) lists `--verbose`/`-v` as a standard subcommand option. This ADR overrides that — verbosity is global only.

**Migration impact**: All 11 current subcommands define their own `-v`/`--verbose` as a boolean flag. These must all be removed, and each command refactored to read `ctx.obj["verbose"]` instead of a local `verbose` parameter. This is the single largest migration item in this spec.

---

### ADR-3: Output Flag Convention

**Context**: `-o` means "file" in most commands but "directory" in `build`.

| Option | Description |
|--------|-------------|
| A. Always `-o`/`--output` | Semantic varies, documented in help |
| **B. Split** | **`-o`/`--output` for files, `--output-dir` for directories** |
| C. Always `--output-dir` | Everything writes to directory |

**Decision**: B — explicit about what you're getting. `compile` produces a single file (with embedded context), so it uses `-o`/`--output`. Only `build` uses `--output-dir`.

---

### ADR-4: Device Input Casing

**Context**: Mixed casing across commands and internal code.

| Option | Description |
|--------|-------------|
| A. Lowercase in, uppercase display | User types `npu`, display shows `NPU` |
| **B. Case-insensitive** | **Accept any casing, normalize to uppercase internally** |

**Decision**: B — Click supports `case_sensitive=False` natively. Most forgiving for users.

---

### ADR-5: Device Choice Set

**Context**: Different commands had different choice orders and sets.

**Decision**: Canonical set is `auto | cpu | gpu | npu` (alphabetical after auto). Same on all 8 device-aware commands.

**Breaking change**: Current commands use `auto|npu|gpu|cpu` order. This is an intentional reordering to follow alphabetical convention.

**Future**: If new devices are added (dsp, fpga), they join the canonical set in `_options.py` — one place to update.

---

### ADR-6: `--device` and `--ep` Scope

**Context**: Inconsistent command coverage for device/EP options.

**Decision**: Same scope — both present on: build, compile, config, perf, eval, analyze, optimize, quantize. Absent from: export, inspect, sys.

**Rationale**: If a command targets a device, it might also need to override the EP.

---

### ADR-7: Model Option Scope

**Context**: `-m`/`--model` is the backbone of wmk — nearly every command needs a model.

**Decision**: Present on all commands except `sys`. Always `-m`/`--model`.

`required=True` by default. Commands with info-only modes (e.g., `--list`, `--list-tasks`, `--list-capabilities`) set `required=False` and handle validation internally — these modes don't need a model.

| Command | Required | Exception |
|---------|----------|-----------|
| export | Yes | — |
| quantize | Yes | — |
| build | Yes | — |
| optimize | No | `--list-capabilities` needs no model |
| compile | No | `--list` needs no model |
| config | No | `--model-type` can work without model |
| inspect | No | `--list-tasks` needs no model |
| perf | No | Can read model from config |
| eval | No | Can read model from config |
| analyze | No | Can read model from config |

---

### ADR-8: Precision as String (not Choice)

**Context**: Precision values include simple (`fp32`, `int8`) and compound (`w8a16`, `w4a16`). The set grows as backends evolve.

| Option | Description |
|--------|-------------|
| A. Fixed Choice | Hard-coded set, rejects unknowns |
| **B. String + validator** | **Accept any string, validate against known set, warn on unknown** |

**Decision**: B — extensible. A callback validator checks against known values and warns (not errors) on unknown, so new precision formats don't require CLI code changes.

---

### ADR-9: Implementation Approach

**Context**: How to standardize options across commands.

| Option | Description |
|--------|-------------|
| A. Shared decorators | `_options.py` with reusable Click decorators |
| B. Class-based commands | `WmkCommand`, `DeviceAwareCommand` classes |
| **C. Shared decorators + test validator** | **Decorators for DRY + test suite for enforcement** |

**Decision**: C — simplicity of decorators, enforcement through tests. Fits project's existing patterns (Click decorators + strong test discipline). Test validator introspects the Click command tree at runtime.

---

### ADR-10: Negation Convention

**Context**: Mix of `--no-X` single flags and `--X/--no-X` pairs.

| Option | Description |
|--------|-------------|
| **A. `--no-X` only** | **Single negative flag, default is "do it"** |
| B. `--X/--no-X` pairs | Click flag pairs |

**Decision**: A — simpler. Default behavior is always "enabled." Users opt out with `--no-quant`, `--no-compile`, etc.

**Note**: This supersedes `compile`'s existing `--quantize/--no-quantize` and `--validate/--no-validate` pairs. Under the new convention, compile uses `--no-quant` and `--no-validate` (single negative flags).

---

## 1. Global Options

These live on the root `wmk` group only. Subcommands inherit them via `ctx.obj` — they never define their own `-v` or `-q`.

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--verbose` | `-v` | count | 0 | Increase verbosity (-v=INFO, -vv=DEBUG) |
| `--quiet` | `-q` | count | 0 | Decrease verbosity (-q=less, -qq=summary only) |
| `--debug` | — | flag | False | Alias for -vv (hidden) |
| `--version` | — | — | — | Show version and exit |
| `--help` | `-h` | — | — | Show help and exit |

### Rules

- `-v` and `-q` are mutually exclusive (error if both given)
- `--debug` sets `verbose=2` and is hidden from help
- Subcommands access via `ctx.obj["verbose"]` and `ctx.obj["quiet"]`
- No subcommand may define its own `-v`, `-q`, `--verbose`, or `--quiet`

### Verbosity Matrix

| Level | Quiet | Verbose | Behavior |
|-------|-------|---------|----------|
| Normal | 0 | 0 | Default output — stage progress, results |
| Quiet | 1 | 0 | Hide detail lines, keep stage status |
| Silent | 2 | 0 | Summary only — no stage output |
| Verbose | 0 | 1 | Step-by-step, INFO logging |
| Debug | 0 | 2 | Full DEBUG logging |

### Context Object Contract

Root `wmk` group stores in `ctx.obj`:

```python
ctx.obj = {
    "verbose": int,   # 0, 1, or 2
    "quiet": int,     # 0, 1, or 2
    "debug": bool,    # True if --debug or -vv
}
```

Subcommands read — never write — these values.

---

## 2. Shared Options

Reusable Click option decorators defined in `modelkit/commands/_options.py`. Each command imports and composes the decorators it needs.

### 2a. Model Option (`-m`/`--model`)

| Flag | Short | Type | Default Required | Description |
|------|-------|------|------------------|-------------|
| `--model` | `-m` | string | Yes (see ADR-7 for exceptions) | HuggingFace model ID, local path, or .onnx file |

**Scope**: All commands except `sys`. See ADR-7 for the per-command required/optional matrix — commands with info-only modes (e.g., `--list`, `--list-tasks`) set `required=False`.

### 2b. Device & EP Options

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--device` | `-d` | Choice(`auto\|cpu\|gpu\|npu`, case_sensitive=False) | `auto` | Target device |
| `--ep` | — | string | None | Force specific execution provider (overrides `--device`) |

**Scope**: build, compile, config, perf, eval, analyze, optimize, quantize.

- Device choices are case-insensitive; normalized to uppercase internally.
- Default is `auto` for all commands.
- `--ep` has no short form — it's an advanced override.

### 2c. Output Options

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--output` | `-o` | Path | Output file (single artifact) |
| `--output-dir` | — | Path | Output directory (multi-artifact) |

**Scope by command:**

| Command | Flag | Reason |
|---------|------|--------|
| export | `-o`/`--output` | Single .onnx |
| config | `-o`/`--output` | Single .json |
| perf | `-o`/`--output` | Single .json |
| eval | `-o`/`--output` | Single .json |
| quantize | `-o`/`--output` | Single .onnx |
| optimize | `-o`/`--output` | Single .onnx |
| analyze | `-o`/`--output` | Single .json |
| compile | `-o`/`--output` | Single .onnx (embedded context) |
| build | `--output-dir` | Directory of artifacts |
| inspect | — | Stdout only |
| sys | — | Stdout only |

### 2d. Task Option

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--task` | `-t` | string | Override auto-detected task (e.g., image-classification) |

**Scope**: export, config, inspect, perf, eval, quantize, analyze.

**Not on**: build (reads from config), compile (task-agnostic), optimize (task-agnostic), sys.

### 2e. Precision Option

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--precision` | `-p` | string | `auto` | Target precision |

**Scope**: config, perf, quantize.

**Validation**: Callback validates against known values. Known set today:

- Simple: `auto`, `fp32`, `fp16`, `int8`, `int16`
- Compound: `w8a8`, `w8a16`, `w4a16`

Unknown values produce a warning (not error) for forward-compatibility.

---

## 3. Command-Specific Options

Beyond shared options, each command has its own specialized flags. Shared options (Section 2) are implied and not repeated here.

### build

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--config` | `-c` | Path | — | WinMLBuildConfig JSON file |
| `--output-dir` | — | Path | — | Output directory for artifacts |
| `--rebuild` | — | flag | False | Overwrite existing artifacts |
| `--no-quant` | — | flag | False | Skip quantization stage |
| `--no-compile` | — | flag | False | Skip compilation stage |
| `--no-analyze` | — | flag | False | Skip analyzer loop |
| `--max-analyze-iterations` | — | int | 3 | Max analyzer iterations |
| `--use-cache` | — | flag | False | Use global cache (~/.cache/winml/) |

### compile

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--compiler` | — | Choice(`ort\|qairt`) | `ort` | Compiler backend |
| `--embed` | — | flag | False | Embed EP context in ONNX file |
| `--no-quant` | — | flag | False | Skip quantization before compilation |
| `--no-validate` | — | flag | False | Skip compiled model validation |
| `--qnn-sdk-root` | — | Path | None | Path to QAIRT SDK root |
| `--list` | `-l` | flag | False | List available compilers and exit |

**Migration note**: Replaces the old `--quantize/--no-quantize` and `--validate/--no-validate` flag pairs (ADR-10).

### config

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--config` | `-c` | Path | — | JSON config with overrides |
| `--model-class` | — | string | None | Override auto-detected model class |
| `--model-type` | — | string | None | Override auto-detected model type |
| `--module` | — | string | None | Submodule class name filter |
| `--shape-config` | — | Path | None | JSON with shape overrides |
| `--library` | — | string | `transformers` | Source library |
| `--no-quant` | — | flag | False | Exclude quantization from config |
| `--no-compile` | — | flag | False | Exclude compilation from config |
| `--trust-remote-code` | — | flag | False | Allow custom code from model repo |

### export

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--with-report` | — | flag | False | Generate export reports (md + json) |
| `--no-hierarchy` | — | flag | False | Skip hierarchy_tag metadata |
| `--dynamo` | — | flag | False | Enable dynamo export |
| `--torch-module` | — | string | None | torch.nn modules to include (comma-sep) |
| `--input-specs` | — | Path | None | JSON input specifications |
| `--export-config` | — | Path | None | ONNX export config JSON |

### inspect

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--format` | `-f` | Choice(`table\|json`) | `table` | Output format |
| `--hierarchy` | `-H` | flag | False | Show HF module hierarchy |
| `--list-tasks` | — | flag | False | List all known tasks and exit |
| `--model-type` | — | string | None | Override model type |
| `--model-class` | — | string | None | Override model class |

### perf

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--iterations` | `-n` | int | 100 | Benchmark iterations |
| `--warmup` | — | int | 10 | Warmup iterations |
| `--batch-size` | — | int | 1 | Batch size |
| `--no-quant` | — | flag | False | Skip quantization during model build |
| `--module` | — | string | None | Per-module benchmarking |
| `--monitor` | — | flag | False | Live NPU utilization chart |
| `--op-tracing` | — | Choice(`basic\|detail`) | None | Operator-level profiling |

Also inherits shared options: `-m`, `-d`/`--device`, `--ep`, `-o`/`--output`, `-t`/`--task`, `-p`/`--precision` (see Section 2).

### quantize

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--samples` | `-n` | int | 10 | Calibration samples |
| `--method` | — | Choice(`minmax\|entropy\|percentile`) | `minmax` | Calibration method |
| `--weight-type` | — | string | None | Weight quantization type |
| `--activation-type` | — | string | None | Activation quantization type |
| `--per-channel` | — | flag | False | Per-channel quantization |
| `--symmetric` | — | flag | False | Symmetric quantization |

### optimize

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--config` | `-c` | Path | None | Config file (YAML/JSON) |
| `--list-capabilities` | `-l` | flag | False | List capabilities and exit |
| *dynamic* | — | — | — | Auto-generated `--enable-X`/`--disable-X` from capability registry |

### analyze

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--format` | `-f` | Choice(`table\|json`) | `table` | Output format |
| `--information` | — | flag | True | Show compatibility info |
| `--htp-metadata` | — | Path | None | HTP metadata for advanced analysis |
| `--no-run-unknown-op` | — | flag | False | Skip unknown op runtime checking |
| `--optim-config` | — | Path | None | Optimization config for analysis |

**Note on `--format` choices**: inspect and analyze use `table|json` (they render Rich tables). sys uses `text|json|compact` (plain text output). The choice sets intentionally differ to match each command's output style.

### eval

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--dataset` | — | string | None | HF dataset path |
| `--dataset-name` | — | string | None | Dataset config name |
| `--samples` | `-n` | int | 100 | Dataset samples |
| `--split` | — | string | `validation` | Dataset split |
| `--shuffle` | — | flag | False | Shuffle before sampling |
| `--streaming` | — | flag | False | Stream dataset |
| `--column` | — | string (multiple) | — | Column mapping key=value |
| `--label-mapping` | — | string | None | Label mapping JSON |
| `--model-id` | — | string | None | HF model ID when -m is .onnx |

### sys

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--format` | `-f` | Choice(`text\|json\|compact`) | `text` | Output format |
| `--list-device` | — | flag | False | List available devices |
| `--list-ep` | — | flag | False | List available EPs |

---

## 4. Short Flag Registry

One source of truth to prevent collisions across the entire CLI.

| Short | Long | Scope | Notes |
|-------|------|-------|-------|
| `-h` | `--help` | global | All commands (Click built-in) |
| `-v` | `--verbose` | global | Count flag |
| `-q` | `--quiet` | global | Count flag |
| `-m` | `--model` | shared | All except sys |
| `-d` | `--device` | shared | 8 commands |
| `-o` | `--output` | shared | File output commands |
| `-t` | `--task` | shared | 7 commands |
| `-p` | `--precision` | shared | config, perf, quantize |
| `-c` | `--config` | command | build, config, optimize |
| `-f` | `--format` | command | inspect, analyze, sys |
| `-n` | `--iterations`/`--samples` | command | perf (`--iterations`), eval (`--samples`), quantize (`--samples`) |
| `-l` | `--list`/`--list-capabilities` | command | compile (`--list`), optimize (`--list-capabilities`) |
| `-H` | `--hierarchy` | command | inspect only |

### Rules

- **Global shorts** (`-h`, `-v`, `-q`) are reserved — no subcommand may redefine them.
- **Shared shorts** (`-m`, `-d`, `-o`, `-t`, `-p`) must always map to the same long form.
- **Command shorts** (`-c`, `-f`, `-n`, `-l`, `-H`) may map to different longs per command, but the semantic should be similar (e.g., `-n` always means "count of something").
- **Available** for future use: `-a`, `-b`, `-e`, `-g`, `-i`, `-j`, `-k`, `-r`, `-s`, `-u`, `-w`, `-x`, `-y`, `-z`, `-P`.

---

## 5. Negation Convention

Use `--no-X` single flags. Default behavior is always "enabled" — flags opt out.

| Pattern | Example | Meaning |
|---------|---------|---------|
| `--no-quant` | `wmk build --no-quant` | Skip quantization |
| `--no-compile` | `wmk build --no-compile` | Skip compilation |
| `--no-analyze` | `wmk build --no-analyze` | Skip analyzer |
| `--no-validate` | `wmk compile --no-validate` | Skip validation |
| `--no-hierarchy` | `wmk export --no-hierarchy` | Skip hierarchy tags |

**Removed**: `--quantize/--no-quantize` pair from compile (was deprecated). Clean slate — just `--no-validate`.

---

## 6. Implementation Architecture

### 6a. Shared Decorator Module

`modelkit/commands/_options.py` — underscore prefix means it's importable but not auto-discovered as a command.

```
modelkit/commands/
├── _options.py          # Shared option decorators
├── build.py
├── compile.py
├── config.py
├── export.py
├── inspect.py
├── perf.py
├── quantize.py
├── optimize.py
├── analyze.py
├── eval.py
└── sys.py
```

Decorator pattern:

```python
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
        type=click.Choice(["auto", "cpu", "gpu", "npu"], case_sensitive=False),
        callback=_normalize_uppercase,
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
```

Commands compose:

```python
@click.command()
@model_option()
@device_option()
@ep_option()
@output_file_option()
def compile(...):
    ...
```

### 6b. Test Validator

`tests/test_cli_spec.py` — enforces this spec in CI.

| Test | What it checks |
|------|----------------|
| `test_all_commands_have_model` | Every command except `sys` has `-m`/`--model` |
| `test_device_commands_have_device_and_ep` | 8 commands have both `--device` and `--ep` |
| `test_no_subcommand_defines_verbose_or_quiet` | No subcommand has `-v`, `-q`, `--verbose`, or `--quiet` |
| `test_short_flag_no_collisions` | No two options on the same command share a short flag |
| `test_output_flag_consistency` | Commands with `-o` use `--output` (not `--output-dir`) |
| `test_device_choice_values` | All `--device` options have exactly `auto\|cpu\|gpu\|npu` |
| `test_device_case_insensitive` | All `--device` choices use `case_sensitive=False` |
| `test_global_flags_not_shadowed` | No subcommand redefines `-h`, `-v`, `-q` |

Tests introspect the Click command tree at runtime — no manual lists to keep in sync.

---

## References

- [CLI Framework PRD](1_prd.md) — Framework-level spec (discovery, global debug, error handling). **Note**: ADR-2 in this spec supersedes the PRD's recommendation of per-command `--verbose`.
- [CLI Core Loop](2_coreloop.md) — Implementation patterns (lazy imports, debug inheritance)
- [CLI Testing Strategy](testing-strategy.md) — Testing approach
- [Click Documentation](https://click.palletsprojects.com/) — CLI framework
