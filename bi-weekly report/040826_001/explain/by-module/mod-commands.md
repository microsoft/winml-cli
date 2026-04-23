# Module: commands
**Path**: `src/winml/modelkit/commands/`
**Period**: 2026-03-23 to 2026-04-08

## 1. Module Overview
The `commands` module implements all CLI subcommands for the `winml` tool (formerly `wmk`). Each subcommand (analyze, build, compile, config, eval, export, hub, inspect, optimize, perf, quantize, sys) is a separate module file.

## 2. Files Changed This Period
| File | PRs | Summary |
|------|-----|---------|
| `hub.py` | #196, #205, #246 | New hub command (+471 lines); renamed wmk→winml (#205); overflow mode fix (#246) |
| `sys.py` | #205, #201, #246 | wmk→winml rename; log leak suppression (#201); JSON/compact output fix (#246) |
| `analyze.py` | #200, #205 | ASCII symbol replacement for charmap fix; wmk→winml rename |
| `build.py` | #15, #205 | Batch update; wmk→winml rename |
| `compile.py` | #205 | wmk→winml rename |
| `config.py` | #205 | wmk→winml rename |
| `eval.py` | #15, #205 | New eval command (+284 lines in #15); wmk→winml rename |
| `export.py` | #205 | wmk→winml rename |
| `inspect.py` | #205 | wmk→winml rename |
| `optimize.py` | #205 | wmk→winml rename |
| `perf.py` | #15, #205 | Batch update; wmk→winml rename |
| `quantize.py` | #205 | wmk→winml rename |
| `__init__.py` | #196, #222 | hub command registration; minor fixup |

## 3. Net Change Summary
- The CLI entry point was renamed from `wmk` to `winml` across all 12 command files and `pyproject.toml` in PR #205.
- A new `hub` subcommand was added in PR #196, providing a Rich console browser for the model registry.
- Windows cp1252 charmap errors were fixed across `analyze.py` and remaining command files in PRs #200 and #208.
- `sys.py` was refactored to suppress EP registry log messages from leaking to stdout and to correctly produce JSON and compact output formats.
- `hub.py` had its list-view overflow mode changed from `'ellipsis'` to `'crop'` to prevent U+2026 output on Windows terminals.

## 4. New APIs/Functions Added
| Symbol | Description |
|--------|-------------|
| `commands/hub.py` | New hub command module — model registry browser with list and detail views |
| `commands/eval.py` | New eval command module added in the #15 batch update |
