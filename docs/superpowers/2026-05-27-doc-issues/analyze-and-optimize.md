# Issues: docs/concepts/analyze-and-optimize.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical (factually wrong; user would hit error)

- **`--output results.json` flag for `winml analyze`** (line 9): The doc says "add `--output results.json` to save the report as JSON". The actual flag is `--output` (source: `commands/analyze.py` line 653 `@cli_utils.output_option("Save JSON output to file")`). This is valid and correct.

- **`--preset` flag on `winml optimize`** (line 21): The doc says "Use presets (`--preset transformer-optimized`, `--preset qnn-compatible`) as a starting point." No `--preset` flag exists on `winml optimize`. The command has `--config` (a config file) and capability flags, but no `--preset` option (source: `commands/optimize.py` — the full file was read and contains no `--preset` option). This is a fabricated flag that would cause `Error: No such option: --preset` if a user tries it.

## Important (misleading or stale claim)

- **Exit codes described as 0/1/2** (line 10): Doc says "zero is full support, one is partial support with unsupported operators, two is a configuration error." Source confirms: `commands/analyze.py` line 1212-1213 (`sys.exit(0 if overall_supported else 1)`) and lines 734-736, 1216-1222 use `sys.exit(2)` for errors. This matches the doc.

- **`--save-node unsupported` or `--save-node partial`** (line 11): Doc says "Use `--save-node unsupported` or `--save-node partial`". Source shows `--save-node` with `multiple=True` and choices `["partial", "unsupported"]` (`commands/analyze.py` lines 673-676). The flag exists and the values are valid.

- **`--max-optim-iterations` default described as "three"** (line 26): Doc says "default: three". Source confirms `default: 3` in the help text (`commands/build.py` line 310) and `hack_max_optim_iterations` defaults to `3` in the build pipeline (`commands/build.py` line 1112, 1234). Correct.

- **`--no-analyze` on `winml build`** (line 27): The doc says "`winml build` runs analyze and optimize in an alternating loop" and "Use `--no-analyze` to skip the loop". Source confirms `--no-analyze` on `winml build` (`commands/build.py` lines 294-298) which sets `hack_max_optim_iterations = 0`. Correct.

- **`--commit a specific combination to a `--config` file`** (line 21): Doc says "commit a specific combination to a `--config` file". The `winml optimize` command has `--config` / `-c` (source: `commands/optimize.py` lines 176-180). This is valid.

## Minor (style, polish, low-impact)

- **`--list-capabilities` and `--list-rewrites` flags** (lines 17, 19): Both exist on `winml optimize` → `commands/optimize.py` lines 153, 160. Correct.

- **Pattern-rewrite flag form `--enable-<source-slug>-<target-slug>`** (line 19): Consistent with source → `commands/optimize.py` lines 217-224, which documents `--enable-gelu-singlegelu` as example. Correct.

- **Cross-links** `[compile-and-epcontext.md]`, `[primitives-and-pipeline.md]`, `[../commands/analyze.md]`, `[../commands/optimize.md]` (lines 31-34): All files exist.

## Verified correct (anchored claims you checked)

- `winml analyze` `--ep` flag exists and takes provider name → `commands/analyze.py` lines 628-639
- `winml analyze` `--device` flag with CPU/GPU/NPU choices → `commands/analyze.py` lines 641-650
- `winml analyze` `--information` / `--no-information` flag (default: enabled) → `commands/analyze.py` lines 654-657
- `winml analyze` `--output` flag for JSON → `commands/analyze.py` line 653
- `winml analyze` exit codes 0/1/2 → `commands/analyze.py` lines 1212-1213, 1216-1222
- `winml optimize` `--enable-<name>` / `--disable-<name>` flag pattern → `commands/optimize.py` lines 124-131
- `winml optimize` `--list-capabilities` flag → `commands/optimize.py` lines 153-158
- `winml optimize` `--list-rewrites` flag → `commands/optimize.py` lines 160-164
- `winml optimize` `--config` file flag → `commands/optimize.py` lines 176-180
- Fusions include GeLU, LayerNorm, MatMul+Add → `optim/pipes/graph.py` lines 242-243
- No `wmk` or `ModelKit` strings in prose → verified by grep
