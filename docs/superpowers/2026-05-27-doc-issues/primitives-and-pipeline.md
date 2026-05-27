# Issues: docs/concepts/primitives-and-pipeline.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical (factually wrong; user would hit error)

- (none)

## Important (misleading or stale claim)

- **`--use-cache` described as alternative to `-o`/`--output-dir`** (line 62-65): Doc says "accepts `--use-cache` in place of `-o`/`--output-dir`". The short flag for output directory in `winml build` is `-o` but the parameter is named `--output-dir`, not `--output`. The doc uses `-o`/`--output-dir` inconsistently: line 62 says "in place of `-o`/`--output-dir`" but elsewhere uses `--output-dir`. Source: `src/winml/modelkit/commands/build.py` lines 249-262 — the option is `--output-dir` / `-o`. This is technically fine but the description shorthand could confuse users.

- **`winml build -c config.json -m microsoft/resnet-50 -o output/`** (line 49): The short flag `-o` maps to `--output-dir` in the build command (source: `commands/build.py` line 250-256). This is valid but worth noting: `-o` is the shorthand for `--output-dir`, not `--output`. The doc uses `-o` which is correct.

- **`WinMLBuildConfig` has six nested sub-configs, not five** (line 51 in config-and-build.md references five — this doc only lists five): The `WinMLBuildConfig` dataclass also has an `eval: WinMLEvaluationConfig | None` field and an `auto: bool` field (source: `src/winml/modelkit/config/build.py` lines 132-138). The doc does not mention these — omission rather than error, but the `eval` field could be relevant to users combining `winml build` and `winml eval`.

## Minor (style, polish, low-impact)

- **Cross-link `[ConvNeXT primitives sample](../samples/convnext-primitives.md)`** (line 104): The file `docs/samples/convnext-primitives.md` exists and the link is valid.

- **`winml build` without `-c`** (lines 49, 62): Doc implies `-c` is required for `winml build`. Source shows `-c` is `required=False` (`commands/build.py` line 236-241) — if omitted, config is auto-generated from `-m`. The doc's initial description of the command is accurate but does not mention the `-c`-less shorthand.

## Verified correct (anchored claims you checked)

- `WinMLBuildConfig` exists as a dataclass → `src/winml/modelkit/config/build.py` line 97
- `winml build` flags `--no-quant`, `--no-compile`, `--no-optimize` all exist → `commands/build.py` lines 273, 275-282, 300-304
- `--use-cache` flag exists and is mutually exclusive with `--output-dir` → `commands/build.py` lines 258-262, 376-379
- `--rebuild` flag exists → `commands/build.py` lines 263-268
- Setting `quant` or `compile` to `null` skips those stages → `config/build.py` lines 133-136 (both are `| None`)
- `~/.cache/winml/` as global cache path → `commands/build.py` line 261 (`~/.cache/winml/`)
- Six primitive commands listed are all real CLI commands → `commands/` directory contains `export.py`, `optimize.py`, `quantize.py`, `compile.py`, `perf.py`, `eval.py`
- No `wmk` or `ModelKit` strings in prose → verified by grep
