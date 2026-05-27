# Issues: docs/commands/optimize.md

Source verified against: `src/winml/modelkit/commands/optimize.py` @ 5e25579

## Critical (flag/behavior wrong; user gets error)

- **`--preset` flag does not exist in source.** The doc (lines 21, 29–35) documents a `--preset / -p` flag accepting `qnn-compatible|transformer-optimized|full|minimal`. There is no such option anywhere in `optimize.py`. The source `@click.command()` definition (lines 151–187) has `--list-capabilities`, `--list-rewrites`, `--model`, `--output`, `--config`, `--verbose`, and the dynamically-generated capability flags. No `--preset` option is defined. Any user running `winml optimize -m model.onnx --preset qnn-compatible` will get "Error: no such option: --preset". The entire "Built-in presets" table (doc lines 29–35) and every preset-based example in the doc are invalid.

- **`-p` short form is documented for `--preset`** (doc line 21) but in source, no `-p` exists. The `--model` flag does have `-m` and `--output` has `-o`, but there is no `-p` anywhere in the command definition.

- **"Configuration precedence" claims preset is step 3** (doc lines 38–43) with order: CLI flags > config file > preset > capability defaults. The actual source precedence (lines 363–383) is: capability defaults, then config file, then CLI options. There is no preset layer. The precedence documented is for a different version or planned feature.

## Important (misleading or stale)

- **`--verbose / -v` flag is absent from the doc flag table.** Source lines 180–185 define `@click.option("--verbose", "-v", is_flag=True, default=False, ...)`. The doc table lists only `--model`, `--output`, `--preset`, `--config`, `--list-capabilities`, `--list-rewrites`, and dynamic flags — `--verbose` is missing entirely.

- **`--model` short form `-m`** is not shown in the doc's flag table (the Short column is empty for `--model` at doc line 19). Source line 167 defines `"--model", "-m"`. Users will not know `-m` works.

- **"Configuration precedence" in source is 3-level, not 4-level.** Source lines 363–383 implement: (1) capability defaults, (2) config file, (3) CLI options. The doc describes 4 levels including "preset". Without the preset, the doc's precedence section incorrectly numbers and describes the chain.

- **Examples use `--preset`** (doc lines 71–85) — all preset-based examples produce errors with the current source. The only valid examples are:
  - `winml optimize -m model.onnx` (default caps)
  - `winml optimize --list-capabilities`
  - `winml optimize --list-rewrites`
  - `winml optimize -m model.onnx --enable-<cap>` / `--disable-<cap>`
  - `winml optimize -m model.onnx -c config.json`

- **`--config` type described as `PATH`** — the doc says "YAML or JSON configuration file" (doc line 23). Source line 175 uses `type=click.Path(exists=True, path_type=Path)` and `load_config()` (lines 48–70) supports `.yaml/.yml` and `.json`. This is correct.

## Minor (polish)

- The doc's dynamic flags section (line 25) correctly describes `--enable-<name>/--disable-<name>` pairs from the capability registry and `--list-capabilities` to discover them. This matches source lines 109–148.
- The claim that "adding a new optimization to the registry automatically makes it available as a CLI flag" matches source — `capability_options` decorator (lines 109–148) auto-generates flags at import time.
- `--list-capabilities` with `-l` short form → source lines 153–157 confirm `-l` is the short form. Correctly documented.
- `--list-rewrites` (no short form) → source lines 159–163 confirm. Correctly documented.
- Output path default `{input}_opt.onnx` → source lines 352–353 confirm.
- Before/after node count reduction report → source lines 419–423 confirm.

## Verified correct (key claims checked)

- `--model / -m` exists, `required=False` (only required when not listing) → source lines 165–171.
- `--output / -o` exists via `cli_utils.output_option` → source line 172.
- `--config / -c` exists, type `Path(exists=True)` → source lines 173–179.
- `--list-capabilities / -l` exists as flag → source lines 151–157.
- `--list-rewrites` exists as flag (no short form) → source lines 159–163.
- Dynamic `--enable-X/--disable-X` flags from capability registry → source lines 109–148.
- Missing `--model` when not listing raises `UsageError` → source lines 336–338.
- Config file supports YAML and JSON → source lines 48–70.
