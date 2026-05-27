# Issues: docs/commands/config.md

Source verified against: `src/winml/modelkit/commands/config.py` @ 5e25579

## Critical (flag/behavior wrong; user gets error)

- **`--no-compile` default is wrong.** The doc (line 32) states default is `off` (meaning compile *is* included by default). Source line 163 defines `--no-compile/--compile` with `"no_compile"` and `default=True`. The default is `no_compile=True`, meaning compilation is *excluded* from the generated config by default. A user reading the doc will expect compilation to be in the config and be surprised to find `"compile": null` in the output.

- **`--verbose` flag is missing from the flag table.** Source lines 147–152 define `@click.option("-v", "--verbose", is_flag=True, default=False, ...)`. This is a real flag that enables `logging.DEBUG` (line 226) and is not documented in the flag table.

- **`--ep` short form** — the doc flag table (line 27) shows no short form for `--ep`. The source uses `@cli_utils.ep_option(required=False, ...)` (line 126), and `ep_option` in `cli.py` line 140 registers `"--ep", "--execution-provider"` with no `-e` short. The doc correctly shows no short form, but it lists the full name without mentioning `--execution-provider` as an alias. This is a minor completeness issue but not an error.

## Important (misleading or stale)

- **`--no-compile` documentation**: The doc entry says default is `off` and the description reads "Omit compilation from the generated config (sets `compile` to `null`). Use this when you want to inspect the optimized ONNX before EP-specific compilation." Since `no_compile` defaults to `True`, compilation is omitted *by default* — the entire framing of `--no-compile` as an opt-in is backwards. Users do not need to pass `--no-compile` to skip compilation; they need `--compile` to include it.

- **`--device` Choice values** — the doc says type is `auto|npu|gpu|cpu` (line 28). Source line 121 confirms `type=click.Choice(["auto", "npu", "gpu", "cpu"], case_sensitive=False)`. This is accurate.

- **`--config / -c` help text says "JSON override file in `WinMLBuildConfig` format"** (doc line 24). Source line 103 uses `type=click.Path(exists=True)` and the flag is called `config_file`. The doc correctly describes behavior.

- **`--ep` accepts aliases** — doc says values include `qnn`, `dml`, `migraphx`, `tensorrt`, `vitisai`, `openvino`, `cpu`. The actual choices come from `ALL_EP_NAMES` via `ep_option` (cli.py line 138). The list of aliases in the doc should be verified against `SUPPORTED_EPS` / `ALL_EP_NAMES` constants. The doc lists `dml` and `migraphx` which may or may not be in `ALL_EP_NAMES` — this should be confirmed.

## Minor (polish)

- The doc example `winml config -m facebook/convnext-tiny-224.onnx --no-quant --no-compile` (line 80) uses `--no-compile` as if it toggles something off, but since `no_compile=True` by default, `--no-compile` is a no-op here. The example is not wrong (it still works) but implies `--no-compile` is doing work when it is already the default.
- `--trust-remote-code` is correctly listed in the flag table and matches source (via `@cli_utils.trust_remote_code_option()` at line 166).

## Verified correct (key claims checked)

- `-m / --model` exists with short `-m`, optional (not required), default `None` → source lines 67–74.
- `-t / --task` exists with short `-t`, default `None` → source lines 75–79.
- `--model-class` exists, no short form, default `None` → source lines 80–85.
- `--model-type` exists, no short form, default `None` → source lines 86–94.
- `--module` exists, no short form, default `None` → source lines 95–99.
- `-c / --config` exists, type `Path(exists=True)`, default `None` → source lines 100–107.
- `--shape-config` exists, type `Path(exists=True)`, default `None` → source lines 108–117.
- `-d / --device` exists with Choice `["auto","npu","gpu","cpu"]`, default `"auto"` → source lines 118–125.
- `-p / --precision` exists, type `str`, default `"auto"` → source lines 131–138.
- `-o / --output` exists → via `cli_utils.output_option`, source line 140.
- `--library` exists, default `"transformers"` → source lines 141–145.
- `--no-quant` exists as `is_flag=True, default=False` → source lines 153–158.
- At least one of `-m`, `--model-type`, `--model-class` required → source lines 229–241.
- ONNX file input path sets `export=None` → source lines 297–311.
