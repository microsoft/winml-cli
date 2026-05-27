# Issues: docs/commands/analyze.md

Source verified against: `src/winml/modelkit/commands/analyze.py` @ 5e25579

## Critical (flag/behavior wrong; user gets error)

- **`--device` default is documented as `NPU`** (doc line 21: "Default: `NPU`") but source line 644 sets `default="auto"` with `show_default=True`. Running `winml analyze --model model.onnx` will use `device="auto"` (infer from local availability), not NPU. A user relying on the doc to know their model will be analyzed against NPU by default will be wrong.

- **`--ep` default is documented as "none — all supported EPs are analyzed"** (doc line 20) but source line 633 sets `default="auto"`. The "auto" mode (source lines 759–768) infers from local availability, not "all supported EPs". Running with no `--ep` is not the same as `--ep all`. The doc's description of the default behavior is wrong.

- **`--run-unknown-op` default is documented as "enabled"** (doc line 26: "flag / enabled") but source line 668 has `default=False`. The pitfall at doc line 84 even says "Disable when the local machine lacks the required libraries" — implying it is on by default — which is incorrect. The correct default is disabled; users must pass `--run-unknown-op` to enable it.

- **`--optim-config` flag is missing from the flag table.** Source lines 677–681 define `@click.option("--optim-config", type=click.Path(path_type=Path), default=None, help="Save auto-discovered optimization config to JSON file")`. This is a functional flag for saving optimization settings and is not documented at all.

- **`--model` has no short form `-m` in the analyze command.** The doc flag table shows no short for `--model` (doc line 19 has empty Short column), which is correct — `model_path_option` in `cli.py` line 68 uses `"--model", "-m"`. Wait — actually it does have `-m`. Let me clarify: the doc table (line 19) shows `| \`--model\` | | \`PATH\` |` with an *empty* Short column, meaning the doc claims there is no short `-m` form. But `model_path_option` (cli.py line 68) uses `click.option("--model", "-m", ...)`, so `-m` is valid. This is a documentation error — users will not know they can use `-m model.onnx`.

- **`--verbose` / `-v` and `--quiet` / `-q` flags are absent from the flag table.** Source uses `@cli_utils.verbosity_options` (line 651) which adds `--verbose / -v` (count) and `--quiet / -q` (flag) — see `cli.py` lines 181–209. Neither appears in the doc.

- **`--config` / `-c` (build config) flag is absent from the flag table.** Source uses `@cli_utils.build_config_option` (line 652) which adds `-c/--config` accepting a `WinMLBuildConfig` JSON file — see `cli.py` lines 212–222. The doc does not mention this.

## Important (misleading or stale)

- **`--ep` choice type** — doc says it accepts full names and short aliases. Source line 634 uses `type=click.Choice([*ALL_EP_NAMES, "all", "auto"], case_sensitive=False)`. The "auto" and "all" values are valid choices but are not mentioned in the doc. The doc's description "When omitted, all supported EPs are analyzed" is wrong (see Critical above); the actual valid special values are "all" and "auto".

- **`--device` choice type** — source line 644 uses `type=click.Choice([*SUPPORTED_DEVICES, "all", "auto"], case_sensitive=False)`. The "all" and "auto" values are not mentioned in the doc.

- **Example "Analyze against all supported EPs"** (doc line 37) runs `winml analyze --model microsoft/resnet-50.onnx` with no `--ep`. Given the actual default is `auto` (not all), the example's described output showing both QNN and OpenVINO may or may not match what runs on a given machine.

## Minor (polish)

- The "Common pitfalls" section says "Omitting `--ep` analyzes every EP" (line 82) — this repeats the incorrect claim from the default description.
- Exit code documentation (codes 0, 1, 2) matches source lines 1212–1214 and is correct.

## Verified correct (key claims checked)

- `--model` exists (via `model_path_option`) and is required → `cli.py` line 57, `analyze.py` line 627.
- `--information/--no-information` flag exists with `default=True` → source lines 654–658.
- `--htp-metadata` flag exists with `type=click.Path(exists=True)`, default `None` → source lines 659–664.
- `--run-unknown-op/--no-run-unknown-op` flag exists → source lines 665–669.
- `--save-node` flag exists as `multiple=True, type=Choice(["partial", "unsupported"])` → source lines 670–676.
- `--output / -o` flag exists → via `cli_utils.output_option`, `cli.py` line 98.
- Static analysis via `ONNXStaticAnalyzer` → source line 819.
- Exit codes 0/1/2 → source lines 1212–1218.
- VitisAI special-cases `--run-unknown-op` to always False → source lines 537–542.
