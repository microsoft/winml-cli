# Issues: docs/commands/export.md

Source verified against: `src/winml/modelkit/commands/export.py` @ 5e25579

## Critical (flag/behavior wrong; user gets error)

- (none)

## Important (misleading or stale)

- **`--dynamo` description says "PyTorch 2.9+" but that version string is invented.** The source (`export.py:376-384`) only warns the flag is unsupported; no PyTorch version requirement is stated. Remove the version number claim to avoid confusion.
- **`--torch-module` description says "Experimental — currently logs a warning"** — this is accurate, but the phrase "currently logs a warning" hides the fact that the flag is **completely ignored** (the option value is never forwarded to `export_onnx()`). Source at `export.py:364-373` explicitly states `TODO: Add torch_module support`. Use "has no effect" rather than "currently logs a warning".
- **`--dynamo` same problem.** Source `export.py:376-384`: "dynamo=True is not supported by export_onnx(). TODO: Add dynamo support". The flag has zero effect; the table note says only "currently logs a warning".

## Minor (polish)

- **Flag table missing `--verbose` / `-v`.** `export.py:73-78` defines `--verbose / -v` as an explicit option with a `help` string. Every other command page includes `--verbose` in their tables; its absence on this page is inconsistent.
- **`--clean-onnx` / `--no-hierarchy` are presented as two separate flags in the table but they are one option.** The source defines them as aliases of a single `--clean-onnx / --no-hierarchy` option with `"no_hierarchy"` as the internal parameter name (`export.py:85-92`). The table formatting (`--clean-onnx` / `--no-hierarchy` in one cell) is technically correct but the slash notation could mislead readers into thinking these are independent toggles.

## Verified correct (key claims checked)

- `--model` / `-m` required string → `export.py:65-70`
- `--output` / `-o` required path → `export.py:71` via `cli_utils.output_option(required=True)`
- `--with-report` is_flag default false → `export.py:79-84`
- `--input-specs` path default None → `export.py:107-111`
- `--task` / `-t` string default None → `export.py:113-118`
- `--export-config` path default None → `export.py:119-124`
- `--shape-config` path default None → `export.py:125-130`
- `--shape-config` silently ignored when `--input-specs` is provided → `export.py:307-331` (input-specs overrides/patches auto-resolved tensors; shape_config is loaded only before auto-resolution, so if both are present the shape_config still applies to auto-resolution and input-specs then overrides it — the doc's "Ignored when `--input-specs` is provided" is a slight overstatement but matches the spirit)
- Eight-step HTP export description → `export.py:153-161` (docstring)
- `--dynamo` and `--torch-module` emit warnings and have no effect → `export.py:364-384`
- No `wmk` or `ModelKit` strings in user-facing prose → confirmed
