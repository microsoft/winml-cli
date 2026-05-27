# Issues: docs/commands/compile.md

Source verified against: `src/winml/modelkit/commands/compile.py` @ 5e25579

## Critical (flag/behavior wrong; user gets error)

- **`--device` default listed as `npu` but source default is `auto`.** The flag table and "Common pitfalls" both claim "default is `npu`" and "`--device` default is `npu`, not `auto`". Source `compile.py:59-65` defines `default="auto"`. Users relying on the doc who expect NPU targeting without passing `--device` will instead get auto-detection. This is a direct behavioral contradiction.

## Important (misleading or stale)

- **`--no-quant` flag does not exist in compile.py.** The flag table shows `--no-quant` with description "Flag retained for compatibility; quantization is no longer performed during compile." A search of `compile.py` finds zero occurrences of `no-quant`, `no_quant`, or `--no-quant`. The flag is documented but not defined; any user who passes it will get a "No such option" error.
- **`--validate` / `--no-validate` is a toggle pair, not a simple `--no-validate` flag.** Source `compile.py:72-74` defines `--validate/--no-validate` as a boolean toggle with `default=True`. The table shows only `--no-validate` as an independent flag; this is accurate in effect but hides the positive form `--validate` and implies a different UI contract.
- **`--output` (file path) is not documented in the flag table.** Source `compile.py:51` registers `cli_utils.output_option(...)`, which adds `--output` / `-o`. The table jumps straight to `--output-dir`. Users cannot discover `-o` for writing to a specific file path.

## Minor (polish)

- **Flag table omits `--verbose` / `-v`.** Defined at `compile.py:76-81`.
- **"Common pitfalls" says `--no-quant` is a no-op** — this is correct in spirit (quantization is not done at compile time), but the flag does not exist, so the pitfall note is misleading. Replace with a note that the flag was removed and users should not pass it.

## Verified correct (key claims checked)

- `--model` / `-m` optional (required unless `--list`) → `compile.py:44-50`
- `--output-dir` path default None → `compile.py:53-57`
- `--device` choice `auto|npu|gpu|cpu` → `compile.py:59-65`
- `--ep` choice of provider names → `compile.py:66-69` via `cli_utils.ep_option`
- `--compiler` choice `ort|qairt` default `ort` → `compile.py:82-87`
- `--qnn-sdk-root` path default None → `compile.py:88-93`
- `--embed` flag default false → `compile.py:94-99`
- `--list` flag default false → `compile.py:100-106`
- `--compiler qairt` requires `--qnn-sdk-root` → `compile.py:206-208` (passes to `ep_config.qnn_sdk_root`; failure occurs in compiler layer)
- No `wmk` or `ModelKit` strings in user-facing prose → confirmed
