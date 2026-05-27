# Issues: docs/commands/quantize.md

Source verified against: `src/winml/modelkit/commands/quantize.py` @ 5e25579

## Critical (flag/behavior wrong; user gets error)

- (none)

## Important (misleading or stale)

- **`--precision` accepted values listed as `int8`, `int16`, or `w8a16` but source also accepts `auto` and the full `w{x}a{y}` family.** The doc's flag table says only "`int8`, `int16`, or mixed-precision like `w8a16`". Source `quantize.py:50-53` documents: "Accepted: auto, int8, int16, or w{x}a{y} where x,y in {8,16} (e.g., w8a8, w8a16, w16a16)." The `auto` value and `w8a8` / `w16a16` forms are silently omitted from the table.
- **Flag table omits `--task` and `--model-name`.** Both are real options defined in source (`quantize.py:92-109`). `--task` selects a calibration dataset; `--model-name` enables task-aware calibration with the model's preprocessor. Users who need task-aware calibration have no documentation to guide them.
- **Flag table omits `--verbose` / `-v`.** Defined at `quantize.py:104-109`.

## Minor (polish)

- **Default output path description says "`{input}_qdq.onnx`" but should clarify stem only.** Source uses `model.stem + "_qdq.onnx"` in the same directory as the input (`quantize.py:189`), which matches, but "`{input}`" is ambiguous about whether the full path or just the stem is used.
- **"Quantizing an already-quantized model is unsupported" pitfall** mentions `winml compile --no-quant` as the alternative. As noted in compile.md, `--no-quant` is a no-op in compile. The pitfall advice is therefore unhelpful and should be updated to reflect actual behavior.

## Verified correct (key claims checked)

- `--model` / `-m` required path → `quantize.py:37-43`
- `--output` / `-o` optional path, default `{stem}_qdq.onnx` → `quantize.py:44` + `quantize.py:189`
- `--precision` / `-p` string default None → `quantize.py:45-53`
- `--samples` integer default 10 → `quantize.py:54-58`
- `--method` choice `minmax|entropy|percentile` default `minmax` → `quantize.py:59-65`
- `--weight-type` choice `uint8|int8|uint16|int16` default None → `quantize.py:66-71`
- `--activation-type` choice `uint8|int8|uint16|int16` default None → `quantize.py:72-77`
- `--per-channel` flag default false → `quantize.py:78-83`
- `--symmetric` flag default false → `quantize.py:84-89`
- Explicit type flags override `--precision` → `quantize.py:271-276`
- Default types when no precision specified: uint8/uint8 → `quantize.py:263` (precision=None or "auto" → default_w/a = "uint8")
- No `wmk` or `ModelKit` strings in user-facing prose → confirmed
