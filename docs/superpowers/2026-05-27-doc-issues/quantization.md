# Issues: docs/concepts/quantization.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical (factually wrong; user would hit error)
- Line 9: "every precision from `_KNOWN_PRECISIONS` in `_options.py`". Neither `_KNOWN_PRECISIONS` nor `_options.py` exist anywhere in the source tree. The actual symbol is `_NAMED_PRECISIONS` (a `frozenset` at `src/winml/modelkit/config/precision.py:71`) and there is no file named `_options.py`. This is a fabricated source citation. A reader trying to cross-reference the table against source code will find nothing.

- Line 9: "the resolved quantization types from `config/precision.py`". The file path should be `src/winml/modelkit/config/precision.py`. The abbreviated form `config/precision.py` is navigable by context, but the companion citation `_options.py` is entirely wrong (see above). The combined sentence creates a misleading impression of where the table data lives.

- Line 18 (table row `int8`): "default for NPU via QNN EP". The actual NPU auto-precision default is `w8a16`, not `int8`. `_AUTO_PRECISION = {"npu": "w8a16", ...}` at `src/winml/modelkit/config/precision.py:32-36`. Using `--precision int8` (or the `int8` named preset) resolves to `uint8/uint8` and is _valid_ for QNN, but it is not the auto-selected default. The annotation "default for NPU via QNN EP" is wrong.

- Line 20 (table row `w4a16`): "Recognized as a precision string but raises an error at quantization time; no 4-bit weight dtype mapping exists in `precision.py` yet." This overstates what the code does. `w4a16` is NOT recognized at all. `is_quantized_precision("w4a16")` returns `False` (because `4` is not in `_BITS_TO_WEIGHT_TYPE`), and `_resolve_quant_types()` in `src/winml/modelkit/commands/quantize.py:260-269` raises `click.BadParameter` for any non-quantized, non-auto precision string — including `w4a16`. The doc's claim that it is "recognized as a precision string" is incorrect; it is rejected before reaching quantization time.

## Important (misleading or stale claim)
- Line 17 (table row `auto`): "Resolves to `int8` (NPU), `fp16` (GPU/CPU) at runtime". Partially wrong. For NPU the auto-precision resolves to `w8a16` (not `int8`), per `_AUTO_PRECISION["npu"] = "w8a16"` at `src/winml/modelkit/config/precision.py:33`. For GPU and CPU the `fp16` claim is correct (`_AUTO_PRECISION["gpu"] = "fp16"`, `_AUTO_PRECISION["cpu"] = "fp16"`, lines 34-35).

- Line 16 (table row `int16`): Weight dtype listed as `int16`, activation dtype as `uint16`. Source at `src/winml/modelkit/config/precision.py:43-50` shows `_WEIGHT_TYPE["int16"] = "int16"` and `_ACTIVATION_TYPE["int16"] = "uint16"`. The weight type `int16` is correct. However, the resolution goes through `_BITS_TO_WEIGHT_TYPE[16] = "int16"` when using the `w{x}a{y}` form. The named-preset path matches the table. This row is correct.

## Minor (style, polish, low-impact)
- Lines 63-65: Cross-links (`weight-and-activation.md`, `eps-and-devices.md`, `../commands/quantize.md`, `../commands/eval.md`) all resolve to files that exist on disk.
- Lines 32-35: `--samples` default `10` and `--method` choices `minmax`, `entropy`, `percentile` — all confirmed at `src/winml/modelkit/commands/quantize.py:57-65`.
- Line 22: "`--weight-type` and `--activation-type` flags on `winml quantize` accept `uint8`, `int8`, `uint16`, or `int16`" — confirmed at `src/winml/modelkit/commands/quantize.py:67-76`.

## Verified correct (anchored claims you checked)
- Line 16 (table row `fp16`): No QDQ nodes, float16 throughout — matches `_WEIGHT_TYPE["fp16"] = None` at `src/winml/modelkit/config/precision.py:41`.
- Line 15 (table row `fp32`): No quantization, baseline — matches `_WEIGHT_TYPE["fp32"] = None` at `src/winml/modelkit/config/precision.py:40`.
- Line 19 (table row `w8a8`): `uint8/uint8`, equivalent to `int8` — matches `_MIXED_RE` path resolving `w8a8` -> `_BITS_TO_WEIGHT_TYPE[8]="uint8"`, `_BITS_TO_ACTIVATION_TYPE[8]="uint8"` at `src/winml/modelkit/config/precision.py:57-65`.
- Line 19 (table row `w8a16`): `uint8` weights, `uint16` activations — matches `_BITS_TO_WEIGHT_TYPE[8]="uint8"`, `_BITS_TO_ACTIVATION_TYPE[16]="uint16"` at `src/winml/modelkit/config/precision.py:57-65`.
- Lines 40-41: `--samples` default `10`, `--method` default `minmax` — confirmed at `src/winml/modelkit/commands/quantize.py:57-65`.
