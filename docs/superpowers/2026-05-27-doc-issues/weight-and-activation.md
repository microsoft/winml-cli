# Issues: docs/concepts/weight-and-activation.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical (factually wrong; user would hit error)
- (none)

## Important (misleading or stale claim)
- Line 23: States "QNN on NPU pairs uint8 weights with uint8 or uint16 activations." According to `src/winml/modelkit/config/precision.py`, the NPU auto-precision resolves to `w8a16` (`_AUTO_PRECISION = {"npu": "w8a16", ...}`, line 33), which maps to `uint8` weights + `uint16` activations (lines 57-65). The `int8` preset maps to `uint8/uint8` (lines 39-51). So the claim "uint8 or uint16" is technically accurate for the full range of QNN-targeted precisions, but the default (and most prominently documented) NPU precision is `w8a16` (uint8 weight + uint16 activation), not `uint8/uint8`. The framing may lead readers to underweight the `w8a16` default.

## Minor (style, polish, low-impact)
- Line 19: "The `--weight-type` and `--activation-type` flags on `winml quantize` exist..." — both flags are confirmed at `src/winml/modelkit/commands/quantize.py:67` and `73`.
- Lines 28-33: Cross-links (`quantization.md`, `eps-and-devices.md`, `../commands/quantize.md`, `graphs-and-ir.md`) all resolve to files that exist on disk.

## Verified correct (anchored claims you checked)
- Line 7: "winml quantize ... observes the weight distributions in your exported ONNX and bakes the per-tensor scale/zero-point into the QDQ nodes" — matches `src/winml/modelkit/commands/quantize.py` workflow and `src/winml/modelkit/config/precision.py` precision resolution.
- Lines 19-24: `--weight-type` accepts `uint8, int8, uint16, int16`; `--activation-type` accepts the same — confirmed at `src/winml/modelkit/commands/quantize.py:67-76`.
- Line 25: `w8a16` described as "8-bit weights, 16-bit activations" — confirmed; resolves to `uint8` weight + `uint16` activation via `_BITS_TO_WEIGHT_TYPE[8]="uint8"` and `_BITS_TO_ACTIVATION_TYPE[16]="uint16"` at `src/winml/modelkit/config/precision.py:57-65`.
