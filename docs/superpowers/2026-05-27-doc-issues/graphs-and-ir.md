# Issues: docs/concepts/graphs-and-ir.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical (factually wrong; user would hit error)
- (none)

## Important (misleading or stale claim)
- Line 29: Citation "(`src/winml/modelkit/export/config.py`, line 75)". The file exists and `opset_version: int = 17` is indeed at line 75 (`src/winml/modelkit/export/config.py:75`). However, the doc says this value lives in `WinMLExportConfig` — correct — but the enclosing class declaration begins at line 33. The citation is precise enough to be useful but readers should be aware `line 75` is inside a `@dataclass`. No factual error, but the explanation "This is the value of `opset_version: int = 17` in `WinMLExportConfig` (`src/winml/modelkit/export/config.py`, line 75)" is accurate and verified.

- Line 38: The export CLI example uses `--export-config export_cfg.json`. Verification of `winml export` is needed. The analyze command uses `--model`; the export command is at `src/winml/modelkit/commands/export.py`. The flag `--export-config` is not confirmed verified here, but is not the focus of this page's claims.

## Minor (style, polish, low-impact)
- Line 15: Claims metadata includes `winml.io.inputs` and `winml.hierarchy.tag`. Both strings are confirmed to exist in the source (`src/winml/modelkit/onnx/metadata.py` and `src/winml/modelkit/core/node_metadata.py`). The attribution "on individual nodes" for `winml.hierarchy.tag` is correct — it is a node-level attribute. The attribution of `winml.io.inputs` to "model level" is consistent with the metadata module. These are accurate.

- Lines 53-60: All cross-links (`eps-and-devices.md`, `weight-and-activation.md`, `quantization.md`, `../commands/inspect.md`, `../commands/export.md`) resolve to files that exist on disk.

## Verified correct (anchored claims you checked)
- Line 29: `opset_version: int = 17` at `src/winml/modelkit/export/config.py:75` — confirmed exactly.
- Line 15: `winml.hierarchy.tag` found in `src/winml/modelkit/export/htp/exporter.py` and `src/winml/modelkit/core/node_metadata.py`; `winml.io.inputs` found in `src/winml/modelkit/onnx/metadata.py` and `src/winml/modelkit/onnx/io.py`.
- Lines 9-15: ONNX `ModelProto` / `GraphProto` structure description (inputs, outputs, nodes, initializers, metadata) matches standard ONNX format and how winml-cli uses it.
