# Issues: docs/concepts/load-and-export.md

Source verified against: microsoft/winml-cli @ 5e25579

## Critical (factually wrong; user would hit error)

- (none)

## Important (misleading or stale claim)

- **`--dynamo` described as "reserved but not yet functional"** (line 19): The doc says "the `--dynamo` flag is reserved for the PyTorch 2.x dynamo exporter but is **not yet functional** in the current release — passing it logs a warning and the flag is ignored." The source confirms this: `commands/export.py` lines 376-384 show that when `dynamo=True`, a warning is printed and the flag is ignored. The note itself is accurate, but the doc still mentions "PyTorch 2.x" while the CLI help text says "PyTorch 2.9+" (`commands/export.py` line 98: `"Enable PyTorch 2.9+ dynamo export for rich node metadata"`). The version reference in the doc is stale/imprecise.

- **`--torch-module` described as "reserved but not yet functional"** (line 35): Similarly, the source confirms (`commands/export.py` lines 362-373) it logs a warning and is ignored. The doc note is accurate. However, the doc says it is "intended to include them as distinct hierarchy nodes" while the CLI help says "Include torch.nn modules in hierarchy (comma-separated)" — consistent.

- **`winml inspect` described as working "without downloading weights"** (line 13): The doc says `winml inspect` "prints the detected task, the HuggingFace model class, the export configuration, and the WinML inference class — all without downloading weights. Add `--hierarchy` to reconstruct the PyTorch module tree from random-weight tracing." The `commands/inspect.py` file was not read, so this specific claim about not downloading weights cannot be confirmed or denied from available sources. This warrants scrutiny.

- **`--shape-config` vs `--input-specs`** (line 33): The doc says "Provide a `--shape-config` JSON file with explicit overrides, or use `--input-specs` to supply a fully specified input manifest." The `winml export` command has both flags: `--shape-config` (line 126 in `commands/export.py`) and `--input-specs` (line 106-111). This is correct. However, the doc describes them as equivalent alternatives — in the source, `--shape-config` passes shape overrides to auto-resolution while `--input-specs` overrides individual tensor specs after auto-resolution. They work differently, not interchangeably.

## Minor (style, polish, low-impact)

- **`winml.hierarchy.tag` metadata key name** (line 21): Doc says nodes carry `winml.hierarchy.tag` and `winml.hierarchy.depth`. Both keys confirmed at `src/winml/modelkit/export/htp/exporter.py` lines 594-595 and `src/winml/modelkit/core/node_metadata.py` lines 71, 74.

- **`winml.io.inputs` and `winml.io.outputs` described as model-level** (line 21): Confirmed at `src/winml/modelkit/export/htp/exporter.py` lines 556, 564.

- **`--no-hierarchy` alias `--clean-onnx`** (line 23): Source confirms both flags exist as aliases: `commands/export.py` lines 87-92 (`--clean-onnx` / `--no-hierarchy`).

- **`--with-report` flag** (line 25): Exists at `commands/export.py` line 80-83.

- **Cross-links** `[graphs-and-ir.md]`, `[../commands/inspect.md]`, `[../commands/export.md]` (lines 39-41): All files exist.

## Verified correct (anchored claims you checked)

- `winml export` uses TorchScript tracing by default → `commands/export.py` line 157 (docstring: "ONNX Export — Convert to ONNX format (TorchScript by default)")
- `--dynamo` flag exists on `winml export` → `commands/export.py` lines 94-98
- `--torch-module` flag exists on `winml export` → `commands/export.py` lines 100-105
- `--task` flag exists on `winml export` → `commands/export.py` lines 112-117
- `--input-specs` flag exists on `winml export` → `commands/export.py` lines 106-111
- `--shape-config` flag exists on `winml export` → `commands/export.py` lines 125-130
- `winml.hierarchy.tag` is a real metadata key → `core/node_metadata.py` line 71
- `winml.hierarchy.depth` is a real metadata key → `core/node_metadata.py` line 74
- `winml.io.inputs` / `winml.io.outputs` are model-level metadata props → `export/htp/exporter.py` lines 556, 564
- `--trust-remote-code` applies to `winml config` (not `winml export` directly) → `commands/config.py` line 166
- No `wmk` or `ModelKit` strings in prose → verified by grep
