# Module: export
**Path**: `src/winml/modelkit/export/`
**Period**: 2026-03-23 to 2026-04-08

## 1. Module Overview
The `export` module handles ONNX model export from HuggingFace models, including I/O specification, HTP (Qualcomm) metadata generation, and config overrides.

## 2. Files Changed This Period
| File | PRs | Summary |
|------|-----|---------|
| `io.py` | #28, #212 | Class rename `OnnxModelOutput` → `ONNXModelOutput` (#28); pixel mask output handling fix for SAM2 (#212) |
| `__init__.py` | #28, #41 | Class rename update (#28); `generate_dummy_inputs` exported (#41) |
| `htp/exporter.py` | #15 | Batch update (+39/-x) |
| `htp/metadata_builder.py` | #28 | Class rename update |
| `htp/htp_metadata_schema.json` | #28 | Schema field rename |

## 3. Net Change Summary
- `OnnxModelOutput` was renamed to `ONNXModelOutput` to comply with the naming convention established in PR #28.
- `io.py` was updated in PR #212 to correctly handle pixel mask outputs in the SAM2 model during export, fixing the `facebook/sam2.1-hiera-small` model.
- `generate_dummy_inputs` was added to `export/__init__.py` in PR #41, allowing test code to import it from the package level.

## 4. New APIs/Functions Added
| Symbol | Description |
|--------|-------------|
| `generate_dummy_inputs` | Exported from `export/__init__.py` (#41) |
