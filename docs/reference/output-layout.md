# Output Layout

When you run `winml build`, the tool writes all artifacts to the output
directory. This page documents what each file is and which ones you need
for deployment.

---

## Directory Structure

After a full pipeline run (export → optimize → quantize → compile):

```text
output/
├── model.onnx                  ← FINAL artifact (deploy this)
├── model.onnx.data             ← External weights (if model ≥ 100 MiB)
├── winml_build_config.json     ← Persisted build config
├── build_manifest.json         ← Build provenance and timing
├── export.onnx                 ← Intermediate: raw ONNX export
├── export.onnx.data
├── optimized.onnx              ← Intermediate: after graph optimization
├── optimized.onnx.data
├── quantized.onnx              ← Intermediate: after QDQ insertion
├── quantized.onnx.data
├── compiled.onnx               ← Intermediate: after EP compilation
└── compiled.onnx.data
```

---

## File Categories

### Final Artifacts (Keep for Deployment)

| File | Purpose |
|------|---------|
| `model.onnx` | The deployment-ready model. Always present. |
| `model.onnx.data` | External weight data (only if model ≥ 100 MiB). Must stay alongside `model.onnx`. |
| `winml_build_config.json` | The config used for this build (includes auto-discovered flags). Useful for reproducibility. |
| `build_manifest.json` | Build metadata: stages run, timings, quantization stats. |

### Intermediate Files (Can Delete After Build)

| File | Stage | Contents |
|------|-------|----------|
| `export.onnx` | Export | Raw PyTorch → ONNX conversion (float32) |
| `optimized.onnx` | Optimize | Graph with fused operators, shape inference applied |
| `quantized.onnx` | Quantize | QDQ nodes inserted, calibrated scales |
| `compiled.onnx` | Compile | EPContext binary embedded or sidecar |

Each intermediate has a corresponding `.onnx.data` file if the model exceeds
100 MiB.

---

## What Gets Written at Each Stage

### Export only (`winml export`)

```text
output/
├── export.onnx
└── export.onnx.data          (if ≥ 100 MiB)
```

### Optimize only (`winml optimize`)

```text
output/
├── optimized.onnx
└── optimized.onnx.data
```

### Full build (`winml build`)

All stages write their intermediate, and `model.onnx` is a copy of the last
successful stage output. If you skip quantization (`--no-quant`), the final
model is a copy of `optimized.onnx`. If you skip compilation too, it's still
a copy of `optimized.onnx`.

---

## External Data

Models larger than **100 MiB** store weights in a separate `.onnx.data` file.
Both files must be kept together — the `.onnx` file contains a reference to the
data file by name.

| Model Size | Files |
|-----------|-------|
| < 100 MiB | `model.onnx` only (weights embedded) |
| ≥ 100 MiB | `model.onnx` + `model.onnx.data` |

!!! warning
    If you move `model.onnx`, always move `model.onnx.data` alongside it.
    The ONNX file references the data file by relative path.

---

## Build Manifest

`build_manifest.json` records provenance for every build:

```json
{
  "schema_version": 1,
  "model_id": "microsoft/resnet-50",
  "task": "image-classification",
  "cache_key": "a1b2c3d4e5f6",
  "config_hash": "f7e8d9c0b1a2",
  "timestamp": "2026-01-15T10:30:00.000000+00:00",
  "elapsed_seconds": 45.1,
  "final_artifact": "model.onnx",
  "analyze_iterations": 2,
  "analyze_unsupported_node_count": 0,
  "analyze_details": { "lint": {}, "autoconf": {} },
  "stages": [
    {
      "name": "export",
      "status": "completed",
      "filename": "export.onnx",
      "elapsed_seconds": 12.5
    },
    {
      "name": "optimize",
      "status": "completed",
      "filename": "optimized.onnx",
      "elapsed_seconds": 8.2
    },
    {
      "name": "quantize",
      "status": "completed",
      "filename": "quantized.onnx",
      "elapsed_seconds": 15.3,
      "nodes_quantized": 150,
      "nodes_skipped": 12
    },
    {
      "name": "compile",
      "status": "completed",
      "filename": "compiled.onnx",
      "elapsed_seconds": 9.1
    }
  ]
}
```

---

## Rebuild Behavior

- If `model.onnx` already exists and `rebuild=False` (default), the build is
  skipped entirely.
- Pass `--rebuild` (CLI) or `force_rebuild=True` (Python API) to force a fresh
  build.
- On rebuild, all old `.onnx` and `.onnx.data` files are deleted before the
  pipeline runs.

---

## See also

- [winml build](../commands/build.md) — build command reference
- [Reference — Build Configuration Schema](index.md) — config file format
- [How winml-cli Works](../concepts/how-it-works.md) — pipeline stages explained
